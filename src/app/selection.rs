//! Cell details on hover, legend highlighting and lasso selection.
//!
//! The viewport records a [`ViewSnapshot`] every frame: where each cell is
//! drawn (the two spaces of the current segment, their normalisation, the
//! camera and the interpolation time). Hover picking and the lasso project
//! cells to the screen with it on the CPU, the same way the shader does.
//!
//! Highlighting a legend group or showing a lasso selection "emphasises" a set
//! of cells: everything else is blended towards the background and drawn
//! first, so the emphasised cells end up on top. Video exports skip this.

use super::{ColorKey, ColorMode, StvizApp};
use crate::{
    camera::Camera2D,
    data::{Dataset, ObsMeta},
};
use eframe::egui;
use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
    io::Write,
    path::Path,
    sync::Arc,
};

/// Lines shown in the hover tooltip before it ends with "+N more".
const MAX_TOOLTIP_FIELDS: usize = 14;
/// How much a dimmed cell keeps of its own colour (the rest is background).
const DIM_KEEP: f32 = 0.18;
/// Pointer distance, in points, within which a cell counts as hovered.
const MIN_PICK_RADIUS: f32 = 6.0;

/// Where one end of the current segment takes its positions from.
#[derive(Clone)]
pub(super) enum PosSource {
    Space(usize),
    /// Sample grid positions, `2 * n_points` floats.
    Grid(Arc<Vec<f32>>),
}

/// How cells were drawn in the last frame.
#[derive(Clone)]
pub(super) struct ViewSnapshot {
    pub(super) rect: egui::Rect,
    pub(super) ppp: f32,
    pub(super) camera: Camera2D,
    pub(super) from: PosSource,
    pub(super) to: PosSource,
    pub(super) from_center: [f32; 2],
    pub(super) from_scale: f32,
    pub(super) to_center: [f32; 2],
    pub(super) to_scale: f32,
    pub(super) t: f32,
    pub(super) draw_indices: Arc<Vec<u32>>,
    pub(super) colors_from: Arc<Vec<u32>>,
    pub(super) colors_to: Arc<Vec<u32>>,
    pub(super) color_t: f32,
    /// The space a reader would say is on screen (the nearer segment end).
    pub(super) shown_space: usize,
    /// The colour key of the nearer segment end.
    pub(super) shown_color: ColorKey,
    pub(super) point_radius_pts: f32,
}

/// Borrowed position arrays for projecting cells to the screen.
struct Projector<'a> {
    view: &'a ViewSnapshot,
    from: &'a [f32],
    to: &'a [f32],
}

impl<'a> Projector<'a> {
    fn new(ds: &'a Dataset, view: &'a ViewSnapshot) -> Option<Self> {
        let n2 = ds.meta.n_points as usize * 2;
        let slice = |src: &'a PosSource| -> Option<&'a [f32]> {
            match src {
                PosSource::Grid(v) if v.len() == n2 => Some(v.as_slice()),
                PosSource::Grid(_) => None,
                PosSource::Space(i) => ds.space_f32_2d(*i).ok(),
            }
        };
        Some(Self {
            view,
            from: slice(&view.from)?,
            to: slice(&view.to)?,
        })
    }

    /// Screen position (egui points) of a cell, `None` if it has no
    /// finite position or is hidden by a filter.
    fn screen_pos(&self, cell: u32) -> Option<egui::Pos2> {
        let v = self.view;
        let i = cell as usize;
        let colors = if v.color_t < 0.5 {
            &v.colors_from
        } else {
            &v.colors_to
        };
        if colors.get(i).is_some_and(|c| c >> 24 == 0) {
            return None;
        }
        let (ax, ay) = (*self.from.get(2 * i)?, *self.from.get(2 * i + 1)?);
        let (bx, by) = (*self.to.get(2 * i)?, *self.to.get(2 * i + 1)?);
        let a = [
            (ax - v.from_center[0]) * v.from_scale,
            (ay - v.from_center[1]) * v.from_scale,
        ];
        let b = [
            (bx - v.to_center[0]) * v.to_scale,
            (by - v.to_center[1]) * v.to_scale,
        ];
        let t = v.t.clamp(0.0, 1.0);
        let p = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
        let vp = [v.rect.width() * v.ppp, v.rect.height() * v.ppp];
        let sx = (p[0] - v.camera.center[0]) * v.camera.pixels_per_unit + 0.5 * vp[0];
        let sy = (p[1] - v.camera.center[1]) * v.camera.pixels_per_unit + 0.5 * vp[1];
        let pos = v.rect.min + egui::vec2(sx / v.ppp, sy / v.ppp);
        (pos.x.is_finite() && pos.y.is_finite()).then_some(pos)
    }
}

/// What the viewport is emphasising this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Emphasis {
    Category { obs_idx: usize, category: usize },
    Selection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LassoMode {
    Replace,
    Add,
    Remove,
}

type DimmedColors = HashMap<(u64, u64, u32), (Arc<Vec<u32>>, u64)>;
type OrderedIndices = Option<((u64, u64), Arc<Vec<u32>>, u64)>;

#[derive(Default)]
struct EmphasisCache {
    /// Signature of the emphasised set and its per-cell mask.
    mask: Option<(u64, Arc<Vec<bool>>)>,
    /// Dimmed colour arrays by (source colour id, set, background).
    colors: DimmedColors,
    /// Reordered draw indices by (source indices id, set).
    indices: OrderedIndices,
}

#[derive(Clone, PartialEq)]
struct HoverKey {
    pointer: egui::Pos2,
    rect: egui::Rect,
    center: [f32; 2],
    ppu: f32,
    t: f32,
    indices: usize,
    dataset_id: u64,
}

#[derive(Default)]
pub(super) struct SelectionState {
    /// Show cell details in a tooltip while hovering the view.
    pub(super) hover_enabled: bool,
    /// Drag draws a lasso without holding Shift.
    pub(super) lasso_tool: bool,
    /// Dim everything outside the selection.
    pub(super) show_selection: bool,
    dataset_id: u64,
    /// Selected cells, sorted dataset indices.
    selected: Vec<u32>,
    selected_version: u64,
    lasso_path: Vec<egui::Pos2>,
    lasso_mode: Option<LassoMode>,
    hovered: Option<u32>,
    hover_key: Option<HoverKey>,
    /// Tooltip rows, cached per (cell, colour key).
    #[allow(clippy::type_complexity)]
    tooltip: Option<(u32, ColorKey, Vec<(String, String)>)>,
    /// Legend rows drawn in the viewport last frame: screen rect and label.
    pub(super) legend_rows: Vec<(egui::Rect, String)>,
    /// Category hovered in the left panel this frame.
    pub(super) panel_hover: Option<(usize, usize)>,
    legend_hover: Option<(usize, usize)>,
    pinned: Option<(usize, usize)>,
    view: Option<ViewSnapshot>,
    cache: EmphasisCache,
    pub(super) status: Option<String>,
}

impl SelectionState {
    pub(super) fn new() -> Self {
        Self {
            hover_enabled: true,
            show_selection: true,
            ..Default::default()
        }
    }
}

impl SelectionState {
    pub(super) fn version(&self) -> u64 {
        self.selected_version
    }

    pub(super) fn selected(&self) -> &[u32] {
        &self.selected
    }
}

impl StvizApp {
    /// Replace the lasso selection, as undo and redo do.
    pub(super) fn restore_selection(&mut self, cells: &[u32]) {
        self.sync_selection_dataset();
        self.selection.selected = cells.to_vec();
        self.selection.selected_version = self.selection.selected_version.wrapping_add(1);
        self.selection.status = None;
    }

    /// Forget selection, highlight and hover when another dataset is loaded.
    pub(super) fn sync_selection_dataset(&mut self) {
        if self.selection.dataset_id != self.dataset_id {
            let keep = (
                self.selection.hover_enabled,
                self.selection.lasso_tool,
                self.selection.show_selection,
            );
            self.selection = SelectionState::new();
            (
                self.selection.hover_enabled,
                self.selection.lasso_tool,
                self.selection.show_selection,
            ) = keep;
            self.selection.dataset_id = self.dataset_id;
        }
    }

    fn video_export_running(&self) -> bool {
        self.exporting_loop || self.export_finishing
    }

    /// Lasso and legend input. Returns true when the drag belongs to the
    /// lasso, so the caller should not pan.
    pub(super) fn selection_input(
        &mut self,
        response: &egui::Response,
        ctx: &egui::Context,
    ) -> bool {
        self.sync_selection_dataset();
        let pointer = response.hover_pos();

        // Legend rows from the overlay drawn last frame.
        self.selection.legend_hover = None;
        if let (Some(p), Some(ds)) = (pointer, self.dataset.clone()) {
            if self.selection.lasso_path.is_empty() {
                if let Some(label) = self
                    .selection
                    .legend_rows
                    .iter()
                    .find(|(r, _)| r.contains(p))
                    .map(|(_, l)| l.clone())
                {
                    if let Some(hit) = self.legend_category(&ds, &label) {
                        self.selection.legend_hover = Some(hit);
                        ctx.set_cursor_icon(egui::CursorIcon::PointingHand);
                        if response.clicked() {
                            self.selection.pinned =
                                (self.selection.pinned != Some(hit)).then_some(hit);
                        }
                    }
                }
            }
        }

        if self.video_export_running() {
            self.selection.lasso_path.clear();
            return false;
        }
        let mods = ctx.input(|i| i.modifiers);
        if response.drag_started() && (mods.shift || self.selection.lasso_tool) {
            let mode = if mods.command {
                LassoMode::Add
            } else if mods.alt {
                LassoMode::Remove
            } else {
                LassoMode::Replace
            };
            self.selection.lasso_mode = Some(mode);
            self.selection.lasso_path.clear();
            if let Some(p) = ctx.input(|i| i.pointer.press_origin()) {
                self.selection.lasso_path.push(p);
            }
        }
        if self.selection.lasso_mode.is_none() {
            return false;
        }
        if response.dragged() {
            if let Some(p) = ctx.input(|i| i.pointer.interact_pos()) {
                let far_enough = self
                    .selection
                    .lasso_path
                    .last()
                    .is_none_or(|last| last.distance(p) >= 2.0);
                if far_enough {
                    self.selection.lasso_path.push(p);
                }
            }
        }
        if response.drag_stopped() || !ctx.input(|i| i.pointer.primary_down()) {
            self.finish_lasso();
        }
        true
    }

    fn finish_lasso(&mut self) {
        let path = std::mem::take(&mut self.selection.lasso_path);
        let Some(mode) = self.selection.lasso_mode.take() else {
            return;
        };
        if path.len() < 3 {
            return;
        }
        let (Some(ds), Some(view)) = (self.dataset.clone(), self.selection.view.clone()) else {
            return;
        };
        let Some(proj) = Projector::new(&ds, &view) else {
            return;
        };
        // Every cell that passes the filters, not only the drawn sample, so a
        // draw cap doesn't thin out the selection. The advanced timeline
        // preview shows its own subset; select from what it draws.
        let previewing =
            self.advanced_timeline_open && !self.playing && self.advanced_preview_card.is_some();
        let candidates: &[u32] = if previewing {
            view.draw_indices.as_slice()
        } else {
            self.base_indices.as_slice()
        };
        let hits = lasso_hits(candidates, &path, |c| proj.screen_pos(c));
        let before = self.selection.selected.len();
        self.selection.selected = combine_selection(&self.selection.selected, hits, mode);
        self.selection.selected_version = self.selection.selected_version.wrapping_add(1);
        self.selection.show_selection = true;
        let after = self.selection.selected.len();
        self.selection.status = Some(match mode {
            LassoMode::Replace => format!("Selected {after} cells."),
            LassoMode::Add => format!("Added {} cells ({after} selected).", after - before),
            LassoMode::Remove => format!("Removed {} cells ({after} selected).", before - after),
        });
    }

    /// The (obs, category) of a legend row label, for the legend on screen.
    fn legend_category(&self, ds: &Dataset, label: &str) -> Option<(usize, usize)> {
        let obs_idx = self
            .selection
            .view
            .as_ref()
            .and_then(|v| match &v.shown_color {
                ColorKey::Categorical(i) => Some(*i),
                ColorKey::Current if self.color_mode == ColorMode::Categorical => {
                    Some(self.active_obs_idx)
                }
                _ => None,
            })?;
        let (_name, _labels, categories, _pal) = ds.obs_categorical(obs_idx).ok()?;
        let category = categories.iter().position(|c| c == label)?;
        Some((obs_idx, category))
    }

    /// Remember how cells are drawn this frame, then dim colours and reorder
    /// draw indices for the current highlight or selection.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply_selection_view(
        &mut self,
        ds: &Dataset,
        snapshot: ViewSnapshot,
        colors_from: &mut Arc<Vec<u32>>,
        colors_from_id: &mut u64,
        colors_to: &mut Arc<Vec<u32>>,
        colors_to_id: &mut u64,
        draw_indices: &mut Arc<Vec<u32>>,
        indices_id: &mut u64,
    ) {
        self.sync_selection_dataset();
        self.selection.view = Some(snapshot);
        let panel = self.selection.panel_hover.take();
        if self.video_export_running() {
            return;
        }
        let emphasis = panel
            .or(self.selection.legend_hover)
            .or(self.selection.pinned)
            .map(|(obs_idx, category)| Emphasis::Category { obs_idx, category })
            .or_else(|| {
                (self.selection.show_selection && !self.selection.selected.is_empty())
                    .then_some(Emphasis::Selection)
            });
        let Some(emphasis) = emphasis else {
            return;
        };
        let Some((sig, mask)) = self.emphasis_mask(ds, emphasis) else {
            return;
        };
        let bg = self.background_color;
        let bg = [bg.r(), bg.g(), bg.b()];
        for (colors, id) in [(colors_from, colors_from_id), (colors_to, colors_to_id)] {
            let key = (*id, sig, u32::from_le_bytes([bg[0], bg[1], bg[2], 0]));
            let entry = self.selection.cache.colors.get(&key).cloned();
            let (dimmed, dimmed_id) = entry.unwrap_or_else(|| {
                let dimmed = Arc::new(dim_colors(colors, &mask, bg));
                let dimmed_id = derived_id(&key);
                if self.selection.cache.colors.len() >= 4 {
                    self.selection.cache.colors.clear();
                }
                self.selection
                    .cache
                    .colors
                    .insert(key, (dimmed.clone(), dimmed_id));
                (dimmed, dimmed_id)
            });
            *colors = dimmed;
            *id = dimmed_id;
        }
        let key = (*indices_id, sig);
        match &self.selection.cache.indices {
            Some((k, idx, id)) if *k == key => {
                *draw_indices = idx.clone();
                *indices_id = *id;
            }
            _ => {
                let ordered = Arc::new(emphasised_last(draw_indices, &mask));
                let id = derived_id(&key);
                self.selection.cache.indices = Some((key, ordered.clone(), id));
                *draw_indices = ordered;
                *indices_id = id;
            }
        }
    }

    fn emphasis_mask(&mut self, ds: &Dataset, emphasis: Emphasis) -> Option<(u64, Arc<Vec<bool>>)> {
        let sig = match emphasis {
            Emphasis::Category { obs_idx, category } => {
                derived_id(&("category", self.dataset_id, obs_idx, category))
            }
            Emphasis::Selection => derived_id(&(
                "selection",
                self.dataset_id,
                self.selection.selected_version,
            )),
        };
        if let Some((s, mask)) = &self.selection.cache.mask {
            if *s == sig {
                return Some((sig, mask.clone()));
            }
        }
        let n = ds.meta.n_points as usize;
        let mask = match emphasis {
            Emphasis::Category { obs_idx, category } => {
                let (_name, labels, _cats, _pal) = ds.obs_categorical(obs_idx).ok()?;
                labels.iter().map(|&l| l as usize == category).collect()
            }
            Emphasis::Selection => {
                let mut mask = vec![false; n];
                for &c in &self.selection.selected {
                    if let Some(m) = mask.get_mut(c as usize) {
                        *m = true;
                    }
                }
                mask
            }
        };
        let mask = Arc::new(mask);
        self.selection.cache.mask = Some((sig, mask.clone()));
        Some((sig, mask))
    }

    /// Hover ring, tooltip and the lasso being drawn.
    pub(super) fn draw_selection_overlay(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
        ctx: &egui::Context,
    ) {
        let painter = ui.painter().with_clip_rect(response.rect);
        let accent = egui::Color32::from_rgb(255, 210, 60);

        if self.selection.lasso_path.len() >= 2 {
            let path = &self.selection.lasso_path;
            let fill = egui::Color32::from_rgba_unmultiplied(255, 210, 60, 28);
            // Fill with a fan from the first point; good enough as a preview
            // even for concave shapes, and the outline is exact.
            let mut mesh = egui::Mesh::default();
            for p in path {
                mesh.colored_vertex(*p, fill);
            }
            for i in 1..path.len() as u32 - 1 {
                mesh.add_triangle(0, i, i + 1);
            }
            painter.add(egui::Shape::mesh(mesh));
            painter.add(egui::Shape::line(
                path.clone(),
                egui::Stroke::new(1.5_f32, accent),
            ));
            painter.add(egui::Shape::dashed_line(
                &[*path.last().unwrap(), path[0]],
                egui::Stroke::new(1.0_f32, accent),
                4.0,
                4.0,
            ));
        }

        // Outline the legend entry being highlighted.
        let active = self.selection.legend_hover.or(self.selection.pinned);
        if let (Some(active), Some(ds)) = (active, self.dataset.clone()) {
            for (r, label) in &self.selection.legend_rows {
                if self.legend_category(&ds, label) == Some(active) {
                    painter.rect_stroke(
                        r.expand(2.0),
                        3.0,
                        egui::Stroke::new(1.0_f32, accent),
                        egui::StrokeKind::Outside,
                    );
                }
            }
        }

        let hovering_legend = self.selection.legend_hover.is_some();
        let busy = self.playing
            || self.video_export_running()
            || response.dragged()
            || !self.selection.lasso_path.is_empty();
        let pointer = response.hover_pos();
        let (Some(pointer), Some(ds), Some(view), true, false, false) = (
            pointer,
            self.dataset.clone(),
            self.selection.view.clone(),
            self.selection.hover_enabled,
            busy,
            hovering_legend,
        ) else {
            self.selection.hovered = None;
            self.selection.hover_key = None;
            return;
        };

        let key = HoverKey {
            pointer,
            rect: view.rect,
            center: view.camera.center,
            ppu: view.camera.pixels_per_unit,
            t: view.t,
            indices: Arc::as_ptr(&view.draw_indices) as usize,
            dataset_id: self.dataset_id,
        };
        if self.selection.hover_key.as_ref() != Some(&key) {
            let radius = (view.point_radius_pts + 3.0).max(MIN_PICK_RADIUS);
            self.selection.hovered = Projector::new(&ds, &view).and_then(|proj| {
                nearest_cell(&view.draw_indices, pointer, radius, |c| proj.screen_pos(c))
            });
            self.selection.hover_key = Some(key);
        }
        let Some(cell) = self.selection.hovered else {
            return;
        };
        if let Some(pos) = Projector::new(&ds, &view).and_then(|p| p.screen_pos(cell)) {
            let r = view.point_radius_pts.max(2.0) + 3.0;
            painter.circle_stroke(pos, r, egui::Stroke::new(2.0_f32, accent));
        }

        let gene = match &view.shown_color {
            ColorKey::Gene(g) => Some(g.clone()),
            ColorKey::Current if self.color_mode == ColorMode::Gene => self.gene_selected.clone(),
            _ => None,
        };
        let shown = view.shown_color.clone();
        let fresh = matches!(&self.selection.tooltip, Some((c, k, _)) if *c == cell && *k == shown);
        if !fresh {
            let colour_obs = match &shown {
                ColorKey::Categorical(i) | ColorKey::Continuous(i) => Some(*i),
                ColorKey::Current if self.color_mode != ColorMode::Gene => {
                    Some(self.active_obs_idx)
                }
                _ => None,
            };
            let lines = cell_details(&ds, cell, colour_obs, gene.as_deref());
            self.selection.tooltip = Some((cell, shown, lines));
        }
        let Some((_, _, lines)) = &self.selection.tooltip else {
            return;
        };
        let title = ds.cell_name(cell);
        egui::Area::new(egui::Id::new("stviz_cell_tooltip"))
            .order(egui::Order::Tooltip)
            .fixed_pos(pointer + egui::vec2(16.0, 16.0))
            .interactable(false)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.set_max_width(320.0);
                    ui.strong(title);
                    if ds.meta.obs_names.is_some() {
                        ui.weak(format!("row {cell}"));
                    }
                    egui::Grid::new("stviz_cell_tooltip_grid")
                        .num_columns(2)
                        .spacing([10.0, 2.0])
                        .show(ui, |ui| {
                            for (k, v) in lines {
                                ui.weak(k);
                                ui.label(v);
                                ui.end_row();
                            }
                        });
                });
            });
    }

    /// Selection controls in the left panel.
    pub(super) fn ui_selection_section(&mut self, ui: &mut egui::Ui, ds: &Dataset) {
        self.sync_selection_dataset();
        ui.separator();
        ui.label("Selection");
        ui.checkbox(&mut self.selection.hover_enabled, "Cell details on hover")
            .on_hover_text("Show a cell's name, annotations and current value under the pointer.");
        ui.checkbox(&mut self.selection.lasso_tool, "Lasso tool")
            .on_hover_text(
                "Drag in the view to draw a selection (without it, hold Shift while dragging). \
                 Hold Ctrl/Cmd to add to the selection, Alt to remove from it.",
            );
        if let Some((obs_idx, category)) = self.selection.pinned {
            let label = ds
                .obs_categorical(obs_idx)
                .ok()
                .and_then(|(name, _, cats, _)| cats.get(category).map(|c| format!("{name}: {c}")))
                .unwrap_or_default();
            ui.horizontal(|ui| {
                ui.label(format!("Highlighting {label}"));
                if ui.small_button("Clear").clicked() {
                    self.selection.pinned = None;
                }
            });
        }
        let n = self.selection.selected.len();
        if n == 0 {
            ui.weak(
                "Shift + drag in the view to select cells. Click a legend entry to highlight it.",
            );
        } else {
            ui.label(format!("{n} cells selected"));
            ui.checkbox(&mut self.selection.show_selection, "Dim unselected cells");
            ui.horizontal(|ui| {
                if ui
                    .button("Export CSV...")
                    .on_hover_text(
                        "Save the selected cells with their name, coordinates in the space on \
                         screen, all annotations and the gene shown (if colouring by a gene).",
                    )
                    .clicked()
                {
                    self.export_selection_csv(ds);
                }
                if ui.button("Clear selection").clicked() {
                    self.selection.selected.clear();
                    self.selection.selected_version =
                        self.selection.selected_version.wrapping_add(1);
                    self.selection.status = None;
                }
            });
        }
        if let Some(status) = &self.selection.status {
            ui.weak(status);
        }
    }

    fn export_selection_csv(&mut self, ds: &Dataset) {
        let Some(view) = self.selection.view.clone() else {
            return;
        };
        let space_idx = view.shown_space;
        let gene = match &view.shown_color {
            ColorKey::Gene(g) => Some(g.clone()),
            ColorKey::Current if self.color_mode == ColorMode::Gene => self.gene_selected.clone(),
            _ => None,
        };
        let space = ds
            .meta
            .spaces
            .get(space_idx)
            .map(|s| s.name.clone())
            .unwrap_or_default();
        let n = self.selection.selected.len();
        let file_name = format!("selection_{n}_cells_{}.csv", sanitize_file_part(&space));
        let Some(path) = rfd::FileDialog::new()
            .add_filter("CSV", &["csv"])
            .set_title("Export selected cells")
            .set_directory(self.dialog_dir())
            .set_file_name(file_name)
            .save_file()
        else {
            return;
        };
        let path = if path.extension().is_none() {
            path.with_extension("csv")
        } else {
            path
        };
        self.selection.status = Some(
            match write_selection_csv_file(
                ds,
                &self.selection.selected,
                space_idx,
                gene.as_deref(),
                &path,
            ) {
                Ok(()) => {
                    self.remember_dir_of(&path);
                    format!("Saved {n} cells to {}", path.display())
                }
                Err(e) => format!("Export failed: {e:#}"),
            },
        );
    }
}

/// The cell under `pointer` (closest within `radius`), preferring cells
/// drawn later (on top) when two are equally close.
fn nearest_cell(
    cells: &[u32],
    pointer: egui::Pos2,
    radius: f32,
    pos: impl Fn(u32) -> Option<egui::Pos2>,
) -> Option<u32> {
    let r2 = radius * radius;
    let mut best: Option<(f32, u32)> = None;
    for &c in cells {
        let Some(p) = pos(c) else { continue };
        let d2 = p.distance_sq(pointer);
        if d2 <= r2 && best.is_none_or(|(bd, _)| d2 <= bd) {
            best = Some((d2, c));
        }
    }
    best.map(|(_, c)| c)
}

/// Cells whose screen position is inside the polygon `path`, sorted.
fn lasso_hits(
    cells: &[u32],
    path: &[egui::Pos2],
    pos: impl Fn(u32) -> Option<egui::Pos2>,
) -> Vec<u32> {
    let bounds = egui::Rect::from_points(path);
    let mut hits: Vec<u32> = cells
        .iter()
        .copied()
        .filter(|&c| pos(c).is_some_and(|p| bounds.contains(p) && point_in_polygon(p, path)))
        .collect();
    hits.sort_unstable();
    hits.dedup();
    hits
}

/// Even-odd rule point-in-polygon test.
fn point_in_polygon(p: egui::Pos2, poly: &[egui::Pos2]) -> bool {
    let mut inside = false;
    let mut j = poly.len().wrapping_sub(1);
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[j]);
        if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Merge a lasso result (sorted) into the selection (sorted).
fn combine_selection(current: &[u32], hits: Vec<u32>, mode: LassoMode) -> Vec<u32> {
    match mode {
        LassoMode::Replace => hits,
        LassoMode::Add => {
            let mut out = Vec::with_capacity(current.len() + hits.len());
            out.extend_from_slice(current);
            out.extend(hits);
            out.sort_unstable();
            out.dedup();
            out
        }
        LassoMode::Remove => current
            .iter()
            .copied()
            .filter(|c| hits.binary_search(c).is_err())
            .collect(),
    }
}

/// Blend cells outside `mask` towards the background, keeping alpha (so
/// filtered-out cells stay hidden).
fn dim_colors(colors: &[u32], mask: &[bool], bg: [u8; 3]) -> Vec<u32> {
    let mix = |c: u32, shift: u32, b: u8| -> u32 {
        let v = ((c >> shift) & 0xFF) as f32;
        let out = v * DIM_KEEP + b as f32 * (1.0 - DIM_KEEP);
        (out.round() as u32).min(255) << shift
    };
    colors
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            if mask.get(i).copied().unwrap_or(false) {
                c
            } else {
                mix(c, 0, bg[0]) | mix(c, 8, bg[1]) | mix(c, 16, bg[2]) | (c & 0xFF00_0000)
            }
        })
        .collect()
}

/// Draw indices with the emphasised cells moved to the end (drawn on top),
/// keeping the order within each group.
fn emphasised_last(indices: &[u32], mask: &[bool]) -> Vec<u32> {
    let on = |c: &u32| mask.get(*c as usize).copied().unwrap_or(false);
    let mut out: Vec<u32> = indices.iter().copied().filter(|c| !on(c)).collect();
    out.extend(indices.iter().copied().filter(on));
    out
}

/// An id for derived GPU data, kept apart from the app's counters by the top bit.
fn derived_id(key: &impl Hash) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut h);
    h.finish() | (1 << 63)
}

fn format_value(v: f32) -> String {
    if !v.is_finite() {
        return "NA".to_string();
    }
    if v == v.trunc() && v.abs() < 1e9 {
        return format!("{}", v as i64);
    }
    let s = format!("{v:.4}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// Tooltip rows for a cell: the colouring first, then every annotation.
fn cell_details(
    ds: &Dataset,
    cell: u32,
    colour_obs: Option<usize>,
    gene: Option<&str>,
) -> Vec<(String, String)> {
    let mut rows = Vec::new();
    if let Some(gene) = gene {
        if let Some(gid) = ds.find_gene(gene) {
            if let Ok(v) = ds.gene_value(gid, cell) {
                rows.push((gene.to_string(), format_value(v)));
            }
        }
    }
    let obs_value = |i: usize| -> Option<(String, String)> {
        match ds.meta.obs.get(i)? {
            ObsMeta::Categorical { .. } => {
                let (name, labels, cats, _) = ds.obs_categorical(i).ok()?;
                let label = labels.get(cell as usize)?;
                let value = cats
                    .get(*label as usize)
                    .cloned()
                    .unwrap_or_else(|| "NA".into());
                Some((name.to_string(), value))
            }
            ObsMeta::Continuous { .. } => {
                let (name, vals) = ds.obs_continuous(i).ok()?;
                Some((name.to_string(), format_value(*vals.get(cell as usize)?)))
            }
        }
    };
    let order = colour_obs
        .into_iter()
        .chain((0..ds.meta.obs.len()).filter(|i| Some(*i) != colour_obs));
    let mut hidden = 0;
    for i in order {
        let Some(row) = obs_value(i) else { continue };
        if rows.len() >= MAX_TOOLTIP_FIELDS {
            hidden += 1;
        } else {
            rows.push(row);
        }
    }
    if hidden > 0 {
        rows.push((String::new(), format!("+{hidden} more")));
    }
    rows
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn sanitize_file_part(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn write_selection_csv_file(
    ds: &Dataset,
    cells: &[u32],
    space_idx: usize,
    gene: Option<&str>,
    path: &Path,
) -> anyhow::Result<()> {
    use anyhow::Context;
    // Write next to the target and rename, so a failed export never leaves
    // half a file under the chosen name.
    let partial = path.with_extension("csv.partial");
    let result = (|| -> anyhow::Result<()> {
        let file = std::fs::File::create(&partial)
            .with_context(|| format!("create {}", partial.display()))?;
        let mut out = std::io::BufWriter::new(file);
        write_selection_csv(ds, cells, space_idx, gene, &mut out)?;
        out.flush()?;
        drop(out);
        std::fs::rename(&partial, path).with_context(|| format!("write {}", path.display()))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&partial);
    }
    result
}

/// One row per cell: name, row number, coordinates in `space_idx`, every
/// annotation and, optionally, one gene's expression.
pub(super) fn write_selection_csv(
    ds: &Dataset,
    cells: &[u32],
    space_idx: usize,
    gene: Option<&str>,
    out: &mut impl Write,
) -> anyhow::Result<()> {
    let space = ds
        .meta
        .spaces
        .get(space_idx)
        .ok_or_else(|| anyhow::anyhow!("invalid space"))?;
    let coords = ds.space_f32_2d(space_idx)?;
    let mut header = vec![
        "cell_id".to_string(),
        "row".to_string(),
        format!("{}_x", space.name),
        format!("{}_y", space.name),
    ];
    enum Col<'a> {
        Cat(&'a [u32], &'a [String]),
        Cont(&'a [f32]),
    }
    let mut cols = Vec::new();
    for i in 0..ds.meta.obs.len() {
        if let Ok((name, labels, cats, _)) = ds.obs_categorical(i) {
            header.push(name.to_string());
            cols.push(Col::Cat(labels, cats));
        } else if let Ok((name, vals)) = ds.obs_continuous(i) {
            header.push(name.to_string());
            cols.push(Col::Cont(vals));
        }
    }
    let gene_vals = match gene.and_then(|g| ds.find_gene(g).map(|id| (g, id))) {
        Some((g, id)) => {
            header.push(g.to_string());
            Some(ds.gene_vector_csc(id)?)
        }
        None => None,
    };
    let line: Vec<String> = header.iter().map(|h| csv_field(h)).collect();
    writeln!(out, "{}", line.join(","))?;

    let num = |v: f32| {
        if v.is_finite() {
            v.to_string()
        } else {
            String::new()
        }
    };
    for &c in cells {
        let i = c as usize;
        let mut row = vec![
            csv_field(&ds.cell_name(c)),
            c.to_string(),
            num(coords.get(2 * i).copied().unwrap_or(f32::NAN)),
            num(coords.get(2 * i + 1).copied().unwrap_or(f32::NAN)),
        ];
        for col in &cols {
            row.push(match col {
                Col::Cat(labels, cats) => labels
                    .get(i)
                    .and_then(|l| cats.get(*l as usize))
                    .map(|s| csv_field(s))
                    .unwrap_or_default(),
                Col::Cont(vals) => num(vals.get(i).copied().unwrap_or(f32::NAN)),
            });
        }
        if let Some(g) = &gene_vals {
            row.push(num(g.get(i).copied().unwrap_or(f32::NAN)));
        }
        writeln!(out, "{}", row.join(","))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::test_support::write_stviz;

    fn pos2(x: f32, y: f32) -> egui::Pos2 {
        egui::pos2(x, y)
    }

    #[test]
    fn point_in_polygon_handles_concave_shapes() {
        // A "U" shape: the notch at the top middle is outside.
        let u = [
            pos2(0.0, 0.0),
            pos2(3.0, 0.0),
            pos2(3.0, 3.0),
            pos2(2.0, 3.0),
            pos2(2.0, 1.0),
            pos2(1.0, 1.0),
            pos2(1.0, 3.0),
            pos2(0.0, 3.0),
        ];
        assert!(point_in_polygon(pos2(0.5, 2.0), &u));
        assert!(point_in_polygon(pos2(1.5, 0.5), &u));
        assert!(!point_in_polygon(pos2(1.5, 2.0), &u));
        assert!(!point_in_polygon(pos2(4.0, 1.0), &u));
    }

    #[test]
    fn lasso_and_selection_modes() {
        let square = [
            pos2(0.0, 0.0),
            pos2(10.0, 0.0),
            pos2(10.0, 10.0),
            pos2(0.0, 10.0),
        ];
        // Cell i sits at (i, i); cell 3 has no position (hidden).
        let pos = |c: u32| (c != 3).then(|| pos2(c as f32 * 3.0, c as f32 * 3.0));
        let hits = lasso_hits(&[5, 0, 1, 2, 3, 4], &square, pos);
        assert_eq!(hits, vec![0, 1, 2]);

        assert_eq!(
            combine_selection(&[1, 7], hits.clone(), LassoMode::Replace),
            vec![0, 1, 2]
        );
        assert_eq!(
            combine_selection(&[1, 7], hits.clone(), LassoMode::Add),
            vec![0, 1, 2, 7]
        );
        assert_eq!(combine_selection(&[1, 7], hits, LassoMode::Remove), vec![7]);
    }

    #[test]
    fn nearest_cell_prefers_the_closest_then_the_topmost() {
        let pos = |c: u32| Some(pos2([0.0, 5.0, 5.0][c as usize], 0.0));
        assert_eq!(nearest_cell(&[0, 1, 2], pos2(1.0, 0.0), 6.0, pos), Some(0));
        assert_eq!(nearest_cell(&[0, 1, 2], pos2(5.0, 0.0), 6.0, pos), Some(2));
        assert_eq!(nearest_cell(&[0, 1, 2], pos2(20.0, 0.0), 6.0, pos), None);
    }

    #[test]
    fn dimming_keeps_emphasised_cells_and_alpha() {
        let colors = [0xFF00_00FF, 0x0000_FF00];
        let out = dim_colors(&colors, &[true, false], [0, 0, 0]);
        assert_eq!(out[0], 0xFF00_00FF);
        // Green dimmed towards black, alpha 0 kept (a filtered cell stays hidden).
        assert_eq!(out[1] >> 24, 0);
        assert!((out[1] >> 8) & 0xFF < 0x40);
        assert_eq!(
            emphasised_last(&[0, 1, 2, 3], &[false, true, false, true]),
            vec![0, 2, 1, 3]
        );
    }

    #[test]
    fn csv_has_names_coordinates_annotations_and_gene() {
        let dir = std::env::temp_dir().join(format!("stviz_sel_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sel.stviz");
        write_stviz(&path, 4, None);
        let ds = Dataset::load(&path).unwrap();
        let mut buf = Vec::new();
        write_selection_csv(&ds, &[1, 3], 1, Some("G0"), &mut buf).unwrap();
        let text = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "cell_id,row,X_umap_x,X_umap_y,leiden,score,G0");
        assert_eq!(lines[1], "1,1,-2,-3,1,0.5,2");
        assert_eq!(lines[2], "3,3,-6,-7,0,1.5,4");
        assert_eq!(lines.len(), 3);

        let out = dir.join("sel.csv");
        write_selection_csv_file(&ds, &[0], 0, None, &out).unwrap();
        let saved = std::fs::read_to_string(&out).unwrap();
        assert!(saved.starts_with("cell_id,row,spatial_x,spatial_y,leiden,score\n0,0,0,1,0,0\n"));
        assert!(!dir.join("sel.csv.partial").exists());
    }

    #[test]
    fn csv_fields_are_quoted_when_needed() {
        assert_eq!(csv_field("plain"), "plain");
        assert_eq!(csv_field("a,b"), "\"a,b\"");
        assert_eq!(csv_field("say \"hi\""), "\"say \"\"hi\"\"\"");
    }

    #[test]
    fn values_are_short() {
        assert_eq!(format_value(3.0), "3");
        assert_eq!(format_value(0.125), "0.125");
        assert_eq!(format_value(1.0 / 3.0), "0.3333");
        assert_eq!(format_value(f32::NAN), "NA");
    }
}
