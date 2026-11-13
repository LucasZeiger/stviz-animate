//! Dataset/workspace file dialogs and workspace save/restore.

use super::{
    overlay::OverlaySettings, AdvancedCard, AdvancedCardFilter, AdvancedConnection,
    CategoricalPalette, ColorKey, ColorMode, EaseMode, ExportEncodingSpeed, ExportQuality,
    ExportVideoQuality, KeyframeFilter, PlaybackMode, SampleGridLabelMode, StvizApp, UiTheme,
    MAX_FILTER_CATEGORIES,
};
use crate::data::{Dataset, ObsMeta};
use anyhow::Context as _;
use eframe::egui;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
};

#[derive(Serialize, Deserialize)]
struct WorkspaceFile {
    version: u32,
    dataset_path: PathBuf,
    /// File name of the dataset, used to find it next to the workspace file
    /// when the saved path no longer exists (e.g. on another computer).
    #[serde(default)]
    dataset_file_name: Option<String>,
    dataset_points: u32,
    camera_center: [f32; 2],
    camera_pixels_per_unit: f32,
    keyframes: Vec<WorkspaceKeyframe>,
    key_times: Vec<f32>,
    color_path_enabled: bool,
    color_mode: ColorMode,
    active_obs: Option<String>,
    gene_selected: Option<String>,
    categorical_palette: CategoricalPalette,
    category_overrides: Vec<WorkspaceOverrides>,
    global_filters: Vec<WorkspaceFilter>,
    playback_mode: PlaybackMode,
    ease_mode: EaseMode,
    speed: f32,
    t: f32,
    point_radius_px: f32,
    max_draw_points: usize,
    #[serde(default)]
    export_encoding_speed: ExportEncodingSpeed,
    background_rgba: [u8; 4],
    ui_scale: f32,
    ui_theme: UiTheme,
    show_axes: bool,
    show_stats: bool,
    sample_grid: WorkspaceSampleGrid,
    #[serde(default)]
    fast_render: Option<bool>,
    #[serde(default)]
    clip_outliers: Option<bool>,
    #[serde(default)]
    export: Option<WorkspaceExport>,
    #[serde(default)]
    advanced: Option<WorkspaceAdvanced>,
    #[serde(default)]
    overlay: Option<OverlaySettings>,
}

#[derive(Serialize, Deserialize)]
struct WorkspaceExport {
    fps: u32,
    duration_sec: f32,
    quality: ExportQuality,
    video_quality: ExportVideoQuality,
    run_ffmpeg: bool,
    keep_frames: bool,
    name: String,
}

/// Layout of the advanced timeline. Each card's space, color and filter come
/// from the keyframe with the same index.
#[derive(Serialize, Deserialize)]
struct WorkspaceAdvanced {
    open: bool,
    grid_mode: bool,
    grid_size: f32,
    cards: Vec<WorkspaceCard>,
    connections: Vec<[usize; 2]>,
}

#[derive(Serialize, Deserialize)]
struct WorkspaceCard {
    duration_sec: f32,
    pos: [f32; 2],
    size: [f32; 2],
    in_enabled: bool,
    out_enabled: bool,
}

#[derive(Serialize, Deserialize)]
struct WorkspaceKeyframe {
    space: String,
    color: WorkspaceColor,
    filter: Option<WorkspaceFilter>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", content = "value")]
enum WorkspaceColor {
    Current,
    Categorical(String),
    Continuous(String),
    Gene(String),
}

#[derive(Serialize, Deserialize)]
struct WorkspaceFilter {
    field: String,
    enabled_categories: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct WorkspaceOverrides {
    field: String,
    colors: HashMap<String, u32>,
}

#[derive(Serialize, Deserialize)]
struct WorkspaceSampleGrid {
    enabled: bool,
    obs: Option<String>,
    space: Option<String>,
    use_filter: bool,
    padding: f32,
    labels_enabled: bool,
    label_mode: SampleGridLabelMode,
    custom_labels: Vec<String>,
}

impl StvizApp {
    pub(super) fn open_dataset_dialog(&mut self) -> anyhow::Result<()> {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("stviz", &["stviz"])
            .set_title("Open .stviz")
            .set_directory(self.dialog_dir())
            .pick_file()
        else {
            return Ok(());
        };

        self.open_data_file(&path);
        Ok(())
    }

    fn workspace_filter(ds: &Dataset, obs_idx: usize, enabled: &[bool]) -> Option<WorkspaceFilter> {
        let (name, _labels, categories, _pal) = ds.obs_categorical(obs_idx).ok()?;
        Some(WorkspaceFilter {
            field: name.to_string(),
            enabled_categories: categories
                .iter()
                .zip(enabled)
                .filter_map(|(name, enabled)| enabled.then_some(name.clone()))
                .collect(),
        })
    }

    fn resolve_workspace_filter(
        ds: &Dataset,
        saved: &WorkspaceFilter,
    ) -> anyhow::Result<KeyframeFilter> {
        let obs_idx = ds
            .meta
            .obs
            .iter()
            .position(
                |obs| matches!(obs, ObsMeta::Categorical { name, .. } if name == &saved.field),
            )
            .ok_or_else(|| anyhow::anyhow!("categorical field not found: {}", saved.field))?;
        let (_name, _labels, categories, _pal) = ds.obs_categorical(obs_idx)?;
        if categories.len() > MAX_FILTER_CATEGORIES {
            anyhow::bail!(
                "categorical field '{}' has {} categories; filter limit is {}",
                saved.field,
                categories.len(),
                MAX_FILTER_CATEGORIES
            );
        }
        let enabled_names: HashSet<&str> = saved
            .enabled_categories
            .iter()
            .map(String::as_str)
            .collect();
        Ok(KeyframeFilter {
            obs_idx,
            enabled: categories
                .iter()
                .map(|name| enabled_names.contains(name.as_str()))
                .collect(),
        })
    }

    fn build_workspace(&self) -> anyhow::Result<WorkspaceFile> {
        let ds = self
            .dataset
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no dataset loaded"))?;
        let dataset_path = self
            .dataset_path
            .clone()
            .ok_or_else(|| anyhow::anyhow!("dataset path unavailable"))?;

        let mut keyframes = Vec::with_capacity(self.space_path.len());
        for (i, &space_idx) in self.space_path.iter().enumerate() {
            let space = ds
                .meta
                .spaces
                .get(space_idx)
                .ok_or_else(|| anyhow::anyhow!("invalid keyframe space"))?
                .name
                .clone();
            let color = match self.color_path.get(i).unwrap_or(&ColorKey::Current) {
                ColorKey::Current => WorkspaceColor::Current,
                ColorKey::Categorical(idx) => match ds.meta.obs.get(*idx) {
                    Some(ObsMeta::Categorical { name, .. }) => {
                        WorkspaceColor::Categorical(name.clone())
                    }
                    _ => return Err(anyhow::anyhow!("invalid categorical color field")),
                },
                ColorKey::Continuous(idx) => match ds.meta.obs.get(*idx) {
                    Some(ObsMeta::Continuous { name, .. }) => {
                        WorkspaceColor::Continuous(name.clone())
                    }
                    _ => return Err(anyhow::anyhow!("invalid continuous color field")),
                },
                ColorKey::Gene(name) => WorkspaceColor::Gene(name.clone()),
            };
            let filter = self
                .filter_path
                .get(i)
                .and_then(Option::as_ref)
                .and_then(|f| Self::workspace_filter(ds, f.obs_idx, &f.enabled));
            keyframes.push(WorkspaceKeyframe {
                space,
                color,
                filter,
            });
        }

        let mut category_overrides = Vec::new();
        for (&obs_idx, overrides) in &self.category_overrides {
            if let Ok((name, _labels, categories, _pal)) = ds.obs_categorical(obs_idx) {
                let colors = categories
                    .iter()
                    .zip(overrides)
                    .filter_map(|(category, color)| color.map(|c| (category.clone(), c)))
                    .collect();
                category_overrides.push(WorkspaceOverrides {
                    field: name.to_string(),
                    colors,
                });
            }
        }

        let global_filters = self
            .active_filters
            .iter()
            .filter_map(|idx| {
                self.category_state
                    .get(idx)
                    .and_then(|enabled| Self::workspace_filter(ds, *idx, enabled))
            })
            .collect();
        let active_obs = ds.meta.obs.get(self.active_obs_idx).map(|obs| match obs {
            ObsMeta::Categorical { name, .. } | ObsMeta::Continuous { name, .. } => name.clone(),
        });
        let bg = self.background_color;

        let advanced = (!self.advanced_cards.is_empty()).then(|| WorkspaceAdvanced {
            open: self.advanced_timeline_open,
            grid_mode: self.advanced_grid_mode,
            grid_size: self.advanced_grid_size,
            cards: self
                .advanced_cards
                .iter()
                .map(|card| WorkspaceCard {
                    duration_sec: card.duration_sec,
                    pos: [card.pos.x, card.pos.y],
                    size: [card.size.x, card.size.y],
                    in_enabled: card.in_enabled,
                    out_enabled: card.out_enabled,
                })
                .collect(),
            connections: self
                .advanced_connections
                .iter()
                .map(|c| [c.from, c.to])
                .collect(),
        });

        Ok(WorkspaceFile {
            version: 1,
            dataset_file_name: dataset_path
                .file_name()
                .map(|n| n.to_string_lossy().to_string()),
            dataset_path,
            dataset_points: ds.meta.n_points,
            camera_center: self.camera.center,
            camera_pixels_per_unit: self.camera.pixels_per_unit,
            keyframes,
            key_times: self.key_times.clone(),
            color_path_enabled: self.color_path_enabled,
            color_mode: self.color_mode,
            active_obs,
            gene_selected: self.gene_selected.clone(),
            categorical_palette: self.categorical_palette,
            category_overrides,
            global_filters,
            playback_mode: self.playback_mode,
            ease_mode: self.ease_mode,
            speed: self.speed,
            t: self.t,
            point_radius_px: self.point_radius_px,
            max_draw_points: self.max_draw_points,
            export_encoding_speed: self.export_encoding_speed,
            background_rgba: [bg.r(), bg.g(), bg.b(), bg.a()],
            ui_scale: self.ui_scale,
            ui_theme: self.ui_theme,
            show_axes: self.show_axes,
            show_stats: self.show_stats,
            sample_grid: WorkspaceSampleGrid {
                enabled: self.sample_grid_enabled,
                obs: self
                    .sample_grid_obs_idx
                    .and_then(|idx| ds.meta.obs.get(idx))
                    .map(|obs| match obs {
                        ObsMeta::Categorical { name, .. } | ObsMeta::Continuous { name, .. } => {
                            name.clone()
                        }
                    }),
                space: self
                    .sample_grid_space_idx
                    .and_then(|idx| ds.meta.spaces.get(idx))
                    .map(|space| space.name.clone()),
                use_filter: self.sample_grid_use_filter,
                padding: self.sample_grid_padding,
                labels_enabled: self.sample_grid_labels_enabled,
                label_mode: self.sample_grid_label_mode,
                custom_labels: self.sample_grid_custom_labels.clone(),
            },
            fast_render: Some(self.fast_render),
            clip_outliers: Some(self.clip_outliers),
            export: Some(WorkspaceExport {
                fps: self.export_fps,
                duration_sec: self.export_duration_sec,
                quality: self.export_quality,
                video_quality: self.export_video_quality,
                run_ffmpeg: self.export_run_ffmpeg,
                keep_frames: self.export_keep_frames,
                name: self.export_name.clone(),
            }),
            advanced,
            overlay: Some(self.overlay.clone()),
        })
    }

    pub(super) fn save_workspace_dialog(&mut self) -> anyhow::Result<()> {
        let workspace = self.build_workspace()?;
        let dataset_path = workspace.dataset_path.clone();
        let stem = dataset_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("stviz")
            .to_string();
        let mut dialog = rfd::FileDialog::new()
            .add_filter("stviz workspace", &["json"])
            .set_title("Save workspace")
            .set_file_name(format!("{stem}.workspace.json"));
        if let Some(parent) = dataset_path.parent() {
            dialog = dialog.set_directory(parent);
        }
        let Some(path) = dialog.save_file() else {
            return Ok(());
        };
        self.remember_dir_of(&path);
        let json = serde_json::to_string_pretty(&workspace)?;
        std::fs::write(&path, json)
            .with_context(|| format!("write workspace: {}", path.display()))?;
        self.workspace_status = Some(format!("Saved workspace: {}", path.display()));
        Ok(())
    }

    pub(super) fn open_workspace_dialog(&mut self) -> anyhow::Result<()> {
        if self.convert_running() || self.dataset_load.is_some() || self.export_busy() {
            anyhow::bail!("wait for the running conversion, load or export to finish");
        }
        let Some(path) = rfd::FileDialog::new()
            .add_filter("stviz workspace", &["json"])
            .set_title("Open workspace")
            .set_directory(self.dialog_dir())
            .pick_file()
        else {
            return Ok(());
        };
        self.remember_dir_of(&path);
        let json = std::fs::read_to_string(&path)
            .with_context(|| format!("read workspace: {}", path.display()))?;
        let workspace: WorkspaceFile =
            serde_json::from_str(&json).context("not a valid workspace file")?;
        if workspace.version != 1 {
            return Err(anyhow::anyhow!(
                "unsupported workspace version {}",
                workspace.version
            ));
        }
        let dataset_path = resolve_dataset_path(&path, &workspace)?;
        self.apply_workspace(workspace, &dataset_path)?;
        self.workspace_status = Some(format!("Opened workspace: {}", path.display()));
        Ok(())
    }

    /// Load the workspace's dataset and restore its state. Everything that
    /// can fail is checked before the current session is touched, so a bad
    /// workspace leaves the app as it was.
    fn apply_workspace(
        &mut self,
        workspace: WorkspaceFile,
        dataset_path: &Path,
    ) -> anyhow::Result<()> {
        let ds = Dataset::load(dataset_path).context("load dataset")?;
        if ds.meta.n_points != workspace.dataset_points {
            return Err(anyhow::anyhow!(
                "workspace expects {} cells, dataset has {}",
                workspace.dataset_points,
                ds.meta.n_points
            ));
        }

        let find_categorical = |field: &str| {
            ds.meta
                .obs
                .iter()
                .position(|obs| matches!(obs, ObsMeta::Categorical { name, .. } if name == field))
        };

        let mut spaces = Vec::with_capacity(workspace.keyframes.len());
        let mut colors = Vec::with_capacity(workspace.keyframes.len());
        let mut filters = Vec::with_capacity(workspace.keyframes.len());
        for key in &workspace.keyframes {
            let space_idx = ds
                .meta
                .spaces
                .iter()
                .position(|space| space.name == key.space)
                .ok_or_else(|| anyhow::anyhow!("space not found: {}", key.space))?;
            let color = match &key.color {
                WorkspaceColor::Current => ColorKey::Current,
                WorkspaceColor::Gene(name) => ColorKey::Gene(name.clone()),
                WorkspaceColor::Categorical(name) => ColorKey::Categorical(
                    find_categorical(name)
                        .ok_or_else(|| anyhow::anyhow!("categorical field not found: {name}"))?,
                ),
                WorkspaceColor::Continuous(name) => ColorKey::Continuous(
                    ds.meta
                        .obs
                        .iter()
                        .position(
                            |obs| matches!(obs, ObsMeta::Continuous { name: n, .. } if n == name),
                        )
                        .ok_or_else(|| anyhow::anyhow!("continuous field not found: {name}"))?,
                ),
            };
            spaces.push(space_idx);
            colors.push(color);
            filters.push(
                key.filter
                    .as_ref()
                    .map(|filter| Self::resolve_workspace_filter(&ds, filter))
                    .transpose()?,
            );
        }
        let global_filters = workspace
            .global_filters
            .iter()
            .map(|saved| Self::resolve_workspace_filter(&ds, saved))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let mut category_overrides = HashMap::new();
        for saved in &workspace.category_overrides {
            if let Some(obs_idx) = find_categorical(&saved.field) {
                let (_name, _labels, categories, _pal) = ds.obs_categorical(obs_idx)?;
                category_overrides.insert(
                    obs_idx,
                    categories
                        .iter()
                        .map(|name| saved.colors.get(name).copied())
                        .collect::<Vec<_>>(),
                );
            }
        }

        // Nothing below can fail in a way that depends on the workspace file.
        self.install_dataset(ds.clone(), dataset_path)?;

        // Datasets with a single space have no timeline; keep the defaults.
        if spaces.len() >= 2 {
            self.space_path = spaces;
            self.color_path = colors;
            self.filter_path = filters;
            self.key_times = workspace.key_times;
            if self.key_times.len() != self.space_path.len()
                || self.key_times.iter().any(|t| !t.is_finite())
                || self.key_times.windows(2).any(|pair| pair[0] > pair[1])
            {
                self.key_times.clear();
                self.ensure_key_times_len(self.space_path.len());
            }
            self.key_collapsed = vec![false; self.space_path.len()];
            self.from_space = self.space_path[0];
            self.to_space = self.space_path[self.space_path.len() - 1];
        }
        self.color_path_enabled = workspace.color_path_enabled;
        self.color_mode = workspace.color_mode;
        self.categorical_palette = workspace.categorical_palette;
        self.playback_mode = workspace.playback_mode;
        self.ease_mode = workspace.ease_mode;
        if workspace.speed.is_finite() && workspace.speed > 0.0 {
            self.speed = workspace.speed;
        }
        self.t = if workspace.t.is_finite() {
            workspace.t.clamp(0.0, 1.0)
        } else {
            0.0
        };
        if workspace.point_radius_px.is_finite() && workspace.point_radius_px > 0.0 {
            self.point_radius_px = workspace.point_radius_px;
        }
        self.max_draw_points = workspace.max_draw_points;
        self.export_encoding_speed = workspace.export_encoding_speed;
        self.background_color = egui::Color32::from_rgba_unmultiplied(
            workspace.background_rgba[0],
            workspace.background_rgba[1],
            workspace.background_rgba[2],
            workspace.background_rgba[3],
        );
        if workspace.ui_scale.is_finite() {
            self.ui_scale = workspace.ui_scale.clamp(0.5, 3.0);
        }
        self.ui_theme = workspace.ui_theme;
        self.show_axes = workspace.show_axes;
        self.show_stats = workspace.show_stats;
        self.overlay = workspace.overlay.clone().unwrap_or_default();
        if workspace.camera_center.iter().all(|v| v.is_finite())
            && workspace.camera_pixels_per_unit.is_finite()
        {
            self.camera.center = workspace.camera_center;
            self.camera.pixels_per_unit = workspace.camera_pixels_per_unit.clamp(1e-6, 1e9);
        }
        if let Some(fast_render) = workspace.fast_render {
            self.fast_render = fast_render;
        }
        if let Some(clip) = workspace.clip_outliers {
            self.clip_outliers = clip;
        }
        if let Some(export) = workspace.export {
            self.export_fps = export.fps.clamp(1, 240);
            if export.duration_sec.is_finite() && export.duration_sec > 0.0 {
                self.export_duration_sec = export.duration_sec;
            }
            self.export_quality = export.quality;
            self.export_video_quality = export.video_quality;
            self.export_run_ffmpeg = export.run_ffmpeg;
            self.export_keep_frames = export.keep_frames;
            if !export.name.trim().is_empty() {
                self.export_name = export.name;
            }
        }

        if let Some(name) = workspace.active_obs {
            if let Some(idx) = ds.meta.obs.iter().position(|obs| match obs {
                ObsMeta::Categorical { name: n, .. } | ObsMeta::Continuous { name: n, .. } => {
                    n == &name
                }
            }) {
                self.active_obs_idx = idx;
            }
        }
        self.category_overrides = category_overrides;
        self.active_filters.clear();
        self.category_state.clear();
        for filter in global_filters {
            self.active_filters.insert(filter.obs_idx);
            self.category_state.insert(filter.obs_idx, filter.enabled);
        }
        self.load_filter_state(&ds);

        self.sample_grid_enabled = workspace.sample_grid.enabled;
        self.sample_grid_obs_idx = workspace
            .sample_grid
            .obs
            .as_ref()
            .and_then(|name| find_categorical(name));
        self.sample_grid_space_idx = workspace
            .sample_grid
            .space
            .as_ref()
            .and_then(|name| ds.meta.spaces.iter().position(|space| &space.name == name));
        self.sample_grid_use_filter = workspace.sample_grid.use_filter;
        self.sample_grid_padding = workspace.sample_grid.padding;
        self.sample_grid_labels_enabled = workspace.sample_grid.labels_enabled;
        self.sample_grid_label_mode = workspace.sample_grid.label_mode;
        self.sample_grid_custom_labels = workspace.sample_grid.custom_labels;
        self.sample_grid_custom_labels_obs_idx = self.sample_grid_obs_idx;

        self.restore_advanced_timeline(workspace.advanced);
        self.color_cache.clear();
        self.filtered_color_cache.clear();
        self.category_palette_cache = None;
        self.recompute_colors_and_filters()?;
        self.gene_selected = workspace.gene_selected;
        if self.color_mode == ColorMode::Gene {
            if let Some(gene) = self.gene_selected.clone() {
                self.gene_query = gene.clone();
                self.apply_gene_colors(&ds, &gene);
            }
        }
        self.mark_grid_dirty();
        Ok(())
    }

    /// Rebuild the advanced timeline cards from the saved layout. The cards
    /// mirror the keyframes, so a layout with a different card count is dropped.
    fn restore_advanced_timeline(&mut self, saved: Option<WorkspaceAdvanced>) {
        self.advanced_cards.clear();
        self.advanced_connections.clear();
        self.advanced_selected_card = None;
        self.advanced_selected_cards.clear();
        self.advanced_preview_card = None;
        self.advanced_timeline_open = false;
        let Some(saved) = saved else {
            return;
        };
        if saved.cards.is_empty() || saved.cards.len() != self.space_path.len() {
            return;
        }
        let finite = |v: [f32; 2]| v.iter().all(|x| x.is_finite());
        if saved
            .cards
            .iter()
            .any(|c| !finite(c.pos) || !finite(c.size) || !c.duration_sec.is_finite())
        {
            return;
        }
        let n = saved.cards.len();
        for (i, card) in saved.cards.into_iter().enumerate() {
            self.advanced_cards.push(AdvancedCard {
                id: i as u64 + 1,
                space_idx: self.space_path[i],
                color_key: self.color_path.get(i).cloned().unwrap_or(ColorKey::Current),
                duration_sec: card.duration_sec.max(0.1),
                pos: egui::pos2(card.pos[0], card.pos[1]),
                size: egui::vec2(card.size[0], card.size[1]),
                in_enabled: card.in_enabled,
                out_enabled: card.out_enabled,
                filter: self
                    .filter_path
                    .get(i)
                    .and_then(Option::as_ref)
                    .map(|filter| AdvancedCardFilter {
                        obs_idx: filter.obs_idx,
                        enabled: filter.enabled.clone(),
                        cached_indices: None,
                        cached_indices_id: 0,
                        cached_dataset_id: 0,
                        cached_max_draw: 0,
                    }),
            });
        }
        self.advanced_next_id = n as u64 + 1;
        self.advanced_connections = saved
            .connections
            .into_iter()
            .filter(|[from, to]| *from < n && *to < n && from != to)
            .map(|[from, to]| AdvancedConnection { from, to })
            .collect();
        self.advanced_grid_mode = saved.grid_mode;
        if saved.grid_size.is_finite() && saved.grid_size > 0.0 {
            self.advanced_grid_size = saved.grid_size;
        }
        self.advanced_timeline_open = saved.open;
    }
}

/// Find the workspace's dataset: at the saved path, or next to the workspace
/// file under its saved name (so a workspace and its .stviz can be moved or
/// shared together).
fn resolve_dataset_path(
    workspace_path: &Path,
    workspace: &WorkspaceFile,
) -> anyhow::Result<PathBuf> {
    let dir = workspace_path.parent().unwrap_or(Path::new("."));
    let mut candidates = Vec::new();
    if workspace.dataset_path.is_relative() {
        candidates.push(dir.join(&workspace.dataset_path));
    } else {
        candidates.push(workspace.dataset_path.clone());
    }
    let file_name = workspace.dataset_file_name.clone().or_else(|| {
        workspace
            .dataset_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
    });
    if let Some(name) = file_name {
        // Keep only the last component, whichever OS wrote the path.
        if let Some(name) = name.rsplit(['/', '\\']).next().filter(|n| !n.is_empty()) {
            candidates.push(dir.join(name));
        }
    }
    candidates
        .iter()
        .find(|p| p.is_file())
        .cloned()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "dataset not found at {}. Put the .stviz file next to the workspace file and try again.",
                candidates
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(" or ")
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workspace_json(dataset_path: &str, extra: &str) -> String {
        format!(
            r#"{{
            "version": 1, "dataset_path": {dataset_path:?}, "dataset_points": 4,
            "camera_center": [0.0, 0.0], "camera_pixels_per_unit": 1.0,
            "keyframes": [], "key_times": [], "color_path_enabled": true,
            "color_mode": "Categorical", "active_obs": null, "gene_selected": null,
            "categorical_palette": "Tableau10", "category_overrides": [],
            "global_filters": [], "playback_mode": "Loop", "ease_mode": "Linear",
            "speed": 0.3, "t": 0.0, "point_radius_px": 1.0, "max_draw_points": 0,
            "background_rgba": [0, 0, 0, 255], "ui_scale": 1.0, "ui_theme": "Dark",
            "show_axes": true, "show_stats": true,
            "sample_grid": {{"enabled": false, "obs": null, "space": null,
                "use_filter": true, "padding": 0.1, "labels_enabled": false,
                "label_mode": "Default", "custom_labels": []}}{extra}
        }}"#
        )
    }

    #[test]
    fn older_workspace_files_still_parse() {
        let ws: WorkspaceFile = serde_json::from_str(&workspace_json("/data/a.stviz", "")).unwrap();
        assert!(ws.advanced.is_none() && ws.export.is_none() && ws.dataset_file_name.is_none());
    }

    #[test]
    fn dataset_is_found_next_to_the_workspace_when_the_saved_path_is_gone() {
        let dir = std::env::temp_dir().join(format!("stviz_ws_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ws_path = dir.join("a.workspace.json");
        let local = dir.join("a.stviz");

        let gone = r#"C:\Users\someone\data\a.stviz"#;
        let ws: WorkspaceFile =
            serde_json::from_str(&workspace_json(gone, r#", "dataset_file_name": "a.stviz""#))
                .unwrap();
        let err = resolve_dataset_path(&ws_path, &ws).unwrap_err().to_string();
        assert!(err.contains("next to the workspace"), "{err}");

        std::fs::write(&local, b"x").unwrap();
        assert_eq!(resolve_dataset_path(&ws_path, &ws).unwrap(), local);

        // Without the saved name, the last component of the old path is used.
        let ws: WorkspaceFile =
            serde_json::from_str(&workspace_json("/somewhere/else/a.stviz", "")).unwrap();
        assert_eq!(resolve_dataset_path(&ws_path, &ws).unwrap(), local);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
