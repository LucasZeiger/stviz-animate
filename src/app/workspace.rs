//! Dataset/workspace file dialogs and workspace save/restore.

use super::{
    CategoricalPalette, ColorKey, ColorMode, EaseMode, ExportEncodingSpeed, KeyframeFilter,
    PlaybackMode, SampleGridLabelMode, StvizApp, UiTheme, MAX_FILTER_CATEGORIES,
};
use crate::data::{Dataset, ObsMeta};
use anyhow::Context as _;
use eframe::egui;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Serialize, Deserialize)]
struct WorkspaceFile {
    version: u32,
    dataset_path: PathBuf,
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
            .set_directory(self.project_dir.clone())
            .pick_file()
        else {
            return Ok(());
        };

        self.load_dataset(&path)
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

        Ok(WorkspaceFile {
            version: 1,
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
        let json = serde_json::to_string_pretty(&workspace)?;
        std::fs::write(&path, json)
            .with_context(|| format!("write workspace: {}", path.display()))?;
        self.workspace_status = Some(format!("Saved workspace: {}", path.display()));
        Ok(())
    }

    pub(super) fn open_workspace_dialog(&mut self) -> anyhow::Result<()> {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("stviz workspace", &["json"])
            .set_title("Open workspace")
            .set_directory(self.project_dir.clone())
            .pick_file()
        else {
            return Ok(());
        };
        let json = std::fs::read_to_string(&path)
            .with_context(|| format!("read workspace: {}", path.display()))?;
        let mut workspace: WorkspaceFile = serde_json::from_str(&json)?;
        if workspace.version != 1 {
            return Err(anyhow::anyhow!(
                "unsupported workspace version {}",
                workspace.version
            ));
        }
        if workspace.dataset_path.is_relative() {
            workspace.dataset_path = path
                .parent()
                .unwrap_or(Path::new("."))
                .join(&workspace.dataset_path);
        }
        self.apply_workspace(workspace)?;
        self.workspace_status = Some(format!("Opened workspace: {}", path.display()));
        Ok(())
    }

    fn apply_workspace(&mut self, workspace: WorkspaceFile) -> anyhow::Result<()> {
        self.load_dataset(&workspace.dataset_path)?;
        let ds = self.dataset.clone().unwrap();
        if ds.meta.n_points != workspace.dataset_points {
            return Err(anyhow::anyhow!(
                "workspace expects {} cells, dataset has {}",
                workspace.dataset_points,
                ds.meta.n_points
            ));
        }
        if workspace.keyframes.len() < 2 {
            return Err(anyhow::anyhow!(
                "workspace must contain at least two keyframes"
            ));
        }

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
                    ds.meta
                        .obs
                        .iter()
                        .position(
                            |obs| matches!(obs, ObsMeta::Categorical { name: n, .. } if n == name),
                        )
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
        self.to_space = *self.space_path.last().unwrap();
        self.color_path_enabled = workspace.color_path_enabled;
        self.color_mode = workspace.color_mode;
        self.categorical_palette = workspace.categorical_palette;
        self.playback_mode = workspace.playback_mode;
        self.ease_mode = workspace.ease_mode;
        self.speed = workspace.speed;
        self.t = workspace.t.clamp(0.0, 1.0);
        self.point_radius_px = workspace.point_radius_px;
        self.max_draw_points = workspace.max_draw_points;
        self.export_encoding_speed = workspace.export_encoding_speed;
        self.background_color = egui::Color32::from_rgba_unmultiplied(
            workspace.background_rgba[0],
            workspace.background_rgba[1],
            workspace.background_rgba[2],
            workspace.background_rgba[3],
        );
        self.ui_scale = workspace.ui_scale.clamp(0.5, 3.0);
        self.ui_theme = workspace.ui_theme;
        self.show_axes = workspace.show_axes;
        self.show_stats = workspace.show_stats;
        self.camera.center = workspace.camera_center;
        self.camera.pixels_per_unit = workspace.camera_pixels_per_unit.clamp(1e-6, 1e9);

        if let Some(name) = workspace.active_obs {
            if let Some(idx) = ds.meta.obs.iter().position(|obs| match obs {
                ObsMeta::Categorical { name: n, .. } | ObsMeta::Continuous { name: n, .. } => {
                    n == &name
                }
            }) {
                self.active_obs_idx = idx;
            }
        }
        self.category_overrides.clear();
        for saved in workspace.category_overrides {
            if let Some(obs_idx) = ds.meta.obs.iter().position(
                |obs| matches!(obs, ObsMeta::Categorical { name, .. } if name == &saved.field),
            ) {
                let (_name, _labels, categories, _pal) = ds.obs_categorical(obs_idx)?;
                self.category_overrides.insert(
                    obs_idx,
                    categories
                        .iter()
                        .map(|name| saved.colors.get(name).copied())
                        .collect(),
                );
            }
        }
        self.active_filters.clear();
        self.category_state.clear();
        for saved in workspace.global_filters {
            let filter = Self::resolve_workspace_filter(&ds, &saved)?;
            self.active_filters.insert(filter.obs_idx);
            self.category_state.insert(filter.obs_idx, filter.enabled);
        }
        self.load_filter_state(&ds);

        self.sample_grid_enabled = workspace.sample_grid.enabled;
        self.sample_grid_obs_idx = workspace.sample_grid.obs.as_ref().and_then(|name| {
            ds.meta
                .obs
                .iter()
                .position(|obs| matches!(obs, ObsMeta::Categorical { name: n, .. } if n == name))
        });
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

        self.advanced_cards.clear();
        self.advanced_connections.clear();
        self.color_cache.clear();
        self.filtered_color_cache.clear();
        self.category_palette_cache = None;
        self.recompute_colors_and_filters()?;
        self.gene_selected = workspace.gene_selected;
        if self.color_mode == ColorMode::Gene {
            if let Some(gene) = self.gene_selected.clone() {
                self.gene_query = gene.clone();
                if let Some((colors, legend, opaque)) =
                    self.compute_colors_for_key(&ds, &ColorKey::Gene(gene))
                {
                    self.colors_rgba8 = Arc::new(colors);
                    self.colors_opaque = opaque;
                    self.colors_id = self.next_color_id();
                    self.legend_range = legend;
                }
            }
        }
        self.mark_grid_dirty();
        Ok(())
    }
}
