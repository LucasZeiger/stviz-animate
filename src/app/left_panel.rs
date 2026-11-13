//! The left-hand settings panel.

use super::{
    convert::ExprSource,
    export::SCREENSHOT_RESOLUTION,
    export_job::ExportJobKind,
    util::{
        categorical_obs, chrono_like_timestamp, color32_from_packed, continuous_obs,
        format_scale_value, obs_name,
    },
    CategoricalPalette, CategoryPaletteCache, ColorMode, ExportEncodingSpeed, ExportQuality,
    ExportVideoQuality, SampleGridLabelMode, StvizApp, UiTheme, MAX_FILTER_CATEGORIES,
    MAX_GRID_CATEGORIES,
};
use crate::{color::pack_rgba8, data::Dataset};
use eframe::egui;
use rand::seq::SliceRandom;
use std::path::PathBuf;

impl StvizApp {
    pub(super) fn ui_left_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("stviz-animate");

        self.ui_data_sources(ui, ctx);
        self.ui_workspace_section(ui);

        self.ui_appearance_controls(ui, ctx);

        ui.separator();
        ui.label("Loaded dataset");

        let Some(ds) = self.dataset.clone() else {
            ui.label("No dataset loaded.");
            return;
        };

        self.ui_dataset_info(ui, &ds);
        self.ui_sample_grid_section(ui, &ds);
        self.ui_color_section(ui, &ds);
        self.ui_filters_and_legend(ui, &ds);
        self.ui_rendering_section(ui, &ds);
        self.ui_labels_section(ui, &ds);
        self.ui_output_section(ui);
    }

    /// Mock dataset generation, the .h5ad/.stviz drop zone and the conversion log.
    fn ui_data_sources(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.separator();
        ui.label("Mock dataset");
        ui.horizontal(|ui| {
            ui.label("Cells");
            ui.add(
                egui::DragValue::new(&mut self.mock_cells)
                    .speed(1_000.0)
                    .range(10_000..=10_000_000),
            );
            ui.label("10k - 10M");
        });
        ui.horizontal(|ui| {
            let generate = ui
                .add_enabled(
                    !self.convert_running(),
                    egui::Button::new("Generate mock dataset"),
                )
                .on_hover_text(
                    "Create a synthetic spatial dataset with clusters, UMAP, tSNE and PCA, \
                 then load it. Needs Python; the first run installs the converter.",
                );
            if generate.clicked() {
                self.start_mock_dataset();
            }
            let regen = ui
                .add_enabled(
                    !self.convert_running() && self.mock_last_h5ad.is_some(),
                    egui::Button::new("Regenerate mock dataset"),
                )
                .on_hover_text("Make a new random mock dataset with the same settings.");
            if regen.clicked() {
                self.start_mock_dataset();
            }
        });

        ui.label("Note: conversion may take a minute or two for datasets >1GB.");
        ui.label("First-time converter setup can take a few minutes; it only happens once.");
        ui.checkbox(
            &mut self.convert_include_expr,
            "Include gene expression data (uncheck for smaller file size)",
        );
        if self.convert_include_expr {
            egui::CollapsingHeader::new("Expression source")
                .id_salt("convert_expr_source")
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut self.convert_expr_source, ExprSource::X, "X")
                            .on_hover_text("Use adata.X (default).");
                        ui.radio_value(&mut self.convert_expr_source, ExprSource::Raw, "raw")
                            .on_hover_text("Use adata.raw.X and its gene names.");
                        ui.radio_value(&mut self.convert_expr_source, ExprSource::Layer, "layer")
                            .on_hover_text("Use one of adata.layers.");
                    });
                    if self.convert_expr_source == ExprSource::Layer {
                        ui.add(
                            egui::TextEdit::singleline(&mut self.convert_layer)
                                .desired_width(200.0)
                                .hint_text("Layer name, e.g. counts"),
                        );
                    }
                    ui.add(
                        egui::TextEdit::singleline(&mut self.convert_gene_symbols)
                            .desired_width(200.0)
                            .hint_text("Gene name column in var (optional)"),
                    )
                    .on_hover_text(
                        "Use this var column for gene names instead of the var index, \
                         e.g. gene_symbols.",
                    );
                });
        }
        ui.checkbox(
            &mut self.convert_generate_only,
            "Generate .stviz file only - don't load",
        );
        if let Some(status) = self.convert_status.as_ref() {
            if self.convert_running() {
                ui.colored_label(egui::Color32::from_rgb(240, 200, 90), status);
            } else {
                ui.label(status);
            }
        }

        ui.separator();
        ui.label("Convert .h5ad or load .stviz");
        let drop_height = 56.0;
        let drop_size = egui::vec2(ui.available_width(), drop_height);
        let (drop_rect, drop_resp) = ui.allocate_exact_size(drop_size, egui::Sense::click());
        let hover_files = ctx.input(|i| !i.raw.hovered_files.is_empty());
        let drop_active = drop_resp.hovered() || hover_files;
        let visuals = ui.visuals();
        let busy_fill = egui::Color32::from_rgb(60, 90, 140);
        let busy_stroke = egui::Stroke::new(1.2_f32, egui::Color32::from_rgb(120, 170, 220));
        let drop_fill = if self.convert_running() {
            busy_fill
        } else if drop_active {
            visuals.widgets.hovered.bg_fill
        } else {
            visuals.widgets.inactive.bg_fill
        };
        let drop_stroke = if self.convert_running() {
            busy_stroke
        } else if drop_active {
            visuals.widgets.hovered.bg_stroke
        } else {
            visuals.widgets.inactive.bg_stroke
        };
        ui.painter().rect_filled(drop_rect, 6.0, drop_fill);
        ui.painter()
            .rect_stroke(drop_rect, 6.0, drop_stroke, egui::StrokeKind::Inside);
        let drop_label = if self.convert_running() {
            "Converting..."
        } else if self.dataset_load.is_some() {
            "Loading..."
        } else {
            "Drop .h5ad or .stviz anywhere (or click to pick)"
        };
        let drop_text_color = if self.convert_running() {
            egui::Color32::from_rgb(230, 240, 255)
        } else {
            visuals.text_color()
        };
        ui.painter().text(
            drop_rect.center(),
            egui::Align2::CENTER_CENTER,
            drop_label,
            egui::FontId::proportional(13.0),
            drop_text_color,
        );
        let drop_resp = drop_resp.on_hover_text("Drop a .h5ad to convert, or a .stviz to load.");
        if drop_resp.clicked() {
            self.pick_data_file();
        }
        if self.convert_running() && ui.button("Cancel conversion").clicked() {
            self.cancel_convert();
        }
        ui.separator();
        egui::CollapsingHeader::new("Conversion log")
            .default_open(true)
            .show(ui, |ui| {
                if ui.button("Copy log").clicked() {
                    ui.ctx().copy_text(self.convert_log_text.clone());
                }
                let row_height = ui.text_style_height(&egui::TextStyle::Body);
                let max_height = row_height * 6.0 + 8.0;
                egui::ScrollArea::vertical()
                    .max_height(max_height)
                    .stick_to_bottom(self.convert_running())
                    .show(ui, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(&mut self.convert_log_text)
                                .desired_rows(6)
                                .interactive(true)
                                .cursor_at_end(true)
                                .desired_width(f32::INFINITY),
                        );
                    });
            });
    }

    /// Open/save buttons for datasets and workspaces, plus the last error.
    fn ui_workspace_section(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        if ui
            .button("Open .stviz")
            .on_hover_text("Open a .stviz dataset.")
            .clicked()
        {
            if let Err(e) = self.open_dataset_dialog() {
                let msg = format!("Open failed: {e:#}");
                eprintln!("{msg}");
                self.last_error = Some(msg);
            }
        }
        if !self.recent_files.is_empty() {
            ui.menu_button("Recent datasets", |ui| {
                for path in self.recent_files.clone() {
                    let name = path
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| path.display().to_string());
                    if ui
                        .button(name)
                        .on_hover_text(path.display().to_string())
                        .clicked()
                    {
                        if path.exists() {
                            self.open_data_file(&path);
                        } else {
                            self.recent_files.retain(|p| p != &path);
                            self.last_error =
                                Some(format!("File no longer exists: {}", path.display()));
                        }
                        ui.close();
                    }
                }
            });
        }
        ui.horizontal(|ui| {
            if ui
                .add_enabled(self.dataset.is_some(), egui::Button::new("Save workspace"))
                .clicked()
            {
                if let Err(e) = self.save_workspace_dialog() {
                    self.workspace_status = Some(format!("Workspace save failed: {e:#}"));
                }
            }
            if ui
                .button("Open workspace")
                .on_hover_text(
                    "Restore a saved timeline, colours, filters and view, with its dataset.",
                )
                .clicked()
            {
                if let Err(e) = self.open_workspace_dialog() {
                    self.workspace_status = Some(format!("Workspace open failed: {e:#}"));
                }
            }
        });
        if let Some(status) = &self.workspace_status {
            ui.small(status);
        }

        ui.horizontal(|ui| {
            let edit = egui::TextEdit::singleline(&mut self.open_path)
                .desired_width(200.0)
                .hint_text("Path to .stviz");
            ui.add(edit);
            if ui
                .button("Load path")
                .on_hover_text("Load the .stviz file from this path.")
                .clicked()
            {
                if self.open_path.trim().is_empty() {
                    self.last_error = Some("Path is empty.".to_string());
                } else {
                    let path = PathBuf::from(self.open_path.trim());
                    self.open_data_file(&path);
                }
            }
        });

        if let Some(msg) = self.last_error.as_ref() {
            ui.colored_label(egui::Color32::RED, msg);
        }
    }

    /// UI scale, theme and fullscreen controls.
    fn ui_appearance_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.separator();
        ui.label("View controls");

        ui.horizontal(|ui| {
            ui.label("UI scale");
            let presets = [75, 90, 100, 110, 125, 150, 175, 200];
            let mut selected = (self.ui_scale * 100.0).round() as i32;
            egui::ComboBox::from_id_salt("ui_scale_presets")
                .selected_text(format!("{selected}%"))
                .show_ui(ui, |ui| {
                    for pct in presets {
                        if ui
                            .selectable_value(&mut selected, pct, format!("{pct}%"))
                            .clicked()
                        {
                            self.ui_scale = pct as f32 / 100.0;
                        }
                    }
                });
            let mut scale_pct = self.ui_scale * 100.0;
            if ui
                .add(
                    egui::DragValue::new(&mut scale_pct)
                        .range(50.0..=250.0)
                        .speed(1.0)
                        .suffix("%"),
                )
                .changed()
            {
                self.ui_scale = (scale_pct / 100.0).clamp(0.5, 2.5);
            }
        });
        let theme_label = match self.ui_theme {
            UiTheme::Dark => "Dark",
            UiTheme::Light => "Light",
            UiTheme::Slate => "Slate",
            UiTheme::Matrix => "Matrix",
        };
        egui::ComboBox::from_label("Theme")
            .selected_text(theme_label)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.ui_theme, UiTheme::Dark, "Dark");
                ui.selectable_value(&mut self.ui_theme, UiTheme::Light, "Light");
                ui.selectable_value(&mut self.ui_theme, UiTheme::Slate, "Slate");
                ui.selectable_value(&mut self.ui_theme, UiTheme::Matrix, "Matrix");
            });

        ui.horizontal(|ui| {
            ui.label("Background");
            ui.color_edit_button_srgba(&mut self.background_color);
        });
        ui.horizontal(|ui| {
            if Self::is_wsl() {
                self.fullscreen = false;
                ui.add_enabled(
                    false,
                    egui::Checkbox::new(&mut self.fullscreen, "Fullscreen"),
                )
                .on_hover_text("Fullscreen is not supported on WSL.");
            } else {
                let changed = ui.checkbox(&mut self.fullscreen, "Fullscreen").changed();
                if changed {
                    self.apply_fullscreen(ctx);
                }
            }
        });
        ui.horizontal(|ui| {
            if Self::is_wsl() {
                self.viewport_fullscreen = false;
                ui.add_enabled(
                    false,
                    egui::Checkbox::new(&mut self.viewport_fullscreen, "Viewport fullscreen"),
                )
                .on_hover_text("Fullscreen is not supported on WSL.");
            } else {
                let changed = ui
                    .checkbox(&mut self.viewport_fullscreen, "Viewport fullscreen")
                    .changed();
                if changed {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(
                        self.viewport_fullscreen,
                    ));
                }
            }
        });
    }

    /// Point count and path of the loaded dataset, plus the axes toggle.
    fn ui_dataset_info(&mut self, ui: &mut egui::Ui, ds: &Dataset) {
        ui.label(format!("Points: {}", ds.meta.n_points));
        if let Some(p) = self.dataset_path.as_ref() {
            ui.label(p.display().to_string());
        }

        ui.separator();
        ui.label("View");
        ui.checkbox(&mut self.show_axes, "Show axes")
            .on_hover_text("Draw small X/Y axes in the bottom-left corner of the view.");
    }

    /// Sample grid (spatial) settings.
    fn ui_sample_grid_section(&mut self, ui: &mut egui::Ui, ds: &Dataset) {
        ui.separator();
        ui.label("Sample grid (spatial)");
        let mut grid_changed = false;
        if ui
            .checkbox(&mut self.sample_grid_enabled, "Enable sample grid")
            .changed()
        {
            grid_changed = true;
        }

        if self.sample_grid_enabled {
            let options = categorical_obs(ds);

            if options.is_empty() {
                ui.label("No categorical .obs for sample grouping.");
            } else {
                let current = self
                    .sample_grid_obs_idx
                    .and_then(|idx| options.iter().find(|(i, _)| *i == idx))
                    .map(|(_, n)| n.clone())
                    .unwrap_or_else(|| options[0].1.clone());

                let mut new_idx = self.sample_grid_obs_idx.unwrap_or(options[0].0);
                egui::ComboBox::from_label("Group by")
                    .selected_text(current)
                    .show_ui(ui, |ui| {
                        for (i, n) in &options {
                            if ui.selectable_value(&mut new_idx, *i, n).changed() {
                                grid_changed = true;
                            }
                        }
                    });
                self.sample_grid_obs_idx = Some(new_idx);

                if let Some(idx) = self.sample_grid_obs_idx {
                    if let Ok((_name, _labels, categories, _pal)) = ds.obs_categorical(idx) {
                        if categories.len() > MAX_GRID_CATEGORIES {
                            ui.colored_label(
                                egui::Color32::YELLOW,
                                format!(
                                    "Too many categories for grid ({}). Choose a smaller field (e.g. sample).",
                                    categories.len()
                                ),
                            );
                        } else {
                            ui.label(format!("Samples: {}", categories.len()));
                        }
                    }
                }
            }

            let mut space_idx = self.sample_grid_space_idx.unwrap_or(0);
            egui::ComboBox::from_label("Grid space (most likely spatial)")
                .selected_text(
                    ds.meta
                        .spaces
                        .get(space_idx)
                        .map(|s| s.name.as_str())
                        .unwrap_or("?"),
                )
                .show_ui(ui, |ui| {
                    for (i, s) in ds.meta.spaces.iter().enumerate() {
                        if ui.selectable_value(&mut space_idx, i, &s.name).changed() {
                            grid_changed = true;
                        }
                    }
                });
            self.sample_grid_space_idx = Some(space_idx);

            if ui
                .checkbox(&mut self.sample_grid_use_filter, "Use filter selection")
                .changed()
            {
                grid_changed = true;
            }
            if ui
                .add(egui::Slider::new(&mut self.sample_grid_padding, 0.0..=0.5).text("padding"))
                .changed()
            {
                grid_changed = true;
            }

            ui.checkbox(&mut self.sample_grid_labels_enabled, "Show sample labels");
            if self.sample_grid_labels_enabled {
                let label_mode = match self.sample_grid_label_mode {
                    SampleGridLabelMode::Default => "Sample (default)",
                    SampleGridLabelMode::Custom => "Custom labels",
                };
                egui::ComboBox::from_label("Label source")
                    .selected_text(label_mode)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.sample_grid_label_mode,
                            SampleGridLabelMode::Default,
                            "Sample (default)",
                        );
                        ui.selectable_value(
                            &mut self.sample_grid_label_mode,
                            SampleGridLabelMode::Custom,
                            "Custom labels",
                        );
                    });

                if self.sample_grid_label_mode == SampleGridLabelMode::Custom {
                    if let Some(obs_idx) = self.sample_grid_obs_idx {
                        if let Ok((_name, _labels, categories, _pal)) = ds.obs_categorical(obs_idx)
                        {
                            if categories.len() <= MAX_GRID_CATEGORIES {
                                self.ensure_sample_grid_custom_labels(obs_idx, categories);
                                if ui.button("Clear custom labels").clicked() {
                                    self.sample_grid_custom_labels = categories.to_vec();
                                }
                                egui::ScrollArea::vertical()
                                    .max_height(140.0)
                                    .show(ui, |ui| {
                                        let row_height = ui.spacing().interact_size.y;
                                        for (i, cat) in categories.iter().enumerate() {
                                            ui.horizontal(|ui| {
                                                let label_width = (ui.available_width() * 0.38)
                                                    .clamp(90.0, 220.0);
                                                ui.add_sized(
                                                    [label_width, row_height],
                                                    egui::Label::new(cat).truncate(),
                                                );
                                                if let Some(val) =
                                                    self.sample_grid_custom_labels.get_mut(i)
                                                {
                                                    let edit_width =
                                                        (ui.available_width() - 8.0).max(120.0);
                                                    ui.add_sized(
                                                        [edit_width, row_height],
                                                        egui::TextEdit::singleline(val),
                                                    );
                                                }
                                            });
                                        }
                                    });
                            }
                        }
                    }
                }
            }
        }

        if grid_changed {
            self.mark_grid_dirty();
        }
    }

    /// Color mode selection and the per-mode color controls.
    fn ui_color_section(&mut self, ui: &mut egui::Ui, ds: &Dataset) {
        ui.separator();
        ui.label("Color");

        ui.horizontal(|ui| {
            if ui
                .selectable_value(&mut self.color_mode, ColorMode::Categorical, "Categorical")
                .changed()
            {
                self.legend_range = None;
            }
            if ui
                .selectable_value(&mut self.color_mode, ColorMode::Continuous, "Continuous")
                .changed()
            {
                self.legend_range = None;
            }
            if ui
                .selectable_value(&mut self.color_mode, ColorMode::Gene, "Gene")
                .changed()
            {
                self.legend_range = None;
            }
        });
        if self.color_mode != ColorMode::Categorical
            && ui
                .checkbox(&mut self.clip_outliers, "Clip outliers (1-99%)")
                .on_hover_text(
                    "Scale colors to the 1st-99th percentile so a few extreme values \
                     don't flatten the gradient.",
                )
                .changed()
        {
            self.color_cache.clear();
            self.filtered_color_cache.clear();
            match (self.color_mode, self.gene_selected.clone()) {
                (ColorMode::Gene, Some(gene)) => {
                    self.apply_gene_colors(ds, &gene);
                }
                _ => {
                    let _ = self.recompute_colors_and_filters();
                }
            }
        }

        match self.color_mode {
            ColorMode::Categorical => {
                // choose categorical obs
                let options = categorical_obs(ds);

                if options.is_empty() {
                    ui.label("No categorical .obs exported.");
                } else {
                    let current = options
                        .iter()
                        .find(|(i, _)| *i == self.active_obs_idx)
                        .map(|(_, n)| n.clone())
                        .unwrap_or_else(|| options[0].1.clone());

                    let mut changed = false;
                    egui::ComboBox::from_label("Obs (cat)")
                        .width(200.0)
                        .selected_text(current)
                        .show_ui(ui, |ui| {
                            for (i, n) in options {
                                if ui
                                    .selectable_value(&mut self.active_obs_idx, i, n)
                                    .changed()
                                {
                                    changed = true;
                                }
                            }
                        });
                    if changed {
                        self.load_filter_state(ds);
                        self.filter_popup_open = false;
                    }

                    if ui
                        .button("Apply categorical")
                        .on_hover_text("Apply categorical colors without enabling filters.")
                        .clicked()
                    {
                        if let Err(e) = self.recompute_colors_and_filters() {
                            eprintln!("color/filter: {e:#}");
                        }
                    }

                    let mut palette_changed = false;
                    let mut colors_changed = false;
                    ui.horizontal(|ui| {
                        ui.label("Palette");
                        let selected = Self::categorical_palette_label(self.categorical_palette);
                        egui::ComboBox::from_id_salt("cat_palette")
                            .width(200.0)
                            .selected_text(selected)
                            .show_ui(ui, |ui| {
                                let options = [
                                    CategoricalPalette::Tableau10,
                                    CategoricalPalette::Tab10,
                                    CategoricalPalette::Tab20,
                                    CategoricalPalette::Category10,
                                    CategoricalPalette::Set1,
                                    CategoricalPalette::Set2,
                                    CategoricalPalette::Set3,
                                    CategoricalPalette::Dark2,
                                    CategoricalPalette::Accent,
                                    CategoricalPalette::Paired,
                                    CategoricalPalette::Pastel1,
                                    CategoricalPalette::Pastel2,
                                    CategoricalPalette::Turbo,
                                    CategoricalPalette::Dataset,
                                ];
                                for pal in options {
                                    if ui
                                        .selectable_value(
                                            &mut self.categorical_palette,
                                            pal,
                                            Self::categorical_palette_label(pal),
                                        )
                                        .changed()
                                    {
                                        palette_changed = true;
                                    }
                                }
                            });
                        if ui.button("Reset category colors").clicked() {
                            self.category_overrides.remove(&self.active_obs_idx);
                            palette_changed = true;
                        }
                    });

                    ui.separator();
                    ui.label("Filter categories");

                    if let Ok((_name, _labels, categories, pal_opt)) =
                        ds.obs_categorical(self.active_obs_idx)
                    {
                        let too_many = categories.len() > MAX_FILTER_CATEGORIES;
                        if !too_many && self.enabled_categories.len() != categories.len() {
                            self.enabled_categories = vec![true; categories.len()];
                        }
                        let palette_slice: Option<&[u32]> = if too_many {
                            None
                        } else {
                            let needs_rebuild = match self.category_palette_cache.as_ref() {
                                Some(cache) => {
                                    cache.obs_idx != self.active_obs_idx
                                        || cache.categories_len != categories.len()
                                        || cache.palette != self.categorical_palette
                                        || cache.has_dataset_palette != pal_opt.is_some()
                                }
                                None => true,
                            };
                            if needs_rebuild {
                                let colors = self.categorical_palette_for(
                                    self.active_obs_idx,
                                    categories.len(),
                                    pal_opt,
                                );
                                self.category_palette_cache = Some(CategoryPaletteCache {
                                    obs_idx: self.active_obs_idx,
                                    categories_len: categories.len(),
                                    palette: self.categorical_palette,
                                    has_dataset_palette: pal_opt.is_some(),
                                    colors,
                                });
                            }
                            self.category_palette_cache
                                .as_ref()
                                .map(|cache| cache.colors.as_slice())
                        };
                        if self.categorical_palette == CategoricalPalette::Dataset
                            && pal_opt.is_none()
                        {
                            ui.label("No dataset palette found; using Tableau10.");
                        }
                        let mut filter_changed = false;
                        if too_many {
                            ui.colored_label(
                                egui::Color32::YELLOW,
                                format!(
                                    "Too many categories ({}). Filtering disabled for this field.",
                                    categories.len()
                                ),
                            );
                        } else {
                            let obs_idx = self.active_obs_idx;
                            let categories_len = categories.len();
                            let (enabled_categories, category_overrides) =
                                (&mut self.enabled_categories, &mut self.category_overrides);
                            let overrides = category_overrides
                                .entry(obs_idx)
                                .or_insert_with(|| vec![None; categories_len]);
                            if overrides.len() != categories_len {
                                overrides.resize(categories_len, None);
                            }
                            let row_height = ui.spacing().interact_size.y.max(20.0);
                            if categories.len() <= 10 {
                                egui::Frame::group(ui.style()).show(ui, |ui| {
                                    render_filter_rows(
                                        ui,
                                        0..categories.len(),
                                        categories,
                                        palette_slice,
                                        enabled_categories,
                                        overrides,
                                        &mut filter_changed,
                                        &mut colors_changed,
                                    );
                                });
                            } else {
                                let preview_max = row_height * 10.0 + 8.0;
                                egui::ScrollArea::vertical()
                                    .max_height(preview_max)
                                    .min_scrolled_height(preview_max)
                                    .show_rows(ui, row_height, categories.len(), |ui, range| {
                                        render_filter_rows(
                                            ui,
                                            range,
                                            categories,
                                            palette_slice,
                                            enabled_categories,
                                            overrides,
                                            &mut filter_changed,
                                            &mut colors_changed,
                                        );
                                    });
                                ui.horizontal(|ui| {
                                    ui.label(format!("{} categories", categories.len()));
                                    if ui.button("Show full list").clicked() {
                                        self.filter_popup_open = true;
                                    }
                                });
                            }
                            if self.filter_popup_open {
                                let mut open = self.filter_popup_open;
                                egui::Window::new("Filter categories")
                                    .open(&mut open)
                                    .resizable(true)
                                    .min_width(260.0)
                                    .show(ui.ctx(), |ui| {
                                        ui.horizontal(|ui| {
                                            if ui.button("All").clicked() {
                                                for v in &mut *enabled_categories {
                                                    *v = true;
                                                }
                                                filter_changed = true;
                                            }
                                            if ui.button("None").clicked() {
                                                for v in &mut *enabled_categories {
                                                    *v = false;
                                                }
                                                filter_changed = true;
                                            }
                                        });
                                        let list_max = (row_height * categories.len() as f32 + 8.0)
                                            .min(ui.ctx().available_rect().height() * 0.7);
                                        egui::ScrollArea::vertical()
                                            .max_height(list_max)
                                            .show_rows(
                                                ui,
                                                row_height,
                                                categories.len(),
                                                |ui, range| {
                                                    render_filter_rows(
                                                        ui,
                                                        range,
                                                        categories,
                                                        palette_slice,
                                                        enabled_categories,
                                                        overrides,
                                                        &mut filter_changed,
                                                        &mut colors_changed,
                                                    );
                                                },
                                            );
                                    });
                                self.filter_popup_open = open;
                            }
                        }

                        ui.horizontal(|ui| {
                            if !too_many {
                                if ui
                                    .button("All")
                                    .on_hover_text("Enable all categories.")
                                    .clicked()
                                {
                                    for v in &mut self.enabled_categories {
                                        *v = true;
                                    }
                                    filter_changed = true;
                                }
                                if ui
                                    .button("None")
                                    .on_hover_text("Disable all categories.")
                                    .clicked()
                                {
                                    for v in &mut self.enabled_categories {
                                        *v = false;
                                    }
                                    filter_changed = true;
                                }
                                let all_on = self.enabled_categories.iter().all(|v| *v);
                                let desired_active = !all_on;
                                let active_now = self.active_filters.contains(&self.active_obs_idx);
                                let state_matches = self
                                    .category_state
                                    .get(&self.active_obs_idx)
                                    .map(|s| s == &self.enabled_categories)
                                    .unwrap_or(false);
                                let needs_apply = filter_changed
                                    || desired_active != active_now
                                    || (desired_active && !state_matches);
                                if needs_apply {
                                    if desired_active {
                                        self.category_state.insert(
                                            self.active_obs_idx,
                                            self.enabled_categories.clone(),
                                        );
                                        self.active_filters.insert(self.active_obs_idx);
                                    } else {
                                        self.active_filters.remove(&self.active_obs_idx);
                                        self.category_state.remove(&self.active_obs_idx);
                                    }
                                    let _ = self.recompute_draw_indices_with_filters();
                                }
                            }
                        });
                    }
                    if palette_changed || colors_changed {
                        self.category_palette_cache = None;
                        self.color_cache.clear();
                        let _ = self.recompute_colors_and_filters();
                    }
                }
            }
            ColorMode::Continuous => {
                let options = continuous_obs(ds);
                if options.is_empty() {
                    ui.label("No continuous .obs exported.");
                } else {
                    let current = options
                        .iter()
                        .find(|(i, _)| *i == self.active_obs_idx)
                        .map(|(_, n)| n.clone())
                        .unwrap_or_else(|| options[0].1.clone());

                    egui::ComboBox::from_label("Obs (cont)")
                        .width(200.0)
                        .selected_text(current)
                        .show_ui(ui, |ui| {
                            for (i, n) in options {
                                ui.selectable_value(&mut self.active_obs_idx, i, n);
                            }
                        });

                    if ui
                        .button("Apply continuous")
                        .on_hover_text("Apply continuous colors.")
                        .clicked()
                    {
                        if let Err(e) = self.recompute_colors_and_filters() {
                            eprintln!("color: {e:#}");
                        }
                    }
                }
            }
            ColorMode::Gene => {
                ui.label("Gene coloring (requires --include-expr in export).");
                if let Some(expr) = ds.meta.expr.as_ref() {
                    let mut apply_gene: Option<String> = None;
                    ui.horizontal(|ui| {
                        ui.label("Gene");
                        if ui.text_edit_singleline(&mut self.gene_query).changed() {
                            self.gene_selected = None;
                        }
                    });
                    if let Some(selected) = self.gene_selected.as_ref() {
                        ui.label(format!("Selected: {selected}"));
                    }
                    let needle = self.gene_query.trim().to_ascii_lowercase();
                    let mut total = 0usize;
                    let gene_row_height = ui.spacing().interact_size.y.max(20.0);
                    let gene_max = gene_row_height * 10.0 + 8.0;
                    egui::ScrollArea::vertical()
                        .max_height(gene_max)
                        .show(ui, |ui| {
                            for name in &expr.var_names {
                                let is_match = if needle.is_empty() {
                                    true
                                } else {
                                    name.to_ascii_lowercase().contains(&needle)
                                };
                                if !is_match {
                                    continue;
                                }
                                total += 1;
                                let is_selected = self
                                    .gene_selected
                                    .as_ref()
                                    .map(|g| g == name)
                                    .unwrap_or(false);
                                if ui.selectable_label(is_selected, name).clicked() {
                                    self.gene_selected = Some(name.clone());
                                    apply_gene = Some(name.clone());
                                }
                            }
                        });
                    if total == 0 {
                        ui.label("No matches.");
                    }

                    if let Some(gene) = apply_gene {
                        if self.apply_gene_colors(ds, &gene) {
                            let _ = self.recompute_draw_indices_no_filter();
                        }
                    }
                } else {
                    ui.label("No gene expression data in this file.");
                }
            }
        }
    }

    /// Active filters summary and the gradient legend.
    fn ui_filters_and_legend(&mut self, ui: &mut egui::Ui, ds: &Dataset) {
        if !self.active_filters.is_empty() {
            ui.separator();
            ui.label("Active filters");
            let mut active = self.active_filters.iter().copied().collect::<Vec<_>>();
            active.sort_unstable();
            egui::ScrollArea::vertical()
                .max_height(90.0)
                .show(ui, |ui| {
                    for obs_idx in active {
                        ui.label(obs_name(ds, obs_idx));
                    }
                });
            if ui
                .button("Clear all filters")
                .on_hover_text("Disable all active filters and reset category masks.")
                .clicked()
            {
                self.clear_all_filters(ds);
            }
        }

        let legend = self
            .active_legend_range
            .as_ref()
            .or(self.legend_range.as_ref());
        if let Some(legend) = legend {
            ui.separator();
            ui.label("Legend");
            draw_gradient_legend(ui, &legend.label, legend.min, legend.max);
        }
    }

    /// Point size, draw cap and draw-order controls.
    fn ui_rendering_section(&mut self, ui: &mut egui::Ui, ds: &Dataset) {
        ui.separator();
        ui.label("Rendering");
        ui.horizontal(|ui| {
            ui.label("point radius");
            let mut slider_value = self.point_radius_px.clamp(0.5, 2.0);
            let slider = egui::Slider::new(&mut slider_value, 0.5..=2.0).show_value(false);
            if ui.add(slider).changed() {
                self.point_radius_px = slider_value;
            }
            let mut manual_value = self.point_radius_px;
            if ui
                .add(
                    egui::DragValue::new(&mut manual_value)
                        .speed(0.05)
                        .range(0.1..=5.0)
                        .suffix(" px"),
                )
                .changed()
            {
                self.point_radius_px = manual_value;
            }
        });
        if ds.meta.n_points > 0 {
            let max_cap = ds.meta.n_points as usize;
            let mut slider = egui::Slider::new(&mut self.max_draw_points, 0..=max_cap)
                .text("max draw (0 = all)");
            slider = slider.logarithmic(true).smallest_positive(1.0);
            if ui.add(slider).changed() {
                self.apply_downsample();
            }
            let draw_count = if self.max_draw_points == 0 {
                self.base_indices.len()
            } else {
                self.draw_indices.len()
            };
            ui.label(format!("Drawn: {} / {}", draw_count, ds.meta.n_points));
        }

        if ui
            .button("Shuffle draw order")
            .on_hover_text("Draw points in random order so no cluster is always painted on top.")
            .clicked()
        {
            if let Some(ds) = self.dataset.as_ref() {
                if self.base_indices.is_empty() {
                    self.base_indices = (0..ds.meta.n_points).collect();
                }
                let mut rng = rand::thread_rng();
                self.base_indices.shuffle(&mut rng);
                self.apply_downsample();
            }
        }

        ui.checkbox(&mut self.fast_render, "Fast render (square points)")
            .on_hover_text("Faster on slow GPUs. Full HD and 4K exports always use round points.");
        ui.checkbox(&mut self.show_stats, "Show stats")
            .on_hover_text("Show GPU, frame rate and the number of drawn points in the view.");
    }

    /// Screenshot and video export controls and the export log.
    /// Title, legend and scale bar drawn over the view and into exports.
    fn ui_labels_section(&mut self, ui: &mut egui::Ui, ds: &Dataset) {
        ui.separator();
        let o = &mut self.overlay;
        ui.checkbox(&mut o.enabled, "Show labels on view and exports")
            .on_hover_text(
                "Draw a title, legend and scale bar over the points. They appear in \
                 the viewport and are burned into screenshots and videos.",
            );
        if !o.enabled {
            return;
        }
        ui.indent("labels_section", |ui| {
            ui.horizontal(|ui| {
                ui.label("Title");
                ui.add(
                    egui::TextEdit::singleline(&mut o.title)
                        .hint_text("Optional, e.g. sample name")
                        .desired_width(170.0),
                );
            });
            ui.checkbox(&mut o.show_legend, "Legend")
                .on_hover_text("Categories or colour bar for the colouring on screen.");
            ui.checkbox(&mut o.show_space_name, "Space name")
                .on_hover_text(
                    "Show the coordinate space, e.g. \"spatial to X_umap\" during a transition.",
                );
            let has_spatial = ds
                .meta
                .spaces
                .iter()
                .any(|s| super::overlay::is_spatial_space(&s.name));
            ui.add_enabled_ui(has_spatial, |ui| {
                ui.checkbox(&mut o.show_scale_bar, "Scale bar")
                    .on_hover_text(
                        "Shown on spaces with \"spatial\" in their name, while the view rests \
                     on that space and the sample grid is off.",
                    );
            })
            .response
            .on_disabled_hover_text("No space with \"spatial\" in its name.");
            if has_spatial && o.show_scale_bar {
                ui.horizontal(|ui| {
                    ui.label("1 coordinate =");
                    ui.add(
                        egui::DragValue::new(&mut o.units_per_coord)
                            .range(1e-6..=1e6)
                            .speed(0.01),
                    )
                    .on_hover_text(
                        "Physical size of one coordinate unit. Visium spot coordinates \
                         are usually in full-resolution image pixels; Xenium and \
                         MERFISH coordinates are usually already in µm.",
                    );
                    ui.add(egui::TextEdit::singleline(&mut o.unit_label).desired_width(40.0));
                });
            }
            ui.add(
                egui::Slider::new(&mut o.text_scale, 0.5..=2.0)
                    .text("Label size")
                    .fixed_decimals(2),
            );
        });
    }

    fn ui_output_section(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        let busy = self.export_busy();
        let mut output_frame = egui::Frame::group(ui.style()).inner_margin(egui::Margin::same(6));
        if busy {
            output_frame = output_frame
                .fill(egui::Color32::from_rgb(40, 60, 90))
                .stroke(egui::Stroke::new(
                    1.2_f32,
                    egui::Color32::from_rgb(120, 170, 220),
                ));
        }
        output_frame.show(ui, |ui| {
            if busy {
                ui.colored_label(egui::Color32::from_rgb(230, 240, 255), "Output (busy)");
            } else {
                ui.label("Output");
            }
            ui.small(format!("Data directory: {}", self.app_data_dir.display()));
            ui.horizontal(|ui| {
                ui.label("Screenshot name");
                ui.add(
                    egui::TextEdit::singleline(&mut self.screenshot_name)
                        .hint_text("stviz-animate_screenshot"),
                );
            });
            if ui
                .button("Screenshot")
                .on_hover_text("Save a 4K PNG of the current view to the screenshots folder.")
                .clicked()
            {
                let ts = chrono_like_timestamp();
                let mut stem = self.screenshot_name.trim().to_string();
                if stem.is_empty() {
                    stem = "stviz-animate_screenshot".to_string();
                }
                if stem.to_ascii_lowercase().ends_with(".png") {
                    stem.truncate(stem.len().saturating_sub(4));
                }
                let path = self.screenshot_dir.join(format!("{stem}_{ts}.png"));
                if let Err(err) = self.render_export_frame(SCREENSHOT_RESOLUTION, &path, false) {
                    self.export_status = Some(format!("Screenshot failed: {err}"));
                }
            }
            ui.add_space(6.0);
            ui.label("Loop export");
            ui.add(
                egui::DragValue::new(&mut self.export_fps)
                    .range(1..=240)
                    .prefix("fps "),
            );
            ui.add(
                egui::DragValue::new(&mut self.export_duration_sec)
                    .range(0.5..=120.0)
                    .speed(0.5)
                    .suffix(" sec")
                    .prefix("duration "),
            );
            ui.horizontal(|ui| {
                ui.label("Quality");
                let label = match self.export_quality {
                    ExportQuality::Current => "Current viewport",
                    ExportQuality::FullHd => "Full HD (1920x1080)",
                    ExportQuality::UltraHd => "4K (3840x2160)",
                };
                egui::ComboBox::from_id_salt("export_quality")
                    .selected_text(label)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.export_quality,
                            ExportQuality::Current,
                            "Current viewport",
                        );
                        ui.selectable_value(
                            &mut self.export_quality,
                            ExportQuality::FullHd,
                            "Full HD (1920x1080)",
                        );
                        ui.selectable_value(
                            &mut self.export_quality,
                            ExportQuality::UltraHd,
                            "4K (3840x2160)",
                        );
                    });
            });
            ui.horizontal(|ui| {
                ui.label("Encoding");
                let label = match self.export_video_quality {
                    ExportVideoQuality::Standard => "Standard (CRF 23, yuv420p)",
                    ExportVideoQuality::High => "High (CRF 18, yuv420p)",
                    ExportVideoQuality::Ultra => "Ultra (CRF 14, yuv420p)",
                };
                egui::ComboBox::from_id_salt("export_video_quality")
                    .selected_text(label)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.export_video_quality,
                            ExportVideoQuality::Standard,
                            "Standard (CRF 23, yuv420p)",
                        );
                        ui.selectable_value(
                            &mut self.export_video_quality,
                            ExportVideoQuality::High,
                            "High (CRF 18, yuv420p)",
                        );
                        ui.selectable_value(
                            &mut self.export_video_quality,
                            ExportVideoQuality::Ultra,
                            "Ultra (CRF 14, yuv420p)",
                        );
                    });
            });
            ui.horizontal(|ui| {
                ui.label("Encoding speed");
                let label = match self.export_encoding_speed {
                    ExportEncodingSpeed::Fast => "Fast (larger file)",
                    ExportEncodingSpeed::Balanced => "Balanced",
                    ExportEncodingSpeed::SmallestFile => "Smallest file (slower)",
                };
                egui::ComboBox::from_id_salt("export_encoding_speed")
                    .selected_text(label)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.export_encoding_speed,
                            ExportEncodingSpeed::Fast,
                            "Fast (larger file)",
                        );
                        ui.selectable_value(
                            &mut self.export_encoding_speed,
                            ExportEncodingSpeed::Balanced,
                            "Balanced",
                        );
                        ui.selectable_value(
                            &mut self.export_encoding_speed,
                            ExportEncodingSpeed::SmallestFile,
                            "Smallest file (slower)",
                        );
                    });
            });
            ui.horizontal(|ui| {
                ui.label("Output");
                ui.text_edit_singleline(&mut self.export_name);
            });
            if ui
                .checkbox(&mut self.export_run_ffmpeg, "Run ffmpeg if available")
                .changed()
            {
                self.refresh_ffmpeg_path();
            }
            let installing = self.export_job_kind() == Some(ExportJobKind::InstallFfmpeg);
            if self.export_run_ffmpeg && !self.ffmpeg_available {
                if installing {
                    ui.colored_label(egui::Color32::YELLOW, "Installing ffmpeg...");
                } else {
                    ui.colored_label(
                        egui::Color32::YELLOW,
                        "ffmpeg not found (will try OpenCV fallback).",
                    );
                }
                ui.checkbox(
                    &mut self.ffmpeg_auto_install,
                    "Install ffmpeg during export if missing",
                )
                .on_hover_text(
                    "Uses the system package manager (winget, choco, brew, apt, dnf, pacman or \
                     zypper). Needs internet access and may need admin rights.",
                );
                if ui
                    .add_enabled(!busy, egui::Button::new("Install ffmpeg now"))
                    .on_hover_text("Runs the system package manager in the background.")
                    .clicked()
                {
                    self.start_ffmpeg_install();
                }
            }
            let keep_forced = !self.export_run_ffmpeg;
            let mut keep_pref = self.export_keep_frames;
            let keep_label = if keep_forced {
                "Keep PNG frames (required without video export)"
            } else {
                "Keep PNG frames"
            };
            if ui
                .add_enabled(
                    !keep_forced,
                    egui::Checkbox::new(&mut keep_pref, keep_label),
                )
                .changed()
            {
                self.export_keep_frames = keep_pref;
            }
            let export_label = match self.export_job_kind() {
                Some(ExportJobKind::Encode) => "Encoding...",
                _ if busy && !installing => "Exporting...",
                _ => "Export loop video",
            };
            if ui
                .add_enabled(!busy, egui::Button::new(export_label))
                .clicked()
            {
                self.start_export_loop();
            }
            if busy {
                let cancel_label = if installing {
                    "Cancel install"
                } else {
                    "Cancel export"
                };
                if ui.button(cancel_label).clicked() {
                    self.cancel_export_loop();
                }
            }
            if let Some(status) = self.export_status.as_ref() {
                if busy {
                    ui.colored_label(egui::Color32::from_rgb(240, 200, 90), status);
                } else {
                    ui.label(status);
                }
            }
            if !self.export_log_text.trim().is_empty() {
                ui.separator();
                let mut log_rect = None;
                let open_override = if self.export_log_open {
                    Some(true)
                } else {
                    None
                };
                egui::CollapsingHeader::new("Export log")
                    .open(open_override)
                    .show(ui, |ui| {
                        if ui.button("Copy log").clicked() {
                            ui.ctx().copy_text(self.export_log_text.clone());
                        }
                        let response = ui.add(
                            egui::TextEdit::multiline(&mut self.export_log_text)
                                .desired_rows(6)
                                .cursor_at_end(true)
                                .interactive(true),
                        );
                        log_rect = Some(response);
                    });
                if self.export_log_open {
                    self.export_log_open = false;
                }
                if self.export_log_focus {
                    if let Some(response) = log_rect {
                        response.scroll_to_me(Some(egui::Align::BOTTOM));
                    }
                    self.export_log_focus = false;
                }
            }
        });
    }
}

fn render_filter_rows(
    ui: &mut egui::Ui,
    range: std::ops::Range<usize>,
    categories: &[String],
    palette_slice: Option<&[u32]>,
    enabled_categories: &mut [bool],
    overrides: &mut [Option<u32>],
    filter_changed: &mut bool,
    colors_changed: &mut bool,
) {
    for i in range {
        let c = &categories[i];
        let mut color = palette_slice
            .and_then(|pal| pal.get(i).copied())
            .unwrap_or(pack_rgba8(200, 200, 200, 255));
        if let Some(override_color) = overrides.get(i).and_then(|c| *c) {
            color = override_color;
        }
        let mut color = color32_from_packed(color);
        ui.horizontal(|ui| {
            if ui.checkbox(&mut enabled_categories[i], "").changed() {
                *filter_changed = true;
            }
            if ui.color_edit_button_srgba(&mut color).changed() {
                if i < overrides.len() {
                    overrides[i] = Some(pack_rgba8(color.r(), color.g(), color.b(), color.a()));
                }
                *colors_changed = true;
            }
            ui.label(c);
        });
    }
}

fn draw_gradient_legend(ui: &mut egui::Ui, label: &str, vmin: f32, vmax: f32) {
    ui.label(label);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(180.0, 14.0), egui::Sense::hover());
    let steps = 32;
    let seg_w = rect.width() / steps as f32;
    for i in 0..steps {
        let t = i as f32 / (steps - 1) as f32;
        let c = colorous::VIRIDIS.eval_continuous(t as f64);
        let color = egui::Color32::from_rgb(c.r, c.g, c.b);
        let x0 = rect.left() + seg_w * i as f32;
        let seg = egui::Rect::from_min_size(
            egui::pos2(x0, rect.top()),
            egui::vec2(seg_w + 1.0, rect.height()),
        );
        ui.painter().rect_filled(seg, 0.0, color);
    }
    ui.label(format!(
        "Range: {} .. {}",
        format_scale_value(vmin),
        format_scale_value(vmax)
    ));
}
