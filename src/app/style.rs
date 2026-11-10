//! Categorical palette choices and the egui theme visuals.

use super::{CategoricalPalette, StvizApp, UiTheme};
use crate::color::{categorical_palette, categorical_palette_named};
use eframe::egui;

impl StvizApp {
    pub(super) fn categorical_palette_label(palette: CategoricalPalette) -> &'static str {
        match palette {
            CategoricalPalette::Tableau10 => "Tableau10",
            CategoricalPalette::Tab10 => "Tab10",
            CategoricalPalette::Tab20 => "Tab20",
            CategoricalPalette::Category10 => "Category10",
            CategoricalPalette::Set1 => "Set1",
            CategoricalPalette::Set2 => "Set2",
            CategoricalPalette::Set3 => "Set3",
            CategoricalPalette::Dark2 => "Dark2",
            CategoricalPalette::Accent => "Accent",
            CategoricalPalette::Paired => "Paired",
            CategoricalPalette::Pastel1 => "Pastel1",
            CategoricalPalette::Pastel2 => "Pastel2",
            CategoricalPalette::Turbo => "Turbo",
            CategoricalPalette::Dataset => "Dataset",
        }
    }

    fn categorical_palette_name(palette: CategoricalPalette) -> &'static str {
        match palette {
            CategoricalPalette::Tableau10 => "tableau10",
            CategoricalPalette::Tab10 => "tab10",
            CategoricalPalette::Tab20 => "tab20",
            CategoricalPalette::Category10 => "category10",
            CategoricalPalette::Set1 => "set1",
            CategoricalPalette::Set2 => "set2",
            CategoricalPalette::Set3 => "set3",
            CategoricalPalette::Dark2 => "dark2",
            CategoricalPalette::Accent => "accent",
            CategoricalPalette::Paired => "paired",
            CategoricalPalette::Pastel1 => "pastel1",
            CategoricalPalette::Pastel2 => "pastel2",
            CategoricalPalette::Turbo => "turbo",
            CategoricalPalette::Dataset => "dataset",
        }
    }

    pub(super) fn categorical_palette_for(
        &self,
        obs_idx: usize,
        categories_len: usize,
        pal_opt: Option<&[u32]>,
    ) -> Vec<u32> {
        let base = match self.categorical_palette {
            CategoricalPalette::Dataset => pal_opt
                .map(|p| p.to_vec())
                .unwrap_or_else(|| categorical_palette_named("tableau10", categories_len)),
            CategoricalPalette::Turbo => categorical_palette(categories_len),
            other => {
                categorical_palette_named(Self::categorical_palette_name(other), categories_len)
            }
        };
        let mut out = base;
        if let Some(overrides) = self.category_overrides.get(&obs_idx) {
            for (i, override_color) in overrides.iter().enumerate() {
                if let Some(color) = override_color {
                    if i < out.len() {
                        out[i] = *color;
                    }
                }
            }
        }
        out
    }

    pub(super) fn visuals_for_theme(theme: UiTheme) -> egui::Visuals {
        match theme {
            UiTheme::Dark => egui::Visuals::dark(),
            UiTheme::Light => egui::Visuals::light(),
            UiTheme::Slate => {
                let mut visuals = egui::Visuals::dark();
                visuals.panel_fill = egui::Color32::from_rgb(22, 26, 32);
                visuals.window_fill = egui::Color32::from_rgb(22, 26, 32);
                visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(32, 37, 44);
                visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(44, 52, 62);
                visuals.widgets.active.bg_fill = egui::Color32::from_rgb(60, 70, 84);
                visuals.selection.bg_fill = egui::Color32::from_rgb(70, 110, 160);
                visuals.selection.stroke =
                    egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(150, 190, 235));
                visuals
            }
            UiTheme::Matrix => {
                let mut visuals = egui::Visuals::dark();
                visuals.panel_fill = egui::Color32::from_rgb(5, 9, 7);
                visuals.window_fill = egui::Color32::from_rgb(5, 9, 7);
                visuals.override_text_color = Some(egui::Color32::from_rgb(95, 210, 120));
                visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(8, 16, 11);
                visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(12, 24, 16);
                visuals.widgets.active.bg_fill = egui::Color32::from_rgb(16, 32, 20);
                visuals.widgets.inactive.bg_stroke =
                    egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(60, 150, 90));
                visuals.widgets.hovered.bg_stroke =
                    egui::Stroke::new(1.2_f32, egui::Color32::from_rgb(90, 210, 130));
                visuals.widgets.active.bg_stroke =
                    egui::Stroke::new(1.4_f32, egui::Color32::from_rgb(120, 245, 160));
                visuals.widgets.inactive.fg_stroke =
                    egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(70, 170, 105));
                visuals.widgets.hovered.fg_stroke =
                    egui::Stroke::new(1.2_f32, egui::Color32::from_rgb(110, 230, 150));
                visuals.selection.bg_fill = egui::Color32::from_rgb(0, 90, 50);
                visuals.selection.stroke =
                    egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(110, 230, 150));
                visuals.hyperlink_color = egui::Color32::from_rgb(80, 190, 110);
                visuals
            }
        }
    }
}
