//! First-run help: the welcome screen shown while no dataset is loaded, the
//! top bar, and the keyboard shortcuts window.

use super::StvizApp;
use eframe::egui;

/// Cells in the dataset made by "Try demo data". Small enough to generate in
/// seconds, large enough to look like real data.
const DEMO_CELLS: u32 = 50_000;

/// Recent datasets listed on the welcome screen.
const WELCOME_RECENT: usize = 5;

/// Shortcut list shown in the shortcuts window, as (keys, action).
pub(super) const SHORTCUTS: &[(&str, &str)] = &[
    ("Space", "Play / pause"),
    ("Right / Left arrow", "Step one frame forward / back"),
    ("Up / Down arrow", "Step ten frames forward / back"),
    ("Esc", "Close this list, or leave fullscreen"),
    ("? or F1", "Show or hide this list"),
    ("Drag in the view", "Pan"),
    ("Scroll in the view", "Zoom at the pointer"),
    (
        "Shift + drag in the view",
        "Lasso-select cells (Ctrl/Cmd adds, Alt removes)",
    ),
    ("Hover a cell", "Show its name and annotations"),
    (
        "Click a legend entry",
        "Highlight that group (click again to clear)",
    ),
    ("Drop a file", "Open a .h5ad (converts it) or a .stviz"),
];

impl StvizApp {
    /// Thin bar above everything: app name and the most used actions.
    pub(super) fn ui_top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.strong("stviz-animate");
            ui.weak(concat!("v", env!("CARGO_PKG_VERSION")));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .button("Shortcuts (?)")
                    .on_hover_text("Keyboard and mouse shortcuts.")
                    .clicked()
                {
                    self.show_shortcuts = !self.show_shortcuts;
                }
                if ui
                    .button("Open data...")
                    .on_hover_text("Open a .h5ad (converted on the fly) or a .stviz file.")
                    .clicked()
                {
                    self.pick_data_file();
                }
            });
        });
    }

    /// File picker for .h5ad and .stviz files.
    pub(super) fn pick_data_file(&mut self) {
        let dialog = rfd::FileDialog::new()
            .add_filter("Datasets", &["h5ad", "stviz"])
            .add_filter("h5ad", &["h5ad"])
            .add_filter("stviz", &["stviz"])
            .set_title("Open a .h5ad or .stviz file")
            .set_directory(self.dialog_dir());
        if let Some(path) = dialog.pick_file() {
            self.open_data_file(&path);
        }
    }

    /// Shown in the viewport while no dataset is loaded.
    pub(super) fn ui_welcome(&mut self, ui: &mut egui::Ui, rect: egui::Rect) {
        let width = rect.width().clamp(240.0, 460.0);
        let card = egui::Rect::from_center_size(
            rect.center(),
            egui::vec2(width, rect.height().min(520.0)),
        )
        .intersect(rect.shrink(12.0));
        let text_color = super::util::contrast_color(self.background_color);
        let weak = text_color.gamma_multiply(0.7);
        ui.scope_builder(
            egui::UiBuilder::new()
                .max_rect(card)
                .layout(egui::Layout::top_down(egui::Align::Center)),
            |ui| {
                ui.visuals_mut().override_text_color = Some(text_color);
                ui.label(egui::RichText::new("stviz-animate").size(28.0).strong());
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(
                        "Animate single-cell and spatial data between spaces \
                         (spatial, UMAP, PCA, ...) and export videos.",
                    )
                    .color(weak),
                );
                ui.add_space(18.0);

                let busy = self.convert_running() || self.dataset_load.is_some();
                if busy {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        let status = if self.convert_running() {
                            self.convert_status.as_deref().unwrap_or("Preparing...")
                        } else {
                            "Loading dataset..."
                        };
                        ui.label(status);
                    });
                    if self.convert_running() && ui.button("Cancel").clicked() {
                        self.cancel_convert();
                    }
                } else {
                    if self.can_install_python() {
                        if let Some(status) = self.convert_status.clone() {
                            ui.label(status);
                        }
                        self.ui_install_python_button(ui);
                        ui.add_space(12.0);
                    }
                    let open = ui.add_sized(
                        [220.0, 32.0],
                        egui::Button::new(egui::RichText::new("Open a dataset...").size(16.0)),
                    );
                    if open
                        .on_hover_text("Pick a .h5ad (converted to .stviz first) or a .stviz file.")
                        .clicked()
                    {
                        self.pick_data_file();
                    }
                    ui.add_space(6.0);
                    let demo = ui.add_sized([220.0, 28.0], egui::Button::new("Try demo data"));
                    if demo
                        .on_hover_text(format!(
                            "Generate a synthetic spatial dataset with {} cells and load it. \
                             The first time takes a few minutes.",
                            DEMO_CELLS
                        ))
                        .clicked()
                    {
                        self.mock_cells = DEMO_CELLS;
                        self.start_mock_dataset();
                    }
                    ui.add_space(6.0);
                    ui.label(
                        egui::RichText::new("or drop a .h5ad / .stviz file anywhere").color(weak),
                    );
                }

                if let Some(err) = self.last_error.clone() {
                    ui.add_space(8.0);
                    ui.colored_label(egui::Color32::from_rgb(230, 110, 100), err);
                }

                let recent: Vec<_> = self
                    .recent_files
                    .iter()
                    .take(WELCOME_RECENT)
                    .cloned()
                    .collect();
                if !recent.is_empty() && !busy {
                    ui.add_space(16.0);
                    ui.label(egui::RichText::new("Recent").strong());
                    for path in recent {
                        let name = path
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_else(|| path.display().to_string());
                        if ui
                            .link(name)
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
                        }
                    }
                }

                ui.add_space(20.0);
                ui.label(egui::RichText::new("How it works").strong());
                ui.add_space(2.0);
                for step in [
                    "1. Load a dataset. Each .obsm embedding becomes a space.",
                    "2. In the timeline below, give each keyframe a space and a colouring.",
                    "3. Press Space to preview, then use Export loop video in the left panel.",
                ] {
                    ui.label(egui::RichText::new(step).color(weak));
                }
                ui.add_space(10.0);
                ui.label(egui::RichText::new("Press ? for keyboard shortcuts.").color(weak));
            },
        );
    }

    /// Floating list of shortcuts, toggled with ? or F1.
    pub(super) fn ui_shortcuts_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_shortcuts;
        egui::Window::new("Keyboard shortcuts")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                egui::Grid::new("shortcuts_grid")
                    .num_columns(2)
                    .spacing([24.0, 6.0])
                    .striped(true)
                    .show(ui, |ui| {
                        for (keys, action) in SHORTCUTS {
                            ui.strong(*keys);
                            ui.label(*action);
                            ui.end_row();
                        }
                    });
                ui.add_space(6.0);
                ui.weak("Shortcuts are ignored while you type in a text field.");
            });
        self.show_shortcuts = open;
    }
}
