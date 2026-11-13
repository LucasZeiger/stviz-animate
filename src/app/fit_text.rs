//! Widgets for showing names of unbounded length (genes, obs fields, categories, file
//! paths) in the left panel without widening it: the text is cut with an ellipsis to fit
//! the space available, and the full name is shown in a tooltip on hover.

use eframe::egui;

/// Narrowest the left side panel can be dragged to.
pub(super) const LEFT_PANEL_MIN_WIDTH: f32 = 240.0;
/// Widest the left side panel can be dragged to, on a large enough window.
pub(super) const LEFT_PANEL_MAX_WIDTH: f32 = 480.0;
/// Largest share of the window width the left side panel may take.
const LEFT_PANEL_MAX_FRACTION: f32 = 0.4;

/// Maximum width of the left side panel for a window of the given width.
pub(super) fn left_panel_max_width(window_width: f32) -> f32 {
    (window_width * LEFT_PANEL_MAX_FRACTION).clamp(LEFT_PANEL_MIN_WIDTH, LEFT_PANEL_MAX_WIDTH)
}

/// Width `text` takes on one line in the given text style.
fn text_width(ui: &egui::Ui, text: &str, style: egui::TextStyle) -> f32 {
    egui::WidgetText::from(text)
        .into_galley(ui, Some(egui::TextWrapMode::Extend), f32::INFINITY, style)
        .size()
        .x
}

/// A single-line label cut to the available width; hovering shows the full text when cut.
pub(super) fn truncated_label(
    ui: &mut egui::Ui,
    text: impl Into<egui::WidgetText>,
) -> egui::Response {
    ui.add(egui::Label::new(text).truncate())
}

/// A selectable list entry cut to the available width, with the full name on hover when cut.
pub(super) fn selectable_name(ui: &mut egui::Ui, selected: bool, name: &str) -> egui::Response {
    let room = ui.available_width() - 2.0 * ui.spacing().button_padding.x;
    let response = ui.add(egui::Button::selectable(selected, name).truncate());
    if text_width(ui, name, egui::TextStyle::Button) > room {
        response.on_hover_text(name)
    } else {
        response
    }
}

/// Like `ComboBox::from_label(label).selected_text(selected)`, but the selected text is cut
/// so that the combo box and its label together fit the available width. Hovering the combo
/// box shows the full selected name when it was cut. `min_width` is the combo box's
/// preferred minimum width (`ComboBox::width`), honoured as far as the space allows.
pub(super) fn fitted_combo<R>(
    ui: &mut egui::Ui,
    label: &str,
    min_width: Option<f32>,
    selected: &str,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) -> Option<R> {
    ui.horizontal(|ui| {
        let spacing = ui.spacing().clone();
        let label_width = text_width(ui, label, egui::TextStyle::Body) + spacing.item_spacing.x;
        let combo_width = (ui.available_width() - label_width).max(80.0);
        let text_room = combo_width
            - 2.0 * spacing.button_padding.x
            - spacing.icon_width
            - spacing.icon_spacing;

        let inner = ui
            .allocate_ui(egui::vec2(combo_width, spacing.interact_size.y), |ui| {
                egui::ComboBox::from_id_salt(label)
                    .width(combo_width.min(min_width.unwrap_or(spacing.combo_width)))
                    .selected_text(selected)
                    .truncate()
                    .show_ui(ui, add_contents)
            })
            .inner;
        if text_width(ui, selected, egui::TextStyle::Button) > text_room {
            inner.response.on_hover_text(selected);
        }
        ui.label(label);
        inner.inner
    })
    .inner
}

#[cfg(test)]
mod tests {
    use super::*;

    const LONG: &str = "a_really_long_gene_or_observation_name_that_goes_on_and_on_and_on_\
                        without_any_spaces_to_wrap_at_and_would_push_the_panel_far_into_the_view";

    /// Lays out the left-panel widgets with a very long name and returns the panel width.
    fn panel_width_with(add: impl Fn(&mut egui::Ui)) -> f32 {
        let ctx = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1600.0, 900.0),
            )),
            ..Default::default()
        };
        let mut width = 0.0;
        // A few frames, since the panel remembers its width from the previous frame.
        for _ in 0..3 {
            let _ = ctx.run(input.clone(), |ctx| {
                let r = egui::SidePanel::left("left_panel")
                    .resizable(true)
                    .default_width(320.0)
                    .min_width(LEFT_PANEL_MIN_WIDTH)
                    .max_width(left_panel_max_width(1600.0))
                    .show(ctx, |ui| {
                        egui::ScrollArea::vertical()
                            .auto_shrink([false; 2])
                            .show(ui, |ui| add(ui));
                    });
                width = r.response.rect.width();
            });
        }
        width
    }

    #[test]
    fn max_width_follows_window() {
        assert_eq!(left_panel_max_width(400.0), LEFT_PANEL_MIN_WIDTH);
        assert_eq!(left_panel_max_width(1000.0), 400.0);
        assert_eq!(left_panel_max_width(4000.0), LEFT_PANEL_MAX_WIDTH);
    }

    #[test]
    fn long_label_in_row_widens_panel_without_truncation() {
        // Guards the test itself: the unfixed widgets do push the panel out.
        let w = panel_width_with(|ui| {
            ui.horizontal(|ui| {
                ui.label(LONG);
            });
        });
        assert!(w > LEFT_PANEL_MAX_WIDTH + 1.0, "panel width {w}");
    }

    #[test]
    fn long_names_keep_panel_within_max_width() {
        let w = panel_width_with(|ui| {
            ui.horizontal(|ui| {
                ui.label("x");
                truncated_label(ui, LONG);
            });
            let _ = selectable_name(ui, true, LONG);
            let _ = fitted_combo(ui, "Obs (cat)", Some(200.0), LONG, |ui| {
                ui.label("item");
            });
        });
        assert!(w <= LEFT_PANEL_MAX_WIDTH + 1.0, "panel width {w}");
    }
}
