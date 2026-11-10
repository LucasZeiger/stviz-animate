//! The advanced (node-based) timeline editor.

use super::{
    util::constrain_card_pos, AdvancedCard, AdvancedCardFilter, AdvancedConnection,
    AdvancedConnectionDrag, ColorKey, KeyColorKind, KeyframeFilter, StvizApp,
    ADVANCED_VIEW_FPS_CAP_HZ, MAX_FILTER_CATEGORIES,
};
use crate::data::{Dataset, ObsMeta};
use eframe::egui;
use std::sync::Arc;

impl StvizApp {
    pub(super) fn ui_advanced_timeline(&mut self, ctx: &egui::Context) {
        if !self.advanced_timeline_open {
            return;
        }
        let viewport_id = egui::ViewportId::from_hash_of("advanced_timeline");
        let viewport_exists = ctx.input(|i| i.raw.viewports.contains_key(&viewport_id));
        let mut builder = egui::ViewportBuilder::default()
            .with_title("Advanced timeline")
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([800.0, 600.0])
            .with_resizable(true)
            .with_clamp_size_to_monitor_size(true);
        if !viewport_exists {
            if let Some(size) = self.advanced_viewport_size {
                if size.x > 0.0 && size.y > 0.0 {
                    builder = builder.with_inner_size(egui::vec2(size.x, size.y));
                }
            }
            if let Some(pos) = self.advanced_viewport_pos {
                builder = builder.with_position(pos);
            }
        }
        let mut close_requested = false;
        ctx.show_viewport_immediate(viewport_id, builder, |ctx, class| {
            let now = ctx.input(|i| i.time);
            let min_dt = 1.0f64 / ADVANCED_VIEW_FPS_CAP_HZ as f64;
            let elapsed = now - self.advanced_last_render_time;
            if elapsed < min_dt {
                ctx.request_repaint_after(std::time::Duration::from_secs_f64(min_dt - elapsed));
            } else {
                self.advanced_last_render_time = now;
            }
            let wants_close = ctx.input(|i| {
                i.raw
                    .viewports
                    .get(&viewport_id)
                    .map(|v| v.close_requested())
                    .unwrap_or(false)
            });
            if wants_close {
                close_requested = true;
            }
            if !matches!(class, egui::ViewportClass::Embedded) {
                if let Some(info) = ctx.input(|i| i.raw.viewports.get(&viewport_id).cloned()) {
                    let is_fullscreen = info.fullscreen.unwrap_or(false);
                    let is_minimized = info.minimized.unwrap_or(false);
                    if !is_fullscreen && !is_minimized {
                        if let Some(rect) = info.inner_rect.or(info.outer_rect) {
                            let size = rect.size();
                            if size.x > 100.0 && size.y > 100.0 {
                                self.advanced_viewport_size = Some(size);
                            }
                        }
                        if let Some(rect) = info.outer_rect.or(info.inner_rect) {
                            let pos = rect.min;
                            self.advanced_viewport_pos = Some(pos);
                        }
                    }
                }
            }
            match class {
                egui::ViewportClass::Embedded => {
                    let mut open = true;
                    egui::Window::new("Advanced timeline")
                        .open(&mut open)
                        .resizable(true)
                        .default_size([1200.0, 800.0])
                        .show(ctx, |ui| {
                            self.advanced_timeline_contents(ctx, ui, viewport_id);
                        });
                    if !open {
                        close_requested = true;
                    }
                }
                _ => {
                    egui::CentralPanel::default().show(ctx, |ui| {
                        self.advanced_timeline_contents(ctx, ui, viewport_id);
                    });
                }
            }
        });
        if close_requested {
            self.advanced_timeline_open = false;
            self.advanced_preview_card = None;
        }
    }

    pub(super) fn advanced_timeline_contents(
        &mut self,
        ctx: &egui::Context,
        ui: &mut egui::Ui,
        viewport_id: egui::ViewportId,
    ) {
        ctx.set_visuals(Self::visuals_for_theme(self.ui_theme));
        let (is_fullscreen, is_maximized) = ctx.input(|i| {
            let info = i.raw.viewports.get(&viewport_id);
            (
                info.and_then(|v| v.fullscreen).unwrap_or(false),
                info.and_then(|v| v.maximized).unwrap_or(false),
            )
        });
        let modifiers = ctx.input(|i| i.modifiers);
        let mut preview_card: Option<usize> = None;
        let mut remove_idx: Option<usize> = None;
        let mut add_card_requested = false;
        ui.horizontal(|ui| {
            if ui.button("Sync from timeline").clicked() {
                self.sync_advanced_from_timeline();
            }
            if ui.button("Apply to timeline").clicked() {
                self.apply_advanced_to_timeline();
            }
            if ui.button("Add card").clicked() {
                add_card_requested = true;
            }
            ui.checkbox(&mut self.advanced_grid_mode, "Grid");
            ui.add_enabled(
                self.advanced_grid_mode,
                egui::DragValue::new(&mut self.advanced_grid_size)
                    .speed(2.0)
                    .range(24.0..=120.0)
                    .prefix("Grid "),
            );
            let fullscreen_label = if is_fullscreen {
                "Exit fullscreen"
            } else {
                "Fullscreen"
            };
            if ui.button(fullscreen_label).clicked() {
                ctx.send_viewport_cmd_to(
                    viewport_id,
                    egui::ViewportCommand::Fullscreen(!is_fullscreen),
                );
                if !is_fullscreen {
                    ctx.send_viewport_cmd_to(viewport_id, egui::ViewportCommand::Maximized(true));
                }
            }
            let maximize_label = if is_maximized {
                "Restore size"
            } else {
                "Maximize"
            };
            if ui.button(maximize_label).clicked() {
                ctx.send_viewport_cmd_to(
                    viewport_id,
                    egui::ViewportCommand::Maximized(!is_maximized),
                );
            }
            if ui.button("Close").clicked() {
                self.advanced_timeline_open = false;
                self.advanced_preview_card = None;
                ctx.send_viewport_cmd_to(viewport_id, egui::ViewportCommand::Close);
            }
        });
        ui.separator();

        let ds_opt = self.dataset.clone();
        let categorical_opts: Vec<(usize, String)> = ds_opt
            .as_ref()
            .map(|ds| {
                ds.meta
                    .obs
                    .iter()
                    .enumerate()
                    .filter_map(|(i, o)| match o {
                        ObsMeta::Categorical { name, .. } => Some((i, name.clone())),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default();
        let filterable_opts: Vec<(usize, String)> = ds_opt
            .as_ref()
            .map(|ds| {
                ds.meta
                    .obs
                    .iter()
                    .enumerate()
                    .filter_map(|(i, o)| match o {
                        ObsMeta::Categorical {
                            name, categories, ..
                        } if categories.len() <= MAX_FILTER_CATEGORIES => Some((i, name.clone())),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default();
        let continuous_opts: Vec<(usize, String)> = ds_opt
            .as_ref()
            .map(|ds| {
                ds.meta
                    .obs
                    .iter()
                    .enumerate()
                    .filter_map(|(i, o)| match o {
                        ObsMeta::Continuous { name, .. } => Some((i, name.clone())),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default();

        let default_card_size = egui::vec2(220.0, 150.0);
        for card in &mut self.advanced_cards {
            if card.size.x < default_card_size.x || card.size.y < default_card_size.y {
                card.size = default_card_size;
            }
        }

        let canvas_rect = ui.available_rect_before_wrap();
        let canvas_response = ui.allocate_rect(canvas_rect, egui::Sense::click());
        {
            let painter = ui.painter();
            painter.rect_filled(canvas_rect, 0.0, ui.visuals().extreme_bg_color);
            if self.advanced_grid_mode {
                let step = self.advanced_grid_size.max(24.0);
                let grid_color = egui::Color32::from_gray(40);
                let cols = (canvas_rect.width() / step).ceil() as i32;
                let rows = (canvas_rect.height() / step).ceil() as i32;
                let max_lines = 200;
                for i in 0..cols.min(max_lines) {
                    let x = canvas_rect.left() + i as f32 * step;
                    painter.line_segment(
                        [
                            egui::pos2(x, canvas_rect.top()),
                            egui::pos2(x, canvas_rect.bottom()),
                        ],
                        egui::Stroke::new(1.0_f32, grid_color),
                    );
                }
                for j in 0..rows.min(max_lines) {
                    let y = canvas_rect.top() + j as f32 * step;
                    painter.line_segment(
                        [
                            egui::pos2(canvas_rect.left(), y),
                            egui::pos2(canvas_rect.right(), y),
                        ],
                        egui::Stroke::new(1.0_f32, grid_color),
                    );
                }
            }
        }

        let inspector_size = egui::vec2(320.0, 360.0);
        let inspector_rect =
            egui::Rect::from_min_size(canvas_rect.min + egui::vec2(12.0, 12.0), inspector_size);
        let pointer_pos = ctx.input(|i| i.pointer.latest_pos());
        if add_card_requested {
            let pos = egui::pos2(inspector_rect.right() + 24.0, inspector_rect.top());
            self.add_advanced_card_at(canvas_rect, inspector_rect, pos);
        }
        for i in 0..self.advanced_cards.len() {
            self.resolve_card_collisions(i, canvas_rect, inspector_rect);
        }
        let mut card_positions: Vec<egui::Pos2> = Vec::with_capacity(self.advanced_cards.len());
        for card in &self.advanced_cards {
            card_positions.push(constrain_card_pos(
                card.pos,
                card.size,
                canvas_rect,
                inspector_rect,
            ));
        }
        let mut connect_state = self.advanced_connecting;
        if let Some(connect) = &mut connect_state {
            if let Some((pos, size)) = self
                .advanced_cards
                .get(connect.from_idx)
                .map(|card| (card_positions[connect.from_idx], card.size))
            {
                let rect = egui::Rect::from_min_size(canvas_rect.min + pos.to_vec2(), size);
                connect.start_pos = if connect.from_is_output {
                    egui::pos2(rect.right(), rect.center().y)
                } else {
                    egui::pos2(rect.left(), rect.center().y)
                };
            }
        }
        let connection_stroke = egui::Stroke::new(1.4_f32, ui.visuals().selection.stroke.color);
        {
            let painter = ui.painter();
            for conn in &self.advanced_connections {
                if conn.from >= card_positions.len() || conn.to >= card_positions.len() {
                    continue;
                }
                let from_card = &self.advanced_cards[conn.from];
                let to_card = &self.advanced_cards[conn.to];
                let from_rect = egui::Rect::from_min_size(
                    canvas_rect.min + card_positions[conn.from].to_vec2(),
                    from_card.size,
                );
                let to_rect = egui::Rect::from_min_size(
                    canvas_rect.min + card_positions[conn.to].to_vec2(),
                    to_card.size,
                );
                let from_pos = egui::pos2(from_rect.right(), from_rect.center().y);
                let to_pos = egui::pos2(to_rect.left(), to_rect.center().y);
                painter.line_segment([from_pos, to_pos], connection_stroke);
            }
        }
        let mut clicked_on_card = false;
        let mut card_drag_started = false;
        let mut selected_card = self.advanced_selected_card;
        let mut selected_cards = self.advanced_selected_cards.clone();
        let mut drag_idx = self.advanced_drag_idx;
        let mut drag_offset = self.advanced_drag_offset;
        let mut group_drag = self.advanced_drag_group.clone();
        let mut group_drag_pointer = self.advanced_drag_pointer_start;
        let mut start_group_drag: Option<egui::Pos2> = None;
        let mut marquee_start = self.advanced_marquee_start;
        let mut context_add_pos: Option<egui::Pos2> = None;
        let mut node_hits: Vec<(usize, bool, egui::Pos2)> = Vec::new();
        let mut resolve_idx: Option<usize> = None;
        let mut remove_indices: Vec<usize> = Vec::new();
        for (i, card) in self.advanced_cards.iter_mut().enumerate() {
            card.pos = constrain_card_pos(card.pos, card.size, canvas_rect, inspector_rect);
            let rect = egui::Rect::from_min_size(canvas_rect.min + card.pos.to_vec2(), card.size);
            let selected = selected_card == Some(i);
            let fill = if selected {
                ui.visuals().widgets.active.bg_fill
            } else {
                ui.visuals().widgets.inactive.bg_fill
            };
            let stroke = if selected {
                egui::Stroke::new(1.5_f32, ui.visuals().selection.stroke.color)
            } else {
                ui.visuals().widgets.noninteractive.bg_stroke
            };
            let node_r = 5.0;
            let in_pos = egui::pos2(rect.left(), rect.center().y);
            let out_pos = egui::pos2(rect.right(), rect.center().y);
            node_hits.push((i, false, in_pos));
            node_hits.push((i, true, out_pos));
            let header_height = 34.0;
            {
                let painter = ui.painter();
                painter.rect_filled(rect, 6.0, fill);
                painter.rect_stroke(rect, 6.0, stroke, egui::StrokeKind::Inside);
                let label = ds_opt
                    .as_ref()
                    .and_then(|ds| ds.meta.spaces.get(card.space_idx))
                    .map(|s| s.name.as_str())
                    .unwrap_or("Card");
                painter.text(
                    rect.min + egui::vec2(8.0, 6.0),
                    egui::Align2::LEFT_TOP,
                    label,
                    egui::FontId::proportional(12.0),
                    ui.visuals().text_color(),
                );
                painter.text(
                    rect.min + egui::vec2(8.0, 20.0),
                    egui::Align2::LEFT_TOP,
                    format!("{:.2}s", card.duration_sec),
                    egui::FontId::proportional(11.0),
                    ui.visuals().text_color(),
                );
                let node_color = ui.visuals().selection.bg_fill;
                let disabled_color = ui.visuals().widgets.noninteractive.fg_stroke.color;
                let in_color = if card.in_enabled {
                    node_color
                } else {
                    disabled_color
                };
                let out_color = if card.out_enabled {
                    node_color
                } else {
                    disabled_color
                };
                painter.circle_filled(in_pos, node_r, in_color);
                painter.circle_filled(out_pos, node_r, out_color);
                painter.circle_stroke(in_pos, node_r, egui::Stroke::new(1.0_f32, stroke.color));
                painter.circle_stroke(out_pos, node_r, egui::Stroke::new(1.0_f32, stroke.color));
            }

            let node_size = egui::vec2(node_r * 2.6, node_r * 2.6);
            let in_rect = egui::Rect::from_center_size(in_pos, node_size);
            let out_rect = egui::Rect::from_center_size(out_pos, node_size);
            let in_id = ui.id().with(("adv_node_in", card.id));
            let out_id = ui.id().with(("adv_node_out", card.id));
            let in_resp = ui.interact(in_rect, in_id, egui::Sense::click_and_drag());
            let out_resp = ui.interact(out_rect, out_id, egui::Sense::click_and_drag());
            if in_resp.drag_started() {
                selected_card = Some(i);
                clicked_on_card = true;
                connect_state = Some(AdvancedConnectionDrag {
                    from_idx: i,
                    from_is_output: false,
                    start_pos: in_pos,
                });
            }
            if out_resp.drag_started() {
                selected_card = Some(i);
                clicked_on_card = true;
                connect_state = Some(AdvancedConnectionDrag {
                    from_idx: i,
                    from_is_output: true,
                    start_pos: out_pos,
                });
            }

            let mut inner_rect = rect.shrink(8.0);
            inner_rect.min.y += header_height;
            if inner_rect.is_positive() {
                ui.scope_builder(egui::UiBuilder::new().max_rect(inner_rect), |ui| {
                    ui.set_min_size(inner_rect.size());
                    ui.spacing_mut().item_spacing = egui::vec2(4.0, 2.0);
                    render_advanced_card_controls(
                        ui,
                        card,
                        ds_opt.as_deref(),
                        &categorical_opts,
                        &continuous_opts,
                        110.0,
                    );
                });
            }

            let pointer_over_controls = pointer_pos
                .map(|pos| inner_rect.contains(pos))
                .unwrap_or(false);
            let pointer_in_rect = pointer_pos.map(|pos| rect.contains(pos)).unwrap_or(false);
            let node_active = in_resp.dragged()
                || in_resp.drag_started()
                || out_resp.dragged()
                || out_resp.drag_started();
            let id = ui.id().with(("adv_card", card.id));
            let sense = if pointer_over_controls {
                egui::Sense::hover()
            } else {
                egui::Sense::click_and_drag()
            };
            let response = ui.interact(rect, id, sense);
            if pointer_over_controls
                && drag_idx.is_none()
                && !node_active
                && ctx.input(|i| i.pointer.any_pressed())
                && pointer_in_rect
            {
                card_drag_started = true;
                if selected_cards.contains(&i) && selected_cards.len() > 1 {
                    start_group_drag = pointer_pos;
                    drag_idx = Some(i);
                } else if let Some(pos) = pointer_pos {
                    drag_idx = Some(i);
                    drag_offset = pos.to_vec2() - rect.min.to_vec2();
                }
            }
            let clicked = if pointer_over_controls {
                pointer_in_rect && ctx.input(|i| i.pointer.primary_clicked())
            } else {
                response.clicked()
            };
            if clicked {
                selected_card = Some(i);
                clicked_on_card = true;
                if modifiers.shift || modifiers.ctrl {
                    if selected_cards.contains(&i) {
                        selected_cards.remove(&i);
                        if selected_card == Some(i) {
                            selected_card = selected_cards.iter().copied().next();
                        }
                    } else {
                        selected_cards.insert(i);
                        selected_card = Some(i);
                    }
                } else {
                    selected_cards.clear();
                    selected_cards.insert(i);
                }
            }
            response.context_menu(|ui| {
                if ui.button("Add card").clicked() {
                    context_add_pos = pointer_pos;
                    ui.close();
                }
                if selected_cards.contains(&i) && selected_cards.len() > 1 {
                    if ui.button("Remove selected cards").clicked() {
                        remove_indices.extend(selected_cards.iter().copied());
                        ui.close();
                    }
                } else if ui.button("Remove card").clicked() {
                    remove_indices.push(i);
                    ui.close();
                }
            });
            if !node_active {
                if response.drag_started() {
                    card_drag_started = true;
                    if selected_cards.contains(&i) && selected_cards.len() > 1 {
                        start_group_drag = response.interact_pointer_pos().or(pointer_pos);
                        drag_idx = Some(i);
                    } else {
                        drag_idx = Some(i);
                        if let Some(pos) = response.interact_pointer_pos() {
                            drag_offset = pos.to_vec2() - rect.min.to_vec2();
                        }
                    }
                }
                let should_drag = response.dragged()
                    || (pointer_over_controls
                        && drag_idx == Some(i)
                        && ctx.input(|i| i.pointer.is_decidedly_dragging()));
                if should_drag && group_drag.is_none() && start_group_drag.is_none() {
                    if let Some(pos) = response.interact_pointer_pos().or(pointer_pos) {
                        let mut local = pos.to_vec2() - canvas_rect.min.to_vec2() - drag_offset;
                        let max_x = (canvas_rect.width() - card.size.x).max(0.0);
                        let max_y = (canvas_rect.height() - card.size.y).max(0.0);
                        if self.advanced_grid_mode {
                            let step = self.advanced_grid_size.max(24.0);
                            local.x = (local.x / step).round() * step;
                            local.y = (local.y / step).round() * step;
                        }
                        local.x = local.x.clamp(0.0, max_x);
                        local.y = local.y.clamp(0.0, max_y);
                        card.pos = constrain_card_pos(
                            egui::pos2(local.x, local.y),
                            card.size,
                            canvas_rect,
                            inspector_rect,
                        );
                        resolve_idx = Some(i);
                    }
                }
            }
        }
        if let Some(start_pos) = start_group_drag {
            let mut group = Vec::with_capacity(selected_cards.len());
            for idx in selected_cards.iter().copied() {
                if let Some(card) = self.advanced_cards.get(idx) {
                    group.push((idx, card.pos));
                }
            }
            if !group.is_empty() {
                group_drag = Some(group);
                group_drag_pointer = Some(start_pos);
            }
        }
        if let (Some(group), Some(start_pos), Some(pos)) =
            (&group_drag, group_drag_pointer, pointer_pos)
        {
            if ctx.input(|i| i.pointer.any_down()) {
                let delta = pos - start_pos;
                for (idx, base_pos) in group {
                    if let Some(card) = self.advanced_cards.get_mut(*idx) {
                        let mut local = base_pos.to_vec2() + delta;
                        let max_x = (canvas_rect.width() - card.size.x).max(0.0);
                        let max_y = (canvas_rect.height() - card.size.y).max(0.0);
                        if self.advanced_grid_mode {
                            let step = self.advanced_grid_size.max(24.0);
                            local.x = (local.x / step).round() * step;
                            local.y = (local.y / step).round() * step;
                        }
                        local.x = local.x.clamp(0.0, max_x);
                        local.y = local.y.clamp(0.0, max_y);
                        card.pos = constrain_card_pos(
                            egui::pos2(local.x, local.y),
                            card.size,
                            canvas_rect,
                            inspector_rect,
                        );
                    }
                }
                for idx in selected_cards.iter().copied() {
                    self.resolve_card_collisions(idx, canvas_rect, inspector_rect);
                }
            }
        }
        if let (Some(connect), Some(pos)) = (connect_state, pointer_pos) {
            ui.painter()
                .line_segment([connect.start_pos, pos], connection_stroke);
        }

        if drag_idx.is_some() && !ctx.input(|i| i.pointer.any_down()) {
            drag_idx = None;
            group_drag = None;
            group_drag_pointer = None;
        }

        if canvas_response.clicked() && !clicked_on_card {
            selected_card = None;
            selected_cards.clear();
        }
        if canvas_response.drag_started() && !card_drag_started {
            marquee_start = canvas_response.interact_pointer_pos();
            selected_cards.clear();
            selected_card = None;
        }
        if let (Some(start), Some(pos)) = (marquee_start, pointer_pos) {
            if ctx.input(|i| i.pointer.any_down()) {
                selected_cards.clear();
                let selection_rect = egui::Rect::from_two_pos(start, pos).intersect(canvas_rect);
                let painter = ui.painter();
                painter.rect_filled(
                    selection_rect,
                    0.0,
                    egui::Color32::from_rgba_unmultiplied(90, 160, 255, 40),
                );
                painter.rect_stroke(
                    selection_rect,
                    0.0,
                    egui::Stroke::new(1.2_f32, egui::Color32::from_rgb(120, 200, 255)),
                    egui::StrokeKind::Inside,
                );
                for (i, card) in self.advanced_cards.iter().enumerate() {
                    let card_rect =
                        egui::Rect::from_min_size(canvas_rect.min + card.pos.to_vec2(), card.size);
                    if selection_rect.intersects(card_rect) {
                        selected_cards.insert(i);
                    }
                }
                selected_card = selected_cards.iter().copied().next();
            } else {
                marquee_start = None;
            }
        }
        if let Some(connect) = connect_state {
            if !ctx.input(|i| i.pointer.any_down()) {
                if let Some(pos) = pointer_pos {
                    let mut best: Option<(usize, bool, f32)> = None;
                    for (idx, is_output, node_pos) in &node_hits {
                        let dist = node_pos.distance(pos);
                        if dist <= 12.0 {
                            if best.as_ref().map(|b| dist < b.2).unwrap_or(true) {
                                best = Some((*idx, *is_output, dist));
                            }
                        }
                    }
                    if let Some((target_idx, target_is_output, _)) = best {
                        if connect.from_is_output && !target_is_output {
                            let from = connect.from_idx;
                            let to = target_idx;
                            if from != to
                                && !self
                                    .advanced_connections
                                    .iter()
                                    .any(|c| c.from == from && c.to == to)
                            {
                                self.advanced_connections
                                    .push(AdvancedConnection { from, to });
                            }
                        } else if !connect.from_is_output && target_is_output {
                            let from = target_idx;
                            let to = connect.from_idx;
                            if from != to
                                && !self
                                    .advanced_connections
                                    .iter()
                                    .any(|c| c.from == from && c.to == to)
                            {
                                self.advanced_connections
                                    .push(AdvancedConnection { from, to });
                            }
                        }
                    }
                }
                self.advanced_connecting = None;
            } else {
                self.advanced_connecting = Some(connect);
            }
        } else {
            self.advanced_connecting = None;
        }

        canvas_response.context_menu(|ui| {
            if ui.button("Add card").clicked() {
                if let Some(pos) = pointer_pos {
                    self.add_advanced_card_at(canvas_rect, inspector_rect, pos);
                }
                ui.close();
            }
        });

        egui::Area::new(egui::Id::new("adv_card_inspector"))
            .fixed_pos(inspector_rect.min)
            .show(ctx, |ui| {
                let mut frame = egui::Frame::NONE
                    .fill(ui.visuals().extreme_bg_color)
                    .stroke(egui::Stroke::new(
                        1.5_f32,
                        ui.visuals().selection.stroke.color,
                    ))
                    .corner_radius(egui::CornerRadius::same(6))
                    .inner_margin(egui::Margin::same(8));
                frame.shadow = egui::epaint::Shadow {
                    color: egui::Color32::from_rgba_unmultiplied(0, 0, 0, 120),
                    offset: [0, 4],
                    blur: 10,
                    spread: 1,
                };
                frame.show(ui, |ui| {
                    ui.set_min_size(inspector_size);
                    ui.label("Card inspector");
                    if selected_cards.len() > 1 {
                        ui.label(format!("{} cards selected", selected_cards.len()));
                        if ui.button("Remove selected cards").clicked() {
                            remove_indices.extend(selected_cards.iter().copied());
                        }
                    } else if let Some(idx) = self.advanced_selected_card {
                        if let Some(card) = self.advanced_cards.get_mut(idx) {
                            if ui.button("Remove card").clicked() {
                                remove_idx = Some(idx);
                            }
                            render_advanced_card_controls(
                                ui,
                                card,
                                ds_opt.as_deref(),
                                &categorical_opts,
                                &continuous_opts,
                                150.0,
                            );
                            ui.horizontal(|ui| {
                                ui.label("Duration");
                                ui.add(
                                    egui::DragValue::new(&mut card.duration_sec)
                                        .speed(0.05)
                                        .range(0.1..=30.0)
                                        .suffix("s"),
                                );
                            });
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut card.in_enabled, "In");
                                ui.checkbox(&mut card.out_enabled, "Out");
                            });

                            if card.filter.as_ref().is_some_and(|filter| {
                                !filterable_opts
                                    .iter()
                                    .any(|(filter_idx, _)| *filter_idx == filter.obs_idx)
                            }) {
                                card.filter = None;
                            }
                            let mut filter_enabled = card.filter.is_some();
                            if ui
                                .add_enabled(
                                    !filterable_opts.is_empty(),
                                    egui::Checkbox::new(&mut filter_enabled, "Filter"),
                                )
                                .on_disabled_hover_text(format!(
                                    "No categorical fields with {} or fewer categories.",
                                    MAX_FILTER_CATEGORIES
                                ))
                                .changed()
                            {
                                if filter_enabled {
                                    let obs_idx =
                                        filterable_opts.first().map(|(idx, _)| *idx).unwrap_or(0);
                                    card.filter = Some(AdvancedCardFilter {
                                        obs_idx,
                                        enabled: Vec::new(),
                                        cached_indices: None,
                                        cached_indices_id: 0,
                                        cached_dataset_id: 0,
                                        cached_max_draw: 0,
                                    });
                                } else {
                                    card.filter = None;
                                    if self.advanced_preview_card == Some(idx) {
                                        self.advanced_preview_card = None;
                                    }
                                }
                            }
                            if let Some(filter) = card.filter.as_mut() {
                                if filterable_opts.is_empty() {
                                    ui.label("No filterable categorical obs.");
                                } else if let Some(ds) = ds_opt.as_ref() {
                                    let mut obs_idx = filter.obs_idx;
                                    egui::ComboBox::from_id_salt(("adv_filter_obs", card.id))
                                        .selected_text(
                                            filterable_opts
                                                .iter()
                                                .find(|(idx, _)| *idx == obs_idx)
                                                .map(|(_, name)| name.as_str())
                                                .unwrap_or("Categorical"),
                                        )
                                        .show_ui(ui, |ui| {
                                            for (idx, name) in &filterable_opts {
                                                ui.selectable_value(&mut obs_idx, *idx, name);
                                            }
                                        });
                                    if obs_idx != filter.obs_idx {
                                        filter.obs_idx = obs_idx;
                                        filter.enabled.clear();
                                        filter.cached_indices = None;
                                        filter.cached_indices_id = 0;
                                        filter.cached_dataset_id = 0;
                                        filter.cached_max_draw = 0;
                                    }
                                    if let Ok((_name, _labels, categories, _pal)) =
                                        ds.obs_categorical(filter.obs_idx)
                                    {
                                        if filter.enabled.len() != categories.len() {
                                            filter.enabled = vec![true; categories.len()];
                                            filter.cached_indices = None;
                                            filter.cached_indices_id = 0;
                                            filter.cached_dataset_id = 0;
                                            filter.cached_max_draw = 0;
                                        }
                                        ui.horizontal(|ui| {
                                            if ui.button("All").clicked() {
                                                for v in &mut filter.enabled {
                                                    *v = true;
                                                }
                                                filter.cached_indices = None;
                                                filter.cached_indices_id = 0;
                                                filter.cached_dataset_id = 0;
                                                filter.cached_max_draw = 0;
                                            }
                                            if ui.button("None").clicked() {
                                                for v in &mut filter.enabled {
                                                    *v = false;
                                                }
                                                filter.cached_indices = None;
                                                filter.cached_indices_id = 0;
                                                filter.cached_dataset_id = 0;
                                                filter.cached_max_draw = 0;
                                            }
                                            if ui.button("Preview").clicked() {
                                                preview_card = Some(idx);
                                            }
                                        });
                                        let row_height = 22.0;
                                        let list_max =
                                            row_height * categories.len().min(10) as f32 + 6.0;
                                        egui::ScrollArea::vertical()
                                            .max_height(list_max)
                                            .auto_shrink([false, false])
                                            .show(ui, |ui| {
                                                for (idx, name) in categories.iter().enumerate() {
                                                    ui.horizontal(|ui| {
                                                        if ui
                                                            .checkbox(&mut filter.enabled[idx], "")
                                                            .changed()
                                                        {
                                                            filter.cached_indices = None;
                                                            filter.cached_indices_id = 0;
                                                            filter.cached_dataset_id = 0;
                                                            filter.cached_max_draw = 0;
                                                        }
                                                        ui.label(name);
                                                    });
                                                }
                                            });
                                    }
                                }
                            }
                        }
                    } else {
                        ui.label("Select a card to edit its settings.");
                    }
                });
            });

        if let Some(pos) = context_add_pos {
            self.add_advanced_card_at(canvas_rect, inspector_rect, pos);
        }
        if let Some(idx) = resolve_idx {
            self.resolve_card_collisions(idx, canvas_rect, inspector_rect);
        }
        if let Some(idx) = remove_idx {
            remove_indices.push(idx);
        }
        if !remove_indices.is_empty() {
            remove_indices.sort_unstable();
            remove_indices.dedup();
            for idx in remove_indices.into_iter().rev() {
                if idx < self.advanced_cards.len() {
                    self.advanced_cards.remove(idx);
                    if selected_card == Some(idx) {
                        selected_card = None;
                    } else if let Some(sel) = selected_card {
                        if sel > idx {
                            selected_card = Some(sel - 1);
                        }
                    }
                    if let Some(preview_idx) = self.advanced_preview_card {
                        if preview_idx == idx {
                            self.advanced_preview_card = None;
                        } else if preview_idx > idx {
                            self.advanced_preview_card = Some(preview_idx - 1);
                        }
                    }
                    selected_cards = selected_cards
                        .iter()
                        .filter_map(|sel| {
                            if *sel == idx {
                                None
                            } else if *sel > idx {
                                Some(sel - 1)
                            } else {
                                Some(*sel)
                            }
                        })
                        .collect();
                }
            }
        }
        if let Some(idx) = preview_card {
            self.advanced_preview_card = Some(idx);
        }

        self.advanced_selected_card = selected_card;
        self.advanced_selected_cards = selected_cards;
        self.advanced_drag_idx = drag_idx;
        self.advanced_drag_offset = drag_offset;
        self.advanced_drag_group = group_drag;
        self.advanced_drag_pointer_start = group_drag_pointer;
        self.advanced_marquee_start = marquee_start;
    }

    pub(super) fn add_advanced_card_at(
        &mut self,
        canvas_rect: egui::Rect,
        inspector_rect: egui::Rect,
        pos: egui::Pos2,
    ) {
        let card_size = egui::vec2(220.0, 150.0);
        let mut local = pos.to_vec2() - canvas_rect.min.to_vec2();
        let max_x = (canvas_rect.width() - card_size.x).max(0.0);
        let max_y = (canvas_rect.height() - card_size.y).max(0.0);
        local.x = local.x.clamp(0.0, max_x);
        local.y = local.y.clamp(0.0, max_y);
        let pos = constrain_card_pos(
            egui::pos2(local.x, local.y),
            card_size,
            canvas_rect,
            inspector_rect,
        );
        let id = self.next_advanced_id();
        let space_idx = self
            .space_path
            .get(self.selected_key_idx.unwrap_or(0))
            .copied()
            .unwrap_or(0);
        let color_key = self
            .color_path
            .get(self.selected_key_idx.unwrap_or(0))
            .cloned()
            .unwrap_or(ColorKey::Current);
        self.advanced_cards.push(AdvancedCard {
            id,
            space_idx,
            color_key,
            duration_sec: 2.0,
            pos,
            size: card_size,
            in_enabled: true,
            out_enabled: true,
            filter: None,
        });
        let idx = self.advanced_cards.len().saturating_sub(1);
        self.resolve_card_collisions(idx, canvas_rect, inspector_rect);
    }

    pub(super) fn resolve_card_collisions(
        &mut self,
        idx: usize,
        canvas_rect: egui::Rect,
        inspector_rect: egui::Rect,
    ) {
        if idx >= self.advanced_cards.len() {
            return;
        }
        let mut iterations = 0;
        while iterations < 8 {
            iterations += 1;
            let (left, rest) = self.advanced_cards.split_at_mut(idx);
            let Some((card, right)) = rest.split_first_mut() else {
                return;
            };
            let mut rect =
                egui::Rect::from_min_size(canvas_rect.min + card.pos.to_vec2(), card.size);
            let mut moved = false;
            for other in left.iter().chain(right.iter()) {
                let other_rect =
                    egui::Rect::from_min_size(canvas_rect.min + other.pos.to_vec2(), other.size);
                if !rect.intersects(other_rect) {
                    continue;
                }
                let move_x = if rect.center().x < other_rect.center().x {
                    other_rect.left() - rect.right()
                } else {
                    other_rect.right() - rect.left()
                };
                let move_y = if rect.center().y < other_rect.center().y {
                    other_rect.top() - rect.bottom()
                } else {
                    other_rect.bottom() - rect.top()
                };
                if move_x.abs() < move_y.abs() {
                    card.pos.x += move_x;
                } else {
                    card.pos.y += move_y;
                }
                card.pos = constrain_card_pos(card.pos, card.size, canvas_rect, inspector_rect);
                rect = egui::Rect::from_min_size(canvas_rect.min + card.pos.to_vec2(), card.size);
                moved = true;
            }
            if !moved {
                break;
            }
        }
    }

    pub(super) fn apply_downsample(&mut self) {
        let max_draw = self.max_draw_points;
        if max_draw == 0 || self.base_indices.len() <= max_draw {
            self.draw_indices = Arc::new(self.base_indices.clone());
            self.indices_id = self.indices_id.wrapping_add(1);
            return;
        }

        let step = (self.base_indices.len() as f32 / max_draw as f32)
            .ceil()
            .max(1.0) as usize;
        let mut out = Vec::with_capacity(max_draw);
        for (i, idx) in self.base_indices.iter().enumerate() {
            if i % step == 0 {
                out.push(*idx);
                if out.len() >= max_draw {
                    break;
                }
            }
        }
        self.draw_indices = Arc::new(out);
        self.indices_id = self.indices_id.wrapping_add(1);
    }

    pub(super) fn load_filter_state(&mut self, ds: &Dataset) {
        if let Ok((_name, _labels, categories, _pal)) = ds.obs_categorical(self.active_obs_idx) {
            if categories.len() > MAX_FILTER_CATEGORIES {
                self.enabled_categories.clear();
                return;
            }
            if let Some(saved) = self.category_state.get(&self.active_obs_idx) {
                if saved.len() == categories.len() {
                    self.enabled_categories = saved.clone();
                    return;
                }
            }
            self.enabled_categories = vec![true; categories.len()];
        }
    }

    pub(super) fn next_advanced_id(&mut self) -> u64 {
        let id = self.advanced_next_id;
        self.advanced_next_id = self.advanced_next_id.wrapping_add(1);
        id
    }

    pub(super) fn sync_advanced_from_timeline(&mut self) {
        self.advanced_connections.clear();
        self.advanced_connecting = None;
        let key_count = self.space_path.len().max(1);
        self.ensure_filter_path_len(key_count);
        let timeline_filters = self.filter_path.clone();
        let card_size = egui::vec2(220.0, 150.0);
        let mut durations = vec![2.0; key_count];
        if self.key_times.len() == key_count {
            let seg_count = key_count.saturating_sub(1);
            for i in 0..seg_count {
                let t0 = self.key_times[i];
                let t1 = self.key_times[i + 1];
                let seg = (t1 - t0).max(0.01);
                durations[i] = seg / self.speed.max(0.01);
            }
            if key_count > 1 {
                durations[key_count - 1] = durations[key_count - 2];
            }
        }
        if self.advanced_cards.len() == key_count {
            for i in 0..key_count {
                if let Some(card) = self.advanced_cards.get_mut(i) {
                    card.space_idx = self.space_path.get(i).copied().unwrap_or(0);
                    card.color_key = self.color_path.get(i).cloned().unwrap_or(ColorKey::Current);
                    card.duration_sec = durations[i].max(0.1);
                    card.filter = timeline_filters[i]
                        .as_ref()
                        .map(|filter| AdvancedCardFilter {
                            obs_idx: filter.obs_idx,
                            enabled: filter.enabled.clone(),
                            cached_indices: None,
                            cached_indices_id: 0,
                            cached_dataset_id: 0,
                            cached_max_draw: 0,
                        });
                }
            }
        } else {
            self.advanced_cards.clear();
            self.advanced_selected_card = None;
            self.advanced_selected_cards.clear();
            self.advanced_preview_card = None;
            let cols = 3usize;
            let start_x = 12.0 + 320.0 + 24.0;
            let start_y = 12.0;
            for i in 0..key_count {
                let col = i % cols;
                let row = i / cols;
                let pos = egui::pos2(
                    start_x + col as f32 * (card_size.x + 16.0),
                    start_y + row as f32 * (card_size.y + 16.0),
                );
                let id = self.next_advanced_id();
                self.advanced_cards.push(AdvancedCard {
                    id,
                    space_idx: self.space_path.get(i).copied().unwrap_or(0),
                    color_key: self.color_path.get(i).cloned().unwrap_or(ColorKey::Current),
                    duration_sec: durations[i].max(0.1),
                    pos,
                    size: card_size,
                    in_enabled: true,
                    out_enabled: true,
                    filter: timeline_filters[i]
                        .as_ref()
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
        }
        if self.advanced_cards.len() >= 2 {
            for i in 0..self.advanced_cards.len() - 1 {
                self.advanced_connections
                    .push(AdvancedConnection { from: i, to: i + 1 });
            }
        }
    }

    pub(super) fn apply_advanced_to_timeline(&mut self) {
        if self.advanced_cards.len() < 2 {
            return;
        }
        let orig_space = self.space_path.clone();
        if orig_space.is_empty() {
            return;
        }
        let mut new_space = Vec::with_capacity(self.advanced_cards.len());
        let mut new_color = Vec::with_capacity(self.advanced_cards.len());
        for card in &self.advanced_cards {
            let idx = card.space_idx.min(orig_space.len().saturating_sub(1));
            new_space.push(idx);
            new_color.push(card.color_key.clone());
        }
        self.space_path = new_space;
        self.color_path = new_color;
        self.filter_path = self
            .advanced_cards
            .iter()
            .map(|card| {
                card.filter.as_ref().map(|filter| KeyframeFilter {
                    obs_idx: filter.obs_idx,
                    enabled: filter.enabled.clone(),
                })
            })
            .collect();
        self.ensure_color_path_len(self.space_path.len());
        let segs = self.space_path.len().saturating_sub(1);
        let mut durations = Vec::with_capacity(segs);
        for card in self.advanced_cards.iter().take(segs) {
            durations.push(card.duration_sec.max(0.05));
        }
        let total = durations.iter().sum::<f32>().max(0.05);
        self.key_times.clear();
        self.key_times.push(0.0);
        let mut acc = 0.0;
        for d in durations {
            acc += d / total;
            self.key_times.push(acc.clamp(0.0, 1.0));
        }
        if let Some(last) = self.key_times.last_mut() {
            *last = 1.0;
        }
        self.speed = (1.0 / total).clamp(0.01, 2.0);
        if self.key_collapsed.len() != self.space_path.len() {
            self.key_collapsed.resize(self.space_path.len(), false);
        }
        self.selected_key_idx = Some(0);
        self.reset_view_key_idx = self
            .reset_view_key_idx
            .min(self.space_path.len().saturating_sub(1));
        self.from_space = *self.space_path.first().unwrap_or(&self.from_space);
        self.to_space = *self.space_path.last().unwrap_or(&self.to_space);
    }

    pub(super) fn mark_grid_dirty(&mut self) {
        self.grid_version = self.grid_version.wrapping_add(1);
        self.grid_cache = None;
    }
}

pub(super) fn render_advanced_card_controls(
    ui: &mut egui::Ui,
    card: &mut AdvancedCard,
    ds: Option<&Dataset>,
    categorical_opts: &[(usize, String)],
    continuous_opts: &[(usize, String)],
    control_width: f32,
) {
    let Some(ds) = ds else {
        ui.label("No dataset.");
        return;
    };
    if ds.meta.spaces.is_empty() {
        ui.label("No spaces.");
        return;
    }

    let max_idx = ds.meta.spaces.len().saturating_sub(1);
    let mut space_idx = card.space_idx.min(max_idx);
    let mut color_key = card.color_key.clone();
    let mut color_kind = match color_key {
        ColorKey::Current => KeyColorKind::Current,
        ColorKey::Categorical(_) => KeyColorKind::Categorical,
        ColorKey::Continuous(_) => KeyColorKind::Continuous,
        ColorKey::Gene(_) => KeyColorKind::Gene,
    };

    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Space").size(10.5));
        egui::ComboBox::from_id_salt(("adv_space", card.id))
            .width(control_width)
            .selected_text(
                ds.meta
                    .spaces
                    .get(space_idx)
                    .map(|s| s.name.as_str())
                    .unwrap_or("?"),
            )
            .show_ui(ui, |ui| {
                for (idx, s) in ds.meta.spaces.iter().enumerate() {
                    ui.selectable_value(&mut space_idx, idx, &s.name);
                }
            });
    });

    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Color").size(10.5));
        let kind_label = match color_kind {
            KeyColorKind::Current => "Current",
            KeyColorKind::Categorical => "Categorical",
            KeyColorKind::Continuous => "Continuous",
            KeyColorKind::Gene => "Gene",
        };
        egui::ComboBox::from_id_salt(("adv_color_kind", card.id))
            .width(control_width)
            .selected_text(kind_label)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut color_kind, KeyColorKind::Current, "Current");
                ui.selectable_value(&mut color_kind, KeyColorKind::Categorical, "Categorical");
                ui.selectable_value(&mut color_kind, KeyColorKind::Continuous, "Continuous");
                ui.selectable_value(&mut color_kind, KeyColorKind::Gene, "Gene");
            });
    });

    match color_kind {
        KeyColorKind::Current => {
            color_key = ColorKey::Current;
        }
        KeyColorKind::Categorical => {
            if categorical_opts.is_empty() {
                ui.label("No categorical obs.");
                color_key = ColorKey::Current;
            } else {
                let mut obs_idx = match color_key {
                    ColorKey::Categorical(idx) => idx,
                    _ => categorical_opts[0].0,
                };
                egui::ComboBox::from_id_salt(("adv_color_cat", card.id))
                    .width(control_width)
                    .selected_text(
                        categorical_opts
                            .iter()
                            .find(|(idx, _)| *idx == obs_idx)
                            .map(|(_, name)| name.as_str())
                            .unwrap_or("?"),
                    )
                    .show_ui(ui, |ui| {
                        for (idx, name) in categorical_opts {
                            ui.selectable_value(&mut obs_idx, *idx, name);
                        }
                    });
                color_key = ColorKey::Categorical(obs_idx);
            }
        }
        KeyColorKind::Continuous => {
            if continuous_opts.is_empty() {
                ui.label("No continuous obs.");
                color_key = ColorKey::Current;
            } else {
                let mut obs_idx = match color_key {
                    ColorKey::Continuous(idx) => idx,
                    _ => continuous_opts[0].0,
                };
                egui::ComboBox::from_id_salt(("adv_color_cont", card.id))
                    .width(control_width)
                    .selected_text(
                        continuous_opts
                            .iter()
                            .find(|(idx, _)| *idx == obs_idx)
                            .map(|(_, name)| name.as_str())
                            .unwrap_or("?"),
                    )
                    .show_ui(ui, |ui| {
                        for (idx, name) in continuous_opts {
                            ui.selectable_value(&mut obs_idx, *idx, name);
                        }
                    });
                color_key = ColorKey::Continuous(obs_idx);
            }
        }
        KeyColorKind::Gene => {
            if let Some(expr) = ds.meta.expr.as_ref() {
                let mut gene = match &color_key {
                    ColorKey::Gene(name) => name.clone(),
                    _ => String::new(),
                };
                let edit = egui::TextEdit::singleline(&mut gene)
                    .id(egui::Id::new(("adv_gene", card.id)))
                    .hint_text("Exact gene name")
                    .desired_width(control_width);
                ui.add(edit);
                if !gene.trim().is_empty() && !expr.var_names.iter().any(|name| name == &gene) {
                    ui.label("No exact match.");
                }
                color_key = ColorKey::Gene(gene);
            } else {
                ui.label("No gene data.");
                color_key = ColorKey::Current;
            }
        }
    }

    card.space_idx = space_idx;
    card.color_key = color_key;
}
