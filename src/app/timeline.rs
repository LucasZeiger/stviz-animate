//! The bottom timeline bar.

use super::*;

impl StvizApp {
    pub(super) fn ui_timeline_bar(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let ds = match self.dataset.clone() {
            Some(ds) => ds,
            None => {
                ui.label("Load a .stviz to use the timeline.");
                return;
            }
        };
        let ds_ref = ds.as_ref();

        let desired_len = self.space_path.len().max(2);
        self.ensure_space_path_len(ds_ref, desired_len);
        self.ensure_color_path_len(self.space_path.len());
        self.ensure_filter_path_len(self.space_path.len());
        if self.key_times.len() != self.space_path.len() {
            self.ensure_key_times_len(self.space_path.len());
        }

        ui.horizontal(|ui| {
            ui.group(|ui| {
                ui.label("Transport");
                ui.horizontal(|ui| {
                    if ui
                        .button(if self.playing { "Pause" } else { "Play" })
                        .on_hover_text("Play or pause the timeline.")
                        .clicked()
                    {
                        self.playing = !self.playing;
                    }
                    if ui
                        .button("Stop")
                        .on_hover_text("Stop and rewind to t = 0.")
                        .clicked()
                    {
                        self.playing = false;
                        self.play_direction = 1.0;
                        self.t = 0.0;
                    }
                    if ui
                        .button("Step -")
                        .on_hover_text("Step back a bit.")
                        .clicked()
                    {
                        self.t = (self.t - 0.01).clamp(0.0, 1.0);
                    }
                    if ui
                        .button("Step +")
                        .on_hover_text("Step forward a bit.")
                        .clicked()
                    {
                        self.t = (self.t + 0.01).clamp(0.0, 1.0);
                    }
                });
            });

            ui.add_space(8.0);

            ui.group(|ui| {
                ui.label("Playback");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.playback_mode, PlaybackMode::Once, "Once");
                    ui.selectable_value(&mut self.playback_mode, PlaybackMode::Loop, "Loop");
                    ui.selectable_value(
                        &mut self.playback_mode,
                        PlaybackMode::PingPong,
                        "Ping-pong",
                    );
                });
                let min_speed = if self.space_path.len() > 5 {
                    0.01
                } else {
                    0.05
                };
                if self.speed < min_speed {
                    self.speed = min_speed;
                }
                ui.add(egui::Slider::new(&mut self.speed, min_speed..=2.0).text("speed"));
            });

            ui.add_space(8.0);

            ui.group(|ui| {
                ui.label("Easing");
                let ease_label = match self.ease_mode {
                    EaseMode::Linear => "Linear",
                    EaseMode::Smoothstep => "Smoothstep",
                    EaseMode::SineInOut => "Sine in-out",
                    EaseMode::QuadInOut => "Quad in-out",
                };
                egui::ComboBox::from_id_salt("timeline_easing")
                    .selected_text(ease_label)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.ease_mode, EaseMode::Linear, "Linear");
                        ui.selectable_value(
                            &mut self.ease_mode,
                            EaseMode::Smoothstep,
                            "Smoothstep",
                        );
                        ui.selectable_value(
                            &mut self.ease_mode,
                            EaseMode::SineInOut,
                            "Sine in-out",
                        );
                        ui.selectable_value(
                            &mut self.ease_mode,
                            EaseMode::QuadInOut,
                            "Quad in-out",
                        );
                    });
            });

            ui.add_space(8.0);

            ui.group(|ui| {
                ui.label("Reset scaling");
                let key_count = self.space_path.len().max(1);
                if self.reset_view_key_idx >= key_count {
                    self.reset_view_key_idx = key_count - 1;
                }
                ui.horizontal(|ui| {
                    let key_label = self
                        .space_path
                        .get(self.reset_view_key_idx)
                        .and_then(|idx| ds_ref.meta.spaces.get(*idx))
                        .map(|s| format!("Key {}: {}", self.reset_view_key_idx + 1, s.name))
                        .unwrap_or_else(|| format!("Key {}", self.reset_view_key_idx + 1));
                    egui::ComboBox::from_id_salt("reset_view_key")
                        .width(160.0)
                        .selected_text(key_label)
                        .show_ui(ui, |ui| {
                            for (i, space_idx) in self.space_path.iter().enumerate() {
                                let label = ds_ref
                                    .meta
                                    .spaces
                                    .get(*space_idx)
                                    .map(|s| format!("Key {}: {}", i + 1, s.name))
                                    .unwrap_or_else(|| format!("Key {}", i + 1));
                                ui.selectable_value(&mut self.reset_view_key_idx, i, label);
                            }
                        });
                    if ui.button("Reset").clicked() {
                        let (active_from, active_to, _color_from, _color_to, segment_t, _seg_idx) =
                            self.current_segment(ds_ref);
                        let mut from_override: Option<Arc<Vec<f32>>> = None;
                        let mut to_override: Option<Arc<Vec<f32>>> = None;
                        if let Some((pos, _bbox)) = self.grid_positions_for(ds_ref, active_from) {
                            from_override = Some(pos);
                        }
                        if let Some((pos, _bbox)) = self.grid_positions_for(ds_ref, active_to) {
                            to_override = Some(pos);
                        }
                        let draw_indices = self.draw_indices.clone();
                        let bbox = self.current_plot_bbox_for_segment(
                            ds_ref,
                            active_from,
                            active_to,
                            segment_t,
                            draw_indices.as_slice(),
                            from_override.as_deref().map(|v| v.as_slice()),
                            to_override.as_deref().map(|v| v.as_slice()),
                        );
                        if let Some(bbox) = bbox {
                            let vp = self.last_viewport_points.size();
                            let ppp = ctx.pixels_per_point();
                            self.camera.fit_bbox(bbox, [vp.x * ppp, vp.y * ppp], 0.9);
                        } else if let Some(space_idx) = self.space_path.get(self.reset_view_key_idx)
                        {
                            if let Some(space) = ds_ref.meta.spaces.get(*space_idx) {
                                let bbox = self.space_bbox_for_view(ds_ref, *space_idx, space);
                                let vp = self.last_viewport_points.size();
                                let ppp = ctx.pixels_per_point();
                                self.camera.fit_bbox(bbox, [vp.x * ppp, vp.y * ppp], 0.9);
                            }
                        }
                    }
                });
            });
        });
        let timeline_height = 36.0;
        let (rect, response) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), timeline_height),
            egui::Sense::click_and_drag(),
        );
        let painter = ui.painter();
        let line_y = rect.center().y;
        painter.line_segment(
            [
                egui::pos2(rect.left(), line_y),
                egui::pos2(rect.right(), line_y),
            ],
            (3.0, egui::Color32::from_gray(100)),
        );

        let key_count = self.space_path.len().max(2);
        let mut marker_handled = false;
        for i in 0..key_count {
            let t = *self.key_times.get(i).unwrap_or(&0.0);
            let x = rect.left() + t * rect.width();
            let pos = egui::pos2(x, line_y);
            let marker_rect = egui::Rect::from_center_size(pos, egui::vec2(14.0, 14.0));
            let marker_id = ui.id().with(("kf_marker", i));
            let marker_resp = ui.interact(marker_rect, marker_id, egui::Sense::click_and_drag());
            if marker_resp.clicked() {
                self.selected_key_idx = Some(i);
                marker_handled = true;
            }
            if marker_resp.dragged() {
                self.selected_key_idx = Some(i);
                marker_handled = true;
                if i > 0 && i + 1 < key_count {
                    if let Some(pos) = marker_resp.interact_pointer_pos() {
                        let mut t = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
                        let min_t = self.key_times[i - 1] + 0.005;
                        let max_t = self.key_times[i + 1] - 0.005;
                        t = t.clamp(min_t, max_t);
                        self.key_times[i] = t;
                    }
                }
            }
            let color = if Some(i) == self.selected_key_idx {
                egui::Color32::from_rgb(255, 210, 90)
            } else {
                egui::Color32::from_gray(170)
            };
            painter.circle_filled(pos, 5.0, color);
        }

        let handle_x = rect.left() + self.t.clamp(0.0, 1.0) * rect.width();
        painter.line_segment(
            [
                egui::pos2(handle_x, rect.top()),
                egui::pos2(handle_x, rect.bottom()),
            ],
            (2.5, egui::Color32::from_rgb(100, 200, 255)),
        );
        if !marker_handled && (response.dragged() || response.clicked()) {
            if let Some(pos) = response.interact_pointer_pos() {
                let t = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
                self.t = t;
            }
        }

        ui.horizontal(|ui| {
            ui.label("t");
            ui.add(
                egui::DragValue::new(&mut self.t)
                    .speed(0.001)
                    .range(0.0..=1.0),
            );
            ui.checkbox(&mut self.color_path_enabled, "Use color keys");
        });

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Keyframes");
            if ui
                .button("Add key")
                .on_hover_text("Insert a keyframe after the current segment.")
                .clicked()
            {
                if self.space_path.len() >= 2 {
                    let space = *self.space_path.last().unwrap_or(&0);
                    let color = self.color_path.last().cloned().unwrap_or(ColorKey::Current);
                    let filter = self.filter_path.last().cloned().unwrap_or(None);
                    self.space_path.push(space);
                    self.color_path.push(color);
                    self.filter_path.push(filter);
                    self.key_times.resize(self.space_path.len(), 0.0);
                    if self.key_collapsed.len() != self.space_path.len() {
                        self.key_collapsed.resize(self.space_path.len(), false);
                    }
                    self.selected_key_idx = Some(self.space_path.len() - 1);
                    self.distribute_key_times_evenly();
                }
            }
            if ui
                .button("Space evenly")
                .on_hover_text("Distribute keyframes evenly across the timeline.")
                .clicked()
            {
                self.distribute_key_times_evenly();
            }
            let can_remove = self.space_path.len() > 2 && self.selected_key_idx.is_some();
            let delete_enabled = can_remove && self.confirm_delete_cards;
            let delete_label = if self.confirm_delete_cards {
                "Delete: enabled"
            } else {
                "Delete: locked"
            };
            ui.checkbox(&mut self.confirm_delete_cards, delete_label);
            let remove_btn = ui
                .add_enabled(delete_enabled, egui::Button::new("Remove key"))
                .on_hover_text(if self.confirm_delete_cards {
                    "Remove the selected keyframe."
                } else {
                    "Enable Delete: yes to remove keyframes."
                });
            if remove_btn.clicked() {
                if let Some(idx) = self.selected_key_idx {
                    self.remove_keyframe(idx);
                }
            }
            if ui
                .button(if self.advanced_timeline_open {
                    "Close advanced timeline"
                } else {
                    "Advanced timeline"
                })
                .clicked()
            {
                self.advanced_timeline_open = !self.advanced_timeline_open;
                if !self.advanced_timeline_open {
                    self.advanced_preview_card = None;
                }
                if self.advanced_timeline_open && self.advanced_cards.is_empty() {
                    self.sync_advanced_from_timeline();
                }
            }
        });

        ui.horizontal(|ui| {
            if ui.button("Collapse all").clicked() {
                self.key_collapsed = vec![true; self.space_path.len()];
            }
            if ui.button("Expand all").clicked() {
                self.key_collapsed = vec![false; self.space_path.len()];
            }
        });

        let categorical_opts: Vec<(usize, String)> = ds_ref
            .meta
            .obs
            .iter()
            .enumerate()
            .filter_map(|(i, o)| match o {
                ObsMeta::Categorical { name, .. } => Some((i, name.clone())),
                _ => None,
            })
            .collect();
        let filterable_opts: Vec<(usize, String)> = ds_ref
            .meta
            .obs
            .iter()
            .enumerate()
            .filter_map(|(i, o)| match o {
                ObsMeta::Categorical {
                    name, categories, ..
                } if categories.len() <= MAX_FILTER_CATEGORIES => Some((i, name.clone())),
                _ => None,
            })
            .collect();
        let continuous_opts: Vec<(usize, String)> = ds_ref
            .meta
            .obs
            .iter()
            .enumerate()
            .filter_map(|(i, o)| match o {
                ObsMeta::Continuous { name, .. } => Some((i, name.clone())),
                _ => None,
            })
            .collect();
        let key_count = self.space_path.len();
        let mut drag_stop_idx: Option<usize> = None;
        let mut card_rects: Vec<(usize, egui::Rect)> = Vec::new();
        let mut remove_idx: Option<usize> = None;
        let pointer_pos = ctx.input(|i| i.pointer.latest_pos());
        let mut drag_target: Option<(usize, bool)> = None;
        egui::ScrollArea::horizontal()
            .max_height(260.0)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let max_idx = ds_ref.meta.spaces.len().saturating_sub(1);
                if self.key_collapsed.len() != key_count {
                    self.key_collapsed.resize(key_count, false);
                }
                ui.horizontal(|ui| {
                    for i in 0..key_count {
                        let mut space_idx = self.space_path[i].min(max_idx);
                        let mut color_key =
                            self.color_path.get(i).cloned().unwrap_or(ColorKey::Current);
                        let mut key_filter = self.filter_path.get(i).cloned().unwrap_or(None);
                        let mut color_kind = match color_key {
                            ColorKey::Current => KeyColorKind::Current,
                            ColorKey::Categorical(_) => KeyColorKind::Categorical,
                            ColorKey::Continuous(_) => KeyColorKind::Continuous,
                            ColorKey::Gene(_) => KeyColorKind::Gene,
                        };
                        let is_selected = Some(i) == self.selected_key_idx;
                        let mut collapsed = self.key_collapsed[i];
                        let mut frame = egui::Frame::group(ui.style())
                            .corner_radius(6)
                            .inner_margin(egui::Margin::same(2));
                        if is_selected {
                            frame = frame.stroke(egui::Stroke::new(
                                1.4,
                                egui::Color32::from_rgb(80, 180, 255),
                            ));
                        }
                        let inner = ui.scope_builder(
                            egui::UiBuilder::new().layout(egui::Layout::top_down(egui::Align::Min)),
                            |ui| {
                                frame.show(ui, |ui| {
                                    let prev_spacing = ui.spacing().item_spacing;
                                    ui.spacing_mut().item_spacing = egui::vec2(prev_spacing.x, 2.0);
                                    ui.spacing_mut().interact_size.y =
                                        ui.spacing().interact_size.y.min(20.0);
                                    let mut drag_handle_resp = None;
                                    ui.allocate_ui_with_layout(
                                        egui::vec2(0.0, 0.0),
                                        egui::Layout::left_to_right(egui::Align::Center),
                                        |ui| {
                                            let drag_handle = ui
                                                .add(
                                                    egui::Button::new(
                                                        egui::RichText::new("⋮⋮")
                                                            .size(13.0)
                                                            .strong()
                                                            .color(ui.visuals().weak_text_color()),
                                                    )
                                                    .frame(false)
                                                    .small()
                                                    .sense(egui::Sense::drag()),
                                                )
                                                .on_hover_text("Drag to reorder keyframes.")
                                                .on_hover_and_drag_cursor(egui::CursorIcon::Grab);
                                            drag_handle_resp = Some(drag_handle);
                                            let header_resp = ui
                                                .selectable_label(
                                                    is_selected,
                                                    egui::RichText::new(format!("Key {}", i + 1))
                                                        .strong(),
                                                )
                                                .on_hover_text("Click to select this keyframe.");
                                            if header_resp.clicked() {
                                                self.selected_key_idx = Some(i);
                                            }
                                            let can_remove = self.space_path.len() > 2;
                                            let delete_enabled =
                                                can_remove && self.confirm_delete_cards;
                                            let delete_btn = ui
                                                .add_enabled(
                                                    delete_enabled,
                                                    egui::Button::new("🗑").small(),
                                                )
                                                .on_hover_text(if self.confirm_delete_cards {
                                                    "Remove this keyframe."
                                                } else {
                                                    "Enable Delete: yes to remove keyframes."
                                                });
                                            if delete_btn.clicked() {
                                                remove_idx = Some(i);
                                            }
                                            if ui
                                                .small_button(if collapsed { "▸" } else { "▾" })
                                                .on_hover_text("Collapse/expand this keyframe.")
                                                .clicked()
                                            {
                                                collapsed = !collapsed;
                                            }
                                        },
                                    );

                                    if let Some(drag_handle) = drag_handle_resp {
                                        if drag_handle.drag_started() || drag_handle.dragged() {
                                            self.dragging_key_idx = Some(i);
                                            ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                                        }
                                        if drag_handle.drag_stopped() {
                                            drag_stop_idx = Some(i);
                                        }
                                    }

                                    if collapsed {
                                        let space_name = ds_ref
                                            .meta
                                            .spaces
                                            .get(space_idx)
                                            .map(|s| s.name.as_str())
                                            .unwrap_or("?");
                                        let color_desc = match &color_key {
                                            ColorKey::Current => "Current".to_string(),
                                            ColorKey::Categorical(idx) => {
                                                format!("Cat: {}", obs_name(ds_ref, *idx))
                                            }
                                            ColorKey::Continuous(idx) => {
                                                format!("Cont: {}", obs_name(ds_ref, *idx))
                                            }
                                            ColorKey::Gene(name) => format!("Gene: {name}"),
                                        };
                                        ui.label(
                                            egui::RichText::new(format!(
                                                "{space_name} | {color_desc}"
                                            ))
                                            .size(10.5),
                                        );
                                        ui.spacing_mut().item_spacing = prev_spacing;
                                        return;
                                    }

                                    let control_width = 110.0;
                                    ui.allocate_ui_with_layout(
                                        egui::vec2(0.0, 0.0),
                                        egui::Layout::left_to_right(egui::Align::Center),
                                        |ui| {
                                            ui.label(egui::RichText::new("Space").size(10.5));
                                            egui::ComboBox::from_id_salt(("kf_space", i))
                                                .width(control_width)
                                                .selected_text(
                                                    ds_ref
                                                        .meta
                                                        .spaces
                                                        .get(space_idx)
                                                        .map(|s| s.name.as_str())
                                                        .unwrap_or("?"),
                                                )
                                                .show_ui(ui, |ui| {
                                                    for (j, s) in
                                                        ds_ref.meta.spaces.iter().enumerate()
                                                    {
                                                        ui.selectable_value(
                                                            &mut space_idx,
                                                            j,
                                                            &s.name,
                                                        );
                                                    }
                                                });
                                        },
                                    );
                                    ui.allocate_ui_with_layout(
                                        egui::vec2(0.0, 0.0),
                                        egui::Layout::left_to_right(egui::Align::Center),
                                        |ui| {
                                            ui.label(egui::RichText::new("Color").size(10.5));
                                            let kind_label = match color_kind {
                                                KeyColorKind::Current => "Current",
                                                KeyColorKind::Categorical => "Categorical",
                                                KeyColorKind::Continuous => "Continuous",
                                                KeyColorKind::Gene => "Gene",
                                            };
                                            egui::ComboBox::from_id_salt(("kf_color_kind", i))
                                                .width(control_width)
                                                .selected_text(kind_label)
                                                .show_ui(ui, |ui| {
                                                    ui.selectable_value(
                                                        &mut color_kind,
                                                        KeyColorKind::Current,
                                                        "Current",
                                                    );
                                                    ui.selectable_value(
                                                        &mut color_kind,
                                                        KeyColorKind::Categorical,
                                                        "Categorical",
                                                    );
                                                    ui.selectable_value(
                                                        &mut color_kind,
                                                        KeyColorKind::Continuous,
                                                        "Continuous",
                                                    );
                                                    ui.selectable_value(
                                                        &mut color_kind,
                                                        KeyColorKind::Gene,
                                                        "Gene",
                                                    );
                                                });
                                        },
                                    );

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
                                                egui::ComboBox::from_id_salt(("kf_color_cat", i))
                                                    .width(control_width)
                                                    .selected_text(
                                                        categorical_opts
                                                            .iter()
                                                            .find(|(idx, _)| *idx == obs_idx)
                                                            .map(|(_, name)| name.as_str())
                                                            .unwrap_or("?"),
                                                    )
                                                    .show_ui(ui, |ui| {
                                                        for (idx, name) in &categorical_opts {
                                                            ui.selectable_value(
                                                                &mut obs_idx,
                                                                *idx,
                                                                name,
                                                            );
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
                                                egui::ComboBox::from_id_salt(("kf_color_cont", i))
                                                    .width(control_width)
                                                    .selected_text(
                                                        continuous_opts
                                                            .iter()
                                                            .find(|(idx, _)| *idx == obs_idx)
                                                            .map(|(_, name)| name.as_str())
                                                            .unwrap_or("?"),
                                                    )
                                                    .show_ui(ui, |ui| {
                                                        for (idx, name) in &continuous_opts {
                                                            ui.selectable_value(
                                                                &mut obs_idx,
                                                                *idx,
                                                                name,
                                                            );
                                                        }
                                                    });
                                                color_key = ColorKey::Continuous(obs_idx);
                                            }
                                        }
                                        KeyColorKind::Gene => {
                                            if let Some(expr) = ds_ref.meta.expr.as_ref() {
                                                let mut gene = match &color_key {
                                                    ColorKey::Gene(name) => name.clone(),
                                                    _ => String::new(),
                                                };
                                                let edit = egui::TextEdit::singleline(&mut gene)
                                                    .id(egui::Id::new(("kf_gene", i)))
                                                    .hint_text("Exact gene name")
                                                    .desired_width(control_width);
                                                ui.add(edit);
                                                if !gene.trim().is_empty()
                                                    && !expr
                                                        .var_names
                                                        .iter()
                                                        .any(|name| name == &gene)
                                                {
                                                    ui.label("No exact match.");
                                                }
                                                color_key = ColorKey::Gene(gene);
                                            } else {
                                                ui.label("No gene data.");
                                                color_key = ColorKey::Current;
                                            }
                                        }
                                    }

                                    if key_filter.as_ref().is_some_and(|filter| {
                                        !filterable_opts
                                            .iter()
                                            .any(|(idx, _)| *idx == filter.obs_idx)
                                    }) {
                                        key_filter = None;
                                    }
                                    let mut filter_enabled = key_filter.is_some();
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
                                            if let Some((obs_idx, _)) = filterable_opts.first() {
                                                if let Ok((_name, _labels, categories, _pal)) =
                                                    ds_ref.obs_categorical(*obs_idx)
                                                {
                                                    key_filter = Some(KeyframeFilter {
                                                        obs_idx: *obs_idx,
                                                        enabled: vec![true; categories.len()],
                                                    });
                                                }
                                            }
                                        } else {
                                            key_filter = None;
                                        }
                                    }
                                    if let Some(filter) = key_filter.as_mut() {
                                        let mut obs_idx = filter.obs_idx;
                                        egui::ComboBox::from_id_salt(("kf_filter_obs", i))
                                            .width(control_width)
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
                                            if let Ok((_name, _labels, categories, _pal)) =
                                                ds_ref.obs_categorical(obs_idx)
                                            {
                                                filter.enabled = vec![true; categories.len()];
                                            }
                                        }
                                        if let Ok((_name, _labels, categories, _pal)) =
                                            ds_ref.obs_categorical(filter.obs_idx)
                                        {
                                            if filter.enabled.len() != categories.len() {
                                                filter.enabled = vec![true; categories.len()];
                                            }
                                            let selected =
                                                filter.enabled.iter().filter(|v| **v).count();
                                            egui::containers::menu::MenuButton::new(format!(
                                                "Categories {selected}/{}",
                                                categories.len()
                                            ))
                                            .config(
                                                egui::containers::menu::MenuConfig::new()
                                                    .close_behavior(
                                                    egui::PopupCloseBehavior::CloseOnClickOutside,
                                                ),
                                            )
                                            .ui(
                                                ui,
                                                |ui| {
                                                    ui.horizontal(|ui| {
                                                        if ui.button("All").clicked() {
                                                            filter.enabled.fill(true);
                                                        }
                                                        if ui.button("None").clicked() {
                                                            filter.enabled.fill(false);
                                                        }
                                                    });
                                                    egui::ScrollArea::vertical()
                                                        .max_height(240.0)
                                                        .show(ui, |ui| {
                                                            for (idx, name) in
                                                                categories.iter().enumerate()
                                                            {
                                                                ui.checkbox(
                                                                    &mut filter.enabled[idx],
                                                                    name,
                                                                );
                                                            }
                                                        });
                                                },
                                            );
                                        }
                                    }
                                    ui.spacing_mut().item_spacing = prev_spacing;
                                })
                            },
                        );
                        let rect = inner.inner.response.rect;
                        card_rects.push((i, rect));
                        self.space_path[i] = space_idx;
                        if i < self.color_path.len() {
                            self.color_path[i] = color_key;
                        } else {
                            self.color_path.push(color_key);
                        }
                        self.filter_path[i] = key_filter;
                        self.key_collapsed[i] = collapsed;
                        ui.add_space(6.0);
                    }
                });
            });
        ui.add_space(8.0);

        if let (Some(drag_idx), Some(pos)) = (self.dragging_key_idx, pointer_pos) {
            if let Some((target_idx, target_rect, insert_after)) = card_rects
                .iter()
                .filter_map(|(idx, rect)| {
                    let center = rect.center();
                    let dist = (center.x - pos.x).powi(2) + (center.y - pos.y).powi(2);
                    Some((*idx, *rect, dist))
                })
                .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, rect, _)| (idx, rect, pos.x > rect.center().x))
            {
                drag_target = Some((target_idx, insert_after));
                let line_x = if insert_after {
                    target_rect.right()
                } else {
                    target_rect.left()
                };
                let line_color = egui::Color32::from_rgb(120, 200, 255);
                ui.painter().line_segment(
                    [
                        egui::pos2(line_x, target_rect.top()),
                        egui::pos2(line_x, target_rect.bottom()),
                    ],
                    egui::Stroke::new(2.0, line_color),
                );
            }
            if let Some((_, rect)) = card_rects.iter().find(|(idx, _)| *idx == drag_idx) {
                let size = rect.size();
                let offset = egui::vec2(12.0, 12.0);
                let float_rect = egui::Rect::from_min_size(pos + offset, size);
                let shadow_rect = float_rect.translate(egui::vec2(3.0, 3.0));
                let shadow_color = egui::Color32::from_rgba_unmultiplied(0, 0, 0, 90);
                let card_fill = ui.visuals().widgets.inactive.bg_fill;
                let card_stroke = ui.visuals().widgets.active.bg_stroke;
                let text_color = ui.visuals().text_color();
                ui.painter().rect_filled(shadow_rect, 6.0, shadow_color);
                ui.painter().rect_filled(float_rect, 6.0, card_fill);
                ui.painter()
                    .rect_stroke(float_rect, 6.0, card_stroke, egui::StrokeKind::Inside);
                ui.painter().text(
                    float_rect.min + egui::vec2(8.0, 8.0),
                    egui::Align2::LEFT_TOP,
                    format!("Key {}", drag_idx + 1),
                    egui::FontId::proportional(12.0),
                    text_color,
                );
            }
        }
        if drag_stop_idx.is_none()
            && self.dragging_key_idx.is_some()
            && !ctx.input(|i| i.pointer.any_down())
        {
            drag_stop_idx = self.dragging_key_idx;
        }

        if let (Some(drag_idx), Some(stop_idx)) = (self.dragging_key_idx, drag_stop_idx) {
            if drag_idx == stop_idx {
                if let Some((target_idx, insert_after)) = drag_target {
                    let mut insert_idx = if insert_after {
                        target_idx.saturating_add(1)
                    } else {
                        target_idx
                    };
                    if insert_idx > drag_idx {
                        insert_idx = insert_idx.saturating_sub(1);
                    }
                    if insert_idx != drag_idx {
                        self.move_keyframe(drag_idx, insert_idx);
                    }
                }
            }
            self.dragging_key_idx = None;
        }
        if let Some(idx) = remove_idx {
            self.remove_keyframe(idx);
        }

        self.ensure_color_path_len(self.space_path.len());
        self.ensure_filter_path_len(self.space_path.len());
        self.from_space = *self.space_path.first().unwrap_or(&self.from_space);
        self.to_space = *self.space_path.last().unwrap_or(&self.to_space);

        if self.playing {
            ctx.request_repaint_after(self.main_view_frame_interval());
        }
    }
}
