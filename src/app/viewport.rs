//! Central viewport drawing and view overlays.

use super::{
    render_callback::PointCloudCallback,
    util::{apply_ease, contrast_color, export_phase_to_t},
    ColorKey, StvizApp,
};
use crate::render::Uniforms;
use eframe::egui;
use std::sync::Arc;

impl StvizApp {
    pub(super) fn ui_viewport(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());
        self.last_viewport_points = rect;
        ui.painter().rect_filled(rect, 0.0, self.background_color);
        if self.dataset.is_none() {
            // Remember the viewport size so the first dataset is fitted to it.
            let ppp = ctx.pixels_per_point();
            self.last_viewport_px = [rect.width() * ppp, rect.height() * ppp];
            self.ui_welcome(ui, rect);
            return;
        }

        // Interactions
        if response.dragged() {
            let delta = response.drag_delta();
            let ppp = ctx.pixels_per_point();
            self.camera.pan_by_pixels([delta.x * ppp, delta.y * ppp]);
        }

        if response.hovered() {
            let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.0 {
                let zoom_factor = (1.0 + scroll * 0.0015).clamp(0.8, 1.25);
                let mouse = ctx
                    .input(|i| i.pointer.hover_pos())
                    .unwrap_or(rect.center());
                let ppp = ctx.pixels_per_point();
                let local = mouse - rect.min;
                self.camera.zoom_at_viewport_pixel(
                    [local.x * ppp, local.y * ppp],
                    [rect.width() * ppp, rect.height() * ppp],
                    zoom_factor,
                );
            }
        }

        if self.exporting_loop && self.export_total_frames > 0 {
            let phase = self.export_frame_index as f32 / self.export_total_frames as f32;
            self.t = export_phase_to_t(phase, self.playback_mode, 1.0);
        }

        if self.show_stats {
            let dt = ctx.input(|i| i.unstable_dt);
            if dt > 0.0 {
                let ms = dt * 1000.0;
                if self.frame_ms_avg <= 0.0 {
                    self.frame_ms_avg = ms;
                } else {
                    self.frame_ms_avg = self.frame_ms_avg * 0.9 + ms * 0.1;
                }
            }
            let fps = if self.frame_ms_avg > 0.0 {
                1000.0 / self.frame_ms_avg
            } else {
                0.0
            };
            let drawn = self.draw_indices.len();
            let label = format!(
                "{} | {:.1} fps | {:.1} ms | drawn {}",
                self.adapter_label, fps, self.frame_ms_avg, drawn
            );
            ui.painter().text(
                rect.right_top() + egui::vec2(-6.0, 6.0),
                egui::Align2::RIGHT_TOP,
                label,
                egui::FontId::proportional(12.0),
                contrast_color(self.background_color),
            );
        }

        // Update uniforms for this viewport
        let ppp = ctx.pixels_per_point();
        let viewport_px = [rect.width() * ppp, rect.height() * ppp];
        self.last_viewport_px = viewport_px;

        let ds_opt = self.dataset.clone();
        let view_camera = self.camera;
        let (active_from, active_to, color_from, color_to, segment_t, seg_idx) =
            if let Some(ds) = ds_opt.as_ref() {
                self.current_segment(ds)
            } else {
                (
                    0,
                    0,
                    ColorKey::Current,
                    ColorKey::Current,
                    self.t.clamp(0.0, 1.0),
                    0,
                )
            };

        let mut from_override: Option<Arc<Vec<f32>>> = None;
        let mut to_override: Option<Arc<Vec<f32>>> = None;
        let mut from_override_id = 0u64;
        let mut to_override_id = 0u64;

        let (from_center, from_scale, to_center, to_scale) = if let Some(ds) = ds_opt.as_ref() {
            if let Some((pos, _bbox)) = self.grid_positions_for(ds, active_from) {
                from_override = Some(pos);
                if let Some(obs_idx) = self.sample_grid_obs_idx {
                    from_override_id = self.sample_grid_override_id(active_from, obs_idx);
                }
            }
            if let Some((pos, _bbox)) = self.grid_positions_for(ds, active_to) {
                to_override = Some(pos);
                if let Some(obs_idx) = self.sample_grid_obs_idx {
                    to_override_id = self.sample_grid_override_id(active_to, obs_idx);
                }
            }
            let from = if let Some(space) = ds.meta.spaces.get(active_from) {
                self.space_transform(ds, active_from, space)
            } else {
                ([0.0, 0.0], 1.0)
            };
            let to = if let Some(space) = ds.meta.spaces.get(active_to) {
                self.space_transform(ds, active_to, space)
            } else {
                ([0.0, 0.0], 1.0)
            };
            (from.0, from.1, to.0, to.1)
        } else {
            ([0.0, 0.0], 1.0, [0.0, 0.0], 1.0)
        };

        let t_eased = apply_ease(segment_t, self.ease_mode);
        let mut color_t = 0.0f32;
        let mut colors_from = self.colors_rgba8.clone();
        let mut colors_from_id = self.colors_id;
        let mut colors_to = self.colors_rgba8.clone();
        let mut colors_to_id = self.colors_id;
        let mut legend_from = self.legend_range.clone();
        let mut colors_from_opaque = self.colors_opaque;
        let mut colors_to_opaque = self.colors_opaque;

        if let Some(ds) = ds_opt.as_ref() {
            if self.color_path_enabled {
                let (cf, cf_id, cf_legend, cf_opaque) = self.colors_for_key(ds, &color_from);
                let (ct, ct_id, _ct_legend, ct_opaque) = self.colors_for_key(ds, &color_to);
                colors_from = cf;
                colors_from_id = cf_id;
                colors_to = ct;
                colors_to_id = ct_id;
                colors_from_opaque = cf_opaque;
                colors_to_opaque = ct_opaque;
                if color_from != color_to {
                    color_t = t_eased;
                }
                legend_from = cf_legend;
            }

            let (filter_from, filter_to) = self.segment_filters(seg_idx);
            let (cf, cf_id, from_filtered) =
                self.colors_with_filter(ds, colors_from, colors_from_id, filter_from.as_ref());
            let (ct, ct_id, to_filtered) =
                self.colors_with_filter(ds, colors_to, colors_to_id, filter_to.as_ref());
            colors_from = cf;
            colors_from_id = cf_id;
            colors_to = ct;
            colors_to_id = ct_id;
            colors_from_opaque &= !from_filtered;
            colors_to_opaque &= !to_filtered;
            if colors_from_id != colors_to_id {
                color_t = t_eased;
            }
        }

        let mut draw_indices = self.draw_indices.clone();
        let mut indices_id = self.indices_id;
        if let Some(ds) = ds_opt.as_ref() {
            if self.advanced_timeline_open && !self.playing {
                if let Some(preview_idx) = self.advanced_preview_card {
                    if let Some((indices, id)) = self.advanced_draw_override(ds, preview_idx) {
                        draw_indices = indices;
                        indices_id = id;
                    }
                }
            }
            let n_points = ds.meta.n_points;
            if draw_indices.iter().any(|&idx| idx >= n_points) {
                let before = draw_indices.len();
                let filtered: Vec<u32> = draw_indices
                    .iter()
                    .copied()
                    .filter(|&idx| idx < n_points)
                    .collect();
                let dropped = before.saturating_sub(filtered.len());
                let msg = format!("Dropped {dropped} invalid draw indices (n_points={n_points}).");
                self.last_error = Some(msg.clone());
                if self.exporting_loop || self.export_finishing {
                    self.append_export_log(&msg);
                    self.export_status = Some(format!("Export failed: {msg}"));
                    self.exporting_loop = false;
                    self.export_finishing = false;
                    self.export_cancelled = false;
                    self.export_camera = None;
                }
                draw_indices = Arc::new(filtered);
                indices_id = indices_id.wrapping_add(1);
            }
        }

        self.active_legend_range = if self.color_path_enabled {
            legend_from.clone()
        } else {
            self.legend_range.clone()
        };

        let uniforms = Uniforms {
            viewport_px,
            _pad0: [0.0; 2],
            center: view_camera.center,
            _pad1: [0.0; 2],
            pixels_per_unit: view_camera.pixels_per_unit,
            t: t_eased,
            point_radius_px: self.point_radius_px * ppp, // scale with dpi so it looks consistent
            mask_mode: if self.fast_render { 0.0 } else { 1.0 },
            color_t,
            _pad2: 0.0,
            _pad2b: [0.0; 2],
            from_center,
            from_scale,
            _pad3: 0.0,
            to_center,
            to_scale,
            _pad4: 0.0,
        };

        // Push render params into shared state
        {
            let mut p = self.shared.params.lock();
            p.dataset = self.dataset.clone();
            p.dataset_id = self.dataset_id;
            p.from_space = active_from as u32;
            p.to_space = active_to as u32;
            p.colors_id = colors_from_id;
            p.colors_rgba8 = colors_from;
            p.colors_to_id = colors_to_id;
            p.colors_to_rgba8 = colors_to.clone();
            p.indices_id = indices_id;
            p.draw_indices = draw_indices;
            p.uniforms = uniforms;
            p.use_opaque = colors_from_opaque && colors_to_opaque;
            p.from_override = from_override.clone();
            p.to_override = to_override.clone();
            p.from_override_id = from_override_id;
            p.to_override_id = to_override_id;
        }

        // Submit paint callback
        let cb = PointCloudCallback {
            shared: self.shared.clone(),
        };
        let paint_cb = egui_wgpu::Callback::new_paint_callback(rect, cb);
        ui.painter().add(paint_cb);

        self.draw_view_overlays(ui, rect);

        if self.exporting_loop && self.export_total_frames > 0 {
            let path = self
                .export_dir
                .join(format!("frame_{:06}.png", self.export_frame_index));
            if let Some(res) = self.export_resolution {
                if let Err(err) = self.render_export_frame(res, &path, true) {
                    self.append_export_log(&format!("Export render failed: {err}"));
                    self.export_status = Some(format!("Export failed: {err}"));
                    self.exporting_loop = false;
                    self.export_camera = None;
                    if self.direct_video_export.is_some() {
                        self.abort_direct_video_export();
                        self.remove_export_output();
                        let export_dir = self.export_dir.clone();
                        self.remove_export_frames(&export_dir);
                    }
                    return;
                }
            } else {
                let crop = Self::viewport_crop_px_even(rect, ppp);
                self.export_pending_frames = self.export_pending_frames.saturating_add(1);
                self.request_screenshot(ctx, path, crop, true);
            }
            self.export_frame_index += 1;
            if self.export_frame_index >= self.export_total_frames {
                self.exporting_loop = false;
                self.export_camera = None;
                if self.export_resolution.is_some() {
                    self.finish_export_loop();
                } else {
                    self.export_finishing = true;
                    self.export_status = Some("Finalizing export...".to_string());
                }
            }
        }

        // Keep repainting while playing/exporting
        if self.playing || self.exporting_loop || self.export_finishing {
            ctx.request_repaint_after(self.main_view_frame_interval());
        }
    }

    fn draw_view_overlays(&mut self, ui: &egui::Ui, rect: egui::Rect) {
        let painter = ui.painter();
        if self.show_axes {
            let margin = 10.0;
            let origin = rect.left_bottom() + egui::vec2(margin, -margin);
            let axis_len = 36.0;
            let x_color = egui::Color32::from_rgb(220, 90, 90);
            let y_color = egui::Color32::from_rgb(90, 220, 140);

            painter.line_segment([origin, origin + egui::vec2(axis_len, 0.0)], (1.2, x_color));
            painter.line_segment(
                [origin, origin + egui::vec2(0.0, -axis_len)],
                (1.2, y_color),
            );
            painter.text(
                origin + egui::vec2(axis_len + 4.0, 0.0),
                egui::Align2::LEFT_CENTER,
                "X",
                egui::FontId::proportional(12.0),
                x_color,
            );
            painter.text(
                origin + egui::vec2(0.0, -axis_len - 2.0),
                egui::Align2::CENTER_BOTTOM,
                "Y",
                egui::FontId::proportional(12.0),
                y_color,
            );
        }
        if !self.sample_grid_labels_enabled {
            return;
        }
        let Some(ds) = self.dataset.as_ref() else {
            return;
        };
        let (active_from, active_to, _color_from, _color_to, seg_t, _seg_idx) =
            self.current_segment(ds);
        let grid_idx = self.sample_grid_space_idx;
        // Show labels while the grid space is on screen: during a grid-to-grid
        // segment, or at the very start/end of a segment touching it.
        let show_labels = grid_idx.is_some_and(|grid_idx| {
            (active_from == grid_idx && (active_to == grid_idx || seg_t <= 0.02))
                || (active_to == grid_idx && seg_t >= 0.98)
        });
        if !show_labels {
            return;
        }
        let Some(labels) = self.sample_grid_label_positions(ds) else {
            return;
        };
        let ppp = ui.ctx().pixels_per_point();
        let viewport_px = [rect.width() * ppp, rect.height() * ppp];
        if viewport_px[0] <= 0.0 || viewport_px[1] <= 0.0 {
            return;
        }
        let color = contrast_color(self.background_color);
        for (label, pos) in labels {
            let screen_px = [
                (pos[0] - self.camera.center[0]) * self.camera.pixels_per_unit
                    + 0.5 * viewport_px[0],
                (pos[1] - self.camera.center[1]) * self.camera.pixels_per_unit
                    + 0.5 * viewport_px[1],
            ];
            let screen = rect.min + egui::vec2(screen_px[0] / ppp, screen_px[1] / ppp);
            painter.text(
                screen,
                egui::Align2::CENTER_BOTTOM,
                label,
                egui::FontId::proportional(12.0),
                color,
            );
        }
    }

    pub(super) fn viewport_fullscreen_overlay(&mut self, ctx: &egui::Context) {
        egui::Area::new("viewport_fullscreen_exit".into())
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-12.0, 12.0))
            .show(ctx, |ui| {
                let frame = egui::Frame::NONE
                    .fill(egui::Color32::from_black_alpha(140))
                    .corner_radius(6)
                    .inner_margin(egui::Margin::symmetric(8, 6));
                frame.show(ui, |ui| {
                    if ui.button("Exit fullscreen").clicked() {
                        self.viewport_fullscreen = false;
                        ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
                    }
                });
            });
    }

    fn viewport_crop_px(rect: egui::Rect, ppp: f32) -> Option<[u32; 4]> {
        if !rect.is_positive() {
            return None;
        }
        let x0 = (rect.min.x * ppp).floor().max(0.0);
        let y0 = (rect.min.y * ppp).floor().max(0.0);
        let x1 = (rect.max.x * ppp).ceil().max(x0 + 1.0);
        let y1 = (rect.max.y * ppp).ceil().max(y0 + 1.0);
        let w = (x1 - x0).max(1.0) as u32;
        let h = (y1 - y0).max(1.0) as u32;
        Some([x0 as u32, y0 as u32, w, h])
    }

    fn viewport_crop_px_even(rect: egui::Rect, ppp: f32) -> Option<[u32; 4]> {
        let mut crop = Self::viewport_crop_px(rect, ppp)?;
        if crop[2] > 1 && crop[2] % 2 == 1 {
            crop[2] -= 1;
        }
        if crop[3] > 1 && crop[3] % 2 == 1 {
            crop[3] -= 1;
        }
        if crop[2] < 2 || crop[3] < 2 {
            return None;
        }
        Some(crop)
    }
}
