//! Playback ticking, hotkeys and drag-and-drop.

use super::{PlaybackMode, StvizApp};
use eframe::egui;

impl StvizApp {
    pub(super) fn maybe_update_playback(&mut self, ctx: &egui::Context) {
        if self.exporting_loop {
            return;
        }
        if !self.playing {
            return;
        }
        let now = ctx.input(|i| i.time);
        let last = self.play_last_time.unwrap_or(now);
        let dt = (now - last) as f32;
        self.play_last_time = Some(now);
        self.t += self.play_direction * self.speed * dt;
        match self.playback_mode {
            PlaybackMode::Once => {
                if self.t >= 1.0 {
                    self.t = 1.0;
                    self.playing = false;
                } else if self.t <= 0.0 {
                    self.t = 0.0;
                    self.playing = false;
                }
            }
            PlaybackMode::Loop => {
                if self.t > 1.0 {
                    self.t -= 1.0;
                } else if self.t < 0.0 {
                    self.t += 1.0;
                }
            }
            PlaybackMode::PingPong => {
                if self.t > 1.0 {
                    self.t = 1.0;
                    self.play_direction = -self.play_direction;
                } else if self.t < 0.0 {
                    self.t = 0.0;
                    self.play_direction = -self.play_direction;
                }
            }
        }
    }

    pub(super) fn handle_hotkeys(&mut self, ctx: &egui::Context) {
        if ctx.wants_keyboard_input() {
            return;
        }
        self.handle_undo_hotkeys(ctx);
        if ctx.input(|i| i.key_pressed(egui::Key::Questionmark) || i.key_pressed(egui::Key::F1)) {
            self.show_shortcuts = !self.show_shortcuts;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            if self.show_shortcuts {
                self.show_shortcuts = false;
            } else if self.viewport_fullscreen {
                self.viewport_fullscreen = false;
                ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
            } else if self.fullscreen {
                self.fullscreen = false;
                self.apply_fullscreen(ctx);
            }
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
            self.playing = !self.playing;
            self.play_last_time = None;
            if self.playing {
                self.play_last_time = Some(ctx.input(|i| i.time));
            }
        }
        let step = 1.0 / self.export_fps.max(1) as f32;
        let big_step = step * 10.0;
        if ctx.input(|i| i.key_pressed(egui::Key::ArrowRight)) {
            self.t = (self.t + step).clamp(0.0, 1.0);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::ArrowLeft)) {
            self.t = (self.t - step).clamp(0.0, 1.0);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::ArrowUp)) {
            self.t = (self.t + big_step).clamp(0.0, 1.0);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::ArrowDown)) {
            self.t = (self.t - big_step).clamp(0.0, 1.0);
        }
    }

    pub(super) fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped_files = ctx.input(|i| i.raw.dropped_files.clone());
        let Some(path) = dropped_files.into_iter().find_map(|file| {
            let path = file.path?;
            matches!(
                Self::lowercase_extension(&path).as_deref(),
                Some("h5ad" | "stviz")
            )
            .then_some(path)
        }) else {
            if ctx.input(|i| !i.raw.dropped_files.is_empty()) {
                self.convert_status = Some("Drop a .h5ad or .stviz file.".to_string());
            }
            return;
        };
        self.open_data_file(&path);
    }

    fn lowercase_extension(path: &std::path::Path) -> Option<String> {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase())
    }

    /// Convert a .h5ad or load a .stviz, unless the app is busy with one.
    pub(super) fn open_data_file(&mut self, path: &std::path::Path) {
        let busy = if self.convert_running() {
            Some("A conversion is running; wait for it or cancel it first.")
        } else if self.dataset_load.is_some() {
            Some("A dataset is still loading.")
        } else if self.export_busy() {
            Some("Finish or cancel the export before opening another dataset.")
        } else {
            None
        };
        if let Some(reason) = busy {
            self.last_error = Some(reason.to_string());
            return;
        }
        match Self::lowercase_extension(path).as_deref() {
            Some("h5ad") => {
                self.last_error = None;
                self.queue_h5ad_convert(path);
            }
            Some("stviz") => self.start_dataset_load(path),
            _ => self.convert_status = Some("Pick a .h5ad or .stviz file.".to_string()),
        }
    }

    pub(super) fn main_view_frame_interval(&self) -> std::time::Duration {
        let hz = self.main_view_fps_cap_hz.max(30.0);
        std::time::Duration::from_secs_f32(1.0 / hz)
    }
}
