//! Preferences that persist between launches (`settings.json` in the app data
//! directory): appearance, a few defaults, the last used folder and recent files.

use super::{StvizApp, UiTheme};
use eframe::egui;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const MAX_RECENT_FILES: usize = 8;
/// Seconds between checks for changed settings.
const SAVE_INTERVAL_SEC: f64 = 2.0;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(super) struct AppSettings {
    ui_scale: f32,
    ui_theme: UiTheme,
    background_rgba: [u8; 4],
    show_axes: bool,
    show_stats: bool,
    clip_outliers: bool,
    convert_include_expr: bool,
    ffmpeg_auto_install: bool,
    export_fps: u32,
    export_duration_sec: f32,
    last_dir: Option<PathBuf>,
    recent_files: Vec<PathBuf>,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            ui_scale: 1.0,
            ui_theme: UiTheme::Dark,
            background_rgba: [0, 0, 0, 255],
            show_axes: true,
            show_stats: true,
            clip_outliers: false,
            convert_include_expr: true,
            ffmpeg_auto_install: false,
            export_fps: 30,
            export_duration_sec: 5.0,
            last_dir: None,
            recent_files: Vec::new(),
        }
    }
}

impl AppSettings {
    /// Read settings, falling back to defaults if the file is missing or unreadable.
    pub(super) fn load(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|json| serde_json::from_str(&json).ok())
            .unwrap_or_default()
    }

    fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        // Write then rename so a crash can't leave a half-written file.
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, path)
    }
}

impl StvizApp {
    pub(super) fn apply_settings(&mut self, s: &AppSettings) {
        self.ui_scale = s.ui_scale.clamp(0.5, 3.0);
        self.ui_theme = s.ui_theme;
        let [r, g, b, a] = s.background_rgba;
        self.background_color = egui::Color32::from_rgba_unmultiplied(r, g, b, a);
        self.show_axes = s.show_axes;
        self.show_stats = s.show_stats;
        self.clip_outliers = s.clip_outliers;
        self.convert_include_expr = s.convert_include_expr;
        self.ffmpeg_auto_install = s.ffmpeg_auto_install;
        self.export_fps = s.export_fps.clamp(1, 240);
        if s.export_duration_sec.is_finite() && s.export_duration_sec > 0.0 {
            self.export_duration_sec = s.export_duration_sec;
        }
        self.last_dir = s.last_dir.clone();
        self.recent_files = s.recent_files.clone();
        self.recent_files.truncate(MAX_RECENT_FILES);
    }

    fn current_settings(&self) -> AppSettings {
        let bg = self.background_color;
        AppSettings {
            ui_scale: self.ui_scale,
            ui_theme: self.ui_theme,
            background_rgba: [bg.r(), bg.g(), bg.b(), bg.a()],
            show_axes: self.show_axes,
            show_stats: self.show_stats,
            clip_outliers: self.clip_outliers,
            convert_include_expr: self.convert_include_expr,
            ffmpeg_auto_install: self.ffmpeg_auto_install,
            export_fps: self.export_fps,
            export_duration_sec: self.export_duration_sec,
            last_dir: self.last_dir.clone(),
            recent_files: self.recent_files.clone(),
        }
    }

    /// Save settings if they changed. With `now` (the frame time) writes are
    /// spaced out, and the return value is how long to wait before calling
    /// again if a change is still unsaved. `None` saves immediately.
    pub(super) fn save_settings_if_changed(&mut self, now: Option<f64>) -> Option<f64> {
        let current = self.current_settings();
        if current == self.saved_settings {
            return None;
        }
        if let Some(now) = now {
            let wait = SAVE_INTERVAL_SEC - (now - self.settings_checked_at);
            if wait > 0.0 {
                return Some(wait);
            }
            self.settings_checked_at = now;
        }
        if current.save(&self.settings_path).is_ok() {
            self.saved_settings = current;
        }
        None
    }

    /// Remember the folder of a file the user picked, for the next file dialog.
    pub(super) fn remember_dir_of(&mut self, path: &Path) {
        if let Some(parent) = path.parent().filter(|p| p.is_dir()) {
            self.last_dir = Some(parent.to_path_buf());
        }
    }

    /// Put a dataset at the top of the recent files list.
    pub(super) fn remember_recent_file(&mut self, path: &Path) {
        push_recent(&mut self.recent_files, path);
    }

    /// Folder a file dialog should start in: the last one used, else home.
    pub(super) fn dialog_dir(&self) -> PathBuf {
        if let Some(dir) = self.last_dir.as_ref().filter(|d| d.is_dir()) {
            return dir.clone();
        }
        std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .map(PathBuf::from)
            .filter(|d| d.is_dir())
            .unwrap_or_else(|| self.output_dir.clone())
    }
}

fn push_recent(recent: &mut Vec<PathBuf>, path: &Path) {
    recent.retain(|p| p != path);
    recent.insert(0, path.to_path_buf());
    recent.truncate(MAX_RECENT_FILES);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_round_trip_and_tolerate_bad_files() {
        let dir = std::env::temp_dir().join(format!("stviz_settings_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");
        assert_eq!(AppSettings::load(&path), AppSettings::default());

        let settings = AppSettings {
            ui_scale: 1.5,
            ui_theme: UiTheme::Light,
            recent_files: vec![PathBuf::from("/data/a.stviz")],
            ..AppSettings::default()
        };
        settings.save(&path).unwrap();
        assert_eq!(AppSettings::load(&path), settings);

        // Unknown or missing keys are fine; garbage falls back to defaults.
        std::fs::write(&path, r#"{"ui_scale": 2.0, "future_key": 1}"#).unwrap();
        assert_eq!(AppSettings::load(&path).ui_scale, 2.0);
        std::fs::write(&path, "not json").unwrap();
        assert_eq!(AppSettings::load(&path), AppSettings::default());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn recent_files_are_deduplicated_and_capped() {
        let mut recent = Vec::new();
        for i in 0..12 {
            push_recent(&mut recent, Path::new(&format!("/d/{i}.stviz")));
        }
        push_recent(&mut recent, Path::new("/d/8.stviz"));
        assert_eq!(recent.len(), MAX_RECENT_FILES);
        assert_eq!(recent[0], PathBuf::from("/d/8.stviz"));
        assert_eq!(recent.iter().filter(|p| p.ends_with("8.stviz")).count(), 1);
    }
}
