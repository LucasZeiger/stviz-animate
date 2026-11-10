//! Small free-standing helpers shared across the app modules.

use super::{EaseMode, PlaybackMode, MAX_FILTER_CATEGORIES};
use crate::data::{Dataset, ObsMeta};
use eframe::egui;

pub(super) fn apply_ease(t: f32, mode: EaseMode) -> f32 {
    let x = t.clamp(0.0, 1.0);
    match mode {
        EaseMode::Linear => x,
        EaseMode::Smoothstep => x * x * (3.0 - 2.0 * x),
        EaseMode::SineInOut => 0.5 - 0.5 * (std::f32::consts::PI * x).cos(),
        EaseMode::QuadInOut => {
            if x < 0.5 {
                2.0 * x * x
            } else {
                1.0 - (-2.0 * x + 2.0).powi(2) / 2.0
            }
        }
    }
}

pub(super) fn export_phase_to_t(phase: f32, mode: PlaybackMode, direction: f32) -> f32 {
    let p = phase.fract();
    let t = match mode {
        PlaybackMode::PingPong => {
            if p < 0.5 {
                p * 2.0
            } else {
                2.0 - 2.0 * p
            }
        }
        _ => p,
    };
    if direction < 0.0 {
        1.0 - t
    } else {
        t
    }
}

pub(super) fn color32_from_packed(c: u32) -> egui::Color32 {
    let r = (c & 255) as u8;
    let g = ((c >> 8) & 255) as u8;
    let b = ((c >> 16) & 255) as u8;
    let a = ((c >> 24) & 255) as u8;
    egui::Color32::from_rgba_unmultiplied(r, g, b, a)
}

pub(super) fn contrast_color(bg: egui::Color32) -> egui::Color32 {
    let r = bg.r() as f32 / 255.0;
    let g = bg.g() as f32 / 255.0;
    let b = bg.b() as f32 / 255.0;
    let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    if lum < 0.5 {
        egui::Color32::WHITE
    } else {
        egui::Color32::BLACK
    }
}

pub(super) fn format_scale_value(v: f32) -> String {
    let av = v.abs();
    if av > 0.0 && !(0.01..1000.0).contains(&av) {
        format!("{v:.2e}")
    } else {
        format!("{v:.3}")
    }
}

pub(super) fn obs_name(ds: &Dataset, idx: usize) -> String {
    match ds.meta.obs.get(idx) {
        Some(ObsMeta::Categorical { name, .. }) => format!("{name} (cat)"),
        Some(ObsMeta::Continuous { name, .. }) => format!("{name} (cont)"),
        None => format!("obs[{idx}]"),
    }
}

pub(super) fn chrono_like_timestamp() -> String {
    // Avoid adding chrono dependency; generate a simple timestamp-like string.
    use std::time::{SystemTime, UNIX_EPOCH};
    let ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("{ms}")
}

/// Observation columns that can be offered in a picker, as `(obs index, name)`.
#[derive(Default)]
pub(super) struct ObsOptions {
    pub(super) categorical: Vec<(usize, String)>,
    /// Categorical columns small enough to filter on (`<= MAX_FILTER_CATEGORIES`).
    pub(super) filterable: Vec<(usize, String)>,
    pub(super) continuous: Vec<(usize, String)>,
}

impl ObsOptions {
    pub(super) fn for_dataset(ds: &Dataset) -> Self {
        let mut out = Self::default();
        for (i, o) in ds.meta.obs.iter().enumerate() {
            match o {
                ObsMeta::Categorical {
                    name, categories, ..
                } => {
                    out.categorical.push((i, name.clone()));
                    if categories.len() <= MAX_FILTER_CATEGORIES {
                        out.filterable.push((i, name.clone()));
                    }
                }
                ObsMeta::Continuous { name, .. } => out.continuous.push((i, name.clone())),
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::pack_rgba8;

    const EASES: [EaseMode; 4] = [
        EaseMode::Linear,
        EaseMode::Smoothstep,
        EaseMode::SineInOut,
        EaseMode::QuadInOut,
    ];

    #[test]
    fn apply_ease_hits_endpoints_and_midpoint() {
        for mode in EASES {
            assert!(apply_ease(0.0, mode).abs() < 1e-6, "{mode:?} at 0");
            assert!((apply_ease(1.0, mode) - 1.0).abs() < 1e-6, "{mode:?} at 1");
            assert!(
                (apply_ease(0.5, mode) - 0.5).abs() < 1e-6,
                "{mode:?} at 0.5"
            );
        }
    }

    #[test]
    fn apply_ease_clamps_input() {
        for mode in EASES {
            assert_eq!(apply_ease(-1.0, mode), apply_ease(0.0, mode));
            assert_eq!(apply_ease(2.0, mode), apply_ease(1.0, mode));
        }
    }

    #[test]
    fn export_phase_to_t_handles_modes_and_direction() {
        assert_eq!(export_phase_to_t(0.25, PlaybackMode::Loop, 1.0), 0.25);
        assert_eq!(export_phase_to_t(1.25, PlaybackMode::Loop, 1.0), 0.25);
        assert_eq!(export_phase_to_t(0.25, PlaybackMode::Loop, -1.0), 0.75);
        assert_eq!(export_phase_to_t(0.25, PlaybackMode::PingPong, 1.0), 0.5);
        assert_eq!(export_phase_to_t(0.75, PlaybackMode::PingPong, 1.0), 0.5);
        assert_eq!(export_phase_to_t(0.5, PlaybackMode::PingPong, 1.0), 1.0);
    }

    #[test]
    fn color32_from_packed_round_trips_pack_rgba8() {
        let c = color32_from_packed(pack_rgba8(10, 20, 30, 255));
        assert_eq!((c.r(), c.g(), c.b(), c.a()), (10, 20, 30, 255));
    }

    #[test]
    fn contrast_color_picks_readable_text() {
        assert_eq!(contrast_color(egui::Color32::BLACK), egui::Color32::WHITE);
        assert_eq!(contrast_color(egui::Color32::WHITE), egui::Color32::BLACK);
    }

    #[test]
    fn format_scale_value_switches_to_scientific_outside_range() {
        assert_eq!(format_scale_value(0.0), "0.000");
        assert_eq!(format_scale_value(1.5), "1.500");
        assert_eq!(format_scale_value(12345.0), "1.23e4");
        assert_eq!(format_scale_value(0.001), "1.00e-3");
    }
}
