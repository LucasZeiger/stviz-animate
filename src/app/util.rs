//! Small free-standing helpers shared across the app modules.

use super::{EaseMode, PlaybackMode};
use crate::{
    color::pack_rgba8,
    data::{Dataset, ObsMeta},
};
use eframe::egui;
use std::path::Path;

pub(super) fn save_color_image_png(
    img: &egui::ColorImage,
    path: &Path,
    crop_px: Option<[u32; 4]>,
) -> anyhow::Result<()> {
    use image::ImageEncoder;

    let w = img.size[0] as u32;
    let h = img.size[1] as u32;

    let (x0, y0, cw, ch) = if let Some(crop) = crop_px {
        let mut x = crop[0].min(w);
        let mut y = crop[1].min(h);
        let mut cw = crop[2].min(w.saturating_sub(x));
        let mut ch = crop[3].min(h.saturating_sub(y));
        if cw == 0 || ch == 0 {
            x = 0;
            y = 0;
            cw = w;
            ch = h;
        }
        (x, y, cw, ch)
    } else {
        (0, 0, w, h)
    };

    let mut rgba = Vec::with_capacity((cw * ch * 4) as usize);
    for row in 0..ch {
        let base = (y0 + row) as usize * w as usize + x0 as usize;
        for col in 0..cw as usize {
            let p = img.pixels[base + col];
            rgba.push(p.r());
            rgba.push(p.g());
            rgba.push(p.b());
            rgba.push(p.a());
        }
    }

    std::fs::create_dir_all(path.parent().unwrap_or(Path::new("."))).ok();

    let encoder = image::codecs::png::PngEncoder::new_with_quality(
        std::fs::File::create(path)?,
        image::codecs::png::CompressionType::Best,
        image::codecs::png::FilterType::Adaptive,
    );
    encoder.write_image(&rgba, cw, ch, image::ColorType::Rgba8.into())?;
    Ok(())
}

pub(super) fn save_rgba_png(
    path: &Path,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> anyhow::Result<()> {
    use image::ImageEncoder;

    std::fs::create_dir_all(path.parent().unwrap_or(Path::new("."))).ok();
    let file = std::fs::File::create(path)?;
    let encoder = image::codecs::png::PngEncoder::new_with_quality(
        file,
        image::codecs::png::CompressionType::Best,
        image::codecs::png::FilterType::Adaptive,
    );
    encoder.write_image(rgba, width, height, image::ExtendedColorType::Rgba8)?;
    Ok(())
}

pub(super) fn draw_gradient_legend(ui: &mut egui::Ui, label: &str, vmin: f32, vmax: f32) {
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

pub(super) fn render_filter_rows(
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

pub(super) fn constrain_card_pos(
    pos: egui::Pos2,
    size: egui::Vec2,
    canvas_rect: egui::Rect,
    inspector_rect: egui::Rect,
) -> egui::Pos2 {
    let mut out = pos;
    let max_x = (canvas_rect.width() - size.x).max(0.0);
    let max_y = (canvas_rect.height() - size.y).max(0.0);
    out.x = out.x.clamp(0.0, max_x);
    out.y = out.y.clamp(0.0, max_y);
    let rect = egui::Rect::from_min_size(canvas_rect.min + out.to_vec2(), size);
    if rect.intersects(inspector_rect) {
        let below_y = inspector_rect.bottom() + 12.0;
        if below_y + size.y <= canvas_rect.bottom() {
            out.y = below_y - canvas_rect.min.y;
        } else {
            let right_x = inspector_rect.right() + 12.0;
            if right_x + size.x <= canvas_rect.right() {
                out.x = right_x - canvas_rect.min.x;
            } else {
                out.x = max_x;
                out.y = max_y;
            }
        }
    }
    out
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

pub(super) fn find_obs_by_name(ds: &Dataset, needle: &str) -> Option<usize> {
    let needle = needle.to_ascii_lowercase();
    ds.meta.obs.iter().enumerate().find_map(|(i, o)| match o {
        ObsMeta::Categorical { name, .. } if name.to_ascii_lowercase().contains(&needle) => Some(i),
        ObsMeta::Continuous { name, .. } if name.to_ascii_lowercase().contains(&needle) => Some(i),
        _ => None,
    })
}

pub(super) fn find_space_by_name(ds: &Dataset, needle: &str) -> Option<usize> {
    let needle = needle.to_ascii_lowercase();
    ds.meta.spaces.iter().enumerate().find_map(|(i, s)| {
        if s.name.to_ascii_lowercase().contains(&needle) {
            Some(i)
        } else {
            None
        }
    })
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

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn constrain_card_pos_keeps_card_inside_canvas_and_off_inspector() {
        let canvas = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(400.0, 300.0));
        let far_away = egui::Rect::from_min_size(egui::pos2(1000.0, 1000.0), egui::vec2(1.0, 1.0));
        let size = egui::vec2(100.0, 50.0);

        let clamped = constrain_card_pos(egui::pos2(-20.0, 500.0), size, canvas, far_away);
        assert_eq!(clamped, egui::pos2(0.0, 250.0));

        let inspector = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(150.0, 100.0));
        let moved = constrain_card_pos(egui::pos2(10.0, 10.0), size, canvas, inspector);
        assert_eq!(moved, egui::pos2(10.0, 112.0));
    }
}
