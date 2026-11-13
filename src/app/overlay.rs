//! Labels drawn on top of the points: a title, the current space name, a
//! legend for the active colouring and a scale bar.
//!
//! The same layout is used twice. The viewport draws it with the egui painter
//! as a live preview, and exports rasterize it on the CPU straight into each
//! rendered frame, so videos and screenshots carry the labels too.

use super::{ColorKey, ColorMode, StvizApp, MAX_FILTER_CATEGORIES};
use crate::data::{Dataset, ObsMeta};
use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use eframe::egui;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// User settings for the overlay. Saved in workspaces.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(super) struct OverlaySettings {
    /// Draw the overlay in the viewport and in exports.
    pub(super) enabled: bool,
    pub(super) title: String,
    pub(super) show_space_name: bool,
    pub(super) show_legend: bool,
    pub(super) show_scale_bar: bool,
    /// Physical size of one coordinate unit, e.g. 1.0 when spatial
    /// coordinates are already in microns.
    pub(super) units_per_coord: f32,
    pub(super) unit_label: String,
    /// Multiplies every text and marker size.
    pub(super) text_scale: f32,
}

impl Default for OverlaySettings {
    fn default() -> Self {
        Self {
            enabled: true,
            title: String::new(),
            show_space_name: false,
            show_legend: true,
            show_scale_bar: true,
            units_per_coord: 1.0,
            unit_label: "µm".to_string(),
            text_scale: 1.0,
        }
    }
}

/// Legend for the colouring shown at the current time.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum OverlayLegend {
    Categorical {
        title: String,
        entries: Vec<(String, [u8; 4])>,
    },
    Continuous {
        title: String,
        min: f32,
        max: f32,
    },
}

/// Everything the overlay shows for one frame, independent of output size.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct OverlayContent {
    pub(super) title: Option<String>,
    pub(super) space_name: Option<String>,
    pub(super) legend: Option<OverlayLegend>,
    /// Output pixels per coordinate unit, when a scale bar applies.
    pub(super) scale_px_per_coord: Option<f32>,
    pub(super) units_per_coord: f32,
    pub(super) unit_label: String,
    pub(super) text_scale: f32,
    pub(super) text_color: [u8; 4],
    pub(super) background: [u8; 4],
}

/// A drawing primitive in output pixels.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Prim {
    Rect {
        rect: [f32; 4],
        color: [u8; 4],
    },
    Circle {
        center: [f32; 2],
        radius: f32,
        color: [u8; 4],
    },
    /// Horizontal viridis gradient.
    Gradient {
        rect: [f32; 4],
    },
    /// Text whose top-left corner is at `pos`.
    Text {
        pos: [f32; 2],
        size: f32,
        text: String,
        color: [u8; 4],
    },
}

/// Most categories listed before the legend ends with "+N more".
const MAX_LEGEND_ENTRIES: usize = 36;

fn font() -> &'static FontRef<'static> {
    static FONT: OnceLock<FontRef<'static>> = OnceLock::new();
    FONT.get_or_init(|| {
        FontRef::try_from_slice(epaint_default_fonts::UBUNTU_LIGHT)
            .expect("bundled Ubuntu-Light font is valid")
    })
}

/// Width in pixels of `text` at `size` pixels.
pub(super) fn text_width(text: &str, size: f32) -> f32 {
    let font = font().as_scaled(PxScale::from(size));
    let mut width = 0.0;
    let mut prev = None;
    for ch in text.chars() {
        let id = font.glyph_id(ch);
        if let Some(prev) = prev {
            width += font.kern(prev, id);
        }
        width += font.h_advance(id);
        prev = Some(id);
    }
    width
}

/// Short number label: no trailing zeros, scientific outside 0.01..10000.
pub(super) fn format_number(v: f32) -> String {
    let av = v.abs();
    if v == 0.0 {
        return "0".to_string();
    }
    if !(0.01..10000.0).contains(&av) {
        return format!("{v:.1e}");
    }
    let decimals = if av >= 100.0 {
        0
    } else if av >= 1.0 {
        2
    } else {
        3
    };
    let s = format!("{v:.decimals$}");
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

/// Largest 1, 2 or 5 x 10^k that is at most `max`.
pub(super) fn nice_length(max: f32) -> f32 {
    if !(max.is_finite() && max > 0.0) {
        return 0.0;
    }
    let exp = max.log10().floor();
    let base = 10f32.powf(exp);
    let frac = max / base;
    let step = if frac >= 5.0 {
        5.0
    } else if frac >= 2.0 {
        2.0
    } else {
        1.0
    };
    step * base
}

fn with_alpha(c: [u8; 4], a: u8) -> [u8; 4] {
    [c[0], c[1], c[2], a]
}

/// Lay the overlay out for an output of `size` pixels.
pub(super) fn layout(content: &OverlayContent, size: [f32; 2]) -> Vec<Prim> {
    let [w, h] = size;
    if w < 16.0 || h < 16.0 {
        return Vec::new();
    }
    // One "unit" is 1 px at 1080p, so labels keep their proportions at any
    // resolution.
    let u = w.min(h) / 1080.0 * content.text_scale.clamp(0.25, 4.0);
    let margin = 36.0 * u;
    let text = content.text_color;
    let dim = with_alpha(text, 200);
    let panel = with_alpha(content.background, 190);
    let mut prims = Vec::new();

    // Title and space name, top left.
    let mut y = margin;
    if let Some(title) = content.title.as_deref().filter(|t| !t.trim().is_empty()) {
        let size = 44.0 * u;
        prims.push(Prim::Text {
            pos: [margin, y],
            size,
            text: title.trim().to_string(),
            color: text,
        });
        y += size * 1.3;
    }
    if let Some(space) = content.space_name.as_deref() {
        prims.push(Prim::Text {
            pos: [margin, y],
            size: 28.0 * u,
            text: space.to_string(),
            color: dim,
        });
    }

    // Legend, top right.
    match &content.legend {
        Some(OverlayLegend::Categorical { title, entries }) if !entries.is_empty() => {
            let title_size = 26.0 * u;
            let label_size = 22.0 * u;
            let row_h = label_size * 1.45;
            let swatch_r = label_size * 0.32;
            let pad = 14.0 * u;
            let col_gap = 24.0 * u;
            let avail_h = h - 2.0 * margin - 2.0 * pad - title_size * 1.5 - row_h;
            let per_col = ((avail_h / row_h).floor() as usize).max(1);
            let max_cols = 3;
            let shown = entries
                .len()
                .min(per_col * max_cols)
                .min(MAX_LEGEND_ENTRIES);
            let hidden = entries.len() - shown;
            let n_cols = shown.div_ceil(per_col).max(1);
            let rows = shown.div_ceil(n_cols);
            let mut col_widths = vec![0.0f32; n_cols];
            for (i, (label, _)) in entries.iter().take(shown).enumerate() {
                let col = i / rows;
                let lw = swatch_r * 2.0 + 10.0 * u + text_width(label, label_size);
                col_widths[col] = col_widths[col].max(lw);
            }
            let more = (hidden > 0).then(|| format!("+{hidden} more"));
            let body_w: f32 =
                col_widths.iter().sum::<f32>() + col_gap * (n_cols.saturating_sub(1)) as f32;
            let title_w = text_width(title, title_size);
            let more_w = more
                .as_deref()
                .map(|m| text_width(m, label_size))
                .unwrap_or(0.0);
            let inner_w = body_w.max(title_w).max(more_w);
            let rows_h = rows as f32 * row_h + if more.is_some() { row_h } else { 0.0 };
            let inner_h = title_size * 1.5 + rows_h;
            let x0 = w - margin - inner_w - 2.0 * pad;
            let y0 = margin;
            prims.push(Prim::Rect {
                rect: [x0, y0, inner_w + 2.0 * pad, inner_h + 2.0 * pad],
                color: panel,
            });
            prims.push(Prim::Text {
                pos: [x0 + pad, y0 + pad],
                size: title_size,
                text: title.clone(),
                color: text,
            });
            let body_y = y0 + pad + title_size * 1.5;
            let mut col_x = x0 + pad;
            for (col, col_w) in col_widths.iter().enumerate() {
                for row in 0..rows {
                    let i = col * rows + row;
                    if i >= shown {
                        break;
                    }
                    let (label, color) = &entries[i];
                    let ry = body_y + row as f32 * row_h;
                    prims.push(Prim::Circle {
                        center: [col_x + swatch_r, ry + label_size * 0.55],
                        radius: swatch_r,
                        color: with_alpha(*color, 255),
                    });
                    prims.push(Prim::Text {
                        pos: [col_x + swatch_r * 2.0 + 10.0 * u, ry],
                        size: label_size,
                        text: label.clone(),
                        color: text,
                    });
                }
                col_x += col_w + col_gap;
            }
            if let Some(more) = more {
                prims.push(Prim::Text {
                    pos: [x0 + pad, body_y + rows as f32 * row_h],
                    size: label_size,
                    text: more,
                    color: dim,
                });
            }
        }
        Some(OverlayLegend::Continuous { title, min, max }) => {
            let title_size = 26.0 * u;
            let label_size = 22.0 * u;
            let pad = 14.0 * u;
            let bar_w = 260.0 * u;
            let bar_h = 18.0 * u;
            let lo = format_number(*min);
            let hi = format_number(*max);
            let inner_w = bar_w.max(text_width(title, title_size));
            let inner_h = title_size * 1.5 + bar_h + 8.0 * u + label_size * 1.2;
            let x0 = w - margin - inner_w - 2.0 * pad;
            let y0 = margin;
            prims.push(Prim::Rect {
                rect: [x0, y0, inner_w + 2.0 * pad, inner_h + 2.0 * pad],
                color: panel,
            });
            prims.push(Prim::Text {
                pos: [x0 + pad, y0 + pad],
                size: title_size,
                text: title.clone(),
                color: text,
            });
            let bar_y = y0 + pad + title_size * 1.5;
            let bar_x = x0 + pad + inner_w - bar_w;
            prims.push(Prim::Gradient {
                rect: [bar_x, bar_y, bar_w, bar_h],
            });
            let label_y = bar_y + bar_h + 8.0 * u;
            prims.push(Prim::Text {
                pos: [bar_x, label_y],
                size: label_size,
                text: lo,
                color: text,
            });
            prims.push(Prim::Text {
                pos: [bar_x + bar_w - text_width(&hi, label_size), label_y],
                size: label_size,
                text: hi,
                color: text,
            });
        }
        _ => {}
    }

    // Scale bar, bottom right.
    if let Some(px_per_coord) = content.scale_px_per_coord {
        let units = content.units_per_coord;
        if px_per_coord.is_finite() && px_per_coord > 0.0 && units.is_finite() && units > 0.0 {
            let target_px = w * 0.15;
            let length = nice_length(target_px / px_per_coord * units);
            let bar_len = length / units * px_per_coord;
            if length > 0.0 && bar_len >= 8.0 * u {
                let label = format!("{} {}", format_number(length), content.unit_label.trim());
                let label_size = 24.0 * u;
                let bar_h = (6.0 * u).max(2.0);
                let x1 = w - margin;
                let y_bar = h - margin - bar_h;
                prims.push(Prim::Rect {
                    rect: [x1 - bar_len, y_bar, bar_len, bar_h],
                    color: text,
                });
                let lw = text_width(&label, label_size);
                prims.push(Prim::Text {
                    pos: [x1 - bar_len * 0.5 - lw * 0.5, y_bar - label_size * 1.35],
                    size: label_size,
                    text: label,
                    color: text,
                });
            }
        }
    }

    prims
}

/// Draw `prims` with egui, offset by `origin`. Used for the viewport preview.
pub(super) fn paint_egui(painter: &egui::Painter, origin: egui::Pos2, prims: &[Prim]) {
    let c32 = |c: [u8; 4]| egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]);
    for prim in prims {
        match prim {
            Prim::Rect { rect, color } => {
                let r = egui::Rect::from_min_size(
                    origin + egui::vec2(rect[0], rect[1]),
                    egui::vec2(rect[2], rect[3]),
                );
                painter.rect_filled(r, rect[3].min(rect[2]) * 0.08, c32(*color));
            }
            Prim::Circle {
                center,
                radius,
                color,
            } => {
                painter.circle_filled(
                    origin + egui::vec2(center[0], center[1]),
                    *radius,
                    c32(*color),
                );
            }
            Prim::Gradient { rect } => {
                let steps = 64;
                let seg_w = rect[2] / steps as f32;
                for i in 0..steps {
                    let t = i as f64 / (steps - 1) as f64;
                    let c = colorous::VIRIDIS.eval_continuous(t);
                    let seg = egui::Rect::from_min_size(
                        origin + egui::vec2(rect[0] + seg_w * i as f32, rect[1]),
                        egui::vec2(seg_w + 0.5, rect[3]),
                    );
                    painter.rect_filled(seg, 0.0, egui::Color32::from_rgb(c.r, c.g, c.b));
                }
            }
            Prim::Text {
                pos,
                size,
                text,
                color,
            } => {
                painter.text(
                    origin + egui::vec2(pos[0], pos[1]),
                    egui::Align2::LEFT_TOP,
                    text,
                    egui::FontId::proportional(*size),
                    c32(*color),
                );
            }
        }
    }
}

/// A frame buffer to draw into: tightly packed 8-bit pixels, RGBA or BGRA.
pub(super) struct Canvas<'a> {
    pub(super) pixels: &'a mut [u8],
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) bgra: bool,
}

impl Canvas<'_> {
    fn blend(&mut self, x: i64, y: i64, color: [u8; 4], coverage: f32) {
        if x < 0 || y < 0 || x >= self.width as i64 || y >= self.height as i64 {
            return;
        }
        let a = (color[3] as f32 / 255.0) * coverage.clamp(0.0, 1.0);
        if a <= 0.0 {
            return;
        }
        let idx = (y as usize * self.width as usize + x as usize) * 4;
        let (r, b) = if self.bgra { (2, 0) } else { (0, 2) };
        let px = &mut self.pixels[idx..idx + 4];
        let mix = |dst: u8, src: u8| (dst as f32 + (src as f32 - dst as f32) * a).round() as u8;
        px[r] = mix(px[r], color[0]);
        px[1] = mix(px[1], color[1]);
        px[b] = mix(px[b], color[2]);
        px[3] = (px[3] as f32 + (255.0 - px[3] as f32) * a).round() as u8;
    }

    /// Fill a rectangle with anti-aliased edges.
    fn fill_rect(&mut self, rect: [f32; 4], color: [u8; 4]) {
        let [x, y, w, h] = rect;
        let (x0, y0, x1, y1) = (x, y, x + w, y + h);
        for py in y0.floor() as i64..y1.ceil() as i64 {
            let cy = ((py as f32 + 1.0).min(y1) - (py as f32).max(y0)).clamp(0.0, 1.0);
            for px in x0.floor() as i64..x1.ceil() as i64 {
                let cx = ((px as f32 + 1.0).min(x1) - (px as f32).max(x0)).clamp(0.0, 1.0);
                self.blend(px, py, color, cx * cy);
            }
        }
    }

    fn fill_circle(&mut self, center: [f32; 2], radius: f32, color: [u8; 4]) {
        let [cx, cy] = center;
        for py in (cy - radius - 1.0).floor() as i64..=(cy + radius + 1.0).ceil() as i64 {
            for px in (cx - radius - 1.0).floor() as i64..=(cx + radius + 1.0).ceil() as i64 {
                let dx = px as f32 + 0.5 - cx;
                let dy = py as f32 + 0.5 - cy;
                let d = (dx * dx + dy * dy).sqrt();
                self.blend(px, py, color, radius + 0.5 - d);
            }
        }
    }

    fn fill_gradient(&mut self, rect: [f32; 4]) {
        let [x, y, w, h] = rect;
        if w <= 0.0 {
            return;
        }
        for px in x.floor() as i64..(x + w).ceil() as i64 {
            let t = ((px as f32 + 0.5 - x) / w).clamp(0.0, 1.0);
            let c = colorous::VIRIDIS.eval_continuous(t as f64);
            let col_rect = [px as f32, y, 1.0, h];
            self.fill_rect(col_rect, [c.r, c.g, c.b, 255]);
        }
    }

    fn draw_text(&mut self, pos: [f32; 2], size: f32, text: &str, color: [u8; 4]) {
        let font = font().as_scaled(PxScale::from(size));
        let baseline = pos[1] + font.ascent();
        let mut x = pos[0];
        let mut prev = None;
        for ch in text.chars() {
            let id = font.glyph_id(ch);
            if let Some(prev) = prev {
                x += font.kern(prev, id);
            }
            let glyph = id.with_scale_and_position(size, ab_glyph::point(x, baseline));
            if let Some(outline) = font.outline_glyph(glyph) {
                let bounds = outline.px_bounds();
                outline.draw(|gx, gy, coverage| {
                    self.blend(
                        bounds.min.x as i64 + gx as i64,
                        bounds.min.y as i64 + gy as i64,
                        color,
                        coverage,
                    );
                });
            }
            x += font.h_advance(id);
            prev = Some(id);
        }
    }

    pub(super) fn draw(&mut self, prims: &[Prim]) {
        for prim in prims {
            match prim {
                Prim::Rect { rect, color } => self.fill_rect(*rect, *color),
                Prim::Circle {
                    center,
                    radius,
                    color,
                } => self.fill_circle(*center, *radius, *color),
                Prim::Gradient { rect } => self.fill_gradient(*rect),
                Prim::Text {
                    pos,
                    size,
                    text,
                    color,
                } => self.draw_text(*pos, *size, text, *color),
            }
        }
    }
}

impl StvizApp {
    /// Draw the overlay over the viewport, as a preview of what exports get.
    pub(super) fn draw_overlay_preview(&mut self, ui: &egui::Ui, rect: egui::Rect, ppp: f32) {
        let Some(ds) = self.dataset.clone() else {
            return;
        };
        let ppu = self.camera.pixels_per_unit;
        let Some(content) = self.overlay_content(&ds, |scale| ppu * scale / ppp) else {
            return;
        };
        // Keep clear of the stats line in the top-right corner.
        let mut area = rect;
        if self.show_stats {
            area.min.y += 18.0;
        }
        let prims = layout(&content, [area.width(), area.height()]);
        paint_egui(&ui.painter().with_clip_rect(rect), area.min, &prims);
    }

    /// What the overlay shows for the current frame.
    ///
    /// `px_per_coord` maps a coordinate unit of a space to output pixels, given
    /// that space's normalisation scale; the caller knows the camera.
    pub(super) fn overlay_content(
        &mut self,
        ds: &Dataset,
        px_per_coord: impl Fn(f32) -> f32,
    ) -> Option<OverlayContent> {
        let settings = self.overlay.clone();
        if !settings.enabled {
            return None;
        }
        let (from, to, color_from, color_to, segment_t, _seg) = self.current_segment(ds);
        let space_name = |idx: usize| ds.meta.spaces.get(idx).map(|s| s.name.clone());

        let space_label = if settings.show_space_name {
            if from == to || segment_t <= 0.0 {
                space_name(from)
            } else if segment_t >= 1.0 {
                space_name(to)
            } else {
                match (space_name(from), space_name(to)) {
                    (Some(a), Some(b)) => Some(format!("{a} to {b}")),
                    (a, b) => a.or(b),
                }
            }
        } else {
            None
        };

        let legend = if settings.show_legend {
            let key = if self.color_path_enabled {
                if segment_t < 0.5 {
                    color_from
                } else {
                    color_to
                }
            } else {
                ColorKey::Current
            };
            self.overlay_legend(ds, &key)
        } else {
            None
        };

        // A scale bar only makes sense for physical coordinates, while the view
        // rests on one space and isn't rearranged into a sample grid.
        let resting_space = if from == to || segment_t <= 0.0 {
            Some(from)
        } else if segment_t >= 1.0 {
            Some(to)
        } else {
            None
        };
        let scale_px_per_coord = if settings.show_scale_bar && !self.sample_grid_enabled {
            resting_space
                .and_then(|idx| ds.meta.spaces.get(idx).map(|s| (idx, s)))
                .filter(|(_, s)| is_spatial_space(&s.name))
                .map(|(idx, s)| {
                    let (_, scale) = self.space_transform(ds, idx, s);
                    px_per_coord(scale)
                })
        } else {
            None
        };

        let text = super::util::contrast_color(self.background_color);
        let bg = self.background_color;
        Some(OverlayContent {
            title: Some(settings.title.clone()).filter(|t| !t.trim().is_empty()),
            space_name: space_label,
            legend,
            scale_px_per_coord,
            units_per_coord: settings.units_per_coord,
            unit_label: settings.unit_label.clone(),
            text_scale: settings.text_scale,
            text_color: [text.r(), text.g(), text.b(), 255],
            background: [bg.r(), bg.g(), bg.b(), 255],
        })
    }

    fn overlay_legend(&self, ds: &Dataset, key: &ColorKey) -> Option<OverlayLegend> {
        let categorical = |obs_idx: usize, only_enabled: bool| -> Option<OverlayLegend> {
            let (name, _labels, categories, pal_opt) = ds.obs_categorical(obs_idx).ok()?;
            if categories.len() > MAX_FILTER_CATEGORIES {
                return None;
            }
            let pal = self.categorical_palette_for(obs_idx, categories.len(), pal_opt);
            let entries = categories
                .iter()
                .enumerate()
                .filter(|(i, _)| {
                    !only_enabled || self.enabled_categories.get(*i).copied().unwrap_or(true)
                })
                .map(|(i, label)| {
                    let c = pal.get(i).copied().unwrap_or(0xFFC8C8C8);
                    let rgba = c.to_le_bytes();
                    (label.clone(), rgba)
                })
                .collect();
            Some(OverlayLegend::Categorical {
                title: name.to_string(),
                entries,
            })
        };
        match key {
            ColorKey::Categorical(idx) => categorical(*idx, false),
            ColorKey::Continuous(_) | ColorKey::Gene(_) => self.cached_continuous_legend(key),
            ColorKey::Current => match self.color_mode {
                ColorMode::Categorical => {
                    let is_cat = matches!(
                        ds.meta.obs.get(self.active_obs_idx),
                        Some(ObsMeta::Categorical { .. })
                    );
                    is_cat
                        .then(|| categorical(self.active_obs_idx, true))
                        .flatten()
                }
                ColorMode::Continuous | ColorMode::Gene => {
                    self.legend_range
                        .as_ref()
                        .map(|l| OverlayLegend::Continuous {
                            title: l.label.clone(),
                            min: l.min,
                            max: l.max,
                        })
                }
            },
        }
    }

    /// Legend of a continuous or gene colour key. The viewport fills the
    /// colour cache for the keys of the current segment every frame, so this
    /// never recomputes a gene vector.
    fn cached_continuous_legend(&self, key: &ColorKey) -> Option<OverlayLegend> {
        let entry = self.color_cache.get(key)?;
        entry.legend.as_ref().map(continuous_legend)
    }
}

fn continuous_legend(l: &super::LegendRange) -> OverlayLegend {
    OverlayLegend::Continuous {
        title: l.label.clone(),
        min: l.min,
        max: l.max,
    }
}

/// Whether a space holds physical coordinates (e.g. `spatial`, `X_spatial`).
pub(super) fn is_spatial_space(name: &str) -> bool {
    name.to_ascii_lowercase().contains("spatial")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn content() -> OverlayContent {
        OverlayContent {
            title: Some("Mouse brain".to_string()),
            space_name: Some("spatial".to_string()),
            legend: Some(OverlayLegend::Categorical {
                title: "cluster".to_string(),
                entries: (0..5)
                    .map(|i| (format!("c{i}"), [200, 40 * i as u8, 10, 255]))
                    .collect(),
            }),
            scale_px_per_coord: Some(2.0),
            units_per_coord: 1.0,
            unit_label: "µm".to_string(),
            text_scale: 1.0,
            text_color: [255, 255, 255, 255],
            background: [0, 0, 0, 255],
        }
    }

    #[test]
    fn font_has_the_glyphs_labels_use() {
        let f = font();
        for ch in "µ+0123456789.-e AZaz".chars() {
            assert_ne!(f.glyph_id(ch).0, 0, "missing glyph for {ch:?}");
        }
    }

    #[test]
    fn nice_length_picks_1_2_5_steps() {
        assert_eq!(nice_length(7.3), 5.0);
        assert_eq!(nice_length(288.0), 200.0);
        assert_eq!(nice_length(1.0), 1.0);
        assert!((nice_length(0.034) - 0.02).abs() < 1e-6);
        assert_eq!(nice_length(0.0), 0.0);
    }

    #[test]
    fn format_number_is_short() {
        assert_eq!(format_number(0.0), "0");
        assert_eq!(format_number(500.0), "500");
        assert_eq!(format_number(2.5), "2.5");
        assert_eq!(format_number(0.125), "0.125");
        assert_eq!(format_number(12345.0), "1.2e4");
    }

    #[test]
    fn layout_keeps_everything_inside_the_frame() {
        let mut continuous = content();
        continuous.legend = Some(OverlayLegend::Continuous {
            title: "Gene: Gad1".to_string(),
            min: 0.0,
            max: 3.25,
        });
        for (c, size) in [content(), continuous].into_iter().flat_map(|c| {
            [[1920.0, 1080.0], [3840.0, 2160.0], [640.0, 480.0]].map(|s| (c.clone(), s))
        }) {
            let prims = layout(&c, size);
            assert!(prims.len() >= 8);
            for p in &prims {
                let (x, y, w, h) = match p {
                    Prim::Rect { rect, .. } | Prim::Gradient { rect } => {
                        (rect[0], rect[1], rect[2], rect[3])
                    }
                    Prim::Circle { center, radius, .. } => (
                        center[0] - radius,
                        center[1] - radius,
                        2.0 * radius,
                        2.0 * radius,
                    ),
                    Prim::Text {
                        pos, size, text, ..
                    } => (pos[0], pos[1], text_width(text, *size), *size),
                };
                assert!(x >= 0.0 && y >= 0.0, "{p:?} starts outside {size:?}");
                assert!(
                    x + w <= size[0] && y + h <= size[1],
                    "{p:?} ends outside {size:?}"
                );
            }
        }
    }

    #[test]
    fn scale_bar_label_matches_bar_length() {
        // 2 px per coordinate, 1 µm per coordinate: target 288 px = 144 µm,
        // so the bar is 100 µm = 200 px.
        let prims = layout(&content(), [1920.0, 1080.0]);
        let bar = prims.iter().rev().find_map(|p| match p {
            Prim::Rect { rect, .. } if rect[3] < 10.0 => Some(rect[2]),
            _ => None,
        });
        assert_eq!(bar, Some(200.0));
        assert!(prims
            .iter()
            .any(|p| matches!(p, Prim::Text { text, .. } if text == "100 µm")));
    }

    #[test]
    fn long_categorical_legends_are_truncated() {
        let mut c = content();
        c.legend = Some(OverlayLegend::Categorical {
            title: "cell type".to_string(),
            entries: (0..80)
                .map(|i| (format!("type {i}"), [1, 2, 3, 255]))
                .collect(),
        });
        let prims = layout(&c, [1920.0, 1080.0]);
        let swatches = prims
            .iter()
            .filter(|p| matches!(p, Prim::Circle { .. }))
            .count();
        assert_eq!(swatches, MAX_LEGEND_ENTRIES);
        assert!(prims
            .iter()
            .any(|p| matches!(p, Prim::Text { text, .. } if text == "+44 more")));
    }

    #[test]
    fn canvas_draws_text_and_respects_channel_order() {
        let (w, h) = (200u32, 60u32);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        let mut canvas = Canvas {
            pixels: &mut rgba,
            width: w,
            height: h,
            bgra: false,
        };
        canvas.draw(&[Prim::Text {
            pos: [4.0, 4.0],
            size: 32.0,
            text: "Ag µm".to_string(),
            color: [255, 0, 0, 255],
        }]);
        let lit = rgba.chunks_exact(4).filter(|p| p[0] > 128).count();
        assert!(lit > 50, "text should cover some pixels, got {lit}");
        assert!(rgba.chunks_exact(4).all(|p| p[2] == 0));

        let mut bgra = vec![0u8; (w * h * 4) as usize];
        Canvas {
            pixels: &mut bgra,
            width: w,
            height: h,
            bgra: true,
        }
        .draw(&[Prim::Rect {
            rect: [0.0, 0.0, 10.0, 10.0],
            color: [255, 0, 0, 255],
        }]);
        assert_eq!(&bgra[0..4], &[0, 0, 255, 255]);
    }
}
