use colorous::{Color, Gradient};

const TAB20: [Color; 20] = [
    Color {
        r: 31,
        g: 119,
        b: 180,
    },
    Color {
        r: 174,
        g: 199,
        b: 232,
    },
    Color {
        r: 255,
        g: 127,
        b: 14,
    },
    Color {
        r: 255,
        g: 187,
        b: 120,
    },
    Color {
        r: 44,
        g: 160,
        b: 44,
    },
    Color {
        r: 152,
        g: 223,
        b: 138,
    },
    Color {
        r: 214,
        g: 39,
        b: 40,
    },
    Color {
        r: 255,
        g: 152,
        b: 150,
    },
    Color {
        r: 148,
        g: 103,
        b: 189,
    },
    Color {
        r: 197,
        g: 176,
        b: 213,
    },
    Color {
        r: 140,
        g: 86,
        b: 75,
    },
    Color {
        r: 196,
        g: 156,
        b: 148,
    },
    Color {
        r: 227,
        g: 119,
        b: 194,
    },
    Color {
        r: 247,
        g: 182,
        b: 210,
    },
    Color {
        r: 127,
        g: 127,
        b: 127,
    },
    Color {
        r: 199,
        g: 199,
        b: 199,
    },
    Color {
        r: 188,
        g: 189,
        b: 34,
    },
    Color {
        r: 219,
        g: 219,
        b: 141,
    },
    Color {
        r: 23,
        g: 190,
        b: 207,
    },
    Color {
        r: 158,
        g: 218,
        b: 229,
    },
];

pub fn pack_rgba8(r: u8, g: u8, b: u8, a: u8) -> u32 {
    (r as u32) | ((g as u32) << 8) | ((b as u32) << 16) | ((a as u32) << 24)
}

/// The range to map onto a gradient: the finite min/max of `values`, or their
/// 1st-99th percentiles when `clip` is set so a few outliers don't flatten
/// the scale. `None` if there are no finite values.
pub fn value_range(values: &[f32], clip: bool) -> Option<(f32, f32)> {
    let mut vmin = f32::INFINITY;
    let mut vmax = f32::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            vmin = vmin.min(v);
            vmax = vmax.max(v);
        }
    }
    if vmin > vmax {
        return None;
    }
    if clip {
        // A sample is plenty for percentiles and keeps this fast on big datasets.
        const MAX_SAMPLE: usize = 200_000;
        let step = values.len().div_ceil(MAX_SAMPLE).max(1);
        let mut sample: Vec<f32> = values
            .iter()
            .step_by(step)
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        if sample.len() >= 100 {
            let last = sample.len() - 1;
            let lo_idx = last / 100;
            let hi_idx = last - last / 100;
            let (_, lo, _) = sample.select_nth_unstable_by(lo_idx, f32::total_cmp);
            let lo = *lo;
            let (_, hi, _) = sample.select_nth_unstable_by(hi_idx, f32::total_cmp);
            let hi = *hi;
            if lo < hi {
                return Some((lo, hi));
            }
        }
    }
    Some((vmin, vmax))
}

/// Upper end of the colour scale for a gene (the scale starts at 0). With
/// `clip`, the 99th percentile of the expressing cells.
pub fn gene_vmax(values: &[f32], clip: bool) -> f32 {
    let vmax = if clip {
        let positive: Vec<f32> = values
            .iter()
            .copied()
            .filter(|v| v.is_finite() && *v > 0.0)
            .collect();
        value_range(&positive, true).map(|r| r.1)
    } else {
        value_range(values, false).map(|r| r.1)
    };
    vmax.unwrap_or(0.0).max(1e-6)
}

pub fn gradient_map(values: &[f32], vmin: f32, vmax: f32, grad: &Gradient) -> Vec<u32> {
    let den = (vmax - vmin).max(1e-12);
    values
        .iter()
        .map(|&v| {
            if !v.is_finite() {
                return pack_rgba8(128, 128, 128, 0);
            }
            let t = ((v - vmin) / den).clamp(0.0, 1.0);
            let c: Color = grad.eval_continuous(t as f64);
            pack_rgba8(c.r, c.g, c.b, 255)
        })
        .collect()
}

pub fn categorical_palette(n: usize) -> Vec<u32> {
    // Evenly sample TURBO for distinct-ish colors.
    let mut out = Vec::with_capacity(n);
    for i in 0..n.max(1) {
        let t = if n <= 1 {
            0.0
        } else {
            i as f64 / (n as f64 - 1.0)
        };
        let c = colorous::TURBO.eval_continuous(t);
        out.push(pack_rgba8(c.r, c.g, c.b, 255));
    }
    out
}

fn palette_from_colors(colors: &[Color], n: usize) -> Vec<u32> {
    let count = n.max(1);
    let mut out = Vec::with_capacity(count);
    if colors.is_empty() {
        return out;
    }
    for i in 0..count {
        let c = colors[i % colors.len()];
        out.push(pack_rgba8(c.r, c.g, c.b, 255));
    }
    out
}

pub fn categorical_palette_named(name: &str, n: usize) -> Vec<u32> {
    match name.to_ascii_lowercase().as_str() {
        "tab10" => palette_from_colors(&colorous::CATEGORY10, n),
        "tab20" => palette_from_colors(&TAB20, n),
        "tableau10" => palette_from_colors(&colorous::TABLEAU10, n),
        "category10" => palette_from_colors(&colorous::CATEGORY10, n),
        "set1" => palette_from_colors(&colorous::SET1, n),
        "set2" => palette_from_colors(&colorous::SET2, n),
        "set3" => palette_from_colors(&colorous::SET3, n),
        "dark2" => palette_from_colors(&colorous::DARK2, n),
        "accent" => palette_from_colors(&colorous::ACCENT, n),
        "paired" => palette_from_colors(&colorous::PAIRED, n),
        "pastel1" => palette_from_colors(&colorous::PASTEL1, n),
        "pastel2" => palette_from_colors(&colorous::PASTEL2, n),
        _ => categorical_palette(n),
    }
}

#[cfg(test)]
mod tests {
    use super::{gene_vmax, value_range};

    #[test]
    fn value_range_ignores_non_finite_values() {
        assert_eq!(value_range(&[], false), None);
        assert_eq!(value_range(&[f32::NAN, f32::INFINITY], false), None);
        assert_eq!(
            value_range(&[3.0, f32::NAN, -1.0], false),
            Some((-1.0, 3.0))
        );
    }

    #[test]
    fn clipping_drops_outliers_but_keeps_constant_data_usable() {
        let mut values: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        values.push(1.0e6);
        assert_eq!(value_range(&values, false).unwrap().1, 1.0e6);
        let (lo, hi) = value_range(&values, true).unwrap();
        assert!(lo > 0.0 && lo < 0.05 && hi > 0.95 && hi < 1.0, "{lo} {hi}");
        // All equal: percentiles collapse, so fall back to the plain range.
        assert_eq!(value_range(&[2.0; 500], true), Some((2.0, 2.0)));
    }

    #[test]
    fn gene_vmax_uses_expressing_cells_when_clipping() {
        let mut values = vec![0.0f32; 10_000];
        for (i, v) in values.iter_mut().take(200).enumerate() {
            *v = 1.0 + i as f32 / 200.0;
        }
        values[0] = 500.0;
        assert_eq!(gene_vmax(&values, false), 500.0);
        let clipped = gene_vmax(&values, true);
        assert!(clipped > 1.9 && clipped < 2.1, "{clipped}");
        assert_eq!(gene_vmax(&[0.0; 10], true), 1e-6);
    }
}
