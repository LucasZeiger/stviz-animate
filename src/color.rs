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
