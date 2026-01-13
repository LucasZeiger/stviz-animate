use colorous::{Color, Gradient};

pub fn pack_rgba8(r: u8, g: u8, b: u8, a: u8) -> u32 {
    (r as u32) | ((g as u32) << 8) | ((b as u32) << 16) | ((a as u32) << 24)
}

pub fn gradient_map(values: &[f32], vmin: f32, vmax: f32, grad: &Gradient) -> Vec<u32> {
    let den = (vmax - vmin).max(1e-12);
    values
        .iter()
        .map(|&v| {
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
        let t = if n <= 1 { 0.0 } else { i as f64 / (n as f64 - 1.0) };
        let c = colorous::TURBO.eval_continuous(t);
        out.push(pack_rgba8(c.r, c.g, c.b, 255));
    }
    out
}
