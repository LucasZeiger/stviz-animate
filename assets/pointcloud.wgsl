struct Uniforms {
    viewport_px: vec2<f32>,
    _pad0: vec2<f32>,

    center: vec2<f32>,
    _pad1: vec2<f32>,

    pixels_per_unit: f32,
    t: f32,
    point_radius_px: f32,
    mask_mode: f32,
    color_t: f32,
    _pad2: f32,
    _pad2b: vec2<f32>,

    from_center: vec2<f32>,
    from_scale: f32,
    _pad3: f32,

    to_center: vec2<f32>,
    to_scale: f32,
    _pad4: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> pos_from: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> pos_to: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> colors_from_rgba8: array<u32>;
@group(0) @binding(4) var<storage, read> colors_to_rgba8: array<u32>;
@group(0) @binding(5) var<storage, read> draw_indices: array<u32>;

struct VsIn {
    @location(0) corner: vec2<f32>, // [-1..1] square corners, two triangles
    @builtin(instance_index) inst: u32,
};

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) corner: vec2<f32>,
    @location(1) color: vec4<f32>,
};

fn unpack_rgba8(c: u32) -> vec4<f32> {
    let r: f32 = f32((c >> 0u) & 255u) / 255.0;
    let g: f32 = f32((c >> 8u) & 255u) / 255.0;
    let b: f32 = f32((c >> 16u) & 255u) / 255.0;
    let a: f32 = f32((c >> 24u) & 255u) / 255.0;
    return vec4<f32>(r, g, b, a);
}

@vertex
fn vs_main(input: VsIn) -> VsOut {
    let idx: u32 = draw_indices[input.inst];

    let a_raw: vec2<f32> = pos_from[idx];
    let b_raw: vec2<f32> = pos_to[idx];

    let a: vec2<f32> = (a_raw - u.from_center) * u.from_scale;
    let b: vec2<f32> = (b_raw - u.to_center) * u.to_scale;

    let tt: f32 = clamp(u.t, 0.0, 1.0);
    let p: vec2<f32> = a + (b - a) * tt;

    // World -> viewport pixel coordinates:
    // screen_px = (p - center) * pixels_per_unit + viewport/2
    let screen_px: vec2<f32> = (p - u.center) * u.pixels_per_unit + (u.viewport_px * 0.5);

    // Add constant-size point sprite in pixels:
    let sprite_px: vec2<f32> = input.corner * u.point_radius_px;
    let screen2_px: vec2<f32> = screen_px + sprite_px;

    // Viewport pixels -> NDC (within the viewport)
    let ndc_x: f32 = (screen2_px.x / u.viewport_px.x) * 2.0 - 1.0;
    let ndc_y: f32 = 1.0 - (screen2_px.y / u.viewport_px.y) * 2.0;

    var out: VsOut;
    out.pos = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.corner = input.corner;
    let c0: vec4<f32> = unpack_rgba8(colors_from_rgba8[idx]);
    let c1: vec4<f32> = unpack_rgba8(colors_to_rgba8[idx]);
    let ct: f32 = clamp(u.color_t, 0.0, 1.0);
    out.color = c0 + (c1 - c0) * ct;
    return out;
}

@fragment
fn fs_main(input: VsOut) -> @location(0) vec4<f32> {
    // Circle mask unless disabled.
    if (u.mask_mode > 0.5 && dot(input.corner, input.corner) > 1.0) {
        discard;
    }
    return input.color;
}
