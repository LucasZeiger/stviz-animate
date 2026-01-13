#[derive(Clone, Copy, Debug)]
pub struct Camera2D {
    pub center: [f32; 2],
    /// Pixels per world unit. Larger = zoom in.
    pub pixels_per_unit: f32,
}

impl Default for Camera2D {
    fn default() -> Self {
        Self {
            center: [0.0, 0.0],
            pixels_per_unit: 1.0,
        }
    }
}

impl Camera2D {
    pub fn fit_bbox(&mut self, bbox: [f32; 4], viewport_px: [f32; 2], padding_frac: f32) {
        let min_x = bbox[0];
        let min_y = bbox[1];
        let max_x = bbox[2];
        let max_y = bbox[3];

        let cx = 0.5 * (min_x + max_x);
        let cy = 0.5 * (min_y + max_y);
        self.center = [cx, cy];

        let w = (max_x - min_x).max(1e-6);
        let h = (max_y - min_y).max(1e-6);

        let fill = padding_frac.clamp(0.05, 0.95);
        let sx = (viewport_px[0] * fill) / w;
        let sy = (viewport_px[1] * fill) / h;
        self.pixels_per_unit = sx.min(sy);
        self.pixels_per_unit = self.pixels_per_unit.clamp(1e-6, 1e9);
    }

    pub fn pan_by_pixels(&mut self, delta_px: [f32; 2]) {
        // world_delta = -delta_px / pixels_per_unit
        self.center[0] -= delta_px[0] / self.pixels_per_unit;
        self.center[1] -= delta_px[1] / self.pixels_per_unit;
    }

    pub fn zoom_at_viewport_pixel(
        &mut self,
        mouse_px: [f32; 2],
        viewport_px: [f32; 2],
        zoom_factor: f32,
    ) {
        let old_ppu = self.pixels_per_unit;
        let new_ppu = (old_ppu * zoom_factor).clamp(1e-6, 1e9);

        // Keep world point under cursor stable:
        // world = center + (mouse - viewport/2)/ppu
        let before_world_x = self.center[0] + (mouse_px[0] - 0.5 * viewport_px[0]) / old_ppu;
        let before_world_y = self.center[1] + (mouse_px[1] - 0.5 * viewport_px[1]) / old_ppu;

        self.pixels_per_unit = new_ppu;

        let after_center_x = before_world_x - (mouse_px[0] - 0.5 * viewport_px[0]) / new_ppu;
        let after_center_y = before_world_y - (mouse_px[1] - 0.5 * viewport_px[1]) / new_ppu;

        self.center = [after_center_x, after_center_y];
    }
}
