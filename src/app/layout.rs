//! Plot bounding boxes, space transforms and sample-grid layout.

use super::{
    util::apply_ease, GridCache, GridCacheKey, SampleGridLabelMode, StvizApp, MAX_GRID_CATEGORIES,
};
use crate::data::Dataset;
use std::sync::Arc;

impl StvizApp {
    pub(super) fn effective_bbox(
        &mut self,
        ds: &Dataset,
        space_idx: usize,
        space: &crate::data::SpaceMeta,
    ) -> [f32; 4] {
        if self.sample_grid_enabled && self.sample_grid_space_idx == Some(space_idx) {
            if let Some((_pos, bbox)) = self.grid_positions_for(ds, space_idx) {
                return bbox;
            }
        }
        space.bbox
    }

    pub(super) fn current_plot_bbox_for_segment(
        &mut self,
        ds: &Dataset,
        from_space: usize,
        to_space: usize,
        segment_t: f32,
        draw_indices: &[u32],
        from_override: Option<&[f32]>,
        to_override: Option<&[f32]>,
    ) -> Option<[f32; 4]> {
        if draw_indices.is_empty() {
            return None;
        }
        let n = ds.meta.n_points as usize;
        if n == 0 {
            return None;
        }
        let from_meta = ds.meta.spaces.get(from_space)?;
        let to_meta = ds.meta.spaces.get(to_space)?;
        let (from_center, from_scale) = self.space_transform(ds, from_space, from_meta);
        let (to_center, to_scale) = self.space_transform(ds, to_space, to_meta);
        let from_pos = if let Some(pos) = from_override {
            if pos.len() == n * 2 {
                pos
            } else {
                ds.space_f32_2d(from_space).ok()?
            }
        } else {
            ds.space_f32_2d(from_space).ok()?
        };
        let to_pos = if let Some(pos) = to_override {
            if pos.len() == n * 2 {
                pos
            } else {
                ds.space_f32_2d(to_space).ok()?
            }
        } else {
            ds.space_f32_2d(to_space).ok()?
        };
        let t = apply_ease(segment_t.clamp(0.0, 1.0), self.ease_mode);
        let mut bbox = [
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ];
        let mut any = false;
        for idx in draw_indices {
            let i = *idx as usize;
            if i >= n {
                continue;
            }
            let base = i * 2;
            let xf = (from_pos[base] - from_center[0]) * from_scale;
            let yf = (from_pos[base + 1] - from_center[1]) * from_scale;
            let xt = (to_pos[base] - to_center[0]) * to_scale;
            let yt = (to_pos[base + 1] - to_center[1]) * to_scale;
            let x = xf + (xt - xf) * t;
            let y = yf + (yt - yf) * t;
            if !x.is_finite() || !y.is_finite() {
                continue;
            }
            any = true;
            if x < bbox[0] {
                bbox[0] = x;
            }
            if y < bbox[1] {
                bbox[1] = y;
            }
            if x > bbox[2] {
                bbox[2] = x;
            }
            if y > bbox[3] {
                bbox[3] = y;
            }
        }
        if any {
            Some(bbox)
        } else {
            None
        }
    }

    pub(super) fn export_fit_bbox(&mut self, ds: &Dataset) -> Option<[f32; 4]> {
        let mut space_indices: Vec<usize> = Vec::new();
        if !self.space_path.is_empty() {
            space_indices.extend(self.space_path.iter().copied());
        } else {
            space_indices.push(self.from_space);
            space_indices.push(self.to_space);
        }
        if space_indices.is_empty() {
            return None;
        }
        space_indices.sort_unstable();
        space_indices.dedup();
        let mut union: Option<[f32; 4]> = None;
        for idx in space_indices {
            if let Some(space) = ds.meta.spaces.get(idx) {
                let bbox = self.space_bbox_for_view(ds, idx, space);
                union = Some(match union {
                    Some(current) => [
                        current[0].min(bbox[0]),
                        current[1].min(bbox[1]),
                        current[2].max(bbox[2]),
                        current[3].max(bbox[3]),
                    ],
                    None => bbox,
                });
            }
        }
        union
    }

    pub(super) fn space_transform(
        &mut self,
        ds: &Dataset,
        space_idx: usize,
        space: &crate::data::SpaceMeta,
    ) -> ([f32; 2], f32) {
        let bbox = self.effective_bbox(ds, space_idx, space);
        let min_x = bbox[0];
        let min_y = bbox[1];
        let max_x = bbox[2];
        let max_y = bbox[3];
        let w = (max_x - min_x).max(1e-6);
        let h = (max_y - min_y).max(1e-6);
        let scale = 1.0 / w.max(h);
        let center = [0.5 * (min_x + max_x), 0.5 * (min_y + max_y)];
        (center, scale)
    }

    pub(super) fn space_bbox_for_view(
        &mut self,
        ds: &Dataset,
        space_idx: usize,
        space: &crate::data::SpaceMeta,
    ) -> [f32; 4] {
        let bbox = self.effective_bbox(ds, space_idx, space);
        let min_x = bbox[0];
        let min_y = bbox[1];
        let max_x = bbox[2];
        let max_y = bbox[3];
        let w = (max_x - min_x).max(1e-6);
        let h = (max_y - min_y).max(1e-6);
        let scale = 1.0 / w.max(h);
        let hw = 0.5 * w * scale;
        let hh = 0.5 * h * scale;
        [-hw, -hh, hw, hh]
    }

    pub(super) fn grid_positions_for(
        &mut self,
        ds: &Dataset,
        space_idx: usize,
    ) -> Option<(Arc<Vec<f32>>, [f32; 4])> {
        self.grid_positions_for_space(ds, space_idx, false)
    }

    pub(super) fn grid_positions_for_space(
        &mut self,
        ds: &Dataset,
        space_idx: usize,
        allow_non_selected: bool,
    ) -> Option<(Arc<Vec<f32>>, [f32; 4])> {
        if !self.sample_grid_enabled {
            return None;
        }
        if !allow_non_selected && self.sample_grid_space_idx != Some(space_idx) {
            return None;
        }
        let obs_idx = self.sample_grid_obs_idx?;
        let filter_sig = self.sample_grid_filter_signature(obs_idx);
        let key = GridCacheKey {
            dataset_id: self.dataset_id,
            space_idx,
            obs_idx,
            use_filter: self.sample_grid_use_filter,
            filter_sig,
            version: self.grid_version,
        };
        if let Some(cache) = &self.grid_cache {
            if cache.key == key {
                return Some((cache.positions.clone(), cache.bbox));
            }
        }

        let (_name, labels, categories, _pal) = ds.obs_categorical(obs_idx).ok()?;
        if categories.len() > MAX_GRID_CATEGORIES {
            return None;
        }

        let n_sel = categories.len().max(1);
        let mut map = vec![-1i32; categories.len()];
        for (i, cat) in (0..categories.len()).enumerate() {
            if cat < map.len() {
                map[cat] = i as i32;
            }
        }

        let pos = ds.space_f32_2d(space_idx).ok()?;
        let n = ds.meta.n_points as usize;
        if pos.len() != n * 2 {
            return None;
        }

        let mut min_xy = vec![[f32::INFINITY; 2]; n_sel];
        let mut max_xy = vec![[-f32::INFINITY; 2]; n_sel];

        for i in 0..n {
            let cat = labels.get(i).copied().unwrap_or(0) as usize;
            if cat >= map.len() {
                continue;
            }
            let si = map[cat];
            if si < 0 {
                continue;
            }
            let si = si as usize;
            let x = pos[i * 2];
            let y = pos[i * 2 + 1];
            if x < min_xy[si][0] {
                min_xy[si][0] = x;
            }
            if y < min_xy[si][1] {
                min_xy[si][1] = y;
            }
            if x > max_xy[si][0] {
                max_xy[si][0] = x;
            }
            if y > max_xy[si][1] {
                max_xy[si][1] = y;
            }
        }

        let mut centers = vec![[0.0f32; 2]; n_sel];
        let mut max_w = 1e-6f32;
        let mut max_h = 1e-6f32;
        for i in 0..n_sel {
            let min_x = min_xy[i][0];
            let min_y = min_xy[i][1];
            let max_x = max_xy[i][0];
            let max_y = max_xy[i][1];
            if !min_x.is_finite() || !max_x.is_finite() {
                centers[i] = [0.0, 0.0];
                continue;
            }
            let w = (max_x - min_x).max(1e-6);
            let h = (max_y - min_y).max(1e-6);
            centers[i] = [0.5 * (min_x + max_x), 0.5 * (min_y + max_y)];
            max_w = max_w.max(w);
            max_h = max_h.max(h);
        }

        let pad = (1.0 + self.sample_grid_padding).max(1.0);
        let tile_w = max_w * pad;
        let tile_h = max_h * pad;
        let cols = (n_sel as f32).sqrt().ceil() as usize;
        let rows = n_sel.div_ceil(cols);
        let origin_x = -((cols as f32 - 1.0) * 0.5);
        let origin_y = -((rows as f32 - 1.0) * 0.5);

        let mut offsets = vec![[0.0f32; 2]; n_sel];
        for i in 0..n_sel {
            let col = (i % cols) as f32;
            let row = (i / cols) as f32;
            offsets[i][0] = (origin_x + col) * tile_w;
            offsets[i][1] = (origin_y + row) * tile_h;
        }

        let mut out = vec![0.0f32; pos.len()];
        let mut bbox = [
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ];
        for i in 0..n {
            let mut x = pos[i * 2];
            let mut y = pos[i * 2 + 1];
            let cat = labels.get(i).copied().unwrap_or(0) as usize;
            if cat < map.len() {
                let si = map[cat];
                if si >= 0 {
                    let si = si as usize;
                    x = x - centers[si][0] + offsets[si][0];
                    y = y - centers[si][1] + offsets[si][1];
                }
            }
            out[i * 2] = x;
            out[i * 2 + 1] = y;
            if x < bbox[0] {
                bbox[0] = x;
            }
            if y < bbox[1] {
                bbox[1] = y;
            }
            if x > bbox[2] {
                bbox[2] = x;
            }
            if y > bbox[3] {
                bbox[3] = y;
            }
        }

        let is_spatial = ds
            .meta
            .spaces
            .get(space_idx)
            .map(|s| s.name.eq_ignore_ascii_case("spatial"))
            .unwrap_or(false);
        if is_spatial {
            let mut max_dim = 0.0f32;
            for space in &ds.meta.spaces {
                let w = (space.bbox[2] - space.bbox[0]).max(1e-6);
                let h = (space.bbox[3] - space.bbox[1]).max(1e-6);
                max_dim = max_dim.max(w.max(h));
            }
            let grid_dim = (bbox[2] - bbox[0])
                .max(1e-6)
                .max((bbox[3] - bbox[1]).max(1e-6));
            if max_dim > grid_dim {
                let scale = max_dim / grid_dim;
                let cx = 0.5 * (bbox[0] + bbox[2]);
                let cy = 0.5 * (bbox[1] + bbox[3]);
                bbox = [
                    f32::INFINITY,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    f32::NEG_INFINITY,
                ];
                for i in 0..n {
                    let x = cx + (out[i * 2] - cx) * scale;
                    let y = cy + (out[i * 2 + 1] - cy) * scale;
                    out[i * 2] = x;
                    out[i * 2 + 1] = y;
                    if x < bbox[0] {
                        bbox[0] = x;
                    }
                    if y < bbox[1] {
                        bbox[1] = y;
                    }
                    if x > bbox[2] {
                        bbox[2] = x;
                    }
                    if y > bbox[3] {
                        bbox[3] = y;
                    }
                }
            }
        }

        let positions = Arc::new(out);
        self.grid_cache = Some(GridCache {
            key,
            positions: positions.clone(),
            bbox,
        });
        Some((positions, bbox))
    }

    pub(super) fn ensure_sample_grid_custom_labels(
        &mut self,
        obs_idx: usize,
        categories: &[String],
    ) {
        if self.sample_grid_custom_labels_obs_idx != Some(obs_idx)
            || self.sample_grid_custom_labels.len() != categories.len()
        {
            self.sample_grid_custom_labels_obs_idx = Some(obs_idx);
            self.sample_grid_custom_labels = categories.to_vec();
        }
    }

    pub(super) fn sample_grid_label_positions(
        &self,
        ds: &Dataset,
    ) -> Option<Vec<(String, [f32; 2])>> {
        if !self.sample_grid_enabled || !self.sample_grid_labels_enabled {
            return None;
        }
        let space_idx = self.sample_grid_space_idx?;
        let obs_idx = self.sample_grid_obs_idx?;
        let (_name, labels, categories, _pal) = ds.obs_categorical(obs_idx).ok()?;
        if categories.len() > MAX_GRID_CATEGORIES {
            return None;
        }
        let mut enabled = vec![true; categories.len()];
        if self.sample_grid_use_filter {
            if let Some(state) = self.category_state.get(&obs_idx) {
                for (i, slot) in enabled.iter_mut().enumerate() {
                    *slot = state.get(i).copied().unwrap_or(true);
                }
            }
            if enabled.iter().all(|v| !*v) {
                enabled.fill(true);
            }
        }

        let n_sel = categories.len().max(1);
        let mut map = vec![-1i32; categories.len()];
        for (i, cat) in (0..categories.len()).enumerate() {
            if cat < map.len() {
                map[cat] = i as i32;
            }
        }

        let pos = ds.space_f32_2d(space_idx).ok()?;
        let n = ds.meta.n_points as usize;
        if pos.len() != n * 2 {
            return None;
        }

        let mut min_xy = vec![[f32::INFINITY; 2]; n_sel];
        let mut max_xy = vec![[-f32::INFINITY; 2]; n_sel];

        for i in 0..n {
            let cat = labels.get(i).copied().unwrap_or(0) as usize;
            if cat >= map.len() {
                continue;
            }
            let si = map[cat];
            if si < 0 {
                continue;
            }
            let si = si as usize;
            let x = pos[i * 2];
            let y = pos[i * 2 + 1];
            if x < min_xy[si][0] {
                min_xy[si][0] = x;
            }
            if y < min_xy[si][1] {
                min_xy[si][1] = y;
            }
            if x > max_xy[si][0] {
                max_xy[si][0] = x;
            }
            if y > max_xy[si][1] {
                max_xy[si][1] = y;
            }
        }

        let mut max_w = 1e-6f32;
        let mut max_h = 1e-6f32;
        for i in 0..n_sel {
            let min_x = min_xy[i][0];
            let min_y = min_xy[i][1];
            let max_x = max_xy[i][0];
            let max_y = max_xy[i][1];
            if !min_x.is_finite() || !max_x.is_finite() {
                continue;
            }
            let w = (max_x - min_x).max(1e-6);
            let h = (max_y - min_y).max(1e-6);
            max_w = max_w.max(w);
            max_h = max_h.max(h);
        }

        let pad = (1.0 + self.sample_grid_padding).max(1.0);
        let tile_w = max_w * pad;
        let tile_h = max_h * pad;
        let cols = (n_sel as f32).sqrt().ceil() as usize;
        let rows = n_sel.div_ceil(cols);
        let origin_x = -((cols as f32 - 1.0) * 0.5);
        let origin_y = -((rows as f32 - 1.0) * 0.5);

        let mut offsets = vec![[0.0f32; 2]; n_sel];
        for i in 0..n_sel {
            let col = (i % cols) as f32;
            let row = (i / cols) as f32;
            offsets[i][0] = (origin_x + col) * tile_w;
            offsets[i][1] = (origin_y + row) * tile_h;
        }

        let mut out = Vec::with_capacity(n_sel);
        for cat_idx in 0..categories.len() {
            if !enabled.get(cat_idx).copied().unwrap_or(true) {
                continue;
            }
            let mut label = categories
                .get(cat_idx)
                .cloned()
                .unwrap_or_else(|| format!("Category {}", cat_idx + 1));
            if self.sample_grid_label_mode == SampleGridLabelMode::Custom
                && self.sample_grid_custom_labels_obs_idx == Some(obs_idx)
                && self.sample_grid_custom_labels.len() == categories.len()
            {
                if let Some(custom) = self.sample_grid_custom_labels.get(cat_idx) {
                    if !custom.trim().is_empty() {
                        label = custom.clone();
                    }
                }
            }
            let i = map[cat_idx].max(0) as usize;
            let label_pos = [offsets[i][0], offsets[i][1] + tile_h * 0.55];
            out.push((label, label_pos));
        }
        Some(out)
    }
}
