//! Keyframe bookkeeping and per-segment color/filter resolution.

use super::{
    AdvancedCardFilter, ColorCacheEntry, ColorKey, KeyframeFilter, LegendRange, StvizApp,
    MAX_FILTER_CATEGORIES,
};
use crate::{
    color::{categorical_palette, gradient_map, pack_rgba8},
    data::Dataset,
};
use std::sync::Arc;

impl StvizApp {
    pub(super) fn ensure_color_path_len(&mut self, len: usize) {
        if self.color_path.is_empty() {
            self.color_path.push(ColorKey::Current);
        }
        if self.color_path.len() < len {
            let last = self.color_path.last().cloned().unwrap_or(ColorKey::Current);
            while self.color_path.len() < len {
                self.color_path.push(last.clone());
            }
        } else if self.color_path.len() > len {
            self.color_path.truncate(len);
        }
    }

    pub(super) fn ensure_filter_path_len(&mut self, len: usize) {
        self.filter_path.resize(len, None);
    }

    pub(super) fn ensure_space_path_len(&mut self, ds: &Dataset, len: usize) {
        let max_idx = ds.meta.spaces.len().saturating_sub(1);
        if self.space_path.is_empty() {
            let start = self.from_space.min(max_idx);
            self.space_path.push(start);
        }
        if self.space_path.len() < len {
            let last = *self.space_path.last().unwrap_or(&0);
            while self.space_path.len() < len {
                self.space_path.push(last);
            }
        } else if self.space_path.len() > len {
            self.space_path.truncate(len);
        }
        for idx in &mut self.space_path {
            *idx = (*idx).min(max_idx);
        }
        if let Some(first) = self.space_path.first().copied() {
            self.from_space = first;
        }
        if let Some(last) = self.space_path.last().copied() {
            self.to_space = last;
        }
        if self.reset_view_key_idx >= self.space_path.len() {
            self.reset_view_key_idx = self.space_path.len().saturating_sub(1);
        }
        if self.key_times.len() != self.space_path.len() {
            self.ensure_key_times_len(self.space_path.len());
        }
    }

    pub(super) fn ensure_key_times_len(&mut self, len: usize) {
        if len == 0 {
            self.key_times.clear();
            self.key_collapsed.clear();
            return;
        }
        if self.key_times.len() == len {
            if let Some(first) = self.key_times.first_mut() {
                *first = 0.0;
            }
            if let Some(last) = self.key_times.last_mut() {
                *last = 1.0;
            }
            if self.key_collapsed.len() != len {
                self.key_collapsed.resize(len, false);
            }
            return;
        }
        if len == 1 {
            self.key_times = vec![0.0];
            self.key_collapsed = vec![false];
            return;
        }
        self.key_times = (0..len).map(|i| i as f32 / (len as f32 - 1.0)).collect();
        if self.key_collapsed.len() != len {
            self.key_collapsed.resize(len, false);
        }
    }

    pub(super) fn distribute_key_times_evenly(&mut self) {
        let len = self.key_times.len();
        if len < 2 {
            return;
        }
        for (i, t) in self.key_times.iter_mut().enumerate() {
            *t = i as f32 / (len as f32 - 1.0);
        }
    }

    pub(super) fn move_keyframe(&mut self, from: usize, to: usize) {
        let len = self.space_path.len();
        if len < 2 || from >= len || to >= len || from == to {
            return;
        }
        if self.color_path.len() != len {
            self.ensure_color_path_len(len);
        }
        self.ensure_filter_path_len(len);
        if self.key_times.len() != len {
            self.ensure_key_times_len(len);
        }
        if self.key_collapsed.len() != len {
            self.key_collapsed.resize(len, false);
        }

        let space = self.space_path.remove(from);
        let color = self.color_path.remove(from);
        let filter = self.filter_path.remove(from);
        let time = self.key_times.remove(from);
        let collapsed = self.key_collapsed.remove(from);

        self.space_path.insert(to, space);
        self.color_path.insert(to, color);
        self.filter_path.insert(to, filter);
        self.key_times.insert(to, time);
        self.key_collapsed.insert(to, collapsed);

        if let Some(selected) = self.selected_key_idx {
            if selected == from {
                self.selected_key_idx = Some(to);
            } else if from < selected && to >= selected {
                self.selected_key_idx = Some(selected.saturating_sub(1));
            } else if from > selected && to <= selected {
                self.selected_key_idx = Some((selected + 1).min(len - 1));
            }
        }

        self.distribute_key_times_evenly();
    }

    pub(super) fn remove_keyframe(&mut self, idx: usize) {
        if self.space_path.len() <= 2 || idx >= self.space_path.len() {
            return;
        }
        self.space_path.remove(idx);
        if idx < self.color_path.len() {
            self.color_path.remove(idx);
        }
        if idx < self.filter_path.len() {
            self.filter_path.remove(idx);
        }
        if idx < self.key_times.len() {
            self.key_times.remove(idx);
        }
        if idx < self.key_collapsed.len() {
            self.key_collapsed.remove(idx);
        }
        self.selected_key_idx = None;
        self.dragging_key_idx = None;
        if let Some(first) = self.key_times.first_mut() {
            *first = 0.0;
        }
        if let Some(last) = self.key_times.last_mut() {
            *last = 1.0;
        }
        if self.reset_view_key_idx >= self.space_path.len() {
            self.reset_view_key_idx = self.space_path.len().saturating_sub(1);
        }
    }

    pub(super) fn segment_for_t(&self) -> (usize, f32) {
        let len = self.key_times.len();
        if len < 2 {
            return (0, self.t.clamp(0.0, 1.0));
        }
        let t = self.t.clamp(0.0, 1.0);
        let mut seg_idx = 0usize;
        while seg_idx + 1 < len && t > self.key_times[seg_idx + 1] {
            seg_idx += 1;
        }
        if seg_idx + 1 >= len {
            seg_idx = len - 2;
        }
        let t0 = self.key_times[seg_idx];
        let t1 = self.key_times[seg_idx + 1];
        let denom = (t1 - t0).max(1e-6);
        let local = ((t - t0) / denom).clamp(0.0, 1.0);
        (seg_idx, local)
    }

    pub(super) fn compute_colors_for_key(
        &self,
        ds: &Dataset,
        key: &ColorKey,
    ) -> Option<(Vec<u32>, Option<LegendRange>, bool)> {
        match key {
            ColorKey::Current => None,
            ColorKey::Categorical(idx) => {
                let (_name, labels, categories, pal_opt) = ds.obs_categorical(*idx).ok()?;
                let too_many = categories.len() > MAX_FILTER_CATEGORIES;
                let pal: Vec<u32> = if too_many {
                    categorical_palette(256)
                } else {
                    self.categorical_palette_for(*idx, categories.len(), pal_opt)
                };
                let mut colors = Vec::with_capacity(labels.len());
                let mut opaque = true;
                for &lab in labels {
                    let li = lab as usize;
                    let c = if too_many {
                        let idx = if pal.is_empty() { 0 } else { li % pal.len() };
                        pal.get(idx)
                            .copied()
                            .unwrap_or(pack_rgba8(200, 200, 200, 255))
                    } else {
                        pal.get(li)
                            .copied()
                            .unwrap_or(pack_rgba8(200, 200, 200, 255))
                    };
                    if (c >> 24) & 0xFF != 255 {
                        opaque = false;
                    }
                    colors.push(c);
                }
                Some((colors, None, opaque))
            }
            ColorKey::Continuous(idx) => {
                let (name, vals) = ds.obs_continuous(*idx).ok()?;
                let mut vmin = f32::INFINITY;
                let mut vmax = f32::NEG_INFINITY;
                for &v in vals {
                    if v.is_finite() {
                        vmin = vmin.min(v);
                        vmax = vmax.max(v);
                    }
                }
                if !vmin.is_finite() || !vmax.is_finite() || vmin == vmax {
                    vmin = 0.0;
                    vmax = 1.0;
                }
                let colors = gradient_map(vals, vmin, vmax, &colorous::VIRIDIS);
                let legend = Some(LegendRange {
                    label: name.to_string(),
                    min: vmin,
                    max: vmax,
                });
                Some((colors, legend, vals.iter().all(|v| v.is_finite())))
            }
            ColorKey::Gene(name) => {
                let gene = name.trim();
                if gene.is_empty() {
                    return None;
                }
                let gid = ds.find_gene(gene)?;
                let vec = ds.gene_vector_csc(gid).ok()?;
                let mut vmax = 0.0f32;
                for &v in &vec {
                    if v.is_finite() {
                        vmax = vmax.max(v);
                    }
                }
                let vmin = 0.0;
                let vmax = vmax.max(1e-6);
                let colors = gradient_map(&vec, vmin, vmax, &colorous::VIRIDIS);
                let legend = Some(LegendRange {
                    label: format!("Gene: {gene}"),
                    min: vmin,
                    max: vmax,
                });
                Some((colors, legend, vec.iter().all(|v| v.is_finite())))
            }
        }
    }

    pub(super) fn colors_for_key(
        &mut self,
        ds: &Dataset,
        key: &ColorKey,
    ) -> (Arc<Vec<u32>>, u64, Option<LegendRange>, bool) {
        if *key == ColorKey::Current {
            return (
                self.colors_rgba8.clone(),
                self.colors_id,
                self.legend_range.clone(),
                self.colors_opaque,
            );
        }
        if let Some(entry) = self.color_cache.get(key) {
            return (
                entry.colors.clone(),
                entry.id,
                entry.legend.clone(),
                entry.opaque,
            );
        }
        if let Some((colors, legend, opaque)) = self.compute_colors_for_key(ds, key) {
            let entry = ColorCacheEntry {
                colors: Arc::new(colors),
                id: self.next_color_id(),
                legend,
                opaque,
            };
            self.color_cache.insert(key.clone(), entry.clone());
            return (entry.colors, entry.id, entry.legend, entry.opaque);
        }
        (
            self.colors_rgba8.clone(),
            self.colors_id,
            self.legend_range.clone(),
            self.colors_opaque,
        )
    }

    pub(super) fn colors_with_filter(
        &mut self,
        ds: &Dataset,
        colors: Arc<Vec<u32>>,
        colors_id: u64,
        filter: Option<&KeyframeFilter>,
    ) -> (Arc<Vec<u32>>, u64, bool) {
        let Some(filter) = filter else {
            return (colors, colors_id, false);
        };
        let Ok((_name, labels, categories, _pal)) = ds.obs_categorical(filter.obs_idx) else {
            return (colors, colors_id, false);
        };
        if filter.enabled.len() != categories.len() || filter.enabled.iter().all(|v| *v) {
            return (colors, colors_id, false);
        }

        let filtered_id = Self::advanced_filter_signature(
            self.dataset_id ^ colors_id.rotate_left(17),
            filter.obs_idx,
            colors.len(),
            &filter.enabled,
        );
        if let Some(cached) = self.filtered_color_cache.get(&filtered_id) {
            return (cached.clone(), filtered_id, true);
        }

        let mut filtered = colors.as_ref().clone();
        for (i, &label) in labels.iter().enumerate() {
            if !filter.enabled.get(label as usize).copied().unwrap_or(false) {
                filtered[i] &= 0x00ff_ffff;
            }
        }
        let filtered = Arc::new(filtered);
        if self.filtered_color_cache.len() >= 4 {
            self.filtered_color_cache.clear();
        }
        self.filtered_color_cache
            .insert(filtered_id, filtered.clone());
        (filtered, filtered_id, true)
    }

    pub(super) fn segment_filters(
        &self,
        seg_idx: usize,
    ) -> (Option<KeyframeFilter>, Option<KeyframeFilter>) {
        if self.advanced_timeline_open && self.advanced_cards.len() == self.space_path.len() {
            let to_key = |filter: &Option<AdvancedCardFilter>| {
                filter.as_ref().map(|filter| KeyframeFilter {
                    obs_idx: filter.obs_idx,
                    enabled: filter.enabled.clone(),
                })
            };
            if !self.playing {
                if let Some(preview_idx) = self.advanced_preview_card {
                    let filter = self
                        .advanced_cards
                        .get(preview_idx)
                        .and_then(|card| to_key(&card.filter));
                    return (filter.clone(), filter);
                }
            }
            return (
                self.advanced_cards
                    .get(seg_idx)
                    .and_then(|card| to_key(&card.filter)),
                self.advanced_cards
                    .get(seg_idx + 1)
                    .and_then(|card| to_key(&card.filter)),
            );
        }
        (
            self.filter_path.get(seg_idx).cloned().flatten(),
            self.filter_path.get(seg_idx + 1).cloned().flatten(),
        )
    }

    pub(super) fn current_segment(
        &self,
        ds: &Dataset,
    ) -> (usize, usize, ColorKey, ColorKey, f32, usize) {
        let n_spaces = ds.meta.spaces.len();
        let clamp_idx = |idx: usize| idx.min(n_spaces.saturating_sub(1));
        let mut path: Vec<usize> = self
            .space_path
            .iter()
            .copied()
            .filter(|i| *i < n_spaces)
            .collect();
        if path.len() < 2 {
            path = vec![clamp_idx(self.from_space), clamp_idx(self.to_space)];
        }
        let (mut seg_idx, mut local_t) = if self.key_times.len() == path.len() && path.len() >= 2 {
            self.segment_for_t()
        } else {
            let segs = path.len().saturating_sub(1).max(1);
            let total = segs as f32;
            let scaled = (self.t.clamp(0.0, 1.0)) * total;
            let mut idx = scaled.floor() as usize;
            let mut local = scaled - idx as f32;
            if idx >= segs {
                idx = segs - 1;
                local = 1.0;
            }
            (idx, local)
        };
        if seg_idx >= path.len().saturating_sub(1) {
            seg_idx = path.len().saturating_sub(2);
            local_t = 1.0;
        }
        let from = path[seg_idx];
        let to = path[seg_idx + 1];
        let (color_from, color_to) = if self.color_path_enabled && self.color_path.len() >= 2 {
            let cf = self
                .color_path
                .get(seg_idx)
                .cloned()
                .unwrap_or(ColorKey::Current);
            let ct = self
                .color_path
                .get(seg_idx + 1)
                .cloned()
                .unwrap_or_else(|| cf.clone());
            (cf, ct)
        } else {
            (ColorKey::Current, ColorKey::Current)
        };
        (
            from,
            to,
            color_from,
            color_to,
            local_t.clamp(0.0, 1.0),
            seg_idx,
        )
    }
}
