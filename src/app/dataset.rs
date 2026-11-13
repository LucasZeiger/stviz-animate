//! Dataset loading, color/filter recomputation and draw-index selection.

use super::{
    AdvancedCardFilter, ColorKey, ColorMode, LegendRange, StvizApp, MAX_FILTER_CATEGORIES,
};
use crate::{
    color::{categorical_palette, gradient_map, pack_rgba8, value_range},
    data::{Dataset, ObsMeta},
};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
    thread::JoinHandle,
};

/// A .stviz being opened and validated on a worker thread.
pub(super) struct DatasetLoad {
    path: PathBuf,
    handle: JoinHandle<anyhow::Result<Arc<Dataset>>>,
}

impl StvizApp {
    /// Open a .stviz without blocking the UI; `poll_dataset_load` finishes it.
    pub(super) fn start_dataset_load(&mut self, path: &Path) {
        if self.export_busy() {
            self.last_error =
                Some("Finish or cancel the export before opening another dataset.".to_string());
            return;
        }
        if self.dataset_load.is_some() {
            self.last_error = Some("Another dataset is still loading.".to_string());
            return;
        }
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| path.display().to_string());
        self.convert_status = Some(format!("Loading {name}..."));
        self.last_error = None;
        let path = path.to_path_buf();
        let worker_path = path.clone();
        self.dataset_load = Some(DatasetLoad {
            path,
            handle: std::thread::spawn(move || Dataset::load(&worker_path)),
        });
    }

    pub(super) fn poll_dataset_load(&mut self) {
        if !self
            .dataset_load
            .as_ref()
            .is_some_and(|load| load.handle.is_finished())
        {
            return;
        }
        let load = self.dataset_load.take().unwrap();
        let result = match load.handle.join() {
            Ok(Ok(ds)) => self.install_dataset(ds, &load.path),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(anyhow::anyhow!("the loader stopped unexpectedly")),
        };
        match result {
            Ok(()) => {
                self.convert_status = Some(format!("Loaded dataset: {}", load.path.display()));
            }
            Err(e) => {
                let msg = format!("Load failed: {e:#}");
                eprintln!("{msg}");
                self.convert_status = None;
                self.last_error = Some(msg);
            }
        }
    }

    /// Close the current dataset and release its file.
    pub(super) fn unload_dataset(&mut self) {
        self.dataset = None;
        self.dataset_path = None;
        self.scenes = Default::default();
        self.dataset_id = self.dataset_id.wrapping_add(1);
        self.shared.params.lock().dataset = None;
        self.base_indices.clear();
        self.draw_indices = Arc::new(Vec::new());
        self.indices_id = self.indices_id.wrapping_add(1);
        self.playing = false;
    }

    /// Make an already opened dataset the current one and reset the view,
    /// timeline, colors and filters to their defaults for it.
    pub(super) fn install_dataset(&mut self, ds: Arc<Dataset>, path: &Path) -> anyhow::Result<()> {
        self.reset_scenes_for(&ds);
        self.dataset = Some(ds.clone());
        self.dataset_path = Some(path.to_path_buf());
        if !is_mock_dataset(path) {
            self.remember_dir_of(path);
            self.remember_recent_file(path);
        }
        self.open_path = path.display().to_string();
        self.last_error = None;
        self.dataset_id = self.dataset_id.wrapping_add(1);

        // default selections
        let from_idx = find_space_by_name(&ds, "spatial").unwrap_or(0);
        let mut to_idx = find_space_by_name(&ds, "x_umap")
            .or_else(|| find_space_by_name(&ds, "umap"))
            .unwrap_or((ds.meta.spaces.len().saturating_sub(1)).min(1));
        if ds.meta.spaces.len() > 1 && from_idx == to_idx {
            to_idx = if from_idx == 0 { 1 } else { 0 };
        }
        self.from_space = from_idx;
        self.to_space = to_idx;
        self.active_obs_idx = 0;
        self.color_mode = ColorMode::Categorical;
        self.space_path.clear();
        if ds.meta.spaces.len() >= 2 {
            self.space_path.push(from_idx);
            self.space_path.push(to_idx);
        }
        self.color_path_enabled = true;
        self.color_path.clear();
        self.color_path.push(ColorKey::Current);
        self.color_path.push(ColorKey::Current);
        self.filter_path = vec![None; self.space_path.len()];
        self.key_times = vec![0.0, 1.0];
        self.key_collapsed = vec![false; self.space_path.len()];
        self.color_cache.clear();
        self.filtered_color_cache.clear();
        self.category_state.clear();
        self.category_overrides.clear();
        self.category_palette_cache = None;
        self.active_filters.clear();
        self.load_filter_state(&ds);

        let is_mock = is_mock_dataset(path);
        if is_mock {
            let mut mock_spaces = Vec::new();
            let mut push_unique = |idx: usize| {
                if !mock_spaces.contains(&idx) {
                    mock_spaces.push(idx);
                }
            };
            if let Some(idx) = find_space_by_name(&ds, "spatial") {
                push_unique(idx);
            }
            if let Some(idx) =
                find_space_by_name(&ds, "x_pca").or_else(|| find_space_by_name(&ds, "pca"))
            {
                push_unique(idx);
            }
            if let Some(idx) =
                find_space_by_name(&ds, "x_umap").or_else(|| find_space_by_name(&ds, "umap"))
            {
                push_unique(idx);
            }
            if let Some(idx) =
                find_space_by_name(&ds, "x_tsne").or_else(|| find_space_by_name(&ds, "tsne"))
            {
                push_unique(idx);
            }
            if mock_spaces.len() >= 2 {
                self.space_path = mock_spaces;
                self.from_space = *self.space_path.first().unwrap_or(&self.from_space);
                self.to_space = *self.space_path.last().unwrap_or(&self.to_space);
                self.color_path = vec![ColorKey::Current; self.space_path.len()];
                self.filter_path = vec![None; self.space_path.len()];
                let n_keys = self.space_path.len();
                self.key_times = if n_keys > 1 {
                    (0..n_keys)
                        .map(|i| i as f32 / (n_keys.saturating_sub(1)) as f32)
                        .collect()
                } else {
                    vec![0.0]
                };
                self.key_collapsed = vec![false; self.space_path.len()];
            }
        }
        if !self.key_collapsed.is_empty() || !self.space_path.is_empty() {
            self.key_collapsed = vec![false; self.space_path.len()];
        }
        if is_mock {
            self.sample_grid_enabled = true;
            self.speed = 0.2;
        }
        if is_mock {
            if let Some(space_idx) = find_space_by_name(&ds, "spatial") {
                if let Some(pos) = self.space_path.iter().position(|idx| *idx == space_idx) {
                    self.reset_view_key_idx = pos;
                }
                if let Some(space) = ds.meta.spaces.get(space_idx) {
                    let bbox = self.space_bbox_for_view(&ds, space_idx, space);
                    let viewport_px =
                        if self.last_viewport_px[0] > 0.0 && self.last_viewport_px[1] > 0.0 {
                            self.last_viewport_px
                        } else {
                            [1000.0, 700.0]
                        };
                    self.camera.fit_bbox(bbox, viewport_px, 0.9);
                }
            }
        }

        self.sample_grid_obs_idx =
            find_categorical_by_names(&ds, &["sample", "samples", "library_id", "batch"]);
        self.sample_grid_space_idx = find_space_by_name(&ds, "spatial")
            .or_else(|| find_space_by_name(&ds, "centroid"))
            .or_else(|| {
                if ds.meta.spaces.is_empty() {
                    None
                } else {
                    Some(0)
                }
            });
        self.grid_cache = None;
        self.grid_version = self.grid_version.wrapping_add(1);
        self.sample_grid_custom_labels.clear();
        self.sample_grid_custom_labels_obs_idx = None;

        // default indices: 0..n
        let n = ds.meta.n_points as usize;
        let idx: Vec<u32> = (0..n as u32).collect();
        self.base_indices = idx;
        if self.max_draw_points == 0 {
            self.max_draw_points = if n > 500_000 { 300_000 } else { 0 };
        }
        self.apply_downsample();

        // default colors
        let cluster_names = [
            "leiden",
            "louvain",
            "cluster",
            "clusters",
            "cell_type",
            "celltype",
        ];
        if let Some(idx) = find_categorical_by_names(&ds, &cluster_names) {
            self.active_obs_idx = idx;
            self.load_filter_state(&ds);
        }
        self.recompute_colors_and_filters()?;

        // Fit camera to "from" space bbox (needs viewport size; approximate now)
        let bbox = self.space_bbox_for_view(&ds, self.from_space, &ds.meta.spaces[self.from_space]);
        self.camera.fit_bbox(bbox, [1000.0, 700.0], 0.9);
        self.legend_range = None;
        self.active_legend_range = None;
        self.reset_view_key_idx = 0;
        if let Some(space_idx) = self.space_path.first().copied() {
            if let Some(space) = ds.meta.spaces.get(space_idx) {
                let bbox = self.space_bbox_for_view(&ds, space_idx, space);
                // Use the real viewport size once a frame has been drawn.
                let viewport_px =
                    if self.last_viewport_px[0] > 0.0 && self.last_viewport_px[1] > 0.0 {
                        self.last_viewport_px
                    } else {
                        [1000.0, 700.0]
                    };
                self.camera.fit_bbox(bbox, viewport_px, 0.9);
            }
        }

        Ok(())
    }

    pub(super) fn recompute_colors_and_filters(&mut self) -> anyhow::Result<()> {
        let Some(ds) = self.dataset.clone() else {
            return Ok(());
        };
        let n = ds.meta.n_points as usize;

        match self.color_mode {
            ColorMode::Categorical => {
                // find first categorical if current isn't
                let mut cat_idx = None;
                for (i, o) in ds.meta.obs.iter().enumerate() {
                    if matches!(o, ObsMeta::Categorical { .. }) {
                        cat_idx = Some(i);
                        break;
                    }
                }
                if cat_idx.is_none() {
                    // fallback: all white
                    self.colors_rgba8 = Arc::new(vec![pack_rgba8(255, 255, 255, 255); n]);
                    self.colors_opaque = true;
                    self.colors_id = self.next_color_id();
                    self.legend_range = None;
                    let _ = self.recompute_draw_indices_with_filters();
                    return Ok(());
                }
                if !matches!(
                    ds.meta.obs[self.active_obs_idx],
                    ObsMeta::Categorical { .. }
                ) {
                    self.active_obs_idx = cat_idx.unwrap();
                    self.load_filter_state(&ds);
                }

                let (_name, labels, categories, pal_opt) =
                    ds.obs_categorical(self.active_obs_idx)?;
                let too_many = categories.len() > MAX_FILTER_CATEGORIES;
                let pal: Vec<u32> = if too_many {
                    categorical_palette(256)
                } else {
                    self.categorical_palette_for(self.active_obs_idx, categories.len(), pal_opt)
                };

                // init filter toggles
                if !too_many && self.enabled_categories.len() != categories.len() {
                    self.enabled_categories = vec![true; categories.len()];
                }

                let mut colors = Vec::with_capacity(n);
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

                self.colors_rgba8 = Arc::new(colors);
                self.colors_opaque = opaque;
                self.colors_id = self.next_color_id();
                self.legend_range = None;

                if !too_many {
                    self.recompute_draw_indices()?;
                } else {
                    self.enabled_categories.clear();
                    self.active_filters.remove(&self.active_obs_idx);
                    let _ = self.recompute_draw_indices_with_filters();
                }
            }
            ColorMode::Continuous => {
                // find first continuous if current isn't
                let mut cont_idx = None;
                for (i, o) in ds.meta.obs.iter().enumerate() {
                    if matches!(o, ObsMeta::Continuous { .. }) {
                        cont_idx = Some(i);
                        break;
                    }
                }
                if cont_idx.is_none() {
                    self.colors_rgba8 = Arc::new(vec![pack_rgba8(255, 255, 255, 255); n]);
                    self.colors_opaque = true;
                    self.colors_id = self.next_color_id();
                    self.legend_range = None;
                    let _ = self.recompute_draw_indices_with_filters();
                    return Ok(());
                }
                if !matches!(ds.meta.obs[self.active_obs_idx], ObsMeta::Continuous { .. }) {
                    self.active_obs_idx = cont_idx.unwrap();
                }

                let (name, vals) = ds.obs_continuous(self.active_obs_idx)?;
                let (vmin, vmax) = match value_range(vals, self.clip_outliers) {
                    Some((lo, hi)) if lo < hi => (lo, hi),
                    _ => (0.0, 1.0),
                };
                let colors = gradient_map(vals, vmin, vmax, &colorous::VIRIDIS);
                self.colors_rgba8 = Arc::new(colors);
                self.colors_opaque = vals.iter().all(|v| v.is_finite());
                self.colors_id = self.next_color_id();
                self.legend_range = Some(LegendRange {
                    label: name.to_string(),
                    min: vmin,
                    max: vmax,
                });

                // no categorical filter in continuous mode
                self.enabled_categories.clear();
                self.recompute_draw_indices_with_filters()?;
            }
            ColorMode::Gene => {
                // gene computed via UI interaction; keep current colors
                self.recompute_draw_indices_with_filters()?;
            }
        }

        Ok(())
    }

    pub(super) fn next_color_id(&mut self) -> u64 {
        self.color_id_gen = self.color_id_gen.wrapping_add(1);
        self.color_id_gen
    }

    pub(super) fn recompute_draw_indices_no_filter(&mut self) -> anyhow::Result<()> {
        let Some(ds) = self.dataset.as_ref() else {
            return Ok(());
        };
        let n = ds.meta.n_points as usize;
        let base: Vec<u32> = (0..n as u32).collect();
        self.base_indices = base;
        self.apply_downsample();
        Ok(())
    }

    fn recompute_draw_indices(&mut self) -> anyhow::Result<()> {
        let Some(ds) = self.dataset.as_ref() else {
            return Ok(());
        };
        if !matches!(
            ds.meta.obs[self.active_obs_idx],
            ObsMeta::Categorical { .. }
        ) {
            return self.recompute_draw_indices_with_filters();
        }

        let (_name, _labels, categories, _pal) = ds.obs_categorical(self.active_obs_idx)?;
        if categories.len() > MAX_FILTER_CATEGORIES {
            self.active_filters.remove(&self.active_obs_idx);
            return Ok(());
        }
        if self.enabled_categories.len() != categories.len() {
            self.enabled_categories = vec![true; categories.len()];
        }

        if self.active_filters.contains(&self.active_obs_idx)
            && !self.category_state.contains_key(&self.active_obs_idx)
        {
            self.category_state
                .insert(self.active_obs_idx, self.enabled_categories.clone());
        }
        self.recompute_draw_indices_with_filters()?;
        Ok(())
    }

    pub(super) fn recompute_draw_indices_with_filters(&mut self) -> anyhow::Result<()> {
        let Some(ds) = self.dataset.as_ref() else {
            return Ok(());
        };
        let n = ds.meta.n_points as usize;
        if self.active_filters.is_empty() {
            if matches!(
                ds.meta.obs.get(self.active_obs_idx),
                Some(ObsMeta::Categorical { .. })
            ) && !self.enabled_categories.is_empty()
                && self.enabled_categories.iter().all(|v| !*v)
            {
                self.base_indices.clear();
                self.apply_downsample();
                self.grid_version = self.grid_version.wrapping_add(1);
                self.grid_cache = None;
                return Ok(());
            }
            let base: Vec<u32> = (0..n as u32).collect();
            self.base_indices = base;
            self.apply_downsample();
            return Ok(());
        }

        let mut mask = vec![true; n];
        for &obs_idx in &self.active_filters {
            let Ok((_name, labels, categories, _pal)) = ds.obs_categorical(obs_idx) else {
                continue;
            };
            if categories.len() > MAX_FILTER_CATEGORIES {
                continue;
            }
            let enabled = self
                .category_state
                .get(&obs_idx)
                .cloned()
                .unwrap_or_else(|| vec![true; categories.len()]);
            for (i, &lab) in labels.iter().enumerate() {
                if !mask[i] {
                    continue;
                }
                let li = lab as usize;
                let keep = enabled.get(li).copied().unwrap_or(false);
                if !keep {
                    mask[i] = false;
                }
            }
        }

        let mut idx = Vec::with_capacity(n);
        for (i, keep) in mask.iter().enumerate() {
            if *keep {
                idx.push(i as u32);
            }
        }
        self.base_indices = idx;
        self.apply_downsample();
        self.grid_version = self.grid_version.wrapping_add(1);
        self.grid_cache = None;
        Ok(())
    }

    pub(super) fn clear_all_filters(&mut self, ds: &Dataset) {
        self.active_filters.clear();
        self.category_state.clear();
        self.filter_popup_open = false;
        if let Ok((_name, _labels, categories, _pal)) = ds.obs_categorical(self.active_obs_idx) {
            if categories.len() <= MAX_FILTER_CATEGORIES {
                self.enabled_categories = vec![true; categories.len()];
            } else {
                self.enabled_categories.clear();
            }
        } else {
            self.enabled_categories.clear();
        }
        let _ = self.recompute_draw_indices_with_filters();
    }

    pub(super) fn downsample_indices(max_draw: usize, indices: &[u32]) -> Vec<u32> {
        if max_draw == 0 || indices.len() <= max_draw {
            return indices.to_vec();
        }
        let step = indices.len().div_ceil(max_draw).max(1);
        let mut out = Vec::with_capacity(max_draw);
        for (i, idx) in indices.iter().enumerate() {
            if i % step == 0 {
                out.push(*idx);
                if out.len() >= max_draw {
                    break;
                }
            }
        }
        out
    }

    pub(super) fn sample_grid_filter_signature(&self, obs_idx: usize) -> u64 {
        if !self.sample_grid_use_filter {
            return 0;
        }
        let Some(enabled) = self.category_state.get(&obs_idx) else {
            return 0;
        };
        Self::advanced_filter_signature(0, obs_idx, 0, enabled)
    }

    pub(super) fn sample_grid_override_id(&self, space_idx: usize, obs_idx: usize) -> u64 {
        self.grid_version
            ^ ((space_idx as u64) << 32)
            ^ self.sample_grid_filter_signature(obs_idx)
            ^ (obs_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }

    pub(super) fn advanced_filter_signature(
        dataset_id: u64,
        obs_idx: usize,
        max_draw: usize,
        enabled: &[bool],
    ) -> u64 {
        let mut hash = 1469598103934665603u64 ^ dataset_id.wrapping_mul(1099511628211);
        hash ^= (obs_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        hash ^= (max_draw as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
        for (i, enabled) in enabled.iter().enumerate() {
            let v = if *enabled { 1u64 } else { 0u64 };
            hash = hash
                .wrapping_mul(1099511628211)
                .wrapping_add((i as u64).wrapping_add(v));
        }
        hash
    }

    fn advanced_filter_indices(
        ds: &Dataset,
        filter: &mut AdvancedCardFilter,
        max_draw: usize,
        dataset_id: u64,
    ) -> Option<(Arc<Vec<u32>>, u64)> {
        let requested_sig =
            Self::advanced_filter_signature(dataset_id, filter.obs_idx, max_draw, &filter.enabled);
        if filter.cached_dataset_id == dataset_id
            && filter.cached_max_draw == max_draw
            && filter.cached_indices_id == requested_sig
        {
            if let Some(indices) = filter.cached_indices.clone() {
                return Some((indices, filter.cached_indices_id));
            }
        }
        let Ok((_name, labels, categories, _pal)) = ds.obs_categorical(filter.obs_idx) else {
            return None;
        };
        if categories.len() > MAX_FILTER_CATEGORIES {
            return None;
        }
        if filter.enabled.len() != categories.len() {
            filter.enabled = vec![true; categories.len()];
        }
        let mut idx = Vec::new();
        if !filter.enabled.iter().all(|v| !*v) {
            idx.reserve(labels.len());
            for (i, &lab) in labels.iter().enumerate() {
                let li = lab as usize;
                if filter.enabled.get(li).copied().unwrap_or(false) {
                    idx.push(i as u32);
                }
            }
        }
        let out = Self::downsample_indices(max_draw, &idx);
        let sig =
            Self::advanced_filter_signature(dataset_id, filter.obs_idx, max_draw, &filter.enabled);
        let arc = Arc::new(out);
        filter.cached_indices = Some(arc.clone());
        filter.cached_indices_id = sig;
        filter.cached_dataset_id = dataset_id;
        filter.cached_max_draw = max_draw;
        Some((arc, sig))
    }

    pub(super) fn advanced_draw_override(
        &mut self,
        ds: &Dataset,
        seg_idx: usize,
    ) -> Option<(Arc<Vec<u32>>, u64)> {
        if self.advanced_cards.len() != self.space_path.len() {
            return None;
        }
        let max_draw = self.max_draw_points;
        let dataset_id = self.dataset_id;
        let card = self.advanced_cards.get_mut(seg_idx)?;
        let filter = card.filter.as_mut()?;
        Self::advanced_filter_indices(ds, filter, max_draw, dataset_id)
    }

    pub(super) fn apply_downsample(&mut self) {
        let max_draw = self.max_draw_points;
        if max_draw == 0 || self.base_indices.len() <= max_draw {
            self.draw_indices = Arc::new(self.base_indices.clone());
            self.indices_id = self.indices_id.wrapping_add(1);
            return;
        }

        let step = self.base_indices.len().div_ceil(max_draw).max(1);
        let mut out = Vec::with_capacity(max_draw);
        for (i, idx) in self.base_indices.iter().enumerate() {
            if i % step == 0 {
                out.push(*idx);
                if out.len() >= max_draw {
                    break;
                }
            }
        }
        self.draw_indices = Arc::new(out);
        self.indices_id = self.indices_id.wrapping_add(1);
    }

    pub(super) fn load_filter_state(&mut self, ds: &Dataset) {
        if let Ok((_name, _labels, categories, _pal)) = ds.obs_categorical(self.active_obs_idx) {
            if categories.len() > MAX_FILTER_CATEGORIES {
                self.enabled_categories.clear();
                return;
            }
            if let Some(saved) = self.category_state.get(&self.active_obs_idx) {
                if saved.len() == categories.len() {
                    self.enabled_categories = saved.clone();
                    return;
                }
            }
            self.enabled_categories = vec![true; categories.len()];
        }
    }
}

/// Mock datasets are temporary and get a showcase timeline when loaded.
fn is_mock_dataset(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with("mock_spatial_"))
}

/// Index of the name that best matches `needle`, ignoring case: an exact
/// match first, then a name with `needle` as a whole word ("X_umap" for
/// "umap", but not "n_samples" for "sample").
fn best_name_match<'a>(
    names: impl Iterator<Item = &'a str> + Clone,
    needle: &str,
) -> Option<usize> {
    let needle = needle.to_ascii_lowercase();
    names
        .clone()
        .position(|name| name.eq_ignore_ascii_case(&needle))
        .or_else(|| {
            names.clone().position(|name| {
                name.to_ascii_lowercase()
                    .split(|c: char| !c.is_ascii_alphanumeric())
                    .any(|word| word == needle)
            })
        })
}

/// First categorical obs column matching one of `needles`, in order of preference.
fn find_categorical_by_names(ds: &Dataset, needles: &[&str]) -> Option<usize> {
    let names = ds.meta.obs.iter().map(|o| match o {
        ObsMeta::Categorical { name, .. } => name.as_str(),
        // Never matches: only categorical columns are candidates.
        ObsMeta::Continuous { .. } => "",
    });
    needles
        .iter()
        .find_map(|needle| best_name_match(names.clone(), needle))
}

fn find_space_by_name(ds: &Dataset, needle: &str) -> Option<usize> {
    best_name_match(ds.meta.spaces.iter().map(|s| s.name.as_str()), needle)
}

#[cfg(test)]
mod tests {
    use super::best_name_match;

    #[test]
    fn name_matching_prefers_exact_then_whole_words() {
        let names = ["n_samples", "Sample_ID", "sample", "X_umap", "umap_3d"];
        let find = |needle| best_name_match(names.iter().copied(), needle);
        assert_eq!(find("sample"), Some(2));
        assert_eq!(find("umap"), Some(3));
        assert_eq!(find("x_umap"), Some(3));
        assert_eq!(find("id"), Some(1));
        assert_eq!(find("samp"), None);
        assert_eq!(best_name_match(["n_samples"].into_iter(), "sample"), None);
    }
}
