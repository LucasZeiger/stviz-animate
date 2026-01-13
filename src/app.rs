use crate::{
    camera::Camera2D,
    color::{categorical_palette, gradient_map, pack_rgba8},
    data::{Dataset, ObsMeta},
    render::{PointCloudGpu, SharedRender, Uniforms},
};
use anyhow::Context as _;
use eframe::egui;
use egui_wgpu::{wgpu, CallbackTrait};
use rand::seq::SliceRandom;
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
    thread,
};

const MAX_FILTER_CATEGORIES: usize = 5000;
const MAX_GRID_CATEGORIES: usize = 512;

pub struct StvizApp {
    dataset: Option<Arc<Dataset>>,
    dataset_path: Option<PathBuf>,
    dataset_id: u64,

    camera: Camera2D,

    from_space: usize,
    to_space: usize,
    transition_mode: TransitionMode,
    space_path: Vec<usize>,

    // Color mode
    color_mode: ColorMode,
    active_obs_idx: usize, // index into meta.obs

    // Categorical filtering (only when categorical mode)
    enabled_categories: Vec<bool>,
    category_state: HashMap<usize, Vec<bool>>,
    active_filters: HashSet<usize>,
    color_path_enabled: bool,
    color_path: Vec<ColorKey>,
    color_cache: HashMap<ColorKey, ColorCacheEntry>,
    color_id_gen: u64,
    selected_key_idx: Option<usize>,
    timeline_height: f32,
    key_times: Vec<f32>,
    keyframe_columns: usize,

    // Playback
    playing: bool,
    t: f32,
    speed: f32, // units per second
    play_direction: f32,
    playback_mode: PlaybackMode,
    ease_mode: EaseMode,
    point_radius_px: f32,
    max_draw_points: usize,
    fast_render: bool,
    opaque_points: bool,

    // Render plumbing
    shared: Arc<SharedRender>,
    colors_id: u64,
    colors_rgba8: Arc<Vec<u32>>,
    indices_id: u64,
    base_indices: Vec<u32>,
    draw_indices: Arc<Vec<u32>>,

    // Screenshot / recording
    project_dir: PathBuf,
    output_dir: PathBuf,
    screenshot_dir: PathBuf,
    record_dir: PathBuf,
    recording: bool,
    frame_counter: u64,
    exporting_loop: bool,
    export_fps: u32,
    export_dir: PathBuf,
    export_name: String,
    export_total_frames: u32,
    export_frame_index: u32,
    export_status: Option<String>,
    export_run_ffmpeg: bool,
    export_camera: Option<Camera2D>,

    // .h5ad -> .stviz conversion
    convert_python_cmd: String,
    convert_include_expr: bool,
    convert_generate_only: bool,
    convert_input: String,
    convert_output: String,
    convert_status: Option<String>,
    convert_running: bool,
    convert_handle: Option<std::thread::JoinHandle<Result<ConvertResult, String>>>,
    convert_last_python_cmd: Option<String>,
    convert_last_python_exe: Option<String>,

    // UI/view settings
    ui_scale: f32,
    ui_theme: UiTheme,
    background_color: egui::Color32,
    normalize_spaces: bool,
    show_axes: bool,
    show_stats: bool,
    adapter_label: String,
    frame_ms_avg: f32,

    // Sample grid for spatial
    sample_grid_enabled: bool,
    sample_grid_obs_idx: Option<usize>,
    sample_grid_space_idx: Option<usize>,
    sample_grid_use_filter: bool,
    sample_grid_padding: f32,
    grid_version: u64,
    grid_cache: Option<GridCache>,

    // Legend / gene input
    legend_range: Option<LegendRange>,
    gene_query: String,
    gene_selected: Option<String>,

    // Open path fallback + status
    open_path: String,
    last_error: Option<String>,

    // Viewport (for interactions)
    last_viewport_points: egui::Rect,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ColorMode {
    Categorical,
    Continuous,
    Gene, // gene vector computed on demand (requires expr in file)
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ColorKey {
    Current,
    Categorical(usize),
    Continuous(usize),
    Gene(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TransitionMode {
    Single,
    Path,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlaybackMode {
    Once,
    Loop,
    PingPong,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EaseMode {
    Linear,
    Smoothstep,
    SineInOut,
    QuadInOut,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KeyColorKind {
    Current,
    Categorical,
    Continuous,
    Gene,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UiTheme {
    Dark,
    Light,
    Slate,
    Matrix,
}

#[derive(Clone, Debug)]
struct LegendRange {
    label: String,
    min: f32,
    max: f32,
}

#[derive(Clone, Debug)]
struct ColorCacheEntry {
    colors: Arc<Vec<u32>>,
    id: u64,
    legend: Option<LegendRange>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct GridCacheKey {
    dataset_id: u64,
    space_idx: usize,
    obs_idx: usize,
    use_filter: bool,
    version: u64,
}

struct GridCache {
    key: GridCacheKey,
    positions: Arc<Vec<f32>>,
    bbox: [f32; 4],
}

#[derive(Clone, Debug)]
struct ScreenshotRequest {
    path: PathBuf,
    crop_px: Option<[u32; 4]>,
}

struct ConvertResult {
    msg: String,
    output: PathBuf,
    load_after: bool,
}

impl StvizApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let rs = cc
            .wgpu_render_state
            .as_ref()
            .expect("eframe must be built with the wgpu renderer");

        // `target_format` is exposed by eframe's wgpu render state in typical setups.
        // If this fails due to API drift, read it from rs and propagate here.
        let target_format = rs.target_format;
        let adapter_info = rs.adapter.get_info();
        let adapter_label = format!(
            "{} ({:?}, {:?})",
            adapter_info.name, adapter_info.device_type, adapter_info.backend
        );

        let project_dir = Self::resolve_project_dir();
        let output_dir = project_dir.join("output");
        let _ = std::fs::create_dir_all(&output_dir);

        Self {
            dataset: None,
            dataset_path: None,
            dataset_id: 1,

            camera: Camera2D::default(),

            from_space: 0,
            to_space: 0,
            transition_mode: TransitionMode::Single,
            space_path: Vec::new(),

            color_mode: ColorMode::Categorical,
            active_obs_idx: 0,
            enabled_categories: Vec::new(),
            category_state: HashMap::new(),
            active_filters: HashSet::new(),
            color_path_enabled: false,
            color_path: Vec::new(),
            color_cache: HashMap::new(),
            color_id_gen: 1,
            selected_key_idx: None,
            timeline_height: 160.0,
            key_times: Vec::new(),
            keyframe_columns: 3,

            playing: false,
            t: 0.0,
            speed: 0.35,
            play_direction: 1.0,
            playback_mode: PlaybackMode::PingPong,
            ease_mode: EaseMode::Smoothstep,
            point_radius_px: 0.5,
            max_draw_points: 0,
            fast_render: false,
            opaque_points: false,

            shared: Arc::new(SharedRender::new(target_format)),
            colors_id: 1,
            colors_rgba8: Arc::new(Vec::new()),
            indices_id: 1,
            base_indices: Vec::new(),
            draw_indices: Arc::new(Vec::new()),

            project_dir: project_dir.clone(),
            output_dir: output_dir.clone(),
            screenshot_dir: output_dir.clone(),
            record_dir: output_dir.clone(),
            recording: false,
            frame_counter: 0,
            exporting_loop: false,
            export_fps: 30,
            export_dir: output_dir.clone(),
            export_name: String::from("stviz-animate_loop.mp4"),
            export_total_frames: 0,
            export_frame_index: 0,
            export_status: None,
            export_run_ffmpeg: true,
            export_camera: None,

            convert_python_cmd: String::new(),
            convert_include_expr: false,
            convert_generate_only: false,
            convert_input: String::new(),
            convert_output: String::new(),
            convert_status: None,
            convert_running: false,
            convert_handle: None,
            convert_last_python_cmd: None,
            convert_last_python_exe: None,

            ui_scale: 1.0,
            ui_theme: UiTheme::Dark,
            background_color: egui::Color32::BLACK,
            normalize_spaces: true,
            show_axes: true,
            show_stats: true,
            adapter_label,
            frame_ms_avg: 0.0,

            sample_grid_enabled: false,
            sample_grid_obs_idx: None,
            sample_grid_space_idx: None,
            sample_grid_use_filter: true,
            sample_grid_padding: 0.15,
            grid_version: 1,
            grid_cache: None,

            legend_range: None,
            gene_query: String::new(),
            gene_selected: None,

            open_path: String::new(),
            last_error: None,

            last_viewport_points: egui::Rect::ZERO,
        }
    }

    fn open_dataset_dialog(&mut self) -> anyhow::Result<()> {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("stviz", &["stviz"])
            .set_title("Open .stviz")
            .set_directory(self.project_dir.clone())
            .pick_file()
        else {
            return Ok(());
        };

        self.load_dataset(&path)
    }

    fn load_dataset(&mut self, path: &Path) -> anyhow::Result<()> {
        let ds = Dataset::load(path).context("load dataset")?;
        self.dataset = Some(ds.clone());
        self.dataset_path = Some(path.to_path_buf());
        self.open_path = path.display().to_string();
        self.last_error = None;
        self.dataset_id = self.dataset_id.wrapping_add(1);

        // default selections
        self.from_space = 0;
        self.to_space = (ds.meta.spaces.len().saturating_sub(1)).min(1);
        self.active_obs_idx = 0;
        self.color_mode = ColorMode::Categorical;
        self.transition_mode = TransitionMode::Single;
        self.space_path.clear();
        if ds.meta.spaces.len() >= 2 {
            self.space_path.push(0);
            self.space_path.push(self.to_space);
        }
        self.color_path_enabled = true;
        self.color_path.clear();
        self.color_path.push(ColorKey::Current);
        self.color_path.push(ColorKey::Current);
        self.key_times = vec![0.0, 1.0];
        self.color_cache.clear();
        self.category_state.clear();
        self.active_filters.clear();
        self.load_filter_state(&ds);

        self.sample_grid_obs_idx = find_obs_by_name(&ds, "sample");
        self.sample_grid_space_idx = find_space_by_name(&ds, "spatial")
            .or_else(|| find_space_by_name(&ds, "centroid"))
            .or_else(|| if ds.meta.spaces.is_empty() { None } else { Some(0) });
        self.grid_cache = None;
        self.grid_version = self.grid_version.wrapping_add(1);

        // default indices: 0..n
        let n = ds.meta.n_points as usize;
        let idx: Vec<u32> = (0..n as u32).collect();
        self.base_indices = idx;
        if self.max_draw_points == 0 {
            self.max_draw_points = if n > 500_000 { 300_000 } else { 0 };
        }
        self.apply_downsample();

        // default colors
        self.recompute_colors_and_filters()?;

        // Fit camera to "from" space bbox (needs viewport size; approximate now)
        let bbox = self.space_bbox_for_view(&ds, self.from_space, &ds.meta.spaces[self.from_space]);
        self.camera.fit_bbox(bbox, [1000.0, 700.0], 0.9);
        self.legend_range = None;

        Ok(())
    }

    fn recompute_colors_and_filters(&mut self) -> anyhow::Result<()> {
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
                    self.colors_id = self.next_color_id();
                    self.legend_range = None;
                    let _ = self.recompute_draw_indices_with_filters();
                    return Ok(());
                }
                if !matches!(ds.meta.obs[self.active_obs_idx], ObsMeta::Categorical { .. }) {
                    self.active_obs_idx = cat_idx.unwrap();
                    self.load_filter_state(&ds);
                }

                let (_name, labels, categories, pal_opt) = ds.obs_categorical(self.active_obs_idx)?;
                let too_many = categories.len() > MAX_FILTER_CATEGORIES;
                let pal: Vec<u32> = if too_many {
                    categorical_palette(256)
                } else if let Some(p) = pal_opt {
                    p.to_vec()
                } else {
                    categorical_palette(categories.len())
                };

                // init filter toggles
                if !too_many && self.enabled_categories.len() != categories.len() {
                    self.enabled_categories = vec![true; categories.len()];
                }

                let mut colors = Vec::with_capacity(n);
                for &lab in labels {
                    let li = lab as usize;
                    let c = if too_many {
                        let idx = if pal.is_empty() { 0 } else { li % pal.len() };
                        pal.get(idx).copied().unwrap_or(pack_rgba8(200, 200, 200, 255))
                    } else {
                        pal.get(li).copied().unwrap_or(pack_rgba8(200, 200, 200, 255))
                    };
                    colors.push(c);
                }

                self.colors_rgba8 = Arc::new(colors);
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
                    self.colors_id = self.next_color_id();
                    self.legend_range = None;
                    let _ = self.recompute_draw_indices_with_filters();
                    return Ok(());
                }
                if !matches!(ds.meta.obs[self.active_obs_idx], ObsMeta::Continuous { .. }) {
                    self.active_obs_idx = cont_idx.unwrap();
                }

                let (name, vals) = ds.obs_continuous(self.active_obs_idx)?;
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
                self.colors_rgba8 = Arc::new(colors);
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

    fn next_color_id(&mut self) -> u64 {
        self.color_id_gen = self.color_id_gen.wrapping_add(1);
        self.color_id_gen
    }

    fn recompute_draw_indices_no_filter(&mut self) -> anyhow::Result<()> {
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
        if !matches!(ds.meta.obs[self.active_obs_idx], ObsMeta::Categorical { .. }) {
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

    fn recompute_draw_indices_with_filters(&mut self) -> anyhow::Result<()> {
        let Some(ds) = self.dataset.as_ref() else {
            return Ok(());
        };
        let n = ds.meta.n_points as usize;
        if self.active_filters.is_empty() {
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

    fn maybe_update_playback(&mut self, ctx: &egui::Context) {
        if self.exporting_loop {
            return;
        }
        if !self.playing {
            return;
        }
        let dt = ctx.input(|i| i.unstable_dt);
        self.t += self.play_direction * self.speed * dt;
        match self.playback_mode {
            PlaybackMode::Once => {
                if self.t >= 1.0 {
                    self.t = 1.0;
                    self.playing = false;
                } else if self.t <= 0.0 {
                    self.t = 0.0;
                    self.playing = false;
                }
            }
            PlaybackMode::Loop => {
                if self.t > 1.0 {
                    self.t -= 1.0;
                } else if self.t < 0.0 {
                    self.t += 1.0;
                }
            }
            PlaybackMode::PingPong => {
                if self.t > 1.0 {
                    self.t = 1.0;
                    self.play_direction = -self.play_direction;
                } else if self.t < 0.0 {
                    self.t = 0.0;
                    self.play_direction = -self.play_direction;
                }
            }
        }
    }

    fn handle_hotkeys(&mut self, ctx: &egui::Context) {
        if ctx.wants_keyboard_input() {
            return;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
            self.playing = !self.playing;
        }
        let step = 1.0 / self.export_fps.max(1) as f32;
        let big_step = step * 10.0;
        if ctx.input(|i| i.key_pressed(egui::Key::ArrowRight)) {
            self.t = (self.t + step).clamp(0.0, 1.0);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::ArrowLeft)) {
            self.t = (self.t - step).clamp(0.0, 1.0);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::ArrowUp)) {
            self.t = (self.t + big_step).clamp(0.0, 1.0);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::ArrowDown)) {
            self.t = (self.t - big_step).clamp(0.0, 1.0);
        }
    }

    fn poll_convert_job(&mut self) {
        let Some(handle) = self.convert_handle.as_ref() else {
            return;
        };
        if !handle.is_finished() {
            return;
        }
        let handle = self.convert_handle.take().unwrap();
        self.convert_running = false;
        match handle.join() {
            Ok(Ok(result)) => {
                self.convert_status = Some(result.msg);
                if result.load_after {
                    if !result.output.exists() {
                        let msg = format!("Load skipped: output not found at {}", result.output.display());
                        self.convert_status = Some(msg.clone());
                        self.last_error = Some(msg);
                        return;
                    }
                    if let Err(e) = self.load_dataset(&result.output) {
                        let msg = format!("Load failed: {e:#}");
                        eprintln!("{msg}");
                        self.last_error = Some(msg.clone());
                        self.convert_status = Some(msg);
                    } else {
                        self.open_path = result.output.display().to_string();
                        self.convert_status = Some(format!(
                            "Loaded dataset: {}",
                            result.output.display()
                        ));
                    }
                }
            }
            Ok(Err(msg)) => self.convert_status = Some(msg),
            Err(_) => self.convert_status = Some("Conversion thread panicked.".to_string()),
        }
    }

    fn queue_h5ad_convert(&mut self, path: &Path) {
        self.convert_input = path.display().to_string();
        self.convert_output = self.default_convert_output(&self.convert_input).display().to_string();
        self.start_convert();
    }

    fn detect_python_cmd(override_cmd: &str) -> Result<(String, String), String> {
        let override_cmd = override_cmd.trim();
        if !override_cmd.is_empty() {
            return Self::check_python_cmd(override_cmd);
        }

        let mut errors = Vec::new();
        if let Some(env_cmd) = Self::python_from_env() {
            match Self::check_python_cmd(&env_cmd) {
                Ok(result) => return Ok(result),
                Err(msg) => errors.push(msg),
            }
        }

        let candidates = ["python", "python3"];
        for candidate in candidates {
            match Self::check_python_cmd(candidate) {
                Ok(result) => return Ok(result),
                Err(msg) => errors.push(msg),
            }
        }
        if errors.is_empty() {
            Err("Python not found. Install Python and ensure `python` or `python3` is on PATH.".to_string())
        } else {
            Err(errors.join("\n"))
        }
    }

    fn check_python_cmd(cmd: &str) -> Result<(String, String), String> {
        let env_root = Self::python_env_root(cmd);
        let mut exe_cmd = Command::new(cmd);
        exe_cmd
            .arg("-c")
            .arg("import sys; print(sys.executable)");
        Self::apply_python_env(&mut exe_cmd, env_root.as_deref());
        let exe_out = exe_cmd
            .output()
            .map_err(|e| format!("`{cmd}` not found: {e}"))?;
        if !exe_out.status.success() {
            return Err(format!("`{cmd}` failed to run (exit {}).", exe_out.status));
        }
        let exe = String::from_utf8_lossy(&exe_out.stdout).trim().to_string();
        let exe_label = if exe.is_empty() { cmd.to_string() } else { exe };

        let mut import_cmd = Command::new(cmd);
        import_cmd
            .arg("-c")
            .arg("import anndata, h5py, numpy, pandas, scipy");
        Self::apply_python_env(&mut import_cmd, env_root.as_deref());
        let import_out = import_cmd
            .output()
            .map_err(|e| format!("Failed to run `{cmd}`: {e}"))?;
        if import_out.status.success() {
            return Ok((cmd.to_string(), exe_label));
        }

        let stderr = String::from_utf8_lossy(&import_out.stderr);
        let stdout = String::from_utf8_lossy(&import_out.stdout);
        let detail = if !stderr.trim().is_empty() {
            stderr.trim().to_string()
        } else {
            stdout.trim().to_string()
        };
        let detail = if detail.is_empty() {
            "Missing required modules.".to_string()
        } else {
            detail
        };
        Err(format!(
            "`{cmd}` ({exe_label}) is missing required modules.\n{detail}\nInstall with: {cmd} -m pip install -U anndata h5py numpy pandas scipy\nClear the Python field to auto-detect from PATH."
        ))
    }

    fn python_from_env() -> Option<String> {
        let env_prefix = std::env::var_os("VIRTUAL_ENV")
            .or_else(|| std::env::var_os("CONDA_PREFIX"))?;
        let base = PathBuf::from(env_prefix);
        let candidates = if cfg!(windows) {
            ["Scripts/python.exe", "python.exe"]
        } else {
            ["bin/python", "bin/python3"]
        };
        for rel in candidates {
            let path = base.join(rel);
            if path.exists() {
                return Some(path.display().to_string());
            }
        }
        None
    }

    fn default_convert_output(&self, input: &str) -> PathBuf {
        let _ = std::fs::create_dir_all(&self.output_dir);
        let input_path = Path::new(input);
        let stem = input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("converted");
        self.output_dir.join(format!("{stem}.stviz"))
    }

    fn python_picker_start_dir() -> Option<PathBuf> {
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("USERPROFILE").map(PathBuf::from));
        let mut candidates = Vec::new();

        if let Some(home_dir) = home.as_ref() {
            candidates.push(home_dir.join(".virtualenvs"));
            candidates.push(home_dir.join(".local").join("share").join("virtualenvs"));
            candidates.push(home_dir.join(".pyenv").join("versions"));
            candidates.push(home_dir.join(".conda").join("envs"));
            candidates.push(home_dir.join("miniconda3").join("envs"));
            candidates.push(home_dir.join("anaconda3").join("envs"));
            candidates.push(home_dir.join("mambaforge").join("envs"));
            candidates.push(home_dir.join("micromamba").join("envs"));
        }

        if let Some(local_app) = std::env::var_os("LOCALAPPDATA").map(PathBuf::from) {
            candidates.push(local_app.join("Programs"));
            candidates.push(local_app.join("Continuum").join("anaconda3").join("envs"));
        }
        if let Some(app_data) = std::env::var_os("APPDATA").map(PathBuf::from) {
            candidates.push(app_data.join("Python"));
        }

        for dir in &candidates {
            if dir.exists() {
                return Some(dir.to_path_buf());
            }
        }

        home
    }

    fn resolve_project_dir() -> PathBuf {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        if cwd.join("python").join("export_stviz.py").exists() {
            return cwd;
        }
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(mut dir) = exe_path.parent().map(|p| p.to_path_buf()) {
                for _ in 0..6 {
                    if dir.join("python").join("export_stviz.py").exists() {
                        return dir;
                    }
                    let Some(parent) = dir.parent() else {
                        break;
                    };
                    dir = parent.to_path_buf();
                }
            }
        }
        cwd
    }

    fn apply_python_env(cmd: &mut Command, env_root: Option<&Path>) -> Option<String> {
        let env_root = env_root?;

        cmd.env_remove("PYTHONHOME");
        cmd.env_remove("PYTHONPATH");
        cmd.env_remove("CONDA_PREFIX");
        cmd.env_remove("CONDA_DEFAULT_ENV");
        cmd.env_remove("CONDA_SHLVL");
        cmd.env_remove("VIRTUAL_ENV");

        if env_root.join("conda-meta").exists() {
            if let Some(name) = env_root.file_name().and_then(|s| s.to_str()) {
                cmd.env("CONDA_DEFAULT_ENV", name);
            }
            cmd.env("CONDA_PREFIX", &env_root);
            cmd.env("CONDA_SHLVL", "1");
            cmd.env("CONDA_DLL_SEARCH_MODIFICATION_ENABLE", "1");
        } else if env_root.join("pyvenv.cfg").exists() {
            cmd.env("VIRTUAL_ENV", &env_root);
        }

        let path_sep = if cfg!(windows) { ";" } else { ":" };
        let mut parts = Vec::new();
        if cfg!(windows) {
            parts.push(env_root.join("Library").join("bin"));
            parts.push(env_root.join("Scripts"));
        } else {
            parts.push(env_root.join("bin"));
        }
        parts.push(env_root.to_path_buf());
        let prefix = parts
            .iter()
            .filter_map(|p| p.to_str())
            .collect::<Vec<_>>()
            .join(path_sep);
        if let Some(old_path) = std::env::var_os("PATH") {
            let old_path = old_path.to_string_lossy();
            let new_path = if prefix.is_empty() {
                old_path.to_string()
            } else {
                format!("{}{}{}", prefix.as_str(), path_sep, old_path)
            };
            cmd.env("PATH", new_path);
        } else if !prefix.is_empty() {
            cmd.env("PATH", &prefix);
        }
        Some(prefix)
    }

    fn python_env_root(python_cmd: &str) -> Option<PathBuf> {
        let path = PathBuf::from(python_cmd);
        let parent = path.parent()?.to_path_buf();
        if let Some(dir_name) = parent.file_name().and_then(|s| s.to_str()) {
            if dir_name.eq_ignore_ascii_case("scripts") || dir_name.eq_ignore_ascii_case("bin") {
                return parent.parent().map(|p| p.to_path_buf());
            }
        }
        Some(parent)
    }

    fn start_convert(&mut self) {
        if self.convert_running {
            return;
        }
        let input = self.convert_input.trim();
        if input.is_empty() {
            self.convert_status = Some("Input .h5ad path is empty.".to_string());
            return;
        }
        if self.convert_output.trim().is_empty() {
            let out = self.default_convert_output(input);
            self.convert_output = out.display().to_string();
        }
        let output = self.convert_output.trim();
        if output.is_empty() {
            self.convert_status = Some("Output .stviz path is empty.".to_string());
            return;
        }

        let script = self.project_dir.join("python").join("export_stviz.py");
        if !script.exists() {
            self.convert_status = Some(format!("Exporter not found: {}", script.display()));
            return;
        }

        let (python_cmd, python_exe) = match Self::detect_python_cmd(&self.convert_python_cmd) {
            Ok(cmd) => cmd,
            Err(msg) => {
                self.convert_status = Some(msg);
                return;
            }
        };
        self.convert_last_python_cmd = Some(python_cmd.clone());
        self.convert_last_python_exe = Some(python_exe.clone());
        let input = input.to_string();
        let output = output.to_string();
        let script = script.to_string_lossy().to_string();
        let include_expr = self.convert_include_expr;
        let python_exe_label = python_exe.clone();
        let generate_only = self.convert_generate_only;
        let project_dir = self.project_dir.clone();
        let log_path = self
            .output_dir
            .join(format!("convert_log_{}.txt", chrono_like_timestamp()));

        self.convert_running = true;
        self.convert_status = Some(format!("Converting with {python_exe}..."));
        self.convert_handle = Some(thread::spawn(move || {
            let mut cmd = Command::new(&python_cmd);
            cmd.arg("-X")
                .arg("faulthandler")
                .arg(script)
                .arg("--input")
                .arg(&input)
                .arg("--output")
                .arg(&output);
            if include_expr {
                cmd.arg("--include-expr");
            }
            cmd.env("PYTHONFAULTHANDLER", "1");
            cmd.current_dir(&project_dir);

            let env_root = Self::python_env_root(&python_cmd);
            let path_prefix = Self::apply_python_env(&mut cmd, env_root.as_deref());

            let mut log = String::new();
            log.push_str("stviz-animate converter log\n");
            log.push_str(&format!("python_cmd: {python_cmd}\n"));
            log.push_str(&format!("python_exe: {python_exe_label}\n"));
            log.push_str(&format!("project_dir: {}\n", project_dir.display()));
            log.push_str(&format!("input: {input}\n"));
            log.push_str(&format!("output: {output}\n"));
            if let Some(root) = env_root.as_ref() {
                log.push_str(&format!("env_root: {}\n", root.display()));
            }
            if let Some(prefix) = path_prefix.as_ref() {
                log.push_str(&format!("path_prefix: {prefix}\n"));
            }

            let out = match cmd.output() {
                Ok(out) => out,
                Err(e) => {
                    log.push_str(&format!("spawn_error: {e}\n"));
                    let _ = std::fs::write(&log_path, log);
                    return Err(format!(
                        "Failed to run exporter: {e}\nLog: {}",
                        log_path.display()
                    ));
                }
            };
            log.push_str(&format!("status: {}\n", out.status));
            log.push_str("stdout:\n");
            log.push_str(&String::from_utf8_lossy(&out.stdout));
            log.push_str("\nstderr:\n");
            log.push_str(&String::from_utf8_lossy(&out.stderr));
            let _ = std::fs::write(&log_path, log);

            if !out.status.success() {
                let stderr = String::from_utf8_lossy(&out.stderr);
                let stdout = String::from_utf8_lossy(&out.stdout);
                return Err(format!(
                    "Exporter failed (python: {python_exe_label}, {}):\n{}\n{}\nLog: {}",
                    out.status,
                    stdout,
                    stderr,
                    log_path.display()
                ));
            }
            let stdout = String::from_utf8_lossy(&out.stdout);
            Ok(ConvertResult {
                msg: format!(
                    "Conversion done (python: {python_exe_label}):\n{}",
                    stdout.trim()
                ),
                output: PathBuf::from(output),
                load_after: !generate_only,
            })
        }));
    }

    fn start_export_loop(&mut self) {
        if self.exporting_loop {
            return;
        }
        if self.speed <= 0.0 {
            self.export_status = Some("Export failed: speed must be > 0.".to_string());
            return;
        }
        let base_period = 1.0 / self.speed.max(1e-6);
        let period = match self.playback_mode {
            PlaybackMode::PingPong => base_period * 2.0,
            _ => base_period,
        };
        let total = (period * self.export_fps as f32).round().max(2.0) as u32;

        let ts = chrono_like_timestamp();
        self.export_dir = self.record_dir.join(format!("loop_{ts}"));
        let _ = std::fs::create_dir_all(&self.export_dir);
        if self.export_name.trim().is_empty() {
            self.export_name = String::from("stviz-animate_loop.mp4");
        }

        self.export_total_frames = total;
        self.export_frame_index = 0;
        self.exporting_loop = true;
        self.export_status = Some(format!("Exporting {total} frames..."));
        self.playing = false;
        self.recording = false;
        self.export_camera = None;
    }

    fn finish_export_loop(&mut self) {
        let frames = self.export_total_frames;
        let frames_dir = self.export_dir.clone();
        self.export_camera = None;
        if !self.export_run_ffmpeg {
            self.export_status = Some(format!(
                "Exported {frames} frames to {}",
                frames_dir.display()
            ));
            return;
        }

        let ffmpeg_ok = std::process::Command::new("ffmpeg")
            .arg("-version")
            .output()
            .is_ok();
        if !ffmpeg_ok {
            self.export_status = Some(format!(
                "Exported {frames} frames to {} (ffmpeg not found)",
                frames_dir.display()
            ));
            return;
        }

        let out_path = self.record_dir.join(self.export_name.trim());
        let pattern = frames_dir.join("frame_%06d.png").to_string_lossy().to_string();
        let out_str = out_path.to_string_lossy().to_string();
        let status = std::process::Command::new("ffmpeg")
            .arg("-y")
            .arg("-framerate")
            .arg(self.export_fps.to_string())
            .arg("-i")
            .arg(&pattern)
            .arg("-c:v")
            .arg("libx264")
            .arg("-pix_fmt")
            .arg("yuv420p")
            .arg(&out_str)
            .status();

        match status {
            Ok(s) if s.success() => {
                self.export_status = Some(format!("Wrote video: {}", out_path.display()));
            }
            Ok(s) => {
                self.export_status = Some(format!("ffmpeg failed: {}", s));
            }
            Err(e) => {
                self.export_status = Some(format!("ffmpeg error: {e}"));
            }
        }
    }

    fn handle_screenshot_events(&mut self, ctx: &egui::Context) {
        // If we receive a screenshot event, save it.
        // We trigger screenshots with a viewport command carrying user_data.
        let events = ctx.input(|i| i.events.clone());
        for ev in events {
            if let egui::Event::Screenshot { image, user_data, .. } = ev {
                if let Some(req) = user_data
                    .data
                    .as_ref()
                    .and_then(|u| u.downcast_ref::<ScreenshotRequest>().cloned())
                {
                    let _ = save_color_image_png(&image, &req.path, req.crop_px);
                } else if let Some(any) = user_data
                    .data
                    .as_ref()
                    .and_then(|u| u.downcast_ref::<PathBuf>().cloned())
                {
                    let _ = save_color_image_png(&image, &any, None);
                }
            }
        }
    }

    fn request_screenshot(&mut self, ctx: &egui::Context, path: PathBuf, crop_px: Option<[u32; 4]>) {
        let req = ScreenshotRequest { path, crop_px };
        ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::new(req)));
    }

    fn ui_left_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("stviz-animate");

        ui.horizontal(|ui| {
            ui.label("Python (optional)");
            let edit = egui::TextEdit::singleline(&mut self.convert_python_cmd)
                .desired_width(180.0)
                .hint_text("python or /path/to/python");
            ui.add(edit);
            if ui.button("Pick python").clicked() {
                let mut dialog = rfd::FileDialog::new().set_title("Select Python");
                if let Some(dir) = Self::python_picker_start_dir() {
                    dialog = dialog.set_directory(dir);
                }
                if let Some(path) = dialog.pick_file() {
                    self.convert_python_cmd = path.display().to_string();
                }
            }
        });
        let override_cmd = self.convert_python_cmd.trim();
        if !override_cmd.is_empty() {
            ui.label(format!("Python override: {override_cmd}"));
        } else if let Some(env_py) = Self::python_from_env() {
            ui.label(format!("Env Python: {env_py}"));
        } else {
            ui.label("Env Python: none (auto-detects from PATH)");
        }
        if let Some(exe) = self.convert_last_python_exe.as_ref() {
            ui.label(format!("Last used: {exe}"));
        }
        ui.checkbox(
            &mut self.convert_include_expr,
            "Include gene expression data",
        );
        ui.checkbox(
            &mut self.convert_generate_only,
            "Generate .stviz file only - don't load",
        );
        if let Some(status) = self.convert_status.as_ref() {
            ui.label(status);
        }

        ui.separator();
        ui.label("Convert .h5ad");
        let drop_height = 56.0;
        let drop_size = egui::vec2(ui.available_width(), drop_height);
        let (drop_rect, drop_resp) = ui.allocate_exact_size(drop_size, egui::Sense::click());
        let hover_files = ctx.input(|i| !i.raw.hovered_files.is_empty());
        let drop_active = drop_resp.hovered() || hover_files;
        let visuals = ui.visuals();
        let drop_fill = if drop_active {
            visuals.widgets.hovered.bg_fill
        } else {
            visuals.widgets.inactive.bg_fill
        };
        let drop_stroke = if drop_active {
            visuals.widgets.hovered.bg_stroke
        } else {
            visuals.widgets.inactive.bg_stroke
        };
        ui.painter().rect_filled(drop_rect, 6.0, drop_fill);
        ui.painter()
            .rect_stroke(drop_rect, 6.0, drop_stroke, egui::StrokeKind::Inside);
        let drop_label = if self.convert_running {
            "Converting..."
        } else {
            "Drop .h5ad here (or click to pick)"
        };
        ui.painter().text(
            drop_rect.center(),
            egui::Align2::CENTER_CENTER,
            drop_label,
            egui::FontId::proportional(13.0),
            visuals.text_color(),
        );
        let drop_resp = drop_resp.on_hover_text("Drop a .h5ad file to convert it to .stviz.");
        if drop_resp.clicked() && !self.convert_running {
            let dialog = rfd::FileDialog::new()
                .add_filter("h5ad", &["h5ad"])
                .set_title("Select .h5ad")
                .set_directory(self.project_dir.clone());
            if let Some(path) = dialog.pick_file() {
                self.queue_h5ad_convert(&path);
            }
        }
        let dropped_files = ctx.input(|i| i.raw.dropped_files.clone());
        if !dropped_files.is_empty() && !self.convert_running {
            let in_rect = ctx
                .input(|i| i.pointer.hover_pos())
                .map(|pos| drop_rect.contains(pos))
                .unwrap_or(drop_resp.hovered());
            if in_rect {
                let mut handled = false;
                for file in dropped_files {
                    if let Some(path) = file.path {
                        if path.extension().and_then(|ext| ext.to_str()) == Some("h5ad") {
                            self.queue_h5ad_convert(&path);
                            handled = true;
                            break;
                        }
                    }
                }
                if !handled {
                    self.convert_status = Some("Drop a .h5ad file to convert.".to_string());
                }
            }
        }

        ui.separator();
        if ui
            .button("Open .stviz")
            .on_hover_text("Open a .stviz dataset.")
            .clicked()
        {
            if let Err(e) = self.open_dataset_dialog() {
                let msg = format!("Open failed: {e:#}");
                eprintln!("{msg}");
                self.last_error = Some(msg);
            }
        }

        ui.horizontal(|ui| {
            let edit = egui::TextEdit::singleline(&mut self.open_path)
                .desired_width(200.0)
                .hint_text("Path to .stviz");
            ui.add(edit);
            if ui
                .button("Load path")
                .on_hover_text("Load the .stviz file from this path.")
                .clicked()
            {
                if self.open_path.trim().is_empty() {
                    self.last_error = Some("Path is empty.".to_string());
                } else {
                    let path = PathBuf::from(self.open_path.trim());
                    if let Err(e) = self.load_dataset(&path) {
                        let msg = format!("Load failed: {e:#}");
                        eprintln!("{msg}");
                        self.last_error = Some(msg);
                    }
                }
            }
        });

        if let Some(msg) = self.last_error.as_ref() {
            ui.colored_label(egui::Color32::RED, msg);
        }

        ui.separator();

        let Some(ds) = self.dataset.clone() else {
            ui.label("No dataset loaded.");
            return;
        };

        ui.label(format!("Points: {}", ds.meta.n_points));
        if let Some(p) = self.dataset_path.as_ref() {
            ui.label(p.display().to_string());
        }

        ui.separator();
        ui.label("View");

        ui.add(egui::Slider::new(&mut self.ui_scale, 0.75..=2.5).text("UI scale"));
        let mut columns = self.keyframe_columns as u32;
        if ui
            .add(egui::Slider::new(&mut columns, 1..=4).text("Keyframe columns"))
            .changed()
        {
            self.keyframe_columns = (columns as usize).max(1);
        }
        let theme_label = match self.ui_theme {
            UiTheme::Dark => "Dark",
            UiTheme::Light => "Light",
            UiTheme::Slate => "Slate",
            UiTheme::Matrix => "Matrix",
        };
        egui::ComboBox::from_label("Theme")
            .selected_text(theme_label)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.ui_theme, UiTheme::Dark, "Dark");
                ui.selectable_value(&mut self.ui_theme, UiTheme::Light, "Light");
                ui.selectable_value(&mut self.ui_theme, UiTheme::Slate, "Slate");
                ui.selectable_value(&mut self.ui_theme, UiTheme::Matrix, "Matrix");
            });

        ui.horizontal(|ui| {
            ui.label("Background");
            ui.color_edit_button_srgba(&mut self.background_color);
        });

        let norm_changed = ui.checkbox(&mut self.normalize_spaces, "Normalize space scales").changed();
        ui.checkbox(&mut self.show_axes, "Show axes");

        if norm_changed {
            let from_idx = self
                .space_path
                .first()
                .copied()
                .unwrap_or(self.from_space)
                .min(ds.meta.spaces.len().saturating_sub(1));
            let bbox = self.space_bbox_for_view(&ds, from_idx, &ds.meta.spaces[from_idx]);
            let vp = self.last_viewport_points.size();
            let ppp = ctx.pixels_per_point();
            self.camera.fit_bbox(bbox, [vp.x * ppp, vp.y * ppp], 0.9);
        }

        ui.separator();
        if ui
            .button("Fit view to FROM")
            .on_hover_text("Fit the camera to the current FROM space.")
            .clicked()
        {
            let from_idx = self
                .space_path
                .first()
                .copied()
                .unwrap_or(self.from_space)
                .min(ds.meta.spaces.len().saturating_sub(1));
            let bbox = self.space_bbox_for_view(&ds, from_idx, &ds.meta.spaces[from_idx]);
            // Use last viewport size if known
            let vp = self.last_viewport_points.size();
            let ppp = ctx.pixels_per_point();
            self.camera.fit_bbox(bbox, [vp.x * ppp, vp.y * ppp], 0.9);
        }

        ui.separator();
        ui.label("Sample grid (spatial)");
        let mut grid_changed = false;
        if ui.checkbox(&mut self.sample_grid_enabled, "Enable sample grid").changed() {
            grid_changed = true;
        }

        if self.sample_grid_enabled {
            let mut options: Vec<(usize, String)> = Vec::new();
            for (i, o) in ds.meta.obs.iter().enumerate() {
                if let ObsMeta::Categorical { name, .. } = o {
                    options.push((i, name.clone()));
                }
            }

            if options.is_empty() {
                ui.label("No categorical .obs for sample grouping.");
            } else {
                let current = self
                    .sample_grid_obs_idx
                    .and_then(|idx| options.iter().find(|(i, _)| *i == idx))
                    .map(|(_, n)| n.clone())
                    .unwrap_or_else(|| options[0].1.clone());

                let mut new_idx = self.sample_grid_obs_idx.unwrap_or(options[0].0);
                egui::ComboBox::from_label("Group by")
                    .selected_text(current)
                    .show_ui(ui, |ui| {
                        for (i, n) in &options {
                            if ui.selectable_value(&mut new_idx, *i, n).changed() {
                                grid_changed = true;
                            }
                        }
                    });
                self.sample_grid_obs_idx = Some(new_idx);

                if let Some(idx) = self.sample_grid_obs_idx {
                    if let Ok((_name, _labels, categories, _pal)) = ds.obs_categorical(idx) {
                        if categories.len() > MAX_GRID_CATEGORIES {
                            ui.colored_label(
                                egui::Color32::YELLOW,
                                format!(
                                    "Too many categories for grid ({}). Choose a smaller field (e.g. sample).",
                                    categories.len()
                                ),
                            );
                        } else {
                            ui.label(format!("Samples: {}", categories.len()));
                        }
                    }
                }
            }

            let mut space_idx = self.sample_grid_space_idx.unwrap_or(0);
            egui::ComboBox::from_label("Grid space")
                .selected_text(ds.meta.spaces.get(space_idx).map(|s| s.name.as_str()).unwrap_or("?"))
                .show_ui(ui, |ui| {
                    for (i, s) in ds.meta.spaces.iter().enumerate() {
                        if ui.selectable_value(&mut space_idx, i, &s.name).changed() {
                            grid_changed = true;
                        }
                    }
                });
            self.sample_grid_space_idx = Some(space_idx);

            if ui
                .checkbox(&mut self.sample_grid_use_filter, "Use filter selection")
                .changed()
            {
                grid_changed = true;
            }
            if ui
                .add(egui::Slider::new(&mut self.sample_grid_padding, 0.0..=0.5).text("padding"))
                .changed()
            {
                grid_changed = true;
            }
        }

        if grid_changed {
            self.mark_grid_dirty();
        }

        ui.separator();
        ui.label("Color");

        ui.horizontal(|ui| {
            if ui
                .selectable_value(&mut self.color_mode, ColorMode::Categorical, "Categorical")
                .changed()
            {
                self.legend_range = None;
            }
            if ui
                .selectable_value(&mut self.color_mode, ColorMode::Continuous, "Continuous")
                .changed()
            {
                self.legend_range = None;
            }
            if ui
                .selectable_value(&mut self.color_mode, ColorMode::Gene, "Gene")
                .changed()
            {
                self.legend_range = None;
            }
        });

        match self.color_mode {
            ColorMode::Categorical => {
                // choose categorical obs
                let mut options: Vec<(usize, String)> = Vec::new();
                for (i, o) in ds.meta.obs.iter().enumerate() {
                    if let ObsMeta::Categorical { name, .. } = o {
                        options.push((i, name.clone()));
                    }
                }

                if options.is_empty() {
                    ui.label("No categorical .obs exported.");
                } else {
                    let current = options
                        .iter()
                        .find(|(i, _)| *i == self.active_obs_idx)
                        .map(|(_, n)| n.clone())
                        .unwrap_or_else(|| options[0].1.clone());

                    let mut changed = false;
                    egui::ComboBox::from_label("Obs (cat)")
                        .selected_text(current)
                        .show_ui(ui, |ui| {
                            for (i, n) in options {
                                if ui.selectable_value(&mut self.active_obs_idx, i, n).changed() {
                                    changed = true;
                                }
                            }
                        });
                    if changed {
                        self.load_filter_state(&ds);
                    }

                    if ui
                        .button("Apply categorical")
                        .on_hover_text("Apply categorical colors without enabling filters.")
                        .clicked()
                    {
                        if let Err(e) = self.recompute_colors_and_filters() {
                            eprintln!("color/filter: {e:#}");
                        }
                    }

                    ui.separator();
                    ui.label("Filter categories");

                    if let Ok((_name, _labels, categories, pal_opt)) = ds.obs_categorical(self.active_obs_idx) {
                        let too_many = categories.len() > MAX_FILTER_CATEGORIES;
                        if !too_many && self.enabled_categories.len() != categories.len() {
                            self.enabled_categories = vec![true; categories.len()];
                        }
                        let palette: Vec<u32> = if too_many {
                            categorical_palette(256)
                        } else if let Some(p) = pal_opt {
                            p.to_vec()
                        } else {
                            categorical_palette(categories.len())
                        };
                        if too_many {
                            ui.colored_label(
                                egui::Color32::YELLOW,
                                format!(
                                    "Too many categories ({}). Filtering disabled for this field.",
                                    categories.len()
                                ),
                            );
                        } else {
                            egui::ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
                                for (i, c) in categories.iter().enumerate() {
                                    let color = palette
                                        .get(i)
                                        .copied()
                                        .unwrap_or(pack_rgba8(200, 200, 200, 255));
                                    let color = color32_from_packed(color);
                                    ui.horizontal(|ui| {
                                        ui.checkbox(&mut self.enabled_categories[i], "");
                                        let (rect, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                                        ui.painter().rect_filled(rect, 2.0, color);
                                        ui.label(c);
                                    });
                                }
                            });
                        }

                        ui.horizontal(|ui| {
                            if !too_many {
                                if ui.button("All").on_hover_text("Enable all categories.").clicked() {
                                    for v in &mut self.enabled_categories {
                                        *v = true;
                                    }
                                }
                                if ui.button("None").on_hover_text("Disable all categories.").clicked() {
                                    for v in &mut self.enabled_categories {
                                        *v = false;
                                    }
                                }
                                let filter_on = self.active_filters.contains(&self.active_obs_idx);
                                if ui
                                    .button(if filter_on { "Disable filter" } else { "Enable filter" })
                                    .on_hover_text("Toggle filtering by the selected categories.")
                                    .clicked()
                                {
                                    if filter_on {
                                        self.active_filters.remove(&self.active_obs_idx);
                                    } else {
                                        self.category_state
                                            .insert(self.active_obs_idx, self.enabled_categories.clone());
                                        self.active_filters.insert(self.active_obs_idx);
                                    }
                                    let _ = self.recompute_draw_indices_with_filters();
                                }
                                if ui
                                    .button("Apply filter")
                                    .on_hover_text("Apply the current category selection as a filter.")
                                    .clicked()
                                {
                                    self.category_state
                                        .insert(self.active_obs_idx, self.enabled_categories.clone());
                                    self.active_filters.insert(self.active_obs_idx);
                                    if let Err(e) = self.recompute_draw_indices_with_filters() {
                                        eprintln!("filter: {e:#}");
                                    }
                                }
                            }
                        });

                        if !too_many && !self.active_filters.is_empty() {
                            ui.separator();
                            ui.label("Active filters");
                            for obs_idx in self.active_filters.iter().copied().collect::<Vec<_>>() {
                                let label = obs_name(&ds, obs_idx);
                                ui.label(label);
                            }
                            if ui
                                .button("Clear all filters")
                                .on_hover_text("Disable all active filters.")
                                .clicked()
                            {
                                self.active_filters.clear();
                                let _ = self.recompute_draw_indices_with_filters();
                            }
                        }
                    }
                }
            }
            ColorMode::Continuous => {
                let mut options: Vec<(usize, String)> = Vec::new();
                for (i, o) in ds.meta.obs.iter().enumerate() {
                    if let ObsMeta::Continuous { name, .. } = o {
                        options.push((i, name.clone()));
                    }
                }
                if options.is_empty() {
                    ui.label("No continuous .obs exported.");
                } else {
                    let current = options
                        .iter()
                        .find(|(i, _)| *i == self.active_obs_idx)
                        .map(|(_, n)| n.clone())
                        .unwrap_or_else(|| options[0].1.clone());

                    egui::ComboBox::from_label("Obs (cont)")
                        .selected_text(current)
                        .show_ui(ui, |ui| {
                            for (i, n) in options {
                                ui.selectable_value(&mut self.active_obs_idx, i, n);
                            }
                        });

                    if ui
                        .button("Apply continuous")
                        .on_hover_text("Apply continuous colors.")
                        .clicked()
                    {
                        if let Err(e) = self.recompute_colors_and_filters() {
                            eprintln!("color: {e:#}");
                        }
                    }
                }
            }
            ColorMode::Gene => {
                ui.label("Gene coloring (requires --include-expr in export).");
                if let Some(expr) = ds.meta.expr.as_ref() {
                    let mut apply_gene: Option<String> = None;
                    ui.horizontal(|ui| {
                        ui.label("Gene");
                        if ui.text_edit_singleline(&mut self.gene_query).changed() {
                            self.gene_selected = None;
                        }
                    });
                    if let Some(selected) = self.gene_selected.as_ref() {
                        ui.label(format!("Selected: {selected}"));
                    }
                    let needle = self.gene_query.trim().to_ascii_lowercase();
                    let mut shown = 0usize;
                    let mut total = 0usize;
                    let limit = 120usize;
                    egui::ScrollArea::vertical().max_height(180.0).show(ui, |ui| {
                        for name in &expr.var_names {
                            let is_match = if needle.is_empty() {
                                true
                            } else {
                                name.to_ascii_lowercase().contains(&needle)
                            };
                            if !is_match {
                                continue;
                            }
                            total += 1;
                            if shown >= limit {
                                continue;
                            }
                            let is_selected = self
                                .gene_selected
                                .as_ref()
                                .map(|g| g == name)
                                .unwrap_or(false);
                            if ui.selectable_label(is_selected, name).clicked() {
                                self.gene_selected = Some(name.clone());
                                apply_gene = Some(name.clone());
                            }
                            shown += 1;
                        }
                    });
                    if total == 0 {
                        ui.label("No matches.");
                    } else if total > limit {
                        ui.label(format!("Showing {limit} of {total} matches."));
                    }

                    if let Some(gene) = apply_gene {
                        if let Some(gid) = ds.find_gene(&gene) {
                            if let Ok(vec) = ds.gene_vector_csc(gid) {
                                let mut vmax = 0.0f32;
                                for &v in &vec {
                                    if v.is_finite() {
                                        vmax = vmax.max(v);
                                    }
                                }
                                let vmin = 0.0;
                                let vmax = vmax.max(1e-6);
                                let colors = gradient_map(&vec, vmin, vmax, &colorous::VIRIDIS);
                                self.colors_rgba8 = Arc::new(colors);
                                self.colors_id = self.next_color_id();
                                self.legend_range = Some(LegendRange {
                                    label: format!("Gene: {gene}"),
                                    min: vmin,
                                    max: vmax,
                                });
                                let _ = self.recompute_draw_indices_no_filter();
                            }
                        }
                    }
                } else {
                    ui.label("No gene expression data in this file.");
                }

                if ui
                    .button("Apply gene")
                    .on_hover_text("Color by the gene expression vector.")
                    .clicked()
                {
                    let g = self
                        .gene_selected
                        .clone()
                        .unwrap_or_else(|| self.gene_query.trim().to_string());
                    if g.is_empty() {
                        return;
                    }
                    match ds.find_gene(&g) {
                        None => {
                            eprintln!("gene not found: {g}");
                        }
                        Some(gid) => match ds.gene_vector_csc(gid) {
                            Ok(vec) => {
                                // map with VIRIDIS using robust max
                                let mut vmax = 0.0f32;
                                for &v in &vec {
                                    if v.is_finite() {
                                        vmax = vmax.max(v);
                                    }
                                }
                                let vmin = 0.0;
                                let vmax = vmax.max(1e-6);
                                let colors = gradient_map(&vec, vmin, vmax, &colorous::VIRIDIS);
                                self.colors_rgba8 = Arc::new(colors);
                                self.colors_id = self.next_color_id();
                                self.legend_range = Some(LegendRange {
                                    label: format!("Gene: {g}"),
                                    min: vmin,
                                    max: vmax,
                                });
                                let _ = self.recompute_draw_indices_no_filter();
                            }
                            Err(e) => eprintln!("gene vector: {e:#}"),
                        },
                    }
                }
            }
        }

        if let Some(legend) = self.legend_range.as_ref() {
            ui.separator();
            ui.label("Legend");
            draw_gradient_legend(ui, &legend.label, legend.min, legend.max);
        }

        ui.separator();
        ui.label("Rendering");
        ui.horizontal(|ui| {
            ui.label("point radius");
            let mut slider_value = self.point_radius_px.clamp(0.5, 2.0);
            let slider = egui::Slider::new(&mut slider_value, 0.5..=2.0).show_value(false);
            if ui.add(slider).changed() {
                self.point_radius_px = slider_value;
            }
            let mut manual_value = self.point_radius_px;
            if ui
                .add(
                    egui::DragValue::new(&mut manual_value)
                        .speed(0.05)
                        .range(0.1..=5.0)
                        .suffix(" px"),
                )
                .changed()
            {
                self.point_radius_px = manual_value;
            }
        });
        if ds.meta.n_points > 0 {
            let max_cap = ds.meta.n_points as usize;
            let mut slider = egui::Slider::new(&mut self.max_draw_points, 0..=max_cap).text("max draw (0 = all)");
            slider = slider.logarithmic(true).smallest_positive(1.0);
            if ui.add(slider).changed() {
                self.apply_downsample();
            }
            let draw_count = if self.max_draw_points == 0 {
                self.base_indices.len()
            } else {
                self.draw_indices.len()
            };
            ui.label(format!("Drawn: {} / {}", draw_count, ds.meta.n_points));
        }

        if ui.button("Shuffle draw order").clicked() {
            if let Some(ds) = self.dataset.as_ref() {
                if self.base_indices.is_empty() {
                    self.base_indices = (0..ds.meta.n_points as u32).collect();
                }
                let mut rng = rand::thread_rng();
                self.base_indices.shuffle(&mut rng);
                self.apply_downsample();
            }
        }

        ui.checkbox(&mut self.fast_render, "Fast render (square points)");
        ui.checkbox(&mut self.opaque_points, "Opaque points (no blending)");
        ui.checkbox(&mut self.show_stats, "Show stats");

        ui.separator();
        ui.label("Output");
        if ui.button("Screenshot").clicked() {
            let ts = chrono_like_timestamp();
            let path = self.screenshot_dir.join(format!("stviz-animate_screenshot_{ts}.png"));
            let crop = Self::viewport_crop_px(self.last_viewport_points, ctx.pixels_per_point());
            self.request_screenshot(ctx, path, crop);
        }

        ui.horizontal(|ui| {
            if ui.button(if self.recording { "Stop recording" } else { "Start recording" }).clicked() {
                self.recording = !self.recording;
                if self.recording {
                    self.frame_counter = 0;
                    let _ = std::fs::create_dir_all(&self.record_dir);
                }
            }
        });
        ui.label(format!("Record dir: {}", self.record_dir.display()));

        ui.separator();
        ui.label("Export loop");
        ui.add(egui::DragValue::new(&mut self.export_fps).range(1..=240).prefix("fps "));
        ui.horizontal(|ui| {
            ui.label("Output");
            ui.text_edit_singleline(&mut self.export_name);
        });
        ui.checkbox(&mut self.export_run_ffmpeg, "Run ffmpeg if available");
        if ui
            .button(if self.exporting_loop { "Exporting..." } else { "Export loop video" })
            .clicked()
        {
            self.start_export_loop();
        }
        if let Some(status) = self.export_status.as_ref() {
            ui.label(status);
        }
    }

    fn visuals_for_theme(theme: UiTheme) -> egui::Visuals {
        match theme {
            UiTheme::Dark => egui::Visuals::dark(),
            UiTheme::Light => egui::Visuals::light(),
            UiTheme::Slate => {
                let mut visuals = egui::Visuals::dark();
                visuals.panel_fill = egui::Color32::from_rgb(22, 26, 32);
                visuals.window_fill = egui::Color32::from_rgb(22, 26, 32);
                visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(32, 37, 44);
                visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(44, 52, 62);
                visuals.widgets.active.bg_fill = egui::Color32::from_rgb(60, 70, 84);
                visuals.selection.bg_fill = egui::Color32::from_rgb(70, 110, 160);
                visuals.selection.stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(150, 190, 235));
                visuals
            }
            UiTheme::Matrix => {
                let mut visuals = egui::Visuals::dark();
                visuals.panel_fill = egui::Color32::from_rgb(6, 10, 8);
                visuals.window_fill = egui::Color32::from_rgb(6, 10, 8);
                visuals.override_text_color = Some(egui::Color32::from_rgb(130, 255, 150));
                visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(10, 18, 12);
                visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(14, 28, 18);
                visuals.widgets.active.bg_fill = egui::Color32::from_rgb(18, 36, 22);
                visuals.widgets.inactive.fg_stroke =
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(80, 200, 120));
                visuals.widgets.hovered.fg_stroke =
                    egui::Stroke::new(1.2, egui::Color32::from_rgb(120, 255, 160));
                visuals.selection.bg_fill = egui::Color32::from_rgb(0, 120, 60);
                visuals.selection.stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(120, 255, 170));
                visuals.hyperlink_color = egui::Color32::from_rgb(90, 220, 120);
                visuals
            }
        }
    }

    fn ui_viewport(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());
        self.last_viewport_points = rect;
        ui.painter().rect_filled(rect, 0.0, self.background_color);

        // Interactions
        if response.dragged() {
            let delta = response.drag_delta();
            let ppp = ctx.pixels_per_point();
            self.camera.pan_by_pixels([delta.x * ppp, delta.y * ppp]);
        }

        if response.hovered() {
            let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.0 {
                let zoom_factor = (1.0 + scroll * 0.0015).clamp(0.8, 1.25);
                let mouse = ctx.input(|i| i.pointer.hover_pos()).unwrap_or(rect.center());
                let ppp = ctx.pixels_per_point();
                let local = mouse - rect.min;
                self.camera.zoom_at_viewport_pixel(
                    [local.x * ppp, local.y * ppp],
                    [rect.width() * ppp, rect.height() * ppp],
                    zoom_factor,
                );
            }
        }

        if self.exporting_loop && self.export_total_frames > 0 {
            let phase = self.export_frame_index as f32 / self.export_total_frames as f32;
            self.t = export_phase_to_t(phase, self.playback_mode, self.play_direction);
        }

        if self.show_stats {
            let dt = ctx.input(|i| i.unstable_dt);
            if dt > 0.0 {
                let ms = dt * 1000.0;
                if self.frame_ms_avg <= 0.0 {
                    self.frame_ms_avg = ms;
                } else {
                    self.frame_ms_avg = self.frame_ms_avg * 0.9 + ms * 0.1;
                }
            }
            let fps = if self.frame_ms_avg > 0.0 {
                1000.0 / self.frame_ms_avg
            } else {
                0.0
            };
            let drawn = self.draw_indices.len();
            let label = format!(
                "{} | {:.1} fps | {:.1} ms | drawn {}",
                self.adapter_label,
                fps,
                self.frame_ms_avg,
                drawn
            );
            ui.painter().text(
                rect.right_top() + egui::vec2(-6.0, 6.0),
                egui::Align2::RIGHT_TOP,
                label,
                egui::FontId::proportional(12.0),
                contrast_color(self.background_color),
            );
        }

        // Update uniforms for this viewport
        let ppp = ctx.pixels_per_point();
        let viewport_px = [rect.width() * ppp, rect.height() * ppp];

        let ds_opt = self.dataset.clone();
        let mut view_camera = self.camera;
        if self.exporting_loop {
            if self.export_camera.is_none() {
                if let Some(ds) = ds_opt.as_ref() {
                    if let Some(bbox) = self.export_fit_bbox(ds) {
                        let mut cam = Camera2D::default();
                        cam.fit_bbox(bbox, viewport_px, 0.98);
                        self.export_camera = Some(cam);
                    }
                }
            }
            if let Some(cam) = self.export_camera {
                view_camera = cam;
            }
        }
        let (active_from, active_to, color_from, color_to, segment_t) = if let Some(ds) = ds_opt.as_ref() {
            self.current_segment(ds)
        } else {
            (0, 0, ColorKey::Current, ColorKey::Current, self.t.clamp(0.0, 1.0))
        };

        let mut from_override: Option<Arc<Vec<f32>>> = None;
        let mut to_override: Option<Arc<Vec<f32>>> = None;
        let mut from_override_id = 0u64;
        let mut to_override_id = 0u64;

        let (from_center, from_scale, to_center, to_scale) = if let Some(ds) = ds_opt.as_ref() {
            if let Some((pos, _bbox)) = self.grid_positions_for(ds, active_from) {
                from_override = Some(pos);
                from_override_id = self.grid_version ^ ((active_from as u64) << 32);
            }
            if let Some((pos, _bbox)) = self.grid_positions_for(ds, active_to) {
                to_override = Some(pos);
                to_override_id = self.grid_version ^ ((active_to as u64) << 32);
            }
            let from = if let Some(space) = ds.meta.spaces.get(active_from) {
                self.space_transform(ds, active_from, space)
            } else {
                ([0.0, 0.0], 1.0)
            };
            let to = if let Some(space) = ds.meta.spaces.get(active_to) {
                self.space_transform(ds, active_to, space)
            } else {
                ([0.0, 0.0], 1.0)
            };
            (from.0, from.1, to.0, to.1)
        } else {
            ([0.0, 0.0], 1.0, [0.0, 0.0], 1.0)
        };

        let t_eased = apply_ease(segment_t, self.ease_mode);
        let mut color_t = 0.0f32;
        let mut colors_from = self.colors_rgba8.clone();
        let mut colors_from_id = self.colors_id;
        let mut colors_to = self.colors_rgba8.clone();
        let mut colors_to_id = self.colors_id;
        let mut legend_from = self.legend_range.clone();

        if let Some(ds) = ds_opt.as_ref() {
            if self.color_path_enabled {
                let (cf, cf_id, cf_legend) = self.colors_for_key(ds, &color_from);
                let (ct, ct_id, _ct_legend) = self.colors_for_key(ds, &color_to);
                colors_from = cf;
                colors_from_id = cf_id;
                colors_to = ct;
                colors_to_id = ct_id;
                if color_from != color_to {
                    color_t = t_eased;
                }
                legend_from = cf_legend;
            }
        }

        if self.color_path_enabled {
            self.legend_range = legend_from.clone();
        }

        let uniforms = Uniforms {
            viewport_px,
            _pad0: [0.0; 2],
            center: view_camera.center,
            _pad1: [0.0; 2],
            pixels_per_unit: view_camera.pixels_per_unit,
            t: t_eased,
            point_radius_px: self.point_radius_px * ppp, // scale with dpi so it looks consistent
            mask_mode: if self.fast_render { 0.0 } else { 1.0 },
            color_t,
            _pad2: 0.0,
            _pad2b: [0.0; 2],
            from_center,
            from_scale,
            _pad3: 0.0,
            to_center,
            to_scale,
            _pad4: 0.0,
        };

        // Push render params into shared state
        {
            let mut p = self.shared.params.lock();
            p.dataset = self.dataset.clone();
            p.dataset_id = self.dataset_id;
            p.from_space = active_from as u32;
            p.to_space = active_to as u32;
            p.colors_id = colors_from_id;
            p.colors_rgba8 = colors_from;
            p.colors_to_id = colors_to_id;
            p.colors_to_rgba8 = colors_to.clone();
            p.indices_id = self.indices_id;
            p.draw_indices = self.draw_indices.clone();
            p.uniforms = uniforms;
            p.use_opaque = self.opaque_points;
            p.from_override = from_override.clone();
            p.to_override = to_override.clone();
            p.from_override_id = from_override_id;
            p.to_override_id = to_override_id;
        }

        // Submit paint callback
        let cb = PointCloudCallback {
            shared: self.shared.clone(),
        };
        let paint_cb = egui_wgpu::Callback::new_paint_callback(rect, cb);
        ui.painter().add(paint_cb);

        self.draw_view_overlays(ui, rect);

        // Recording: request screenshots every frame while recording
        if self.recording {
            let path = self.record_dir.join(format!("frame_{:06}.png", self.frame_counter));
            self.frame_counter += 1;
            let crop = Self::viewport_crop_px_even(rect, ppp);
            self.request_screenshot(ctx, path, crop);
        }

        if self.exporting_loop && self.export_total_frames > 0 {
            let path = self.export_dir.join(format!("frame_{:06}.png", self.export_frame_index));
            let crop = Self::viewport_crop_px_even(rect, ppp);
            self.request_screenshot(ctx, path, crop);
            self.export_frame_index += 1;
            if self.export_frame_index >= self.export_total_frames {
                self.exporting_loop = false;
                self.export_camera = None;
                self.finish_export_loop();
            }
        }

        // Keep repainting while playing/recording
        if self.playing || self.recording || self.exporting_loop {
            ctx.request_repaint();
        }
    }

    fn effective_bbox(&mut self, ds: &Dataset, space_idx: usize, space: &crate::data::SpaceMeta) -> [f32; 4] {
        if self.sample_grid_enabled && self.sample_grid_space_idx == Some(space_idx) {
            if let Some((_pos, bbox)) = self.grid_positions_for(ds, space_idx) {
                return bbox;
            }
        }
        space.bbox
    }

    fn export_fit_bbox(&mut self, ds: &Dataset) -> Option<[f32; 4]> {
        let mut space_indices: Vec<usize> = Vec::new();
        if self.transition_mode == TransitionMode::Path && !self.space_path.is_empty() {
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
        let mut best = None;
        let mut best_area = -1.0f32;
        for idx in space_indices {
            if let Some(space) = ds.meta.spaces.get(idx) {
                let bbox = self.space_bbox_for_view(ds, idx, space);
                let w = (bbox[2] - bbox[0]).max(1e-6);
                let h = (bbox[3] - bbox[1]).max(1e-6);
                let area = w * h;
                if area > best_area {
                    best_area = area;
                    best = Some(bbox);
                }
            }
        }
        best
    }

    fn space_transform(&mut self, ds: &Dataset, space_idx: usize, space: &crate::data::SpaceMeta) -> ([f32; 2], f32) {
        if !self.normalize_spaces {
            return ([0.0, 0.0], 1.0);
        }
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

    fn space_bbox_for_view(&mut self, ds: &Dataset, space_idx: usize, space: &crate::data::SpaceMeta) -> [f32; 4] {
        let bbox = self.effective_bbox(ds, space_idx, space);
        if !self.normalize_spaces {
            return bbox;
        }
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

    fn grid_positions_for(&mut self, ds: &Dataset, space_idx: usize) -> Option<(Arc<Vec<f32>>, [f32; 4])> {
        if !self.sample_grid_enabled {
            return None;
        }
        if self.sample_grid_space_idx != Some(space_idx) {
            return None;
        }
        let obs_idx = self.sample_grid_obs_idx?;
        let key = GridCacheKey {
            dataset_id: self.dataset_id,
            space_idx,
            obs_idx,
            use_filter: self.sample_grid_use_filter,
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

        let mut selected: Vec<usize> = if self.sample_grid_use_filter {
            if let Some(state) = self.category_state.get(&obs_idx) {
                state.iter().enumerate().filter_map(|(i, v)| if *v { Some(i) } else { None }).collect()
            } else {
                (0..categories.len()).collect()
            }
        } else {
            (0..categories.len()).collect()
        };
        if selected.is_empty() {
            selected = (0..categories.len()).collect();
        }

        let n_sel = selected.len().max(1);
        let mut map = vec![-1i32; categories.len()];
        for (i, &cat) in selected.iter().enumerate() {
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
        let rows = (n_sel + cols - 1) / cols;
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
        let mut bbox = [f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
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

        let positions = Arc::new(out);
        self.grid_cache = Some(GridCache { key, positions: positions.clone(), bbox });
        Some((positions, bbox))
    }

    fn ensure_color_path_len(&mut self, len: usize) {
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

    fn ensure_space_path_len(&mut self, ds: &Dataset, len: usize) {
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
        if self.key_times.len() != self.space_path.len() {
            self.ensure_key_times_len(self.space_path.len());
        }
    }

    fn ensure_key_times_len(&mut self, len: usize) {
        if len == 0 {
            self.key_times.clear();
            return;
        }
        if self.key_times.len() == len {
            if let Some(first) = self.key_times.first_mut() {
                *first = 0.0;
            }
            if let Some(last) = self.key_times.last_mut() {
                *last = 1.0;
            }
            return;
        }
        if len == 1 {
            self.key_times = vec![0.0];
            return;
        }
        self.key_times = (0..len)
            .map(|i| i as f32 / (len as f32 - 1.0))
            .collect();
    }

    fn segment_for_t(&self) -> (usize, f32) {
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

    fn timeline_segment_index(&self) -> usize {
        let len = self.key_times.len();
        if len < 2 {
            return 0;
        }
        let t = self.t.clamp(0.0, 1.0);
        let mut seg_idx = 0usize;
        while seg_idx + 1 < len && t > self.key_times[seg_idx + 1] {
            seg_idx += 1;
        }
        seg_idx.min(len.saturating_sub(2))
    }

    fn compute_colors_for_key(ds: &Dataset, key: &ColorKey) -> Option<(Vec<u32>, Option<LegendRange>)> {
        match key {
            ColorKey::Current => None,
            ColorKey::Categorical(idx) => {
                let (_name, labels, categories, pal_opt) = ds.obs_categorical(*idx).ok()?;
                let too_many = categories.len() > MAX_FILTER_CATEGORIES;
                let pal: Vec<u32> = if too_many {
                    categorical_palette(256)
                } else if let Some(p) = pal_opt {
                    p.to_vec()
                } else {
                    categorical_palette(categories.len())
                };
                let mut colors = Vec::with_capacity(labels.len());
                for &lab in labels {
                    let li = lab as usize;
                    let c = if too_many {
                        let idx = if pal.is_empty() { 0 } else { li % pal.len() };
                        pal.get(idx).copied().unwrap_or(pack_rgba8(200, 200, 200, 255))
                    } else {
                        pal.get(li).copied().unwrap_or(pack_rgba8(200, 200, 200, 255))
                    };
                    colors.push(c);
                }
                Some((colors, None))
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
                Some((colors, legend))
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
                Some((colors, legend))
            }
        }
    }

    fn colors_for_key(&mut self, ds: &Dataset, key: &ColorKey) -> (Arc<Vec<u32>>, u64, Option<LegendRange>) {
        if *key == ColorKey::Current {
            return (self.colors_rgba8.clone(), self.colors_id, self.legend_range.clone());
        }
        if let Some(entry) = self.color_cache.get(key) {
            return (entry.colors.clone(), entry.id, entry.legend.clone());
        }
        if let Some((colors, legend)) = Self::compute_colors_for_key(ds, key) {
            let entry = ColorCacheEntry {
                colors: Arc::new(colors),
                id: self.next_color_id(),
                legend,
            };
            self.color_cache.insert(key.clone(), entry.clone());
            return (entry.colors, entry.id, entry.legend);
        }
        (self.colors_rgba8.clone(), self.colors_id, self.legend_range.clone())
    }

    fn current_segment(&self, ds: &Dataset) -> (usize, usize, ColorKey, ColorKey, f32) {
        let n_spaces = ds.meta.spaces.len();
        let clamp_idx = |idx: usize| idx.min(n_spaces.saturating_sub(1));
        match self.transition_mode {
            TransitionMode::Single => {
                let mut from = clamp_idx(self.from_space);
                let mut to = clamp_idx(self.to_space);
                if self.space_path.len() >= 2 {
                    from = clamp_idx(self.space_path[0]);
                    to = clamp_idx(self.space_path[1]);
                }
                let (color_from, color_to) = if self.color_path_enabled && self.color_path.len() >= 2 {
                    let cf = self.color_path.get(0).cloned().unwrap_or(ColorKey::Current);
                    let ct = self.color_path.get(1).cloned().unwrap_or(ColorKey::Current);
                    (cf, ct)
                } else {
                    (ColorKey::Current, ColorKey::Current)
                };
                (from, to, color_from, color_to, self.t.clamp(0.0, 1.0))
            }
            TransitionMode::Path => {
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
                (from, to, color_from, color_to, local_t.clamp(0.0, 1.0))
            }
        }
    }

    fn draw_view_overlays(&self, ui: &egui::Ui, rect: egui::Rect) {
        if !self.show_axes {
            return;
        }

        let painter = ui.painter();
        let margin = 10.0;
        let origin = rect.left_bottom() + egui::vec2(margin, -margin);
        let axis_len = 36.0;

        if self.show_axes {
            let x_color = egui::Color32::from_rgb(220, 90, 90);
            let y_color = egui::Color32::from_rgb(90, 220, 140);

            painter.line_segment([origin, origin + egui::vec2(axis_len, 0.0)], (1.2, x_color));
            painter.line_segment([origin, origin + egui::vec2(0.0, -axis_len)], (1.2, y_color));
            painter.text(
                origin + egui::vec2(axis_len + 4.0, 0.0),
                egui::Align2::LEFT_CENTER,
                "X",
                egui::FontId::proportional(12.0),
                x_color,
            );
            painter.text(
                origin + egui::vec2(0.0, -axis_len - 2.0),
                egui::Align2::CENTER_BOTTOM,
                "Y",
                egui::FontId::proportional(12.0),
                y_color,
            );
        }

    }

    fn viewport_crop_px(rect: egui::Rect, ppp: f32) -> Option<[u32; 4]> {
        if rect.is_positive() == false {
            return None;
        }
        let x0 = (rect.min.x * ppp).floor().max(0.0);
        let y0 = (rect.min.y * ppp).floor().max(0.0);
        let x1 = (rect.max.x * ppp).ceil().max(x0 + 1.0);
        let y1 = (rect.max.y * ppp).ceil().max(y0 + 1.0);
        let w = (x1 - x0).max(1.0) as u32;
        let h = (y1 - y0).max(1.0) as u32;
        Some([x0 as u32, y0 as u32, w, h])
    }

    fn viewport_crop_px_even(rect: egui::Rect, ppp: f32) -> Option<[u32; 4]> {
        let mut crop = Self::viewport_crop_px(rect, ppp)?;
        if crop[2] > 1 && crop[2] % 2 == 1 {
            crop[2] -= 1;
        }
        if crop[3] > 1 && crop[3] % 2 == 1 {
            crop[3] -= 1;
        }
        if crop[2] < 2 || crop[3] < 2 {
            return None;
        }
        Some(crop)
    }

    fn ui_timeline_bar(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let ds = match self.dataset.clone() {
            Some(ds) => ds,
            None => {
                ui.label("Load a .stviz to use the timeline.");
                return;
            }
        };
        let ds_ref = ds.as_ref();

        let desired_len = self.space_path.len().max(2);
        self.ensure_space_path_len(ds_ref, desired_len);
        self.ensure_color_path_len(self.space_path.len());
        if self.key_times.len() != self.space_path.len() {
            self.ensure_key_times_len(self.space_path.len());
        }

        let prev_mode = self.transition_mode;
        ui.horizontal(|ui| {
            if ui
                .button(if self.playing { "Pause" } else { "Play" })
                .on_hover_text("Play or pause the timeline.")
                .clicked()
            {
                self.playing = !self.playing;
            }
            if ui.button("Stop").on_hover_text("Stop and rewind to t = 0.").clicked() {
                self.playing = false;
                self.t = 0.0;
            }
            if ui
                .button("Reverse")
                .on_hover_text("Reverse playback direction.")
                .clicked()
            {
                self.play_direction *= -1.0;
            }
            if ui.button("Step -").on_hover_text("Step back a bit.").clicked() {
                self.t = (self.t - 0.01).clamp(0.0, 1.0);
            }
            if ui.button("Step +").on_hover_text("Step forward a bit.").clicked() {
                self.t = (self.t + 0.01).clamp(0.0, 1.0);
            }

            ui.separator();
            ui.label("Mode");
            ui.selectable_value(&mut self.transition_mode, TransitionMode::Single, "Single");
            ui.selectable_value(&mut self.transition_mode, TransitionMode::Path, "Path");
            ui.separator();
            ui.label("Playback");
            ui.selectable_value(&mut self.playback_mode, PlaybackMode::Once, "Once");
            ui.selectable_value(&mut self.playback_mode, PlaybackMode::Loop, "Loop");
            ui.selectable_value(&mut self.playback_mode, PlaybackMode::PingPong, "Ping-pong");
            ui.separator();
            ui.add(egui::Slider::new(&mut self.speed, 0.05..=2.0).text("speed"));
            ui.separator();
            ui.label("Easing");
            let ease_label = match self.ease_mode {
                EaseMode::Linear => "Linear",
                EaseMode::Smoothstep => "Smoothstep",
                EaseMode::SineInOut => "Sine in-out",
                EaseMode::QuadInOut => "Quad in-out",
            };
            egui::ComboBox::from_id_salt("timeline_easing")
                .selected_text(ease_label)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.ease_mode, EaseMode::Linear, "Linear");
                    ui.selectable_value(&mut self.ease_mode, EaseMode::Smoothstep, "Smoothstep");
                    ui.selectable_value(&mut self.ease_mode, EaseMode::SineInOut, "Sine in-out");
                    ui.selectable_value(&mut self.ease_mode, EaseMode::QuadInOut, "Quad in-out");
                });
        });
        if prev_mode != self.transition_mode {
            if self.transition_mode == TransitionMode::Single {
                self.ensure_space_path_len(ds_ref, 2);
            } else {
                let desired_len = self.space_path.len().max(2);
                self.ensure_space_path_len(ds_ref, desired_len);
            }
            self.ensure_color_path_len(self.space_path.len());
        }

        let timeline_height = 36.0;
        let (rect, response) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), timeline_height),
            egui::Sense::click_and_drag(),
        );
        let painter = ui.painter();
        let line_y = rect.center().y;
        painter.line_segment(
            [egui::pos2(rect.left(), line_y), egui::pos2(rect.right(), line_y)],
            (3.0, egui::Color32::from_gray(100)),
        );

        let key_count = self.space_path.len().max(2);
        let mut marker_handled = false;
        for i in 0..key_count {
            let t = *self.key_times.get(i).unwrap_or(&0.0);
            let x = rect.left() + t * rect.width();
            let pos = egui::pos2(x, line_y);
            let marker_rect = egui::Rect::from_center_size(pos, egui::vec2(14.0, 14.0));
            let marker_id = ui.id().with(("kf_marker", i));
            let marker_resp = ui.interact(marker_rect, marker_id, egui::Sense::click_and_drag());
            if marker_resp.clicked() {
                self.selected_key_idx = Some(i);
                marker_handled = true;
            }
            if marker_resp.dragged() {
                self.selected_key_idx = Some(i);
                marker_handled = true;
                if i > 0 && i + 1 < key_count {
                    if let Some(pos) = marker_resp.interact_pointer_pos() {
                        let mut t = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
                        let min_t = self.key_times[i - 1] + 0.005;
                        let max_t = self.key_times[i + 1] - 0.005;
                        t = t.clamp(min_t, max_t);
                        self.key_times[i] = t;
                    }
                }
            }
            let color = if Some(i) == self.selected_key_idx {
                egui::Color32::from_rgb(255, 210, 90)
            } else {
                egui::Color32::from_gray(170)
            };
            painter.circle_filled(pos, 5.0, color);
        }

        let handle_x = rect.left() + self.t.clamp(0.0, 1.0) * rect.width();
        painter.line_segment(
            [egui::pos2(handle_x, rect.top()), egui::pos2(handle_x, rect.bottom())],
            (2.5, egui::Color32::from_rgb(100, 200, 255)),
        );
        if !marker_handled && (response.dragged() || response.clicked()) {
            if let Some(pos) = response.interact_pointer_pos() {
                let t = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
                self.t = t;
            }
        }

        ui.horizontal(|ui| {
            ui.label("t");
            ui.add(egui::DragValue::new(&mut self.t).speed(0.001).range(0.0..=1.0));
            ui.checkbox(&mut self.color_path_enabled, "Use color keys");
        });

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Keyframes");
            if ui
                .button("Add key")
                .on_hover_text("Insert a keyframe after the current segment.")
                .clicked()
            {
                if self.transition_mode == TransitionMode::Single {
                    self.transition_mode = TransitionMode::Path;
                }
                if self.space_path.len() >= 2 {
                    let seg = self.timeline_segment_index();
                    let insert_at = (seg + 1).min(self.space_path.len() - 1);
                    let space = self.space_path.get(seg).copied().unwrap_or(0);
                    let color = self.color_path.get(seg).cloned().unwrap_or(ColorKey::Current);
                    let t0 = *self.key_times.get(seg).unwrap_or(&0.0);
                    let t1 = *self.key_times.get(seg + 1).unwrap_or(&1.0);
                    let new_t = 0.5 * (t0 + t1);
                    self.space_path.insert(insert_at, space);
                    self.color_path.insert(insert_at, color);
                    if self.key_times.len() >= insert_at {
                        self.key_times.insert(insert_at, new_t);
                    }
                    self.selected_key_idx = Some(insert_at);
                }
            }
            if ui
                .button("Space evenly")
                .on_hover_text("Distribute keyframes evenly across the timeline.")
                .clicked()
            {
                let len = self.key_times.len();
                if len >= 2 {
                    for (i, t) in self.key_times.iter_mut().enumerate() {
                        *t = i as f32 / (len as f32 - 1.0);
                    }
                }
            }
            let can_remove = self.space_path.len() > 2 && self.selected_key_idx.is_some();
            if ui
                .add_enabled(can_remove, egui::Button::new("Remove key"))
                .on_hover_text("Remove the selected keyframe.")
                .clicked()
            {
                if let Some(idx) = self.selected_key_idx {
                    if self.space_path.len() > 2 {
                        self.space_path.remove(idx);
                        if idx < self.color_path.len() {
                            self.color_path.remove(idx);
                        }
                        if idx < self.key_times.len() {
                            self.key_times.remove(idx);
                        }
                        self.selected_key_idx = None;
                        if let Some(first) = self.key_times.first_mut() {
                            *first = 0.0;
                        }
                        if let Some(last) = self.key_times.last_mut() {
                            *last = 1.0;
                        }
                    }
                }
            }
        });

        let categorical_opts: Vec<(usize, String)> = ds_ref
            .meta
            .obs
            .iter()
            .enumerate()
            .filter_map(|(i, o)| match o {
                ObsMeta::Categorical { name, .. } => Some((i, name.clone())),
                _ => None,
            })
            .collect();
        let continuous_opts: Vec<(usize, String)> = ds_ref
            .meta
            .obs
            .iter()
            .enumerate()
            .filter_map(|(i, o)| match o {
                ObsMeta::Continuous { name, .. } => Some((i, name.clone())),
                _ => None,
            })
            .collect();
        egui::ScrollArea::both()
            .max_height(240.0)
            .auto_shrink([false; 2])
            .show(ui, |ui| {
                let max_idx = ds_ref.meta.spaces.len().saturating_sub(1);
                let key_count = self.space_path.len();
                let mut row_start = 0;
                let columns = self.keyframe_columns.max(1);
                while row_start < key_count {
                    let row_end = (row_start + columns).min(key_count);
                    ui.horizontal(|ui| {
                        for i in row_start..row_end {
                            let mut space_idx = self.space_path[i].min(max_idx);
                            let mut color_key =
                                self.color_path.get(i).cloned().unwrap_or(ColorKey::Current);
                            let mut color_kind = match color_key {
                                ColorKey::Current => KeyColorKind::Current,
                                ColorKey::Categorical(_) => KeyColorKind::Categorical,
                                ColorKey::Continuous(_) => KeyColorKind::Continuous,
                                ColorKey::Gene(_) => KeyColorKind::Gene,
                            };
                            let is_selected = Some(i) == self.selected_key_idx;
                            ui.push_id(i, |ui| {
                                ui.group(|ui| {
                                    let label = format!("Key {i}");
                                    if ui.selectable_label(is_selected, label).clicked() {
                                        self.selected_key_idx = Some(i);
                                    }
                                    ui.label("Space");
                                    egui::ComboBox::from_id_salt(("kf_space", i))
                                        .selected_text(
                                            ds_ref
                                                .meta
                                                .spaces
                                                .get(space_idx)
                                                .map(|s| s.name.as_str())
                                                .unwrap_or("?"),
                                        )
                                        .show_ui(ui, |ui| {
                                            for (j, s) in ds_ref.meta.spaces.iter().enumerate() {
                                                ui.selectable_value(&mut space_idx, j, &s.name);
                                            }
                                        });
                                    ui.label("Color");
                                    let kind_label = match color_kind {
                                        KeyColorKind::Current => "Current",
                                        KeyColorKind::Categorical => "Categorical",
                                        KeyColorKind::Continuous => "Continuous",
                                        KeyColorKind::Gene => "Gene",
                                    };
                                    egui::ComboBox::from_id_salt(("kf_color_kind", i))
                                        .selected_text(kind_label)
                                        .show_ui(ui, |ui| {
                                            ui.selectable_value(
                                                &mut color_kind,
                                                KeyColorKind::Current,
                                                "Current",
                                            );
                                            ui.selectable_value(
                                                &mut color_kind,
                                                KeyColorKind::Categorical,
                                                "Categorical",
                                            );
                                            ui.selectable_value(
                                                &mut color_kind,
                                                KeyColorKind::Continuous,
                                                "Continuous",
                                            );
                                            ui.selectable_value(
                                                &mut color_kind,
                                                KeyColorKind::Gene,
                                                "Gene",
                                            );
                                        });

                                    match color_kind {
                                        KeyColorKind::Current => {
                                            color_key = ColorKey::Current;
                                        }
                                        KeyColorKind::Categorical => {
                                            if categorical_opts.is_empty() {
                                                ui.label("No categorical obs.");
                                                color_key = ColorKey::Current;
                                            } else {
                                                let mut obs_idx = match color_key {
                                                    ColorKey::Categorical(idx) => idx,
                                                    _ => categorical_opts[0].0,
                                                };
                                                egui::ComboBox::from_id_salt(("kf_color_cat", i))
                                                    .selected_text(
                                                        categorical_opts
                                                            .iter()
                                                            .find(|(idx, _)| *idx == obs_idx)
                                                            .map(|(_, name)| name.as_str())
                                                            .unwrap_or("?"),
                                                    )
                                                    .show_ui(ui, |ui| {
                                                        for (idx, name) in &categorical_opts {
                                                            ui.selectable_value(&mut obs_idx, *idx, name);
                                                        }
                                                    });
                                                color_key = ColorKey::Categorical(obs_idx);
                                            }
                                        }
                                        KeyColorKind::Continuous => {
                                            if continuous_opts.is_empty() {
                                                ui.label("No continuous obs.");
                                                color_key = ColorKey::Current;
                                            } else {
                                                let mut obs_idx = match color_key {
                                                    ColorKey::Continuous(idx) => idx,
                                                    _ => continuous_opts[0].0,
                                                };
                                                egui::ComboBox::from_id_salt(("kf_color_cont", i))
                                                    .selected_text(
                                                        continuous_opts
                                                            .iter()
                                                            .find(|(idx, _)| *idx == obs_idx)
                                                            .map(|(_, name)| name.as_str())
                                                            .unwrap_or("?"),
                                                    )
                                                    .show_ui(ui, |ui| {
                                                        for (idx, name) in &continuous_opts {
                                                            ui.selectable_value(&mut obs_idx, *idx, name);
                                                        }
                                                    });
                                                color_key = ColorKey::Continuous(obs_idx);
                                            }
                                        }
                                        KeyColorKind::Gene => {
                                            if let Some(expr) = ds_ref.meta.expr.as_ref() {
                                                let mut gene = match &color_key {
                                                    ColorKey::Gene(name) => name.clone(),
                                                    _ => String::new(),
                                                };
                                                let edit = egui::TextEdit::singleline(&mut gene)
                                                    .hint_text("Exact gene name")
                                                    .desired_width(140.0);
                                                ui.add(edit);
                                                if !gene.trim().is_empty()
                                                    && !expr.var_names.iter().any(|name| name == &gene)
                                                {
                                                    ui.label("No exact match.");
                                                }
                                                color_key = ColorKey::Gene(gene);
                                            } else {
                                                ui.label("No gene data.");
                                                color_key = ColorKey::Current;
                                            }
                                        }
                                    }
                                });
                            });
                            self.space_path[i] = space_idx;
                            if i < self.color_path.len() {
                                self.color_path[i] = color_key;
                            } else {
                                self.color_path.push(color_key);
                            }
                        }
                    });
                    row_start += columns;
                }
            });

        self.ensure_color_path_len(self.space_path.len());
        self.from_space = *self.space_path.first().unwrap_or(&self.from_space);
        self.to_space = *self.space_path.last().unwrap_or(&self.to_space);

        if self.transition_mode == TransitionMode::Single {
            self.ensure_space_path_len(ds_ref, 2);
        }

        if self.playing {
            ctx.request_repaint();
        }
    }

    fn apply_downsample(&mut self) {
        let max_draw = self.max_draw_points;
        if max_draw == 0 || self.base_indices.len() <= max_draw {
            self.draw_indices = Arc::new(self.base_indices.clone());
            self.indices_id = self.indices_id.wrapping_add(1);
            return;
        }

        let step = (self.base_indices.len() as f32 / max_draw as f32).ceil().max(1.0) as usize;
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

    fn load_filter_state(&mut self, ds: &Dataset) {
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

    fn mark_grid_dirty(&mut self) {
        self.grid_version = self.grid_version.wrapping_add(1);
        self.grid_cache = None;
    }
}

impl eframe::App for StvizApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_zoom_factor(self.ui_scale);
        ctx.set_visuals(Self::visuals_for_theme(self.ui_theme));
        self.handle_screenshot_events(ctx);
        self.handle_hotkeys(ctx);
        self.poll_convert_job();
        self.maybe_update_playback(ctx);

        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("GPU point transitions");
            });
        });

        egui::SidePanel::left("left_panel").resizable(true).default_width(320.0).show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
                self.ui_left_panel(ui, ctx);
            });
        });

        egui::TopBottomPanel::bottom("timeline_bar")
            .resizable(true)
            .default_height(self.timeline_height)
            .min_height(120.0)
            .show(ctx, |ui| {
                self.ui_timeline_bar(ui, ctx);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.ui_viewport(ui, ctx);
        });

        if self.convert_running {
            ctx.request_repaint();
        }
    }
}

struct PointCloudCallback {
    shared: Arc<SharedRender>,
}

impl CallbackTrait for PointCloudCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let gpu = if let Some(g) = callback_resources.get_mut::<PointCloudGpu>() {
            g
        } else {
            // Create GPU resources once
            let target_format = self.shared.params.lock().target_format;
            callback_resources.insert(PointCloudGpu::new(device, target_format));
            callback_resources.get_mut::<PointCloudGpu>().unwrap()
        };

        let p = self.shared.params.lock();
        let Some(ds) = p.dataset.as_ref() else {
            return Vec::new();
        };

        // Ensure default buffers exist when dataset just loaded
        if p.colors_rgba8.is_empty() || p.colors_to_rgba8.is_empty() {
            return Vec::new();
        }
        if p.draw_indices.is_empty() {
            return Vec::new();
        }

        let _ = gpu.prepare(
            device,
            queue,
            p.target_format,
            ds,
            p.dataset_id,
            p.from_space,
            p.to_space,
            p.colors_id,
            &p.colors_rgba8,
            p.colors_to_id,
            &p.colors_to_rgba8,
            p.indices_id,
            &p.draw_indices,
            p.from_override.as_deref().map(|v| v.as_slice()),
            p.to_override.as_deref().map(|v| v.as_slice()),
            p.from_override_id,
            p.to_override_id,
            p.uniforms,
        );

        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(gpu) = callback_resources.get::<PointCloudGpu>() else {
            return;
        };
        let use_opaque = self.shared.params.lock().use_opaque;
        gpu.paint(render_pass, use_opaque);
    }
}

// Minimal PNG save for egui::ColorImage
fn save_color_image_png(
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

    let encoder = image::codecs::png::PngEncoder::new(std::fs::File::create(path)?);
    encoder.write_image(&rgba, cw, ch, image::ColorType::Rgba8.into())?;
    Ok(())
}

fn draw_gradient_legend(ui: &mut egui::Ui, label: &str, vmin: f32, vmax: f32) {
    ui.label(label);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(180.0, 14.0), egui::Sense::hover());
    let steps = 32;
    let seg_w = rect.width() / steps as f32;
    for i in 0..steps {
        let t = i as f32 / (steps - 1) as f32;
        let c = colorous::VIRIDIS.eval_continuous(t as f64);
        let color = egui::Color32::from_rgb(c.r, c.g, c.b);
        let x0 = rect.left() + seg_w * i as f32;
        let seg = egui::Rect::from_min_size(egui::pos2(x0, rect.top()), egui::vec2(seg_w + 1.0, rect.height()));
        ui.painter().rect_filled(seg, 0.0, color);
    }
    ui.label(format!("Range: {} .. {}", format_scale_value(vmin), format_scale_value(vmax)));
}

fn apply_ease(t: f32, mode: EaseMode) -> f32 {
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

fn export_phase_to_t(phase: f32, mode: PlaybackMode, direction: f32) -> f32 {
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

fn color32_from_packed(c: u32) -> egui::Color32 {
    let r = (c & 255) as u8;
    let g = ((c >> 8) & 255) as u8;
    let b = ((c >> 16) & 255) as u8;
    let a = ((c >> 24) & 255) as u8;
    egui::Color32::from_rgba_unmultiplied(r, g, b, a)
}

fn contrast_color(bg: egui::Color32) -> egui::Color32 {
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

fn format_scale_value(v: f32) -> String {
    let av = v.abs();
    if av > 0.0 && (av < 0.01 || av >= 1000.0) {
        format!("{v:.2e}")
    } else {
        format!("{v:.3}")
    }
}

fn obs_name(ds: &Dataset, idx: usize) -> String {
    match ds.meta.obs.get(idx) {
        Some(ObsMeta::Categorical { name, .. }) => format!("{name} (cat)"),
        Some(ObsMeta::Continuous { name, .. }) => format!("{name} (cont)"),
        None => format!("obs[{idx}]"),
    }
}

fn find_obs_by_name(ds: &Dataset, needle: &str) -> Option<usize> {
    let needle = needle.to_ascii_lowercase();
    ds.meta
        .obs
        .iter()
        .enumerate()
        .find_map(|(i, o)| match o {
            ObsMeta::Categorical { name, .. } if name.to_ascii_lowercase().contains(&needle) => Some(i),
            ObsMeta::Continuous { name, .. } if name.to_ascii_lowercase().contains(&needle) => Some(i),
            _ => None,
        })
}

fn find_space_by_name(ds: &Dataset, needle: &str) -> Option<usize> {
    let needle = needle.to_ascii_lowercase();
    ds.meta
        .spaces
        .iter()
        .enumerate()
        .find_map(|(i, s)| if s.name.to_ascii_lowercase().contains(&needle) { Some(i) } else { None })
}

fn chrono_like_timestamp() -> String {
    // Avoid adding chrono dependency; generate a simple timestamp-like string.
    use std::time::{SystemTime, UNIX_EPOCH};
    let ms = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis();
    format!("{ms}")
}
