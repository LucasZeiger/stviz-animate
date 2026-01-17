use crate::{
    camera::Camera2D,
    color::{categorical_palette, categorical_palette_named, gradient_map, pack_rgba8},
    data::{Dataset, ObsMeta},
    render::{PointCloudGpu, SharedRender, Uniforms},
};
use anyhow::Context as _;
use eframe::egui;
use egui_wgpu::{wgpu, CallbackTrait};
use rand::{seq::SliceRandom, Rng};
#[cfg(windows)]
use raw_window_handle::HasWindowHandle;
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
    thread,
};

const MAX_FILTER_CATEGORIES: usize = 5000;
const MAX_GRID_CATEGORIES: usize = 512;
const ADVANCED_VIEW_FPS_CAP_HZ: f32 = 60.0;
const MIN_PYTHON_VERSION: (u32, u32) = (3, 8);
const SCREENSHOT_RESOLUTION: [u32; 2] = [3840, 2160];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExportQuality {
    Current,
    FullHd,
    UltraHd,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExportVideoQuality {
    Standard,
    High,
    Ultra,
}

pub struct StvizApp {
    dataset: Option<Arc<Dataset>>,
    dataset_path: Option<PathBuf>,
    dataset_id: u64,

    camera: Camera2D,

    from_space: usize,
    to_space: usize,
    space_path: Vec<usize>,

    // Color mode
    color_mode: ColorMode,
    active_obs_idx: usize, // index into meta.obs
    categorical_palette: CategoricalPalette,
    category_overrides: HashMap<usize, Vec<Option<u32>>>,
    category_palette_cache: Option<CategoryPaletteCache>,

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
    dragging_key_idx: Option<usize>,
    key_collapsed: Vec<bool>,

    // Playback
    playing: bool,
    t: f32,
    speed: f32, // units per second
    play_direction: f32,
    play_last_time: Option<f64>,
    playback_mode: PlaybackMode,
    ease_mode: EaseMode,
    point_radius_px: f32,
    max_draw_points: usize,
    fast_render: bool,

    // Render plumbing
    shared: Arc<SharedRender>,
    render_state: Option<egui_wgpu::RenderState>,
    offscreen_gpu: Option<PointCloudGpu>,
    offscreen_gpu_msaa: Option<PointCloudGpu>,
    colors_id: u64,
    colors_rgba8: Arc<Vec<u32>>,
    colors_opaque: bool,
    indices_id: u64,
    base_indices: Vec<u32>,
    draw_indices: Arc<Vec<u32>>,

    // Screenshot / export
    project_dir: PathBuf,
    output_dir: PathBuf,
    screenshot_dir: PathBuf,
    exporting_loop: bool,
    export_fps: u32,
    export_duration_sec: f32,
    export_quality: ExportQuality,
    export_resolution: Option<[u32; 2]>,
    export_video_quality: ExportVideoQuality,
    export_dir: PathBuf,
    export_name: String,
    export_output_path: Option<PathBuf>,
    export_total_frames: u32,
    export_frame_index: u32,
    export_status: Option<String>,
    export_run_ffmpeg: bool,
    export_keep_frames: bool,
    export_camera: Option<Camera2D>,
    export_pending_frames: u32,
    export_finishing: bool,
    export_cancelled: bool,
    ffmpeg_available: bool,
    ffmpeg_path: Option<PathBuf>,
    export_log_path: Option<PathBuf>,
    export_log_text: String,
    export_log_open: bool,
    export_log_focus: bool,

    // .h5ad -> .stviz conversion
    convert_generate_only: bool,
    convert_include_expr: bool,
    convert_input: String,
    convert_output: String,
    convert_status: Option<String>,
    convert_running: bool,
    convert_handle: Option<std::thread::JoinHandle<Result<ConvertResult, String>>>,
    convert_last_python_exe: Option<String>,
    convert_log_path: Option<PathBuf>,
    convert_log_text: String,
    mock_cells: u32,
    mock_last_h5ad: Option<PathBuf>,
    mock_last_stviz: Option<PathBuf>,
    mock_last_log: Option<PathBuf>,

    // UI/view settings
    ui_scale: f32,
    ui_theme: UiTheme,
    background_color: egui::Color32,
    fullscreen: bool,
    viewport_fullscreen: bool,
    show_axes: bool,
    show_stats: bool,
    reset_view_key_idx: usize,
    confirm_delete_cards: bool,
    filter_popup_open: bool,
    advanced_timeline_open: bool,
    advanced_cards: Vec<AdvancedCard>,
    advanced_next_id: u64,
    advanced_grid_mode: bool,
    advanced_grid_size: f32,
    advanced_drag_idx: Option<usize>,
    advanced_drag_offset: egui::Vec2,
    advanced_selected_card: Option<usize>,
    advanced_selected_cards: HashSet<usize>,
    advanced_preview_card: Option<usize>,
    advanced_viewport_pos: Option<egui::Pos2>,
    advanced_viewport_size: Option<egui::Vec2>,
    advanced_drag_group: Option<Vec<(usize, egui::Pos2)>>,
    advanced_drag_pointer_start: Option<egui::Pos2>,
    advanced_marquee_start: Option<egui::Pos2>,
    advanced_last_render_time: f64,
    advanced_connections: Vec<AdvancedConnection>,
    advanced_connecting: Option<AdvancedConnectionDrag>,
    main_view_refresh_hz: f32,
    main_view_fps_cap_hz: f32,
    main_view_last_refresh_query: f64,
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
    sample_grid_labels_enabled: bool,
    sample_grid_label_mode: SampleGridLabelMode,
    sample_grid_custom_labels_obs_idx: Option<usize>,
    sample_grid_custom_labels: Vec<String>,

    // Legend / gene input
    legend_range: Option<LegendRange>,
    active_legend_range: Option<LegendRange>,
    gene_query: String,
    gene_selected: Option<String>,

    // Open path fallback + status
    open_path: String,
    last_error: Option<String>,

    // Viewport (for interactions)
    last_viewport_points: egui::Rect,
    last_viewport_px: [f32; 2],
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
enum PlaybackMode {
    Once,
    Loop,
    PingPong,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SampleGridLabelMode {
    Default,
    Custom,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CategoricalPalette {
    Tableau10,
    Tab10,
    Tab20,
    Category10,
    Set1,
    Set2,
    Set3,
    Dark2,
    Accent,
    Paired,
    Pastel1,
    Pastel2,
    Turbo,
    Dataset,
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
    opaque: bool,
}

#[derive(Clone, Debug)]
struct CategoryPaletteCache {
    obs_idx: usize,
    categories_len: usize,
    palette: CategoricalPalette,
    has_dataset_palette: bool,
    colors: Vec<u32>,
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
struct AdvancedCardFilter {
    obs_idx: usize,
    enabled: Vec<bool>,
    cached_indices: Option<Arc<Vec<u32>>>,
    cached_indices_id: u64,
    cached_dataset_id: u64,
}

#[derive(Clone, Debug)]
struct AdvancedCard {
    id: u64,
    space_idx: usize,
    color_key: ColorKey,
    duration_sec: f32,
    pos: egui::Pos2,
    size: egui::Vec2,
    in_enabled: bool,
    out_enabled: bool,
    filter: Option<AdvancedCardFilter>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AdvancedConnection {
    from: usize,
    to: usize,
}

#[derive(Clone, Copy, Debug)]
struct AdvancedConnectionDrag {
    from_idx: usize,
    from_is_output: bool,
    start_pos: egui::Pos2,
}

#[derive(Clone, Debug)]
struct ScreenshotRequest {
    path: PathBuf,
    crop_px: Option<[u32; 4]>,
    is_export_frame: bool,
}

struct ConvertResult {
    msg: String,
    output: PathBuf,
    load_after: bool,
    python_exe: Option<String>,
}

impl StvizApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_embed_viewports(false);
        let rs = cc
            .wgpu_render_state
            .as_ref()
            .expect("eframe must be built with the wgpu renderer");
        let render_state = Some(rs.clone());

        // `target_format` is exposed by eframe's wgpu render state in typical setups.
        // If this fails due to API drift, read it from rs and propagate here.
        let target_format = rs.target_format;
        let adapter_info = rs.adapter.get_info();
        let adapter_label = format!(
            "{} ({:?}, {:?})",
            adapter_info.name, adapter_info.device_type, adapter_info.backend
        );
        let cpu_adapter = matches!(adapter_info.device_type, wgpu::DeviceType::Cpu);

        let project_dir = Self::resolve_project_dir();
        let output_dir = project_dir.join("output");
        let _ = std::fs::create_dir_all(&output_dir);
        Self::cleanup_mock_artifacts(&output_dir);
        let ffmpeg_path = Self::resolve_ffmpeg_path(&project_dir);

        let app = Self {
            dataset: None,
            dataset_path: None,
            dataset_id: 1,

            camera: Camera2D::default(),

            from_space: 0,
            to_space: 0,
            space_path: Vec::new(),

            color_mode: ColorMode::Categorical,
            active_obs_idx: 0,
            categorical_palette: CategoricalPalette::Tableau10,
            category_overrides: HashMap::new(),
            category_palette_cache: None,
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
            dragging_key_idx: None,
            key_collapsed: Vec::new(),

            playing: false,
            t: 0.0,
            speed: 0.35,
            play_direction: 1.0,
            play_last_time: None,
            playback_mode: PlaybackMode::PingPong,
            ease_mode: EaseMode::Smoothstep,
            point_radius_px: 0.5,
            max_draw_points: 0,
            fast_render: cpu_adapter,

            shared: Arc::new(SharedRender::new(target_format)),
            render_state,
            offscreen_gpu: None,
            offscreen_gpu_msaa: None,
            colors_id: 1,
            colors_rgba8: Arc::new(Vec::new()),
            colors_opaque: true,
            indices_id: 1,
            base_indices: Vec::new(),
            draw_indices: Arc::new(Vec::new()),

            project_dir: project_dir.clone(),
            output_dir: output_dir.clone(),
            screenshot_dir: output_dir.clone(),
            exporting_loop: false,
            export_fps: 30,
            export_duration_sec: 5.0,
            export_quality: ExportQuality::Current,
            export_resolution: None,
            export_video_quality: ExportVideoQuality::High,
            export_dir: output_dir.clone(),
            export_name: String::from("stviz-animate_loop.mp4"),
            export_output_path: None,
            export_total_frames: 0,
            export_frame_index: 0,
            export_status: None,
            export_run_ffmpeg: true,
            export_keep_frames: false,
            export_camera: None,
            export_pending_frames: 0,
            export_finishing: false,
            export_cancelled: false,
            ffmpeg_path: ffmpeg_path.clone(),
            ffmpeg_available: ffmpeg_path.is_some(),
            export_log_path: None,
            export_log_text: String::new(),
            export_log_open: false,
            export_log_focus: false,

            convert_generate_only: false,
            convert_include_expr: true,
            convert_input: String::new(),
            convert_output: String::new(),
            convert_status: None,
            convert_running: false,
            convert_handle: None,
            convert_last_python_exe: None,
            convert_log_path: None,
            convert_log_text: String::new(),
            mock_cells: 300_000,
            mock_last_h5ad: None,
            mock_last_stviz: None,
            mock_last_log: None,

            ui_scale: 1.0,
            ui_theme: UiTheme::Dark,
            background_color: egui::Color32::BLACK,
            fullscreen: false,
            viewport_fullscreen: false,
            show_axes: true,
            show_stats: true,
            reset_view_key_idx: 0,
            confirm_delete_cards: false,
            filter_popup_open: false,
            advanced_timeline_open: false,
            advanced_cards: Vec::new(),
            advanced_next_id: 1,
            advanced_grid_mode: false,
            advanced_grid_size: 48.0,
            advanced_drag_idx: None,
            advanced_drag_offset: egui::Vec2::ZERO,
            advanced_selected_card: None,
            advanced_selected_cards: HashSet::new(),
            advanced_preview_card: None,
            advanced_viewport_pos: None,
            advanced_viewport_size: None,
            advanced_drag_group: None,
            advanced_drag_pointer_start: None,
            advanced_marquee_start: None,
            advanced_last_render_time: 0.0,
            advanced_connections: Vec::new(),
            advanced_connecting: None,
            main_view_refresh_hz: 60.0,
            main_view_fps_cap_hz: 59.0,
            main_view_last_refresh_query: 0.0,
            adapter_label,
            frame_ms_avg: 0.0,

            sample_grid_enabled: false,
            sample_grid_obs_idx: None,
            sample_grid_space_idx: None,
            sample_grid_use_filter: true,
            sample_grid_padding: 0.15,
            grid_version: 1,
            grid_cache: None,
            sample_grid_labels_enabled: false,
            sample_grid_label_mode: SampleGridLabelMode::Default,
            sample_grid_custom_labels_obs_idx: None,
            sample_grid_custom_labels: Vec::new(),

            legend_range: None,
            active_legend_range: None,
            gene_query: String::new(),
            gene_selected: None,

            open_path: String::new(),
            last_error: None,

            last_viewport_points: egui::Rect::ZERO,
            last_viewport_px: [0.0, 0.0],
        };
        if app.fullscreen {
            app.apply_fullscreen(&cc.egui_ctx);
        } else {
            cc.egui_ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(true));
        }
        app
    }

    fn apply_fullscreen(&self, ctx: &egui::Context) {
        if Self::is_wsl() {
            return;
        }
        let enable = self.fullscreen;
        let is_wayland = std::env::var("XDG_SESSION_TYPE")
            .map(|v| v.eq_ignore_ascii_case("wayland"))
            .unwrap_or(false);
        if is_wayland {
            ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(true));
            if !enable {
                ctx.send_viewport_cmd(egui::ViewportCommand::Decorations(true));
            }
        } else {
            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(enable));
            ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(true));
            ctx.send_viewport_cmd(egui::ViewportCommand::Decorations(!enable));
        }
    }

    fn is_wsl() -> bool {
        std::env::var_os("WSL_DISTRO_NAME").is_some()
            || std::env::var_os("WSL_INTEROP").is_some()
    }

    fn update_main_view_refresh(&mut self, ctx: &egui::Context, _frame: &eframe::Frame) {
        let now = ctx.input(|i| i.time);
        if now - self.main_view_last_refresh_query < 2.0 {
            return;
        }
        self.main_view_last_refresh_query = now;
        let mut refresh_hz = self.main_view_refresh_hz;
        #[cfg(windows)]
        {
            if let Some(hz) = Self::query_refresh_hz_windows(_frame) {
                refresh_hz = hz;
            }
        }
        if refresh_hz <= 1.0 {
            refresh_hz = 60.0;
        }
        self.main_view_refresh_hz = refresh_hz;
        self.main_view_fps_cap_hz = (refresh_hz - 1.0).max(30.0);
    }

    #[cfg(windows)]
    fn query_refresh_hz_windows(frame: &eframe::Frame) -> Option<f32> {
        use raw_window_handle::RawWindowHandle;
        use windows::Win32::Foundation::HWND;
        use windows::Win32::Graphics::Gdi::{GetDC, GetDeviceCaps, ReleaseDC, VREFRESH};

        let handle = frame.window_handle().ok()?;
        let RawWindowHandle::Win32(handle) = handle.as_raw() else {
            return None;
        };
        let hwnd = HWND(handle.hwnd.get() as *mut core::ffi::c_void);
        unsafe {
            let hdc = GetDC(hwnd);
            if hdc.is_invalid() {
                return None;
            }
            let refresh = GetDeviceCaps(hdc, VREFRESH) as i32;
            ReleaseDC(hwnd, hdc);
            if refresh > 1 {
                Some(refresh as f32)
            } else {
                None
            }
        }
    }

    fn categorical_palette_label(palette: CategoricalPalette) -> &'static str {
        match palette {
            CategoricalPalette::Tableau10 => "Tableau10",
            CategoricalPalette::Tab10 => "Tab10",
            CategoricalPalette::Tab20 => "Tab20",
            CategoricalPalette::Category10 => "Category10",
            CategoricalPalette::Set1 => "Set1",
            CategoricalPalette::Set2 => "Set2",
            CategoricalPalette::Set3 => "Set3",
            CategoricalPalette::Dark2 => "Dark2",
            CategoricalPalette::Accent => "Accent",
            CategoricalPalette::Paired => "Paired",
            CategoricalPalette::Pastel1 => "Pastel1",
            CategoricalPalette::Pastel2 => "Pastel2",
            CategoricalPalette::Turbo => "Turbo",
            CategoricalPalette::Dataset => "Dataset",
        }
    }

    fn categorical_palette_name(palette: CategoricalPalette) -> &'static str {
        match palette {
            CategoricalPalette::Tableau10 => "tableau10",
            CategoricalPalette::Tab10 => "tab10",
            CategoricalPalette::Tab20 => "tab20",
            CategoricalPalette::Category10 => "category10",
            CategoricalPalette::Set1 => "set1",
            CategoricalPalette::Set2 => "set2",
            CategoricalPalette::Set3 => "set3",
            CategoricalPalette::Dark2 => "dark2",
            CategoricalPalette::Accent => "accent",
            CategoricalPalette::Paired => "paired",
            CategoricalPalette::Pastel1 => "pastel1",
            CategoricalPalette::Pastel2 => "pastel2",
            CategoricalPalette::Turbo => "turbo",
            CategoricalPalette::Dataset => "dataset",
        }
    }

    fn categorical_palette_for(
        &self,
        obs_idx: usize,
        categories_len: usize,
        pal_opt: Option<&[u32]>,
    ) -> Vec<u32> {
        let base = match self.categorical_palette {
            CategoricalPalette::Dataset => pal_opt
                .map(|p| p.to_vec())
                .unwrap_or_else(|| categorical_palette_named("tableau10", categories_len)),
            CategoricalPalette::Turbo => categorical_palette(categories_len),
            other => categorical_palette_named(Self::categorical_palette_name(other), categories_len),
        };
        let mut out = base;
        if let Some(overrides) = self.category_overrides.get(&obs_idx) {
            for (i, override_color) in overrides.iter().enumerate() {
                if let Some(color) = override_color {
                    if i < out.len() {
                        out[i] = *color;
                    }
                }
            }
        }
        out
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
        let max_idx = ds.meta.spaces.len().saturating_sub(1);
        let from_idx = find_space_by_name(&ds, "spatial").unwrap_or(0.min(max_idx));
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
        self.key_times = vec![0.0, 1.0];
        self.key_collapsed = vec![false; self.space_path.len()];
        self.color_cache.clear();
        self.category_state.clear();
        self.category_overrides.clear();
        self.category_palette_cache = None;
        self.active_filters.clear();
        self.load_filter_state(&ds);

        let is_mock = path
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.starts_with("mock_spatial_"))
            .unwrap_or(false);
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
            if let Some(idx) = find_space_by_name(&ds, "x_pca")
                .or_else(|| find_space_by_name(&ds, "pca"))
            {
                push_unique(idx);
            }
            if let Some(idx) = find_space_by_name(&ds, "x_umap")
                .or_else(|| find_space_by_name(&ds, "umap"))
            {
                push_unique(idx);
            }
            if let Some(idx) = find_space_by_name(&ds, "x_tsne")
                .or_else(|| find_space_by_name(&ds, "tsne"))
            {
                push_unique(idx);
            }
            if mock_spaces.len() >= 2 {
                self.space_path = mock_spaces;
                self.from_space = *self.space_path.first().unwrap_or(&self.from_space);
                self.to_space = *self.space_path.last().unwrap_or(&self.to_space);
                self.color_path = vec![ColorKey::Current; self.space_path.len()];
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
                    let viewport_px = if self.last_viewport_px[0] > 0.0
                        && self.last_viewport_px[1] > 0.0
                    {
                        self.last_viewport_px
                    } else {
                        [1000.0, 700.0]
                    };
                    self.camera.fit_bbox(bbox, viewport_px, 0.9);
                }
            }
        }

        self.sample_grid_obs_idx = find_obs_by_name(&ds, "sample");
        self.sample_grid_space_idx = find_space_by_name(&ds, "spatial")
            .or_else(|| find_space_by_name(&ds, "centroid"))
            .or_else(|| if ds.meta.spaces.is_empty() { None } else { Some(0) });
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
        if let Some(idx) = find_obs_by_name(&ds, "leiden") {
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
                self.camera.fit_bbox(bbox, [1000.0, 700.0], 0.9);
            }
        }

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
                self.colors_opaque = true;
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
                        pal.get(idx).copied().unwrap_or(pack_rgba8(200, 200, 200, 255))
                    } else {
                        pal.get(li).copied().unwrap_or(pack_rgba8(200, 200, 200, 255))
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
                self.colors_opaque = true;
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
            if matches!(ds.meta.obs.get(self.active_obs_idx), Some(ObsMeta::Categorical { .. }))
                && !self.enabled_categories.is_empty()
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

    fn downsample_indices(max_draw: usize, indices: &[u32]) -> Vec<u32> {
        if max_draw == 0 || indices.len() <= max_draw {
            return indices.to_vec();
        }
        let step = (indices.len() as f32 / max_draw as f32).ceil().max(1.0) as usize;
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

    fn filter_signature(dataset_id: u64, obs_idx: usize, enabled: &[bool]) -> u64 {
        let mut hash = 1469598103934665603u64 ^ dataset_id.wrapping_mul(1099511628211);
        hash ^= obs_idx as u64;
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
        if filter.cached_dataset_id == dataset_id {
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
        let sig = Self::filter_signature(dataset_id, filter.obs_idx, &filter.enabled);
        let arc = Arc::new(out);
        filter.cached_indices = Some(arc.clone());
        filter.cached_indices_id = sig;
        filter.cached_dataset_id = dataset_id;
        Some((arc, sig))
    }

    fn advanced_draw_override(
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

    fn maybe_update_playback(&mut self, ctx: &egui::Context) {
        if self.exporting_loop {
            return;
        }
        if !self.playing {
            return;
        }
        let now = ctx.input(|i| i.time);
        let last = self.play_last_time.unwrap_or(now);
        let dt = (now - last) as f32;
        self.play_last_time = Some(now);
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
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            if self.viewport_fullscreen {
                self.viewport_fullscreen = false;
                ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
            } else if self.fullscreen {
                self.fullscreen = false;
                self.apply_fullscreen(ctx);
            }
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
            self.playing = !self.playing;
            self.play_last_time = None;
            if self.playing {
                self.play_last_time = Some(ctx.input(|i| i.time));
            }
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

    fn main_view_frame_interval(&self) -> std::time::Duration {
        let hz = self.main_view_fps_cap_hz.max(30.0);
        std::time::Duration::from_secs_f32(1.0 / hz)
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
                if let Some(python_exe) = result.python_exe {
                    self.convert_last_python_exe = Some(python_exe);
                }
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
        self.refresh_convert_log();
        if let Some(path) = self.convert_log_path.clone() {
            let is_mock_log = path
                .file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.starts_with("mock_convert_log_"))
                .unwrap_or(false);
            if is_mock_log {
                let _ = std::fs::remove_file(&path);
                self.convert_log_path = None;
            }
        }
    }

    fn refresh_convert_log(&mut self) {
        let Some(path) = self.convert_log_path.as_ref() else {
            return;
        };
        if let Ok(text) = std::fs::read_to_string(path) {
            self.convert_log_text = text;
        }
    }

    fn queue_h5ad_convert(&mut self, path: &Path) {
        self.convert_input = path.display().to_string();
        self.convert_output = self.default_convert_output(&self.convert_input).display().to_string();
        self.start_convert();
    }

    fn clamp_mock_cells(&mut self) -> u32 {
        let clamped = self.mock_cells.clamp(10_000, 10_000_000);
        if clamped != self.mock_cells {
            self.mock_cells = clamped;
        }
        clamped
    }

    fn default_mock_paths(&self, cells: u32) -> (PathBuf, PathBuf) {
        let _ = std::fs::create_dir_all(&self.output_dir);
        let ts = chrono_like_timestamp();
        let h5ad = self
            .output_dir
            .join(format!("mock_spatial_{cells}_{ts}.h5ad"));
        let stviz = self
            .output_dir
            .join(format!("mock_spatial_{cells}_{ts}.stviz"));
        (h5ad, stviz)
    }

    fn cleanup_mock_artifacts(output_dir: &Path) {
        let Ok(entries) = std::fs::read_dir(output_dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            if name.starts_with("mock_spatial_") || name.starts_with("mock_convert_log_") {
                let _ = std::fs::remove_file(&path);
            }
        }
    }

    fn cleanup_previous_mock(&mut self) {
        if let Some(path) = self.mock_last_h5ad.as_ref() {
            if path.starts_with(&self.output_dir) {
                let _ = std::fs::remove_file(path);
            }
        }
        if let Some(path) = self.mock_last_stviz.as_ref() {
            if path.starts_with(&self.output_dir) {
                let _ = std::fs::remove_file(path);
            }
        }
        if let Some(path) = self.mock_last_log.as_ref() {
            if path.starts_with(&self.output_dir) {
                let _ = std::fs::remove_file(path);
            }
        }
    }

    fn detect_python_cmd() -> Result<(String, String), String> {
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
        let hint = "If Python is installed but not found, run:\n\
  python -c \"import sys; print(sys.executable)\"\n\
Then add that path to your system environment variables (PATH).";
        if errors.is_empty() {
            Err(format!(
                "Python not found. Install Python and ensure `python` or `python3` is on PATH.\n{hint}"
            ))
        } else {
            Err(format!("{}\n\n{hint}", errors.join("\n")))
        }
    }

    fn check_python_cmd(cmd: &str) -> Result<(String, String), String> {
        let env_root = Self::python_env_root(cmd);
        let mut exe_cmd = Command::new(cmd);
        exe_cmd
            .arg("-c")
            .arg(
                "import sys; print(sys.executable); print(f\"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}\")",
            );
        Self::apply_python_env(&mut exe_cmd, env_root.as_deref());
        let exe_out = exe_cmd
            .output()
            .map_err(|e| format!("`{cmd}` not found: {e}"))?;
        if !exe_out.status.success() {
            return Err(format!("`{cmd}` failed to run (exit {}).", exe_out.status));
        }
        let stdout = String::from_utf8_lossy(&exe_out.stdout);
        let mut lines = stdout.lines();
        let exe = lines.next().unwrap_or("").trim().to_string();
        let version = lines.next().unwrap_or("").trim().to_string();
        if !version.is_empty() {
            let mut parts = version.split(|c| c == '.' || c == ' ');
            let major = parts.next().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
            let minor = parts.next().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
            if major > 0 && (major, minor) < MIN_PYTHON_VERSION {
                return Err(format!(
                    "`{cmd}` is Python {version}, but {}.{}+ is required for the converter scripts.",
                    MIN_PYTHON_VERSION.0, MIN_PYTHON_VERSION.1
                ));
            }
        }
        let exe_label = if exe.is_empty() { cmd.to_string() } else { exe };

        Ok((cmd.to_string(), exe_label))
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

    fn resolve_project_dir() -> PathBuf {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        if cwd.join("python").join("export_stviz.py").exists() {
            return cwd;
        }
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(mut dir) = exe_path.parent().map(|p| p.to_path_buf()) {
                #[cfg(target_os = "macos")]
                if let Some(contents_dir) = dir.parent() {
                    let resources_dir = contents_dir.join("Resources");
                    if resources_dir
                        .join("python")
                        .join("export_stviz.py")
                        .exists()
                    {
                        return resources_dir;
                    }
                }
                if let Some(parent) = dir.parent() {
                    let share_dir = parent.join("share").join("stviz-animate");
                    if share_dir
                        .join("python")
                        .join("export_stviz.py")
                        .exists()
                    {
                        return share_dir;
                    }
                }
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

    fn resolve_ffmpeg_path(project_dir: &Path) -> Option<PathBuf> {
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Ok(env) = std::env::var("STVIZ_FFMPEG") {
            if !env.trim().is_empty() {
                candidates.push(PathBuf::from(env));
            }
        }
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(dir) = exe_path.parent() {
                candidates.push(dir.join("ffmpeg"));
                candidates.push(dir.join("ffmpeg.exe"));
                candidates.push(dir.join("bin").join("ffmpeg"));
                candidates.push(dir.join("bin").join("ffmpeg.exe"));
            }
        }
        candidates.push(project_dir.join("ffmpeg"));
        candidates.push(project_dir.join("ffmpeg.exe"));
        candidates.push(project_dir.join("bin").join("ffmpeg"));
        candidates.push(project_dir.join("bin").join("ffmpeg.exe"));
        candidates.push(project_dir.join("ffmpeg").join("ffmpeg"));
        candidates.push(project_dir.join("ffmpeg").join("ffmpeg.exe"));
        candidates.push(project_dir.join("ffmpeg").join("bin").join("ffmpeg"));
        candidates.push(project_dir.join("ffmpeg").join("bin").join("ffmpeg.exe"));

        for candidate in candidates {
            if candidate.exists() {
                let mut cmd = Command::new(&candidate);
                Self::apply_subprocess_flags(&mut cmd);
                if cmd.arg("-version").output().is_ok() {
                    return Some(candidate);
                }
            }
        }

        let mut cmd = Command::new("ffmpeg");
        Self::apply_subprocess_flags(&mut cmd);
        if cmd.arg("-version").output().is_ok() {
            return Some(PathBuf::from("ffmpeg"));
        }
        None
    }

    fn refresh_ffmpeg_path(&mut self) {
        let path = Self::resolve_ffmpeg_path(&self.project_dir);
        self.ffmpeg_available = path.is_some();
        self.ffmpeg_path = path;
    }

    fn run_python_cv2_export(&mut self, frames_dir: &Path, out_path: &Path) -> Result<(), String> {
        let script = self.project_dir.join("python").join("export_video_cv2.py");
        if !script.exists() {
            return Err(format!("OpenCV exporter not found: {}", script.display()));
        }

        let (python_cmd, python_exe) = match Self::detect_python_cmd() {
            Ok(cmd) => cmd,
            Err(msg) => {
                self.append_export_log(&msg);
                return Err(msg);
            }
        };
        self.convert_last_python_exe = Some(python_exe);
        let venv_dir = self.project_dir.join(".stviz_venv");
        let venv_python = Self::venv_python_path(&venv_dir);
        let base_env_root = Self::python_env_root(&python_cmd);

        if !venv_python.exists() {
            self.append_export_log("Creating a private converter environment for OpenCV...");
            let mut cmd = Command::new(&python_cmd);
            cmd.arg("-m").arg("venv").arg(&venv_dir);
            Self::apply_python_env(&mut cmd, base_env_root.as_deref());
            let out = cmd.output().map_err(|e| {
                let msg = format!("Failed to create virtual environment: {e}");
                self.append_export_log(&msg);
                msg
            })?;
            if !out.status.success() {
                let msg = "Virtual environment setup failed.".to_string();
                self.append_export_log(&msg);
                self.append_export_log(&String::from_utf8_lossy(&out.stdout));
                self.append_export_log(&String::from_utf8_lossy(&out.stderr));
                return Err(msg);
            }
        }

        self.append_export_log("Checking OpenCV (cv2) dependency...");
        let mut check_cmd = Command::new(&venv_python);
        check_cmd.arg("-c").arg("import cv2");
        Self::apply_python_env(&mut check_cmd, Some(&venv_dir));
        let check_out = check_cmd.output().map_err(|e| {
            let msg = format!("Failed to check OpenCV dependency: {e}");
            self.append_export_log(&msg);
            msg
        })?;

        if !check_out.status.success() {
            self.append_export_log(
                "Installing OpenCV (opencv-python-headless). This may take a few minutes...",
            );
            let mut pip_cmd = Command::new(&venv_python);
            pip_cmd
                .arg("-m")
                .arg("pip")
                .arg("install")
                .arg("-U")
                .arg("opencv-python-headless");
            Self::apply_python_env(&mut pip_cmd, Some(&venv_dir));
            let pip_out = pip_cmd.output().map_err(|e| {
                let msg = format!("Failed to install OpenCV: {e}");
                self.append_export_log(&msg);
                msg
            })?;
            if !pip_out.status.success() {
                let msg = "OpenCV install failed.".to_string();
                self.append_export_log(&msg);
                self.append_export_log(&String::from_utf8_lossy(&pip_out.stdout));
                self.append_export_log(&String::from_utf8_lossy(&pip_out.stderr));
                return Err(msg);
            }
        }

        self.append_export_log("Encoding video with OpenCV...");
        let mut cmd = Command::new(&venv_python);
        cmd.arg("-X")
            .arg("faulthandler")
            .arg(&script)
            .arg("--input-dir")
            .arg(frames_dir)
            .arg("--output")
            .arg(out_path)
            .arg("--fps")
            .arg(self.export_fps.to_string());
        cmd.env("PYTHONFAULTHANDLER", "1");
        cmd.current_dir(&self.project_dir);
        Self::apply_python_env(&mut cmd, Some(&venv_dir));
        let out = cmd.output().map_err(|e| {
            let msg = format!("Failed to run OpenCV exporter: {e}");
            self.append_export_log(&msg);
            msg
        })?;
        if !out.status.success() {
            let msg = "OpenCV export failed.".to_string();
            self.append_export_log(&msg);
            self.append_export_log(&String::from_utf8_lossy(&out.stdout));
            self.append_export_log(&String::from_utf8_lossy(&out.stderr));
            return Err(msg);
        }
        let stdout = String::from_utf8_lossy(&out.stdout);
        if !stdout.trim().is_empty() {
            self.append_export_log(stdout.trim());
        }
        Ok(())
    }

    fn apply_python_env(cmd: &mut Command, env_root: Option<&Path>) -> Option<String> {
        Self::apply_subprocess_flags(cmd);
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

    fn apply_subprocess_flags(cmd: &mut Command) {
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
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

    fn venv_python_path(venv_dir: &Path) -> PathBuf {
        if cfg!(windows) {
            venv_dir.join("Scripts").join("python.exe")
        } else {
            venv_dir.join("bin").join("python")
        }
    }

    fn start_mock_dataset(&mut self) {
        if self.convert_running {
            return;
        }
        self.cleanup_previous_mock();
        let cells = self.clamp_mock_cells();
        let (h5ad_path, stviz_path) = self.default_mock_paths(cells);
        self.mock_last_h5ad = Some(h5ad_path.clone());
        self.mock_last_stviz = Some(stviz_path.clone());
        self.convert_input = h5ad_path.display().to_string();
        self.convert_output = stviz_path.display().to_string();

        let script = self.project_dir.join("python").join("mock_spatial_dataset.py");
        if !script.exists() {
            self.convert_status = Some(format!(
                "Mock dataset script not found: {}",
                script.display()
            ));
            self.last_error = Some(format!(
                "Mock dataset script not found: {}",
                script.display()
            ));
            return;
        }

        let (python_cmd, python_exe) = match Self::detect_python_cmd() {
            Ok(cmd) => cmd,
            Err(msg) => {
                self.convert_status = Some(msg);
                return;
            }
        };
        self.convert_last_python_exe = Some(python_exe.clone());

        let input = h5ad_path.display().to_string();
        let output = stviz_path.display().to_string();
        let script = script.to_string_lossy().to_string();
        let include_expr = self.convert_include_expr;
        let generate_only = self.convert_generate_only;
        let project_dir = self.project_dir.clone();
        let log_path = self
            .output_dir
            .join(format!("mock_convert_log_{}.txt", chrono_like_timestamp()));
        let seed: u64 = rand::thread_rng().gen();

        self.convert_running = true;
        self.convert_log_path = Some(log_path.clone());
        self.convert_log_text.clear();
        self.convert_status = Some("Preparing mock dataset...".to_string());
        self.convert_handle = Some(thread::spawn(move || {
            let mut log = String::new();
            let mut append = |line: &str| {
                log.push_str(line);
                log.push('\n');
                let _ = std::fs::write(&log_path, &log);
            };

            append("Mock dataset generation started.");
            append(&format!("Requested cells: {cells}"));
            append(if include_expr {
                "Including gene expression data."
            } else {
                "Skipping gene expression data for smaller file size."
            });
            append("Preparing the converter environment...");
            append(&format!("Project dir: {}", project_dir.display()));
            append(&format!("Python cmd: {}", python_cmd));

            let venv_dir = project_dir.join(".stviz_venv");
            let venv_python = Self::venv_python_path(&venv_dir);
            let base_env_root = Self::python_env_root(&python_cmd);
            append(&format!("Venv dir: {}", venv_dir.display()));
            append(&format!("Venv python: {}", venv_python.display()));
            if !venv_python.exists() {
                append("Creating a private converter environment...");
                let mut cmd = Command::new(&python_cmd);
                cmd.arg("-m").arg("venv").arg(&venv_dir);
                Self::apply_python_env(&mut cmd, base_env_root.as_deref());
                let out = cmd.output().map_err(|e| {
                    append("Failed to create the converter environment.");
                    append(&format!("Details: {e}"));
                    format!("Failed to create virtual environment: {e}\nLog: {}", log_path.display())
                })?;
                if !out.status.success() {
                    append("Environment setup output:");
                    append(&String::from_utf8_lossy(&out.stdout));
                    append("Environment setup error:");
                    append(&String::from_utf8_lossy(&out.stderr));
                    return Err(format!(
                        "Virtual environment setup failed.\nLog: {}",
                        log_path.display()
                    ));
                }
            } else {
                append("Converter environment found.");
            }

            append("Checking converter dependencies...");
            let mut check_cmd = Command::new(&venv_python);
            check_cmd
                .arg("-c")
                .arg("import anndata, h5py, numpy, pandas, scipy");
            Self::apply_python_env(&mut check_cmd, Some(&venv_dir));
            append("Dependency check: import anndata, h5py, numpy, pandas, scipy");
            let check_out = check_cmd.output().map_err(|e| {
                append("Dependency check failed.");
                append(&format!("Details: {e}"));
                format!("Failed to check dependencies: {e}\nLog: {}", log_path.display())
            })?;

            if !check_out.status.success() {
                append("Installing converter dependencies (this may take a minute)...");
                append("pip install -U anndata h5py numpy pandas scipy");
                let mut pip_cmd = Command::new(&venv_python);
                pip_cmd
                    .arg("-m")
                    .arg("pip")
                    .arg("install")
                    .arg("-U")
                    .arg("anndata")
                    .arg("h5py")
                    .arg("numpy")
                    .arg("pandas")
                    .arg("scipy");
                Self::apply_python_env(&mut pip_cmd, Some(&venv_dir));
                let pip_out = pip_cmd.output().map_err(|e| {
                    append("Dependency install failed.");
                    append(&format!("Details: {e}"));
                    format!("Failed to install dependencies: {e}\nLog: {}", log_path.display())
                })?;
                if !pip_out.status.success() {
                    append("Install output:");
                    append(&String::from_utf8_lossy(&pip_out.stdout));
                    append("Install error:");
                    append(&String::from_utf8_lossy(&pip_out.stderr));
                    return Err(format!(
                        "Dependency install failed.\nLog: {}",
                        log_path.display()
                    ));
                }
                append("Dependencies installed.");
            } else {
                append("Dependencies already installed.");
            }

            append("Generating the mock .h5ad file...");
            let mut mock_cmd = Command::new(&venv_python);
            mock_cmd
                .arg("-X")
                .arg("faulthandler")
                .arg(&script)
                .arg("--out")
                .arg(&input)
                .arg("--cells")
                .arg(cells.to_string())
                .arg("--seed")
                .arg(seed.to_string());
            if !include_expr {
                mock_cmd.arg("--no-expr");
            }
            mock_cmd.env("PYTHONFAULTHANDLER", "1");
            mock_cmd.current_dir(&project_dir);
            Self::apply_python_env(&mut mock_cmd, Some(&venv_dir));
            append(&format!(
                "Command: {} {} --out {} --cells {} --seed {}{}",
                venv_python.display(),
                script,
                input,
                cells,
                seed,
                if include_expr { "" } else { " --no-expr" }
            ));

            let out = mock_cmd.output().map_err(|e| {
                append("Mock dataset generation failed to start.");
                append(&format!("Details: {e}"));
                format!("Failed to run mock generator: {e}\nLog: {}", log_path.display())
            })?;
            if !out.status.success() {
                append("Mock generation output:");
                append(&String::from_utf8_lossy(&out.stdout));
                append("Mock generation error:");
                append(&String::from_utf8_lossy(&out.stderr));
                return Err(format!(
                    "Mock dataset generation failed.\nLog: {}",
                    log_path.display()
                ));
            }
            append("Mock dataset created.");

            append("Converting dataset...");
            let mut cmd = Command::new(&venv_python);
            cmd.arg("-X")
                .arg("faulthandler")
                .arg(project_dir.join("python").join("export_stviz.py"))
                .arg("--input")
                .arg(&input)
                .arg("--output")
                .arg(&output);
            if include_expr {
                cmd.arg("--include-expr");
            }
            cmd.env("PYTHONFAULTHANDLER", "1");
            cmd.current_dir(&project_dir);
            Self::apply_python_env(&mut cmd, Some(&venv_dir));
            append(&format!(
                "Command: {} export_stviz.py --input {} --output {}{}",
                venv_python.display(),
                input,
                output,
                if include_expr { " --include-expr" } else { "" }
            ));

            let out = cmd.output().map_err(|e| {
                append("Conversion failed to start.");
                append(&format!("Details: {e}"));
                format!("Failed to run exporter: {e}\nLog: {}", log_path.display())
            })?;
            if !out.status.success() {
                append("Conversion output:");
                append(&String::from_utf8_lossy(&out.stdout));
                append("Conversion error:");
                append(&String::from_utf8_lossy(&out.stderr));
                return Err(format!(
                    "Conversion failed.\nLog: {}",
                    log_path.display()
                ));
            }
            append("Conversion completed.");
            let stdout = String::from_utf8_lossy(&out.stdout);
            Ok(ConvertResult {
                msg: format!("Mock dataset ready.\n{}", stdout.trim()),
                output: PathBuf::from(output),
                load_after: !generate_only,
                python_exe: Some(venv_python.display().to_string()),
            })
        }));
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

        let (python_cmd, python_exe) = match Self::detect_python_cmd() {
            Ok(cmd) => cmd,
            Err(msg) => {
                self.convert_status = Some(msg);
                return;
            }
        };
        self.convert_last_python_exe = Some(python_exe.clone());
        let input = input.to_string();
        let output = output.to_string();
        let script = script.to_string_lossy().to_string();
        let include_expr = self.convert_include_expr;
        let generate_only = self.convert_generate_only;
        let project_dir = self.project_dir.clone();
        let log_path = self
            .output_dir
            .join(format!("convert_log_{}.txt", chrono_like_timestamp()));
        self.convert_running = true;
        self.convert_log_path = Some(log_path.clone());
        self.convert_log_text.clear();
        self.convert_status = Some("Preparing converter environment...".to_string());
        self.convert_handle = Some(thread::spawn(move || {
            let mut log = String::new();
            let mut append = |line: &str| {
                log.push_str(line);
                log.push('\n');
                let _ = std::fs::write(&log_path, &log);
            };

            append("Conversion started.");
            append(if include_expr {
                "Including gene expression data."
            } else {
                "Skipping gene expression data for smaller file size."
            });
            append("Preparing the converter environment...");
            append(&format!("Input file: {input}"));
            append(&format!("Output file: {output}"));
            append(&format!("Project dir: {}", project_dir.display()));
            append(&format!("Python cmd: {}", python_cmd));

            let venv_dir = project_dir.join(".stviz_venv");
            let venv_python = Self::venv_python_path(&venv_dir);
            let base_env_root = Self::python_env_root(&python_cmd);
            append(&format!("Venv dir: {}", venv_dir.display()));
            append(&format!("Venv python: {}", venv_python.display()));
            if !venv_python.exists() {
                append("Creating a private converter environment...");
                let mut cmd = Command::new(&python_cmd);
                cmd.arg("-m").arg("venv").arg(&venv_dir);
                Self::apply_python_env(&mut cmd, base_env_root.as_deref());
                let out = cmd.output().map_err(|e| {
                    append("Failed to create the converter environment.");
                    append(&format!("Details: {e}"));
                    format!("Failed to create virtual environment: {e}\nLog: {}", log_path.display())
                })?;
                if !out.status.success() {
                    append("Environment setup output:");
                    append(&String::from_utf8_lossy(&out.stdout));
                    append("Environment setup error:");
                    append(&String::from_utf8_lossy(&out.stderr));
                    return Err(format!(
                        "Virtual environment setup failed.\nLog: {}",
                        log_path.display()
                    ));
                }
            } else {
                append("Converter environment found.");
            }

            append("Checking converter dependencies...");
            let mut check_cmd = Command::new(&venv_python);
            check_cmd
                .arg("-c")
                .arg("import anndata, h5py, numpy, pandas, scipy");
            Self::apply_python_env(&mut check_cmd, Some(&venv_dir));
            append("Dependency check: import anndata, h5py, numpy, pandas, scipy");
            let check_out = check_cmd.output().map_err(|e| {
                append("Dependency check failed.");
                append(&format!("Details: {e}"));
                format!("Failed to check dependencies: {e}\nLog: {}", log_path.display())
            })?;

            if !check_out.status.success() {
                append("Installing converter dependencies (this may take a minute)...");
                append("pip install -U anndata h5py numpy pandas scipy");
                let mut pip_cmd = Command::new(&venv_python);
                pip_cmd
                    .arg("-m")
                    .arg("pip")
                    .arg("install")
                    .arg("-U")
                    .arg("anndata")
                    .arg("h5py")
                    .arg("numpy")
                    .arg("pandas")
                    .arg("scipy");
                Self::apply_python_env(&mut pip_cmd, Some(&venv_dir));
                let pip_out = pip_cmd.output().map_err(|e| {
                    append("Dependency install failed.");
                    append(&format!("Details: {e}"));
                    format!("Failed to install dependencies: {e}\nLog: {}", log_path.display())
                })?;
                if !pip_out.status.success() {
                    append("Install output:");
                    append(&String::from_utf8_lossy(&pip_out.stdout));
                    append("Install error:");
                    append(&String::from_utf8_lossy(&pip_out.stderr));
                    return Err(format!(
                        "Dependency install failed.\nLog: {}",
                        log_path.display()
                    ));
                }
                append("Dependencies installed.");
            } else {
                append("Dependencies already installed.");
            }

            append("Converting dataset...");
            let mut cmd = Command::new(&venv_python);
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
            Self::apply_python_env(&mut cmd, Some(&venv_dir));
            append(&format!(
                "Command: {} export_stviz.py --input {} --output {}{}",
                venv_python.display(),
                input,
                output,
                if include_expr { " --include-expr" } else { "" }
            ));

            let out = cmd.output().map_err(|e| {
                append("Conversion failed to start.");
                append(&format!("Details: {e}"));
                format!("Failed to run exporter: {e}\nLog: {}", log_path.display())
            })?;
            if !out.status.success() {
                append("Conversion output:");
                append(&String::from_utf8_lossy(&out.stdout));
                append("Conversion error:");
                append(&String::from_utf8_lossy(&out.stderr));
                return Err(format!(
                    "Conversion failed.\nLog: {}",
                    log_path.display()
                ));
            }
            append("Conversion completed.");
            let stdout = String::from_utf8_lossy(&out.stdout);
            Ok(ConvertResult {
                msg: format!("Conversion completed.\n{}", stdout.trim()),
                output: PathBuf::from(output),
                load_after: !generate_only,
                python_exe: Some(venv_python.display().to_string()),
            })
        }));
    }

    fn start_export_loop(&mut self) {
        if self.exporting_loop {
            return;
        }
        if self.export_duration_sec <= 0.0 {
            self.export_status = Some("Export failed: duration must be > 0.".to_string());
            return;
        }
        self.refresh_ffmpeg_path();
        let duration = self.export_duration_sec.max(0.01);
        let total = (duration * self.export_fps as f32).round().max(2.0) as u32;

        let ts = chrono_like_timestamp();
        self.export_dir = self.output_dir.join(format!("loop_{ts}"));
        let _ = std::fs::create_dir_all(&self.export_dir);
        self.export_log_path = Some(self.output_dir.join(format!("export_log_{ts}.txt")));
        self.export_log_text.clear();
        self.export_log_open = true;
        self.export_log_focus = true;
        if self.export_name.trim().is_empty() {
            self.export_name = String::from("stviz-animate_loop.mp4");
        }

        self.export_total_frames = total;
        self.export_frame_index = 0;
        self.exporting_loop = true;
        self.export_pending_frames = 0;
        self.export_finishing = false;
        self.export_cancelled = false;
        self.export_resolution = self.export_resolution_for_quality();
        self.export_status = Some(format!(
            "Exporting {total} frames ({:.2}s)...",
            duration
        ));
        self.playing = false;
        self.export_camera = None;

        self.append_export_log("Loop export started.");
        self.append_export_log(&format!("Frames directory: {}", self.export_dir.display()));
        let out_path = self.output_dir.join(self.export_name.trim());
        self.export_output_path = Some(out_path.clone());
        self.append_export_log(&format!("Output video: {}", out_path.display()));
        self.append_export_log(&format!("FPS: {}", self.export_fps));
        self.append_export_log(&format!("Total frames: {}", total));
        self.append_export_log(&format!("Duration (sec): {:.2}", duration));
        if let Some(res) = self.export_resolution {
            self.append_export_log(&format!("Resolution: {}x{}", res[0], res[1]));
        } else {
            self.append_export_log("Resolution: current viewport");
        }
        self.append_export_log(&format!("Playback mode: {:?}", self.playback_mode));
        self.append_export_log(&format!("Speed: {:.4}", self.speed));
        let keep_label = if !self.export_run_ffmpeg {
            "yes (forced)"
        } else if self.export_keep_frames {
            "yes"
        } else {
            "no"
        };
        self.append_export_log(&format!("Keep PNG frames: {keep_label}"));
        if self.export_run_ffmpeg {
            let ffmpeg_label = self
                .ffmpeg_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "not found".to_string());
            self.append_export_log(&format!("ffmpeg: {}", ffmpeg_label));
            let (crf, preset, pix_fmt) = match self.export_video_quality {
                ExportVideoQuality::Standard => (23, "medium", "yuv420p"),
                ExportVideoQuality::High => (18, "slow", "yuv420p"),
                ExportVideoQuality::Ultra => (14, "slow", "yuv420p"),
            };
            self.append_export_log(&format!(
                "Encoding: libx264, CRF {crf}, preset {preset}, {pix_fmt}"
            ));
            if self.ffmpeg_path.is_none() {
                self.append_export_log("ffmpeg missing; will try OpenCV fallback.");
            }
        } else {
            self.append_export_log("ffmpeg disabled.");
        }
        let mut dataset_lines = Vec::new();
        if let Some(ds) = self.dataset.as_ref() {
            dataset_lines.push(format!("Dataset points: {}", ds.meta.n_points));
            dataset_lines.push(format!("Spaces: {}", ds.meta.spaces.len()));
            for (i, space_idx) in self.space_path.iter().enumerate() {
                let space_name = ds
                    .meta
                    .spaces
                    .get(*space_idx)
                    .map(|s| s.name.as_str())
                    .unwrap_or("?");
                let color_key = self.color_path.get(i).cloned().unwrap_or(ColorKey::Current);
                let color_desc = match color_key {
                    ColorKey::Current => "Current".to_string(),
                    ColorKey::Categorical(idx) => format!("Cat: {}", obs_name(ds, idx)),
                    ColorKey::Continuous(idx) => format!("Cont: {}", obs_name(ds, idx)),
                    ColorKey::Gene(name) => format!("Gene: {}", name),
                };
                dataset_lines.push(format!(
                    "Key {}: space={}, color={}",
                    i + 1,
                    space_name,
                    color_desc
                ));
            }
            if !self.active_filters.is_empty() {
                let filters = self
                    .active_filters
                    .iter()
                    .copied()
                    .map(|idx| obs_name(ds, idx))
                    .collect::<Vec<_>>()
                    .join(", ");
                dataset_lines.push(format!("Active filters: {}", filters));
            }
            if self.sample_grid_enabled {
                let grid_obs = self
                    .sample_grid_obs_idx
                    .map(|idx| obs_name(ds, idx))
                    .unwrap_or_else(|| "unknown".to_string());
                dataset_lines.push(format!(
                    "Sample grid: enabled (group by {grid_obs})"
                ));
            }
        }
        for line in dataset_lines {
            self.append_export_log(&line);
        }
    }

    fn finish_export_loop(&mut self) {
        let frames = self.export_total_frames;
        let frames_dir = self.export_dir.clone();
        self.export_camera = None;
        let keep_frames = self.export_keep_frames;
        if !self.export_run_ffmpeg {
            self.export_status = Some(format!(
                "Exported {frames} frames to {}",
                frames_dir.display()
            ));
            self.append_export_log("Export finished (ffmpeg disabled).");
            if !keep_frames {
                self.remove_export_frames(&frames_dir);
            }
            return;
        }

        let out_path = self.output_dir.join(self.export_name.trim());
        if let Some(ffmpeg_bin) = self.ffmpeg_path.clone() {
            let pattern = frames_dir.join("frame_%06d.png").to_string_lossy().to_string();
            let out_str = out_path.to_string_lossy().to_string();
            let mut cmd = std::process::Command::new(&ffmpeg_bin);
            Self::apply_subprocess_flags(&mut cmd);
            let (crf, preset, pix_fmt) = match self.export_video_quality {
                ExportVideoQuality::Standard => (23, "medium", "yuv420p"),
                ExportVideoQuality::High => (18, "slow", "yuv420p"),
                ExportVideoQuality::Ultra => (14, "slow", "yuv420p"),
            };
            let output = cmd
                .arg("-y")
                .arg("-framerate")
                .arg(self.export_fps.to_string())
                .arg("-i")
                .arg(&pattern)
                .arg("-c:v")
                .arg("libx264")
                .arg("-preset")
                .arg(preset)
                .arg("-crf")
                .arg(crf.to_string())
                .arg("-pix_fmt")
                .arg(pix_fmt)
                .arg(&out_str)
                .output();

            match output {
                Ok(out) if out.status.success() => {
                    if keep_frames {
                        self.export_status = Some(format!(
                            "Wrote video: {} (kept frames)",
                            out_path.display()
                        ));
                        self.append_export_log("Video render succeeded (frames kept).");
                    } else {
                        self.export_status = Some(format!("Wrote video: {}", out_path.display()));
                        self.append_export_log("Video render succeeded; cleaning PNG frames.");
                        self.remove_export_frames(&frames_dir);
                    }
                }
                Ok(out) => {
                    let status = out.status;
                    self.export_status = Some(format!("ffmpeg failed: {}", status));
                    self.append_export_log(&format!("ffmpeg failed: {}", status));
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    if !stdout.trim().is_empty() {
                        self.append_export_log("ffmpeg stdout:");
                        self.append_export_log(stdout.trim());
                    }
                    if !stderr.trim().is_empty() {
                        self.append_export_log("ffmpeg stderr:");
                        self.append_export_log(stderr.trim());
                    }
                }
                Err(e) => {
                    self.export_status = Some(format!("ffmpeg error: {e}"));
                    self.append_export_log(&format!("ffmpeg error: {e}"));
                }
            }
            return;
        }

        self.append_export_log("ffmpeg not found; trying OpenCV fallback.");
        match self.run_python_cv2_export(&frames_dir, &out_path) {
            Ok(()) => {
                if keep_frames {
                    self.export_status = Some(format!(
                        "Wrote video (OpenCV): {} (kept frames)",
                        out_path.display()
                    ));
                    self.append_export_log("OpenCV export succeeded (frames kept).");
                } else {
                    self.export_status = Some(format!("Wrote video (OpenCV): {}", out_path.display()));
                    self.append_export_log("OpenCV export succeeded; cleaning PNG frames.");
                    self.remove_export_frames(&frames_dir);
                }
            }
            Err(e) => {
                self.export_status = Some(format!(
                    "Exported {frames} frames to {} (OpenCV fallback failed)",
                    frames_dir.display()
                ));
                self.append_export_log(&format!("OpenCV fallback failed: {e}"));
            }
        }
    }

    fn cancel_export_loop(&mut self) {
        if !self.exporting_loop && !self.export_finishing {
            return;
        }
        self.exporting_loop = false;
        self.export_finishing = false;
        self.export_cancelled = true;
        self.export_camera = None;
        self.export_status = Some("Export cancelled.".to_string());
        self.append_export_log("Export cancelled.");
        if self.export_pending_frames == 0 {
            self.export_cancelled = false;
            let frames_dir = self.export_dir.clone();
            self.remove_export_frames(&frames_dir);
            self.remove_export_output();
        }
    }

    fn remove_export_frames(&mut self, frames_dir: &Path) {
        if let Err(e) = std::fs::remove_dir_all(frames_dir) {
            self.append_export_log(&format!(
                "Failed to remove frames: {e} ({})",
                frames_dir.display()
            ));
        } else {
            self.append_export_log(&format!(
                "Removed frames directory: {}",
                frames_dir.display()
            ));
        }
    }

    fn remove_export_output(&mut self) {
        let Some(path) = self.export_output_path.take() else {
            return;
        };
        if path.exists() {
            if let Err(e) = std::fs::remove_file(&path) {
                self.append_export_log(&format!(
                    "Failed to remove output: {e} ({})",
                    path.display()
                ));
            } else {
                self.append_export_log(&format!(
                    "Removed output file: {}",
                    path.display()
                ));
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
                    if req.is_export_frame {
                        self.export_pending_frames = self.export_pending_frames.saturating_sub(1);
                    }
                } else if let Some(any) = user_data
                    .data
                    .as_ref()
                    .and_then(|u| u.downcast_ref::<PathBuf>().cloned())
                {
                    let _ = save_color_image_png(&image, &any, None);
                }
            }
        }
        if self.export_finishing && self.export_pending_frames == 0 {
            self.export_finishing = false;
            self.finish_export_loop();
        }
        if self.export_cancelled && self.export_pending_frames == 0 {
            self.export_cancelled = false;
            let frames_dir = self.export_dir.clone();
            self.remove_export_frames(&frames_dir);
            self.remove_export_output();
        }
    }

    fn request_screenshot(
        &mut self,
        ctx: &egui::Context,
        path: PathBuf,
        crop_px: Option<[u32; 4]>,
        is_export_frame: bool,
    ) {
        let req = ScreenshotRequest {
            path,
            crop_px,
            is_export_frame,
        };
        ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::new(req)));
    }

    fn append_export_log(&mut self, line: &str) {
        if !self.export_log_text.is_empty() {
            self.export_log_text.push('\n');
        }
        self.export_log_text.push_str(line);
        if let Some(path) = self.export_log_path.as_ref() {
            let _ = std::fs::write(path, &self.export_log_text);
        }
    }

    fn export_resolution_for_quality(&self) -> Option<[u32; 2]> {
        match self.export_quality {
            ExportQuality::Current => None,
            ExportQuality::FullHd => Some([1920, 1080]),
            ExportQuality::UltraHd => Some([3840, 2160]),
        }
    }

    fn render_export_frame(
        &mut self,
        size: [u32; 2],
        path: &Path,
        use_export_bbox: bool,
    ) -> Result<(), String> {
        let render_state = self
            .render_state
            .clone()
            .ok_or_else(|| "Render state unavailable for high-quality export.".to_string())?;
        let device = &render_state.device;
        let queue = &render_state.queue;
        let target_format = render_state.target_format;

        let params = self.shared.params.lock();
        let Some(ds) = params.dataset.clone() else {
            return Err("No dataset loaded for export.".to_string());
        };
        let dataset_id = params.dataset_id;
        let colors_id = params.colors_id;
        let colors_rgba8 = params.colors_rgba8.clone();
        let colors_to_id = params.colors_to_id;
        let colors_to_rgba8 = params.colors_to_rgba8.clone();
        let indices_id = params.indices_id;
        let draw_indices = params.draw_indices.clone();
        let from_override = params.from_override.clone();
        let to_override = params.to_override.clone();
        let from_override_id = params.from_override_id;
        let to_override_id = params.to_override_id;
        let use_opaque = params.use_opaque;
        drop(params);

        let viewport_px = [size[0] as f32, size[1] as f32];
        let mut view_camera = self.camera;
        if use_export_bbox {
            if let Some(bbox) = self.export_fit_bbox(&ds) {
                let mut cam = Camera2D::default();
                cam.fit_bbox(bbox, viewport_px, 0.98);
                view_camera = cam;
            }
        }
        let (active_from, active_to, color_from, color_to, segment_t, _seg_idx) =
            self.current_segment(&ds);
        let from_space = active_from as u32;
        let to_space = active_to as u32;
        let (from_center, from_scale) = if let Some(space) = ds.meta.spaces.get(active_from) {
            self.space_transform(&ds, active_from, space)
        } else {
            ([0.0, 0.0], 1.0)
        };
        let (to_center, to_scale) = if let Some(space) = ds.meta.spaces.get(active_to) {
            self.space_transform(&ds, active_to, space)
        } else {
            ([0.0, 0.0], 1.0)
        };

        let t_eased = apply_ease(segment_t, self.ease_mode);
        let color_t = if color_from != color_to { t_eased } else { 0.0 };
        let mut point_radius_px = self.point_radius_px;
        if self.last_viewport_points.width() > 0.0 {
            let ppp = self.last_viewport_px[0] / self.last_viewport_points.width();
            point_radius_px *= ppp;
        }
        if self.last_viewport_px[0] > 0.0 && self.last_viewport_px[1] > 0.0 {
            let scale_x = size[0] as f32 / self.last_viewport_px[0];
            let scale_y = size[1] as f32 / self.last_viewport_px[1];
            point_radius_px *= scale_x.min(scale_y);
        }
        let uniforms = Uniforms {
            viewport_px,
            _pad0: [0.0; 2],
            center: view_camera.center,
            _pad1: [0.0; 2],
            pixels_per_unit: view_camera.pixels_per_unit,
            t: t_eased,
            point_radius_px,
            mask_mode: 1.0,
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

        let sample_count = 4u32;
        let use_msaa = sample_count > 1;
        let gpu = if use_msaa {
            if self.offscreen_gpu_msaa.is_none() {
                self.offscreen_gpu_msaa =
                    Some(PointCloudGpu::new_with_sample_count(device, target_format, sample_count));
            }
            self.offscreen_gpu_msaa.as_mut().unwrap()
        } else {
            if self.offscreen_gpu.is_none() {
                self.offscreen_gpu = Some(PointCloudGpu::new(device, target_format));
            }
            self.offscreen_gpu.as_mut().unwrap()
        };
        let _ = gpu.prepare(
            device,
            queue,
            target_format,
            &ds,
            dataset_id,
            from_space,
            to_space,
            colors_id,
            &colors_rgba8,
            colors_to_id,
            &colors_to_rgba8,
            indices_id,
            &draw_indices,
            from_override.as_deref().map(|v| v.as_slice()),
            to_override.as_deref().map(|v| v.as_slice()),
            from_override_id,
            to_override_id,
            uniforms,
        );

        let resolve_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("export_frame_resolve"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let resolve_view = resolve_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let msaa_texture = if use_msaa {
            Some(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("export_frame_msaa"),
                size: wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: wgpu::TextureDimension::D2,
                format: target_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            }))
        } else {
            None
        };
        let msaa_view = msaa_texture
            .as_ref()
            .map(|tex| tex.create_view(&wgpu::TextureViewDescriptor::default()));
        let color_view = msaa_view.as_ref().unwrap_or(&resolve_view);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("export_encoder"),
        });
        let bg = self.background_color;
        let clear = wgpu::Color {
            r: bg.r() as f64 / 255.0,
            g: bg.g() as f64 / 255.0,
            b: bg.b() as f64 / 255.0,
            a: bg.a() as f64 / 255.0,
        };
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("export_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    depth_slice: None,
                    resolve_target: if use_msaa { Some(&resolve_view) } else { None },
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            gpu.paint(&mut render_pass, use_opaque);
        }

        let bytes_per_pixel = 4u32;
        let unpadded_bytes_per_row = size[0] * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = ((unpadded_bytes_per_row + align - 1) / align) * align;
        let buffer_size = padded_bytes_per_row as u64 * size[1] as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("export_buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &resolve_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(size[1]),
                },
            },
            wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        let buffer_slice = output_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = sender.send(v);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        receiver
            .recv()
            .map_err(|_| "Failed to read export buffer.".to_string())?
            .map_err(|e| format!("Export buffer map failed: {e:?}"))?;

        let data = buffer_slice.get_mapped_range();
        let mut rgba = vec![0u8; (size[0] * size[1] * bytes_per_pixel) as usize];
        let is_bgra = matches!(
            target_format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );
        for row in 0..size[1] as usize {
            let src = row * padded_bytes_per_row as usize;
            let dst = row * unpadded_bytes_per_row as usize;
            let row_slice = &data[src..src + unpadded_bytes_per_row as usize];
            if is_bgra {
                for (i, chunk) in row_slice.chunks_exact(4).enumerate() {
                    let idx = dst + i * 4;
                    rgba[idx] = chunk[2];
                    rgba[idx + 1] = chunk[1];
                    rgba[idx + 2] = chunk[0];
                    rgba[idx + 3] = chunk[3];
                }
            } else {
                rgba[dst..dst + unpadded_bytes_per_row as usize].copy_from_slice(row_slice);
            }
        }
        drop(data);
        output_buffer.unmap();

        save_rgba_png(path, size[0], size[1], &rgba)
            .map_err(|e| format!("Export PNG failed: {e:#}"))?;
        Ok(())
    }

    fn ui_left_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("stviz-animate");

        ui.separator();
        ui.label("Mock dataset");
        ui.horizontal(|ui| {
            ui.label("Cells");
            ui.add(
                egui::DragValue::new(&mut self.mock_cells)
                    .speed(1_000.0)
                    .range(10_000..=10_000_000),
            );
            ui.label("10k - 10M");
        });
        ui.horizontal(|ui| {
            let generate = ui.add_enabled(
                !self.convert_running,
                egui::Button::new("Generate mock dataset"),
            );
            if generate.clicked() {
                self.start_mock_dataset();
            }
            let regen = ui.add_enabled(
                !self.convert_running && self.mock_last_h5ad.is_some(),
                egui::Button::new("Regenerate mock dataset"),
            );
            if regen.clicked() {
                self.start_mock_dataset();
            }
        });

        ui.label("Note: conversion may take a minute or two for datasets >1GB.");
        ui.label("First-time converter setup can take a few minutes; it only happens once.");
        ui.checkbox(
            &mut self.convert_include_expr,
            "Include gene expression data (uncheck for smaller file size)",
        );
        ui.checkbox(
            &mut self.convert_generate_only,
            "Generate .stviz file only - don't load",
        );
        if let Some(status) = self.convert_status.as_ref() {
            if self.convert_running {
                ui.colored_label(egui::Color32::from_rgb(240, 200, 90), status);
            } else {
                ui.label(status);
            }
        }

        ui.separator();
        ui.label("Convert .h5ad or load .stviz");
        let drop_height = 56.0;
        let drop_size = egui::vec2(ui.available_width(), drop_height);
        let (drop_rect, drop_resp) = ui.allocate_exact_size(drop_size, egui::Sense::click());
        let hover_files = ctx.input(|i| !i.raw.hovered_files.is_empty());
        let drop_active = drop_resp.hovered() || hover_files;
        let visuals = ui.visuals();
        let busy_fill = egui::Color32::from_rgb(60, 90, 140);
        let busy_stroke = egui::Stroke::new(1.2, egui::Color32::from_rgb(120, 170, 220));
        let drop_fill = if self.convert_running {
            busy_fill
        } else if drop_active {
            visuals.widgets.hovered.bg_fill
        } else {
            visuals.widgets.inactive.bg_fill
        };
        let drop_stroke = if self.convert_running {
            busy_stroke
        } else if drop_active {
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
            "Drop .h5ad or .stviz here (or click to pick)"
        };
        let drop_text_color = if self.convert_running {
            egui::Color32::from_rgb(230, 240, 255)
        } else {
            visuals.text_color()
        };
        ui.painter().text(
            drop_rect.center(),
            egui::Align2::CENTER_CENTER,
            drop_label,
            egui::FontId::proportional(13.0),
            drop_text_color,
        );
        let drop_resp =
            drop_resp.on_hover_text("Drop a .h5ad to convert, or a .stviz to load.");
        if drop_resp.clicked() && !self.convert_running {
            let dialog = rfd::FileDialog::new()
                .add_filter("h5ad", &["h5ad"])
                .add_filter("stviz", &["stviz"])
                .set_title("Select .h5ad or .stviz")
                .set_directory(self.project_dir.clone());
            if let Some(path) = dialog.pick_file() {
                match path.extension().and_then(|ext| ext.to_str()).map(|ext| ext.to_ascii_lowercase()).as_deref() {
                    Some("h5ad") => self.queue_h5ad_convert(&path),
                    Some("stviz") => {
                        if let Err(e) = self.load_dataset(&path) {
                            let msg = format!("Load failed: {e:#}");
                            eprintln!("{msg}");
                            self.last_error = Some(msg);
                        }
                    }
                    _ => {
                        self.convert_status =
                            Some("Pick a .h5ad or .stviz file.".to_string());
                    }
                }
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
                        let ext = path
                            .extension()
                            .and_then(|ext| ext.to_str())
                            .map(|ext| ext.to_ascii_lowercase());
                        match ext.as_deref() {
                            Some("h5ad") => {
                                self.queue_h5ad_convert(&path);
                                handled = true;
                                break;
                            }
                            Some("stviz") => {
                                if let Err(e) = self.load_dataset(&path) {
                                    let msg = format!("Load failed: {e:#}");
                                    eprintln!("{msg}");
                                    self.last_error = Some(msg);
                                }
                                handled = true;
                                break;
                            }
                            _ => {}
                        }
                    }
                }
                if !handled {
                    self.convert_status =
                        Some("Drop a .h5ad or .stviz file.".to_string());
                }
            }
        }

        ui.separator();
        egui::CollapsingHeader::new("Conversion log")
            .default_open(true)
            .show(ui, |ui| {
                if ui.button("Copy log").clicked() {
                    ui.ctx()
                        .copy_text(self.convert_log_text.clone());
                }
                let row_height = ui.text_style_height(&egui::TextStyle::Body);
                let max_height = row_height * 6.0 + 8.0;
                egui::ScrollArea::vertical()
                    .max_height(max_height)
                    .stick_to_bottom(self.convert_running)
                    .show(ui, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(&mut self.convert_log_text)
                                .desired_rows(6)
                                .interactive(true)
                                .cursor_at_end(true)
                                .desired_width(f32::INFINITY),
                        );
                    });
            });

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
        ui.label("View controls");

        ui.horizontal(|ui| {
            ui.label("UI scale");
            let presets = [75, 90, 100, 110, 125, 150, 175, 200];
            let mut selected = (self.ui_scale * 100.0).round() as i32;
            egui::ComboBox::from_id_salt("ui_scale_presets")
                .selected_text(format!("{selected}%"))
                .show_ui(ui, |ui| {
                    for pct in presets {
                        if ui.selectable_value(&mut selected, pct, format!("{pct}%")).clicked() {
                            self.ui_scale = pct as f32 / 100.0;
                        }
                    }
                });
            let mut scale_pct = self.ui_scale * 100.0;
            if ui
                .add(
                    egui::DragValue::new(&mut scale_pct)
                        .range(50.0..=250.0)
                        .speed(1.0)
                        .suffix("%"),
                )
                .changed()
            {
                self.ui_scale = (scale_pct / 100.0).clamp(0.5, 2.5);
            }
        });
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
        ui.horizontal(|ui| {
            if Self::is_wsl() {
                self.fullscreen = false;
                ui.add_enabled(false, egui::Checkbox::new(&mut self.fullscreen, "Fullscreen"))
                    .on_hover_text("Fullscreen is not supported on WSL.");
            } else {
                let changed = ui.checkbox(&mut self.fullscreen, "Fullscreen").changed();
                if changed {
                    self.apply_fullscreen(ctx);
                }
            }
        });
        ui.horizontal(|ui| {
            if Self::is_wsl() {
                self.viewport_fullscreen = false;
                ui.add_enabled(
                    false,
                    egui::Checkbox::new(&mut self.viewport_fullscreen, "Viewport fullscreen"),
                )
                .on_hover_text("Fullscreen is not supported on WSL.");
            } else {
                let changed = ui
                    .checkbox(&mut self.viewport_fullscreen, "Viewport fullscreen")
                    .changed();
                if changed {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(
                        self.viewport_fullscreen,
                    ));
                }
            }
        });

        ui.separator();
        ui.label("Loaded dataset");

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
        ui.checkbox(&mut self.show_axes, "Show axes");

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
            egui::ComboBox::from_label("Grid space (most likely spatial)")
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

            ui.checkbox(&mut self.sample_grid_labels_enabled, "Show sample labels");
            if self.sample_grid_labels_enabled {
                let label_mode = match self.sample_grid_label_mode {
                    SampleGridLabelMode::Default => "Sample (default)",
                    SampleGridLabelMode::Custom => "Custom labels",
                };
                egui::ComboBox::from_label("Label source")
                    .selected_text(label_mode)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.sample_grid_label_mode,
                            SampleGridLabelMode::Default,
                            "Sample (default)",
                        );
                        ui.selectable_value(
                            &mut self.sample_grid_label_mode,
                            SampleGridLabelMode::Custom,
                            "Custom labels",
                        );
                    });

                if self.sample_grid_label_mode == SampleGridLabelMode::Custom {
                    if let Some(obs_idx) = self.sample_grid_obs_idx {
                        if let Ok((_name, _labels, categories, _pal)) =
                            ds.obs_categorical(obs_idx)
                        {
                            if categories.len() <= MAX_GRID_CATEGORIES {
                                self.ensure_sample_grid_custom_labels(obs_idx, &categories);
                                if ui.button("Clear custom labels").clicked() {
                                    self.sample_grid_custom_labels = categories.to_vec();
                                }
                                egui::ScrollArea::vertical()
                                    .max_height(140.0)
                                    .show(ui, |ui| {
                                        for (i, cat) in categories.iter().enumerate() {
                                            ui.horizontal(|ui| {
                                                ui.label(cat);
                                                if let Some(val) =
                                                    self.sample_grid_custom_labels.get_mut(i)
                                                {
                                                    ui.text_edit_singleline(val);
                                                }
                                            });
                                        }
                                    });
                            }
                        }
                    }
                }
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
                        .width(200.0)
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
                        self.filter_popup_open = false;
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

                    let mut palette_changed = false;
                    let mut colors_changed = false;
                    ui.horizontal(|ui| {
                        ui.label("Palette");
                        let selected = Self::categorical_palette_label(self.categorical_palette);
                        egui::ComboBox::from_id_salt("cat_palette")
                            .width(200.0)
                            .selected_text(selected)
                            .show_ui(ui, |ui| {
                                let options = [
                                    CategoricalPalette::Tableau10,
                                    CategoricalPalette::Tab10,
                                    CategoricalPalette::Tab20,
                                    CategoricalPalette::Category10,
                                    CategoricalPalette::Set1,
                                    CategoricalPalette::Set2,
                                    CategoricalPalette::Set3,
                                    CategoricalPalette::Dark2,
                                    CategoricalPalette::Accent,
                                    CategoricalPalette::Paired,
                                    CategoricalPalette::Pastel1,
                                    CategoricalPalette::Pastel2,
                                    CategoricalPalette::Turbo,
                                    CategoricalPalette::Dataset,
                                ];
                                for pal in options {
                                    if ui
                                        .selectable_value(
                                            &mut self.categorical_palette,
                                            pal,
                                            Self::categorical_palette_label(pal),
                                        )
                                        .changed()
                                    {
                                        palette_changed = true;
                                    }
                                }
                            });
                        if ui.button("Reset category colors").clicked() {
                            self.category_overrides.remove(&self.active_obs_idx);
                            palette_changed = true;
                        }
                    });

                    ui.separator();
                    ui.label("Filter categories");

                    if let Ok((_name, _labels, categories, pal_opt)) =
                        ds.obs_categorical(self.active_obs_idx)
                    {
                        let too_many = categories.len() > MAX_FILTER_CATEGORIES;
                        if !too_many && self.enabled_categories.len() != categories.len() {
                            self.enabled_categories = vec![true; categories.len()];
                        }
                        let palette_slice: Option<&[u32]> = if too_many {
                            None
                        } else {
                            let needs_rebuild = match self.category_palette_cache.as_ref() {
                                Some(cache) => {
                                    cache.obs_idx != self.active_obs_idx
                                        || cache.categories_len != categories.len()
                                        || cache.palette != self.categorical_palette
                                        || cache.has_dataset_palette != pal_opt.is_some()
                                }
                                None => true,
                            };
                            if needs_rebuild {
                                let colors = self.categorical_palette_for(
                                    self.active_obs_idx,
                                    categories.len(),
                                    pal_opt,
                                );
                                self.category_palette_cache = Some(CategoryPaletteCache {
                                    obs_idx: self.active_obs_idx,
                                    categories_len: categories.len(),
                                    palette: self.categorical_palette,
                                    has_dataset_palette: pal_opt.is_some(),
                                    colors,
                                });
                            }
                            self.category_palette_cache
                                .as_ref()
                                .map(|cache| cache.colors.as_slice())
                        };
                        if self.categorical_palette == CategoricalPalette::Dataset
                            && pal_opt.is_none()
                        {
                            ui.label("No dataset palette found; using Tableau10.");
                        }
                        let mut filter_changed = false;
                        if too_many {
                            ui.colored_label(
                                egui::Color32::YELLOW,
                                format!(
                                    "Too many categories ({}). Filtering disabled for this field.",
                                    categories.len()
                                ),
                            );
                        } else {
                            let obs_idx = self.active_obs_idx;
                            let categories_len = categories.len();
                            let (enabled_categories, category_overrides) =
                                (&mut self.enabled_categories, &mut self.category_overrides);
                            let overrides = category_overrides
                                .entry(obs_idx)
                                .or_insert_with(|| vec![None; categories_len]);
                            if overrides.len() != categories_len {
                                overrides.resize(categories_len, None);
                            }
                            let row_height = ui.spacing().interact_size.y.max(20.0);
                            if categories.len() <= 10 {
                                egui::Frame::group(ui.style()).show(ui, |ui| {
                                    render_filter_rows(
                                        ui,
                                        0..categories.len(),
                                        categories,
                                        palette_slice,
                                        enabled_categories,
                                        overrides,
                                        &mut filter_changed,
                                        &mut colors_changed,
                                    );
                                });
                            } else {
                                let preview_max = row_height * 10.0 + 8.0;
                                egui::ScrollArea::vertical()
                                    .max_height(preview_max)
                                    .min_scrolled_height(preview_max)
                                    .show_rows(ui, row_height, categories.len(), |ui, range| {
                                        render_filter_rows(
                                            ui,
                                            range,
                                            categories,
                                            palette_slice,
                                            enabled_categories,
                                            overrides,
                                            &mut filter_changed,
                                            &mut colors_changed,
                                        );
                                    });
                                ui.horizontal(|ui| {
                                    ui.label(format!("{} categories", categories.len()));
                                    if ui.button("Show full list").clicked() {
                                        self.filter_popup_open = true;
                                    }
                                });
                            }
                            if self.filter_popup_open {
                                let mut open = self.filter_popup_open;
                                egui::Window::new("Filter categories")
                                    .open(&mut open)
                                    .resizable(true)
                                    .min_width(260.0)
                                    .show(ui.ctx(), |ui| {
                                        ui.horizontal(|ui| {
                                            if ui.button("All").clicked() {
                                                for v in &mut *enabled_categories {
                                                    *v = true;
                                                }
                                                filter_changed = true;
                                            }
                                            if ui.button("None").clicked() {
                                                for v in &mut *enabled_categories {
                                                    *v = false;
                                                }
                                                filter_changed = true;
                                            }
                                        });
                                        let list_max = (row_height * categories.len() as f32 + 8.0)
                                            .min(ui.ctx().available_rect().height() * 0.7);
                                        egui::ScrollArea::vertical()
                                            .max_height(list_max)
                                            .show_rows(ui, row_height, categories.len(), |ui, range| {
                                                render_filter_rows(
                                                    ui,
                                                    range,
                                                    categories,
                                                    palette_slice,
                                                    enabled_categories,
                                                    overrides,
                                                    &mut filter_changed,
                                                    &mut colors_changed,
                                                );
                                            });
                                    });
                                self.filter_popup_open = open;
                            }
                        }

                        ui.horizontal(|ui| {
                            if !too_many {
                                if ui.button("All").on_hover_text("Enable all categories.").clicked() {
                                    for v in &mut self.enabled_categories {
                                        *v = true;
                                    }
                                    filter_changed = true;
                                }
                                if ui.button("None").on_hover_text("Disable all categories.").clicked() {
                                    for v in &mut self.enabled_categories {
                                        *v = false;
                                    }
                                    filter_changed = true;
                                }
                                let all_on = self.enabled_categories.iter().all(|v| *v);
                                let desired_active = !all_on;
                                let active_now = self.active_filters.contains(&self.active_obs_idx);
                                let state_matches = self
                                    .category_state
                                    .get(&self.active_obs_idx)
                                    .map(|s| s == &self.enabled_categories)
                                    .unwrap_or(false);
                                let needs_apply =
                                    filter_changed || desired_active != active_now || (desired_active && !state_matches);
                                if needs_apply {
                                    if desired_active {
                                        self.category_state
                                            .insert(self.active_obs_idx, self.enabled_categories.clone());
                                        self.active_filters.insert(self.active_obs_idx);
                                    } else {
                                        self.active_filters.remove(&self.active_obs_idx);
                                        self.category_state.remove(&self.active_obs_idx);
                                    }
                                    let _ = self.recompute_draw_indices_with_filters();
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
                    if palette_changed || colors_changed {
                        self.category_palette_cache = None;
                        self.color_cache.clear();
                        let _ = self.recompute_colors_and_filters();
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
                        .width(200.0)
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
                    let gene_row_height = ui.spacing().interact_size.y.max(20.0);
                    let gene_max = gene_row_height * 10.0 + 8.0;
                    egui::ScrollArea::vertical()
                        .max_height(gene_max)
                        .show(ui, |ui| {
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
                                self.colors_opaque = true;
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

            }
        }

        let legend = self.active_legend_range.as_ref().or(self.legend_range.as_ref());
        if let Some(legend) = legend {
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
        ui.checkbox(&mut self.show_stats, "Show stats");

        ui.separator();
        let mut output_frame = egui::Frame::group(ui.style()).inner_margin(egui::Margin::same(6));
        if self.exporting_loop {
            output_frame = output_frame
                .fill(egui::Color32::from_rgb(40, 60, 90))
                .stroke(egui::Stroke::new(1.2, egui::Color32::from_rgb(120, 170, 220)));
        }
        output_frame.show(ui, |ui| {
            if self.exporting_loop {
                ui.colored_label(egui::Color32::from_rgb(230, 240, 255), "Output (busy)");
            } else {
                ui.label("Output");
            }
            if ui.button("Screenshot").clicked() {
                let ts = chrono_like_timestamp();
                let path = self.screenshot_dir.join(format!("stviz-animate_screenshot_{ts}.png"));
                if let Err(err) = self.render_export_frame(SCREENSHOT_RESOLUTION, &path, false) {
                    self.export_status = Some(format!("Screenshot failed: {err}"));
                }
            }
            ui.add_space(6.0);
            ui.label("Loop export");
            ui.add(egui::DragValue::new(&mut self.export_fps).range(1..=240).prefix("fps "));
            ui.add(
                egui::DragValue::new(&mut self.export_duration_sec)
                    .range(0.5..=120.0)
                    .speed(0.5)
                    .suffix(" sec")
                    .prefix("duration "),
            );
            ui.horizontal(|ui| {
                ui.label("Quality");
                let label = match self.export_quality {
                    ExportQuality::Current => "Current viewport",
                    ExportQuality::FullHd => "Full HD (1920x1080)",
                    ExportQuality::UltraHd => "4K (3840x2160)",
                };
                egui::ComboBox::from_id_salt("export_quality")
                    .selected_text(label)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.export_quality,
                            ExportQuality::Current,
                            "Current viewport",
                        );
                        ui.selectable_value(
                            &mut self.export_quality,
                            ExportQuality::FullHd,
                            "Full HD (1920x1080)",
                        );
                        ui.selectable_value(
                            &mut self.export_quality,
                            ExportQuality::UltraHd,
                            "4K (3840x2160)",
                        );
                    });
            });
            ui.horizontal(|ui| {
                ui.label("Encoding");
                let label = match self.export_video_quality {
                    ExportVideoQuality::Standard => "Standard (CRF 23, yuv420p)",
                    ExportVideoQuality::High => "High (CRF 18, yuv420p)",
                ExportVideoQuality::Ultra => "Ultra (CRF 14, yuv420p)",
            };
                egui::ComboBox::from_id_salt("export_video_quality")
                    .selected_text(label)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.export_video_quality,
                            ExportVideoQuality::Standard,
                            "Standard (CRF 23, yuv420p)",
                        );
                        ui.selectable_value(
                            &mut self.export_video_quality,
                            ExportVideoQuality::High,
                            "High (CRF 18, yuv420p)",
                        );
                        ui.selectable_value(
                            &mut self.export_video_quality,
                            ExportVideoQuality::Ultra,
                            "Ultra (CRF 14, yuv420p)",
                        );
                    });
            });
            ui.horizontal(|ui| {
                ui.label("Output");
                ui.text_edit_singleline(&mut self.export_name);
            });
            if ui.checkbox(&mut self.export_run_ffmpeg, "Run ffmpeg if available").changed() {
                self.refresh_ffmpeg_path();
            }
            if self.export_run_ffmpeg && !self.ffmpeg_available {
                ui.colored_label(
                    egui::Color32::YELLOW,
                    "ffmpeg not found (will try OpenCV fallback).",
                );
            }
            let keep_forced = !self.export_run_ffmpeg;
            let mut keep_pref = self.export_keep_frames;
            let keep_label = if keep_forced {
                "Keep PNG frames (required without video export)"
            } else {
                "Keep PNG frames"
            };
            if ui
                .add_enabled(!keep_forced, egui::Checkbox::new(&mut keep_pref, keep_label))
                .changed()
            {
                self.export_keep_frames = keep_pref;
            }
            if ui
                .button(if self.exporting_loop { "Exporting..." } else { "Export loop video" })
                .clicked()
            {
                self.start_export_loop();
            }
            if (self.exporting_loop || self.export_finishing)
                && ui.button("Cancel export").clicked()
            {
                self.cancel_export_loop();
            }
            if let Some(status) = self.export_status.as_ref() {
                if self.exporting_loop {
                    ui.colored_label(egui::Color32::from_rgb(240, 200, 90), status);
                } else {
                    ui.label(status);
                }
            }
            if !self.export_log_text.trim().is_empty() {
                ui.separator();
                let mut log_rect = None;
                let open_override = if self.export_log_open { Some(true) } else { None };
                egui::CollapsingHeader::new("Export log")
                    .open(open_override)
                    .show(ui, |ui| {
                        if ui.button("Copy log").clicked() {
                            ui.ctx().copy_text(self.export_log_text.clone());
                        }
                        let response = ui.add(
                            egui::TextEdit::multiline(&mut self.export_log_text)
                                .desired_rows(6)
                                .cursor_at_end(true)
                                .interactive(true),
                        );
                        log_rect = Some(response);
                    });
                if self.export_log_open {
                    self.export_log_open = false;
                }
                if self.export_log_focus {
                    if let Some(response) = log_rect {
                        response.scroll_to_me(Some(egui::Align::BOTTOM));
                    }
                    self.export_log_focus = false;
                }
            }
        });
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
                visuals.panel_fill = egui::Color32::from_rgb(5, 9, 7);
                visuals.window_fill = egui::Color32::from_rgb(5, 9, 7);
                visuals.override_text_color = Some(egui::Color32::from_rgb(95, 210, 120));
                visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(8, 16, 11);
                visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(12, 24, 16);
                visuals.widgets.active.bg_fill = egui::Color32::from_rgb(16, 32, 20);
                visuals.widgets.inactive.bg_stroke =
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(60, 150, 90));
                visuals.widgets.hovered.bg_stroke =
                    egui::Stroke::new(1.2, egui::Color32::from_rgb(90, 210, 130));
                visuals.widgets.active.bg_stroke =
                    egui::Stroke::new(1.4, egui::Color32::from_rgb(120, 245, 160));
                visuals.widgets.inactive.fg_stroke =
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(70, 170, 105));
                visuals.widgets.hovered.fg_stroke =
                    egui::Stroke::new(1.2, egui::Color32::from_rgb(110, 230, 150));
                visuals.selection.bg_fill = egui::Color32::from_rgb(0, 90, 50);
                visuals.selection.stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(110, 230, 150));
                visuals.hyperlink_color = egui::Color32::from_rgb(80, 190, 110);
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
            self.t = export_phase_to_t(phase, self.playback_mode, 1.0);
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
        self.last_viewport_px = viewport_px;

        let ds_opt = self.dataset.clone();
        let view_camera = self.camera;
        let (active_from, active_to, color_from, color_to, segment_t, seg_idx) =
            if let Some(ds) = ds_opt.as_ref() {
                self.current_segment(ds)
            } else {
                (
                    0,
                    0,
                    ColorKey::Current,
                    ColorKey::Current,
                    self.t.clamp(0.0, 1.0),
                    0,
                )
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
        let mut colors_from_opaque = self.colors_opaque;
        let mut colors_to_opaque = self.colors_opaque;

        if let Some(ds) = ds_opt.as_ref() {
            if self.color_path_enabled {
                let (cf, cf_id, cf_legend, cf_opaque) = self.colors_for_key(ds, &color_from);
                let (ct, ct_id, _ct_legend, ct_opaque) = self.colors_for_key(ds, &color_to);
                colors_from = cf;
                colors_from_id = cf_id;
                colors_to = ct;
                colors_to_id = ct_id;
                colors_from_opaque = cf_opaque;
                colors_to_opaque = ct_opaque;
                if color_from != color_to {
                    color_t = t_eased;
                }
                legend_from = cf_legend;
            }
        }

        let mut draw_indices = self.draw_indices.clone();
        let mut indices_id = self.indices_id;
        if let Some(ds) = ds_opt.as_ref() {
            let mut override_idx = None;
            if self.advanced_timeline_open && !self.playing {
                if let Some(preview_idx) = self.advanced_preview_card {
                    override_idx = Some(preview_idx);
                }
            }
            if let Some(idx) = override_idx {
                if let Some((indices, id)) = self.advanced_draw_override(ds, idx) {
                    draw_indices = indices;
                    indices_id = id;
                }
            } else if let Some((indices, id)) = self.advanced_draw_override(ds, seg_idx) {
                draw_indices = indices;
                indices_id = id;
            }
        }

        self.active_legend_range = if self.color_path_enabled {
            legend_from.clone()
        } else {
            self.legend_range.clone()
        };

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
            p.indices_id = indices_id;
            p.draw_indices = draw_indices;
            p.uniforms = uniforms;
            p.use_opaque = colors_from_opaque && colors_to_opaque;
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

        if self.exporting_loop && self.export_total_frames > 0 {
            let path = self.export_dir.join(format!("frame_{:06}.png", self.export_frame_index));
            if let Some(res) = self.export_resolution {
                if let Err(err) = self.render_export_frame(res, &path, true) {
                    self.append_export_log(&format!("Export render failed: {err}"));
                    self.export_status = Some(format!("Export failed: {err}"));
                    self.exporting_loop = false;
                    self.export_camera = None;
                    return;
                }
            } else {
                let crop = Self::viewport_crop_px_even(rect, ppp);
                self.export_pending_frames = self.export_pending_frames.saturating_add(1);
                self.request_screenshot(ctx, path, crop, true);
            }
            self.export_frame_index += 1;
            if self.export_frame_index >= self.export_total_frames {
                self.exporting_loop = false;
                self.export_camera = None;
                if self.export_resolution.is_some() {
                    self.finish_export_loop();
                } else {
                    self.export_finishing = true;
                    self.export_status = Some("Finalizing export...".to_string());
                }
            }
        }

        // Keep repainting while playing/exporting
        if self.playing || self.exporting_loop || self.export_finishing {
            ctx.request_repaint_after(self.main_view_frame_interval());
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

    fn grid_positions_for(
        &mut self,
        ds: &Dataset,
        space_idx: usize,
    ) -> Option<(Arc<Vec<f32>>, [f32; 4])> {
        self.grid_positions_for_space(ds, space_idx, false)
    }

    fn grid_positions_for_space(
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
            let grid_dim = (bbox[2] - bbox[0]).max(1e-6).max((bbox[3] - bbox[1]).max(1e-6));
            if max_dim > grid_dim {
                let scale = max_dim / grid_dim;
                let cx = 0.5 * (bbox[0] + bbox[2]);
                let cy = 0.5 * (bbox[1] + bbox[3]);
                bbox = [f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
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

    fn ensure_sample_grid_custom_labels(&mut self, obs_idx: usize, categories: &[String]) {
        if self.sample_grid_custom_labels_obs_idx != Some(obs_idx)
            || self.sample_grid_custom_labels.len() != categories.len()
        {
            self.sample_grid_custom_labels_obs_idx = Some(obs_idx);
            self.sample_grid_custom_labels = categories.to_vec();
        }
    }

    fn sample_grid_label_positions(
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
        let mut selected: Vec<usize> = if self.sample_grid_use_filter {
            if let Some(state) = self.category_state.get(&obs_idx) {
                state
                    .iter()
                    .enumerate()
                    .filter_map(|(i, v)| if *v { Some(i) } else { None })
                    .collect()
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

        let mut out = Vec::with_capacity(n_sel);
        for (i, &cat_idx) in selected.iter().enumerate() {
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
            let label_pos = [offsets[i][0], offsets[i][1] + tile_h * 0.55];
            out.push((label, label_pos));
        }
        Some(out)
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
        if self.reset_view_key_idx >= self.space_path.len() {
            self.reset_view_key_idx = self.space_path.len().saturating_sub(1);
        }
        if self.key_times.len() != self.space_path.len() {
            self.ensure_key_times_len(self.space_path.len());
        }
    }

    fn ensure_key_times_len(&mut self, len: usize) {
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
        self.key_times = (0..len)
            .map(|i| i as f32 / (len as f32 - 1.0))
            .collect();
        if self.key_collapsed.len() != len {
            self.key_collapsed.resize(len, false);
        }
    }

    fn distribute_key_times_evenly(&mut self) {
        let len = self.key_times.len();
        if len < 2 {
            return;
        }
        for (i, t) in self.key_times.iter_mut().enumerate() {
            *t = i as f32 / (len as f32 - 1.0);
        }
    }

    fn move_keyframe(&mut self, from: usize, to: usize) {
        let len = self.space_path.len();
        if len < 2 || from >= len || to >= len || from == to {
            return;
        }
        if self.color_path.len() != len {
            self.ensure_color_path_len(len);
        }
        if self.key_times.len() != len {
            self.ensure_key_times_len(len);
        }
        if self.key_collapsed.len() != len {
            self.key_collapsed.resize(len, false);
        }

        let space = self.space_path.remove(from);
        let color = self.color_path.remove(from);
        let time = self.key_times.remove(from);
        let collapsed = self.key_collapsed.remove(from);

        self.space_path.insert(to, space);
        self.color_path.insert(to, color);
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

    fn remove_keyframe(&mut self, idx: usize) {
        if self.space_path.len() <= 2 || idx >= self.space_path.len() {
            return;
        }
        self.space_path.remove(idx);
        if idx < self.color_path.len() {
            self.color_path.remove(idx);
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

    fn compute_colors_for_key(
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
                        pal.get(idx).copied().unwrap_or(pack_rgba8(200, 200, 200, 255))
                    } else {
                        pal.get(li).copied().unwrap_or(pack_rgba8(200, 200, 200, 255))
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
                Some((colors, legend, true))
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
                Some((colors, legend, true))
            }
        }
    }

    fn colors_for_key(
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

    fn current_segment(&self, ds: &Dataset) -> (usize, usize, ColorKey, ColorKey, f32, usize) {
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

    fn draw_view_overlays(&mut self, ui: &egui::Ui, rect: egui::Rect) {
        let painter = ui.painter();
        if self.show_axes {
            let margin = 10.0;
            let origin = rect.left_bottom() + egui::vec2(margin, -margin);
            let axis_len = 36.0;
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
        if !self.sample_grid_labels_enabled {
            return;
        }
        let Some(ds) = self.dataset.as_ref() else {
            return;
        };
        let (active_from, active_to, _color_from, _color_to, seg_t, _seg_idx) =
            self.current_segment(ds);
        let grid_idx = self.sample_grid_space_idx;
        let show_labels = if let Some(grid_idx) = grid_idx {
            if active_from == grid_idx && active_to == grid_idx {
                true
            } else if active_from == grid_idx && seg_t <= 0.02 {
                true
            } else {
                active_to == grid_idx && seg_t >= 0.98
            }
        } else {
            false
        };
        if !show_labels {
            return;
        }
        let Some(labels) = self.sample_grid_label_positions(ds) else {
            return;
        };
        let ppp = ui.ctx().pixels_per_point();
        let viewport_px = [rect.width() * ppp, rect.height() * ppp];
        if viewport_px[0] <= 0.0 || viewport_px[1] <= 0.0 {
            return;
        }
        let color = contrast_color(self.background_color);
        for (label, pos) in labels {
            let screen_px = [
                (pos[0] - self.camera.center[0]) * self.camera.pixels_per_unit
                    + 0.5 * viewport_px[0],
                (pos[1] - self.camera.center[1]) * self.camera.pixels_per_unit
                    + 0.5 * viewport_px[1],
            ];
            let screen = rect.min
                + egui::vec2(screen_px[0] / ppp, screen_px[1] / ppp);
            painter.text(
                screen,
                egui::Align2::CENTER_BOTTOM,
                label,
                egui::FontId::proportional(12.0),
                color,
            );
        }

    }

    fn viewport_fullscreen_overlay(&mut self, ctx: &egui::Context) {
        egui::Area::new("viewport_fullscreen_exit".into())
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-12.0, 12.0))
            .show(ctx, |ui| {
                let frame = egui::Frame::NONE
                    .fill(egui::Color32::from_black_alpha(140))
                    .corner_radius(6)
                    .inner_margin(egui::Margin::symmetric(8, 6));
                frame.show(ui, |ui| {
                    if ui.button("Exit fullscreen").clicked() {
                        self.viewport_fullscreen = false;
                        ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
                    }
                });
            });
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

        ui.horizontal(|ui| {
            ui.group(|ui| {
                ui.label("Transport");
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
                        self.play_direction = 1.0;
                        self.t = 0.0;
                    }
                    if ui.button("Step -").on_hover_text("Step back a bit.").clicked() {
                        self.t = (self.t - 0.01).clamp(0.0, 1.0);
                    }
                    if ui.button("Step +").on_hover_text("Step forward a bit.").clicked() {
                        self.t = (self.t + 0.01).clamp(0.0, 1.0);
                    }
                });
            });

            ui.add_space(8.0);

            ui.group(|ui| {
                ui.label("Playback");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.playback_mode, PlaybackMode::Once, "Once");
                    ui.selectable_value(&mut self.playback_mode, PlaybackMode::Loop, "Loop");
                    ui.selectable_value(&mut self.playback_mode, PlaybackMode::PingPong, "Ping-pong");
                });
                let min_speed = if self.space_path.len() > 5 { 0.01 } else { 0.05 };
                if self.speed < min_speed {
                    self.speed = min_speed;
                }
                ui.add(egui::Slider::new(&mut self.speed, min_speed..=2.0).text("speed"));
            });

            ui.add_space(8.0);

            ui.group(|ui| {
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

            ui.add_space(8.0);

            ui.group(|ui| {
                ui.label("Reset scaling");
                let key_count = self.space_path.len().max(1);
                if self.reset_view_key_idx >= key_count {
                    self.reset_view_key_idx = key_count - 1;
                }
                ui.horizontal(|ui| {
                    let key_label = self
                        .space_path
                        .get(self.reset_view_key_idx)
                        .and_then(|idx| ds_ref.meta.spaces.get(*idx))
                        .map(|s| format!("Key {}: {}", self.reset_view_key_idx + 1, s.name))
                        .unwrap_or_else(|| format!("Key {}", self.reset_view_key_idx + 1));
                    egui::ComboBox::from_id_salt("reset_view_key")
                        .width(160.0)
                        .selected_text(key_label)
                        .show_ui(ui, |ui| {
                            for (i, space_idx) in self.space_path.iter().enumerate() {
                                let label = ds_ref
                                    .meta
                                    .spaces
                                    .get(*space_idx)
                                    .map(|s| format!("Key {}: {}", i + 1, s.name))
                                    .unwrap_or_else(|| format!("Key {}", i + 1));
                                ui.selectable_value(&mut self.reset_view_key_idx, i, label);
                            }
                        });
                    if ui.button("Reset").clicked() {
                        if let Some(space_idx) = self.space_path.get(self.reset_view_key_idx) {
                            if let Some(space) = ds_ref.meta.spaces.get(*space_idx) {
                                let bbox = self.space_bbox_for_view(ds_ref, *space_idx, space);
                                let vp = self.last_viewport_points.size();
                                let ppp = ctx.pixels_per_point();
                                self.camera.fit_bbox(bbox, [vp.x * ppp, vp.y * ppp], 0.9);
                            }
                        }
                    }
                });
            });
        });
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
                if self.space_path.len() >= 2 {
                    let space = *self.space_path.last().unwrap_or(&0);
                    let color = self
                        .color_path
                        .last()
                        .cloned()
                        .unwrap_or(ColorKey::Current);
                    self.space_path.push(space);
                    self.color_path.push(color);
                    self.key_times.resize(self.space_path.len(), 0.0);
                    if self.key_collapsed.len() != self.space_path.len() {
                        self.key_collapsed.resize(self.space_path.len(), false);
                    }
                    self.selected_key_idx = Some(self.space_path.len() - 1);
                    self.distribute_key_times_evenly();
                }
            }
            if ui
                .button("Space evenly")
                .on_hover_text("Distribute keyframes evenly across the timeline.")
                .clicked()
            {
                self.distribute_key_times_evenly();
            }
            let can_remove = self.space_path.len() > 2 && self.selected_key_idx.is_some();
            let delete_enabled = can_remove && self.confirm_delete_cards;
            let delete_label = if self.confirm_delete_cards {
                "Delete: enabled"
            } else {
                "Delete: locked"
            };
            ui.checkbox(&mut self.confirm_delete_cards, delete_label);
            let remove_btn = ui
                .add_enabled(delete_enabled, egui::Button::new("Remove key"))
                .on_hover_text(if self.confirm_delete_cards {
                    "Remove the selected keyframe."
                } else {
                    "Enable Delete: yes to remove keyframes."
                });
            if remove_btn.clicked() {
                if let Some(idx) = self.selected_key_idx {
                    self.remove_keyframe(idx);
                }
            }
            if ui
                .button(if self.advanced_timeline_open {
                    "Close advanced timeline"
                } else {
                    "Advanced timeline"
                })
                .clicked()
            {
                self.advanced_timeline_open = !self.advanced_timeline_open;
                if !self.advanced_timeline_open {
                    self.advanced_preview_card = None;
                }
                if self.advanced_timeline_open && self.advanced_cards.is_empty() {
                    self.sync_advanced_from_timeline();
                }
            }

        });

        ui.horizontal(|ui| {
            if ui.button("Collapse all").clicked() {
                self.key_collapsed = vec![true; self.space_path.len()];
            }
            if ui.button("Expand all").clicked() {
                self.key_collapsed = vec![false; self.space_path.len()];
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
        let key_count = self.space_path.len();
        let mut drag_stop_idx: Option<usize> = None;
        let mut card_rects: Vec<(usize, egui::Rect)> = Vec::new();
        let mut remove_idx: Option<usize> = None;
        let pointer_pos = ctx.input(|i| i.pointer.latest_pos());
        let mut drag_target: Option<(usize, bool)> = None;
        egui::ScrollArea::horizontal()
            .max_height(260.0)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let max_idx = ds_ref.meta.spaces.len().saturating_sub(1);
                if self.key_collapsed.len() != key_count {
                    self.key_collapsed.resize(key_count, false);
                }
                ui.horizontal(|ui| {
                    for i in 0..key_count {
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
                        let mut collapsed = self.key_collapsed[i];
                        let mut frame = egui::Frame::group(ui.style())
                            .corner_radius(6)
                            .inner_margin(egui::Margin::same(2));
                        if is_selected {
                            frame = frame.stroke(egui::Stroke::new(
                                1.4,
                                egui::Color32::from_rgb(80, 180, 255),
                            ));
                        }
                        let inner = ui.scope_builder(
                            egui::UiBuilder::new().layout(egui::Layout::top_down(egui::Align::Min)),
                            |ui| {
                                frame.show(ui, |ui| {
                                    let prev_spacing = ui.spacing().item_spacing;
                                    ui.spacing_mut().item_spacing = egui::vec2(prev_spacing.x, 2.0);
                                    ui.spacing_mut().interact_size.y =
                                        ui.spacing().interact_size.y.min(20.0);
                                    let mut drag_handle_resp = None;
                                    ui.allocate_ui_with_layout(
                                        egui::vec2(0.0, 0.0),
                                        egui::Layout::left_to_right(egui::Align::Center),
                                        |ui| {
                                        let drag_handle = ui
                                            .add(
                                                egui::Button::new(
                                                    egui::RichText::new("⋮⋮")
                                                        .size(13.0)
                                                        .strong()
                                                        .color(ui.visuals().weak_text_color()),
                                                )
                                                .frame(false)
                                                .small()
                                                .sense(egui::Sense::drag()),
                                            )
                                            .on_hover_text("Drag to reorder keyframes.")
                                            .on_hover_and_drag_cursor(egui::CursorIcon::Grab);
                                        drag_handle_resp = Some(drag_handle);
                                        let header_resp = ui
                                            .selectable_label(
                                                is_selected,
                                                egui::RichText::new(format!("Key {}", i + 1))
                                                    .strong(),
                                            )
                                            .on_hover_text("Click to select this keyframe.");
                                        if header_resp.clicked() {
                                            self.selected_key_idx = Some(i);
                                        }
                                        let can_remove = self.space_path.len() > 2;
                                        let delete_enabled = can_remove && self.confirm_delete_cards;
                                        let delete_btn = ui
                                            .add_enabled(
                                                delete_enabled,
                                                egui::Button::new("🗑").small(),
                                            )
                                            .on_hover_text(if self.confirm_delete_cards {
                                                "Remove this keyframe."
                                            } else {
                                                "Enable Delete: yes to remove keyframes."
                                            });
                                        if delete_btn.clicked() {
                                            remove_idx = Some(i);
                                        }
                                        if ui
                                            .small_button(if collapsed { "▸" } else { "▾" })
                                            .on_hover_text("Collapse/expand this keyframe.")
                                            .clicked()
                                        {
                                            collapsed = !collapsed;
                                        }
                                    },
                                    );

                                    if let Some(drag_handle) = drag_handle_resp {
                                        if drag_handle.drag_started() || drag_handle.dragged() {
                                            self.dragging_key_idx = Some(i);
                                            ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                                        }
                                        if drag_handle.drag_stopped() {
                                            drag_stop_idx = Some(i);
                                        }
                                    }

                                    if collapsed {
                                        let space_name = ds_ref
                                            .meta
                                            .spaces
                                            .get(space_idx)
                                            .map(|s| s.name.as_str())
                                            .unwrap_or("?");
                                        let color_desc = match &color_key {
                                            ColorKey::Current => "Current".to_string(),
                                            ColorKey::Categorical(idx) => {
                                                format!("Cat: {}", obs_name(ds_ref, *idx))
                                            }
                                            ColorKey::Continuous(idx) => {
                                                format!("Cont: {}", obs_name(ds_ref, *idx))
                                            }
                                            ColorKey::Gene(name) => format!("Gene: {name}"),
                                        };
                                        ui.label(
                                            egui::RichText::new(format!("{space_name} | {color_desc}"))
                                                .size(10.5),
                                        );
                                        ui.spacing_mut().item_spacing = prev_spacing;
                                        return;
                                    }

                                    let control_width = 110.0;
                                    ui.allocate_ui_with_layout(
                                        egui::vec2(0.0, 0.0),
                                        egui::Layout::left_to_right(egui::Align::Center),
                                        |ui| {
                                        ui.label(egui::RichText::new("Space").size(10.5));
                                        egui::ComboBox::from_id_salt(("kf_space", i))
                                            .width(control_width)
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
                                    },
                                    );
                                    ui.allocate_ui_with_layout(
                                        egui::vec2(0.0, 0.0),
                                        egui::Layout::left_to_right(egui::Align::Center),
                                        |ui| {
                                        ui.label(egui::RichText::new("Color").size(10.5));
                                        let kind_label = match color_kind {
                                            KeyColorKind::Current => "Current",
                                            KeyColorKind::Categorical => "Categorical",
                                            KeyColorKind::Continuous => "Continuous",
                                            KeyColorKind::Gene => "Gene",
                                        };
                                        egui::ComboBox::from_id_salt(("kf_color_kind", i))
                                            .width(control_width)
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
                                    },
                                    );

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
                                                    .width(control_width)
                                                    .selected_text(
                                                        categorical_opts
                                                            .iter()
                                                            .find(|(idx, _)| *idx == obs_idx)
                                                            .map(|(_, name)| name.as_str())
                                                            .unwrap_or("?"),
                                                    )
                                                    .show_ui(ui, |ui| {
                                                        for (idx, name) in &categorical_opts {
                                                            ui.selectable_value(
                                                                &mut obs_idx,
                                                                *idx,
                                                                name,
                                                            );
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
                                                    .width(control_width)
                                                    .selected_text(
                                                        continuous_opts
                                                            .iter()
                                                            .find(|(idx, _)| *idx == obs_idx)
                                                            .map(|(_, name)| name.as_str())
                                                            .unwrap_or("?"),
                                                    )
                                                    .show_ui(ui, |ui| {
                                                        for (idx, name) in &continuous_opts {
                                                            ui.selectable_value(
                                                                &mut obs_idx,
                                                                *idx,
                                                                name,
                                                            );
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
                                                    .id(egui::Id::new(("kf_gene", i)))
                                                    .hint_text("Exact gene name")
                                                    .desired_width(control_width);
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
                                    ui.spacing_mut().item_spacing = prev_spacing;
                                })
                            },
                        );
                        let rect = inner.inner.response.rect;
                        card_rects.push((i, rect));
                        self.space_path[i] = space_idx;
                        if i < self.color_path.len() {
                            self.color_path[i] = color_key;
                        } else {
                            self.color_path.push(color_key);
                        }
                        self.key_collapsed[i] = collapsed;
                        ui.add_space(6.0);
                    }
                });
            });
        ui.add_space(8.0);

        if let (Some(drag_idx), Some(pos)) = (self.dragging_key_idx, pointer_pos) {
            if let Some((target_idx, target_rect, insert_after)) = card_rects
                .iter()
                .filter_map(|(idx, rect)| {
                    let center = rect.center();
                    let dist = (center.x - pos.x).powi(2) + (center.y - pos.y).powi(2);
                    Some((*idx, *rect, dist))
                })
                .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, rect, _)| (idx, rect, pos.x > rect.center().x))
            {
                drag_target = Some((target_idx, insert_after));
                let line_x = if insert_after {
                    target_rect.right()
                } else {
                    target_rect.left()
                };
                let line_color = egui::Color32::from_rgb(120, 200, 255);
                ui.painter().line_segment(
                    [egui::pos2(line_x, target_rect.top()), egui::pos2(line_x, target_rect.bottom())],
                    egui::Stroke::new(2.0, line_color),
                );
            }
            if let Some((_, rect)) = card_rects.iter().find(|(idx, _)| *idx == drag_idx) {
                let size = rect.size();
                let offset = egui::vec2(12.0, 12.0);
                let float_rect = egui::Rect::from_min_size(pos + offset, size);
                let shadow_rect = float_rect.translate(egui::vec2(3.0, 3.0));
                let shadow_color = egui::Color32::from_rgba_unmultiplied(0, 0, 0, 90);
                let card_fill = ui.visuals().widgets.inactive.bg_fill;
                let card_stroke = ui.visuals().widgets.active.bg_stroke;
                let text_color = ui.visuals().text_color();
                ui.painter()
                    .rect_filled(shadow_rect, 6.0, shadow_color);
                ui.painter().rect_filled(float_rect, 6.0, card_fill);
                ui.painter()
                    .rect_stroke(float_rect, 6.0, card_stroke, egui::StrokeKind::Inside);
                ui.painter().text(
                    float_rect.min + egui::vec2(8.0, 8.0),
                    egui::Align2::LEFT_TOP,
                    format!("Key {}", drag_idx + 1),
                    egui::FontId::proportional(12.0),
                    text_color,
                );
            }
        }
        if drag_stop_idx.is_none()
            && self.dragging_key_idx.is_some()
            && !ctx.input(|i| i.pointer.any_down())
        {
            drag_stop_idx = self.dragging_key_idx;
        }

        if let (Some(drag_idx), Some(stop_idx)) = (self.dragging_key_idx, drag_stop_idx) {
            if drag_idx == stop_idx {
                if let Some((target_idx, insert_after)) = drag_target {
                    let mut insert_idx = if insert_after {
                        target_idx.saturating_add(1)
                    } else {
                        target_idx
                    };
                    if insert_idx > drag_idx {
                        insert_idx = insert_idx.saturating_sub(1);
                    }
                    if insert_idx != drag_idx {
                        self.move_keyframe(drag_idx, insert_idx);
                    }
                }
            }
            self.dragging_key_idx = None;
        }
        if let Some(idx) = remove_idx {
            self.remove_keyframe(idx);
        }

        self.ensure_color_path_len(self.space_path.len());
        self.from_space = *self.space_path.first().unwrap_or(&self.from_space);
        self.to_space = *self.space_path.last().unwrap_or(&self.to_space);

        if self.playing {
            ctx.request_repaint_after(self.main_view_frame_interval());
        }
    }

    fn ui_advanced_timeline(&mut self, ctx: &egui::Context) {
        if !self.advanced_timeline_open {
            return;
        }
        let viewport_id = egui::ViewportId::from_hash_of("advanced_timeline");
        let viewport_exists = ctx.input(|i| i.raw.viewports.contains_key(&viewport_id));
        let mut builder = egui::ViewportBuilder::default()
            .with_title("Advanced timeline")
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([800.0, 600.0])
            .with_resizable(true)
            .with_clamp_size_to_monitor_size(true);
        if !viewport_exists {
            if let Some(size) = self.advanced_viewport_size {
                if size.x > 0.0 && size.y > 0.0 {
                    builder = builder.with_inner_size(egui::vec2(size.x, size.y));
                }
            }
            if let Some(pos) = self.advanced_viewport_pos {
                builder = builder.with_position(pos);
            }
        }
        let mut close_requested = false;
        ctx.show_viewport_immediate(viewport_id, builder, |ctx, class| {
            let now = ctx.input(|i| i.time);
            let min_dt = 1.0f64 / ADVANCED_VIEW_FPS_CAP_HZ as f64;
            let elapsed = now - self.advanced_last_render_time;
            if elapsed < min_dt {
                ctx.request_repaint_after(std::time::Duration::from_secs_f64(min_dt - elapsed));
            } else {
                self.advanced_last_render_time = now;
            }
            let wants_close = ctx.input(|i| {
                i.raw
                    .viewports
                    .get(&viewport_id)
                    .map(|v| v.close_requested())
                    .unwrap_or(false)
            });
            if wants_close {
                close_requested = true;
            }
            if !matches!(class, egui::ViewportClass::Embedded) {
                if let Some(info) = ctx.input(|i| i.raw.viewports.get(&viewport_id).cloned()) {
                    let is_fullscreen = info.fullscreen.unwrap_or(false);
                    let is_minimized = info.minimized.unwrap_or(false);
                    if !is_fullscreen && !is_minimized {
                        if let Some(rect) = info.inner_rect.or(info.outer_rect) {
                            let size = rect.size();
                            if size.x > 100.0 && size.y > 100.0 {
                                self.advanced_viewport_size = Some(size);
                            }
                        }
                        if let Some(rect) = info.outer_rect.or(info.inner_rect) {
                            let pos = rect.min;
                            self.advanced_viewport_pos = Some(pos);
                        }
                    }
                }
            }
            match class {
                egui::ViewportClass::Embedded => {
                    let mut open = true;
                    egui::Window::new("Advanced timeline")
                        .open(&mut open)
                        .resizable(true)
                        .default_size([1200.0, 800.0])
                        .show(ctx, |ui| {
                            self.advanced_timeline_contents(ctx, ui, viewport_id);
                        });
                    if !open {
                        close_requested = true;
                    }
                }
                _ => {
                    egui::CentralPanel::default().show(ctx, |ui| {
                        self.advanced_timeline_contents(ctx, ui, viewport_id);
                    });
                }
            }
        });
        if close_requested {
            self.advanced_timeline_open = false;
            self.advanced_preview_card = None;
        }
    }

    fn advanced_timeline_contents(
        &mut self,
        ctx: &egui::Context,
        ui: &mut egui::Ui,
        viewport_id: egui::ViewportId,
    ) {
        ctx.set_visuals(Self::visuals_for_theme(self.ui_theme));
        let (is_fullscreen, is_maximized) = ctx.input(|i| {
            let info = i.raw.viewports.get(&viewport_id);
            (
                info.and_then(|v| v.fullscreen).unwrap_or(false),
                info.and_then(|v| v.maximized).unwrap_or(false),
            )
        });
        let modifiers = ctx.input(|i| i.modifiers);
        let mut preview_card: Option<usize> = None;
        let mut remove_idx: Option<usize> = None;
        let mut add_card_requested = false;
        ui.horizontal(|ui| {
            if ui.button("Sync from timeline").clicked() {
                self.sync_advanced_from_timeline();
            }
            if ui.button("Apply to timeline").clicked() {
                self.apply_advanced_to_timeline();
            }
            if ui.button("Add card").clicked() {
                add_card_requested = true;
            }
            ui.checkbox(&mut self.advanced_grid_mode, "Grid");
            ui.add_enabled(
                self.advanced_grid_mode,
                egui::DragValue::new(&mut self.advanced_grid_size)
                    .speed(2.0)
                    .range(24.0..=120.0)
                    .prefix("Grid "),
            );
            let fullscreen_label = if is_fullscreen {
                "Exit fullscreen"
            } else {
                "Fullscreen"
            };
            if ui.button(fullscreen_label).clicked() {
                ctx.send_viewport_cmd_to(
                    viewport_id,
                    egui::ViewportCommand::Fullscreen(!is_fullscreen),
                );
                if !is_fullscreen {
                    ctx.send_viewport_cmd_to(
                        viewport_id,
                        egui::ViewportCommand::Maximized(true),
                    );
                }
            }
            let maximize_label = if is_maximized {
                "Restore size"
            } else {
                "Maximize"
            };
            if ui.button(maximize_label).clicked() {
                ctx.send_viewport_cmd_to(
                    viewport_id,
                    egui::ViewportCommand::Maximized(!is_maximized),
                );
            }
            if ui.button("Close").clicked() {
                self.advanced_timeline_open = false;
                self.advanced_preview_card = None;
                ctx.send_viewport_cmd_to(viewport_id, egui::ViewportCommand::Close);
            }
        });
        ui.separator();

        let ds_opt = self.dataset.clone();
        let categorical_opts: Vec<(usize, String)> = ds_opt
            .as_ref()
            .map(|ds| {
                ds.meta
                    .obs
                    .iter()
                    .enumerate()
                    .filter_map(|(i, o)| match o {
                        ObsMeta::Categorical { name, .. } => Some((i, name.clone())),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default();
        let continuous_opts: Vec<(usize, String)> = ds_opt
            .as_ref()
            .map(|ds| {
                ds.meta
                    .obs
                    .iter()
                    .enumerate()
                    .filter_map(|(i, o)| match o {
                        ObsMeta::Continuous { name, .. } => Some((i, name.clone())),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default();

        let default_card_size = egui::vec2(220.0, 150.0);
        for card in &mut self.advanced_cards {
            if card.size.x < default_card_size.x || card.size.y < default_card_size.y {
                card.size = default_card_size;
            }
        }

        let canvas_rect = ui.available_rect_before_wrap();
        let canvas_response = ui.allocate_rect(canvas_rect, egui::Sense::click());
        {
            let painter = ui.painter();
            painter.rect_filled(canvas_rect, 0.0, ui.visuals().extreme_bg_color);
            if self.advanced_grid_mode {
                let step = self.advanced_grid_size.max(24.0);
                let grid_color = egui::Color32::from_gray(40);
                let cols = (canvas_rect.width() / step).ceil() as i32;
                let rows = (canvas_rect.height() / step).ceil() as i32;
                let max_lines = 200;
                for i in 0..cols.min(max_lines) {
                    let x = canvas_rect.left() + i as f32 * step;
                    painter.line_segment(
                        [
                            egui::pos2(x, canvas_rect.top()),
                            egui::pos2(x, canvas_rect.bottom()),
                        ],
                        egui::Stroke::new(1.0, grid_color),
                    );
                }
                for j in 0..rows.min(max_lines) {
                    let y = canvas_rect.top() + j as f32 * step;
                    painter.line_segment(
                        [
                            egui::pos2(canvas_rect.left(), y),
                            egui::pos2(canvas_rect.right(), y),
                        ],
                        egui::Stroke::new(1.0, grid_color),
                    );
                }
            }
        }

        let inspector_size = egui::vec2(320.0, 360.0);
        let inspector_rect =
            egui::Rect::from_min_size(canvas_rect.min + egui::vec2(12.0, 12.0), inspector_size);
        let pointer_pos = ctx.input(|i| i.pointer.latest_pos());
        if add_card_requested {
            let pos = egui::pos2(inspector_rect.right() + 24.0, inspector_rect.top());
            self.add_advanced_card_at(canvas_rect, inspector_rect, pos);
        }
        for i in 0..self.advanced_cards.len() {
            self.resolve_card_collisions(i, canvas_rect, inspector_rect);
        }
        let mut card_positions: Vec<egui::Pos2> = Vec::with_capacity(self.advanced_cards.len());
        for card in &self.advanced_cards {
            card_positions.push(constrain_card_pos(
                card.pos,
                card.size,
                canvas_rect,
                inspector_rect,
            ));
        }
        let mut connect_state = self.advanced_connecting;
        if let Some(connect) = &mut connect_state {
            if let Some((pos, size)) = self
                .advanced_cards
                .get(connect.from_idx)
                .map(|card| (card_positions[connect.from_idx], card.size))
            {
                let rect =
                    egui::Rect::from_min_size(canvas_rect.min + pos.to_vec2(), size);
                connect.start_pos = if connect.from_is_output {
                    egui::pos2(rect.right(), rect.center().y)
                } else {
                    egui::pos2(rect.left(), rect.center().y)
                };
            }
        }
        let connection_stroke =
            egui::Stroke::new(1.4, ui.visuals().selection.stroke.color);
        {
            let painter = ui.painter();
            for conn in &self.advanced_connections {
                if conn.from >= card_positions.len() || conn.to >= card_positions.len() {
                    continue;
                }
                let from_card = &self.advanced_cards[conn.from];
                let to_card = &self.advanced_cards[conn.to];
                let from_rect = egui::Rect::from_min_size(
                    canvas_rect.min + card_positions[conn.from].to_vec2(),
                    from_card.size,
                );
                let to_rect = egui::Rect::from_min_size(
                    canvas_rect.min + card_positions[conn.to].to_vec2(),
                    to_card.size,
                );
                let from_pos = egui::pos2(from_rect.right(), from_rect.center().y);
                let to_pos = egui::pos2(to_rect.left(), to_rect.center().y);
                painter.line_segment([from_pos, to_pos], connection_stroke);
            }
        }
        let mut clicked_on_card = false;
        let mut card_drag_started = false;
        let mut selected_card = self.advanced_selected_card;
        let mut selected_cards = self.advanced_selected_cards.clone();
        let mut drag_idx = self.advanced_drag_idx;
        let mut drag_offset = self.advanced_drag_offset;
        let mut group_drag = self.advanced_drag_group.clone();
        let mut group_drag_pointer = self.advanced_drag_pointer_start;
        let mut start_group_drag: Option<egui::Pos2> = None;
        let mut marquee_start = self.advanced_marquee_start;
        let mut context_add_pos: Option<egui::Pos2> = None;
        let mut node_hits: Vec<(usize, bool, egui::Pos2)> = Vec::new();
        let mut resolve_idx: Option<usize> = None;
        let mut remove_indices: Vec<usize> = Vec::new();
        for (i, card) in self.advanced_cards.iter_mut().enumerate() {
            card.pos = constrain_card_pos(card.pos, card.size, canvas_rect, inspector_rect);
            let rect = egui::Rect::from_min_size(
                canvas_rect.min + card.pos.to_vec2(),
                card.size,
            );
            let selected = selected_card == Some(i);
            let fill = if selected {
                ui.visuals().widgets.active.bg_fill
            } else {
                ui.visuals().widgets.inactive.bg_fill
            };
            let stroke = if selected {
                egui::Stroke::new(1.5, ui.visuals().selection.stroke.color)
            } else {
                ui.visuals().widgets.noninteractive.bg_stroke
            };
            let node_r = 5.0;
            let in_pos = egui::pos2(rect.left(), rect.center().y);
            let out_pos = egui::pos2(rect.right(), rect.center().y);
            node_hits.push((i, false, in_pos));
            node_hits.push((i, true, out_pos));
            let header_height = 34.0;
            {
                let painter = ui.painter();
                painter.rect_filled(rect, 6.0, fill);
                painter.rect_stroke(rect, 6.0, stroke, egui::StrokeKind::Inside);
                let label = ds_opt
                    .as_ref()
                    .and_then(|ds| ds.meta.spaces.get(card.space_idx))
                    .map(|s| s.name.as_str())
                    .unwrap_or("Card");
                painter.text(
                    rect.min + egui::vec2(8.0, 6.0),
                    egui::Align2::LEFT_TOP,
                    label,
                    egui::FontId::proportional(12.0),
                    ui.visuals().text_color(),
                );
                painter.text(
                    rect.min + egui::vec2(8.0, 20.0),
                    egui::Align2::LEFT_TOP,
                    format!("{:.2}s", card.duration_sec),
                    egui::FontId::proportional(11.0),
                    ui.visuals().text_color(),
                );
                let node_color = ui.visuals().selection.bg_fill;
                let disabled_color = ui.visuals().widgets.noninteractive.fg_stroke.color;
                let in_color = if card.in_enabled { node_color } else { disabled_color };
                let out_color = if card.out_enabled { node_color } else { disabled_color };
                painter.circle_filled(in_pos, node_r, in_color);
                painter.circle_filled(out_pos, node_r, out_color);
                painter.circle_stroke(in_pos, node_r, egui::Stroke::new(1.0, stroke.color));
                painter.circle_stroke(out_pos, node_r, egui::Stroke::new(1.0, stroke.color));
            }

            let node_size = egui::vec2(node_r * 2.6, node_r * 2.6);
            let in_rect = egui::Rect::from_center_size(in_pos, node_size);
            let out_rect = egui::Rect::from_center_size(out_pos, node_size);
            let in_id = ui.id().with(("adv_node_in", card.id));
            let out_id = ui.id().with(("adv_node_out", card.id));
            let in_resp = ui.interact(in_rect, in_id, egui::Sense::click_and_drag());
            let out_resp = ui.interact(out_rect, out_id, egui::Sense::click_and_drag());
            if in_resp.drag_started() {
                selected_card = Some(i);
                clicked_on_card = true;
                connect_state = Some(AdvancedConnectionDrag {
                    from_idx: i,
                    from_is_output: false,
                    start_pos: in_pos,
                });
            }
            if out_resp.drag_started() {
                selected_card = Some(i);
                clicked_on_card = true;
                connect_state = Some(AdvancedConnectionDrag {
                    from_idx: i,
                    from_is_output: true,
                    start_pos: out_pos,
                });
            }

            let mut inner_rect = rect.shrink(8.0);
            inner_rect.min.y += header_height;
            if inner_rect.is_positive() {
                ui.scope_builder(egui::UiBuilder::new().max_rect(inner_rect), |ui| {
                    ui.set_min_size(inner_rect.size());
                    ui.spacing_mut().item_spacing = egui::vec2(4.0, 2.0);
                render_advanced_card_controls(
                    ui,
                    card,
                    ds_opt.as_deref(),
                    &categorical_opts,
                    &continuous_opts,
                    110.0,
                );
                });
            }

            let pointer_over_controls = pointer_pos
                .map(|pos| inner_rect.contains(pos))
                .unwrap_or(false);
            let pointer_in_rect = pointer_pos.map(|pos| rect.contains(pos)).unwrap_or(false);
            let node_active = in_resp.dragged()
                || in_resp.drag_started()
                || out_resp.dragged()
                || out_resp.drag_started();
            let id = ui.id().with(("adv_card", card.id));
            let sense = if pointer_over_controls {
                egui::Sense::hover()
            } else {
                egui::Sense::click_and_drag()
            };
            let response = ui.interact(rect, id, sense);
            if pointer_over_controls
                && drag_idx.is_none()
                && !node_active
                && ctx.input(|i| i.pointer.any_pressed())
                && pointer_in_rect
            {
                card_drag_started = true;
                if selected_cards.contains(&i) && selected_cards.len() > 1 {
                    start_group_drag = pointer_pos;
                    drag_idx = Some(i);
                } else if let Some(pos) = pointer_pos {
                    drag_idx = Some(i);
                    drag_offset = pos.to_vec2() - rect.min.to_vec2();
                }
            }
            let clicked = if pointer_over_controls {
                pointer_in_rect && ctx.input(|i| i.pointer.primary_clicked())
            } else {
                response.clicked()
            };
            if clicked {
                selected_card = Some(i);
                clicked_on_card = true;
                if modifiers.shift || modifiers.ctrl {
                    if selected_cards.contains(&i) {
                        selected_cards.remove(&i);
                        if selected_card == Some(i) {
                            selected_card = selected_cards.iter().copied().next();
                        }
                    } else {
                        selected_cards.insert(i);
                        selected_card = Some(i);
                    }
                } else {
                    selected_cards.clear();
                    selected_cards.insert(i);
                }
            }
            response.context_menu(|ui| {
                if ui.button("Add card").clicked() {
                    context_add_pos = pointer_pos;
                    ui.close();
                }
                if selected_cards.contains(&i) && selected_cards.len() > 1 {
                    if ui.button("Remove selected cards").clicked() {
                        remove_indices.extend(selected_cards.iter().copied());
                        ui.close();
                    }
                } else if ui.button("Remove card").clicked() {
                    remove_indices.push(i);
                    ui.close();
                }
            });
            if !node_active {
                if response.drag_started() {
                    card_drag_started = true;
                    if selected_cards.contains(&i) && selected_cards.len() > 1 {
                        start_group_drag =
                            response.interact_pointer_pos().or(pointer_pos);
                        drag_idx = Some(i);
                    } else {
                        drag_idx = Some(i);
                        if let Some(pos) = response.interact_pointer_pos() {
                            drag_offset = pos.to_vec2() - rect.min.to_vec2();
                        }
                    }
                }
                let should_drag = response.dragged()
                    || (pointer_over_controls
                        && drag_idx == Some(i)
                        && ctx.input(|i| i.pointer.is_decidedly_dragging()));
                if should_drag && group_drag.is_none() && start_group_drag.is_none() {
                    if let Some(pos) = response.interact_pointer_pos().or(pointer_pos) {
                        let mut local = pos.to_vec2() - canvas_rect.min.to_vec2() - drag_offset;
                        let max_x = (canvas_rect.width() - card.size.x).max(0.0);
                        let max_y = (canvas_rect.height() - card.size.y).max(0.0);
                        if self.advanced_grid_mode {
                            let step = self.advanced_grid_size.max(24.0);
                            local.x = (local.x / step).round() * step;
                            local.y = (local.y / step).round() * step;
                        }
                        local.x = local.x.clamp(0.0, max_x);
                        local.y = local.y.clamp(0.0, max_y);
                        card.pos = constrain_card_pos(
                            egui::pos2(local.x, local.y),
                            card.size,
                            canvas_rect,
                            inspector_rect,
                        );
                        resolve_idx = Some(i);
                    }
                }
            }
        }
        if let Some(start_pos) = start_group_drag {
            let mut group = Vec::with_capacity(selected_cards.len());
            for idx in selected_cards.iter().copied() {
                if let Some(card) = self.advanced_cards.get(idx) {
                    group.push((idx, card.pos));
                }
            }
            if !group.is_empty() {
                group_drag = Some(group);
                group_drag_pointer = Some(start_pos);
            }
        }
        if let (Some(group), Some(start_pos), Some(pos)) =
            (&group_drag, group_drag_pointer, pointer_pos)
        {
            if ctx.input(|i| i.pointer.any_down()) {
                let delta = pos - start_pos;
                for (idx, base_pos) in group {
                    if let Some(card) = self.advanced_cards.get_mut(*idx) {
                        let mut local = base_pos.to_vec2() + delta;
                        let max_x = (canvas_rect.width() - card.size.x).max(0.0);
                        let max_y = (canvas_rect.height() - card.size.y).max(0.0);
                        if self.advanced_grid_mode {
                            let step = self.advanced_grid_size.max(24.0);
                            local.x = (local.x / step).round() * step;
                            local.y = (local.y / step).round() * step;
                        }
                        local.x = local.x.clamp(0.0, max_x);
                        local.y = local.y.clamp(0.0, max_y);
                        card.pos = constrain_card_pos(
                            egui::pos2(local.x, local.y),
                            card.size,
                            canvas_rect,
                            inspector_rect,
                        );
                    }
                }
                for idx in selected_cards.iter().copied() {
                    self.resolve_card_collisions(idx, canvas_rect, inspector_rect);
                }
            }
        }
        if let (Some(connect), Some(pos)) = (connect_state, pointer_pos) {
            ui.painter()
                .line_segment([connect.start_pos, pos], connection_stroke);
        }

        if drag_idx.is_some() && !ctx.input(|i| i.pointer.any_down()) {
            drag_idx = None;
            group_drag = None;
            group_drag_pointer = None;
        }

        if canvas_response.clicked() && !clicked_on_card {
            selected_card = None;
            selected_cards.clear();
        }
        if canvas_response.drag_started() && !card_drag_started {
            marquee_start = canvas_response.interact_pointer_pos();
            selected_cards.clear();
            selected_card = None;
        }
        if let (Some(start), Some(pos)) = (marquee_start, pointer_pos) {
            if ctx.input(|i| i.pointer.any_down()) {
                selected_cards.clear();
                let selection_rect =
                    egui::Rect::from_two_pos(start, pos).intersect(canvas_rect);
                let painter = ui.painter();
                painter.rect_filled(
                    selection_rect,
                    0.0,
                    egui::Color32::from_rgba_unmultiplied(90, 160, 255, 40),
                );
                painter.rect_stroke(
                    selection_rect,
                    0.0,
                    egui::Stroke::new(1.2, egui::Color32::from_rgb(120, 200, 255)),
                    egui::StrokeKind::Inside,
                );
                for (i, card) in self.advanced_cards.iter().enumerate() {
                    let card_rect = egui::Rect::from_min_size(
                        canvas_rect.min + card.pos.to_vec2(),
                        card.size,
                    );
                    if selection_rect.intersects(card_rect) {
                        selected_cards.insert(i);
                    }
                }
                selected_card = selected_cards.iter().copied().next();
            } else {
                marquee_start = None;
            }
        }
        if let Some(connect) = connect_state {
            if !ctx.input(|i| i.pointer.any_down()) {
                if let Some(pos) = pointer_pos {
                    let mut best: Option<(usize, bool, f32)> = None;
                    for (idx, is_output, node_pos) in &node_hits {
                        let dist = node_pos.distance(pos);
                        if dist <= 12.0 {
                            if best.as_ref().map(|b| dist < b.2).unwrap_or(true) {
                                best = Some((*idx, *is_output, dist));
                            }
                        }
                    }
                    if let Some((target_idx, target_is_output, _)) = best {
                        if connect.from_is_output && !target_is_output {
                            let from = connect.from_idx;
                            let to = target_idx;
                            if from != to
                                && !self
                                    .advanced_connections
                                    .iter()
                                    .any(|c| c.from == from && c.to == to)
                            {
                                self.advanced_connections.push(AdvancedConnection { from, to });
                            }
                        } else if !connect.from_is_output && target_is_output {
                            let from = target_idx;
                            let to = connect.from_idx;
                            if from != to
                                && !self
                                    .advanced_connections
                                    .iter()
                                    .any(|c| c.from == from && c.to == to)
                            {
                                self.advanced_connections.push(AdvancedConnection { from, to });
                            }
                        }
                    }
                }
                self.advanced_connecting = None;
            } else {
                self.advanced_connecting = Some(connect);
            }
        } else {
            self.advanced_connecting = None;
        }

        canvas_response.context_menu(|ui| {
            if ui.button("Add card").clicked() {
                if let Some(pos) = pointer_pos {
                    self.add_advanced_card_at(canvas_rect, inspector_rect, pos);
                }
                ui.close();
            }
        });

        egui::Area::new(egui::Id::new("adv_card_inspector"))
            .fixed_pos(inspector_rect.min)
            .show(ctx, |ui| {
                let mut frame = egui::Frame::NONE
                    .fill(ui.visuals().extreme_bg_color)
                    .stroke(egui::Stroke::new(1.5, ui.visuals().selection.stroke.color))
                    .corner_radius(egui::CornerRadius::same(6))
                    .inner_margin(egui::Margin::same(8));
                frame.shadow = egui::epaint::Shadow {
                    color: egui::Color32::from_rgba_unmultiplied(0, 0, 0, 120),
                    offset: [0, 4],
                    blur: 10,
                    spread: 1,
                };
                frame.show(ui, |ui| {
                        ui.set_min_size(inspector_size);
                        ui.label("Card inspector");
                        if selected_cards.len() > 1 {
                            ui.label(format!("{} cards selected", selected_cards.len()));
                            if ui.button("Remove selected cards").clicked() {
                                remove_indices.extend(selected_cards.iter().copied());
                            }
                        } else if let Some(idx) = self.advanced_selected_card {
                            if let Some(card) = self.advanced_cards.get_mut(idx) {
                                if ui.button("Remove card").clicked() {
                                    remove_idx = Some(idx);
                                }
                                render_advanced_card_controls(
                                    ui,
                                    card,
                                    ds_opt.as_deref(),
                                    &categorical_opts,
                                    &continuous_opts,
                                    150.0,
                                );
                                ui.horizontal(|ui| {
                                    ui.label("Duration");
                                    ui.add(
                                        egui::DragValue::new(&mut card.duration_sec)
                                            .speed(0.05)
                                            .range(0.1..=30.0)
                                            .suffix("s"),
                                    );
                                });
                                ui.horizontal(|ui| {
                                    ui.checkbox(&mut card.in_enabled, "In");
                                    ui.checkbox(&mut card.out_enabled, "Out");
                                });

                                let mut filter_enabled = card.filter.is_some();
                                if ui.checkbox(&mut filter_enabled, "Filter").changed() {
                                    if filter_enabled {
                                        let obs_idx = categorical_opts
                                            .first()
                                            .map(|(idx, _)| *idx)
                                            .unwrap_or(0);
                                        card.filter = Some(AdvancedCardFilter {
                                            obs_idx,
                                            enabled: Vec::new(),
                                            cached_indices: None,
                                            cached_indices_id: 0,
                                            cached_dataset_id: 0,
                                        });
                                    } else {
                                        card.filter = None;
                                        if self.advanced_preview_card == Some(idx) {
                                            self.advanced_preview_card = None;
                                        }
                                    }
                                }
                                if let Some(filter) = card.filter.as_mut() {
                                    if categorical_opts.is_empty() {
                                        ui.label("No categorical obs.");
                                    } else if let Some(ds) = ds_opt.as_ref() {
                                        let mut obs_idx = filter.obs_idx;
                                        egui::ComboBox::from_id_salt(("adv_filter_obs", card.id))
                                            .selected_text(
                                                categorical_opts
                                                    .iter()
                                                    .find(|(idx, _)| *idx == obs_idx)
                                                    .map(|(_, name)| name.as_str())
                                                    .unwrap_or("Categorical"),
                                            )
                                            .show_ui(ui, |ui| {
                                                for (idx, name) in &categorical_opts {
                                                    ui.selectable_value(
                                                        &mut obs_idx,
                                                        *idx,
                                                        name,
                                                    );
                                                }
                                            });
                                        if obs_idx != filter.obs_idx {
                                            filter.obs_idx = obs_idx;
                                            filter.enabled.clear();
                                            filter.cached_indices = None;
                                            filter.cached_indices_id = 0;
                                            filter.cached_dataset_id = 0;
                                        }
                                        if let Ok((_name, _labels, categories, _pal)) =
                                            ds.obs_categorical(filter.obs_idx)
                                        {
                                            if filter.enabled.len() != categories.len() {
                                                filter.enabled = vec![true; categories.len()];
                                                filter.cached_indices = None;
                                                filter.cached_indices_id = 0;
                                                filter.cached_dataset_id = 0;
                                            }
                                            ui.horizontal(|ui| {
                                                if ui.button("All").clicked() {
                                                    for v in &mut filter.enabled {
                                                        *v = true;
                                                    }
                                                    filter.cached_indices = None;
                                                    filter.cached_indices_id = 0;
                                                    filter.cached_dataset_id = 0;
                                                }
                                                if ui.button("None").clicked() {
                                                    for v in &mut filter.enabled {
                                                        *v = false;
                                                    }
                                                    filter.cached_indices = None;
                                                    filter.cached_indices_id = 0;
                                                    filter.cached_dataset_id = 0;
                                                }
                                                if ui.button("Preview").clicked() {
                                                    preview_card = Some(idx);
                                                }
                                            });
                                            let row_height = 22.0;
                                            let list_max =
                                                row_height * categories.len().min(10) as f32 + 6.0;
                                            egui::ScrollArea::vertical()
                                                .max_height(list_max)
                                                .auto_shrink([false, false])
                                                .show(ui, |ui| {
                                                    for (idx, name) in
                                                        categories.iter().enumerate()
                                                    {
                                                        ui.horizontal(|ui| {
                                                            if ui
                                                                .checkbox(
                                                                    &mut filter.enabled[idx],
                                                                    "",
                                                                )
                                                                .changed()
                                                            {
                                                                filter.cached_indices = None;
                                                                filter.cached_indices_id = 0;
                                                                filter.cached_dataset_id = 0;
                                                            }
                                                            ui.label(name);
                                                        });
                                                    }
                                                });
                                        }
                                    }
                                }
                            }
                        } else {
                            ui.label("Select a card to edit its settings.");
                        }
                    });
            });

        if let Some(pos) = context_add_pos {
            self.add_advanced_card_at(canvas_rect, inspector_rect, pos);
        }
        if let Some(idx) = resolve_idx {
            self.resolve_card_collisions(idx, canvas_rect, inspector_rect);
        }
        if let Some(idx) = remove_idx {
            remove_indices.push(idx);
        }
        if !remove_indices.is_empty() {
            remove_indices.sort_unstable();
            remove_indices.dedup();
            for idx in remove_indices.into_iter().rev() {
                if idx < self.advanced_cards.len() {
                    self.advanced_cards.remove(idx);
                    if selected_card == Some(idx) {
                        selected_card = None;
                    } else if let Some(sel) = selected_card {
                        if sel > idx {
                            selected_card = Some(sel - 1);
                        }
                    }
                    if let Some(preview_idx) = self.advanced_preview_card {
                        if preview_idx == idx {
                            self.advanced_preview_card = None;
                        } else if preview_idx > idx {
                            self.advanced_preview_card = Some(preview_idx - 1);
                        }
                    }
                    selected_cards = selected_cards
                        .iter()
                        .filter_map(|sel| {
                            if *sel == idx {
                                None
                            } else if *sel > idx {
                                Some(sel - 1)
                            } else {
                                Some(*sel)
                            }
                        })
                        .collect();
                }
            }
        }
        if let Some(idx) = preview_card {
            self.advanced_preview_card = Some(idx);
        }

        self.advanced_selected_card = selected_card;
        self.advanced_selected_cards = selected_cards;
        self.advanced_drag_idx = drag_idx;
        self.advanced_drag_offset = drag_offset;
        self.advanced_drag_group = group_drag;
        self.advanced_drag_pointer_start = group_drag_pointer;
        self.advanced_marquee_start = marquee_start;

    }

    fn add_advanced_card_at(
        &mut self,
        canvas_rect: egui::Rect,
        inspector_rect: egui::Rect,
        pos: egui::Pos2,
    ) {
        let card_size = egui::vec2(220.0, 150.0);
        let mut local = pos.to_vec2() - canvas_rect.min.to_vec2();
        let max_x = (canvas_rect.width() - card_size.x).max(0.0);
        let max_y = (canvas_rect.height() - card_size.y).max(0.0);
        local.x = local.x.clamp(0.0, max_x);
        local.y = local.y.clamp(0.0, max_y);
        let pos = constrain_card_pos(
            egui::pos2(local.x, local.y),
            card_size,
            canvas_rect,
            inspector_rect,
        );
        let id = self.next_advanced_id();
        let space_idx = self.space_path.get(self.selected_key_idx.unwrap_or(0)).copied().unwrap_or(0);
        let color_key = self
            .color_path
            .get(self.selected_key_idx.unwrap_or(0))
            .cloned()
            .unwrap_or(ColorKey::Current);
        self.advanced_cards.push(AdvancedCard {
            id,
            space_idx,
            color_key,
            duration_sec: 2.0,
            pos,
            size: card_size,
            in_enabled: true,
            out_enabled: true,
            filter: None,
        });
        let idx = self.advanced_cards.len().saturating_sub(1);
        self.resolve_card_collisions(idx, canvas_rect, inspector_rect);
    }

    fn resolve_card_collisions(
        &mut self,
        idx: usize,
        canvas_rect: egui::Rect,
        inspector_rect: egui::Rect,
    ) {
        if idx >= self.advanced_cards.len() {
            return;
        }
        let mut iterations = 0;
        while iterations < 8 {
            iterations += 1;
            let (left, rest) = self.advanced_cards.split_at_mut(idx);
            let Some((card, right)) = rest.split_first_mut() else {
                return;
            };
            let mut rect = egui::Rect::from_min_size(
                canvas_rect.min + card.pos.to_vec2(),
                card.size,
            );
            let mut moved = false;
            for other in left.iter().chain(right.iter()) {
                let other_rect = egui::Rect::from_min_size(
                    canvas_rect.min + other.pos.to_vec2(),
                    other.size,
                );
                if !rect.intersects(other_rect) {
                    continue;
                }
                let move_x = if rect.center().x < other_rect.center().x {
                    other_rect.left() - rect.right()
                } else {
                    other_rect.right() - rect.left()
                };
                let move_y = if rect.center().y < other_rect.center().y {
                    other_rect.top() - rect.bottom()
                } else {
                    other_rect.bottom() - rect.top()
                };
                if move_x.abs() < move_y.abs() {
                    card.pos.x += move_x;
                } else {
                    card.pos.y += move_y;
                }
                card.pos = constrain_card_pos(card.pos, card.size, canvas_rect, inspector_rect);
                rect = egui::Rect::from_min_size(
                    canvas_rect.min + card.pos.to_vec2(),
                    card.size,
                );
                moved = true;
            }
            if !moved {
                break;
            }
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

    fn next_advanced_id(&mut self) -> u64 {
        let id = self.advanced_next_id;
        self.advanced_next_id = self.advanced_next_id.wrapping_add(1);
        id
    }

    fn sync_advanced_from_timeline(&mut self) {
        self.advanced_connections.clear();
        self.advanced_connecting = None;
        let key_count = self.space_path.len().max(1);
        let card_size = egui::vec2(220.0, 150.0);
        let mut durations = vec![2.0; key_count];
        if self.key_times.len() == key_count {
            let seg_count = key_count.saturating_sub(1);
            for i in 0..seg_count {
                let t0 = self.key_times[i];
                let t1 = self.key_times[i + 1];
                let seg = (t1 - t0).max(0.01);
                durations[i] = seg / self.speed.max(0.01);
            }
            if key_count > 1 {
                durations[key_count - 1] = durations[key_count - 2];
            }
        }
        if self.advanced_cards.len() == key_count {
            for i in 0..key_count {
                if let Some(card) = self.advanced_cards.get_mut(i) {
                    card.space_idx = self.space_path.get(i).copied().unwrap_or(0);
                    card.color_key = self
                        .color_path
                        .get(i)
                        .cloned()
                        .unwrap_or(ColorKey::Current);
                    card.duration_sec = durations[i].max(0.1);
                }
            }
        } else {
            self.advanced_cards.clear();
            self.advanced_selected_card = None;
            self.advanced_selected_cards.clear();
            self.advanced_preview_card = None;
            let cols = 3usize;
            let start_x = 12.0 + 320.0 + 24.0;
            let start_y = 12.0;
            for i in 0..key_count {
                let col = i % cols;
                let row = i / cols;
                let pos = egui::pos2(
                    start_x + col as f32 * (card_size.x + 16.0),
                    start_y + row as f32 * (card_size.y + 16.0),
                );
                let id = self.next_advanced_id();
                self.advanced_cards.push(AdvancedCard {
                    id,
                    space_idx: self.space_path.get(i).copied().unwrap_or(0),
                    color_key: self.color_path.get(i).cloned().unwrap_or(ColorKey::Current),
                    duration_sec: durations[i].max(0.1),
                    pos,
                    size: card_size,
                    in_enabled: true,
                    out_enabled: true,
                    filter: None,
                });
            }
        }
        if self.advanced_cards.len() >= 2 {
            for i in 0..self.advanced_cards.len() - 1 {
                self.advanced_connections
                    .push(AdvancedConnection { from: i, to: i + 1 });
            }
        }
    }

    fn apply_advanced_to_timeline(&mut self) {
        if self.advanced_cards.len() < 2 {
            return;
        }
        let orig_space = self.space_path.clone();
        if orig_space.is_empty() {
            return;
        }
        let mut new_space = Vec::with_capacity(self.advanced_cards.len());
        let mut new_color = Vec::with_capacity(self.advanced_cards.len());
        for card in &self.advanced_cards {
            let idx = card.space_idx.min(orig_space.len().saturating_sub(1));
            new_space.push(idx);
            new_color.push(card.color_key.clone());
        }
        self.space_path = new_space;
        self.color_path = new_color;
        self.ensure_color_path_len(self.space_path.len());
        let segs = self.space_path.len().saturating_sub(1);
        let mut durations = Vec::with_capacity(segs);
        for card in self.advanced_cards.iter().take(segs) {
            durations.push(card.duration_sec.max(0.05));
        }
        let total = durations.iter().sum::<f32>().max(0.05);
        self.key_times.clear();
        self.key_times.push(0.0);
        let mut acc = 0.0;
        for d in durations {
            acc += d / total;
            self.key_times.push(acc.clamp(0.0, 1.0));
        }
        if let Some(last) = self.key_times.last_mut() {
            *last = 1.0;
        }
        self.speed = (1.0 / total).clamp(0.01, 2.0);
        if self.key_collapsed.len() != self.space_path.len() {
            self.key_collapsed.resize(self.space_path.len(), false);
        }
        self.selected_key_idx = Some(0);
        self.reset_view_key_idx = self
            .reset_view_key_idx
            .min(self.space_path.len().saturating_sub(1));
        self.from_space = *self.space_path.first().unwrap_or(&self.from_space);
        self.to_space = *self.space_path.last().unwrap_or(&self.to_space);
    }

    fn mark_grid_dirty(&mut self) {
        self.grid_version = self.grid_version.wrapping_add(1);
        self.grid_cache = None;
    }
}

fn render_advanced_card_controls(
    ui: &mut egui::Ui,
    card: &mut AdvancedCard,
    ds: Option<&Dataset>,
    categorical_opts: &[(usize, String)],
    continuous_opts: &[(usize, String)],
    control_width: f32,
) {
    let Some(ds) = ds else {
        ui.label("No dataset.");
        return;
    };
    if ds.meta.spaces.is_empty() {
        ui.label("No spaces.");
        return;
    }

    let max_idx = ds.meta.spaces.len().saturating_sub(1);
    let mut space_idx = card.space_idx.min(max_idx);
    let mut color_key = card.color_key.clone();
    let mut color_kind = match color_key {
        ColorKey::Current => KeyColorKind::Current,
        ColorKey::Categorical(_) => KeyColorKind::Categorical,
        ColorKey::Continuous(_) => KeyColorKind::Continuous,
        ColorKey::Gene(_) => KeyColorKind::Gene,
    };

    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Space").size(10.5));
        egui::ComboBox::from_id_salt(("adv_space", card.id))
            .width(control_width)
            .selected_text(
                ds.meta
                    .spaces
                    .get(space_idx)
                    .map(|s| s.name.as_str())
                    .unwrap_or("?"),
            )
            .show_ui(ui, |ui| {
                for (idx, s) in ds.meta.spaces.iter().enumerate() {
                    ui.selectable_value(&mut space_idx, idx, &s.name);
                }
            });
    });

    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Color").size(10.5));
        let kind_label = match color_kind {
            KeyColorKind::Current => "Current",
            KeyColorKind::Categorical => "Categorical",
            KeyColorKind::Continuous => "Continuous",
            KeyColorKind::Gene => "Gene",
        };
        egui::ComboBox::from_id_salt(("adv_color_kind", card.id))
            .width(control_width)
            .selected_text(kind_label)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut color_kind, KeyColorKind::Current, "Current");
                ui.selectable_value(&mut color_kind, KeyColorKind::Categorical, "Categorical");
                ui.selectable_value(&mut color_kind, KeyColorKind::Continuous, "Continuous");
                ui.selectable_value(&mut color_kind, KeyColorKind::Gene, "Gene");
            });
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
                egui::ComboBox::from_id_salt(("adv_color_cat", card.id))
                    .width(control_width)
                    .selected_text(
                        categorical_opts
                            .iter()
                            .find(|(idx, _)| *idx == obs_idx)
                            .map(|(_, name)| name.as_str())
                            .unwrap_or("?"),
                    )
                    .show_ui(ui, |ui| {
                        for (idx, name) in categorical_opts {
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
                egui::ComboBox::from_id_salt(("adv_color_cont", card.id))
                    .width(control_width)
                    .selected_text(
                        continuous_opts
                            .iter()
                            .find(|(idx, _)| *idx == obs_idx)
                            .map(|(_, name)| name.as_str())
                            .unwrap_or("?"),
                    )
                    .show_ui(ui, |ui| {
                        for (idx, name) in continuous_opts {
                            ui.selectable_value(&mut obs_idx, *idx, name);
                        }
                    });
                color_key = ColorKey::Continuous(obs_idx);
            }
        }
        KeyColorKind::Gene => {
            if let Some(expr) = ds.meta.expr.as_ref() {
                let mut gene = match &color_key {
                    ColorKey::Gene(name) => name.clone(),
                    _ => String::new(),
                };
                let edit = egui::TextEdit::singleline(&mut gene)
                    .id(egui::Id::new(("adv_gene", card.id)))
                    .hint_text("Exact gene name")
                    .desired_width(control_width);
                ui.add(edit);
                if !gene.trim().is_empty() && !expr.var_names.iter().any(|name| name == &gene) {
                    ui.label("No exact match.");
                }
                color_key = ColorKey::Gene(gene);
            } else {
                ui.label("No gene data.");
                color_key = ColorKey::Current;
            }
        }
    }

    card.space_idx = space_idx;
    card.color_key = color_key;
}

impl Drop for StvizApp {
    fn drop(&mut self) {
        Self::cleanup_mock_artifacts(&self.output_dir);
    }
}

impl eframe::App for StvizApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        ctx.set_zoom_factor(self.ui_scale);
        ctx.set_visuals(Self::visuals_for_theme(self.ui_theme));
        self.update_main_view_refresh(ctx, frame);
        self.handle_screenshot_events(ctx);
        self.handle_hotkeys(ctx);
        self.poll_convert_job();
        if self.convert_running {
            self.refresh_convert_log();
        }
        self.maybe_update_playback(ctx);

        if self.viewport_fullscreen {
            let is_fullscreen = ctx
                .input(|i| i.viewport().fullscreen)
                .unwrap_or(false);
            if !is_fullscreen {
                self.viewport_fullscreen = false;
            }
        }

        if self.viewport_fullscreen {
            egui::CentralPanel::default().show(ctx, |ui| {
                self.ui_viewport(ui, ctx);
            });
            self.viewport_fullscreen_overlay(ctx);
            return;
        }

        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("GPU point transitions");
            });
        });

        egui::SidePanel::left("left_panel")
            .resizable(true)
            .default_width(320.0)
            .max_width(480.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
                    self.ui_left_panel(ui, ctx);
                });
            });

        egui::TopBottomPanel::bottom("timeline_bar")
            .resizable(true)
            .default_height(self.timeline_height)
            .min_height(120.0)
            .max_height(280.0)
            .show(ctx, |ui| {
                self.ui_timeline_bar(ui, ctx);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.ui_viewport(ui, ctx);
        });
        self.ui_advanced_timeline(ctx);

        if self.convert_running {
            ctx.request_repaint_after(self.main_view_frame_interval());
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

    let encoder = image::codecs::png::PngEncoder::new_with_quality(
        std::fs::File::create(path)?,
        image::codecs::png::CompressionType::Best,
        image::codecs::png::FilterType::Adaptive,
    );
    encoder.write_image(&rgba, cw, ch, image::ColorType::Rgba8.into())?;
    Ok(())
}

fn save_rgba_png(path: &Path, width: u32, height: u32, rgba: &[u8]) -> anyhow::Result<()> {
    use image::ImageEncoder;

    std::fs::create_dir_all(path.parent().unwrap_or(Path::new("."))).ok();
    let file = std::fs::File::create(path)?;
    let encoder = image::codecs::png::PngEncoder::new_with_quality(
        file,
        image::codecs::png::CompressionType::Best,
        image::codecs::png::FilterType::Adaptive,
    );
    encoder.write_image(
        rgba,
        width,
        height,
        image::ExtendedColorType::Rgba8,
    )?;
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

fn render_filter_rows(
    ui: &mut egui::Ui,
    range: std::ops::Range<usize>,
    categories: &[String],
    palette_slice: Option<&[u32]>,
    enabled_categories: &mut [bool],
    overrides: &mut [Option<u32>],
    filter_changed: &mut bool,
    colors_changed: &mut bool,
) {
    for i in range {
        let c = &categories[i];
        let mut color = palette_slice
            .and_then(|pal| pal.get(i).copied())
            .unwrap_or(pack_rgba8(200, 200, 200, 255));
        if let Some(override_color) = overrides.get(i).and_then(|c| *c) {
            color = override_color;
        }
        let mut color = color32_from_packed(color);
        ui.horizontal(|ui| {
            if ui.checkbox(&mut enabled_categories[i], "").changed() {
                *filter_changed = true;
            }
            if ui.color_edit_button_srgba(&mut color).changed() {
                if i < overrides.len() {
                    overrides[i] = Some(pack_rgba8(
                        color.r(),
                        color.g(),
                        color.b(),
                        color.a(),
                    ));
                }
                *colors_changed = true;
            }
            ui.label(c);
        });
    }
}

fn constrain_card_pos(
    pos: egui::Pos2,
    size: egui::Vec2,
    canvas_rect: egui::Rect,
    inspector_rect: egui::Rect,
) -> egui::Pos2 {
    let mut out = pos;
    let max_x = (canvas_rect.width() - size.x).max(0.0);
    let max_y = (canvas_rect.height() - size.y).max(0.0);
    out.x = out.x.clamp(0.0, max_x);
    out.y = out.y.clamp(0.0, max_y);
    let rect = egui::Rect::from_min_size(canvas_rect.min + out.to_vec2(), size);
    if rect.intersects(inspector_rect) {
        let below_y = inspector_rect.bottom() + 12.0;
        if below_y + size.y <= canvas_rect.bottom() {
            out.y = below_y - canvas_rect.min.y;
        } else {
            let right_x = inspector_rect.right() + 12.0;
            if right_x + size.x <= canvas_rect.right() {
                out.x = right_x - canvas_rect.min.x;
            } else {
                out.x = max_x;
                out.y = max_y;
            }
        }
    }
    out
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
