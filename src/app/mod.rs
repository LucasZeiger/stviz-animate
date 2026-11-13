use crate::{
    camera::Camera2D,
    data::Dataset,
    render::{PointCloudGpu, SharedRender},
};
use eframe::egui;
use egui_wgpu::wgpu;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    process::{Child, ChildStdin},
    sync::Arc,
};

mod advanced;
mod convert;
mod dataset;
mod env;
mod export;
mod export_job;
mod fit_text;
mod help;
mod history;
mod input;
mod keyframes;
mod layout;
mod left_panel;
mod overlay;
mod process;
mod render_callback;
mod scenes;
mod selection;
mod settings;
mod style;
mod timeline;
mod util;
mod viewport;
mod window;
mod workspace;

const MAX_FILTER_CATEGORIES: usize = 5000;
const MAX_GRID_CATEGORIES: usize = 512;
const APP_DATA_DIR_NAME: &str = "stviz-animate";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum ExportQuality {
    Current,
    FullHd,
    UltraHd,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum ExportVideoQuality {
    Standard,
    High,
    Ultra,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
enum ExportEncodingSpeed {
    Fast,
    Balanced,
    #[default]
    SmallestFile,
}

impl ExportVideoQuality {
    /// libx264 constant rate factor (lower is higher quality).
    fn crf(self) -> u32 {
        match self {
            Self::Standard => 23,
            Self::High => 18,
            Self::Ultra => 14,
        }
    }
}

impl ExportEncodingSpeed {
    fn preset(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Balanced => "medium",
            Self::SmallestFile => "slow",
        }
    }
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
    filter_path: Vec<Option<KeyframeFilter>>,
    color_cache: HashMap<ColorKey, ColorCacheEntry>,
    filtered_color_cache: HashMap<u64, Arc<Vec<u32>>>,
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
    /// Scale points with zoom (fixed size in world units) instead of fixed pixels.
    point_auto_scale: bool,
    /// Largest radius (logical px) auto-scaling may grow points to.
    point_auto_max_px: f32,
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
    app_data_dir: PathBuf,
    output_dir: PathBuf,
    logs_dir: PathBuf,
    export_root_dir: PathBuf,
    converter_venv_dir: PathBuf,
    screenshot_dir: PathBuf,
    screenshot_name: String,
    exporting_loop: bool,
    export_fps: u32,
    export_duration_sec: f32,
    export_quality: ExportQuality,
    export_resolution: Option<[u32; 2]>,
    export_video_quality: ExportVideoQuality,
    export_encoding_speed: ExportEncodingSpeed,
    export_dir: PathBuf,
    export_name: String,
    export_output_path: Option<PathBuf>,
    export_temp_output_path: Option<PathBuf>,
    export_total_frames: u32,
    export_frame_index: u32,
    export_status: Option<String>,
    export_run_ffmpeg: bool,
    export_keep_frames: bool,
    export_camera: Option<Camera2D>,
    direct_video_export: Option<DirectVideoExport>,
    /// Background encode or ffmpeg install; see `export_job`.
    export_job: Option<export_job::ExportJob>,
    export_pending_frames: u32,
    export_finishing: bool,
    export_cancelled: bool,
    ffmpeg_available: bool,
    ffmpeg_path: Option<PathBuf>,
    ffmpeg_auto_install: bool,
    ffmpeg_auto_install_attempted: bool,
    export_log_path: Option<PathBuf>,
    export_log_text: String,
    export_log_open: bool,
    export_log_focus: bool,

    // .h5ad -> .stviz conversion
    convert_generate_only: bool,
    convert_include_expr: bool,
    convert_expr_source: convert::ExprSource,
    convert_layer: String,
    convert_gene_symbols: String,
    convert_status: Option<String>,
    convert_job: Option<convert::ConvertJob>,
    /// A conversion that stopped because Python is missing; the Install
    /// Python button installs it and runs this again.
    convert_retry: Option<convert::ConvertRequest>,
    /// A .stviz being opened on a worker thread; see `start_dataset_load`.
    dataset_load: Option<dataset::DatasetLoad>,
    convert_last_python_exe: Option<String>,
    convert_log_path: Option<PathBuf>,
    convert_log_text: String,
    mock_cells: u32,
    mock_last_h5ad: Option<PathBuf>,
    mock_last_stviz: Option<PathBuf>,

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

    /// Title, legend and scale bar drawn over the view and into exports.
    overlay: overlay::OverlaySettings,
    /// Hover details, legend highlighting and the lasso selection.
    selection: selection::SelectionState,
    /// Named scenes stored in the open .stviz file.
    scenes: scenes::SceneState,
    /// Undo and redo of view edits.
    history: history::History,

    // Legend / gene input
    legend_range: Option<LegendRange>,
    active_legend_range: Option<LegendRange>,
    gene_query: String,
    gene_selected: Option<String>,

    /// Map continuous/gene colours onto the 1st-99th percentile range.
    clip_outliers: bool,
    // Persistent settings
    settings_path: PathBuf,
    saved_settings: settings::AppSettings,
    settings_checked_at: f64,
    last_dir: Option<PathBuf>,
    recent_files: Vec<PathBuf>,
    /// Keyboard shortcuts window, toggled with ? or F1.
    show_shortcuts: bool,
    /// A file passed on the command line, opened on the first frame.
    startup_file: Option<PathBuf>,
    // Open path fallback + status
    open_path: String,
    workspace_status: Option<String>,
    last_error: Option<String>,

    // Viewport (for interactions)
    last_viewport_points: egui::Rect,
    last_viewport_px: [f32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum PlaybackMode {
    Once,
    Loop,
    PingPong,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum SampleGridLabelMode {
    Default,
    Custom,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum UiTheme {
    Dark,
    Light,
    Slate,
    Matrix,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
    filter_sig: u64,
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
    cached_max_draw: usize,
}

#[derive(Clone, Debug)]
struct KeyframeFilter {
    obs_idx: usize,
    enabled: Vec<bool>,
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

struct DirectVideoExport {
    child: Child,
    stdin: Option<ChildStdin>,
    stderr_path: PathBuf,
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
        let mut app_data_dir = Self::resolve_app_data_dir();
        if std::fs::create_dir_all(&app_data_dir).is_err() {
            app_data_dir = std::env::temp_dir().join(APP_DATA_DIR_NAME);
            let _ = std::fs::create_dir_all(&app_data_dir);
        }
        let output_dir = app_data_dir.join("output");
        let logs_dir = app_data_dir.join("logs");
        let export_root_dir = output_dir.join("exports");
        let screenshot_dir = output_dir.join("screenshots");
        let converter_venv_dir = app_data_dir.join(".stviz_venv");
        let _ = std::fs::create_dir_all(&output_dir);
        let _ = std::fs::create_dir_all(&logs_dir);
        let _ = std::fs::create_dir_all(&export_root_dir);
        let _ = std::fs::create_dir_all(&screenshot_dir);
        let _ = std::fs::create_dir_all(&converter_venv_dir);
        Self::cleanup_mock_artifacts(&output_dir);
        let ffmpeg_path = Self::resolve_ffmpeg_path(&project_dir);
        let settings_path = app_data_dir.join("settings.json");
        let saved_settings = settings::AppSettings::load(&settings_path);

        let mut app = Self {
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
            filter_path: Vec::new(),
            color_cache: HashMap::new(),
            filtered_color_cache: HashMap::new(),
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
            point_auto_scale: false,
            point_auto_max_px: 12.0,
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
            app_data_dir: app_data_dir.clone(),
            output_dir: output_dir.clone(),
            logs_dir: logs_dir.clone(),
            export_root_dir: export_root_dir.clone(),
            converter_venv_dir: converter_venv_dir.clone(),
            screenshot_dir: screenshot_dir.clone(),
            screenshot_name: String::from("stviz-animate_screenshot"),
            exporting_loop: false,
            export_fps: 30,
            export_duration_sec: 5.0,
            export_quality: ExportQuality::Current,
            export_resolution: None,
            export_video_quality: ExportVideoQuality::High,
            export_encoding_speed: ExportEncodingSpeed::SmallestFile,
            export_dir: export_root_dir.clone(),
            export_name: String::from("stviz-animate_loop.mp4"),
            export_output_path: None,
            export_temp_output_path: None,
            export_total_frames: 0,
            export_frame_index: 0,
            export_status: None,
            export_run_ffmpeg: true,
            export_keep_frames: false,
            export_camera: None,
            direct_video_export: None,
            export_job: None,
            export_pending_frames: 0,
            export_finishing: false,
            export_cancelled: false,
            ffmpeg_path: ffmpeg_path.clone(),
            ffmpeg_available: ffmpeg_path.is_some(),
            // Off by default: installing runs the system package manager.
            ffmpeg_auto_install: false,
            ffmpeg_auto_install_attempted: false,
            export_log_path: None,
            export_log_text: String::new(),
            export_log_open: false,
            export_log_focus: false,

            convert_generate_only: false,
            convert_include_expr: true,
            convert_expr_source: convert::ExprSource::X,
            convert_layer: String::new(),
            convert_gene_symbols: String::new(),
            convert_status: None,
            convert_job: None,
            convert_retry: None,
            dataset_load: None,
            convert_last_python_exe: None,
            convert_log_path: None,
            convert_log_text: String::new(),
            mock_cells: 300_000,
            mock_last_h5ad: None,
            mock_last_stviz: None,

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

            overlay: overlay::OverlaySettings::default(),
            selection: selection::SelectionState::new(),
            scenes: scenes::SceneState::default(),
            history: history::History::default(),
            legend_range: None,
            active_legend_range: None,
            gene_query: String::new(),
            gene_selected: None,

            clip_outliers: false,
            settings_path,
            saved_settings: saved_settings.clone(),
            settings_checked_at: f64::NEG_INFINITY,
            last_dir: None,
            recent_files: Vec::new(),
            show_shortcuts: false,
            // `stviz-animate <file>` (or "Open with...") opens a .h5ad/.stviz.
            startup_file: std::env::args_os().nth(1).map(PathBuf::from),
            open_path: String::new(),
            workspace_status: None,
            last_error: None,

            last_viewport_points: egui::Rect::ZERO,
            last_viewport_px: [0.0, 0.0],
        };
        app.apply_settings(&saved_settings);
        if app.fullscreen {
            app.apply_fullscreen(&cc.egui_ctx);
        } else {
            cc.egui_ctx
                .send_viewport_cmd(egui::ViewportCommand::Maximized(true));
        }
        app
    }
}

impl Drop for StvizApp {
    fn drop(&mut self) {
        self.abort_direct_video_export();
        if let Some(job) = self.convert_job.take() {
            job.abort();
        }
        if let Some(job) = self.export_job.take() {
            job.cancel();
            let _ = job.join();
        }
        // Release the memory-mapped dataset first: Windows cannot delete a
        // mapped file, so a loaded mock dataset would otherwise be left behind.
        self.dataset = None;
        self.shared.params.lock().dataset = None;
        Self::cleanup_mock_artifacts(&self.output_dir);
        self.save_settings_if_changed(None);
    }
}

impl eframe::App for StvizApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        ctx.set_zoom_factor(self.ui_scale);
        ctx.set_visuals(Self::visuals_for_theme(self.ui_theme));
        self.update_main_view_refresh(ctx, frame);
        self.handle_screenshot_events(ctx);
        self.poll_export_job(ctx);
        self.handle_hotkeys(ctx);
        self.handle_dropped_files(ctx);
        if let Some(path) = self.startup_file.take() {
            self.open_data_file(&path);
        }
        self.poll_convert_job();
        self.poll_dataset_load();
        if let Some(wait) = self.save_settings_if_changed(Some(ctx.input(|i| i.time))) {
            // The UI only repaints on input; come back to save the pending change.
            ctx.request_repaint_after(std::time::Duration::from_secs_f64(wait));
        }
        self.maybe_update_playback(ctx);
        let render_prepare_err = { self.shared.last_prepare_error.lock().take() };
        if let Some(err) = render_prepare_err {
            self.last_error = Some(err.clone());
            if self.exporting_loop || self.export_finishing {
                self.append_export_log(&err);
                self.export_status = Some(format!("Export failed: {err}"));
                self.exporting_loop = false;
                self.export_finishing = false;
                self.export_cancelled = false;
                self.export_camera = None;
                let frames_dir = self.export_dir.clone();
                self.remove_export_frames(&frames_dir);
                self.remove_export_output();
            }
        }

        if self.viewport_fullscreen {
            let is_fullscreen = ctx.input(|i| i.viewport().fullscreen).unwrap_or(false);
            if !is_fullscreen {
                self.viewport_fullscreen = false;
            }
        }

        if self.viewport_fullscreen {
            egui::CentralPanel::default().show(ctx, |ui| {
                self.ui_viewport(ui, ctx);
            });
            self.viewport_fullscreen_overlay(ctx);
            self.ui_advanced_timeline(ctx);
            self.ui_shortcuts_window(ctx);
            self.update_history(ctx);
            return;
        }

        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            self.ui_top_bar(ui);
        });

        egui::SidePanel::left("left_panel")
            .resizable(true)
            .default_width(320.0)
            .min_width(fit_text::LEFT_PANEL_MIN_WIDTH)
            .max_width(fit_text::left_panel_max_width(ctx.content_rect().width()))
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .show(ui, |ui| {
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
        self.ui_shortcuts_window(ctx);
        self.update_history(ctx);

        if self.convert_running() || self.dataset_load.is_some() {
            ctx.request_repaint_after(self.main_view_frame_interval());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{GridCacheKey, StvizApp};

    #[test]
    fn advanced_filter_signature_tracks_max_draw() {
        let enabled = [true, false, true, true];
        let a = StvizApp::advanced_filter_signature(7, 3, 1000, &enabled);
        let b = StvizApp::advanced_filter_signature(7, 3, 250, &enabled);
        assert_ne!(a, b);
    }

    #[test]
    fn downsample_indices_respects_limit_and_order() {
        let idx: Vec<u32> = (0..10).collect();
        let out = StvizApp::downsample_indices(3, &idx);
        assert_eq!(out, vec![0, 4, 8]);
    }

    #[test]
    fn grid_cache_key_includes_filter_signature() {
        let base = GridCacheKey {
            dataset_id: 1,
            space_idx: 2,
            obs_idx: 3,
            use_filter: true,
            filter_sig: 111,
            version: 4,
        };
        let different_filter = GridCacheKey {
            filter_sig: 222,
            ..base.clone()
        };
        assert_ne!(base, different_filter);
    }
}
