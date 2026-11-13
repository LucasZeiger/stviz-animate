#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod camera;
mod color;
mod data;
mod render;

fn load_app_icon() -> Option<eframe::egui::IconData> {
    let bytes = include_bytes!("../assets/icon.jpg");
    let image = image::load_from_memory(bytes).ok()?.into_rgba8();
    let (width, height) = image.dimensions();
    Some(eframe::egui::IconData {
        rgba: image.into_raw(),
        width,
        height,
    })
}

type DeviceDescriptorFn = std::sync::Arc<
    dyn Fn(&egui_wgpu::wgpu::Adapter) -> egui_wgpu::wgpu::DeviceDescriptor<'static> + Send + Sync,
>;

/// Point positions and colours live in one storage buffer each. wgpu's default
/// limits cap a storage binding at 128 MiB, which is exactly 2^24 (16.7M) cells
/// of xy positions, so ask for whatever the GPU actually supports instead
/// (typically 2-4 GiB, i.e. hundreds of millions of cells).
fn with_large_buffer_limits(base: DeviceDescriptorFn) -> DeviceDescriptorFn {
    std::sync::Arc::new(move |adapter| {
        let mut desc = base(adapter);
        let supported = adapter.limits();
        let limits = &mut desc.required_limits;
        limits.max_storage_buffer_binding_size = limits
            .max_storage_buffer_binding_size
            .max(supported.max_storage_buffer_binding_size);
        limits.max_buffer_size = limits.max_buffer_size.max(supported.max_buffer_size);
        desc
    })
}

fn main() -> eframe::Result<()> {
    let mut viewport = eframe::egui::ViewportBuilder::default()
        .with_title("stviz-animate")
        .with_inner_size([1200.0, 800.0]);
    if let Some(icon) = load_app_icon() {
        viewport = viewport.with_icon(icon);
    }
    let mut native_options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };
    let mut wgpu_config = egui_wgpu::WgpuConfiguration::default();
    if cfg!(target_os = "windows") {
        wgpu_config.desired_maximum_frame_latency = Some(1);
    }
    if let egui_wgpu::WgpuSetup::CreateNew(mut setup) = wgpu_config.wgpu_setup {
        if cfg!(target_os = "windows") && std::env::var("WGPU_BACKEND").is_err() {
            setup.instance_descriptor.backends =
                egui_wgpu::wgpu::Backends::DX12 | egui_wgpu::wgpu::Backends::GL;
        }
        setup.device_descriptor = with_large_buffer_limits(setup.device_descriptor);
        wgpu_config.wgpu_setup = egui_wgpu::WgpuSetup::CreateNew(setup);
    }
    native_options.wgpu_options = wgpu_config;

    eframe::run_native(
        "stviz-animate",
        native_options,
        Box::new(|cc| Ok(Box::new(app::StvizApp::new(cc)))),
    )
}
