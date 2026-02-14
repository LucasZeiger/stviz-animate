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
    if cfg!(target_os = "windows") {
        let mut wgpu_config = egui_wgpu::WgpuConfiguration {
            desired_maximum_frame_latency: Some(1),
            ..Default::default()
        };
        if std::env::var("WGPU_BACKEND").is_err() {
            if let egui_wgpu::WgpuSetup::CreateNew(mut setup) = wgpu_config.wgpu_setup {
                setup.instance_descriptor.backends =
                    egui_wgpu::wgpu::Backends::DX12 | egui_wgpu::wgpu::Backends::GL;
                wgpu_config.wgpu_setup = egui_wgpu::WgpuSetup::CreateNew(setup);
            }
        }
        native_options.wgpu_options = wgpu_config;
    }

    eframe::run_native(
        "stviz-animate",
        native_options,
        Box::new(|cc| Ok(Box::new(app::StvizApp::new(cc)))),
    )
}
