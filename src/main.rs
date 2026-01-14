mod app;
mod camera;
mod color;
mod data;
mod render;

fn main() -> eframe::Result<()> {
    let mut native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("stviz-animate")
            .with_inner_size([1200.0, 800.0]),
        ..Default::default()
    };
    if cfg!(target_os = "windows") {
        native_options.vsync = false;
        native_options.wgpu_options = egui_wgpu::WgpuConfiguration {
            present_mode: egui_wgpu::wgpu::PresentMode::AutoNoVsync,
            desired_maximum_frame_latency: Some(1),
            ..Default::default()
        };
    }

    eframe::run_native(
        "stviz-animate",
        native_options,
        Box::new(|cc| Ok(Box::new(app::StvizApp::new(cc)))),
    )
}
