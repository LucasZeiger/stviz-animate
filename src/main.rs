mod app;
mod camera;
mod color;
mod data;
mod render;

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("stviz-animate")
            .with_inner_size([1200.0, 800.0]),
        ..Default::default()
    };

    eframe::run_native(
        "stviz-animate",
        native_options,
        Box::new(|cc| Ok(Box::new(app::StvizApp::new(cc)))),
    )
}
