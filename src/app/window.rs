//! Window-level behaviour: fullscreen, display refresh rate and WSL detection.

use super::StvizApp;
use eframe::egui;

impl StvizApp {
    pub(super) fn apply_fullscreen(&self, ctx: &egui::Context) {
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

    pub(super) fn update_main_view_refresh(&mut self, ctx: &egui::Context, _frame: &eframe::Frame) {
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
        use raw_window_handle::{HasWindowHandle, RawWindowHandle};
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

    pub(super) fn is_wsl() -> bool {
        std::env::var_os("WSL_DISTRO_NAME").is_some() || std::env::var_os("WSL_INTEROP").is_some()
    }
}
