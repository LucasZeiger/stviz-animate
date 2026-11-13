//! Video/frame export and screenshots.

use super::{
    export_job::{
        install_ffmpeg, remove_frames_dir, remove_temp_output, run_encode, EncodePlan, ExportJob,
        ExportJobKind, JobOutcome,
    },
    overlay::Canvas,
    util::{
        apply_ease, chrono_like_timestamp, effective_point_radius, obs_name, zoom_relative_to_fit,
    },
    ColorKey, DirectVideoExport, ExportQuality, StvizApp,
};
use crate::{
    camera::Camera2D,
    render::{PointCloudGpu, Uniforms},
};
use eframe::egui;
use egui_wgpu::wgpu;
use std::{
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

/// Output size of the "Screenshot" button.
pub(super) const SCREENSHOT_RESOLUTION: [u32; 2] = [3840, 2160];

impl StvizApp {
    fn try_start_direct_video_export(&mut self) -> Result<bool, String> {
        if !self.export_run_ffmpeg || self.export_keep_frames {
            return Ok(false);
        }
        let Some(size) = self.export_resolution else {
            return Ok(false);
        };
        let Some(ffmpeg) = self.ffmpeg_path.clone() else {
            return Ok(false);
        };
        let Some(output) = self.export_temp_output_path.clone() else {
            return Ok(false);
        };
        let Some(render_state) = self.render_state.as_ref() else {
            return Ok(false);
        };
        let input_format = match render_state.target_format {
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => "bgra",
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => "rgba",
            _ => return Ok(false),
        };
        let crf = self.export_video_quality.crf();
        let output_format = "yuv420p";
        let preset = self.export_encoding_speed.preset();
        let stderr_path = self.export_dir.join("ffmpeg_stderr.log");
        let stderr =
            std::fs::File::create(&stderr_path).map_err(|e| format!("create ffmpeg log: {e}"))?;
        let mut cmd = Command::new(ffmpeg);
        Self::apply_subprocess_flags(&mut cmd);
        let mut child = cmd
            .arg("-y")
            .arg("-f")
            .arg("rawvideo")
            .arg("-pix_fmt")
            .arg(input_format)
            .arg("-video_size")
            .arg(format!("{}x{}", size[0], size[1]))
            .arg("-framerate")
            .arg(self.export_fps.to_string())
            .arg("-i")
            .arg("pipe:0")
            .arg("-an")
            .arg("-c:v")
            .arg("libx264")
            .arg("-preset")
            .arg(preset)
            .arg("-crf")
            .arg(crf.to_string())
            .arg("-pix_fmt")
            .arg(output_format)
            .arg("-movflags")
            .arg("+faststart")
            .arg(&output)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::from(stderr))
            .spawn()
            .map_err(|e| format!("start direct ffmpeg export: {e}"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "ffmpeg stdin unavailable".to_string())?;
        self.direct_video_export = Some(DirectVideoExport {
            child,
            stdin: Some(stdin),
            stderr_path,
        });
        self.append_export_log(&format!(
            "Direct ffmpeg streaming enabled ({input_format} raw frames)."
        ));
        Ok(true)
    }

    pub(super) fn abort_direct_video_export(&mut self) {
        if let Some(mut direct) = self.direct_video_export.take() {
            direct.stdin.take();
            super::process::kill_tree(&mut direct.child);
        }
    }

    pub(super) fn start_export_loop(&mut self) {
        if self.export_busy() {
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
        self.export_dir = self.export_root_dir.join(format!("loop_{ts}"));
        let _ = std::fs::create_dir_all(&self.export_dir);
        self.export_log_path = Some(self.logs_dir.join(format!("export_log_{ts}.txt")));
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
        self.export_status = Some(format!("Exporting {total} frames ({:.2}s)...", duration));
        self.playing = false;
        self.export_camera = None;
        self.abort_direct_video_export();

        if let (Some(size), Some(ds)) = (self.export_resolution, self.dataset.clone()) {
            if let Some(bbox) = self.export_fit_bbox(&ds) {
                let mut camera = Camera2D::default();
                camera.fit_bbox(bbox, [size[0] as f32, size[1] as f32], 0.98);
                self.export_camera = Some(camera);
            }
        }

        self.append_export_log("Loop export started.");
        self.append_export_log(&format!("Frames directory: {}", self.export_dir.display()));
        let out_path = self.managed_export_output_path();
        if let Some(name) = out_path.file_name().and_then(|s| s.to_str()) {
            self.export_name = name.to_string();
        }
        let temp_out_path = self.managed_export_temp_path(&out_path);
        self.export_output_path = Some(out_path.clone());
        self.export_temp_output_path = Some(temp_out_path.clone());
        self.remove_export_output();
        self.export_temp_output_path = Some(temp_out_path.clone());
        self.append_export_log(&format!("Output video: {}", out_path.display()));
        self.append_export_log(&format!("Temporary video: {}", temp_out_path.display()));
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
            let crf = self.export_video_quality.crf();
            let preset = self.export_encoding_speed.preset();
            self.append_export_log(&format!(
                "Encoding: libx264, CRF {crf}, preset {preset}, yuv420p"
            ));
            if self.ffmpeg_path.is_none() {
                if self.ffmpeg_auto_install && !self.ffmpeg_auto_install_attempted {
                    self.append_export_log(
                        "ffmpeg missing; will try to install it after frames are rendered.",
                    );
                } else {
                    self.append_export_log("ffmpeg missing; will try OpenCV fallback.");
                }
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
                dataset_lines.push(format!("Sample grid: enabled (group by {grid_obs})"));
            }
        }
        for line in dataset_lines {
            self.append_export_log(&line);
        }
        match self.try_start_direct_video_export() {
            Ok(true) => self.append_export_log("PNG frame staging bypassed."),
            Ok(false) => self.append_export_log("Using PNG frame staging."),
            Err(e) => self.append_export_log(&format!(
                "Direct ffmpeg streaming unavailable ({e}); using PNG frame staging."
            )),
        }
    }

    /// Hand the finished frames to a background worker for encoding, so the
    /// UI stays responsive while ffmpeg/OpenCV run. See `export_job`.
    pub(super) fn finish_export_loop(&mut self) {
        let final_out = self
            .export_output_path
            .clone()
            .unwrap_or_else(|| self.managed_export_output_path());
        let temp_out = self
            .export_temp_output_path
            .take()
            .unwrap_or_else(|| self.managed_export_temp_path(&final_out));
        self.export_camera = None;
        let install_ffmpeg = self.export_run_ffmpeg
            && self.ffmpeg_path.is_none()
            && self.ffmpeg_auto_install
            && !self.ffmpeg_auto_install_attempted;
        if install_ffmpeg {
            self.ffmpeg_auto_install_attempted = true;
        }
        let plan = EncodePlan {
            direct: self.direct_video_export.take(),
            frames: self.export_total_frames,
            frames_dir: self.export_dir.clone(),
            temp_out,
            final_out,
            export_root: self.export_root_dir.clone(),
            output_root: self.output_dir.clone(),
            keep_frames: self.export_keep_frames,
            run_ffmpeg: self.export_run_ffmpeg,
            ffmpeg: self.ffmpeg_path.clone(),
            install_ffmpeg,
            fps: self.export_fps,
            crf: self.export_video_quality.crf(),
            preset: self.export_encoding_speed.preset(),
            project_dir: self.project_dir.clone(),
            venv_dir: self.converter_venv_dir.clone(),
        };
        self.export_status = Some(if plan.run_ffmpeg {
            "Encoding video...".to_string()
        } else {
            "Finishing export...".to_string()
        });
        self.export_job = Some(ExportJob::spawn(
            ExportJobKind::Encode,
            move |log, cancel| run_encode(plan, log, cancel),
        ));
    }

    /// Start the opt-in ffmpeg install in the background.
    pub(super) fn start_ffmpeg_install(&mut self) {
        if self.export_job.is_some() {
            return;
        }
        self.ffmpeg_auto_install_attempted = true;
        self.export_log_open = true;
        self.append_export_log("Installing ffmpeg...");
        self.export_status = Some("Installing ffmpeg...".to_string());
        let project_dir = self.project_dir.clone();
        self.export_job = Some(ExportJob::spawn(
            ExportJobKind::InstallFfmpeg,
            move |log, cancel| {
                let status = match install_ffmpeg(&project_dir, log, cancel) {
                    Ok(Some(path)) => format!("ffmpeg ready: {}", path.display()),
                    Ok(None) => "ffmpeg install did not complete; see the export log.".to_string(),
                    Err(_) => "ffmpeg install cancelled.".to_string(),
                };
                JobOutcome {
                    status: Some(status),
                    ffmpeg_changed: true,
                    ..JobOutcome::default()
                }
            },
        ));
    }

    /// Forward worker log lines into the export log and apply the result once
    /// the worker finishes. Called every frame from `update`.
    pub(super) fn poll_export_job(&mut self, ctx: &egui::Context) {
        let Some(job) = self.export_job.as_ref() else {
            return;
        };
        let lines = job.drain_log();
        let finished = job.is_finished();
        for line in lines {
            self.append_export_log(&line);
        }
        if !finished {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
            return;
        }
        let (outcome, lines) = self.export_job.take().unwrap().join();
        for line in lines {
            self.append_export_log(&line);
        }
        if let Some(python_exe) = outcome.python_exe {
            self.convert_last_python_exe = Some(python_exe);
        }
        if outcome.ffmpeg_changed {
            self.refresh_ffmpeg_path();
        }
        if outcome.status.is_some() {
            self.export_status = outcome.status;
        }
        ctx.request_repaint();
    }

    /// True while frames are rendering or a background export job is running.
    pub(super) fn export_busy(&self) -> bool {
        self.exporting_loop || self.export_finishing || self.export_job.is_some()
    }

    pub(super) fn export_job_kind(&self) -> Option<ExportJobKind> {
        self.export_job.as_ref().map(|job| job.kind)
    }

    pub(super) fn cancel_export_loop(&mut self) {
        if let Some(job) = self.export_job.as_ref() {
            if !job.is_cancelling() {
                job.cancel();
                self.export_status = Some("Cancelling...".to_string());
            }
            return;
        }
        if !self.exporting_loop && !self.export_finishing {
            return;
        }
        self.exporting_loop = false;
        self.export_finishing = false;
        self.export_cancelled = true;
        self.export_camera = None;
        self.export_status = Some("Export cancelled.".to_string());
        self.append_export_log("Export cancelled.");
        self.abort_direct_video_export();
        if self.export_pending_frames == 0 {
            self.export_cancelled = false;
            let frames_dir = self.export_dir.clone();
            self.remove_export_frames(&frames_dir);
            self.remove_export_output();
        }
    }

    pub(super) fn remove_export_frames(&mut self, frames_dir: &Path) {
        let mut lines = Vec::new();
        remove_frames_dir(&self.export_root_dir, frames_dir, &mut |s| {
            lines.push(s.to_string())
        });
        for line in lines {
            self.append_export_log(&line);
        }
    }

    pub(super) fn remove_export_output(&mut self) {
        let Some(path) = self.export_temp_output_path.take() else {
            return;
        };
        let mut lines = Vec::new();
        remove_temp_output(&self.export_root_dir, &path, &mut |s| {
            lines.push(s.to_string())
        });
        for line in lines {
            self.append_export_log(&line);
        }
    }

    pub(super) fn handle_screenshot_events(&mut self, ctx: &egui::Context) {
        // If we receive a screenshot event, save it.
        // We trigger screenshots with a viewport command carrying user_data.
        let events = ctx.input(|i| i.events.clone());
        for ev in events {
            if let egui::Event::Screenshot {
                image, user_data, ..
            } = ev
            {
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

    pub(super) fn request_screenshot(
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

    pub(super) fn append_export_log(&mut self, line: &str) {
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

    pub(super) fn render_export_frame(
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

        let (active_from, active_to, color_from, color_to, segment_t, _seg_idx) =
            self.current_segment(&ds);
        let viewport_px = [size[0] as f32, size[1] as f32];
        // How much bigger (or smaller) the export is than the on-screen viewport.
        let export_scale = if self.last_viewport_px[0] > 0.0 && self.last_viewport_px[1] > 0.0 {
            let scale_x = size[0] as f32 / self.last_viewport_px[0];
            let scale_y = size[1] as f32 / self.last_viewport_px[1];
            scale_x.min(scale_y)
        } else {
            1.0
        };
        let mut view_camera = self.export_camera.unwrap_or(self.camera);
        if self.export_camera.is_none() && !use_export_bbox {
            // Screenshot of the current view: keep the same framing at the
            // export resolution instead of revealing more of the plot.
            view_camera.pixels_per_unit =
                (view_camera.pixels_per_unit * export_scale).clamp(1e-6, 1e9);
        }
        if use_export_bbox && self.export_camera.is_none() {
            let bbox = self
                .current_plot_bbox_for_segment(
                    &ds,
                    active_from,
                    active_to,
                    segment_t,
                    draw_indices.as_slice(),
                    from_override.as_deref().map(|v| v.as_slice()),
                    to_override.as_deref().map(|v| v.as_slice()),
                )
                .or_else(|| self.export_fit_bbox(&ds));
            if let Some(bbox) = bbox {
                let mut cam = Camera2D::default();
                cam.fit_bbox(bbox, viewport_px, 0.98);
                view_camera = cam;
            }
        }
        // Labels are laid out now, while `self` is free to borrow, and drawn
        // into the frame after it is read back.
        let overlay_prims = self
            .overlay_content(&ds, |scale| view_camera.pixels_per_unit * scale)
            .map(|content| super::overlay::layout(&content, viewport_px))
            .unwrap_or_default();
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
        let color_t = if color_from != color_to || colors_id != colors_to_id {
            t_eased
        } else {
            0.0
        };
        // Same sizing as the viewport (DPI, then export resolution), so the
        // export looks like what is on screen.
        let ppp = if self.last_viewport_points.width() > 0.0 {
            self.last_viewport_px[0] / self.last_viewport_points.width()
        } else {
            1.0
        };
        let px_scale = ppp * export_scale;
        let point_radius_px = effective_point_radius(
            self.point_radius_px * px_scale,
            self.point_auto_scale,
            zoom_relative_to_fit(view_camera.pixels_per_unit, viewport_px),
            self.point_auto_max_px * px_scale,
        )
        // 4x MSAA spreads a 1 px point over several pixels and dims it.
        .max(1.0);
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
                self.offscreen_gpu_msaa = Some(PointCloudGpu::new_with_sample_count(
                    device,
                    target_format,
                    sample_count,
                ));
            }
            self.offscreen_gpu_msaa.as_mut().unwrap()
        } else {
            if self.offscreen_gpu.is_none() {
                self.offscreen_gpu = Some(PointCloudGpu::new(device, target_format));
            }
            self.offscreen_gpu.as_mut().unwrap()
        };
        gpu.prepare(
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
        )
        .map_err(|e| format!("GPU prepare failed: {e}"))?;

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
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
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
        if use_export_bbox && self.direct_video_export.is_some() {
            let mut frame = vec![0u8; (size[0] * size[1] * bytes_per_pixel) as usize];
            for row in 0..size[1] as usize {
                let src = row * padded_bytes_per_row as usize;
                let dst = row * unpadded_bytes_per_row as usize;
                frame[dst..dst + unpadded_bytes_per_row as usize]
                    .copy_from_slice(&data[src..src + unpadded_bytes_per_row as usize]);
            }
            drop(data);
            output_buffer.unmap();
            let is_bgra = matches!(
                target_format,
                wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
            );
            Canvas {
                pixels: &mut frame,
                width: size[0],
                height: size[1],
                bgra: is_bgra,
            }
            .draw(&overlay_prims);
            let stdin = self
                .direct_video_export
                .as_mut()
                .and_then(|direct| direct.stdin.as_mut())
                .ok_or_else(|| "ffmpeg input stream unavailable".to_string())?;
            stdin
                .write_all(&frame)
                .map_err(|e| format!("write raw frame to ffmpeg: {e}"))?;
            return Ok(());
        }

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
        Canvas {
            pixels: &mut rgba,
            width: size[0],
            height: size[1],
            bgra: false,
        }
        .draw(&overlay_prims);

        save_rgba_png(path, size[0], size[1], &rgba)
            .map_err(|e| format!("Export PNG failed: {e:#}"))?;
        Ok(())
    }
}

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
    encoder.write_image(rgba, width, height, image::ExtendedColorType::Rgba8)?;
    Ok(())
}

#[derive(Clone, Debug)]
struct ScreenshotRequest {
    path: PathBuf,
    crop_px: Option<[u32; 4]>,
    is_export_frame: bool,
}
