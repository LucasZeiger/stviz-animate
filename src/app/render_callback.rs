//! egui paint callback that draws the point cloud with wgpu.

use crate::render::{PointCloudGpu, SharedRender};
use eframe::egui;
use egui_wgpu::{wgpu, CallbackTrait};
use std::sync::Arc;

pub(super) struct PointCloudCallback {
    pub(super) shared: Arc<SharedRender>,
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

        match gpu.prepare(
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
        ) {
            Ok(()) => {
                self.shared.last_prepare_error.lock().take();
            }
            Err(e) => {
                *self.shared.last_prepare_error.lock() = Some(format!("GPU prepare failed: {e}"));
                return Vec::new();
            }
        }

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
