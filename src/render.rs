use anyhow::anyhow;
use bytemuck::{Pod, Zeroable};
use egui_wgpu::wgpu;
use egui_wgpu::wgpu::util::DeviceExt;
use parking_lot::Mutex;
use std::sync::Arc;

use crate::data::Dataset;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Uniforms {
    pub viewport_px: [f32; 2],
    pub _pad0: [f32; 2],
    pub center: [f32; 2],
    pub _pad1: [f32; 2],
    pub pixels_per_unit: f32,
    pub t: f32,
    pub point_radius_px: f32,
    pub mask_mode: f32,
    pub color_t: f32,
    pub _pad2: f32,
    pub _pad2b: [f32; 2],
    pub from_center: [f32; 2],
    pub from_scale: f32,
    pub _pad3: f32,
    pub to_center: [f32; 2],
    pub to_scale: f32,
    pub _pad4: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CornerVert {
    corner: [f32; 2],
}

pub struct PointCloudGpu {
    pub pipeline_alpha: wgpu::RenderPipeline,
    pub pipeline_opaque: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
    pub uniform_buf: wgpu::Buffer,

    pub pos_from: wgpu::Buffer,
    pub pos_to: wgpu::Buffer,
    pub colors_from: wgpu::Buffer,
    pub colors_to: wgpu::Buffer,
    pub draw_indices: wgpu::Buffer,

    pub corners: wgpu::Buffer,

    pub n_points: u32,
    pub n_draw: u32,

    pub target_format: wgpu::TextureFormat,

    // Change detection:
    pub last_dataset_id: u64,
    pub last_from_space: u32,
    pub last_to_space: u32,
    pub last_colors_from_id: u64,
    pub last_colors_to_id: u64,
    pub last_indices_id: u64,
    pub last_from_override_id: u64,
    pub last_to_override_id: u64,
}

impl PointCloudGpu {
    fn create_pipeline(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        blend: Option<wgpu::BlendState>,
        sample_count: u32,
    ) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
        let shader_src = include_str!("../assets/pointcloud.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pointcloud.wgsl"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pc_bgl"),
            entries: &[
                // uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            wgpu::BufferSize::new(std::mem::size_of::<Uniforms>() as u64).unwrap(),
                        ),
                    },
                    count: None,
                },
                // pos_from
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // pos_to
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // colors_from
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // colors_to
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // draw_indices
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pc_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pc_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<CornerVert>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: sample_count,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        (pipeline, bgl)
    }

    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        Self::new_with_sample_count(device, format, 1)
    }

    pub fn new_with_sample_count(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> Self {
        let (pipeline_alpha, bgl) = Self::create_pipeline(
            device,
            format,
            Some(wgpu::BlendState::ALPHA_BLENDING),
            sample_count,
        );
        let (pipeline_opaque, _) = Self::create_pipeline(device, format, None, sample_count);

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pc_uniform"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Start tiny, grow on demand
        let pos_from = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pc_pos_from"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pos_to = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pc_pos_to"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let colors_from = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pc_colors_from"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let colors_to = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pc_colors_to"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let draw_indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pc_draw_indices"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let corners_data = [
            CornerVert {
                corner: [-1.0, -1.0],
            },
            CornerVert {
                corner: [1.0, -1.0],
            },
            CornerVert { corner: [1.0, 1.0] },
            CornerVert {
                corner: [-1.0, -1.0],
            },
            CornerVert { corner: [1.0, 1.0] },
            CornerVert {
                corner: [-1.0, 1.0],
            },
        ];
        let corners = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pc_corners"),
            contents: bytemuck::cast_slice(&corners_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pc_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pos_from.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pos_to.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: colors_from.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: colors_to.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: draw_indices.as_entire_binding(),
                },
            ],
        });

        Self {
            pipeline_alpha,
            pipeline_opaque,
            bind_group,
            uniform_buf,
            pos_from,
            pos_to,
            colors_from,
            colors_to,
            draw_indices,
            corners,
            n_points: 0,
            n_draw: 0,
            target_format: format,
            last_dataset_id: 0,
            last_from_space: u32::MAX,
            last_to_space: u32::MAX,
            last_colors_from_id: 0,
            last_colors_to_id: 0,
            last_indices_id: 0,
            last_from_override_id: 0,
            last_to_override_id: 0,
        }
    }

    fn ensure_storage_buffer(
        device: &wgpu::Device,
        buf: &mut wgpu::Buffer,
        label: &str,
        need_bytes: u64,
    ) {
        if buf.size() >= need_bytes {
            return;
        }
        *buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: need_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    }

    fn write_chunked(queue: &wgpu::Queue, buf: &wgpu::Buffer, data: &[u8]) {
        const CHUNK: usize = 16 * 1024 * 1024; // 16MB
        let mut offset = 0usize;
        while offset < data.len() {
            let end = (offset + CHUNK).min(data.len());
            queue.write_buffer(buf, offset as u64, &data[offset..end]);
            offset = end;
        }
    }

    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        dataset: &Dataset,
        dataset_id: u64,
        from_space: u32,
        to_space: u32,
        colors_from_id: u64,
        colors_from_rgba8: &[u32],
        colors_to_id: u64,
        colors_to_rgba8: &[u32],
        indices_id: u64,
        draw_indices: &[u32],
        from_override: Option<&[f32]>,
        to_override: Option<&[f32]>,
        from_override_id: u64,
        to_override_id: u64,
        uniforms: Uniforms,
    ) -> anyhow::Result<()> {
        // If pipeline must be recreated (format change), rebuild everything.
        if self.target_format != format {
            *self = Self::new(device, format);
        }

        // Draw nothing until everything below has succeeded, so a failed
        // prepare never paints with buffers that don't match the dataset.
        self.n_draw = 0;

        let n_points = dataset.meta.n_points;
        let indices_changed =
            self.last_dataset_id != dataset_id || self.last_indices_id != indices_id;
        if indices_changed {
            if let Some(bad) = draw_indices.iter().copied().find(|&idx| idx >= n_points) {
                return Err(anyhow!(
                    "draw index out of range: {} (n_points={})",
                    bad,
                    n_points
                ));
            }
        }

        // Grow buffers
        let pos_bytes_need = (n_points as u64) * 2 * 4;
        let col_bytes_need = (n_points as u64) * 4;
        let idx_bytes_need = (draw_indices.len() as u64) * 4;
        check_gpu_capacity(&device.limits(), n_points as u64, pos_bytes_need)?;
        self.n_points = n_points;

        Self::ensure_storage_buffer(device, &mut self.pos_from, "pc_pos_from", pos_bytes_need);
        Self::ensure_storage_buffer(device, &mut self.pos_to, "pc_pos_to", pos_bytes_need);
        Self::ensure_storage_buffer(
            device,
            &mut self.colors_from,
            "pc_colors_from",
            col_bytes_need,
        );
        Self::ensure_storage_buffer(device, &mut self.colors_to, "pc_colors_to", col_bytes_need);
        Self::ensure_storage_buffer(
            device,
            &mut self.draw_indices,
            "pc_draw_indices",
            idx_bytes_need,
        );

        // Rebuild bind group if buffers were replaced
        let bgl = self.pipeline_alpha.get_bind_group_layout(0);
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pc_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.pos_from.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.pos_to.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.colors_from.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.colors_to.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.draw_indices.as_entire_binding(),
                },
            ],
        });

        // Upload spaces when changed
        if self.last_dataset_id != dataset_id
            || self.last_from_space != from_space
            || self.last_from_override_id != from_override_id
        {
            let bytes = if let Some(override_pos) = from_override {
                if override_pos.len() == n_points as usize * 2 {
                    bytemuck::cast_slice(override_pos)
                } else {
                    let a = dataset.space_f32_2d(from_space as usize)?;
                    bytemuck::cast_slice(a)
                }
            } else {
                let a = dataset.space_f32_2d(from_space as usize)?;
                bytemuck::cast_slice(a)
            };
            Self::write_chunked(queue, &self.pos_from, bytes);
            self.last_from_space = from_space;
            self.last_from_override_id = from_override_id;
        }
        if self.last_dataset_id != dataset_id
            || self.last_to_space != to_space
            || self.last_to_override_id != to_override_id
        {
            let bytes = if let Some(override_pos) = to_override {
                if override_pos.len() == n_points as usize * 2 {
                    bytemuck::cast_slice(override_pos)
                } else {
                    let b = dataset.space_f32_2d(to_space as usize)?;
                    bytemuck::cast_slice(b)
                }
            } else {
                let b = dataset.space_f32_2d(to_space as usize)?;
                bytemuck::cast_slice(b)
            };
            Self::write_chunked(queue, &self.pos_to, bytes);
            self.last_to_space = to_space;
            self.last_to_override_id = to_override_id;
        }

        // Upload colors when changed
        if self.last_dataset_id != dataset_id || self.last_colors_from_id != colors_from_id {
            if colors_from_rgba8.len() != n_points as usize {
                return Err(anyhow!("colors_from len != n_points"));
            }
            let bytes = bytemuck::cast_slice(colors_from_rgba8);
            Self::write_chunked(queue, &self.colors_from, bytes);
            self.last_colors_from_id = colors_from_id;
        }
        if self.last_dataset_id != dataset_id || self.last_colors_to_id != colors_to_id {
            if colors_to_rgba8.len() != n_points as usize {
                return Err(anyhow!("colors_to len != n_points"));
            }
            let bytes = bytemuck::cast_slice(colors_to_rgba8);
            Self::write_chunked(queue, &self.colors_to, bytes);
            self.last_colors_to_id = colors_to_id;
        }

        // Upload indices when changed
        if indices_changed {
            let bytes = bytemuck::cast_slice(draw_indices);
            Self::write_chunked(queue, &self.draw_indices, bytes);
            self.last_indices_id = indices_id;
        }

        // Uniforms every frame
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        self.last_dataset_id = dataset_id;
        self.n_draw = draw_indices.len() as u32;
        Ok(())
    }

    pub fn paint<'rp>(&self, render_pass: &mut wgpu::RenderPass<'rp>, opaque: bool) {
        if self.n_draw == 0 {
            return;
        }
        if opaque {
            render_pass.set_pipeline(&self.pipeline_opaque);
        } else {
            render_pass.set_pipeline(&self.pipeline_alpha);
        }
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.corners.slice(..));
        render_pass.draw(0..6, 0..self.n_draw);
    }
}

/// The largest per-point GPU buffer holds xy positions (8 bytes per cell). If
/// the device can't bind or allocate one that big, wgpu would raise a
/// validation error and abort the app, so refuse up front with a message that
/// says how far over the limit the dataset is.
pub fn check_gpu_capacity(
    limits: &wgpu::Limits,
    n_points: u64,
    pos_bytes_need: u64,
) -> anyhow::Result<()> {
    let max_bytes = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
    if pos_bytes_need <= max_bytes {
        return Ok(());
    }
    let max_cells = max_bytes / 8;
    Err(anyhow!(
        "This dataset has {} cells, but this GPU can only draw up to {} cells \
         (largest buffer it allows is {} MiB). Subset the dataset before converting it, \
         or try a GPU with more memory.",
        group_thousands(n_points),
        group_thousands(max_cells),
        max_bytes / (1024 * 1024)
    ))
}

fn group_thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

pub struct RenderParams {
    pub dataset: Option<Arc<Dataset>>,
    pub dataset_id: u64,

    pub target_format: wgpu::TextureFormat,

    pub from_space: u32,
    pub to_space: u32,

    pub colors_id: u64,
    pub colors_rgba8: Arc<Vec<u32>>,
    pub colors_to_id: u64,
    pub colors_to_rgba8: Arc<Vec<u32>>,

    pub indices_id: u64,
    pub draw_indices: Arc<Vec<u32>>,

    pub uniforms: Uniforms,
    pub use_opaque: bool,
    pub from_override: Option<Arc<Vec<f32>>>,
    pub to_override: Option<Arc<Vec<f32>>>,
    pub from_override_id: u64,
    pub to_override_id: u64,
}

pub struct SharedRender {
    pub params: Mutex<RenderParams>,
    pub last_prepare_error: Mutex<Option<String>>,
}

impl SharedRender {
    pub fn new(target_format: wgpu::TextureFormat) -> Self {
        Self {
            params: Mutex::new(RenderParams {
                dataset: None,
                dataset_id: 0,
                target_format,
                from_space: 0,
                to_space: 0,
                colors_id: 0,
                colors_rgba8: Arc::new(Vec::new()),
                colors_to_id: 0,
                colors_to_rgba8: Arc::new(Vec::new()),
                indices_id: 0,
                draw_indices: Arc::new(Vec::new()),
                uniforms: Uniforms::zeroed(),
                use_opaque: false,
                from_override: None,
                to_override: None,
                from_override_id: 0,
                to_override_id: 0,
            }),
            last_prepare_error: Mutex::new(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_limits_stop_at_2_pow_24_cells() {
        let limits = wgpu::Limits::default();
        let ok = 1u64 << 24;
        assert!(check_gpu_capacity(&limits, ok, ok * 8).is_ok());
        let over = ok + 1;
        let err = check_gpu_capacity(&limits, over, over * 8)
            .unwrap_err()
            .to_string();
        assert!(err.contains("16,777,217 cells"), "{err}");
        assert!(err.contains("up to 16,777,216 cells"), "{err}");
    }

    #[test]
    fn raised_limits_allow_large_datasets() {
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: i32::MAX as u32,
            max_buffer_size: 1 << 40,
            ..wgpu::Limits::default()
        };
        let n = 50_000_000u64;
        assert!(check_gpu_capacity(&limits, n, n * 8).is_ok());
    }

    #[test]
    fn thousands_grouping() {
        assert_eq!(group_thousands(0), "0");
        assert_eq!(group_thousands(999), "999");
        assert_eq!(group_thousands(1_000), "1,000");
        assert_eq!(group_thousands(16_777_216), "16,777,216");
    }

    /// Renders a dataset just over 2^24 cells on a real GPU device, once with
    /// wgpu's default limits (must fail with a message, not abort) and once
    /// with the adapter's limits as `main.rs` requests them (must draw).
    /// Needs a GPU or a software Vulkan driver (e.g. Mesa lavapipe) and ~3 GB RAM:
    /// `cargo test --release large_dataset_on_gpu -- --ignored`
    #[test]
    #[ignore]
    fn large_dataset_on_gpu() {
        let n: u32 = (1 << 24) + 1_000;
        let path = std::env::temp_dir().join(format!("stviz_large_{}.stviz", std::process::id()));
        crate::data::test_support::write_stviz(&path, n, None);
        let ds = Dataset::load(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("no GPU adapter");
        let supported = adapter.limits();

        let run = |limits: wgpu::Limits| -> anyhow::Result<()> {
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    required_limits: limits,
                    ..Default::default()
                }))
                .unwrap();
            let format = wgpu::TextureFormat::Rgba8Unorm;
            let target = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 64,
                    height: 64,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = target.create_view(&Default::default());
            device.push_error_scope(wgpu::ErrorFilter::Validation);
            let mut gpu = PointCloudGpu::new(&device, format);
            let colors = vec![0xFFFF_FFFFu32; n as usize];
            let indices: Vec<u32> = (0..n).collect();
            let mut uniforms = Uniforms::zeroed();
            uniforms.viewport_px = [64.0, 64.0];
            uniforms.pixels_per_unit = 1.0;
            uniforms.point_radius_px = 1.0;
            uniforms.from_scale = 1.0;
            uniforms.to_scale = 1.0;
            let res = gpu.prepare(
                &device, &queue, format, &ds, 1, 0, 1, 1, &colors, 1, &colors, 1, &indices, None,
                None, 0, 0, uniforms,
            );
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                gpu.paint(&mut pass, false);
            }
            queue.submit([enc.finish()]);
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            if let Some(e) = pollster::block_on(device.pop_error_scope()) {
                panic!("wgpu validation error: {e}");
            }
            res
        };

        let default_err = run(wgpu::Limits::default()).unwrap_err().to_string();
        assert!(
            default_err.contains("can only draw up to 16,777,216 cells"),
            "{default_err}"
        );

        let raised = wgpu::Limits {
            max_storage_buffer_binding_size: supported.max_storage_buffer_binding_size,
            max_buffer_size: supported.max_buffer_size,
            ..wgpu::Limits::default()
        };
        let fits = (n as u64) * 8
            <= (supported.max_storage_buffer_binding_size as u64).min(supported.max_buffer_size);
        eprintln!(
            "adapter: {:?}, max storage binding {} MiB, dataset fits: {fits}",
            adapter.get_info().name,
            supported.max_storage_buffer_binding_size >> 20
        );
        let res = run(raised);
        if fits {
            res.unwrap();
        } else {
            // Software drivers such as lavapipe really are capped at 128 MiB.
            assert!(res.is_err());
        }
    }
}
