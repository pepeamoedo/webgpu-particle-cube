use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

// =====================================================================
// ÁLGEBRA SIMD CON GLAM (MIGRADO A PRODUCCIÓN)
// =====================================================================

// =====================================================================
// ESTRUCTURAS DE DATOS DE UNIFORMS (REPR C)
// =====================================================================

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct CameraUniform {
    view_proj: [f32; 16],
    position: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct LightingConfig {
    ambient_color: [f32; 4],
    light_dir: [f32; 4],
    params: [f32; 4], // [size, spacing, intensity, 0.0]
}

// Utilidad unsafe estándar para convertir estructuras repr(C) a bytes y enviarlos a la GPU
unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    std::slice::from_raw_parts(
        (p as *const T) as *const u8,
        std::mem::size_of::<T>(),
    )
}

// Utilidad unsafe para convertir vectores completos a slice de bytes
unsafe fn slice_as_u8_slice<T: Sized>(p: &[T]) -> &[u8] {
    std::slice::from_raw_parts(
        p.as_ptr() as *const u8,
        p.len() * std::mem::size_of::<T>(),
    )
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct ParticleStruct {
    pos: [f32; 4],
    vel: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct ComputeParams {
    spacing: f32,
    delta_time: f32,
    intensity: f32,
    dummy: f32,
}

// =====================================================================
// ESTADO DE CÁMARA ORBITAL Y VIEWPORT
// =====================================================================

struct CameraState {
    theta: f32,
    phi: f32,
    radius: f32,
    is_dragging: bool,
    last_mouse_x: f32,
    last_mouse_y: f32,
}

struct ViewportState {
    logical_width: f64,
    logical_height: f64,
    dirty: bool,
}

fn create_depth_texture(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::TextureView {
    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth Texture"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 4, // MSAA 4x - Debe coincidir con el sample count del framebuffer de color!
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth24Plus,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    depth_texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn create_multisampled_framebuffer(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::TextureView {
    let multisampled_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("MSAA Framebuffer"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 4, // MSAA 4x
        dimension: wgpu::TextureDimension::D2,
        format: config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    multisampled_texture.create_view(&wgpu::TextureViewDescriptor::default())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub async fn start() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Debug);
    
    log::info!("Iniciando Motor de Partículas WebGPU...");

    let window = web_sys::window().ok_or("No existe la ventana global 'window'")?;
    let document = window.document().ok_or("No existe el objeto 'document'")?;
    
    let canvas = document
        .get_element_by_id("canvas")
        .ok_or("No se encontró el canvas con id='canvas'")?
        .dyn_into::<web_sys::HtmlCanvasElement>()?;

    let dpr = window.device_pixel_ratio();
    let logical_width = window.inner_width()?.as_f64().ok_or("Ancho de ventana inválido")?;
    let logical_height = window.inner_height()?.as_f64().ok_or("Alto de ventana inválido")?;

    let mut physical_width = (logical_width * dpr) as u32;
    let mut physical_height = (logical_height * dpr) as u32;

    canvas.set_width(physical_width);
    canvas.set_height(physical_height);

    let canvas_style = canvas.style();
    canvas_style.set_property("width", &format!("{}px", logical_width))?;
    canvas_style.set_property("height", &format!("{}px", logical_height))?;

    let size_slider = document
        .get_element_by_id("size-slider")
        .and_then(|el| el.dyn_into::<web_sys::HtmlInputElement>().ok());

    let spacing_slider = document
        .get_element_by_id("spacing-slider")
        .and_then(|el| el.dyn_into::<web_sys::HtmlInputElement>().ok());

    let light_slider = document
        .get_element_by_id("light-slider")
        .and_then(|el| el.dyn_into::<web_sys::HtmlInputElement>().ok());

    // Inicializar WebGPU
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
        .map_err(|e| JsValue::from_str(&format!("Fallo al crear superficie: {:?}", e)))?;

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .ok_or("No se encontró un adaptador compatible")?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Wgpu Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .map_err(|e| JsValue::from_str(&format!("Error al solicitar dispositivo: {:?}", e)))?;

    // Configurar Superficie
    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(surface_caps.formats[0]);

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: physical_width,
        height: physical_height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);

    // Bindeo de Eventos Interactivos de la Cámara (Ángulos esféricos)
    let camera_state = Rc::new(RefCell::new(CameraState {
        theta: 0.8,
        phi: 1.2,
        radius: 4.5,
        is_dragging: false,
        last_mouse_x: 0.0,
        last_mouse_y: 0.0,
    }));

    {
        let cam = camera_state.clone();
        let on_mousedown = Closure::wrap(Box::new(move |e: web_sys::MouseEvent| {
            let mut c = cam.borrow_mut();
            c.is_dragging = true;
            c.last_mouse_x = e.client_x() as f32;
            c.last_mouse_y = e.client_y() as f32;
        }) as Box<dyn FnMut(_)>);
        canvas.add_event_listener_with_callback("mousedown", on_mousedown.as_ref().unchecked_ref())?;
        on_mousedown.forget();
    }

    {
        let cam = camera_state.clone();
        let on_mouseup = Closure::wrap(Box::new(move |_: web_sys::MouseEvent| {
            let mut c = cam.borrow_mut();
            c.is_dragging = false;
        }) as Box<dyn FnMut(_)>);
        canvas.add_event_listener_with_callback("mouseup", on_mouseup.as_ref().unchecked_ref())?;
        on_mouseup.forget();
    }

    {
        let cam = camera_state.clone();
        let on_mousemove = Closure::wrap(Box::new(move |e: web_sys::MouseEvent| {
            let mut c = cam.borrow_mut();
            if c.is_dragging {
                let x = e.client_x() as f32;
                let y = e.client_y() as f32;
                let dx = x - c.last_mouse_x;
                let dy = y - c.last_mouse_y;

                c.theta += dx * 0.005;
                c.phi = (c.phi - dy * 0.005).clamp(0.1, std::f32::consts::PI - 0.1);

                c.last_mouse_x = x;
                c.last_mouse_y = y;
            }
        }) as Box<dyn FnMut(_)>);
        canvas.add_event_listener_with_callback("mousemove", on_mousemove.as_ref().unchecked_ref())?;
        on_mousemove.forget();
    }

    {
        let cam = camera_state.clone();
        let on_wheel = Closure::wrap(Box::new(move |e: web_sys::WheelEvent| {
            let mut c = cam.borrow_mut();
            c.radius = (c.radius + e.delta_y() as f32 * 0.004).clamp(1.5, 12.0);
        }) as Box<dyn FnMut(_)>);
        canvas.add_event_listener_with_callback("wheel", on_wheel.as_ref().unchecked_ref())?;
        on_wheel.forget();
    }

    // =====================================================================
    // INICIALIZACIÓN DE DATOS DEL STORAGE BUFFER (1,728 PARTÍCULAS)
    // =====================================================================
    let num_particles = 1728;
    let mut initial_particles = Vec::with_capacity(num_particles);
    let spacing_init = 1.0f32;
    
    for iz in 0..12 {
        for iy in 0..12 {
            for ix in 0..12 {
                // Generar coordenadas del cubo de cristal de -1.0 a 1.0 (centrado en el origen)
                let px = (ix as f32 - 5.5) * (spacing_init / 5.5);
                let py = (iy as f32 - 5.5) * (spacing_init / 5.5);
                let pz = (iz as f32 - 5.5) * (spacing_init / 5.5);
                
                // Velocidades orbitales iniciales en el eje Y (tangente)
                let vx = -pz * 0.1;
                let vy = 0.0f32;
                let vz = px * 0.1;
                
                initial_particles.push(ParticleStruct {
                    pos: [px, py, pz, 1.0],
                    vel: [vx, vy, vz, 0.0],
                });
            }
        }
    }

    // Crear Storage Buffer para las posiciones dinámicas de las partículas
    let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Particle Storage Buffer"),
        size: (initial_particles.len() * std::mem::size_of::<ParticleStruct>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    unsafe {
        queue.write_buffer(&storage_buffer, 0, slice_as_u8_slice(&initial_particles));
    }

    // Crear Uniform Buffer para parámetros de Cómputo
    let compute_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Compute Params Buffer"),
        size: std::mem::size_of::<ComputeParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let initial_compute_params = ComputeParams {
        spacing: 1.0,
        delta_time: 0.016,
        intensity: 0.8,
        dummy: 0.0,
    };
    unsafe {
        queue.write_buffer(&compute_params_buffer, 0, any_as_u8_slice(&initial_compute_params));
    }

    // =====================================================================
    // PREPARACIÓN DE BUFFERS DE UNIFORMS (PROCESOS DE CÁMARA Y LUZ)
    // =====================================================================

    // 1. Buffer para el Proceso de Cámara (80 bytes)
    let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Camera Uniform Buffer"),
        size: std::mem::size_of::<CameraUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 2. Buffer para el Proceso de Luz y Material (32 bytes)
    let lighting_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Lighting Uniform Buffer"),
        size: std::mem::size_of::<LightingConfig>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Escribir datos iniciales de luz
    let initial_lighting = LightingConfig {
        ambient_color: [0.15, 0.1, 0.25, 0.8], // Violeta neón ambiental
        light_dir: [1.0, 1.5, 1.0, 0.0],       // Luz desde el cuadrante superior derecho
        params: [0.005, 1.0, 0.8, 0.0],
    };
    unsafe {
        queue.write_buffer(&lighting_buffer, 0, any_as_u8_slice(&initial_lighting));
    }

    // =====================================================================
    // CREACIÓN DEL BIND GROUP LAYOUT Y BIND GROUP (UNIFORMS)
    // =====================================================================
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Uniforms Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Uniforms Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: lighting_buffer.as_entire_binding(),
            },
        ],
    });

    // =====================================================================
    // CREACIÓN DEL BIND GROUP LAYOUT Y BIND GROUP (STORAGE RENDER)
    // =====================================================================
    let bind_group_layout_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Storage Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
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

    let bind_group_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Storage Bind Group"),
        layout: &bind_group_layout_1,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: storage_buffer.as_entire_binding(),
            },
        ],
    });

    // =====================================================================
    // CREACIÓN DEL BIND GROUP LAYOUT Y BIND GROUP (COMPUTE)
    // =====================================================================
    let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Compute Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Compute Bind Group"),
        layout: &compute_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: storage_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: compute_params_buffer.as_entire_binding(),
            },
        ],
    });

    // Cargar Shaders WGSL (Modular shader para rendering)
    let shader_source = concat!(
        include_str!("shaders/common.wgsl"),
        include_str!("shaders/particles.wgsl"),
        include_str!("shaders/lines.wgsl"),
        include_str!("shaders/glass.wgsl")
    );

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Modular Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Cargar Compute Shader WGSL
    let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Compute Shader Module"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/compute.wgsl").into()),
    });

    // Crear Pipeline Layout con ambos Bind Group Layouts (Uniforms en slot 0, Storage en slot 1)
    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Render Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout, &bind_group_layout_1],
        push_constant_ranges: &[],
    });

    // Crear Glass Pipeline Layout (solo slot 0 con Uniforms, igual al original)
    let glass_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Glass Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // Crear Compute Pipeline Layout
    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[&compute_bind_group_layout],
        push_constant_ranges: &[],
    });

    // =====================================================================
    // COMPILACIÓN DE PIPELINES (SINCRÓNICOS PARA COMPATIBILIDAD CON WGPU 0.19)
    // =====================================================================
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &compute_shader,
        entry_point: "main",
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24Plus,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 4, // MSAA 4x
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    });

    let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Line Pipeline"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_line",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_line",
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24Plus,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 4, // MSAA 4x
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    });

    let glass_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Glass Pipeline"),
        layout: Some(&glass_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_glass",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_glass",
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24Plus,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 4, // MSAA 4x
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    });

    log::info!("Pipelines, Storage y Uniforms inicializados. Lanzando bucle interactivo.");

    // =====================================================================
    // NATIVE RESIZEOBSERVER PARA ELIMINAR EL LAYOUT THRASHING
    // =====================================================================
    let viewport_state = Rc::new(RefCell::new(ViewportState {
        logical_width,
        logical_height,
        dirty: true, // Forzar creación inicial de texturas MSAA y profundidad
    }));

    let observer = {
        let state = viewport_state.clone();
        let on_resize = Closure::wrap(Box::new(move |entries: js_sys::Array| {
            if entries.length() > 0 {
                if let Ok(entry) = entries.get(0).dyn_into::<web_sys::ResizeObserverEntry>() {
                    let rect = entry.content_rect();
                    let mut s = state.borrow_mut();
                    s.logical_width = rect.width();
                    s.logical_height = rect.height();
                    s.dirty = true;
                }
            }
        }) as Box<dyn FnMut(js_sys::Array)>);

        let obs = web_sys::ResizeObserver::new(on_resize.as_ref().unchecked_ref())
            .expect("No se pudo crear el ResizeObserver");
        obs.observe(&canvas);
        on_resize.forget();
        obs
    };

    // Inicializar buffers multisampled framebuffer y depth texture
    let mut msaa_texture_view = create_multisampled_framebuffer(&device, &config);
    let mut depth_texture_view = create_depth_texture(&device, &config);

    // Reloj de alta precisión para medir delta_time estable
    let performance = window.performance().ok_or("No existe performance object")?;
    let mut last_frame_time = performance.now();

    // Bucle de Renderizado
    let f = Rc::new(RefCell::new(None));
    let g = f.clone();

    let device = Rc::new(device);
    let queue = Rc::new(queue);
    let surface = Rc::new(surface);
    let render_pipeline = Rc::new(render_pipeline);
    let line_pipeline = Rc::new(line_pipeline);
    let glass_pipeline = Rc::new(glass_pipeline);
    let compute_pipeline = Rc::new(compute_pipeline);
    let bind_group = Rc::new(bind_group);
    let bind_group_1 = Rc::new(bind_group_1);
    let compute_bind_group = Rc::new(compute_bind_group);
    let camera_buffer = Rc::new(camera_buffer);
    let lighting_buffer = Rc::new(lighting_buffer);
    let compute_params_buffer = Rc::new(compute_params_buffer);
    let viewport_state_clone = viewport_state.clone();
    let performance_clone = performance.clone();

    let canvas_clone = canvas.clone();
    let window_clone = window.clone();
    let size_slider_clone = size_slider.clone();
    let spacing_slider_clone = spacing_slider.clone();
    let light_slider_clone = light_slider.clone();
    let lighting_buffer_clone = lighting_buffer.clone();
    let line_pipeline_clone = line_pipeline.clone();
    let glass_pipeline_clone = glass_pipeline.clone();
    let _observer_keep_alive = observer; // Mantener vivo el ResizeObserver moviéndolo al contexto

    let mut last_size_val = -1.0f32;
    let mut last_spacing_val = -1.0f32;
    let mut last_light_val = -1.0f32;

    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        // A. Resize dinámico libre de Layout Thrashing (ResizeObserver)
        let mut resized = false;
        let (w, h) = {
            let mut state = viewport_state_clone.borrow_mut();
            if state.dirty {
                state.dirty = false;
                resized = true;
            }
            (state.logical_width, state.logical_height)
        };

        if resized {
            let dpr = window_clone.device_pixel_ratio();
            let new_physical_width = (w * dpr) as u32;
            let new_physical_height = (h * dpr) as u32;
            
            if new_physical_width > 0 && new_physical_height > 0 && 
               (new_physical_width != physical_width || new_physical_height != physical_height) {
                
                physical_width = new_physical_width;
                physical_height = new_physical_height;
                
                canvas_clone.set_width(physical_width);
                canvas_clone.set_height(physical_height);
                
                config.width = physical_width;
                config.height = physical_height;
                surface.configure(&device, &config);
                
                // Recrear texturas MSAA y profundidad para coincidir con la resolución física exacta
                msaa_texture_view = create_multisampled_framebuffer(&device, &config);
                depth_texture_view = create_depth_texture(&device, &config);
            }
        }

        // A2. ACTUALIZAR PARÁMETROS DEL PANEL DE CONTROL EN TIEMPO REAL
        let size_val = size_slider_clone
            .as_ref()
            .map(|s| s.value().parse::<f32>().unwrap_or(0.005))
            .unwrap_or(0.005);
        let spacing_val = spacing_slider_clone
            .as_ref()
            .map(|s| s.value().parse::<f32>().unwrap_or(1.0))
            .unwrap_or(1.0);
        let light_val = light_slider_clone
            .as_ref()
            .map(|s| s.value().parse::<f32>().unwrap_or(0.8))
            .unwrap_or(0.8);

        if size_val != last_size_val || spacing_val != last_spacing_val || light_val != last_light_val {
            last_size_val = size_val;
            last_spacing_val = spacing_val;
            last_light_val = light_val;

            let lighting_data = LightingConfig {
                ambient_color: [0.15 * light_val, 0.1 * light_val, 0.25 * light_val, light_val],
                light_dir: [1.0, 1.5, 1.0, 0.0],
                params: [size_val, spacing_val, light_val, 0.0],
            };
            unsafe {
                queue.write_buffer(&lighting_buffer_clone, 0, any_as_u8_slice(&lighting_data));
            }
        }

        // Medir delta_time preciso
        let now = performance_clone.now();
        let mut dt = ((now - last_frame_time) / 1000.0) as f32;
        last_frame_time = now;
        if dt > 0.1 {
            dt = 0.016; // Prevenir saltos de físicas al suspender pestaña
        }

        // Escribir los parámetros de cómputo (físicas dinámicas en GPU)
        let compute_params_data = ComputeParams {
            spacing: spacing_val,
            delta_time: dt,
            intensity: light_val,
            dummy: 0.0,
        };
        unsafe {
            queue.write_buffer(&compute_params_buffer, 0, any_as_u8_slice(&compute_params_data));
        }

        // B. CALCULAR PROCESO DE CÁMARA E INTERACCIÓN (SIMD Glam Math)
        let (eye, view_proj) = {
            let state = camera_state.borrow();
            
            // Convertir ángulos esféricos a posición 3D
            let eye_x = state.radius * state.phi.sin() * state.theta.cos();
            let eye_y = state.radius * state.phi.cos();
            let eye_z = state.radius * state.phi.sin() * state.theta.sin();
            let eye_pos = [eye_x, eye_y, eye_z];

            let eye_vec = glam::Vec3::new(eye_x, eye_y, eye_z);
            let target_vec = glam::Vec3::ZERO;
            let up_vec = glam::Vec3::Y;
            
            let view = glam::Mat4::look_at_rh(eye_vec, target_vec, up_vec);
            let aspect = if physical_height > 0 {
                physical_width as f32 / physical_height as f32
            } else {
                1.0
            };
            let proj = glam::Mat4::perspective_rh(45.0f32.to_radians(), aspect, 0.1, 100.0);
            
            let vp = proj * view;
            
            (eye_pos, vp.to_cols_array())
        };

        // C. SUBIR DATOS DE CÁMARA A LA GPU
        let camera_data = CameraUniform {
            view_proj,
            position: [eye[0], eye[1], eye[2], 1.0],
        };
        unsafe {
            queue.write_buffer(&camera_buffer, 0, any_as_u8_slice(&camera_data));
        }

        // D. Obtener fotograma de superficie
        let frame = match surface.get_current_texture() {
            Ok(f) => f,
            Err(e) => {
                log::error!("Fallo de superficie: {:?}", e);
                return;
            }
        };

        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Frame Encoder"),
        });

        // =====================================================================
        // COMPUTE PASS: SIMULACIÓN DE FÍSICAS PARALELAS EN GPU
        // =====================================================================
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Physics Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &compute_bind_group, &[]);
            compute_pass.dispatch_workgroups(7, 1, 1); // 1,728 partículas / 256 workgroup_size = 6.75 -> 7 grupos
        }

        // =====================================================================
        // RENDER PASS: DIBUJADO DE ESCENA CON MSAA 4x E ILUMINACIÓN
        // =====================================================================
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Interactive Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &msaa_texture_view,
                    resolve_target: Some(&view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.015,
                            g: 0.012,
                            b: 0.02,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            // 1. Dibujamos el lattice de líneas conectadas en 3D (9504 vértices)
            render_pass.set_pipeline(&line_pipeline_clone);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.set_bind_group(1, &bind_group_1, &[]);
            render_pass.draw(0..9504, 0..1);

            // 2. Dibujamos los puntos nítidos de las partículas (10368 vértices)
            render_pass.set_pipeline(&render_pipeline);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.set_bind_group(1, &bind_group_1, &[]);
            render_pass.draw(0..10368, 0..1);

            // 3. Dibujamos el cristal encasillador exterior (36 vértices procedimentales)
            render_pass.set_pipeline(&glass_pipeline_clone);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.draw(0..36, 0..1);
        }

        queue.submit(std::iter::once(encoder.finish()));
        frame.present();

        request_animation_frame(f.borrow().as_ref().unwrap());
    }) as Box<dyn FnMut()>));

    request_animation_frame(g.borrow().as_ref().unwrap());
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window()
        .unwrap()
        .request_animation_frame(f.as_ref().unchecked_ref())
        .unwrap();
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_modular_shader() {
        let common = include_str!("shaders/common.wgsl");
        let particles = include_str!("shaders/particles.wgsl");
        let lines = include_str!("shaders/lines.wgsl");
        let glass = include_str!("shaders/glass.wgsl");
        
        let shader_source = format!("{}{}{}{}", common, particles, lines, glass);
        
        let mut frontend = wgpu::naga::front::wgsl::Frontend::new();
        match frontend.parse(&shader_source) {
            Ok(module) => {
                println!("Shader parsed successfully!");
                let mut validator = wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::all(),
                    wgpu::naga::valid::Capabilities::all(),
                );
                match validator.validate(&module) {
                    Ok(_) => println!("Shader validated successfully!"),
                    Err(e) => {
                        let error_msg = e.emit_to_string(&shader_source);
                        panic!("Validation Error:\n{}", error_msg);
                    }
                }
            }
            Err(e) => {
                let error_msg = e.emit_to_string(&shader_source);
                panic!("Parsing Error:\n{}", error_msg);
            }
        }
    }
}
