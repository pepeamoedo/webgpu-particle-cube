use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
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
    mouse_active: f32,
    mouse_pos: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(dead_code)]
struct ParticleKey {
    hash: u32,
    index: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(dead_code)]
struct GridParams {
    cell_size: f32,
    grid_size: u32,
    num_particles: u32,
    dummy: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(dead_code)]
struct SortParams {
    stage: u32,
    step: u32,
    pad0: u32,
    pad1: u32,
}

// =====================================================================
// ESTADO DE CÁMARA ORBITAL Y VIEWPORT
// =====================================================================

struct CameraState {
    theta: f32,
    phi: f32,
    radius: f32,
    target_theta: f32,
    target_phi: f32,
    target_radius: f32,
    is_dragging: bool,
    last_mouse_x: f32,
    last_mouse_y: f32,
    mouse_ndc_x: f32,
    mouse_ndc_y: f32,
    mouse_active: bool,
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

fn create_multisampled_framebuffer(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    format: wgpu::TextureFormat,
) -> wgpu::TextureView {
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
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    multisampled_texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn create_hdr_texture(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("HDR Color Texture"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn create_bloom_texture(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration, label: &str) -> wgpu::TextureView {
    let width = (config.width / 4).max(1);
    let height = (config.height / 4).max(1);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
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

    let supports_timestamp = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    let mut required_features = wgpu::Features::empty();
    if supports_timestamp {
        required_features |= wgpu::Features::TIMESTAMP_QUERY;
    }

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Wgpu Device"),
                required_features,
                required_limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .map_err(|e| JsValue::from_str(&format!("Error al solicitar dispositivo: {:?}", e)))?;

    let timestamp_period = if supports_timestamp {
        queue.get_timestamp_period() as f64
    } else {
        1.0
    };

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

    // Bindeo de Eventos Interactivos de la Cámara (Ángulos esféricos e Interacción 3D)
    let camera_state = Rc::new(RefCell::new(CameraState {
        theta: 0.8,
        phi: 1.2,
        radius: 4.5,
        target_theta: 0.8,
        target_phi: 1.2,
        target_radius: 4.5,
        is_dragging: false,
        last_mouse_x: 0.0,
        last_mouse_y: 0.0,
        mouse_ndc_x: 0.0,
        mouse_ndc_y: 0.0,
        mouse_active: false,
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
        let canvas_for_move = canvas.clone();
        let on_mousemove = Closure::wrap(Box::new(move |e: web_sys::MouseEvent| {
            let el: &web_sys::Element = canvas_for_move.as_ref();
            let rect = el.get_bounding_client_rect();
            let client_x = e.client_x() as f64 - rect.left();
            let client_y = e.client_y() as f64 - rect.top();
            let ndc_x = (client_x / rect.width()) * 2.0 - 1.0;
            let ndc_y = 1.0 - (client_y / rect.height()) * 2.0;

            let mut c = cam.borrow_mut();
            c.mouse_ndc_x = ndc_x as f32;
            c.mouse_ndc_y = ndc_y as f32;
            c.mouse_active = true;

            if c.is_dragging {
                let x = e.client_x() as f32;
                let y = e.client_y() as f32;
                let dx = x - c.last_mouse_x;
                let dy = y - c.last_mouse_y;

                c.target_theta += dx * 0.005;
                c.target_phi = (c.target_phi - dy * 0.005).clamp(0.1, std::f32::consts::PI - 0.1);

                c.last_mouse_x = x;
                c.last_mouse_y = y;
            }
        }) as Box<dyn FnMut(_)>);
        canvas.add_event_listener_with_callback("mousemove", on_mousemove.as_ref().unchecked_ref())?;
        on_mousemove.forget();
    }

    {
        let cam = camera_state.clone();
        let on_mouseleave = Closure::wrap(Box::new(move |_: web_sys::MouseEvent| {
            let mut c = cam.borrow_mut();
            c.mouse_active = false;
        }) as Box<dyn FnMut(_)>);
        canvas.add_event_listener_with_callback("mouseleave", on_mouseleave.as_ref().unchecked_ref())?;
        on_mouseleave.forget();
    }

    {
        let cam = camera_state.clone();
        let on_wheel = Closure::wrap(Box::new(move |e: web_sys::WheelEvent| {
            let mut c = cam.borrow_mut();
            c.target_radius = (c.target_radius + e.delta_y() as f32 * 0.004).clamp(1.5, 12.0);
        }) as Box<dyn FnMut(_)>);
        canvas.add_event_listener_with_callback("wheel", on_wheel.as_ref().unchecked_ref())?;
        on_wheel.forget();
    }

    // =====================================================================
    // INICIALIZACIÓN DE DATOS DEL STORAGE BUFFER (65,536 PARTÍCULAS SPH)
    // =====================================================================
    fn lcg_random(state: &mut u32) -> f32 {
        *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (*state as f32) / (u32::MAX as f32)
    }

    let num_particles = 8192;
    let grid_size = 8192;
    let mut initial_particles = Vec::with_capacity(num_particles);
    let mut seed = 123456789u32;
    
    // Inicialización en cuadrícula 3D alineada con pequeño jitter para estabilidad SPH
    // 8192 = 16 × 16 × 32 → distribuimos en un cubo de [-0.8, 0.8]^3
    let gx: usize = 16;
    let gy: usize = 16;
    let gz: usize = 32;
    let extent = 0.82f32; // radio del bloque inicial dentro de la urna
    let jitter = 0.008f32; // pequeño desorden para iniciar el SPH sin singularidades
    
    'outer: for iz in 0..gz {
        for iy in 0..gy {
            for ix in 0..gx {
                if initial_particles.len() >= num_particles { break 'outer; }
                
                let px_base = -extent + (ix as f32 + 0.5) * (2.0 * extent / gx as f32);
                let py_base = -extent + (iy as f32 + 0.5) * (2.0 * extent / gy as f32);
                let pz_base = -extent + (iz as f32 + 0.5) * (2.0 * extent / gz as f32);
                
                let jx = (lcg_random(&mut seed) - 0.5) * jitter;
                let jy = (lcg_random(&mut seed) - 0.5) * jitter;
                let jz = (lcg_random(&mut seed) - 0.5) * jitter;
                
                let px = px_base + jx;
                let py = py_base + jy;
                let pz = pz_base + jz;
                
                initial_particles.push(ParticleStruct {
                    pos: [px, py, pz, 1.0], // w: densidad
                    vel: [0.0, 0.0, 0.0, 0.0], // reposo inicial — la gravedad y SPH lo ponen en movimiento
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

    // =====================================================================
    // CREACIÓN DE BUFFERS AUXILIARES PARA SPATIAL HASHING & SORT
    // =====================================================================
    
    // Keys buffer for bitonic sort: u32 hash + u32 index = 8 bytes per particle
    let keys_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Particle Keys Buffer"),
        size: (num_particles * std::mem::size_of::<ParticleKey>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Cell starts buffer (grid_size * 4 bytes)
    let cell_starts_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Cell Starts Buffer"),
        size: (grid_size * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Cell ends buffer (grid_size * 4 bytes)
    let cell_ends_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Cell Ends Buffer"),
        size: (grid_size * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Uniform buffer for GridParams
    let grid_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Grid Params Buffer"),
        size: std::mem::size_of::<GridParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let initial_grid_params = GridParams {
        cell_size: 0.18, // radius interaction H
        grid_size: grid_size as u32,
        num_particles: num_particles as u32,
        dummy: 0,
    };
    unsafe {
        queue.write_buffer(&grid_params_buffer, 0, any_as_u8_slice(&initial_grid_params));
    }

    // Uniform buffer for SortParams (with pre-allocated deterministic dynamic offset entries)
    let mut sort_params_data = Vec::new();
    let mut stage = 2u32;
    let mut step_offsets = Vec::new();
    while stage <= num_particles as u32 {
        let mut step = stage / 2;
        while step > 0 {
            step_offsets.push(sort_params_data.len() as u32);
            // Each entry is SortParams aligned to 256 bytes (which is 64 u32s / 256 bytes)
            let mut entry = vec![0u32; 64];
            entry[0] = stage;
            entry[1] = step;
            sort_params_data.extend_from_slice(&entry);
            step /= 2;
        }
        stage *= 2;
    }

    let sort_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sort Params Buffer"),
        size: (sort_params_data.len() * 4) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    unsafe {
        queue.write_buffer(&sort_params_buffer, 0, slice_as_u8_slice(&sort_params_data));
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
        mouse_active: 0.0,
        mouse_pos: [0.0, 0.0, 0.0, 0.0],
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
    // =====================================================================
    // CREACIÓN DEL BIND GROUP LAYOUT Y BIND GROUP (COMPUTE)
    // =====================================================================
    let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Compute Bind Group Layout"),
        entries: &[
            // 0: particles (storage read_write)
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
            // 1: keys (storage read_write)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 2: cell_starts (storage read_write)
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 3: cell_ends (storage read_write)
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 4: params (uniform)
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 5: grid_params (uniform)
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 6: sort_params (uniform, has_dynamic_offset: true)
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
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
                resource: keys_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: cell_starts_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: cell_ends_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: compute_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: grid_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &sort_params_buffer,
                    offset: 0,
                    size: Some(std::num::NonZeroU64::new(256).unwrap()),
                }),
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
    let hash_gen_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Hash Gen Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &compute_shader,
        entry_point: "hash_gen",
    });

    let bitonic_sort_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Bitonic Sort Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &compute_shader,
        entry_point: "bitonic_sort",
    });

    let cell_offsets_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Cell Offsets Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &compute_shader,
        entry_point: "cell_offsets",
    });

    let sph_density_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("SPH Density Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &compute_shader,
        entry_point: "sph_density",
    });

    let sph_force_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("SPH Force Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &compute_shader,
        entry_point: "sph_force",
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
                format: wgpu::TextureFormat::Rgba16Float,
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
                format: wgpu::TextureFormat::Rgba16Float,
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
                format: wgpu::TextureFormat::Rgba16Float,
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

    // =====================================================================
    // CONFIGURACIÓN DE POST-PROCESAMIENTO: HDR BLOOM Y BLUR GAUSSIANO
    // =====================================================================
    let postprocess_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Postprocess Shader Module"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/postprocess.wgsl").into()),
    });

    let post_process_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Post Process Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    struct BlurParams {
        texel_size: [f32; 2],
        dummy: [f32; 2],
    }

    let blur_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Blur Params Buffer"),
        size: std::mem::size_of::<BlurParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let texture_sampler_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Texture Sampler Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let blur_params_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Blur Params Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let composite_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Composite Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let bright_extract_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Bright Extract Pipeline Layout"),
        bind_group_layouts: &[&texture_sampler_bind_group_layout],
        push_constant_ranges: &[],
    });

    let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Blur Pipeline Layout"),
        bind_group_layouts: &[&texture_sampler_bind_group_layout, &blur_params_bind_group_layout],
        push_constant_ranges: &[],
    });

    let post_process_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Post Process Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout, &composite_bind_group_layout],
        push_constant_ranges: &[],
    });

    let bright_extract_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Bright Extract Pipeline"),
        layout: Some(&bright_extract_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &postprocess_shader,
            entry_point: "vs_post_process",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &postprocess_shader,
            entry_point: "fs_bright_extract",
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let blur_h_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Blur H Pipeline"),
        layout: Some(&blur_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &postprocess_shader,
            entry_point: "vs_post_process",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &postprocess_shader,
            entry_point: "fs_blur_h",
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let blur_v_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Blur V Pipeline"),
        layout: Some(&blur_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &postprocess_shader,
            entry_point: "vs_post_process",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &postprocess_shader,
            entry_point: "fs_blur_v",
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let post_process_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Post Process Pipeline"),
        layout: Some(&post_process_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &postprocess_shader,
            entry_point: "vs_post_process",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &postprocess_shader,
            entry_point: "fs_composite",
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let blur_params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Blur Params Bind Group"),
        layout: &blur_params_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: blur_params_buffer.as_entire_binding(),
            },
        ],
    });

    // Escribir parámetros de desenfoque iniciales
    {
        let low_res_w = (physical_width / 4).max(1) as f32;
        let low_res_h = (physical_height / 4).max(1) as f32;
        let blur_params = BlurParams {
            texel_size: [1.0 / low_res_w, 1.0 / low_res_h],
            dummy: [0.0, 0.0],
        };
        unsafe {
            queue.write_buffer(&blur_params_buffer, 0, any_as_u8_slice(&blur_params));
        }
    }

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

    // Inicializar buffers multisampled framebuffer, depth texture y texturas de post-procesamiento
    let mut msaa_texture_view = create_multisampled_framebuffer(&device, &config, wgpu::TextureFormat::Rgba16Float);
    let mut depth_texture_view = create_depth_texture(&device, &config);
    let mut hdr_texture_view = create_hdr_texture(&device, &config);
    let mut brights_texture_view = create_bloom_texture(&device, &config, "Brights Texture");
    let mut blur_temp_texture_view = create_bloom_texture(&device, &config, "Blur Temp Texture");

    // Inicializar QuerySet y buffers de perfilado si está soportado
    let query_set = if supports_timestamp {
        Some(device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("GPU Timing Query Set"),
            ty: wgpu::QueryType::Timestamp,
            count: 6,
        }))
    } else {
        None
    };

    let query_buffer = if supports_timestamp {
        Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Timing Query Buffer"),
            size: 48, // 6 queries * 8 bytes
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    } else {
        None
    };

    let query_readback_buffer = if supports_timestamp {
        Some(Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Timing Query Readback Buffer"),
            size: 48,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })))
    } else {
        None
    };

    // Crear bind groups iniciales
    let mut extract_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Extract Bind Group"),
        layout: &texture_sampler_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&hdr_texture_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&post_process_sampler) },
        ],
    });

    let mut blur_h_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Blur H Bind Group"),
        layout: &texture_sampler_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&brights_texture_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&post_process_sampler) },
        ],
    });

    let mut blur_v_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Blur V Bind Group"),
        layout: &texture_sampler_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&blur_temp_texture_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&post_process_sampler) },
        ],
    });

    let mut composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Composite Bind Group"),
        layout: &composite_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&hdr_texture_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&brights_texture_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&post_process_sampler) },
        ],
    });

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
    let hash_gen_pipeline = Rc::new(hash_gen_pipeline);
    let bitonic_sort_pipeline = Rc::new(bitonic_sort_pipeline);
    let cell_offsets_pipeline = Rc::new(cell_offsets_pipeline);
    let sph_density_pipeline = Rc::new(sph_density_pipeline);
    let sph_force_pipeline = Rc::new(sph_force_pipeline);
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

    // Clones de recursos de post-procesamiento para el bucle de renderizado
    let bright_extract_pipeline_clone = Rc::new(bright_extract_pipeline);
    let blur_h_pipeline_clone = Rc::new(blur_h_pipeline);
    let blur_v_pipeline_clone = Rc::new(blur_v_pipeline);
    let post_process_pipeline_clone = Rc::new(post_process_pipeline);
    let blur_params_bind_group_clone = Rc::new(blur_params_bind_group);
    
    let texture_sampler_bind_group_layout_clone = Rc::new(texture_sampler_bind_group_layout);
    let composite_bind_group_layout_clone = Rc::new(composite_bind_group_layout);
    let post_process_sampler_clone = Rc::new(post_process_sampler);
    let blur_params_buffer_clone = Rc::new(blur_params_buffer);

    let _observer_keep_alive = observer; // Mantener vivo el ResizeObserver moviéndolo al contexto

    let mut last_size_val = -1.0f32;
    let mut last_spacing_val = -1.0f32;
    let mut last_light_val = -1.0f32;

    let query_pending = Arc::new(AtomicBool::new(false));

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
                msaa_texture_view = create_multisampled_framebuffer(&device, &config, wgpu::TextureFormat::Rgba16Float);
                depth_texture_view = create_depth_texture(&device, &config);
                hdr_texture_view = create_hdr_texture(&device, &config);
                brights_texture_view = create_bloom_texture(&device, &config, "Brights Texture");
                blur_temp_texture_view = create_bloom_texture(&device, &config, "Blur Temp Texture");

                // Escribir los nuevos parámetros de desenfoque
                let low_res_w = (physical_width / 4).max(1) as f32;
                let low_res_h = (physical_height / 4).max(1) as f32;
                let blur_params = BlurParams {
                    texel_size: [1.0 / low_res_w, 1.0 / low_res_h],
                    dummy: [0.0, 0.0],
                };
                unsafe {
                    queue.write_buffer(&blur_params_buffer_clone, 0, any_as_u8_slice(&blur_params));
                }

                // Recrear bind groups con las nuevas vistas de texturas
                extract_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Extract Bind Group"),
                    layout: &texture_sampler_bind_group_layout_clone,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&hdr_texture_view) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&post_process_sampler_clone) },
                    ],
                });

                blur_h_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Blur H Bind Group"),
                    layout: &texture_sampler_bind_group_layout_clone,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&brights_texture_view) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&post_process_sampler_clone) },
                    ],
                });

                blur_v_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Blur V Bind Group"),
                    layout: &texture_sampler_bind_group_layout_clone,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&blur_temp_texture_view) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&post_process_sampler_clone) },
                    ],
                });

                composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Composite Bind Group"),
                    layout: &composite_bind_group_layout_clone,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&hdr_texture_view) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&brights_texture_view) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&post_process_sampler_clone) },
                    ],
                });
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

        // B. CALCULAR PROCESO DE CÁMARA E INTERACCIÓN (SIMD Glam Math con Inercia)
        let (eye, view_proj, mouse_active_val, mouse_pos_val) = {
            let mut state = camera_state.borrow_mut();
            
            // Interpolación exponencial amortiguada independiente de la tasa de refresco
            let factor = 1.0 - (-12.0 * dt).exp();
            state.theta += (state.target_theta - state.theta) * factor;
            state.phi += (state.target_phi - state.phi) * factor;
            state.radius += (state.target_radius - state.radius) * factor;
            
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

            // Proyección del Cursor 3D (Ray casting asíncrono e Intersección con Plano)
            let mut mouse_active_val = 0.0f32;
            let mut mouse_pos_val = [0.0f32; 4];
            
            if state.mouse_active {
                // Si el usuario arrastra (clic sostenido), modo 2.0 (vórtice atractor), de lo contrario modo 1.0 (soplido repulsivo)
                mouse_active_val = if state.is_dragging { 2.0 } else { 1.0 };
                
                let inv_vp = vp.inverse();
                let near_point = inv_vp.project_point3(glam::Vec3::new(state.mouse_ndc_x, state.mouse_ndc_y, 0.0));
                let far_point = inv_vp.project_point3(glam::Vec3::new(state.mouse_ndc_x, state.mouse_ndc_y, 1.0));
                let ray_dir = (far_point - near_point).normalize();
                
                let plane_normal = (target_vec - eye_vec).normalize();
                let denom = ray_dir.dot(plane_normal);
                if denom.abs() > 1e-6 {
                    let t = -eye_vec.dot(plane_normal) / denom;
                    let mouse_3d = eye_vec + ray_dir * t;
                    
                    // Acotar la fuerza dentro de la urna de cristal
                    let limit = spacing_val * 1.06;
                    let mx = mouse_3d.x.clamp(-limit, limit);
                    let my = mouse_3d.y.clamp(-limit, limit);
                    let mz = mouse_3d.z.clamp(-limit, limit);
                    
                    mouse_pos_val = [mx, my, mz, 0.0];
                } else {
                    mouse_active_val = 0.0;
                }
            }
            
            (eye_pos, vp.to_cols_array(), mouse_active_val, mouse_pos_val)
        };

        // Escribir los parámetros de cómputo (físicas dinámicas en GPU)
        let compute_params_data = ComputeParams {
            spacing: spacing_val,
            delta_time: dt,
            intensity: light_val,
            mouse_active: mouse_active_val,
            mouse_pos: mouse_pos_val,
        };
        unsafe {
            queue.write_buffer(&compute_params_buffer, 0, any_as_u8_slice(&compute_params_data));
        }

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
                timestamp_writes: query_set.as_ref().map(|qs| wgpu::ComputePassTimestampWrites {
                    query_set: qs,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
            });

            // 1. Paso 1: Generación de Hash Spatial Hashing
            compute_pass.set_pipeline(&hash_gen_pipeline);
            compute_pass.set_bind_group(0, &compute_bind_group, &[0]);
            compute_pass.dispatch_workgroups((num_particles as u32) / 256, 1, 1);

            // 2. Paso 2: Ordenamiento masivo en GPU (Bitonic Merge Sort)
            compute_pass.set_pipeline(&bitonic_sort_pipeline);
            let mut step_idx = 0;
            let mut stage = 2u32;
            while stage <= num_particles as u32 {
                let mut step = stage / 2;
                while step > 0 {
                    let offset = step_offsets[step_idx] * 4; // offset en bytes
                    compute_pass.set_bind_group(0, &compute_bind_group, &[offset]);
                    compute_pass.dispatch_workgroups((num_particles as u32) / 256, 1, 1);
                    step_idx += 1;
                    step /= 2;
                }
                stage *= 2;
            }

            // 3. Paso 3: offsets de celdas
            compute_pass.set_pipeline(&cell_offsets_pipeline);
            compute_pass.set_bind_group(0, &compute_bind_group, &[0]);
            compute_pass.dispatch_workgroups((num_particles as u32) / 256, 1, 1);

            // 4. Paso 4: densidad y presión de fluido SPH
            compute_pass.set_pipeline(&sph_density_pipeline);
            compute_pass.set_bind_group(0, &compute_bind_group, &[0]);
            compute_pass.dispatch_workgroups((num_particles as u32) / 256, 1, 1);

            // 5. Paso 5: integración de fuerzas SPH (Presión + Viscosidad + Remolino)
            compute_pass.set_pipeline(&sph_force_pipeline);
            compute_pass.set_bind_group(0, &compute_bind_group, &[0]);
            compute_pass.dispatch_workgroups((num_particles as u32) / 256, 1, 1);
        }

        // =====================================================================
        // RENDER PASS: DIBUJADO DE ESCENA CON MSAA 4x E ILUMINACIÓN
        // =====================================================================
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Interactive Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &msaa_texture_view,
                    resolve_target: Some(&hdr_texture_view),
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
                timestamp_writes: query_set.as_ref().map(|qs| wgpu::RenderPassTimestampWrites {
                    query_set: qs,
                    beginning_of_pass_write_index: Some(2),
                    end_of_pass_write_index: Some(3),
                }),
            });

            // 1. Líneas de fuerza / velocidad: 2 vértices por partícula (cola + cabeza del vector)
            render_pass.set_pipeline(&line_pipeline_clone);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.set_bind_group(1, &bind_group_1, &[]);
            render_pass.draw(0..((num_particles as u32) * 2), 0..1);

            // 2. Dibujamos los puntos nítidos de las partículas (65,536 partículas * 6 vértices por billboard)
            render_pass.set_pipeline(&render_pipeline);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.set_bind_group(1, &bind_group_1, &[]);
            render_pass.draw(0..((num_particles as u32) * 6), 0..1);

            // 3. Dibujamos el cristal encasillador exterior (36 vértices procedimentales)
            render_pass.set_pipeline(&glass_pipeline_clone);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.draw(0..36, 0..1);
        }

        // =====================================================================
        // PASS 2: EXTRACCIÓN DE BRILLOS (THRESHOLD) Y DOWNSAMPLING A 1/4
        // =====================================================================
        {
            let mut extract_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bright Extract Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &brights_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: query_set.as_ref().map(|qs| wgpu::RenderPassTimestampWrites {
                    query_set: qs,
                    beginning_of_pass_write_index: Some(4),
                    end_of_pass_write_index: None,
                }),
            });
            extract_pass.set_pipeline(&bright_extract_pipeline_clone);
            extract_pass.set_bind_group(0, &extract_bind_group, &[]);
            extract_pass.draw(0..3, 0..1);
        }

        // =====================================================================
        // PASS 3: BLUR HORIZONTAL EN 1/4 RESOLUCIÓN
        // =====================================================================
        {
            let mut blur_h_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Blur Horizontal Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &blur_temp_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            blur_h_pass.set_pipeline(&blur_h_pipeline_clone);
            blur_h_pass.set_bind_group(0, &blur_h_bind_group, &[]);
            blur_h_pass.set_bind_group(1, &blur_params_bind_group_clone, &[]);
            blur_h_pass.draw(0..3, 0..1);
        }

        // =====================================================================
        // PASS 4: BLUR VERTICAL EN 1/4 RESOLUCIÓN
        // =====================================================================
        {
            let mut blur_v_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Blur Vertical Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &brights_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            blur_v_pass.set_pipeline(&blur_v_pipeline_clone);
            blur_v_pass.set_bind_group(0, &blur_v_bind_group, &[]);
            blur_v_pass.set_bind_group(1, &blur_params_bind_group_clone, &[]);
            blur_v_pass.draw(0..3, 0..1);
        }

        // =====================================================================
        // PASS 5: COMPOSICIÓN DE BLOOM, ACES TONEMAPPING Y CORRECCIÓN GAMMA
        // =====================================================================
        {
            let mut compose_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Post Process Compose Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
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
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: query_set.as_ref().map(|qs| wgpu::RenderPassTimestampWrites {
                    query_set: qs,
                    beginning_of_pass_write_index: None,
                    end_of_pass_write_index: Some(5),
                }),
            });
            compose_pass.set_pipeline(&post_process_pipeline_clone);
            compose_pass.set_bind_group(0, &bind_group, &[]);
            compose_pass.set_bind_group(1, &composite_bind_group, &[]);
            compose_pass.draw(0..3, 0..1);
        }

        queue.submit(std::iter::once(encoder.finish()));

        if supports_timestamp {
            if query_set.is_some() && query_buffer.is_some() && query_readback_buffer.is_some() {
                let query_pending_clone = query_pending.clone();
                if !query_pending_clone.load(Ordering::SeqCst) {
                    query_pending_clone.store(true, Ordering::SeqCst);

                    let qs = query_set.as_ref().unwrap();
                    let qb = query_buffer.as_ref().unwrap();
                    let qrb = query_readback_buffer.as_ref().unwrap().clone();

                    map_and_read_timestamps(qrb.clone(), query_pending_clone, timestamp_period);

                    let mut resolve_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Resolve Query Encoder"),
                    });
                    resolve_encoder.resolve_query_set(qs, 0..6, qb, 0);
                    resolve_encoder.copy_buffer_to_buffer(qb, 0, &qrb, 0, 48);
                    queue.submit(std::iter::once(resolve_encoder.finish()));
                }
            }
        }

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

#[allow(unused_variables)]
fn update_gpu_timing_hud(compute_ms: f64, render_ms: f64, postprocess_ms: f64, total_ms: f64) {
    #[cfg(target_arch = "wasm32")]
    {
        if let Some(window) = web_sys::window() {
            if let Ok(func) = js_sys::Reflect::get(&window, &wasm_bindgen::JsValue::from_str("updateGpuTimingHud")) {
                if !func.is_undefined() && !func.is_null() {
                    if let Some(f) = func.dyn_ref::<js_sys::Function>() {
                        let this_val = wasm_bindgen::JsValue::NULL;
                        let args = js_sys::Array::new();
                        args.push(&wasm_bindgen::JsValue::from_f64(compute_ms));
                        args.push(&wasm_bindgen::JsValue::from_f64(render_ms));
                        args.push(&wasm_bindgen::JsValue::from_f64(postprocess_ms));
                        args.push(&wasm_bindgen::JsValue::from_f64(total_ms));
                        let _ = f.apply(&this_val, &args);
                    }
                }
            }
        }
    }
}

fn map_and_read_timestamps(
    qrb: Arc<wgpu::Buffer>,
    query_pending: Arc<AtomicBool>,
    timestamp_period: f64,
) {
    let qrb_clone = qrb.clone();
    qrb.slice(..).map_async(wgpu::MapMode::Read, move |result| {
        if result.is_ok() {
            let data = qrb_clone.slice(..).get_mapped_range();
            let timestamps = unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const u64,
                    data.len() / 8,
                )
            };

            if timestamps.len() >= 6 {
                let t0 = timestamps[0];
                let t1 = timestamps[1];
                let t2 = timestamps[2];
                let t3 = timestamps[3];
                let t4 = timestamps[4];
                let t5 = timestamps[5];

                let compute_time = if t1 >= t0 { (t1 - t0) as f64 * timestamp_period } else { 0.0 };
                let render_time = if t3 >= t2 { (t3 - t2) as f64 * timestamp_period } else { 0.0 };
                let postprocess_time = if t5 >= t4 { (t5 - t4) as f64 * timestamp_period } else { 0.0 };

                let compute_ms = compute_time / 1_000_000.0;
                let render_ms = render_time / 1_000_000.0;
                let postprocess_ms = postprocess_time / 1_000_000.0;
                let total_ms = compute_ms + render_ms + postprocess_ms;

                update_gpu_timing_hud(compute_ms, render_ms, postprocess_ms, total_ms);
            }
            drop(data);
            qrb_clone.unmap();
        }
        query_pending.store(false, Ordering::SeqCst);
    });
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

    #[test]
    fn test_postprocess_shader() {
        let postprocess = include_str!("shaders/postprocess.wgsl");
        let mut frontend = wgpu::naga::front::wgsl::Frontend::new();
        match frontend.parse(postprocess) {
            Ok(module) => {
                println!("Postprocess Shader parsed successfully!");
                let mut validator = wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::all(),
                    wgpu::naga::valid::Capabilities::all(),
                );
                match validator.validate(&module) {
                    Ok(_) => println!("Postprocess Shader validated successfully!"),
                    Err(e) => {
                        let error_msg = e.emit_to_string(postprocess);
                        panic!("Postprocess Validation Error:\n{}", error_msg);
                    }
                }
            }
            Err(e) => {
                let error_msg = e.emit_to_string(postprocess);
                panic!("Postprocess Parsing Error:\n{}", error_msg);
            }
        }
    }

    #[test]
    fn test_compute_shader() {
        let compute = include_str!("shaders/compute.wgsl");
        let mut frontend = wgpu::naga::front::wgsl::Frontend::new();
        match frontend.parse(compute) {
            Ok(module) => {
                println!("Compute Shader parsed successfully!");
                let mut validator = wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::all(),
                    wgpu::naga::valid::Capabilities::all(),
                );
                match validator.validate(&module) {
                    Ok(_) => println!("Compute Shader validated successfully!"),
                    Err(e) => {
                        let error_msg = e.emit_to_string(compute);
                        panic!("Compute Validation Error:\n{}", error_msg);
                    }
                }
            }
            Err(e) => {
                let error_msg = e.emit_to_string(compute);
                panic!("Compute Parsing Error:\n{}", error_msg);
            }
        }
    }
}
