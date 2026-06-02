use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

// =====================================================================
// ÁLGEBRA 3D MINIMALISTA (DIRECTO AL METAL)
// =====================================================================

fn perspective(aspect: f32, fovy: f32, znear: f32, zfar: f32) -> [f32; 16] {
    let f = 1.0 / (fovy / 2.0).tan();
    let r_depth = 1.0 / (znear - zfar);
    [
        f / aspect, 0.0, 0.0, 0.0,
        0.0, f, 0.0, 0.0,
        0.0, 0.0, zfar * r_depth, -1.0,
        0.0, 0.0, znear * zfar * r_depth, 0.0,
    ]
}

fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [f32; 16] {
    let f = {
        let mut v = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]];
        let len = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
        v[0] /= len; v[1] /= len; v[2] /= len;
        v
    };
    let s = {
        let mut v = [
            f[1] * up[2] - f[2] * up[1],
            f[2] * up[0] - f[0] * up[2],
            f[0] * up[1] - f[1] * up[0],
        ];
        let len = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
        v[0] /= len; v[1] /= len; v[2] /= len;
        v
    };
    let u = [
        s[1] * f[2] - s[2] * f[1],
        s[2] * f[0] - s[0] * f[2],
        s[0] * f[1] - s[1] * f[0],
    ];
    [
        s[0], u[0], -f[0], 0.0,
        s[1], u[1], -f[1], 0.0,
        s[2], u[2], -f[2], 0.0,
        -(s[0]*eye[0] + s[1]*eye[1] + s[2]*eye[2]),
        -(u[0]*eye[0] + u[1]*eye[1] + u[2]*eye[2]),
        (f[0]*eye[0] + f[1]*eye[1] + f[2]*eye[2]),
        1.0,
    ]
}

fn mat4_mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0; 16];
    for col in 0..4 {
        for row in 0..4 {
            out[col * 4 + row] = 
                a[0 * 4 + row] * b[col * 4 + 0] +
                a[1 * 4 + row] * b[col * 4 + 1] +
                a[2 * 4 + row] * b[col * 4 + 2] +
                a[3 * 4 + row] * b[col * 4 + 3];
        }
    }
    out
}

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

// =====================================================================
// ESTADO DE CÁMARA ORBITAL
// =====================================================================

struct CameraState {
    theta: f32,
    phi: f32,
    radius: f32,
    is_dragging: bool,
    last_mouse_x: f32,
    last_mouse_y: f32,
}

#[wasm_bindgen(start)]
pub async fn start() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Debug).expect("No se pudo inicializar console_log");
    
    log::info!("Iniciando Motor de Partículas WebGPU...");

    let window = web_sys::window().ok_or("No existe la ventana global 'window'")?;
    let document = window.document().ok_or("No existe el objeto 'document'")?;
    
    let canvas = document
        .get_element_by_id("canvas")
        .ok_or("No se encontró el canvas con id='canvas'")?
        .dyn_into::<web_sys::HtmlCanvasElement>()?;

    let mut width = window.inner_width()?.as_f64().ok_or("Ancho de ventana inválido")? as u32;
    let mut height = window.inner_height()?.as_f64().ok_or("Alto de ventana inválido")? as u32;
    canvas.set_width(width);
    canvas.set_height(height);

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
                required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
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
        width,
        height,
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

    // Escribir datos iniciales de luz (Ambiental violeta profundo + Luz direccional blanca inclinada)
    let initial_lighting = LightingConfig {
        ambient_color: [0.15, 0.1, 0.25, 0.8], // Violeta neón ambiental
        light_dir: [1.0, 1.5, 1.0, 0.0],       // Luz desde el cuadrante superior derecho
        params: [0.005, 1.0, 0.8, 0.0],
    };
    unsafe {
        queue.write_buffer(&lighting_buffer, 0, any_as_u8_slice(&initial_lighting));
    }

    // =====================================================================
    // CREACIÓN DEL BIND GROUP LAYOUT Y BIND GROUP
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

    // Cargar Shaders WGSL (Compilados/Concatenados en tiempo de compilación para cero sobrecarga de ejecución)
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

    // Crear Pipeline Layout con nuestro Bind Group Layout
    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Render Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[], // Expansión de geometría interna en el Shader
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                // Activamos mezcla de colores (Alpha Blending) para dar un efecto de brillo neón acumulado
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::One, // Aditivo para el brillo
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
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
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
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let glass_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Glass Pipeline"),
        layout: Some(&render_pipeline_layout),
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
            cull_mode: None, // Sin culling para ver tanto el cristal frontal como el trasero
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    log::info!("Pipeline y Uniforms inicializados. Lanzando bucle interactivo.");

    // Bucle de Renderizado
    let f = Rc::new(RefCell::new(None));
    let g = f.clone();

    let device = Rc::new(device);
    let queue = Rc::new(queue);
    let surface = Rc::new(surface);
    let render_pipeline = Rc::new(render_pipeline);
    let line_pipeline = Rc::new(line_pipeline);
    let glass_pipeline = Rc::new(glass_pipeline);
    let bind_group = Rc::new(bind_group);
    let camera_buffer = Rc::new(camera_buffer);
    let lighting_buffer = Rc::new(lighting_buffer);

    let canvas_clone = canvas.clone();
    let window_clone = window.clone();
    let size_slider_clone = size_slider.clone();
    let spacing_slider_clone = spacing_slider.clone();
    let light_slider_clone = light_slider.clone();
    let lighting_buffer_clone = lighting_buffer.clone();
    let line_pipeline_clone = line_pipeline.clone();
    let glass_pipeline_clone = glass_pipeline.clone();

    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        // A. Resize dinámico
        let current_width = window_clone.inner_width().unwrap().as_f64().unwrap() as u32;
        let current_height = window_clone.inner_height().unwrap().as_f64().unwrap() as u32;
        
        if current_width != width || current_height != height {
            width = current_width;
            height = current_height;
            canvas_clone.set_width(width);
            canvas_clone.set_height(height);
            
            config.width = width;
            config.height = height;
            surface.configure(&device, &config);
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

        let lighting_data = LightingConfig {
            ambient_color: [0.15 * light_val, 0.1 * light_val, 0.25 * light_val, light_val],
            light_dir: [1.0, 1.5, 1.0, 0.0],
            params: [size_val, spacing_val, light_val, 0.0],
        };
        unsafe {
            queue.write_buffer(&lighting_buffer_clone, 0, any_as_u8_slice(&lighting_data));
        }

        // B. CALCULAR PROCESO DE CÁMARA E INTERACCIÓN
        let (eye, view_proj) = {
            let state = camera_state.borrow();
            
            // Convertir ángulos esféricos a posición 3D
            let eye_x = state.radius * state.phi.sin() * state.theta.cos();
            let eye_y = state.radius * state.phi.cos();
            let eye_z = state.radius * state.phi.sin() * state.theta.sin();
            let eye_pos = [eye_x, eye_y, eye_z];

            let view = look_at(eye_pos, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
            let aspect = width as f32 / height as f32;
            let proj = perspective(aspect, 45.0f32.to_radians(), 0.1, 100.0);
            
            (eye_pos, mat4_mul(&proj, &view))
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

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Interactive Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Fondo modo oscuro elegante
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
                timestamp_writes: None,
            });

            // 1. Dibujamos el lattice de líneas conectadas en 3D (9504 vértices)
            render_pass.set_pipeline(&line_pipeline_clone);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.draw(0..9504, 0..1);

            // 2. Dibujamos los puntos nítidos de las partículas (10368 vértices)
            render_pass.set_pipeline(&render_pipeline);
            render_pass.draw(0..10368, 0..1);

            // 3. Dibujamos el cristal encasillador exterior (36 vértices procedimentales)
            render_pass.set_pipeline(&glass_pipeline_clone);
            render_pass.draw(0..36, 0..1);
        }

        queue.submit(std::iter::once(encoder.finish()));
        frame.present();

        request_animation_frame(f.borrow().as_ref().unwrap());
    }) as Box<dyn FnMut()>));

    request_animation_frame(g.borrow().as_ref().unwrap());
    Ok(())
}

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window()
        .unwrap()
        .request_animation_frame(f.as_ref().unchecked_ref())
        .unwrap();
}
