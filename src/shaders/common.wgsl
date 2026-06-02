// =====================================================================
// ESTADOS Y UNIFORMS COMUNES
// =====================================================================
struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec4<f32>, // Posición del ojo de la cámara
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct LightingConfig {
    ambient_color: vec4<f32>, // Color (RGB) + Intensidad (Alpha)
    light_dir: vec4<f32>,     // Dirección de la luz
    params: vec4<f32>,        // x: size, y: spacing, z: intensity, w: reserved
};

@group(0) @binding(1)
var<uniform> lighting: LightingConfig;

struct Particle {
    pos: vec4<f32>,
    vel: vec4<f32>,
};

@group(1) @binding(0)
var<storage, read> particles: array<Particle>;
