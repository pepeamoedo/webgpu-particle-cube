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
    position: vec3<f32>,
    color: vec3<f32>,
    size: f32,
};

// Genera un punto de la cuadrícula del cubo en el espacio 3D
fn generate_particle_grid(in_index: u32) -> Particle {
    var p: Particle;
    
    // Grilla de 12x12x12 = 1,728 partículas en total
    let CUBE_SIZE: i32 = 12;
    let total_particles: i32 = CUBE_SIZE * CUBE_SIZE * CUBE_SIZE;
    
    let index = i32(in_index) % total_particles;
    let ix = index % CUBE_SIZE;
    let iy = (index / CUBE_SIZE) % CUBE_SIZE;
    let iz = index / (CUBE_SIZE * CUBE_SIZE);
    
    // Mapear a rango NDC [-1.0, 1.0] multiplicado por el espaciado dinámico (params.y)
    let spacing = lighting.params.y;
    let fx = (f32(ix) / f32(CUBE_SIZE - 1) - 0.5) * 2.0 * spacing;
    let fy = (f32(iy) / f32(CUBE_SIZE - 1) - 0.5) * 2.0 * spacing;
    let fz = (f32(iz) / f32(CUBE_SIZE - 1) - 0.5) * 2.0 * spacing;
    
    p.position = vec3<f32>(fx, fy, fz);
    
    // Proceso de Color Premium: Degradado tridimensional basado en coordenadas X, Y, Z locales
    let r = f32(ix) / f32(CUBE_SIZE - 1);
    let g = f32(iy) / f32(CUBE_SIZE - 1);
    let b = f32(iz) / f32(CUBE_SIZE - 1);
    
    // Colores RGB neón sumamente vibrantes y estables
    p.color = vec3<f32>(r * 0.9 + 0.1, g * 0.9 + 0.1, b * 0.9 + 0.1);
    
    p.size = 5.0;
    return p;
}
