// =====================================================================
// PROCESO: Simulación de Físicas Dinámicas (Compute Shader)
// =====================================================================
struct Particle {
    pos: vec4<f32>,
    vel: vec4<f32>,
};

@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

struct Params {
    spacing: f32,
    delta_time: f32,
    intensity: f32,
    dummy: f32,
};

@group(0) @binding(1)
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let TOTAL_PARTICLES: u32 = 1728u;
    if (index >= TOTAL_PARTICLES) {
        return;
    }
    
    var p = particles[index];
    
    // Atracción gravitatoria hacia el origen (centro del cubo de cristal)
    let center_dir = -normalize(p.pos.xyz);
    let dist = length(p.pos.xyz);
    let gravity = center_dir * 0.15 / (dist * dist + 0.1);
    
    // Fuerza orbital sutil de rotación en el eje Y (remolino neón)
    let orbit = vec3<f32>(-p.pos.z, 0.0, p.pos.x) * 0.25;
    
    // Actualizar velocidad basada en fuerzas orbitales y gravitatorias
    p.vel = vec4<f32>(p.vel.xyz + (gravity + orbit) * params.delta_time, 0.0);
    
    // Limitar la velocidad máxima para evitar inestabilidades numéricas
    let speed = length(p.vel.xyz);
    if (speed > 2.0) {
        p.vel = vec4<f32>(normalize(p.vel.xyz) * 2.0, 0.0);
    }
    
    // Actualizar posición física
    p.pos = vec4<f32>(p.pos.xyz + p.vel.xyz * params.delta_time, 1.0);
    
    // Contención y rebotes elásticos contra los bordes de la urna de cristal
    let limit = params.spacing * 1.06;
    if (abs(p.pos.x) > limit) {
        p.pos.x = sign(p.pos.x) * limit;
        p.vel.x = -p.vel.x * 0.8;
    }
    if (abs(p.pos.y) > limit) {
        p.pos.y = sign(p.pos.y) * limit;
        p.vel.y = -p.vel.y * 0.8;
    }
    if (abs(p.pos.z) > limit) {
        p.pos.z = sign(p.pos.z) * limit;
        p.vel.z = -p.vel.z * 0.8;
    }
    
    particles[index] = p;
}
