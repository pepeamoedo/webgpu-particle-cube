// =====================================================================
// PROCESO: Simulación de Fluidos SPH y Spatial Hashing (5 Pasos de Cómputo)
// =====================================================================

struct Particle {
    pos: vec4<f32>, // xyz: posición, w: densidad del fluido
    vel: vec4<f32>, // xyz: velocidad, w: presión del fluido
};

struct ParticleKey {
    hash: u32,
    index: u32,
};

struct Params {
    spacing: f32,
    delta_time: f32,
    intensity: f32,
    mouse_active: f32,
    mouse_pos: vec4<f32>,
};

struct GridParams {
    cell_size: f32,
    grid_size: u32,
    num_particles: u32,
    dummy: u32,
};

struct SortParams {
    stage: u32,
    step: u32,
};

// =====================================================================
// BINDINGS COMUNES
// =====================================================================
@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1)
var<storage, read_write> keys: array<ParticleKey>;
@group(0) @binding(2)
var<storage, read_write> cell_starts: array<u32>;
@group(0) @binding(3)
var<storage, read_write> cell_ends: array<u32>;

@group(0) @binding(4)
var<uniform> params: Params;
@group(0) @binding(5)
var<uniform> grid_params: GridParams;
@group(0) @binding(6)
var<uniform> sort_params: SortParams;

// Parámetros de SPH (Fluidos Físicos de Neón)
const H: f32 = 0.18; // Radio de vecindad de partículas
const HSQ: f32 = 0.0324; // H al cuadrado
const POLY6: f32 = 315.0 / (64.0 * 3.14159265 * 0.00188957); // Kernel poly6 para densidad (H^9 = 0.00188957)
const SPIKY_GRAD: f32 = -45.0 / (3.14159265 * 0.000034); // Kernel spiky gradiente para presión (H^6 = 0.000034)
const VISC_LAP: f32 = 45.0 / (3.14159265 * 0.000034); // Kernel viscosidad laplaciano
const REST_DENS: f32 = 800.0; // Densidad de reposo
const GAS_CONST: f32 = 150.0; // Constante de presión de gases
const VISCOSITY: f32 = 0.25; // Viscosidad del fluido
const MASS: f32 = 1.0; // Masa de la partícula

// =====================================================================
// PASO 1: GENERACIÓN DE HASH / CLAVES SPATIAL HASHING
// =====================================================================
@compute @workgroup_size(256)
fn hash_gen(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= grid_params.num_particles) { return; }

    // Clear cells in-place on GPU (completely CPU-overhead free!)
    cell_starts[idx] = 0xffffffffu;
    cell_ends[idx] = 0xffffffffu;

    let p = particles[idx];
    let cell_size = grid_params.cell_size;
    let grid_size = grid_params.grid_size;
    
    // Coordenadas enteras de la celda 3D
    let cell = vec3<i32>(floor(p.pos.xyz / cell_size));
    
    // Función de dispersión espacial robusta
    let h = (u32(cell.x) * 73856093u ^ u32(cell.y) * 19349663u ^ u32(cell.z) * 83492791u) % grid_size;
    
    keys[idx] = ParticleKey(h, idx);
}

// =====================================================================
// PASO 2: ORDENAMIENTO EN PARALELO (BITONIC MERGE SORT)
// =====================================================================
@compute @workgroup_size(256)
fn bitonic_sort(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let keys_idx = global_id.x;
    let N = grid_params.num_particles;
    if (keys_idx >= N) { return; }

    let stage = sort_params.stage;
    let step = sort_params.step;

    let partner_idx = keys_idx ^ step;
    if (partner_idx > keys_idx) {
        let is_descending = ((keys_idx & stage) == 0u);
        let key_a = keys[keys_idx];
        let key_b = keys[partner_idx];

        if (is_descending) {
            if (key_a.hash > key_b.hash) {
                keys[keys_idx] = key_b;
                keys[partner_idx] = key_a;
            }
        } else {
            if (key_a.hash < key_b.hash) {
                keys[keys_idx] = key_b;
                keys[partner_idx] = key_a;
            }
        }
    }
}

// =====================================================================
// PASO 3: REGISTRO DE INTERVALOS DE CELDAS (OFFSETS)
// =====================================================================
@compute @workgroup_size(256)
fn cell_offsets(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let N = grid_params.num_particles;
    if (idx >= N) { return; }

    let hash = keys[idx].hash;
    
    if (idx == 0u) {
        cell_starts[hash] = idx;
    } else {
        let prev_hash = keys[idx - 1u].hash;
        if (hash != prev_hash) {
            cell_starts[hash] = idx;
            cell_ends[prev_hash] = idx;
        }
    }

    if (idx == N - 1u) {
        cell_ends[hash] = N;
    }
}

// =====================================================================
// PASO 4: CÁLCULO DE DENSIDAD Y PRESIÓN DE FLUIDO (SPH)
// =====================================================================
@compute @workgroup_size(256)
fn sph_density(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= grid_params.num_particles) { return; }

    var p = particles[idx];
    let pos = p.pos.xyz;

    var density: f32 = 0.0;
    let cell_size = grid_params.cell_size;
    let grid_size = grid_params.grid_size;
    let center_cell = vec3<i32>(floor(pos / cell_size));

    // Iteración por las 27 celdas espaciales vecinas directas O(1)
    for (var z = -1; z <= 1; z = z + 1) {
        for (var y = -1; y <= 1; y = y + 1) {
            for (var x = -1; x <= 1; x = x + 1) {
                let cell = center_cell + vec3<i32>(x, y, z);
                let h_cell = (u32(cell.x) * 73856093u ^ u32(cell.y) * 19349663u ^ u32(cell.z) * 83492791u) % grid_size;
                
                let start = cell_starts[h_cell];
                let end = cell_ends[h_cell];
                
                if (start != 0xffffffffu) {
                    for (var i = start; i < end; i = i + 1u) {
                        let n_idx = keys[i].index;
                        let n_pos = particles[n_idx].pos.xyz;
                        let r_vec = pos - n_pos;
                        let r2 = dot(r_vec, r_vec);
                        
                        if (r2 < HSQ) {
                            density += POLY6 * pow(HSQ - r2, 3.0);
                        }
                    }
                }
            }
        }
    }

    // Densidad mínima para evitar divisiones por cero en el paso de fuerzas
    p.pos.w = max(0.1, density);
    
    // Ecuación de estado (Gas Law)
    p.vel.w = max(0.0, GAS_CONST * (density - REST_DENS));
    
    particles[idx] = p;
}

// =====================================================================
// PASO 5: INTEGRACIÓN Y FUERZAS FÍSICAS REVOLVENTES (SPH)
// =====================================================================
@compute @workgroup_size(256)
fn sph_force(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= grid_params.num_particles) { return; }

    var p = particles[idx];
    let pos = p.pos.xyz;
    let vel = p.vel.xyz;
    let density = p.pos.w;
    let pressure = p.vel.w;

    var pressure_force = vec3<f32>(0.0);
    var viscosity_force = vec3<f32>(0.0);

    let cell_size = grid_params.cell_size;
    let grid_size = grid_params.grid_size;
    let center_cell = vec3<i32>(floor(pos / cell_size));

    // Búsqueda espacial optimizada O(1) vecinos
    for (var z = -1; z <= 1; z = z + 1) {
        for (var y = -1; y <= 1; y = y + 1) {
            for (var x = -1; x <= 1; x = x + 1) {
                let cell = center_cell + vec3<i32>(x, y, z);
                let h_cell = (u32(cell.x) * 73856093u ^ u32(cell.y) * 19349663u ^ u32(cell.z) * 83492791u) % grid_size;
                
                let start = cell_starts[h_cell];
                let end = cell_ends[h_cell];
                
                if (start != 0xffffffffu) {
                    for (var i = start; i < end; i = i + 1u) {
                        let n_idx = keys[i].index;
                        if (n_idx == idx) { continue; }
                        
                        let n_part = particles[n_idx];
                        let n_pos = n_part.pos.xyz;
                        let r_vec = pos - n_pos;
                        let r2 = dot(r_vec, r_vec);
                        
                        if (r2 < HSQ) {
                            let r = sqrt(max(0.0001, r2));
                            let n_density = n_part.pos.w;
                            let n_pressure = n_part.vel.w;
                            
                            // 1. Fuerza de Presión SPH (Fórmula de Spiky Gradiente)
                            let p_term = -MASS * (pressure + n_pressure) / (2.0 * n_density) * SPIKY_GRAD * pow(H - r, 2.0);
                            pressure_force += normalize(r_vec) * p_term;
                            
                            // 2. Fuerza de Viscosidad SPH (Fórmula Laplaciana)
                            let v_term = VISCOSITY * MASS * VISC_LAP * (H - r) / n_density;
                            viscosity_force += (n_part.vel.xyz - vel) * v_term;
                        }
                    }
                }
            }
        }
    }

    // Acotar fuerzas SPH para prevenir inestabilidades extremas
    let max_force = 150.0;
    if (length(pressure_force) > max_force) {
        pressure_force = normalize(pressure_force) * max_force;
    }

    // Gravedad terrestre hacia abajo (-Y) escalada para la urna cúbica
    let gravity = vec3<f32>(0.0, -9.81 * 0.45, 0.0);
    
    // Fuerza orbital interactiva (remolino de neon suave)
    let orbit = vec3<f32>(-pos.z, 0.0, pos.x) * 0.08;

    // Fuerza de Interacción 3D del Cursor
    var mouse_force = vec3<f32>(0.0);
    if (params.mouse_active > 0.5) {
        let r_vec = pos - params.mouse_pos.xyz;
        let r2 = dot(r_vec, r_vec);
        
        if (params.mouse_active < 1.5) {
            // Modo Hover: Soplido repulsivo radial
            let interaction_radius: f32 = 0.5;
            let interaction_radius_sq = interaction_radius * interaction_radius;
            if (r2 < interaction_radius_sq) {
                let r = sqrt(max(0.0001, r2));
                let force_strength = 24.0 * (1.0 - r / interaction_radius);
                mouse_force = normalize(r_vec) * force_strength;
            }
        } else {
            // Modo Drag: Succión atractora y agitación en vórtice (dedo en el agua)
            let interaction_radius: f32 = 0.65;
            let interaction_radius_sq = interaction_radius * interaction_radius;
            if (r2 < interaction_radius_sq) {
                let r = sqrt(max(0.0001, r2));
                // Succión hacia el cursor
                let attract_strength = -28.0 * (1.0 - r / interaction_radius);
                let attract_force = normalize(r_vec) * attract_strength;
                
                // Vórtice orbital alrededor del eje vertical del cursor
                let tangent = normalize(vec3<f32>(-r_vec.z, 0.0, r_vec.x));
                let vortex_strength = 25.0 * (1.0 - r / interaction_radius);
                let vortex_force = tangent * vortex_strength;
                
                mouse_force = attract_force + vortex_force;
            }
        }
    }

    // Sumar fuerzas combinadas
    let total_force = pressure_force + viscosity_force + gravity + orbit + mouse_force;

    // Actualizar velocidad física
    p.vel = vec4<f32>(vel + total_force * params.delta_time, p.vel.w);

    // Limitar velocidad para un flujo uniforme y suave
    let speed = length(p.vel.xyz);
    if (speed > 2.5) {
        p.vel = vec4<f32>(normalize(p.vel.xyz) * 2.5, p.vel.w);
    }

    // Integrar posición física (Euler-Cromer)
    p.pos = vec4<f32>(pos + p.vel.xyz * params.delta_time, p.pos.w);

    // Colisión elástica contra la urna cúbica flotante
    let limit = params.spacing * 1.06;
    let damping = 0.45; // Rebote viscoso amortiguado
    
    if (abs(p.pos.x) > limit) {
        p.pos.x = sign(p.pos.x) * limit;
        p.vel.x = -p.vel.x * damping;
    }
    if (abs(p.pos.y) > limit) {
        p.pos.y = sign(p.pos.y) * limit;
        p.vel.y = -p.vel.y * damping;
    }
    if (abs(p.pos.z) > limit) {
        p.pos.z = sign(p.pos.z) * limit;
        p.vel.z = -p.vel.z * damping;
    }

    particles[idx] = p;
}
