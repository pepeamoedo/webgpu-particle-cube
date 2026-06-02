// =====================================================================
// SSFR — PASO 1: PROFUNDIDAD DE ESFERAS (Sphere Depth Pass)
// Cada partícula SPH se renderiza como una esfera de profundidad lineal.
// Salida en canal R de Rgba16Float: profundidad en unidades mundo (positiva).
// =====================================================================

struct CameraUniform {
    view_proj: mat4x4<f32>,
    position:  vec4<f32>,
};

struct LightingConfig {
    ambient_color: vec4<f32>,
    light_dir:     vec4<f32>,
    params:        vec4<f32>,  // x: size, y: spacing, z: intensity, w: reserved
};

struct Particle {
    pos: vec4<f32>,
    vel: vec4<f32>,
};

struct SsfrParams {
    near:          f32,
    far:           f32,
    particle_r:    f32,
    aspect:        f32,
    fluid_color:   vec4<f32>,
    texel_size:    vec2<f32>,
    tan_half_fov:  f32,
    _pad:          f32,
    light_dir_vs:  vec4<f32>,
};

@group(0) @binding(0) var<uniform> ssfr_cam:      CameraUniform;
@group(0) @binding(1) var<uniform> ssfr_light:    LightingConfig;
@group(1) @binding(0) var<storage, read> ssfr_particles: array<Particle>;
@group(2) @binding(0) var<uniform> ssfr_params:   SsfrParams;

struct SphereOut {
    @builtin(position) clip_pos:     vec4<f32>,
    @location(0)       uv:           vec2<f32>,
    @location(1)       center_depth: f32,
};

@vertex
fn vs_sphere_depth(@builtin(vertex_index) idx: u32) -> SphereOut {
    var out: SphereOut;

    let p_idx = idx / 6u;
    let q_idx = idx % 6u;
    let p     = ssfr_particles[p_idx];
    let pos   = p.pos.xyz;

    // Profundidad lineal del centro de la partícula (distancia proyectada al plano de vista)
    let to_p    = pos - ssfr_cam.position.xyz;
    let forward = normalize(-ssfr_cam.position.xyz);  // cámara siempre mira al origen
    let depth   = max(ssfr_params.near, dot(to_p, forward));

    // Radio del billboard en NDC para cubrir exactamente la esfera proyectada
    let ndc_r = (ssfr_params.particle_r / depth) * 1.15;  // 15% de margen de seguridad

    var uv = vec2<f32>(0.0, 0.0);
    if      (q_idx == 0u) { uv = vec2<f32>(-1.0, -1.0); }
    else if (q_idx == 1u) { uv = vec2<f32>( 1.0, -1.0); }
    else if (q_idx == 2u) { uv = vec2<f32>(-1.0,  1.0); }
    else if (q_idx == 3u) { uv = vec2<f32>(-1.0,  1.0); }
    else if (q_idx == 4u) { uv = vec2<f32>( 1.0, -1.0); }
    else                   { uv = vec2<f32>( 1.0,  1.0); }

    var clip  = ssfr_cam.view_proj * vec4<f32>(pos, 1.0);
    clip.x   += uv.x * ndc_r * clip.w;
    clip.y   += uv.y * ndc_r * clip.w;

    out.clip_pos     = clip;
    out.uv           = uv;
    out.center_depth = depth;
    return out;
}

@fragment
fn fs_sphere_depth(in: SphereOut) -> @location(0) vec4<f32> {
    let d2 = dot(in.uv, in.uv);
    if (d2 > 1.0) { discard; }

    // Avanzar la profundidad hacia la cámara según la superficie esférica
    let advance = sqrt(1.0 - d2) * ssfr_params.particle_r;
    let sphere_depth = max(ssfr_params.near, in.center_depth - advance);

    // Canal R = profundidad, GBA = 0 (sin fluido en fondo = 0.0)
    return vec4<f32>(sphere_depth, 0.0, 0.0, 1.0);
}
