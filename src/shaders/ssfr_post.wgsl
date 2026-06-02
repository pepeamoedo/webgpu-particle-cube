// =====================================================================
// SSFR — Passes 2-5:
//   2. Blur Bilateral Horizontal (funde esferas, preserva siluetas)
//   3. Blur Bilateral Vertical
//   4. Reconstrucción de Normales en espacio de vista
//   5. Sombreado: MERCURIO LÍQUIDO DE NEÓN
// =====================================================================

struct PostOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       uv:       vec2<f32>,
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

// @group(0): textura fuente + sampler (bilateral → textura depth | shade → textura normal)
@group(0) @binding(0) var t_ssfr:  texture_2d<f32>;
@group(0) @binding(1) var s_ssfr:  sampler;
// @group(1): parámetros SSFR (todos los passes)
@group(1) @binding(0) var<uniform> pp: SsfrParams;
// @group(2): escena de fondo — solo fs_fluid_shade
@group(2) @binding(0) var t_scene: texture_2d<f32>;

// Triángulo de pantalla completa (sin vertex buffers)
@vertex
fn vs_ssfr_quad(@builtin(vertex_index) idx: u32) -> PostOut {
    var out: PostOut;
    let x        = f32(i32(idx == 1u) * 4 - 1);
    let y        = f32(i32(idx == 2u) * 4 - 1);
    out.uv       = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    out.clip_pos = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

// ─────────────────────────────────────────────────────────────────────
// BLUR BILATERAL GAUSSIANO — 11 taps separables
// Funde esferas vecinas · Preserva bordes nítidos del fluido
// ─────────────────────────────────────────────────────────────────────
const SIGMA_S: f32 = 6.0;   // sigma espacial (píxeles)
const SIGMA_D: f32 = 0.22;  // sigma de profundidad (unidades mundo)

fn gauss(x: f32, sig: f32) -> f32 {
    return exp(-(x * x) / (2.0 * sig * sig));
}

fn bilateral(uv: vec2<f32>, center_d: f32, step: vec2<f32>) -> f32 {
    var d_acc: f32 = 0.0;
    var w_acc: f32 = 0.0;
    for (var i: i32 = -5; i <= 5; i++) {
        let s  = textureSample(t_ssfr, s_ssfr, uv + step * f32(i)).r;
        if (s < 0.001) { continue; }  // fondo sin fluido, ignorar
        let ws = gauss(f32(i), SIGMA_S);
        let wd = gauss(s - center_d, SIGMA_D);
        d_acc += s * ws * wd;
        w_acc += ws * wd;
    }
    return select(center_d, d_acc / w_acc, w_acc > 0.001);
}

@fragment
fn fs_bilateral_h(in: PostOut) -> @location(0) vec4<f32> {
    let d = textureSample(t_ssfr, s_ssfr, in.uv).r;
    if (d < 0.001) { return vec4<f32>(0.0); }
    let smoothed_d = bilateral(in.uv, d, vec2<f32>(pp.texel_size.x, 0.0));
    return vec4<f32>(smoothed_d, 0.0, 0.0, 1.0);
}

@fragment
fn fs_bilateral_v(in: PostOut) -> @location(0) vec4<f32> {
    let d = textureSample(t_ssfr, s_ssfr, in.uv).r;
    if (d < 0.001) { return vec4<f32>(0.0); }
    let smoothed_d = bilateral(in.uv, d, vec2<f32>(0.0, pp.texel_size.y));
    return vec4<f32>(smoothed_d, 0.0, 0.0, 1.0);
}

// ─────────────────────────────────────────────────────────────────────
// RECONSTRUCCIÓN DE NORMALES en espacio de vista
// ─────────────────────────────────────────────────────────────────────

// Reconstruye la posición en espacio de vista desde UV + profundidad lineal
fn view_pos(uv: vec2<f32>, d: f32) -> vec3<f32> {
    let nx = uv.x * 2.0 - 1.0;
    let ny = 1.0 - uv.y * 2.0;
    return vec3<f32>(
        nx * d * pp.aspect * pp.tan_half_fov,
        ny * d * pp.tan_half_fov,
        -d
    );
}

@fragment
fn fs_normal_reconstruct(in: PostOut) -> @location(0) vec4<f32> {
    let d0 = textureSample(t_ssfr, s_ssfr, in.uv).r;
    if (d0 < 0.001) { return vec4<f32>(0.0); }  // sin fluido → máscara = 0

    let tx = pp.texel_size.x;
    let ty = pp.texel_size.y;

    var dx = textureSample(t_ssfr, s_ssfr, in.uv + vec2<f32>( tx, 0.0)).r;
    var dy = textureSample(t_ssfr, s_ssfr, in.uv + vec2<f32>(0.0,  ty)).r;
    // Diferencias simétricas en bordes del fluido (evitar artefactos)
    if (dx < 0.001) { dx = textureSample(t_ssfr, s_ssfr, in.uv + vec2<f32>(-tx, 0.0)).r; }
    if (dy < 0.001) { dy = textureSample(t_ssfr, s_ssfr, in.uv + vec2<f32>(0.0, -ty)).r; }

    let p0 = view_pos(in.uv, d0);
    let px = view_pos(in.uv + vec2<f32>(tx, 0.0), max(0.001, dx));
    let py = view_pos(in.uv + vec2<f32>(0.0, ty), max(0.001, dy));

    var n = normalize(cross(px - p0, py - p0));
    if (n.z < 0.0) { n = -n; }  // la normal siempre apunta hacia la cámara

    return vec4<f32>(n * 0.5 + 0.5, 1.0);  // RGB: normal empaquetada | A: máscara fluido
}

// ─────────────────────────────────────────────────────────────────────
// SOMBREADO — MERCURIO LÍQUIDO DE NEÓN
// F0=0.82 · Especular 512 · Entorno procedimental cian/blanco/magenta
// ─────────────────────────────────────────────────────────────────────
@fragment
fn fs_fluid_shade(in: PostOut) -> @location(0) vec4<f32> {
    let ns = textureSample(t_ssfr,  s_ssfr, in.uv);
    let bg = textureSample(t_scene, s_ssfr, in.uv).rgb;

    // Sin fluido → pasar el fondo transparente sin modificar
    if (ns.a < 0.5) { return vec4<f32>(bg, 1.0); }

    // Desempaquetar normal en espacio de vista
    let N     = normalize(ns.rgb * 2.0 - 1.0);
    let V     = vec3<f32>(0.0, 0.0, 1.0);           // hacia cámara, siempre +Z en view-space
    let L     = normalize(pp.light_dir_vs.xyz);      // luz key transformada al view-space en CPU
    let H     = normalize(V + L);

    let NdotV = max(0.001, dot(N, V));
    let NdotH = max(0.0,   dot(N, H));

    // Fresnel de Schlick — mercurio líquido
    let F0      = 0.82;
    let fresnel = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);

    // Refracción mínima (mercurio es casi opaco)
    let roff    = clamp(in.uv + N.xy * 0.004, vec2<f32>(0.001), vec2<f32>(0.999));
    let refracted = textureSample(t_scene, s_ssfr, roff).rgb;

    // Entorno procedural de neón — ilumina la superficie metálica
    let env_top   = max(0.0,  N.y) * vec3<f32>(0.90, 0.95, 1.00);    // blanco frío (techo)
    let env_bot   = max(0.0, -N.y) * vec3<f32>(0.02, 0.05, 0.15);    // azul marino (suelo)
    let env_cyan  = abs(N.x)       * vec3<f32>(0.00, 0.55, 0.90) * 0.45; // lateral cian
    let env_magen = max(0.0, -N.z) * vec3<f32>(0.55, 0.00, 0.80) * 0.25; // magenta (interior)
    let env_color = env_top + env_bot + env_cyan + env_magen;

    // Especular ultra-concentrado (mercurio → shininess 512)
    let spec       = pow(NdotH, 512.0) * 4.5;
    let spec_color = vec3<f32>(1.0, 0.97, 0.92) * spec;

    // Color base (plata azulada fría del mercurio)
    let base_col = pp.fluid_color.rgb * (1.0 - fresnel) * 0.10;

    // Composición física: reflexión dominante + transmisión mínima
    let refl   = env_color * fresnel;
    let trans  = refracted * (1.0 - fresnel) * 0.06;
    let color  = refl + trans + base_col + spec_color;

    return vec4<f32>(color, 1.0);
}
