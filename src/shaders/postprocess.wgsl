// =====================================================================
// POST-PROCESAMIENTO: HDR Bloom, Blur Gaussiano y Tonemapping
// =====================================================================

struct PostProcessVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_post_process(
    @builtin(vertex_index) in_vertex_index: u32,
) -> PostProcessVertexOutput {
    var out: PostProcessVertexOutput;
    let x = f32(i32(in_vertex_index == 1u) * 4 - 1);
    let y = f32(i32(in_vertex_index == 2u) * 4 - 1);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

// ---------------------------------------------------------------------
// 1. EXTRACCIÓN DE BRILLOS (THRESHOLD) Y DOWNSAMPLING
// ---------------------------------------------------------------------
@group(0) @binding(0)
var t_hdr: texture_2d<f32>;
@group(0) @binding(1)
var s_hdr: sampler;

@fragment
fn fs_bright_extract(in: PostProcessVertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(t_hdr, s_hdr, in.uv).rgb;
    let luminance = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    
    var bright_color = vec3<f32>(0.0);
    if (luminance > 1.0) {
        let knee = 0.1;
        let soft = clamp(luminance - 1.0 + knee, 0.0, 2.0 * knee);
        let soft_weight = (soft * soft) / (4.0 * knee + 0.0001);
        let contribution = max(soft_weight, luminance - 1.0) / max(luminance, 0.0001);
        bright_color = color * contribution;
    }
    return vec4<f32>(bright_color, 1.0);
}

// ---------------------------------------------------------------------
// 2. DESENFOQUE GAUSSIANO (HORIZONTAL Y VERTICAL)
// ---------------------------------------------------------------------
@group(0) @binding(0)
var t_blur_source: texture_2d<f32>;
@group(0) @binding(1)
var s_blur_source: sampler;

struct BlurParams {
    texel_size: vec2<f32>,
    dummy: vec2<f32>,
};

@group(1) @binding(0)
var<uniform> blur_params: BlurParams;

@fragment
fn fs_blur_h(in: PostProcessVertexOutput) -> @location(0) vec4<f32> {
    var result = vec3<f32>(0.0);
    let offset = blur_params.texel_size.x;
    
    result += textureSample(t_blur_source, s_blur_source, in.uv + vec2<f32>(-2.0 * offset, 0.0)).rgb * 0.06136;
    result += textureSample(t_blur_source, s_blur_source, in.uv + vec2<f32>(-1.0 * offset, 0.0)).rgb * 0.24477;
    result += textureSample(t_blur_source, s_blur_source, in.uv).rgb * 0.38774;
    result += textureSample(t_blur_source, s_blur_source, in.uv + vec2<f32>(1.0 * offset, 0.0)).rgb * 0.24477;
    result += textureSample(t_blur_source, s_blur_source, in.uv + vec2<f32>(2.0 * offset, 0.0)).rgb * 0.06136;
    
    return vec4<f32>(result, 1.0);
}

@fragment
fn fs_blur_v(in: PostProcessVertexOutput) -> @location(0) vec4<f32> {
    var result = vec3<f32>(0.0);
    let offset = blur_params.texel_size.y;
    
    result += textureSample(t_blur_source, s_blur_source, in.uv + vec2<f32>(0.0, -2.0 * offset)).rgb * 0.06136;
    result += textureSample(t_blur_source, s_blur_source, in.uv + vec2<f32>(0.0, -1.0 * offset)).rgb * 0.24477;
    result += textureSample(t_blur_source, s_blur_source, in.uv).rgb * 0.38774;
    result += textureSample(t_blur_source, s_blur_source, in.uv + vec2<f32>(0.0, 1.0 * offset)).rgb * 0.24477;
    result += textureSample(t_blur_source, s_blur_source, in.uv + vec2<f32>(0.0, 2.0 * offset)).rgb * 0.06136;
    
    return vec4<f32>(result, 1.0);
}

// ---------------------------------------------------------------------
// 3. COMPOSICIÓN FINAL, TONEMAPPING Y CORRECCIÓN GAMMA
// ---------------------------------------------------------------------
struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec4<f32>,
};

struct LightingConfig {
    ambient_color: vec4<f32>,
    light_dir: vec4<f32>,
    params: vec4<f32>, // x: size, y: spacing, z: intensity (HUD slider), w: reserved
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(0) @binding(1)
var<uniform> lighting: LightingConfig;

@group(1) @binding(0)
var t_scene: texture_2d<f32>;
@group(1) @binding(1)
var t_bloom: texture_2d<f32>;
@group(1) @binding(2)
var s_composite: sampler;

@fragment
fn fs_composite(in: PostProcessVertexOutput) -> @location(0) vec4<f32> {
    let scene = textureSample(t_scene, s_composite, in.uv).rgb;
    let bloom = textureSample(t_bloom, s_composite, in.uv).rgb;
    
    let bloom_strength = lighting.params.z * 1.5;
    let color = scene + bloom * bloom_strength;
    
    let x = color;
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    let tonemapped = clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
    
    let final_color = pow(tonemapped, vec3<f32>(1.0 / 2.2));
    
    return vec4<f32>(final_color, 1.0);
}
