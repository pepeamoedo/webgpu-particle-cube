// =====================================================================
// PROCESO: Cubo de Cristal Encasillador (Glassmorphism / Transparencia)
// =====================================================================
struct GlassVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) local_pos: vec3<f32>,
};

@vertex
fn vs_glass(
    @builtin(vertex_index) in_vertex_index: u32,
) -> GlassVertexOutput {
    var out: GlassVertexOutput;
    
    let face_idx = in_vertex_index / 6u;
    let quad_idx = in_vertex_index % 6u;
    
    var uv = vec2<f32>(0.0, 0.0);
    if (quad_idx == 0u) {
        uv = vec2<f32>(-1.0, -1.0);
    } else if (quad_idx == 1u) {
        uv = vec2<f32>(1.0, -1.0);
    } else if (quad_idx == 2u) {
        uv = vec2<f32>(-1.0, 1.0);
    } else if (quad_idx == 3u) {
        uv = vec2<f32>(-1.0, 1.0);
    } else if (quad_idx == 4u) {
        uv = vec2<f32>(1.0, -1.0);
    } else {
        uv = vec2<f32>(1.0, 1.0);
    }
    
    var local_pos = vec3<f32>(0.0, 0.0, 0.0);
    var local_norm = vec3<f32>(0.0, 0.0, 0.0);
    
    if (face_idx == 0u) {
        // Front (Z = +1)
        local_pos = vec3<f32>(uv.x, uv.y, 1.0);
        local_norm = vec3<f32>(0.0, 0.0, 1.0);
    } else if (face_idx == 1u) {
        // Back (Z = -1)
        local_pos = vec3<f32>(-uv.x, uv.y, -1.0);
        local_norm = vec3<f32>(0.0, 0.0, -1.0);
    } else if (face_idx == 2u) {
        // Left (X = -1)
        local_pos = vec3<f32>(-1.0, uv.y, uv.x);
        local_norm = vec3<f32>(-1.0, 0.0, 0.0);
    } else if (face_idx == 3u) {
        // Right (X = +1)
        local_pos = vec3<f32>(1.0, uv.y, -uv.x);
        local_norm = vec3<f32>(1.0, 0.0, 0.0);
    } else if (face_idx == 4u) {
        // Top (Y = +1)
        local_pos = vec3<f32>(uv.x, 1.0, -uv.y);
        local_norm = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        // Bottom (Y = -1)
        local_pos = vec3<f32>(uv.x, -1.0, uv.y);
        local_norm = vec3<f32>(0.0, -1.0, 0.0);
    }
    
    // El tamaño del cristal recubre el cubo de partículas.
    // Damos un margen extra de 8% (1.08) sobre el espaciado dinámico (spacing)
    let spacing = lighting.params.y;
    let crystal_size = spacing * 1.08;
    
    let world_pos = local_pos * crystal_size;
    out.world_pos = world_pos;
    out.normal = local_norm;
    out.local_pos = local_pos;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    
    return out;
}

@fragment
fn fs_glass(in: GlassVertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.normal);
    let V = normalize(camera.position.xyz - in.world_pos);
    let L = normalize(lighting.light_dir.xyz);
    let H = normalize(V + L);
    
    // 1. Efecto Fresnel (mayor reflexión e impermeabilidad visual en bordes oblicuos)
    let dot_vn = max(0.0, dot(V, N));
    let fresnel = pow(1.0 - dot_vn, 4.0);
    
    // 2. Reflexión de Entorno de Neón Procedimental
    let R = reflect(-V, N);
    let reflection_color = vec3<f32>(0.0, 0.9, 1.0) * max(0.0, R.y) + vec3<f32>(1.0, 0.0, 0.6) * max(0.0, -R.y);
    let stripes = smoothstep(0.97, 1.0, sin(R.x * 8.0) * sin(R.z * 8.0));
    let reflection = (reflection_color * 0.6 + stripes * vec3<f32>(1.0)) * fresnel;
    
    // 3. Brillo Especular Físico de Alta Frecuencia (Punto de luz directo)
    let specular = pow(max(0.0, dot(N, H)), 64.0) * 0.7 * lighting.params.z;
    let spec_color = vec3<f32>(1.0, 1.0, 1.0) * specular;
    
    // 4. Bordes Luminiscentes de Cristal de 1px (Corte de vidrio láser)
    let edge_val = vec3<f32>(1.0) - abs(in.local_pos);
    var dist_to_edge = 0.0;
    if (abs(N.x) > 0.9) {
        dist_to_edge = min(edge_val.y, edge_val.z);
    } else if (abs(N.y) > 0.9) {
        dist_to_edge = min(edge_val.x, edge_val.z);
    } else {
        dist_to_edge = min(edge_val.x, edge_val.y);
    }
    // Crear un borde ultra fino y brillante de neón
    let border_glow = smoothstep(0.02, 0.0, dist_to_edge);
    let border_color = vec3<f32>(0.0, 0.95, 1.0) * border_glow * 0.9 * lighting.params.z;
    
    // 5. Cuerpo Interno Semitransparente (Tinte violeta súper premium de cristal oscuro)
    let body_color = vec3<f32>(0.08, 0.04, 0.15) * (1.0 - fresnel);
    
    // Suma de componentes (física de luz de cristal)
    let final_color = body_color + reflection * 0.5 + spec_color + border_color;
    
    // Canal alfa adaptativo dinámico
    let alpha = clamp(0.12 + fresnel * 0.35 + specular * 0.8 + border_glow * 0.7, 0.0, 0.95);
    
    return vec4<f32>(final_color, alpha);
}
