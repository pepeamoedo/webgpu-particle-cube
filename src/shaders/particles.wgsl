// =====================================================================
// PROCESO: Renderizado de Partículas (Billboarding)
// =====================================================================
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>, // Coordenadas locales de la partícula
};

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    
    // Generamos quads de 6 vértices (2 triángulos) por partícula
    let particle_index = in_vertex_index / 6u;
    let quad_vertex_index = in_vertex_index % 6u;
    
    let p = generate_particle_grid(particle_index);
    
    // Determinar UV y offset local para hacer el quad alineado a la pantalla (Billboard)
    var uv = vec2<f32>(0.0, 0.0);
    var offset = vec2<f32>(0.0, 0.0);
    
    // Tamaño de partícula dinámico desde el uniform buffer (params.x)
    let p_size = lighting.params.x;
    
    // Mapeo simétrico de los 6 vértices del quad
    if (quad_vertex_index == 0u) {
        uv = vec2<f32>(-1.0, -1.0);
        offset = vec2<f32>(-p_size, -p_size);
    } else if (quad_vertex_index == 1u) {
        uv = vec2<f32>(1.0, -1.0);
        offset = vec2<f32>(p_size, -p_size);
    } else if (quad_vertex_index == 2u) {
        uv = vec2<f32>(-1.0, 1.0);
        offset = vec2<f32>(-p_size, p_size);
    } else if (quad_vertex_index == 3u) {
        uv = vec2<f32>(-1.0, 1.0);
        offset = vec2<f32>(-p_size, p_size);
    } else if (quad_vertex_index == 4u) {
        uv = vec2<f32>(1.0, -1.0);
        offset = vec2<f32>(p_size, -p_size);
    } else {
        uv = vec2<f32>(1.0, 1.0);
        offset = vec2<f32>(p_size, p_size);
    }
    
    let world_pos = vec4<f32>(p.position, 1.0);
    var clip_pos = camera.view_proj * world_pos;
    
    // Aplicar el tamaño del billboard de forma independiente en espacio de proyección
    clip_pos.x += offset.x * clip_pos.w;
    clip_pos.y += offset.y * clip_pos.w;
    
    out.clip_position = clip_pos;
    out.color = p.color;
    out.uv = uv;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // 1. Dar forma circular perfectamente definida y afilada (sin blur / raw)
    let dist = dot(in.uv, in.uv);
    if (dist > 1.0) {
        discard;
    }
    
    // 2. Proceso de Normales Esféricas y Luz Difusa Directa
    let normal = vec3<f32>(in.uv, sqrt(max(0.0, 1.0 - dist)));
    let diffuse = max(0.0, dot(normal, normalize(lighting.light_dir.xyz))) * 0.45;
    
    // 3. Combinación de Iluminación Ambiental
    let ambient = lighting.ambient_color.rgb;
    
    // Proceso de Color Nítido y Sólido sin desvanecimiento (Alpha = 1.0)
    let final_color = in.color * (ambient + diffuse) * 1.5;
    
    return vec4<f32>(final_color, 1.0);
}
