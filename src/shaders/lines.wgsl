// =====================================================================
// PROCESO: Generador de Líneas del Lattice (Line List)
// =====================================================================




struct LineVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_line(
    @builtin(vertex_index) in_vertex_index: u32,
) -> LineVertexOutput {
    var out: LineVertexOutput;
    
    let segment_index = in_vertex_index / 2u;
    let vertex_in_segment = in_vertex_index % 2u;
    
    var idx: u32 = 0u;
    
    if (segment_index < 1584u) {
        // Segmentos alineados en X
        let ix = segment_index % 11u;
        let iy = (segment_index / 11u) % 12u;
        let iz = segment_index / 132u;
        
        let start_idx = iz * 144u + iy * 12u + ix;
        if (vertex_in_segment == 0u) {
            idx = start_idx;
        } else {
            idx = start_idx + 1u;
        }
    } else if (segment_index < 3168u) {
        // Segmentos alineados en Y
        let local_seg = segment_index - 1584u;
        let ix = local_seg % 12u;
        let iy = (local_seg / 12u) % 11u;
        let iz = local_seg / 132u;
        
        let start_idx = iz * 144u + iy * 12u + ix;
        if (vertex_in_segment == 0u) {
            idx = start_idx;
        } else {
            idx = start_idx + 12u;
        }
    } else {
        // Segmentos alineados en Z
        let local_seg = segment_index - 3168u;
        let ix = local_seg % 12u;
        let iy = (local_seg / 12u) % 12u;
        let iz = local_seg / 144u;
        
        let start_idx = iz * 144u + iy * 12u + ix;
        if (vertex_in_segment == 0u) {
            idx = start_idx;
        } else {
            idx = start_idx + 144u;
        }
    }
    
    let p = particles[idx];
    let pos = p.pos.xyz;
    
    // Mapeo dinámico y fluido de degradado cromático en caliente según la posición física real
    let spacing = lighting.params.y;
    let r = (p.pos.x / (spacing * 2.0) + 0.5);
    let g = (p.pos.y / (spacing * 2.0) + 0.5);
    let b = (p.pos.z / (spacing * 2.0) + 0.5);
    out.color = vec3<f32>(r * 0.9 + 0.1, g * 0.9 + 0.1, b * 0.9 + 0.1);
    
    out.clip_position = camera.view_proj * vec4<f32>(pos, 1.0);
    return out;
}

@fragment
fn fs_line(in: LineVertexOutput) -> @location(0) vec4<f32> {
    let intensity = lighting.params.z;
    // Multiplicar el color del degradado directamente por la intensidad para un brillo neón potente y visible
    let final_color = in.color * intensity * 0.8;
    return vec4<f32>(final_color, 0.45); // Alfa del 45% para que se vean como hilos finos y elegantes con mezcla aditiva
}
