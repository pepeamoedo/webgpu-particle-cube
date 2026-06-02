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
    
    let CUBE_SIZE: i32 = 12;
    let spacing = lighting.params.y;
    
    var ix: i32 = 0;
    var iy: i32 = 0;
    var iz: i32 = 0;
    
    if (segment_index < 1584u) {
        // Segmentos alineados en X
        ix = i32(segment_index % 11u);
        iy = i32((segment_index / 11u) % 12u);
        iz = i32(segment_index / 132u);
        if (vertex_in_segment == 1u) {
            ix = ix + 1;
        }
    } else if (segment_index < 3168u) {
        // Segmentos alineados en Y
        let local_seg = segment_index - 1584u;
        ix = i32(local_seg % 12u);
        iy = i32((local_seg / 12u) % 11u);
        iz = i32(local_seg / 132u);
        if (vertex_in_segment == 1u) {
            iy = iy + 1;
        }
    } else {
        // Segmentos alineados en Z
        let local_seg = segment_index - 3168u;
        ix = i32(local_seg % 12u);
        iy = i32((local_seg / 12u) % 12u);
        iz = i32(local_seg / 144u);
        if (vertex_in_segment == 1u) {
            iz = iz + 1;
        }
    }
    
    // Mapear coordenadas a NDC [-1.0, 1.0] con espaciado
    let fx = (f32(ix) / f32(CUBE_SIZE - 1) - 0.5) * 2.0 * spacing;
    let fy = (f32(iy) / f32(CUBE_SIZE - 1) - 0.5) * 2.0 * spacing;
    let fz = (f32(iz) / f32(CUBE_SIZE - 1) - 0.5) * 2.0 * spacing;
    
    let pos = vec3<f32>(fx, fy, fz);
    
    // Color degradado igual al cubo
    let r = f32(ix) / f32(CUBE_SIZE - 1);
    let g = f32(iy) / f32(CUBE_SIZE - 1);
    let b = f32(iz) / f32(CUBE_SIZE - 1);
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
