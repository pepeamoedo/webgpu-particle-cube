// =====================================================================
// PROCESO: Líneas de Fuerza / Velocidad (Streamlines del Fluido)
// Cada partícula genera un segmento: tail=pos, head=pos+vel*scale
// =====================================================================


struct LineVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) alpha: f32,
};

@vertex
fn vs_line(
    @builtin(vertex_index) in_vertex_index: u32,
) -> LineVertexOutput {
    var out: LineVertexOutput;

    // Cada partícula genera 2 vértices: índice par = cola, índice impar = cabeza
    let particle_idx = in_vertex_index / 2u;
    let is_head = (in_vertex_index % 2u) == 1u;

    let p = particles[particle_idx];
    let pos = p.pos.xyz;
    let vel = p.vel.xyz;

    // Escala de las líneas de fuerza: ajusta para la urna unitaria
    let line_scale: f32 = 0.12;

    var world_pos: vec3<f32>;
    if (is_head) {
        world_pos = pos + vel * line_scale;
    } else {
        world_pos = pos;
    }

    // Velocidad: magnitud para colorear el gradiente
    let speed = length(vel);
    let speed_norm = clamp(speed / 2.5, 0.0, 1.0); // normalizar contra v_max del compute

    // Espectro de calor: azul → cian → verde → amarillo → naranja → rojo
    var col: vec3<f32>;
    if (speed_norm < 0.25) {
        let t = speed_norm / 0.25;
        col = mix(vec3<f32>(0.05, 0.1, 0.9),  vec3<f32>(0.0,  0.8, 0.9),  t);
    } else if (speed_norm < 0.5) {
        let t = (speed_norm - 0.25) / 0.25;
        col = mix(vec3<f32>(0.0,  0.8, 0.9),  vec3<f32>(0.1,  0.9, 0.2),  t);
    } else if (speed_norm < 0.75) {
        let t = (speed_norm - 0.5) / 0.25;
        col = mix(vec3<f32>(0.1,  0.9, 0.2),  vec3<f32>(1.0,  0.8, 0.0),  t);
    } else {
        let t = (speed_norm - 0.75) / 0.25;
        col = mix(vec3<f32>(1.0,  0.8, 0.0),  vec3<f32>(1.0,  0.15, 0.05), t);
    }

    // La cabeza del vector brilla más (énfasis en la dirección)
    let brightness = select(0.5, 1.2, is_head);
    out.color = col * brightness;

    // Transparencia proporcional a la velocidad (líneas muy lentas son casi invisibles)
    out.alpha = clamp(speed_norm * 2.5, 0.08, 0.85);

    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    return out;
}

@fragment
fn fs_line(in: LineVertexOutput) -> @location(0) vec4<f32> {
    let intensity = lighting.params.z;
    let final_color = in.color * intensity * 1.4;
    return vec4<f32>(final_color, in.alpha);
}
