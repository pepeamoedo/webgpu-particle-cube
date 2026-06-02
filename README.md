# WebGPU SPH Viscous Fluid — Crystal Box

[![Rust](https://img.shields.io/badge/rust-v1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![WebGPU](https://img.shields.io/badge/WebGPU-wgpu--0.19-brightgreen.svg)](https://wgpu.rs/)
[![WASM](https://img.shields.io/badge/WASM-compiled-blueviolet.svg)](https://webassembly.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **physically-based viscous fluid simulation** running entirely on the GPU, built from scratch in **Rust + WebAssembly + raw WebGPU (`wgpu v0.19`)**. Eight thousand particles simulate Smoothed-Particle Hydrodynamics (SPH) inside a glassmorphic crystal box, reacting in real time to cursor forces and terrestrial gravity. The pipeline includes HDR rendering, real Bloom post-processing, and GPU timestamp profiling.

---

## ⚙️ Physics Architecture — 5-Stage Compute Pipeline

All physics run in a parallel GPU compute pipeline dispatched every frame. No physics code executes on the CPU.

| Stage | Kernel | Work |
|---|---|---|
| 1 | `hash_gen` | Assign each particle to a spatial grid cell (hash) and clear cell arrays |
| 2 | `bitonic_sort` | GPU-parallel Bitonic Merge Sort to order particles by cell hash |
| 3 | `cell_offsets` | Compute cell start/end indices in the sorted array |
| 4 | `sph_density` | Poly6 kernel density estimation; store density & pressure per particle |
| 5 | `sph_force` | Spiky-gradient pressure + Laplacian viscosity + gravity + cursor interaction → Euler-Cromer integration |

### SPH Constants
```wgsl
const H         : f32 = 0.18;   // Smoothing radius
const REST_DENS : f32 = 800.0;  // Rest density
const GAS_CONST : f32 = 150.0;  // Equation of state stiffness
const VISCOSITY : f32 = 0.25;   // Dynamic viscosity
```

### Spatial Hashing
The grid subdivides 3D space into cells of size `H`. Each particle only queries its **27 direct neighbour cells**, reducing neighbour search from O(N²) to O(N) per frame. Particles are sorted by cell hash using a fully GPU-parallel **Bitonic Merge Sort**, making `cell_starts`/`cell_ends` lookup arrays valid without any CPU readback.

---

## 🖱️ Cursor Interaction (3D Ray Casting)

The cursor is projected from 2D screen space into 3D world space every frame using the **inverse View-Projection matrix**:

```
NDC → Ray (near/far unproject) → Intersect view-center plane → Clamp to box limits → Upload to GPU
```

- **Hover** → Repulsive radial push (radius `0.5`, strength `24.0`) — parts the fluid like a blowing fan.
- **Click + Drag** → Attractive vortex (radius `0.65`, strength `28.0`) + tangential swirl — stirs the fluid like a finger in water.

---

## 🎨 Rendering Pipeline

```
┌─────────────────────┐    ┌──────────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────┐
│  Physics Compute    │ → │  MSAA 4x HDR │ → │  Bright  │ → │ Gaussian │ → │  Composite Bloom │
│  (5 stages, GPU)    │    │  Scene Pass  │    │  Extract │    │  Blur H+V│    │  ACES Tonemapping│
└─────────────────────┘    └──────────────┘    └──────────┘    └──────────┘    └──────────────────┘
```

### Scene Pass — MSAA 4x → HDR (`Rgba16Float`)
- **Particles**: Screen-space billboard quads (6 vertices each), spherical-normal diffuse shading. Additive blending for neon glow.
- **Glass box**: 36-vertex procedural cube with Fresnel effect, environment reflections, specular highlights, and glowing neon laser borders.

### Post-Processing (Bloom)
1. **Bright Extract** — threshold pass downsampled to ¼ resolution.
2. **Gaussian Blur H** — horizontal 9-tap kernel at ¼ res.
3. **Gaussian Blur V** — vertical 9-tap kernel at ¼ res.
4. **Composite** — HDR scene + bloom addition → **ACES filmic tonemapping** → gamma correction (sRGB output).

---

## 📷 Camera — Inertia & Damping

Orbital camera with **exponential spring damping** (no abrupt stops):

```rust
let factor = 1.0 - (-12.0 * dt).exp();
state.theta  += (state.target_theta  - state.theta)  * factor;
state.phi    += (state.target_phi    - state.phi)    * factor;
state.radius += (state.target_radius - state.radius) * factor;
```

Controls: **Drag** to orbit · **Scroll** to zoom.

---

## 📊 GPU Timestamp Profiling (Micro-Profiler)

When the browser/GPU supports `timestamp_query`, the engine injects **6 GPU timestamps** bracketing compute and render passes. Results are mapped asynchronously to a staging buffer and surfaced to the HUD as:

> `Compute: 0.2 ms | Render: 1.1 ms | Post: 0.4 ms | Total GPU: 1.7 ms`

---

## 🛠️ Technology Stack

| | |
|---|---|
| **Language** | Rust (`cdylib` target) |
| **Build** | `wasm-pack` → WebAssembly |
| **Graphics API** | WebGPU via `wgpu v0.19.3` |
| **Math** | `glam` (SIMD-accelerated linear algebra) |
| **Shaders** | Modular WGSL — `common` · `particles` · `lines` · `glass` · `compute` · `postprocess` |
| **Frontend** | Vanilla JS + HTML + CSS (glassmorphic HUD) |

---

## 🎮 Controls

| Control | Action |
|---|---|
| Drag mouse | Orbit camera |
| Scroll wheel | Zoom in / out |
| Hover over fluid | Radial repulsion force |
| Click + Drag | Vortex attraction + swirl |
| Size slider | Billboard particle size |
| Spacing slider | Box scale |
| Glow slider | Bloom intensity |

---

## 🚀 Build & Run Locally

### Prerequisites
1. [Rust](https://www.rust-lang.org/tools/install)
2. `wasm-pack`:
   ```bash
   cargo install wasm-pack
   ```
3. A WebGPU-capable browser (Chrome 113+, Edge 113+, Safari TP).

### Steps
```bash
git clone https://github.com/pepeamoedo/webgpu-particle-cube.git
cd webgpu-particle-cube

# Build optimised WASM
wasm-pack build --target web --release

# Serve locally
python3 -m http.server 8000
```

Open **http://localhost:8000** — use **Cmd + Shift + R** after rebuilding to bypass cache.

---

## 📁 Project Structure

```
src/
├── lib.rs                  # WASM entry point, render loop, event handling
└── shaders/
    ├── common.wgsl         # Shared uniforms & particle struct
    ├── compute.wgsl        # 5-stage SPH physics pipeline
    ├── particles.wgsl      # Billboard particle renderer
    ├── lines.wgsl          # Force-line renderer (velocity vectors)
    ├── glass.wgsl          # Glassmorphic crystal box
    └── postprocess.wgsl    # HDR Bloom + ACES tonemapping
pkg/                        # wasm-pack output (gitignored)
index.html                  # Canvas + HUD overlay
```
