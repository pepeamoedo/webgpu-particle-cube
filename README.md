# WebGPU 3D Particle Constellation Engine

[![Spanish Version](https://img.shields.io/badge/lang-español-ff5722.svg)](README.es.md)
[![Rust](https://img.shields.io/badge/rust-v1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![WebGPU](https://img.shields.io/badge/WebGPU-wgpu--0.19-brightgreen.svg)](https://wgpu.rs/)
[![WASM](https://img.shields.io/badge/WASM-compiled-blueviolet.svg)](https://webassembly.org/)

A high-performance, direct-to-metal 3D interactive graphics engine built entirely from scratch in **Rust, WebAssembly, and raw WebGPU (`wgpu v0.19.3`)**. The experience renders a volumetric lattice of $1,728$ dynamic neon particles ($12 \times 12 \times 12$), mathematically interconnected by ultra-thin 1-pixel glowing wireframe paths, all encased inside a refractive, reflective **Glassmorphic Crystal Box** with real-time HUD controls.

---

## 💎 Optical & Rendering Architecture (The 6 Processes)

To maintain maximum performance without any CPU memory footprint, the engine is structured mathematically around **6 concurrent GPU-driven rendering processes** loaded from highly modular, compile-time concatenated WGSL shaders:

### 1. Camera & View Transformation Space
* Calculated in Rust using col-major matrices (LookAt & Perspective projection).
* Injected into `@group(0) @binding(0)` and bound at slot 0, shared across both vertex and fragment stages to allow unified spatial projection.

### 2. Algorithmic Particle Generator
* Generates coordinates for $1,728$ particles procedurally on the GPU at 60 FPS in 3D space, based entirely on the vertex index.
* Maps particles to NDC `[-1.0, 1.0]` multiplied by a dynamic grid spacing parameter.
* Computes a vibrant, three-dimensional RGB neon color gradient based on local particle coordinates.

### 3. Screen-Aligned Quad Billboarding
* Expands each point vertex mathematically into screen-aligned quad billboards (composed of 2 triangles/6 vertices) on-the-fly.
* Renders particles with perfectly sharp, raw circular profiles (no Gaussian blur/post-processing overhead) by discarding fragments outside a unit dot-product radius.
* Incorporates mathematical spherical normals and a diffuse direct lighting model.

### 4. Zero-Buffer Holographic Lattice (Lines)
* Connects the $1,728$ particles regularly in a 3D wireframe network (X, Y, and Z axes).
* Uses a procedurally generated `LineList` of **9,504 vertices** (4,752 connecting segments) calculated inside `vs_line` with conditional branching.
* **No CPU-to-GPU vertex/index buffers** are used for lines; all lines are drawn procedurally, saving massive memory bandwidth.

### 5. Multi-Pass Glow & Additive Blending
* Blends particles and lines additively using `wgpu::BlendState` with a custom `BlendFactor::One` target.
* Colors are mathematically scaled directly by the HUD glow slider parameter, allowing real-time specular glow adjustments.

### 6. Crystalline Enclosure (Glassmorphism)
* Wraps the entire constellation inside a 36-vertex procedural glass cube slightly larger than the grid spacing (`spacing * 1.08`).
* Implements realistic **optical glass physics**:
  * **Fresnel Effect**: Increases reflection/opacity at grazing angles (edges) and reduces it when viewed straight-on.
  * **Procedural Environment Reflection**: Models a dynamic, responsive neon skybox (cyan/magenta) that shifts interactively based on camera movement.
  * **White Specular Highlight**: Evaluates high-frequency physical specularity from the direct light direction.
  * **Neon Laser Borders**: Automatically detects face boundaries using local fragment coordinates and overlays a sharp, glowing, 1px cyan border.
  * **Translucent Blending**: Uses alpha blending (`SrcAlpha`/`OneMinusSrcAlpha`) to tint the background particles in a premium dark violet glass color.


---

## 🚀 AAA Engine Performance Upgrades

To scale the engine to modern commercial standards, the following advanced optimizations have been fully integrated:

### 1. Parallel GPGPU Physics (Compute Shaders)
* **Zero-CPU Physics Loop**: Migrated particle simulation entirely to a highly parallel GPU Compute Pass using raw WGSL compute shaders (`compute.wgsl`).
* **GPU Orbital Physics**: Integrates gravitational attraction towards the origin, a subtle spiral vortex force field, damping limits to ensure stability, and high-performance elastic collision boundary reflections against the walls of the glass box.
* **Shared Storage Buffer**: The compute shader reads and updates the particle coordinates in a read-write storage buffer (`wgpu::BufferUsages::STORAGE`), which is then bound as a read-only buffer in the vertex shader, eliminating any CPU-to-GPU data copies per frame.

### 2. High-Fidelity 4x MSAA (Multi-Sample Anti-Aliasing)
* **Retina Sharpness**: Configured multisampled framebuffers and resolved color targets dynamically to achieve flawless, hardware-level anti-aliased glowing neon lines and particle quad billboards.
* **Multi-Sampled Depth Testing**: Aligned the depth attachment with `sample_count: 4` to preserve precise z-buffer checks for anti-aliased geometries.

### 3. Layout-Reflow-Free DOM (ResizeObserver)
* **Zero Layout Thrashing**: Resizing events are managed using a native, asynchronous `ResizeObserver` layout guard.
* **Stable Rendering Loops**: Avoids accessing layout-blocking properties like `window.innerWidth`/`innerHeight` inside requestAnimationFrame loops, preserving 60 FPS performance without thread reflow delays.

### 4. Native & WebGPU DevTools Diagnostics
* **Naga Validation Unit Tests**: Added a native Rust test suite executing wgpu's `naga` front-end under `cargo test` to validate concatenated WGSL shaders locally on native desktop platforms.
* **Browser Compilation Hook**: Monkey-patched `GPUDevice.prototype.createShaderModule` in `index.html` to intercept runtime shader initialization and render rich, color-coded diagnostic reports with code carets directly in browser consoles if any warnings or compilation errors are found.

---

## 🛠️ Technology Stack
* **Language**: Rust (`cdylib` target).
* **Compilation**: `wasm-pack` compiling to high-performance WebAssembly.
* **Graphics API**: WebGPU (`wgpu v0.19.3` with WebGL2 fallback capabilities).
* **HTML5/DOM**: Pure Vanilla JS, HTML, and CSS (Glassmorphic floating HUD overlay) communicating directly with the Rust main loop at 60fps.
* **Architecture**: Statically concatenated modular shaders (`common.wgsl`, `particles.wgsl`, `lines.wgsl`, `glass.wgsl`) joined at compile-time via Rust `concat!` macro to ensure zero runtime overhead.

---

## 🎮 Real-time Controls
The floating Glassmorphic HUD overlay lets you control the system parameters dynamically:
1. **Particle Size**: Scales the screen-space quad billboard dimension.
2. **Cube Spacing**: Spans or shrinks the 3D particle lattice in real time.
3. **Glow Intensity**: Directly controls the neon light scaling and the glass box reflections/glowing laser borders.
4. **Orbit Camera**: Drag the mouse to orbit around the cube; use the mouse wheel to zoom in and out.

---

## 🚀 Setup & Local Running

### Prerequisites
1. Install [Rust](https://www.rust-lang.org/tools/install).
2. Install `wasm-pack`:
   ```bash
   cargo install wasm-pack
   ```
3. A WebGPU-capable browser (Chrome, Edge, Safari Technology Preview, or Firefox with WebGPU enabled).

### Build & Run
1. Clone this repository:
   ```bash
   git clone https://github.com/pepeamoedo/webgpu-particle-cube.git
   cd webgpu-particle-cube
   ```
2. Build the WebAssembly package:
   ```bash
   wasm-pack build --target web
   ```
3. Run a local HTTP web server to host the project directory:
   ```bash
   python3 -m http.server 8000
   ```
4. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```
   *(Note: Remember to use a **Hard Refresh** (`Cmd + Shift + R`) to clear browser caches when modifying files!)*
