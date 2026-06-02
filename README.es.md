# Motor de Constelación de Partículas 3D en WebGPU

[![English Version](https://img.shields.io/badge/lang-english-blue.svg)](README.md)
[![Rust](https://img.shields.io/badge/rust-v1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![WebGPU](https://img.shields.io/badge/WebGPU-wgpu--0.19-brightgreen.svg)](https://wgpu.rs/)
[![WASM](https://img.shields.io/badge/WASM-compiled-blueviolet.svg)](https://webassembly.org/)

Un motor gráfico interactivo 3D de alto rendimiento y acceso directo al hardware (direct-to-metal) desarrollado desde cero en **Rust, WebAssembly y WebGPU nativo (`wgpu v0.19.3`)**. La experiencia renderiza una red volumétrica de $1,728$ partículas de neón dinámicas ($12 \times 12 \times 12$), interconectadas matemáticamente por hilos ultrafinos de 1 píxel y protegidas dentro de una urna de **Cristal Translúcido Refractivo (Glassmorphism)** interactiva con controles HUD en tiempo real.

---

## 💎 Arquitectura Óptica y de Renderizado (Los 6 Procesos)

Para garantizar un rendimiento absoluto de 60 FPS sin consumo de memoria en la CPU, el motor gráfico se estructura en **6 procesos de renderizado paralelos en la GPU** mediante sombreadores WGSL modulares y altamente organizados:

### 1. Espacio de Proyección y Cámara
* Calculado en Rust a través de matrices en formato column-major (LookAt y Proyección Perspectiva).
* Inyectado en `@group(0) @binding(0)` con visibilidad compartida en las etapas de vértices y fragmentos para permitir un cálculo espacial unificado.

### 2. Generador Algorítmico de Partículas
* Genera dinámicamente en GPU las coordenadas espaciales de los $1,728$ puntos en 3D en base al índice de vértices.
* Mapea las posiciones al espacio NDC `[-1.0, 1.0]` multiplicado por el slider de espaciado en tiempo real.
* Calcula un degradado tridimensional RGB sumamente vibrante y estable basado en las posiciones relativas locales.

### 3. Expansión Billboard de Quads Alineados
* Expande matemáticamente en la GPU cada vértice de punto en quads alineados a la cámara (billboards formados por 2 triángulos / 6 vértices) al vuelo.
* Renderiza perfiles circulares perfectamente afilados y sólidos (sin desenfoque gaussiano ni sobrecarga de post-procesamiento) mediante el descarte selectivo de fragmentos (`discard`) fuera del radio unitario.
* Incorpora normales esféricas matemáticas e iluminación difusa directa.

### 4. Lattice Holográfico sin Buffers (Líneas)
* Conecta las $1,728$ partículas de forma regular a lo largo de los ejes X, Y y Z en una constelación tridimensional.
* Dibuja una `LineList` de **9,504 vértices** (4,752 segmentos de conexión de 1 píxel) calculados en `vs_line` mediante condicionales.
* **Cero consumo de memoria de CPU**: Las líneas se calculan procedimentalmente al vuelo en la GPU, ahorrando un ancho de banda masivo de buffers de vértices/índices.

### 5. Brillo Neon y Mezcla Aditiva Multi-Pass
* Combina las partículas y las líneas mediante mezcla aditiva utilizando `wgpu::BlendState` con factor de destino `BlendFactor::One`.
* La luminiscencia de los hilos de constelación se escala directamente en el shader mediante el slider de brillo para permitir una regulación de luminosidad reactiva premium.

### 6. Urna de Cristal Protectora (Glassmorphism)
* Envuelve la constelación en un cubo de cristal de 36 vértices procedimentales (12 triángulos) ligeramente mayor que la cuadrícula (`spacing * 1.08`).
* Implementa **físicas ópticas realistas de cristal**:
  * **Efecto Fresnel**: Aumenta la reflectividad y opacidad del cristal en los bordes oblicuos según el ángulo de visión de la cámara y lo vuelve transparente en el centro.
  * **Reflexión de Entorno Procedimental**: Simula un cielo de neón dinámico (cian y magenta) que reacciona a la órbita de la cámara.
  * **Destello Especular Blanco**: Evalúa la física especular directa de alta frecuencia.
  * **Bordes Láser de Neón**: Detecta matemáticamente la cercanía a los límites de las caras del cubo y dibuja un borde luminoso cian ultrafino de 1px.
  * **Cuerpo Semitransparente**: Utiliza mezcla de color alfa estándar (`SrcAlpha`/`OneMinusSrcAlpha`) aportando un tinte violeta oscuro premium.

---

## 🛠️ Stack Tecnológico
* **Lenguaje**: Rust (compilado con target `cdylib`).
* **Entorno**: WebAssembly compilado a alta velocidad con `wasm-pack`.
* **API de Gráficos**: WebGPU (`wgpu v0.19.3` con soporte downlevel para WebGL2).
* **DOM/Frontend**: HTML5, CSS (interfaz flotante Glassmorphism) y JS Vanilla para orquestación e interacción de eventos a 60fps.
* **Compilación de Shaders**: Sombreadores modulares (`common.wgsl`, `particles.wgsl`, `lines.wgsl`, `glass.wgsl`) fusionados estáticamente en tiempo de compilación mediante la macro `concat!` de Rust.

---

## 🎮 Controles Interactivos
El panel Glassmorphism flotante te permite variar los parámetros en tiempo real:
1. **Tamaño Partículas**: Escala el tamaño del billboard de las partículas.
2. **Espaciado del Cubo**: Contrae o expande los nodos de la constelación.
3. **Intensidad Brillo**: Controla directamente la potencia del brillo neón, la visibilidad de los reflejos del cristal y los bordes láser.
4. **Cámara Orbital**: Arrastra el ratón para orbitar en 3D; rueda el scroll para hacer zoom in/out.

---

## 🚀 Instalación y Ejecución Local

### Requisitos previos
1. Instalar [Rust](https://www.rust-lang.org/tools/install).
2. Instalar `wasm-pack`:
   ```bash
   cargo install wasm-pack
   ```
3. Un navegador con soporte para WebGPU (Chrome, Edge, Safari Technology Preview o Firefox con el flag activado).

### Compilar y Desplegar
1. Clona este repositorio:
   ```bash
   git clone https://github.com/pepeamoedo/webgpu-particle-cube.git
   cd webgpu-particle-cube
   ```
2. Compila el paquete de WebAssembly:
   ```bash
   wasm-pack build --target web
   ```
3. Levanta un servidor HTTP local para servir el directorio:
   ```bash
   python3 -m http.server 8000
   ```
4. Abre tu navegador en:
   ```
   http://localhost:8000
   ```
   *(Nota: Recuerda realizar un **Hard Refresh** (`Cmd + Shift + R`) al modificar tus archivos para limpiar la caché del navegador).*
