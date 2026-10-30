# Argon Voxel — Offline Open-World WebGL2 Voxel Engine

A complete, zero-dependency, 100% offline open-world voxel game built with **HTML + ES Modules + Raw WebGL2**.

**Why Raw WebGL2:** Raw WebGL2 provides native integer vertex attributes (`gl.vertexAttribIPointer` packing each voxel vertex into **8 bytes**), a single global static quad index buffer, zero scene-graph/matrix CPU overhead across thousands of chunks, and zero external runtime dependencies for guaranteed offline execution.

---

## 1. Quick Start (Run Locally Offline)

Browsers require an HTTP server for ES modules (`type="module"`) and Web Workers. No `npm install` or internet connection is required:

### Option A — Included Zero-Dependency Node Dev Server (Recommended)
```bash
node tools/serve.mjs
```
Then open **http://127.0.0.1:8080** in Chrome, Edge, or Firefox. *(Pass `--port 3000` to change the port.)*

### Option B — Python Built-in HTTP Server
```bash
python -m http.server 8080
```
Then open **http://127.0.0.1:8080**.

---

## 2. AI Texture Generation Pipeline (`gemini-nano-banana-2.1`)

The game ships with pre-baked padded offline textures (`assets/atlas.png` + `assets/atlas.json`) **and** an automatic runtime procedural fallback (`generateAtlas(seed)` in `textures/atlas.js`), so it runs immediately without an API key or network access.

To regenerate all block textures using Google's **Nano Banana 2.1** image model (`gemini-nano-banana-2.1`) via the Gemini API:

### Prerequisites
- Python 3.9+ with Pillow (`pip install pillow`)
- A Gemini API key set in `GEMINI_API_KEY` (never hardcoded or committed)

### Commands
```powershell
# Windows PowerShell
$env:GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
python tools/generate_textures.py

# Force-regenerate specific tiles (cached tiles are skipped automatically to prevent re-billing)
python tools/generate_textures.py --force stone cobblestone leaves

# Force-regenerate all 22 tiles from scratch
python tools/generate_textures.py --force
```

```bash
# macOS / Linux
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
python3 tools/generate_textures.py
```

To re-bake the deterministic procedural fallback atlas to disk without an API key:
```bash
node tools/bake_fallback_atlas.mjs 1337
```

### Cost Estimate for a Full Texture Regeneration
`gemini-nano-banana-2.1` is priced by output image tokens (~$30 / 1M output tokens, ~$1.50 / 1M input tokens):
- **1K resolution (default, `--image-size 1K`):** ~$0.0336 per tile × 22 tiles = **~$0.74** (up to **~$0.85–$0.95** if 2–3 tiles trigger automatic QA retries).
- **2K resolution (`--image-size 2K`):** ~$0.0504 per tile × 22 tiles = **~$1.11**.
- **Reruns:** **$0.00** (raw API outputs are cached in `tools/.cache/raw/*.png`; animated water frames `water_1..3` are synthesized locally at zero API cost).

### Automated Post-Processing & QA Pipeline
Every generated tile in `tools/generate_textures.py` undergoes:
1. Raw QA verification (rejects non-square outputs, flat colors, visible frames/borders, or corner text/watermark stroke anomalies; retries up to 3 times with prompt modifiers from `tools/prompts.json`).
2. Center-crop to square + toroidal cosine cross-fade (`xy` 2D wrap for terrain materials, `x` horizontal wrap for `grass_side` / `snow_grass_side`, radial symmetry for `log_top`, frame preservation for `glass`).
3. Numeric 3×3 tiled seam verification (compares wrap-boundary gradient against interior adjacent-pixel gradient).
4. Box-filter downscale to `32×32`, HSV brightness/saturation normalization, and 16-color median-cut palette quantization.
5. Cutout/translucent alpha channel derivation (`leaves`, `tall_grass`, `glass`, `water`, `ice`, `clouds`).
6. Packing into `assets/atlas.png` (`512×512`, `8×8` cells of `64×64` px: `32×32` tile + `16` px wrap bleed on all 4 sides, `TEXTURE_MAX_LEVEL = 4`) + `assets/atlas.json` + `assets/atlas_preview_3x3.png`.

---

## 3. Controls

| Input | Action |
| :--- | :--- |
| **Click Canvas** | Lock mouse pointer for first-person look |
| **W / A / S / D** (or Arrow Keys) | Move forward / strafe left / back / strafe right |
| **Mouse** | First-person look |
| **Space** | Jump (ground) / Swim upward (water) / Ascend (fly mode) |
| **F** or **Double-Tap Space** | Toggle **Fly Mode** on/off |
| **Shift** | Sprint (ground) / Descend (fly mode) |
| **Ctrl** | Sprint (ground) / Turbo speed (fly mode) |
| **Left Click** | Break targeted block (6.5-block DDA raycast) |
| **Right Click** | Place selected hotbar block |
| **Middle Click** | Pick targeted block into hotbar |
| **1 – 9** or **Scroll Wheel** | Select hotbar slot |
| **[** / **]** | Scrub time of day backward / forward |
| **F3** | Toggle Debug Telemetry Overlay (FPS, 1% low, chunks, tris, draw calls, VRAM, biome) |
| **Esc** or **O** (or `⚙ Settings`) | Release pointer / Open Engine Settings panel |

---

## 4. Technical Highlights
- **8-Byte Packed Voxel Vertices (`uvec2`):** Encodes local `(x, y, z)`, face normal, 4-level smooth ambient occlusion, 16-level sky light, 8-bit tile ID, greedy quad `(u, v)` span, LOD stride scale, and material animation flags in 64 bits per vertex.
- **Mipmap-Safe Greedy Atlas Sampling:** Computes `dFdx(vUV)` and `dFdy(vUV)` on continuous UVs *before* `fract(vUV)` and samples via `textureGrad()`, eliminating mipmap wrap lines across greedy-merged quads.
- **Quadtree 3-Tier Horizon LOD:** Combines full `16×256×16` 3D voxel chunks (`LOD 0`) with greedy-merged `32×32` (`LOD 1`, stride 2) and `64×64` (`LOD 2`, stride 4) heightfield super-chunks + perimeter terrain skirts so distant 230-block alpine ranges render cheaply out to 32–36+ chunks.
- **16³ Section BFS Cave Culling:** Computes a 15-bit face-to-face connectivity bitmask per `16×16×16` vertical section in the mesher worker and runs a zero-allocation stamped BFS from the camera section every frame, skipping subterranean cave VAOs when sealed beneath solid ground.
- **Off-Thread Worker Pools + 2 ms GPU Upload Budget:** Terrain generation (`terrain.worker.js`) and sky-light/AO greedy meshing (`mesher.worker.js`) run entirely in Web Workers using recycled transferable `ArrayBuffer`s.

---

## 5. Known Limitations
- **Static Fluids & Non-Falling Sand:** Water is generated at sea level and in elevated lake/river basins and can be built into, but broken shorelines do not dynamically flood-simulate into newly dug pits, and sand/gravel blocks do not fall under gravity.
- **Far-Horizon Block Edits (`> 12` chunks away):** Player block edits persist in IndexedDB at arbitrary coordinates and render at full voxel detail within the `LOD 0` radius (`12` chunks / `192` blocks); beyond `192` blocks, `LOD 1/2` horizon super-chunks render the base procedural heightfield until you walk back within range.
- **Point-Light Block Emissives:** Lighting uses baked 16-level sky-light flood propagation, directional sun/moon shading, and per-vertex ambient occlusion (no per-block torch RGB point-light channel).
