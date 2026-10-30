// Offline build helper that invokes `generateAtlas(seed)` from `textures/atlas.js`
// and exports `assets/atlas.png`, `assets/atlas.json`, and a 3x3 tiled visual QA preview
// (`assets/atlas_preview_3x3.png`) via Python Pillow so local files exist even without GEMINI_API_KEY.

import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { generateAtlas, TILE_SIZE } from '../textures/atlas.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');
const assetsDir = path.join(rootDir, 'assets');
fs.mkdirSync(assetsDir, { recursive: true });

const seed = Number(process.argv[2] || 1337);
const atlas = generateAtlas(seed);

const jsonPath = path.join(assetsDir, 'atlas.json');
fs.writeFileSync(jsonPath, JSON.stringify(atlas.metadata, null, 2), 'utf8');

const rawBinPath = path.join(assetsDir, '.atlas_rgba.tmp');
fs.writeFileSync(rawBinPath, Buffer.from(atlas.pixels.buffer));

// Also dump individual 32x32 tiles into a temp buffer for 3x3 seam verification + QA contact sheet
const tileEntries = Array.from(atlas.rawTiles.entries());
const tilesBinPath = path.join(assetsDir, '.tiles_rgba.tmp');
const concatTiles = Buffer.alloc(tileEntries.length * TILE_SIZE * TILE_SIZE * 4);
tileEntries.forEach(([, buf], i) => {
  Buffer.from(buf.buffer).copy(concatTiles, i * TILE_SIZE * TILE_SIZE * 4);
});
fs.writeFileSync(tilesBinPath, concatTiles);

const pyScript = `
import json, math, sys
from pathlib import Path
from PIL import Image, ImageDraw

assets = Path(r"${assetsDir.replace(/\\/g, '\\\\')}")
w, h = ${atlas.width}, ${atlas.height}
raw = (assets / ".atlas_rgba.tmp").read_bytes()
img = Image.frombytes("RGBA", (w, h), raw)
img.save(assets / "atlas.png")

names = ${JSON.stringify(tileEntries.map(([n]) => n))}
traw = (assets / ".tiles_rgba.tmp").read_bytes()
stride = 32 * 32 * 4

cols = 5
rows = math.ceil(len(names) / cols)
preview_sz = 96
cell_w, cell_h = preview_sz + 28, preview_sz + 36
sheet = Image.new("RGBA", (cols * cell_w + 16, rows * cell_h + 16), (24, 26, 32, 255))
draw = ImageDraw.Draw(sheet)

max_seam_ratio = 0.0
for idx, name in enumerate(names):
    timg = Image.frombytes("RGBA", (32, 32), traw[idx * stride:(idx + 1) * stride])
    col, row = idx % cols, idx // cols
    x0, y0 = 12 + col * cell_w, 12 + row * cell_h
    for cy in range(0, preview_sz, 8):
        for cx in range(0, preview_sz, 8):
            c = (54, 58, 68, 255) if ((cx // 8 + cy // 8) % 2 == 0) else (38, 41, 49, 255)
            draw.rectangle([x0 + cx, y0 + cy, x0 + cx + 7, y0 + cy + 7], fill=c)
    grid = Image.new("RGBA", (96, 96))
    for gy in range(3):
        for gx in range(3):
            grid.paste(timg, (gx * 32, gy * 32))
            sheet.alpha_composite(timg, (x0 + gx * 32, y0 + gy * 32))

    # Numeric 3x3 seam check on XY-tileable materials (comparing X seam to X interior, Y seam to Y interior)
    if name not in ("grass_side", "snow_grass_side", "log_top", "glass", "tall_grass", "planks"):
        px = grid.convert("RGB").load()
        seam_x = sum(abs(px[31, y][c] - px[32, y][c]) for y in range(32, 64) for c in range(3)) / (32 * 3.0)
        seam_y = sum(abs(px[x, 31][c] - px[x, 32][c]) for x in range(32, 64) for c in range(3)) / (32 * 3.0)
        inter_x = max(1.0, sum(abs(px[x, y][c] - px[x + 1, y][c]) for y in range(32, 64) for x in range(32, 63) for c in range(3)) / (32 * 31 * 3.0))
        inter_y = max(1.0, sum(abs(px[x, y][c] - px[x, y + 1][c]) for y in range(32, 63) for x in range(32, 64) for c in range(3)) / (31 * 32 * 3.0))
        ratio = max(seam_x / inter_x, seam_y / inter_y)
        max_seam_ratio = max(max_seam_ratio, ratio)

    draw.rectangle([x0 - 1, y0 - 1, x0 + preview_sz, y0 + preview_sz], outline=(96, 106, 126, 255))
    draw.text((x0, y0 + preview_sz + 6), f"{idx}: {name}", fill=(226, 232, 242, 255))

sheet.save(assets / "atlas_preview_3x3.png")
print(f"Saved assets/atlas.png ({w}x{h}) and assets/atlas_preview_3x3.png | max 3x3 seam/interior ratio = {max_seam_ratio:.3f}")
`;

const res = spawnSync('python', ['-c', pyScript], { encoding: 'utf8' });
fs.rmSync(rawBinPath, { force: true });
fs.rmSync(tilesBinPath, { force: true });

if (res.status !== 0) {
  console.error(res.stderr);
  process.exit(1);
}
console.log(res.stdout.trim());
