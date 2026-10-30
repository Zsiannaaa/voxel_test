// Texture Atlas Loader + Deterministic Procedural Fallback Generator (`generateAtlas(seed)`).
// Produces the exact same 512x512 padded layout (8x8 grid of 64x64 cells: 32x32 tile + 16px bleed)
// and `atlas.json` metadata schema as `tools/generate_textures.py`, so AI-generated and procedural
// atlases are 100% interchangeable offline.

import { TILE, TILE_NAMES } from '../chunk/blocks.js';

export const ATLAS_GRID = 8;
export const TILE_SIZE = 32;
export const TILE_PAD = 16;
export const CELL_SIZE = TILE_SIZE + TILE_PAD * 2; // 64
export const ATLAS_SIZE = ATLAS_GRID * CELL_SIZE;   // 512

// Deterministic integer hash -> [0, 1)
function hash2(ix, iy, seed) {
  let h = Math.imul(ix, 374761393) ^ Math.imul(iy, 668265263) ^ Math.imul(seed, 1442695041);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function fade(t) {
  return t * t * t * (t * (t * 6 - 15) + 10);
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function clamp(v, lo, hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// 2D Periodic Value-Gradient Noise on a torus of period (px, py) -> [-1, 1]
function pNoise(x, y, px, py, seed) {
  const fx = ((x % px) + px) % px;
  const fy = ((y % py) + py) % py;
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const tx = fx - x0;
  const ty = fy - y0;
  const x1 = (x0 + 1) % px;
  const y1 = (y0 + 1) % py;

  const u = fade(tx);
  const v = fade(ty);

  function gradDot(ix, iy, dx, dy) {
    const angle = hash2(ix, iy, seed) * Math.PI * 2.0;
    return Math.cos(angle) * dx + Math.sin(angle) * dy;
  }

  const n00 = gradDot(x0, y0, tx, ty);
  const n10 = gradDot(x1, y0, tx - 1, ty);
  const n01 = gradDot(x0, y1, tx, ty - 1);
  const n11 = gradDot(x1, y1, tx - 1, ty - 1);

  return lerp(lerp(n00, n10, u), lerp(n01, n11, u), v) * 1.4142;
}

// Multi-octave periodic fBm -> roughly [0, 1]
function pFbm(u, v, baseFreqX, baseFreqY, octaves, seed) {
  let sum = 0;
  let amp = 0.5;
  let norm = 0;
  let fx = baseFreqX;
  let fy = baseFreqY;
  for (let o = 0; o < octaves; o++) {
    sum += amp * pNoise(u * fx, v * fy, fx, fy, seed + o * 1013);
    norm += amp;
    amp *= 0.5;
    fx *= 2;
    fy *= 2;
  }
  return clamp(0.5 + 0.5 * (sum / norm), 0, 1);
}

// 2D Periodic Worley (cellular) noise on (cellsX, cellsY) torus -> { f1, f2, edge, id }
function pWorley(u, v, cellsX, cellsY, seed, jitter = 0.82) {
  const gx = u * cellsX;
  const gy = v * cellsY;
  const ix = Math.floor(gx);
  const iy = Math.floor(gy);

  let f1 = 1e9;
  let f2 = 1e9;
  let bestId = 0;

  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      const cx = (((ix + dx) % cellsX) + cellsX) % cellsX;
      const cy = (((iy + dy) % cellsY) + cellsY) % cellsY;
      const jx = (hash2(cx, cy, seed) - 0.5) * jitter + 0.5;
      const jy = (hash2(cx, cy, seed + 7919) - 0.5) * jitter + 0.5;
      const px = ix + dx + jx;
      const py = iy + dy + jy;
      const dist = Math.hypot(gx - px, gy - py);
      if (dist < f1) {
        f2 = f1;
        f1 = dist;
        bestId = hash2(cx, cy, seed + 31337);
      } else if (dist < f2) {
        f2 = dist;
      }
    }
  }
  return { f1, f2, edge: f2 - f1, id: bestId };
}

// Quantized pixel-art palette ramp sampler
function samplePalette(ramp, t, quantSteps = 0) {
  let k = clamp(t, 0, 0.9999);
  if (quantSteps > 0) {
    k = Math.floor(k * quantSteps) / (quantSteps - 1);
    k = clamp(k, 0, 1);
  }
  const scaled = k * (ramp.length - 1);
  const idx = Math.min(ramp.length - 2, Math.floor(scaled));
  const frac = scaled - idx;
  const c0 = ramp[idx];
  const c1 = ramp[idx + 1];
  return [
    Math.round(lerp(c0[0], c1[0], frac)),
    Math.round(lerp(c0[1], c1[1], frac)),
    Math.round(lerp(c0[2], c1[2], frac)),
  ];
}

const PALETTES = {
  grass: [[40, 96, 41], [54, 118, 50], [70, 141, 58], [89, 163, 68], [112, 184, 82]],
  dirt: [[68, 46, 31], [86, 59, 40], [105, 73, 50], [124, 88, 61], [141, 102, 73]],
  stone: [[84, 88, 94], [102, 106, 113], [120, 124, 131], [137, 141, 148], [154, 158, 164]],
  cobble: [[62, 65, 70], [88, 92, 98], [112, 116, 123], [134, 138, 145], [156, 160, 166]],
  sand: [[194, 161, 101], [211, 179, 117], [225, 196, 135], [237, 211, 153], [246, 224, 172]],
  snow: [[212, 226, 239], [226, 237, 247], [238, 245, 252], [247, 251, 255], [255, 255, 255]],
  gravel: [[76, 75, 78], [99, 96, 98], [121, 116, 116], [142, 136, 133], [163, 157, 152]],
  logRing: [[118, 86, 50], [146, 109, 66], [173, 133, 83], [196, 155, 102], [212, 173, 120]],
  bark: [[45, 29, 18], [62, 41, 26], [80, 54, 35], [97, 67, 44], [113, 79, 53]],
  planks: [[124, 89, 50], [149, 109, 63], [172, 128, 76], [191, 146, 90], [207, 162, 104]],
  leaves: [[29, 73, 30], [42, 97, 41], [57, 121, 51], [74, 145, 62], [95, 167, 75]],
  water: [[22, 88, 148], [32, 112, 179], [48, 137, 206], [84, 173, 232], [146, 216, 250]],
  ice: [[120, 178, 222], [146, 200, 238], [174, 219, 247], [204, 236, 253], [234, 249, 255]],
  bedrock: [[20, 21, 25], [36, 38, 44], [55, 58, 66], [76, 80, 89], [98, 102, 112]],
  clouds: [[219, 227, 239], [232, 238, 247], [243, 247, 253], [250, 252, 255], [255, 255, 255]],
};

function generateStonePixel(u, v, x, y, seed) {
  const strata = pFbm(u, v * 2.0, 4, 8, 3, seed + 301);
  const detail = pFbm(u, v, 8, 8, 2, seed + 347);
  const crack = pWorley(u, v, 5, 5, seed + 389);
  let t = strata * 0.62 + detail * 0.38;
  if (crack.edge < 0.07) t -= 0.16;
  return samplePalette(PALETTES.stone, t, 6);
}

function generateDirtPixel(u, v, x, y, seed) {
  const base = pFbm(u, v, 6, 6, 3, seed + 201);
  const clod = pWorley(u, v, 7, 7, seed + 233);
  let t = base * 0.65 + (1.0 - clamp(clod.f1 * 1.3, 0, 1)) * 0.25 + (clod.id - 0.5) * 0.18;
  // Occasional mineral pebble fleck
  if (hash2(x, y, seed + 277) > 0.94) t += 0.24;
  return samplePalette(PALETTES.dirt, t, 6);
}

function generateTilePixels(tileId, seed) {
  const buf = new Uint8ClampedArray(TILE_SIZE * TILE_SIZE * 4);

  for (let y = 0; y < TILE_SIZE; y++) {
    const v = y / TILE_SIZE;
    for (let x = 0; x < TILE_SIZE; x++) {
      const u = x / TILE_SIZE;
      const off = (y * TILE_SIZE + x) * 4;
      let rgb = [128, 128, 128];
      let alpha = 255;
      const grain = (hash2(x, y, seed + tileId * 977) - 0.5) * 0.08;

      switch (tileId) {
        case TILE.GRASS_TOP: {
          const base = pFbm(u, v, 5, 5, 3, seed + 101);
          const blades = pWorley(u, v, 8, 8, seed + 131);
          const tuft = hash2(x, y, seed + 157);
          let t = base * 0.55 + (1.0 - clamp(blades.f1 * 1.35, 0, 1)) * 0.35 + grain;
          if (tuft > 0.88) t += 0.18;
          else if (tuft < 0.10) t -= 0.14;
          rgb = samplePalette(PALETTES.grass, t, 6);
          break;
        }

        case TILE.GRASS_SIDE: {
          // Horizontally periodic turf lip overhanging warm dirt
          const lipWave = pNoise(u * 4, 0.5, 4, 4, seed + 173) * 2.6 + pNoise(u * 8, 1.5, 8, 8, seed + 191) * 1.4;
          const lipY = Math.round(7.5 + lipWave);
          if (y < lipY) {
            const base = pFbm(u, v, 6, 6, 3, seed + 101);
            let t = base * 0.7 + (1.0 - y / Math.max(1, lipY)) * 0.22 + grain;
            rgb = samplePalette(PALETTES.grass, t, 6);
          } else {
            rgb = generateDirtPixel(u, v, x, y, seed);
            if (y === lipY || y === lipY + 1) {
              const shade = y === lipY ? 0.74 : 0.88;
              rgb = [Math.round(rgb[0] * shade), Math.round(rgb[1] * shade), Math.round(rgb[2] * shade)];
            }
          }
          break;
        }

        case TILE.SNOW_GRASS_SIDE: {
          const lipWave = pNoise(u * 4, 0.5, 4, 4, seed + 177) * 2.5 + pNoise(u * 8, 1.2, 8, 8, seed + 199) * 1.5;
          const lipY = Math.round(8.5 + lipWave);
          if (y < lipY) {
            const t = pFbm(u, v, 5, 5, 2, seed + 601) * 0.7 + 0.25 + grain;
            rgb = samplePalette(PALETTES.snow, t, 5);
          } else {
            rgb = generateDirtPixel(u, v, x, y, seed);
            if (y === lipY) {
              rgb = [Math.round(rgb[0] * 0.78), Math.round(rgb[1] * 0.80), Math.round(rgb[2] * 0.88)];
            }
          }
          break;
        }

        case TILE.DIRT: {
          rgb = generateDirtPixel(u, v, x, y, seed);
          break;
        }

        case TILE.STONE: {
          rgb = generateStonePixel(u, v, x, y, seed);
          break;
        }

        case TILE.COBBLESTONE: {
          const wx = u + pNoise(u * 6, v * 6, 6, 6, seed + 401) * 0.035;
          const wy = v + pNoise(u * 6, v * 6, 6, 6, seed + 419) * 0.035;
          const cell = pWorley(wx, wy, 5, 5, seed + 433, 0.85);
          const micro = pFbm(u, v, 8, 8, 2, seed + 457);
          if (cell.edge < 0.085) {
            rgb = samplePalette(PALETTES.cobble, 0.05 + cell.edge * 1.5, 5);
          } else {
            const dome = clamp((cell.edge - 0.085) * 1.6, 0, 0.45);
            const t = 0.25 + dome + (cell.id - 0.5) * 0.28 + (micro - 0.5) * 0.22 + grain;
            rgb = samplePalette(PALETTES.cobble, t, 6);
          }
          break;
        }

        case TILE.SAND: {
          // Integer wave periods (1 in u, 2 in v) guarantee exact 2D toroidal continuity
          const warp = pNoise(u * 4, v * 4, 4, 4, seed + 503) * 0.28;
          const ripple = 0.5 + 0.5 * Math.sin((v * 2.0 + u * 1.0 + warp) * Math.PI * 2.0);
          const fine = pFbm(u, v, 8, 8, 2, seed + 521);
          const grainDot = (hash2(x, y, seed + 547) - 0.5) * 0.12;
          const t = ripple * 0.34 + fine * 0.50 + 0.14 + grainDot;
          rgb = samplePalette(PALETTES.sand, t, 6);
          break;
        }

        case TILE.SNOW: {
          const drift = pFbm(u, v, 4, 4, 3, seed + 601);
          const sparkle = hash2(x, y, seed + 619) > 0.92 ? 0.18 : 0.0;
          rgb = samplePalette(PALETTES.snow, drift * 0.72 + 0.2 + sparkle, 5);
          break;
        }

        case TILE.GRAVEL: {
          const peb = pWorley(u, v, 8, 8, seed + 701, 0.9);
          const detail = pFbm(u, v, 8, 8, 2, seed + 733);
          let t = peb.id * 0.65 + detail * 0.35 + grain;
          if (peb.edge < 0.09) t *= 0.45;
          rgb = samplePalette(PALETTES.gravel, t, 6);
          // Subtle warm/cool mineral tint variation per pebble
          const warm = Math.round((peb.id - 0.5) * 12);
          rgb = [clamp(rgb[0] + warm, 0, 255), rgb[1], clamp(rgb[2] - warm, 0, 255)];
          break;
        }

        case TILE.LOG_TOP: {
          const dx = (u - 0.5) * 2.0;
          const dy = (v - 0.5) * 2.0;
          const edgeDist = Math.max(Math.abs(dx), Math.abs(dy));
          const rad = Math.hypot(dx, dy) + pNoise(u * 6, v * 6, 6, 6, seed + 809) * 0.04;
          if (edgeDist > 0.81) {
            const bt = pFbm(u, v, 8, 8, 2, seed + 823) * 0.8 + grain;
            rgb = samplePalette(PALETTES.bark, bt, 5);
          } else {
            const rings = 0.5 + 0.5 * Math.sin(rad * Math.PI * 11.0);
            const rays = pNoise(Math.atan2(dy, dx) * 2.546, rad * 4, 16, 4, seed + 853) * 0.12;
            rgb = samplePalette(PALETTES.logRing, rings * 0.68 + 0.18 + rays + grain, 6);
          }
          break;
        }

        case TILE.LOG_SIDE: {
          const ridge = pFbm(u, v, 8, 2, 3, seed + 907);
          const fissure = pWorley(u, v, 7, 2, seed + 929);
          let t = ridge * 0.72 + (1.0 - clamp(fissure.f1 * 1.4, 0, 1)) * 0.25 + grain;
          if (fissure.edge < 0.07) t *= 0.55;
          rgb = samplePalette(PALETTES.bark, t, 6);
          break;
        }

        case TILE.PLANKS: {
          const board = y >> 3; // 4 horizontal boards (0..3)
          const localY = y & 7;
          const jointX = (board & 1) === 0 ? 15 : 27;
          const isHorizGroove = localY === 7;
          const isVertJoint = x === jointX;
          if (isHorizGroove || isVertJoint) {
            rgb = samplePalette(PALETTES.planks, 0.05, 5);
          } else {
            const boardTint = ((board * 37) % 5) * 0.05 - 0.08;
            const woodGrain = pFbm(u, v + board * 0.23, 2, 12, 3, seed + 1009 + board * 31);
            let t = woodGrain * 0.68 + 0.22 + boardTint + grain * 0.6;
            if (localY === 0) t += 0.10; // Top bevel highlight on each plank
            // Tiny nail dots near butt joint
            if ((x === ((jointX + 2) & 31) || x === ((jointX + 30) & 31)) && (localY === 2 || localY === 5)) {
              t = 0.08;
            }
            rgb = samplePalette(PALETTES.planks, t, 6);
          }
          break;
        }

        case TILE.LEAVES: {
          const cluster = pWorley(u, v, 7, 7, seed + 1103, 0.88);
          const detail = pFbm(u, v, 8, 8, 3, seed + 1129);
          const score = (1.0 - clamp(cluster.f1 * 1.25, 0, 1)) * 0.55 + detail * 0.55 + grain;
          if (score < 0.33 && cluster.edge < 0.16) {
            rgb = [0, 0, 0];
            alpha = 0; // Cutout foliage gap
          } else {
            const t = clamp((score - 0.26) * 1.45, 0, 1);
            rgb = samplePalette(PALETTES.leaves, t, 5);
          }
          break;
        }

        case TILE.WATER:
        case TILE.WATER_1:
        case TILE.WATER_2:
        case TILE.WATER_3: {
          const frameIdx = tileId === TILE.WATER ? 0 : (tileId - TILE.WATER_1 + 1);
          const phase = frameIdx * 0.25;
          const su = (u + phase) % 1.0;
          const sv = (v + phase * 0.5) % 1.0;
          const w1 = pWorley(su, sv, 4, 4, seed + 1201, 0.85);
          const w2 = pWorley((u - phase + 1.0) % 1.0, (v + phase) % 1.0, 6, 6, seed + 1237, 0.8);
          const swell = pFbm(su, sv, 3, 3, 2, seed + 1277);
          const caustic = Math.pow(clamp(1.0 - w1.edge * 2.1, 0, 1), 2.2) * 0.55 +
                          Math.pow(clamp(1.0 - w2.edge * 2.4, 0, 1), 2.5) * 0.30;
          const t = swell * 0.45 + caustic + 0.12;
          rgb = samplePalette(PALETTES.water, t, 6);
          alpha = 186;
          break;
        }

        case TILE.ICE: {
          const wx = u + pNoise(u * 4, v * 4, 4, 4, seed + 1307) * 0.04;
          const wy = v + pNoise(u * 4, v * 4, 4, 4, seed + 1319) * 0.04;
          const cell = pWorley(wx, wy, 3, 3, seed + 1301, 0.78);
          const frost = pFbm(u, v, 6, 6, 3, seed + 1327);
          let t = frost * 0.52 + cell.id * 0.28 + 0.14 + grain * 0.5;
          if (cell.edge < 0.045) t = clamp(t + 0.28, 0, 0.96);
          rgb = samplePalette(PALETTES.ice, t, 6);
          alpha = 216;
          break;
        }

        case TILE.COAL_ORE:
        case TILE.IRON_ORE:
        case TILE.GOLD_ORE: {
          rgb = generateStonePixel(u, v, x, y, seed);
          // Chunky 3-6px mineral vein clusters embedded naturally in the granite matrix
          const oreSeed = seed + tileId * 409;
          const wx = u + pNoise(u * 6, v * 6, 6, 6, oreSeed + 17) * 0.045;
          const wy = v + pNoise(u * 6, v * 6, 6, 6, oreSeed + 31) * 0.045;
          const vein = pWorley(wx, wy, 3, 3, oreSeed, 0.72);
          if (vein.f1 < 0.27 && vein.id > 0.28) {
            const core = clamp(1.0 - vein.f1 / 0.27, 0, 1);
            // Add top-left pixel specular highlight inside each mineral nugget
            const hi = clamp(core * 0.85 + grain * 1.4, 0, 1);
            if (tileId === TILE.COAL_ORE) {
              rgb = samplePalette([[16, 18, 22], [28, 31, 38], [46, 50, 60], [72, 77, 88]], hi, 4);
            } else if (tileId === TILE.IRON_ORE) {
              rgb = samplePalette([[118, 70, 46], [166, 108, 74], [208, 154, 116], [238, 198, 164]], hi, 4);
            } else {
              rgb = samplePalette([[146, 96, 14], [204, 150, 24], [244, 204, 52], [255, 245, 150]], hi, 4);
            }
          }
          break;
        }

        case TILE.BEDROCK: {
          const warpU = u + pNoise(u * 4, v * 4, 4, 4, seed + 1709) * 0.14;
          const warpV = v + pNoise(u * 4, v * 4, 4, 4, seed + 1741) * 0.14;
          const strata = pFbm(warpU, warpV, 5, 5, 3, seed + 1723);
          const ridge = Math.abs(strata - 0.5) * 2.2;
          const cell = pWorley(u, v, 5, 5, seed + 1777);
          const t = clamp(ridge * 0.72 + (1.0 - clamp(cell.f1, 0, 1)) * 0.35 + grain, 0, 1);
          rgb = samplePalette([[18, 19, 23], [36, 38, 45], [62, 66, 75], [92, 97, 108], [124, 129, 140]], t, 5);
          break;
        }

        case TILE.GLASS: {
          const isOuter = x === 0 || x === TILE_SIZE - 1 || y === 0 || y === TILE_SIZE - 1;
          const isInnerBevel = (x === 1 || y === 1) && x < TILE_SIZE - 1 && y < TILE_SIZE - 1 && !isOuter;
          const streak1 = (x + y === 11 && x >= 4 && x <= 8) || (x + y === 12 && x >= 4 && x <= 7);
          const streak2 = (x + y === 15 && x >= 6 && x <= 9);
          const streak3 = (x + y === 45 && x >= 21 && x <= 25);
          if (isOuter) {
            rgb = [164, 206, 226];
            alpha = 236;
          } else if (isInnerBevel) {
            rgb = [224, 246, 255];
            alpha = 155;
          } else if (streak1 || streak2 || streak3) {
            rgb = [244, 253, 255];
            alpha = 215;
          } else {
            rgb = [182, 228, 246];
            alpha = 42;
          }
          break;
        }

        case TILE.CLOUDS: {
          const puff = pWorley(u, v, 4, 4, seed + 1901, 0.7);
          const soft = pFbm(u, v, 4, 4, 2, seed + 1931);
          const t = (1.0 - clamp(puff.f1 * 0.9, 0, 1)) * 0.55 + soft * 0.45;
          rgb = samplePalette(PALETTES.clouds, t, 5);
          alpha = 224;
          break;
        }

        case TILE.TALL_GRASS: {
          // Cross-quad meadow grass blades growing upward from y = 31
          alpha = 0;
          rgb = [0, 0, 0];
          const bladeCol = Math.floor(x / 2);
          const colSeed = hash2(bladeCol, 0, seed + 2111);
          if (x >= 2 && x <= 29 && colSeed > 0.16) {
            const tipY = Math.round(5 + hash2(bladeCol, 1, seed + 2137) * 14);
            if (y >= tipY) {
              const growT = (y - tipY) / Math.max(1, 31 - tipY); // 0 at tip, 1 at root
              const sway = Math.sin((1.0 - growT) * 2.1 + (colSeed - 0.5) * 3.0) * (1.0 - growT) * 1.8;
              const center = bladeCol * 2 + 0.5 + sway;
              const halfWidth = 0.45 + growT * 0.65;
              if (Math.abs(x - center) <= halfWidth) {
                alpha = 255;
                rgb = samplePalette(PALETTES.grass, (1.0 - growT) * 0.72 + colSeed * 0.25, 5);
              }
            }
          }
          break;
        }
      }

      buf[off] = rgb[0];
      buf[off + 1] = rgb[1];
      buf[off + 2] = rgb[2];
      buf[off + 3] = alpha;
    }
  }
  return buf;
}

function getTileWrapMode(tileId) {
  if (tileId === TILE.GRASS_SIDE || tileId === TILE.SNOW_GRASS_SIDE) return 'x';
  if (tileId === TILE.LOG_TOP || tileId === TILE.GLASS || tileId === TILE.TALL_GRASS) return 'clamp';
  return 'xy';
}

export function buildAtlasMetadata(generatorLabel = 'procedural-fallback') {
  const tilesMeta = {};
  for (let idx = 0; idx < TILE_NAMES.length; idx++) {
    const name = TILE_NAMES[idx];
    const col = idx % ATLAS_GRID;
    const row = Math.floor(idx / ATLAS_GRID);
    const u0 = (col * CELL_SIZE + TILE_PAD) / ATLAS_SIZE;
    const v0 = (row * CELL_SIZE + TILE_PAD) / ATLAS_SIZE;
    const u1 = u0 + TILE_SIZE / ATLAS_SIZE;
    const v1 = v0 + TILE_SIZE / ATLAS_SIZE;
    tilesMeta[name] = {
      index: idx,
      uv: [+u0.toFixed(6), +v0.toFixed(6), +u1.toFixed(6), +v1.toFixed(6)],
    };
  }

  const blockFaces = {
    1: { name: 'grass', faces: { px: 'grass_side', nx: 'grass_side', py: 'grass_top', ny: 'dirt', pz: 'grass_side', nz: 'grass_side' } },
    2: { name: 'dirt', all: 'dirt' },
    3: { name: 'stone', all: 'stone' },
    4: { name: 'cobblestone', all: 'cobblestone' },
    5: { name: 'sand', all: 'sand' },
    6: { name: 'snow', all: 'snow' },
    7: { name: 'gravel', all: 'gravel' },
    8: { name: 'log', faces: { px: 'log_side', nx: 'log_side', py: 'log_top', ny: 'log_top', pz: 'log_side', nz: 'log_side' } },
    9: { name: 'planks', all: 'planks' },
    10: { name: 'leaves', all: 'leaves' },
    11: { name: 'water', all: 'water', frames: ['water', 'water_1', 'water_2', 'water_3'] },
    12: { name: 'ice', all: 'ice' },
    13: { name: 'coal_ore', all: 'coal_ore' },
    14: { name: 'iron_ore', all: 'iron_ore' },
    15: { name: 'gold_ore', all: 'gold_ore' },
    16: { name: 'bedrock', all: 'bedrock' },
    17: { name: 'glass', all: 'glass' },
    18: { name: 'snow_grass', faces: { px: 'snow_grass_side', nx: 'snow_grass_side', py: 'snow', ny: 'dirt', pz: 'snow_grass_side', nz: 'snow_grass_side' } },
    19: { name: 'tall_grass', all: 'tall_grass' },
  };

  const blocks = {};
  for (const [bid, spec] of Object.entries(blockFaces)) {
    const faces = {};
    for (const fk of ['px', 'nx', 'py', 'ny', 'pz', 'nz']) {
      const tname = spec.faces ? spec.faces[fk] : spec.all;
      faces[fk] = { tile: tname, index: tilesMeta[tname].index, uv: tilesMeta[tname].uv };
    }
    blocks[bid] = {
      name: spec.name,
      faces,
      ...(spec.frames ? { frames: spec.frames.map((fn) => tilesMeta[fn].index) } : {}),
    };
  }

  return {
    version: 1,
    generator: generatorLabel,
    atlasWidth: ATLAS_SIZE,
    atlasHeight: ATLAS_SIZE,
    gridCols: ATLAS_GRID,
    gridRows: ATLAS_GRID,
    tileSize: TILE_SIZE,
    padding: TILE_PAD,
    cellSize: CELL_SIZE,
    tiles: tilesMeta,
    blocks,
  };
}

/**
 * Deterministic procedural texture atlas generator (multi-octave periodic gradient fBm,
 * periodic Worley cellular noise, and quantized pixel-art palette ramps).
 * Works identically in browser and headless Node.js.
 */
export function generateAtlas(seed = 1337) {
  const pixels = new Uint8ClampedArray(ATLAS_SIZE * ATLAS_SIZE * 4);
  const rawTiles = new Map();

  for (let idx = 0; idx < TILE_NAMES.length; idx++) {
    const tileBuf = generateTilePixels(idx, seed);
    rawTiles.set(TILE_NAMES[idx], tileBuf);

    const col = idx % ATLAS_GRID;
    const row = Math.floor(idx / ATLAS_GRID);
    const ox = col * CELL_SIZE;
    const oy = row * CELL_SIZE;
    const wrapMode = getTileWrapMode(idx);
    const wrapX = wrapMode === 'xy' || wrapMode === 'x';
    const wrapY = wrapMode === 'xy';

    // Blit 32x32 tile + 16px wrap/edge bleed padding on all 4 sides
    for (let dy = -TILE_PAD; dy < TILE_SIZE + TILE_PAD; dy++) {
      const sy = wrapY
        ? (((dy % TILE_SIZE) + TILE_SIZE) % TILE_SIZE)
        : clamp(dy, 0, TILE_SIZE - 1);
      const dstY = oy + TILE_PAD + dy;
      for (let dx = -TILE_PAD; dx < TILE_SIZE + TILE_PAD; dx++) {
        const sx = wrapX
          ? (((dx % TILE_SIZE) + TILE_SIZE) % TILE_SIZE)
          : clamp(dx, 0, TILE_SIZE - 1);
        const dstX = ox + TILE_PAD + dx;

        const srcOff = (sy * TILE_SIZE + sx) * 4;
        const dstOff = (dstY * ATLAS_SIZE + dstX) * 4;
        pixels[dstOff] = tileBuf[srcOff];
        pixels[dstOff + 1] = tileBuf[srcOff + 1];
        pixels[dstOff + 2] = tileBuf[srcOff + 2];
        pixels[dstOff + 3] = tileBuf[srcOff + 3];
      }
    }
  }

  let canvas = null;
  if (typeof document !== 'undefined') {
    canvas = document.createElement('canvas');
    canvas.width = ATLAS_SIZE;
    canvas.height = ATLAS_SIZE;
    const ctx = canvas.getContext('2d');
    const imgData = new ImageData(pixels, ATLAS_SIZE, ATLAS_SIZE);
    ctx.putImageData(imgData, 0, 0);
  }

  return {
    width: ATLAS_SIZE,
    height: ATLAS_SIZE,
    pixels,
    source: canvas,
    metadata: buildAtlasMetadata(`procedural-seed-${seed}`),
    isFallback: true,
    rawTiles,
  };
}

/**
 * Loads local `./assets/atlas.json` and `./assets/atlas.png` offline;
 * automatically falls back to `generateAtlas(seed)` if files are absent.
 */
export async function loadAtlas(seed = 1337) {
  try {
    const [metaResp, pngResp] = await Promise.all([
      fetch('./assets/atlas.json'),
      fetch('./assets/atlas.png'),
    ]);
    if (!metaResp.ok || !pngResp.ok) {
      throw new Error('Local atlas files not present');
    }
    const metadata = await metaResp.json();
    const blob = await pngResp.blob();
    const bmp = await createImageBitmap(blob);

    // Extract canvas so HUD hotbar can slice crisp 32x32 block icons directly
    const canvas = document.createElement('canvas');
    canvas.width = bmp.width;
    canvas.height = bmp.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(bmp, 0, 0);
    bmp.close();

    return {
      width: canvas.width,
      height: canvas.height,
      source: canvas,
      metadata,
      isFallback: false,
    };
  } catch (_err) {
    console.info('[Atlas] Local assets/atlas.png unavailable; using deterministic procedural generateAtlas() fallback.');
    return generateAtlas(seed);
  }
}
