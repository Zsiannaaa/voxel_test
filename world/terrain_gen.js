// Full 16x256x16 Chunk Terrain Generator + Fast Far-Distance LOD Heightfield Generator.
// Includes:
//  - Slope-aware cliff rock exposure on mountains & gorges
//  - Trilinearly-interpolated 3D overhangs & dual-field 3D spaghetti caves (< 0.6 ms/chunk!)
//  - Depth-stratified coal, iron, gold ore clusters & gravel pockets
//  - Sea-level & alpine lake water/ice fill
//  - Deterministic cross-chunk-safe oak & alpine pine trees + tall grass flora

import { BLOCK } from '../chunk/blocks.js';
import { CHUNK_D, CHUNK_H, CHUNK_VOLUME, CHUNK_W, SEA_LEVEL, voxelIndex } from '../chunk/chunk_data.js';
import { BIOME, BiomeField } from './biome.js';
import { SimplexNoise, fastHash2, fastHash3 } from './noise.js';

export class TerrainGenerator {
  constructor(seed = 2026) {
    this.seed = seed;
    this.biomeField = new BiomeField(seed);
    this.overhangNoise = new SimplexNoise(seed ^ 0x51a9e2c3);
    this.caveNoiseA = new SimplexNoise(seed ^ 0x6c4d2e1f);
    this.caveNoiseB = new SimplexNoise(seed ^ 0xa39f81b7);

    // Pre-allocated scratch arrays for zero GC churn inside worker
    this.colHeights = new Int16Array(18 * 18);
    this.colWater = new Int16Array(18 * 18);
    this.colBiomes = new Uint8Array(18 * 18);
    this.colTop = new Uint8Array(18 * 18);
    this.colSub = new Uint8Array(18 * 18);
    this.colTemp = new Float32Array(18 * 18);
    this.colSnowLine = new Int16Array(18 * 18);
    this.colSample = {
      height: 64, waterLevel: 0, biome: 0, temp: 0, humid: 0,
      erosion: 0, cont: 0, snowLine: 132, topBlock: 1, subBlock: 2,
    };

    // Coarse 5 x 33 x 5 lattice (stride 4 in XZ, stride 8 in Y) for 3D overhangs & caves
    const LATTICE_VOL = 5 * 33 * 5;
    this.latOverhang = new Float32Array(LATTICE_VOL);
    this.latCaveA = new Float32Array(LATTICE_VOL);
    this.latCaveB = new Float32Array(LATTICE_VOL);
  }

  getBiomeAt(wx, wz) {
    return this.biomeField.sampleColumn(wx, wz, this.colSample);
  }

  generateChunk(cx, cz) {
    const voxels = new Uint8Array(CHUNK_VOLUME);
    const worldX0 = cx * CHUNK_W;
    const worldZ0 = cz * CHUNK_D;

    let minSurfaceY = 255;
    let maxY = SEA_LEVEL + 1;

    // 1. Sample 18x18 column halo (-1..16) so exact central-difference slope is available
    for (let pz = 0; pz < 18; pz++) {
      const wz = worldZ0 + pz - 1;
      for (let px = 0; px < 18; px++) {
        const wx = worldX0 + px - 1;
        const idx = pz * 18 + px;
        const s = this.biomeField.sampleColumn(wx, wz, this.colSample);
        this.colHeights[idx] = s.height;
        this.colWater[idx] = s.waterLevel;
        this.colBiomes[idx] = s.biome;
        this.colTop[idx] = s.topBlock;
        this.colSub[idx] = s.subBlock;
        this.colTemp[idx] = s.temp;
        this.colSnowLine[idx] = s.snowLine;

        if (px >= 1 && px <= 16 && pz >= 1 && pz <= 16) {
          if (s.height < minSurfaceY) minSurfaceY = s.height;
          if (s.height > maxY) maxY = s.height;
          if (s.waterLevel > maxY) maxY = s.waterLevel;
        }
      }
    }

    const latticeTopY = Math.min(32, ((maxY + 15) >> 3));

    // 2. Populate coarse 3D noise lattice (5 x (latticeTopY+1) x 5)
    for (let gy = 0; gy <= latticeTopY; gy++) {
      const wy = gy * 8;
      for (let gz = 0; gz < 5; gz++) {
        const wz = worldZ0 + gz * 4;
        for (let gx = 0; gx < 5; gx++) {
          const wx = worldX0 + gx * 4;
          const lIdx = (gy * 5 + gz) * 5 + gx;
          this.latOverhang[lIdx] = this.overhangNoise.fbm3D(wx * 0.018, wy * 0.022, wz * 0.018, 2);
          this.latCaveA[lIdx] = this.caveNoiseA.noise3D(wx * 0.024, wy * 0.030, wz * 0.024);
          this.latCaveB[lIdx] = this.caveNoiseB.noise3D(wx * 0.024 + 31.7, wy * 0.030 - 19.3, wz * 0.024 + 53.1);
        }
      }
    }

    // 3. Trilinearly interpolate 3D caves/overhangs & fill stone, ores, surface & water
    for (let z = 0; z < 16; z++) {
      const gz = z >> 2;
      const fz = (z & 3) * 0.25;
      const wz = worldZ0 + z;

      for (let x = 0; x < 16; x++) {
        const gx = x >> 2;
        const fx = (x & 3) * 0.25;
        const wx = worldX0 + x;

        const cIdx = (z + 1) * 18 + (x + 1);
        const baseSurf = this.colHeights[cIdx];
        const waterLv = this.colWater[cIdx];
        const biome = this.colBiomes[cIdx];
        const temp = this.colTemp[cIdx];
        const snowLine = this.colSnowLine[cIdx];

        // Compute terrain slope for natural exposed rock cliffs on mountains & canyons
        const slopeX = Math.abs(this.colHeights[cIdx + 1] - this.colHeights[cIdx - 1]) * 0.5;
        const slopeZ = Math.abs(this.colHeights[cIdx + 18] - this.colHeights[cIdx - 18]) * 0.5;
        const slope = Math.max(slopeX, slopeZ);

        let topB = this.colTop[cIdx];
        let subB = this.colSub[cIdx];
        if (slope > 2.6 && baseSurf > SEA_LEVEL + 4) {
          topB = (slope > 4.2 || baseSurf > 110) ? BLOCK.STONE : BLOCK.COBBLESTONE;
          subB = BLOCK.STONE;
        }

        const colCeilY = Math.min(CHUNK_H - 2, Math.max(baseSurf + 10, waterLv));
        const colBase = (z << 4) | x;

        // Bedrock floor (Y = 0..2)
        voxels[colBase] = BLOCK.BEDROCK;

        let highestSolidY = 0;

        for (let y = 1; y <= colCeilY; y++) {
          const vIdx = (y << 8) | colBase;
          if (y <= 2 && fastHash3(wx, y, wz, this.seed + 11) < (3 - y) * 0.45) {
            voxels[vIdx] = BLOCK.BEDROCK;
            highestSolidY = y;
            continue;
          }

          const gy = y >> 3;
          const fy = (y & 7) * 0.125;
          const l000 = (gy * 5 + gz) * 5 + gx;
          const l100 = l000 + 1;
          const l010 = l000 + 5;
          const l110 = l010 + 1;
          const l001 = l000 + 25;
          const l101 = l001 + 1;
          const l011 = l001 + 5;
          const l111 = l011 + 1;

          // Trilinear overhang density near surface on hills & mountains
          let density = baseSurf - y;
          if (y > SEA_LEVEL + 3 && Math.abs(y - baseSurf) < 14) {
            const oh00 = this.latOverhang[l000] + (this.latOverhang[l100] - this.latOverhang[l000]) * fx;
            const oh10 = this.latOverhang[l010] + (this.latOverhang[l110] - this.latOverhang[l010]) * fx;
            const oh01 = this.latOverhang[l001] + (this.latOverhang[l101] - this.latOverhang[l001]) * fx;
            const oh11 = this.latOverhang[l011] + (this.latOverhang[l111] - this.latOverhang[l011]) * fx;
            const oh0 = oh00 + (oh10 - oh00) * fz;
            const oh1 = oh01 + (oh11 - oh01) * fz;
            const oh = oh0 + (oh1 - oh0) * fy;
            const ohAmp = biome === BIOME.SNOWY_MOUNTAINS ? 6.5 : 3.2;
            density += oh * ohAmp;
          }

          if (density >= 0) {
            // 3D Spaghetti caves (intersection of two zero-isosurfaces caveA^2 + caveB^2 < thresh)
            let carved = false;
            // Prevent caves from punching holes straight through shallow seabed/riverbed floors
            const underWaterShield = (waterLv > 0 && y >= baseSurf - 4 && y <= waterLv + 1);
            if (!underWaterShield && y > 3) {
              const ca00 = this.latCaveA[l000] + (this.latCaveA[l100] - this.latCaveA[l000]) * fx;
              const ca10 = this.latCaveA[l010] + (this.latCaveA[l110] - this.latCaveA[l010]) * fx;
              const ca01 = this.latCaveA[l001] + (this.latCaveA[l101] - this.latCaveA[l001]) * fx;
              const ca11 = this.latCaveA[l011] + (this.latCaveA[l111] - this.latCaveA[l011]) * fx;
              const ca = (ca00 + (ca10 - ca00) * fz) * (1 - fy) + (ca01 + (ca11 - ca01) * fz) * fy;

              const cb00 = this.latCaveB[l000] + (this.latCaveB[l100] - this.latCaveB[l000]) * fx;
              const cb10 = this.latCaveB[l010] + (this.latCaveB[l110] - this.latCaveB[l010]) * fx;
              const cb01 = this.latCaveB[l001] + (this.latCaveB[l101] - this.latCaveB[l001]) * fx;
              const cb11 = this.latCaveB[l011] + (this.latCaveB[l111] - this.latCaveB[l011]) * fx;
              const cb = (cb00 + (cb10 - cb00) * fz) * (1 - fy) + (cb01 + (cb11 - cb01) * fz) * fy;

              const radiusSq = y < 44 ? 0.024 : 0.017;
              if (ca * ca + cb * cb < radiusSq) {
                carved = true;
              }
            }

            if (!carved) {
              voxels[vIdx] = BLOCK.STONE;
              highestSolidY = y;
            }
          }
        }

        if (highestSolidY > maxY) maxY = highestSolidY;
        if (highestSolidY < minSurfaceY) minSurfaceY = highestSolidY;

        // 4. Dress exposed surface crust & ore distribution by depth
        let depthFromAir = 0;
        const subDepth = biome === BIOME.DESERT ? 5 : 3;

        for (let y = highestSolidY; y >= 1; y--) {
          const vIdx = (y << 8) | colBase;
          const cur = voxels[vIdx];
          if (cur === BLOCK.AIR) {
            depthFromAir = 0;
            continue;
          }
          if (cur === BLOCK.BEDROCK) continue;

          if (depthFromAir === 0 && y >= baseSurf - 6) {
            let cap = topB;
            if (biome === BIOME.SNOWY_MOUNTAINS && (cap === BLOCK.SNOW || cap === BLOCK.SNOW_GRASS)) {
              cap = y >= snowLine ? BLOCK.SNOW : BLOCK.SNOW_GRASS;
            }
            if (waterLv > 0 && y < waterLv - 1) {
              cap = subB === BLOCK.SAND ? BLOCK.SAND : BLOCK.GRAVEL;
            }
            voxels[vIdx] = cap;
          } else if (depthFromAir <= subDepth && y >= baseSurf - 8) {
            voxels[vIdx] = subB;
          } else {
            // Depth-stratified 3D ore veins on coarse 2x2x2 cells for realistic multi-block clusters!
            const vx = wx >> 1;
            const vy = y >> 1;
            const vz = wz >> 1;
            const oreRoll = fastHash3(vx, vy, vz, this.seed + 701);
            const subMask = fastHash3(wx, y, wz, this.seed + 739);

            if (y <= 34 && oreRoll > 0.986 && subMask > 0.30) {
              voxels[vIdx] = BLOCK.GOLD_ORE;
            } else if (y <= 96 && oreRoll > 0.968 && oreRoll <= 0.986 && subMask > 0.26) {
              voxels[vIdx] = BLOCK.IRON_ORE;
            } else if (y >= 8 && y <= 138 && oreRoll > 0.940 && oreRoll <= 0.968 && subMask > 0.22) {
              voxels[vIdx] = BLOCK.COAL_ORE;
            } else if (y <= 68 && oreRoll < 0.022 && subMask > 0.20) {
              voxels[vIdx] = BLOCK.GRAVEL;
            }
          }
          depthFromAir++;
        }

        // 5. Sea-level / Lake water & polar ice fill
        if (waterLv > 0 && highestSolidY < waterLv) {
          for (let y = highestSolidY + 1; y <= waterLv; y++) {
            const vIdx = (y << 8) | colBase;
            if (voxels[vIdx] === BLOCK.AIR) {
              const freeze = (y === waterLv && temp < -0.36);
              voxels[vIdx] = freeze ? BLOCK.ICE : BLOCK.WATER;
            }
          }
        }

        // 6. Meadow & forest Tall Grass flora
        if (
          waterLv === 0 &&
          highestSolidY >= SEA_LEVEL + 1 &&
          voxels[(highestSolidY << 8) | colBase] === BLOCK.GRASS
        ) {
          const grassChance = biome === BIOME.PLAINS ? 0.16 : (biome === BIOME.FOREST ? 0.11 : 0.03);
          if (fastHash2(wx, wz, this.seed + 911) < grassChance && highestSolidY + 1 < CHUNK_H - 1) {
            voxels[((highestSolidY + 1) << 8) | colBase] = BLOCK.TALL_GRASS;
            if (highestSolidY + 1 > maxY) maxY = highestSolidY + 1;
          }
        }
      }
    }

    // 7. Deterministic Cross-Chunk-Safe Trees (queries jittered 6x6 cells overlapping chunk +-3 blocks)
    const cellMinX = Math.floor((worldX0 - 3) / 6);
    const cellMaxX = Math.floor((worldX0 + 18) / 6);
    const cellMinZ = Math.floor((worldZ0 - 3) / 6);
    const cellMaxZ = Math.floor((worldZ0 + 18) / 6);

    for (let czCell = cellMinZ; czCell <= cellMaxZ; czCell++) {
      for (let cxCell = cellMinX; cxCell <= cellMaxX; cxCell++) {
        const tx = cxCell * 6 + Math.floor(fastHash2(cxCell, czCell, this.seed + 1301) * 5);
        const tz = czCell * 6 + Math.floor(fastHash2(cxCell, czCell, this.seed + 1327) * 5);
        const prob = fastHash2(cxCell, czCell, this.seed + 1361);

        const ts = this.biomeField.sampleColumn(tx, tz, this.colSample);
        if (ts.waterLevel > 0 || ts.height <= SEA_LEVEL + 1 || ts.height > 134) continue;

        let density = 0;
        let isPine = false;
        if (ts.biome === BIOME.FOREST) {
          density = 0.56;
        } else if (ts.biome === BIOME.PLAINS) {
          density = 0.055;
        } else if (ts.biome === BIOME.SNOWY_MOUNTAINS && ts.height < ts.snowLine - 3) {
          density = 0.24;
          isPine = true;
        }

        if (prob >= density) continue;

        // Verify tree site isn't on a sheer cliff
        const hE = this.biomeField.sampleColumn(tx + 1, tz, this.colSample).height;
        const hN = this.biomeField.sampleColumn(tx, tz + 1, this.colSample).height;
        if (Math.abs(hE - ts.height) > 2 || Math.abs(hN - ts.height) > 2) continue;

        const baseY = ts.height + 1;
        const heightVar = Math.floor(fastHash2(tx, tz, this.seed + 1399) * 3);
        const trunkH = isPine ? (6 + heightVar) : (4 + heightVar);
        if (baseY + trunkH + 3 >= CHUNK_H - 1) continue;

        const lxCenter = tx - worldX0;
        const lzCenter = tz - worldZ0;

        // Stamp trunk if inside this chunk
        if (lxCenter >= 0 && lxCenter < 16 && lzCenter >= 0 && lzCenter < 16) {
          for (let dy = 0; dy < trunkH; dy++) {
            voxels[voxelIndex(lxCenter, baseY + dy, lzCenter)] = BLOCK.LOG;
          }
          if (baseY + trunkH + 2 > maxY) maxY = baseY + trunkH + 2;
        }

        // Stamp leaf canopy overlapping this chunk
        if (isPine) {
          for (let dy = 2; dy <= trunkH + 2; dy++) {
            const rad = dy >= trunkH + 1 ? 1 : (((trunkH + 2 - dy) & 1) === 0 ? 2 : 1);
            const wy = baseY + dy;
            for (let dz = -rad; dz <= rad; dz++) {
              const lz = lzCenter + dz;
              if (lz < 0 || lz >= 16) continue;
              for (let dx = -rad; dx <= rad; dx++) {
                const lx = lxCenter + dx;
                if (lx < 0 || lx >= 16) continue;
                if (dx === 0 && dz === 0 && dy < trunkH) continue;
                if (rad === 2 && Math.abs(dx) === 2 && Math.abs(dz) === 2) continue;
                const idx = voxelIndex(lx, wy, lz);
                if (voxels[idx] === BLOCK.AIR || voxels[idx] === BLOCK.TALL_GRASS) {
                  voxels[idx] = (dy === trunkH + 2 && dx === 0 && dz === 0) ? BLOCK.LEAVES : BLOCK.LEAVES;
                  if (wy > maxY) maxY = wy;
                }
              }
            }
          }
        } else {
          for (let dy = trunkH - 2; dy <= trunkH + 1; dy++) {
            const rad = dy >= trunkH ? 1 : 2;
            const wy = baseY + dy;
            for (let dz = -rad; dz <= rad; dz++) {
              const lz = lzCenter + dz;
              if (lz < 0 || lz >= 16) continue;
              for (let dx = -rad; dx <= rad; dx++) {
                const lx = lxCenter + dx;
                if (lx < 0 || lx >= 16) continue;
                if (dx === 0 && dz === 0 && dy < trunkH) continue;
                if (Math.abs(dx) === rad && Math.abs(dz) === rad && (dy === trunkH - 2 || fastHash3(tx + dx, wy, tz + dz, this.seed) > 0.55)) {
                  continue;
                }
                const idx = voxelIndex(lx, wy, lz);
                if (voxels[idx] === BLOCK.AIR || voxels[idx] === BLOCK.TALL_GRASS) {
                  voxels[idx] = BLOCK.LEAVES;
                  if (wy > maxY) maxY = wy;
                }
              }
            }
          }
        }
      }
    }

    return {
      voxels,
      minSurfaceY: Math.max(1, minSurfaceY),
      maxY: Math.min(CHUNK_H - 2, maxY),
    };
  }

  /**
   * Generates a `(gridN + 2) x (gridN + 2)` heightfield & block grid at `stride` blocks
   * directly from 2D biome fields in ~0.2 ms (used for far-horizon LOD super-chunks).
   */
  generateLODGrid(worldX0, worldZ0, gridN, stride) {
    const pitch = gridN + 2;
    const len = pitch * pitch;
    const heights = new Int16Array(len);
    const topBlocks = new Uint8Array(len);
    const subBlocks = new Uint8Array(len);

    for (let gz = 0; gz < pitch; gz++) {
      const wz = worldZ0 + (gz - 1) * stride + (stride >> 1);
      for (let gx = 0; gx < pitch; gx++) {
        const wx = worldX0 + (gx - 1) * stride + (stride >> 1);
        const idx = gz * pitch + gx;
        const s = this.biomeField.sampleColumn(wx, wz, this.colSample);
        heights[idx] = s.height;
        topBlocks[idx] = (s.waterLevel > 0 && s.height < s.waterLevel && s.temp < -0.36)
          ? BLOCK.ICE
          : s.topBlock;
        subBlocks[idx] = s.subBlock;
      }
    }

    // Apply cliff rock exposure on steep LOD cells so distant alpine peaks show crisp rock faces
    for (let gz = 1; gz <= gridN; gz++) {
      for (let gx = 1; gx <= gridN; gx++) {
        const idx = gz * pitch + gx;
        const h = heights[idx];
        if (h <= SEA_LEVEL + 4) continue;
        const sx = Math.abs(heights[idx + 1] - heights[idx - 1]) / (2 * stride);
        const sz = Math.abs(heights[idx + pitch] - heights[idx - pitch]) / (2 * stride);
        const slope = Math.max(sx, sz);
        if (slope > 2.2) {
          topBlocks[idx] = slope > 3.5 ? BLOCK.STONE : BLOCK.COBBLESTONE;
          subBlocks[idx] = BLOCK.STONE;
        }
      }
    }

    return { heights, topBlocks, subBlocks, waterLevel: SEA_LEVEL };
  }
}
