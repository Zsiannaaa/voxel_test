// Tectonic & Climate Biome Synthesis:
// Layered fBm + Domain Warping + Ridged Multifractal Mountains (200-232+ blocks)
// + River Channels + Plateaus + Continentalness / Erosion / Temperature / Humidity biomes.

import { BLOCK } from '../chunk/blocks.js';
import { SEA_LEVEL } from '../chunk/chunk_data.js';
import { SimplexNoise } from './noise.js';

export const BIOME = {
  OCEAN: 0,
  BEACH: 1,
  PLAINS: 2,
  FOREST: 3,
  DESERT: 4,
  SNOWY_MOUNTAINS: 5,
};

export const BIOME_NAMES = [
  'Ocean',
  'Beach',
  'Plains',
  'Forest',
  'Desert',
  'Snowy Mountains',
];

function smoothstep(e0, e1, x) {
  const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

export class BiomeField {
  constructor(seed = 2026) {
    this.seed = seed;
    this.warpA = new SimplexNoise(seed ^ 0x1a2b3c4d);
    this.warpB = new SimplexNoise(seed ^ 0x5f6e7d8c);
    this.continentNoise = new SimplexNoise(seed ^ 0x13579bdf);
    this.erosionNoise = new SimplexNoise(seed ^ 0x2468ace0);
    this.ridgeNoise = new SimplexNoise(seed ^ 0x97531eca);
    this.riverNoise = new SimplexNoise(seed ^ 0x41c3b2a9);
    this.tempNoise = new SimplexNoise(seed ^ 0x77b1d3f5);
    this.humidNoise = new SimplexNoise(seed ^ 0x39e4a1c7);
    this.detailNoise = new SimplexNoise(seed ^ 0x8b6d4f21);
  }

  /**
   * Evaluates column climate, biome, surface height (reaching 205-232+ in alpine peaks),
   * lake/water level, and default surface/subsurface blocks into `out` (zero allocation).
   */
  sampleColumn(wx, wz, out) {
    // 1. Domain warping for organic coastlines, meanders, and twisted mountain ranges
    const dx = this.warpA.fbm2D(wx * 0.0032, wz * 0.0032, 2) * 34.0;
    const dz = this.warpB.fbm2D(wx * 0.0032, wz * 0.0032, 2) * 34.0;
    const qx = wx + dx;
    const qz = wz + dz;

    // 2. Tectonic parameters: Continentalness [-1..1], Erosion [-1..1], Ridged Peaks [0..1]
    const cont = this.continentNoise.fbm2D(qx * 0.00145, qz * 0.00145, 4, 2.0, 0.5);
    const erosion = this.erosionNoise.fbm2D(qx * 0.0021, qz * 0.0021, 3, 2.0, 0.5);
    const ridge = this.ridgeNoise.ridged2D(qx * 0.0030, qz * 0.0030, 5, 2.08, 0.53);
    const detail = this.detailNoise.fbm2D(wx * 0.014, wz * 0.014, 3, 2.0, 0.5);

    // 3. Climate: Temperature & Humidity
    const rawTemp = this.tempNoise.fbm2D(qx * 0.0013, qz * 0.0013, 3);
    const humid = this.humidNoise.fbm2D(qx * 0.0016, qz * 0.0016, 3);

    // 4. Piecewise terrain height spline from Continentalness, Erosion & Ridged noise
    let baseH;
    if (cont < -0.20) {
      // Deep & shelf ocean (Y = 32..59)
      const oceanT = smoothstep(-0.75, -0.20, cont);
      baseH = lerp(33.0, 59.5, oceanT) + detail * 2.5;
    } else if (cont < -0.06) {
      // Coastal shelf -> beach -> lowlands (Y = 59.5..66)
      const coastT = smoothstep(-0.20, -0.06, cont);
      baseH = lerp(59.5, 66.5, coastT) + detail * 2.2;
    } else {
      // Inland terrain: plains/valleys + plateaus + dramatic alpine mountain massifs
      const inland = smoothstep(-0.06, 0.44, cont);
      const lowErosion = smoothstep(0.24, -0.38, erosion); // 1 where un-eroded -> tall mountains!
      const plateauMask = smoothstep(0.18, 0.50, erosion) * smoothstep(0.06, 0.32, cont);

      // Rolling hills & valleys base (65..88)
      baseH = 65.5 + inland * 16.0 + detail * 5.2;

      // Stepped desert/highland plateaus (+24..38 blocks with flat mesa tops)
      if (plateauMask > 0.01) {
        const terrace = smoothstep(0.32, 0.52, 0.5 + 0.5 * this.detailNoise.noise2D(qx * 0.005, qz * 0.005));
        baseH += plateauMask * (14.0 + terrace * 24.0);
      }

      // Dramatic alpine mountain ranges reaching Y = 210..235!
      const mountainWeight = Math.pow(inland, 0.82) * lowErosion;
      if (mountainWeight > 0.005) {
        const peak = Math.pow(ridge, 1.24);
        baseH += mountainWeight * (38.0 + peak * 154.0);
      }
    }

    // 5. Meandering river network & inland lake basins (carves valleys down below SEA_LEVEL)
    const riverField = Math.abs(this.riverNoise.fbm2D(qx * 0.0019, qz * 0.0019, 3));
    let isRiver = false;
    if (cont > -0.14 && riverField < 0.062) {
      // Suppress rivers straight through extreme 170+ alpine summits; carve deep gorges through hills/valleys!
      const gorgeSuppress = smoothstep(175.0, 120.0, baseH);
      if (gorgeSuppress > 0.02) {
        const channel = 1.0 - smoothstep(0.010, 0.062, riverField);
        const riverBed = SEA_LEVEL - 4.5 + detail * 1.4;
        baseH = lerp(baseH, riverBed, channel * gorgeSuppress);
        if (channel * gorgeSuppress > 0.55) isRiver = true;
      }
    }

    // Occasional high-valley / forest lakes where erosion is high and detail dips
    let localWaterLevel = SEA_LEVEL;
    const lakeBasin = smoothstep(0.45, 0.72, erosion) * smoothstep(-0.32, -0.58, detail);
    if (cont > 0.05 && baseH > SEA_LEVEL + 4 && baseH < 105 && lakeBasin > 0.25) {
      const rim = Math.floor(baseH - 1);
      baseH -= lakeBasin * 6.5;
      localWaterLevel = Math.max(SEA_LEVEL, rim - 2);
    }

    let height = Math.max(4, Math.min(244, Math.round(baseH)));

    // Altitude temperature lapse rate (cools rapidly above Y = 105)
    const temp = rawTemp - Math.max(0, height - 92) * 0.0095;

    // 6. Biome classification (prevent desert dunes from abutting snowy pine forests directly)
    let biome;
    if (height < SEA_LEVEL - 1 && !isRiver && localWaterLevel === SEA_LEVEL) {
      biome = BIOME.OCEAN;
    } else if (height <= SEA_LEVEL + 2 && cont < 0.06 && temp > -0.28) {
      biome = BIOME.BEACH;
    } else if (height >= 122 + detail * 6 || temp < -0.32) {
      biome = BIOME.SNOWY_MOUNTAINS;
    } else if (rawTemp > 0.25 && humid < -0.03 && height < 98 + detail * 11) {
      biome = BIOME.DESERT;
    } else if (humid > 0.01) {
      biome = BIOME.FOREST;
    } else {
      biome = BIOME.PLAINS;
    }

    // 7. Default surface & subsurface block selection (refined by slope in terrain_gen.js)
    const snowLine = 136 + Math.round(detail * 7 + Math.max(0, rawTemp) * 18);
    let topBlock = BLOCK.GRASS;
    let subBlock = BLOCK.DIRT;

    if (height < localWaterLevel - 1) {
      topBlock = (temp < -0.2 || detail > 0.22) ? BLOCK.GRAVEL : BLOCK.SAND;
      subBlock = BLOCK.GRAVEL;
    } else if (biome === BIOME.BEACH || (isRiver && height <= SEA_LEVEL + 1)) {
      topBlock = temp < -0.25 ? BLOCK.GRAVEL : BLOCK.SAND;
      subBlock = topBlock;
    } else if (biome === BIOME.DESERT) {
      topBlock = BLOCK.SAND;
      subBlock = BLOCK.SAND;
    } else if (biome === BIOME.SNOWY_MOUNTAINS) {
      if (height >= snowLine) {
        topBlock = BLOCK.SNOW;
        subBlock = BLOCK.SNOW;
      } else if (rawTemp > 0.16) {
        // Warm dry high-country foothills -> rocky highland stone/grass instead of snow stripes
        topBlock = height > 128 ? BLOCK.STONE : BLOCK.GRASS;
        subBlock = height > 128 ? BLOCK.STONE : BLOCK.DIRT;
      } else {
        topBlock = BLOCK.SNOW_GRASS;
        subBlock = BLOCK.DIRT;
      }
    }

    out.height = height;
    out.waterLevel = height <= localWaterLevel ? localWaterLevel : (height <= SEA_LEVEL ? SEA_LEVEL : 0);
    out.biome = biome;
    out.temp = temp;
    out.humid = humid;
    out.erosion = erosion;
    out.cont = cont;
    out.snowLine = snowLine;
    out.topBlock = topBlock;
    out.subBlock = subBlock;
    return out;
  }
}
