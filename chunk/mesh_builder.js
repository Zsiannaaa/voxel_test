// Greedy Voxel Mesher + Sky-Light Flood Propagation + Smooth 0fps Vertex AO
// + 15-bit Section Cave Visibility Graph + Multi-Tier LOD Heightfield Mesher.
// Zero per-quad heap allocations: reuses module-scoped typed scratch buffers.

import {
  BLOCK,
  BLOCK_AO_OCCLUDER,
  BLOCK_CROSS_QUAD,
  BLOCK_FACE_TILES,
  BLOCK_FLAGS,
  BLOCK_LIGHT_ATTEN,
  BLOCK_OPAQUE,
  BLOCK_PASS,
  FACE_NX,
  FACE_NY,
  FACE_NZ,
  FACE_PX,
  FACE_PY,
  FACE_PZ,
  PASS_CUTOUT,
  PASS_OPAQUE,
  PASS_TRANSPARENT,
} from './blocks.js';
import {
  CHUNK_D,
  CHUNK_H,
  CHUNK_SECTIONS,
  CHUNK_W,
  PADDED_D,
  PADDED_SLICE,
  PADDED_VOLUME,
  PADDED_W,
  packVertex,
  sectionPairBit,
} from './chunk_data.js';

// Growable Uint32 buffer collector for 8-byte vertices (8 uint32s = 4 vertices per quad)
class VertexStream {
  constructor(initialUint32s = 8192) {
    this.data = new Uint32Array(initialUint32s);
    this.length = 0;
    this.quadCount = 0;
  }

  reset() {
    this.length = 0;
    this.quadCount = 0;
  }

  ensure(extra) {
    if (this.length + extra <= this.data.length) return;
    let nextLen = this.data.length * 2;
    while (nextLen < this.length + extra) nextLen *= 2;
    const grown = new Uint32Array(nextLen);
    grown.set(this.data.subarray(0, this.length));
    this.data = grown;
  }

  emitQuad(
    x0, y0, z0, u0, v0, ao0,
    x1, y1, z1, u1, v1, ao1,
    x2, y2, z2, u2, v2, ao2,
    x3, y3, z3, u3, v3, ao3,
    normal, sky, tileId, scale, flags
  ) {
    this.ensure(8);
    const out = this.data;
    let p = this.length;

    // Flip triangulation diagonal by cyclically rotating CCW vertices when AO is anisotropic
    if (ao0 + ao2 < ao1 + ao3) {
      packVertex(out, p,     x1, y1, z1, normal, ao1, sky, tileId, u1, v1, scale, flags);
      packVertex(out, p + 2, x2, y2, z2, normal, ao2, sky, tileId, u2, v2, scale, flags);
      packVertex(out, p + 4, x3, y3, z3, normal, ao3, sky, tileId, u3, v3, scale, flags);
      packVertex(out, p + 6, x0, y0, z0, normal, ao0, sky, tileId, u0, v0, scale, flags);
    } else {
      packVertex(out, p,     x0, y0, z0, normal, ao0, sky, tileId, u0, v0, scale, flags);
      packVertex(out, p + 2, x1, y1, z1, normal, ao1, sky, tileId, u1, v1, scale, flags);
      packVertex(out, p + 4, x2, y2, z2, normal, ao2, sky, tileId, u2, v2, scale, flags);
      packVertex(out, p + 6, x3, y3, z3, normal, ao3, sky, tileId, u3, v3, scale, flags);
    }

    this.length = p + 8;
    this.quadCount++;
  }

  toTransferBuffer() {
    if (this.length === 0) return new Uint32Array(0);
    return this.data.slice(0, this.length);
  }
}

// Reusable scratch buffers per worker/thread instance
const skyBuf = new Uint8Array(PADDED_VOLUME);
const bfsQueue = new Int32Array(PADDED_VOLUME);
const greedyMask = new Int32Array(CHUNK_H * CHUNK_W); // max slice is 16 x 256 = 4,096
const sectionVisited = new Uint8Array(16 * 16 * 16);
const sectionQueue = new Int32Array(16 * 16 * 16);

const streamSurface = new VertexStream(16384);
const streamUnderground = new VertexStream(8192);
const streamCutout = new VertexStream(8192);
const streamTrans = new VertexStream(4096);

function computeVertexAO(s1, s2, corner) {
  if (s1 && s2) return 0;
  return 3 - (s1 + s2 + corner);
}

/**
 * Propagates vertical sunlight down each column of the 18xH*18 padded volume
 * and runs a bounded 6-neighbor BFS flood so cave entrances and overhangs
 * receive smooth lateral sky-light falloff.
 */
function computeSkyLight(padded, topY) {
  const ceilY = Math.min(CHUNK_H - 1, topY + 2);
  skyBuf.fill(0, 0, (ceilY + 1) * PADDED_SLICE);
  if (ceilY + 1 < CHUNK_H) {
    skyBuf.fill(15, (ceilY + 1) * PADDED_SLICE, Math.min(CHUNK_H, ceilY + 3) * PADDED_SLICE);
  }

  let qHead = 0;
  let qTail = 0;

  // Pass 1: top-down column attenuation
  for (let pz = 0; pz < PADDED_D; pz++) {
    for (let px = 0; px < PADDED_W; px++) {
      let light = 15;
      const colBase = pz * PADDED_W + px;
      for (let y = ceilY; y >= 0; y--) {
        const idx = y * PADDED_SLICE + colBase;
        const b = padded[idx];
        const atten = BLOCK_LIGHT_ATTEN[b];
        if (atten >= 15) {
          light = 0;
          skyBuf[idx] = 0;
        } else {
          light = light > atten ? (light - atten) : 0;
          skyBuf[idx] = light;
        }
      }
    }
  }

  // Seed BFS queue only with lit voxels adjacent to darker non-opaque voxels
  for (let y = ceilY; y >= 0; y--) {
    const yOff = y * PADDED_SLICE;
    for (let pz = 1; pz < PADDED_D - 1; pz++) {
      const zOff = yOff + pz * PADDED_W;
      for (let px = 1; px < PADDED_W - 1; px++) {
        const idx = zOff + px;
        const lv = skyBuf[idx];
        if (lv <= 1) continue;
        if (
          (skyBuf[idx - 1] + 1 < lv && !BLOCK_OPAQUE[padded[idx - 1]]) ||
          (skyBuf[idx + 1] + 1 < lv && !BLOCK_OPAQUE[padded[idx + 1]]) ||
          (skyBuf[idx - PADDED_W] + 1 < lv && !BLOCK_OPAQUE[padded[idx - PADDED_W]]) ||
          (skyBuf[idx + PADDED_W] + 1 < lv && !BLOCK_OPAQUE[padded[idx + PADDED_W]]) ||
          (y > 0 && skyBuf[idx - PADDED_SLICE] + 1 < lv && !BLOCK_OPAQUE[padded[idx - PADDED_SLICE]])
        ) {
          bfsQueue[qTail++] = idx;
        }
      }
    }
  }

  const maxIdx = (ceilY + 1) * PADDED_SLICE;
  const neighborOffsets = [-1, 1, -PADDED_W, PADDED_W, -PADDED_SLICE, PADDED_SLICE];

  while (qHead < qTail) {
    const idx = bfsQueue[qHead++];
    const lv = skyBuf[idx];
    if (lv <= 1) continue;

    const rem = idx % PADDED_SLICE;
    const px = rem % PADDED_W;
    const pz = (rem / PADDED_W) | 0;

    for (let d = 0; d < 6; d++) {
      if (d === 0 && px === 0) continue;
      if (d === 1 && px === PADDED_W - 1) continue;
      if (d === 2 && pz === 0) continue;
      if (d === 3 && pz === PADDED_D - 1) continue;
      const nIdx = idx + neighborOffsets[d];
      if (nIdx < 0 || nIdx >= maxIdx) continue;
      const nb = padded[nIdx];
      if (BLOCK_OPAQUE[nb]) continue;
      const stepCost = 1 + BLOCK_LIGHT_ATTEN[nb];
      const nextLv = lv - stepCost;
      if (nextLv > skyBuf[nIdx]) {
        skyBuf[nIdx] = nextLv;
        if (nextLv > 1 && qTail < bfsQueue.length) {
          bfsQueue[qTail++] = nIdx;
        }
      }
    }
  }
}

/**
 * Computes 15-bit face-to-face visibility connectivity for each of the 16 vertical
 * 16x16x16 sections in a chunk, enabling Tommaso Checchi-style BFS cave culling.
 */
export function computeSectionConnectivity(padded, maxY) {
  const masks = new Uint16Array(CHUNK_SECTIONS);
  const topSection = Math.min(CHUNK_SECTIONS - 1, (maxY >> 4) + 1);

  // Sections entirely above terrain connect all 6 faces freely (all 15 bits = 0x7FFF)
  for (let sy = topSection + 1; sy < CHUNK_SECTIONS; sy++) {
    masks[sy] = 0x7fff;
  }

  for (let sy = 0; sy <= topSection; sy++) {
    const baseY = sy << 4;
    let opaqueCount = 0;
    sectionVisited.fill(0);

    for (let ly = 0; ly < 16; ly++) {
      const yOff = (baseY + ly) * PADDED_SLICE;
      for (let lz = 0; lz < 16; lz++) {
        const rowOff = yOff + (lz + 1) * PADDED_W + 1;
        for (let lx = 0; lx < 16; lx++) {
          if (BLOCK_OPAQUE[padded[rowOff + lx]]) {
            sectionVisited[(ly << 8) | (lz << 4) | lx] = 1;
            opaqueCount++;
          }
        }
      }
    }

    if (opaqueCount === 0) {
      masks[sy] = 0x7fff;
      continue;
    }
    if (opaqueCount === 4096) {
      masks[sy] = 0;
      continue;
    }

    let sectionMask = 0;
    for (let start = 0; start < 4096; start++) {
      if (sectionVisited[start]) continue;
      // Only need to flood-fill components touching at least one boundary face
      const sx = start & 15;
      const sz = (start >> 4) & 15;
      const sly = start >> 8;
      if (sx !== 0 && sx !== 15 && sz !== 0 && sz !== 15 && sly !== 0 && sly !== 15) continue;

      let touchedFaces = 0;
      let head = 0;
      let tail = 0;
      sectionVisited[start] = 1;
      sectionQueue[tail++] = start;

      while (head < tail) {
        const cur = sectionQueue[head++];
        const x = cur & 15;
        const z = (cur >> 4) & 15;
        const y = cur >> 8;

        if (x === 15) touchedFaces |= (1 << FACE_PX);
        if (x === 0)  touchedFaces |= (1 << FACE_NX);
        if (y === 15) touchedFaces |= (1 << FACE_PY);
        if (y === 0)  touchedFaces |= (1 << FACE_NY);
        if (z === 15) touchedFaces |= (1 << FACE_PZ);
        if (z === 0)  touchedFaces |= (1 << FACE_NZ);

        if (x < 15 && !sectionVisited[cur + 1])   { sectionVisited[cur + 1] = 1;   sectionQueue[tail++] = cur + 1; }
        if (x > 0  && !sectionVisited[cur - 1])   { sectionVisited[cur - 1] = 1;   sectionQueue[tail++] = cur - 1; }
        if (z < 15 && !sectionVisited[cur + 16])  { sectionVisited[cur + 16] = 1;  sectionQueue[tail++] = cur + 16; }
        if (z > 0  && !sectionVisited[cur - 16])  { sectionVisited[cur - 16] = 1;  sectionQueue[tail++] = cur - 16; }
        if (y < 15 && !sectionVisited[cur + 256]) { sectionVisited[cur + 256] = 1; sectionQueue[tail++] = cur + 256; }
        if (y > 0  && !sectionVisited[cur - 256]) { sectionVisited[cur - 256] = 1; sectionQueue[tail++] = cur - 256; }
      }

      for (let a = 0; a < 6; a++) {
        if (!(touchedFaces & (1 << a))) continue;
        for (let b = a + 1; b < 6; b++) {
          if (touchedFaces & (1 << b)) {
            sectionMask |= sectionPairBit(a, b);
          }
        }
      }
    }
    masks[sy] = sectionMask;
  }

  return masks;
}

function shouldRenderFace(blockId, neighborId) {
  if (neighborId === BLOCK.AIR || BLOCK_CROSS_QUAD[neighborId]) return true;
  if (blockId === neighborId) {
    // Cull internal faces between identical water/ice/glass blocks, keep internal leaf cluster depth sparse
    return false;
  }
  const bPass = BLOCK_PASS[blockId];
  if (bPass === PASS_OPAQUE) {
    return !BLOCK_OPAQUE[neighborId];
  }
  if (bPass === PASS_CUTOUT) {
    return !BLOCK_OPAQUE[neighborId];
  }
  // Transparent (water, ice, glass): render against air, cutout, or different transparent block
  return !BLOCK_OPAQUE[neighborId];
}

// Packs face attributes into a nonzero 30-bit integer key for greedy quad merging:
// tileId(8b) | sky(4b) | ao0(2b) | ao1(2b) | ao2(2b) | ao3(2b) | pass(2b) | flags(3b) | under(1b) | valid(1b)
function packMaskKey(tileId, sky, ao0, ao1, ao2, ao3, pass, flags, isUnder) {
  return (
    1 |
    ((tileId & 255) << 1) |
    ((sky & 15) << 9) |
    ((ao0 & 3) << 13) |
    ((ao1 & 3) << 15) |
    ((ao2 & 3) << 17) |
    ((ao3 & 3) << 19) |
    ((pass & 3) << 21) |
    ((flags & 7) << 23) |
    ((isUnder & 1) << 26)
  );
}

function selectStream(pass, isUnder) {
  if (pass === PASS_OPAQUE) return isUnder ? streamUnderground : streamSurface;
  if (pass === PASS_CUTOUT) return streamCutout;
  return streamTrans;
}

/**
 * Builds greedy-meshed, AO-lit, sky-lit packed vertex buffers for a 16x256x16 chunk
 * wrapped in an 18x256x18 halo (`padded`), splitting opaque geometry into surface vs
 * underground buffers for zero-cost cave skipping when above ground.
 */
export function buildChunkMesh(padded, maxYHint = 255, minSurfaceYHint = 48) {
  streamSurface.reset();
  streamUnderground.reset();
  streamCutout.reset();
  streamTrans.reset();

  const scanMaxY = Math.min(CHUNK_H - 2, Math.max(1, maxYHint + 1));
  const undergroundCutoff = Math.max(6, minSurfaceYHint - 5);

  computeSkyLight(padded, scanMaxY);
  const sectionMasks = computeSectionConnectivity(padded, scanMaxY);

  let minMeshY = CHUNK_H;
  let maxMeshY = 0;

  // Pass A: Emit diagonal X-quads for cross-quad flora (e.g. tall_grass)
  for (let y = 1; y <= scanMaxY; y++) {
    const yOff = y * PADDED_SLICE;
    for (let z = 0; z < CHUNK_D; z++) {
      const rowOff = yOff + (z + 1) * PADDED_W + 1;
      for (let x = 0; x < CHUNK_W; x++) {
        const b = padded[rowOff + x];
        if (!BLOCK_CROSS_QUAD[b]) continue;
        const sky = skyBuf[rowOff + x];
        const tileId = BLOCK_FACE_TILES[b * 6 + FACE_PZ];
        if (y < minMeshY) minMeshY = y;
        if (y + 1 > maxMeshY) maxMeshY = y + 1;

        // Cross-quad diagonal A (flags = 3) and diagonal B (flags = 4); shader offsets corners inward by 0.146
        streamCutout.emitQuad(
          x,     y,     z,     0, 1, 3,
          x + 1, y,     z + 1, 1, 1, 3,
          x + 1, y + 1, z + 1, 1, 0, 3,
          x,     y + 1, z,     0, 0, 3,
          FACE_PY, sky, tileId, 1, 3
        );
        streamCutout.emitQuad(
          x,     y,     z + 1, 0, 1, 3,
          x + 1, y,     z,     1, 1, 3,
          x + 1, y + 1, z,     1, 0, 3,
          x,     y + 1, z + 1, 0, 0, 3,
          FACE_PY, sky, tileId, 1, 4
        );
      }
    }
  }

  // Pass B: 6-direction greedy meshing over solid/liquid block faces
  // 1) +Y and -Y horizontal slices (u = x in 0..16, v = z in 0..16)
  for (let dirIdx = 0; dirIdx < 2; dirIdx++) {
    const face = dirIdx === 0 ? FACE_PY : FACE_NY;
    const nyStep = dirIdx === 0 ? PADDED_SLICE : -PADDED_SLICE;

    for (let y = 0; y <= scanMaxY; y++) {
      if (face === FACE_NY && y === 0) continue; // Cull bottom of world Y=0 bedrock
      let count = 0;
      const yOff = y * PADDED_SLICE;

      for (let z = 0; z < CHUNK_D; z++) {
        const pz = z + 1;
        const rowOff = yOff + pz * PADDED_W;
        for (let x = 0; x < CHUNK_W; x++) {
          const px = x + 1;
          const idx = rowOff + px;
          const b = padded[idx];
          if (b === BLOCK.AIR || BLOCK_CROSS_QUAD[b]) {
            greedyMask[z * 16 + x] = 0;
            continue;
          }
          const nIdx = idx + nyStep;
          const nb = padded[nIdx];
          if (!shouldRenderFace(b, nb)) {
            greedyMask[z * 16 + x] = 0;
            continue;
          }

          const sky = skyBuf[nIdx];
          let ao0 = 3, ao1 = 3, ao2 = 3, ao3 = 3;
          if (BLOCK_PASS[b] !== PASS_TRANSPARENT) {
            const oN  = BLOCK_AO_OCCLUDER[padded[nIdx - PADDED_W]];
            const oS  = BLOCK_AO_OCCLUDER[padded[nIdx + PADDED_W]];
            const oW  = BLOCK_AO_OCCLUDER[padded[nIdx - 1]];
            const oE  = BLOCK_AO_OCCLUDER[padded[nIdx + 1]];
            const oNW = BLOCK_AO_OCCLUDER[padded[nIdx - PADDED_W - 1]];
            const oNE = BLOCK_AO_OCCLUDER[padded[nIdx - PADDED_W + 1]];
            const oSW = BLOCK_AO_OCCLUDER[padded[nIdx + PADDED_W - 1]];
            const oSE = BLOCK_AO_OCCLUDER[padded[nIdx + PADDED_W + 1]];

            if (face === FACE_PY) {
              // v0=(x,z+h)[SW], v1=(x+w,z+h)[SE], v2=(x+w,z)[NE], v3=(x,z)[NW]
              ao0 = computeVertexAO(oW, oS, oSW);
              ao1 = computeVertexAO(oE, oS, oSE);
              ao2 = computeVertexAO(oE, oN, oNE);
              ao3 = computeVertexAO(oW, oN, oNW);
            } else {
              // v0=(x,z)[NW], v1=(x+w,z)[NE], v2=(x+w,z+h)[SE], v3=(x,z+h)[SW]
              ao0 = computeVertexAO(oW, oN, oNW);
              ao1 = computeVertexAO(oE, oN, oNE);
              ao2 = computeVertexAO(oE, oS, oSE);
              ao3 = computeVertexAO(oW, oS, oSW);
            }
          }

          const fIdx = b * 6 + face;
          const isUnder = y < undergroundCutoff ? 1 : 0;
          greedyMask[z * 16 + x] = packMaskKey(
            BLOCK_FACE_TILES[fIdx], sky, ao0, ao1, ao2, ao3,
            BLOCK_PASS[b], BLOCK_FLAGS[fIdx], isUnder
          );
          count++;
        }
      }

      if (count === 0) continue;
      const planeY = face === FACE_PY ? y + 1 : y;
      if (y < minMeshY) minMeshY = y;
      if (y + 1 > maxMeshY) maxMeshY = y + 1;

      for (let v = 0; v < 16; v++) {
        for (let u = 0; u < 16; ) {
          const key = greedyMask[v * 16 + u];
          if (key === 0) { u++; continue; }

          let w = 1;
          while (u + w < 16 && greedyMask[v * 16 + u + w] === key) w++;

          let h = 1;
          outerY: while (v + h < 16) {
            const rOff = (v + h) * 16 + u;
            for (let k = 0; k < w; k++) {
              if (greedyMask[rOff + k] !== key) break outerY;
            }
            h++;
          }

          for (let dv = 0; dv < h; dv++) {
            const rOff = (v + dv) * 16 + u;
            for (let du = 0; du < w; du++) greedyMask[rOff + du] = 0;
          }

          const tileId = (key >>> 1) & 255;
          const sky = (key >>> 9) & 15;
          const ao0 = (key >>> 13) & 3;
          const ao1 = (key >>> 15) & 3;
          const ao2 = (key >>> 17) & 3;
          const ao3 = (key >>> 19) & 3;
          const pass = (key >>> 21) & 3;
          const flags = (key >>> 23) & 7;
          const isUnder = (key >>> 26) & 1;
          const stream = selectStream(pass, isUnder);

          if (face === FACE_PY) {
            stream.emitQuad(
              u,     planeY, v + h, 0, h, ao0,
              u + w, planeY, v + h, w, h, ao1,
              u + w, planeY, v,     w, 0, ao2,
              u,     planeY, v,     0, 0, ao3,
              FACE_PY, sky, tileId, 1, flags
            );
          } else {
            stream.emitQuad(
              u,     planeY, v,     0, 0, ao0,
              u + w, planeY, v,     w, 0, ao1,
              u + w, planeY, v + h, w, h, ao2,
              u,     planeY, v + h, 0, h, ao3,
              FACE_NY, sky, tileId, 1, flags
            );
          }
          u += w;
        }
      }
    }
  }

  // 2) +Z and -Z vertical slices (u = x in 0..16, v = y in 0..scanMaxY)
  const sliceH = scanMaxY + 1;
  for (let dirIdx = 0; dirIdx < 2; dirIdx++) {
    const face = dirIdx === 0 ? FACE_PZ : FACE_NZ;
    const nzStep = dirIdx === 0 ? PADDED_W : -PADDED_W;

    for (let z = 0; z < CHUNK_D; z++) {
      let count = 0;
      const pz = z + 1;

      for (let y = 0; y < sliceH; y++) {
        const rowOff = y * PADDED_SLICE + pz * PADDED_W;
        for (let x = 0; x < CHUNK_W; x++) {
          const idx = rowOff + x + 1;
          const b = padded[idx];
          if (b === BLOCK.AIR || BLOCK_CROSS_QUAD[b]) {
            greedyMask[y * 16 + x] = 0;
            continue;
          }
          const nIdx = idx + nzStep;
          const nb = padded[nIdx];
          if (!shouldRenderFace(b, nb)) {
            greedyMask[y * 16 + x] = 0;
            continue;
          }

          const sky = skyBuf[nIdx];
          let ao0 = 3, ao1 = 3, ao2 = 3, ao3 = 3;
          if (BLOCK_PASS[b] !== PASS_TRANSPARENT) {
            const oW  = BLOCK_AO_OCCLUDER[padded[nIdx - 1]];
            const oE  = BLOCK_AO_OCCLUDER[padded[nIdx + 1]];
            const oD  = y > 0 ? BLOCK_AO_OCCLUDER[padded[nIdx - PADDED_SLICE]] : 1;
            const oU  = BLOCK_AO_OCCLUDER[padded[nIdx + PADDED_SLICE]];
            const oDW = y > 0 ? BLOCK_AO_OCCLUDER[padded[nIdx - PADDED_SLICE - 1]] : 1;
            const oDE = y > 0 ? BLOCK_AO_OCCLUDER[padded[nIdx - PADDED_SLICE + 1]] : 1;
            const oUW = BLOCK_AO_OCCLUDER[padded[nIdx + PADDED_SLICE - 1]];
            const oUE = BLOCK_AO_OCCLUDER[padded[nIdx + PADDED_SLICE + 1]];

            if (face === FACE_PZ) {
              // v0=(x,y)[DW], v1=(x+w,y)[DE], v2=(x+w,y+h)[UE], v3=(x,y+h)[UW]
              ao0 = computeVertexAO(oW, oD, oDW);
              ao1 = computeVertexAO(oE, oD, oDE);
              ao2 = computeVertexAO(oE, oU, oUE);
              ao3 = computeVertexAO(oW, oU, oUW);
            } else {
              // v0=(x+w,y)[DE], v1=(x,y)[DW], v2=(x,y+h)[UW], v3=(x+w,y+h)[UE]
              ao0 = computeVertexAO(oE, oD, oDE);
              ao1 = computeVertexAO(oW, oD, oDW);
              ao2 = computeVertexAO(oW, oU, oUW);
              ao3 = computeVertexAO(oE, oU, oUE);
            }
          }

          const fIdx = b * 6 + face;
          // Do not merge grass_side/snow_grass_side vertically, because their texture only tiles horizontally!
          const isSideCap = (b === BLOCK.GRASS || b === BLOCK.SNOW_GRASS);
          const isUnder = y < undergroundCutoff ? 1 : 0;
          greedyMask[y * 16 + x] = packMaskKey(
            BLOCK_FACE_TILES[fIdx], sky, ao0, ao1, ao2, ao3,
            BLOCK_PASS[b], BLOCK_FLAGS[fIdx] | (isSideCap ? 4 : 0), isUnder
          );
          count++;
        }
      }

      if (count === 0) continue;
      const planeZ = face === FACE_PZ ? z + 1 : z;

      for (let v = 0; v < sliceH; v++) {
        for (let u = 0; u < 16; ) {
          const key = greedyMask[v * 16 + u];
          if (key === 0) { u++; continue; }

          let w = 1;
          while (u + w < 16 && greedyMask[v * 16 + u + w] === key) w++;

          const rawFlags = (key >>> 23) & 7;
          const noVertMerge = (rawFlags & 4) !== 0;
          let h = 1;
          if (!noVertMerge) {
            outerZ: while (v + h < sliceH) {
              const rOff = (v + h) * 16 + u;
              for (let k = 0; k < w; k++) {
                if (greedyMask[rOff + k] !== key) break outerZ;
              }
              h++;
            }
          }

          for (let dv = 0; dv < h; dv++) {
            const rOff = (v + dv) * 16 + u;
            for (let du = 0; du < w; du++) greedyMask[rOff + du] = 0;
          }

          if (v < minMeshY) minMeshY = v;
          if (v + h > maxMeshY) maxMeshY = v + h;

          const tileId = (key >>> 1) & 255;
          const sky = (key >>> 9) & 15;
          const ao0 = (key >>> 13) & 3;
          const ao1 = (key >>> 15) & 3;
          const ao2 = (key >>> 17) & 3;
          const ao3 = (key >>> 19) & 3;
          const pass = (key >>> 21) & 3;
          const flags = rawFlags & 3;
          const isUnder = (key >>> 26) & 1;
          const stream = selectStream(pass, isUnder);

          if (face === FACE_PZ) {
            stream.emitQuad(
              u,     v,     planeZ, 0, h, ao0,
              u + w, v,     planeZ, w, h, ao1,
              u + w, v + h, planeZ, w, 0, ao2,
              u,     v + h, planeZ, 0, 0, ao3,
              FACE_PZ, sky, tileId, 1, flags
            );
          } else {
            stream.emitQuad(
              u + w, v,     planeZ, 0, h, ao0,
              u,     v,     planeZ, w, h, ao1,
              u,     v + h, planeZ, w, 0, ao2,
              u + w, v + h, planeZ, 0, 0, ao3,
              FACE_NZ, sky, tileId, 1, flags
            );
          }
          u += w;
        }
      }
    }
  }

  // 3) +X and -X vertical slices (u = z in 0..16, v = y in 0..scanMaxY)
  for (let dirIdx = 0; dirIdx < 2; dirIdx++) {
    const face = dirIdx === 0 ? FACE_PX : FACE_NX;
    const nxStep = dirIdx === 0 ? 1 : -1;

    for (let x = 0; x < CHUNK_W; x++) {
      let count = 0;
      const px = x + 1;

      for (let y = 0; y < sliceH; y++) {
        const yOff = y * PADDED_SLICE + px;
        for (let z = 0; z < CHUNK_D; z++) {
          const idx = yOff + (z + 1) * PADDED_W;
          const b = padded[idx];
          if (b === BLOCK.AIR || BLOCK_CROSS_QUAD[b]) {
            greedyMask[y * 16 + z] = 0;
            continue;
          }
          const nIdx = idx + nxStep;
          const nb = padded[nIdx];
          if (!shouldRenderFace(b, nb)) {
            greedyMask[y * 16 + z] = 0;
            continue;
          }

          const sky = skyBuf[nIdx];
          let ao0 = 3, ao1 = 3, ao2 = 3, ao3 = 3;
          if (BLOCK_PASS[b] !== PASS_TRANSPARENT) {
            const oN  = BLOCK_AO_OCCLUDER[padded[nIdx - PADDED_W]];
            const oS  = BLOCK_AO_OCCLUDER[padded[nIdx + PADDED_W]];
            const oD  = y > 0 ? BLOCK_AO_OCCLUDER[padded[nIdx - PADDED_SLICE]] : 1;
            const oU  = BLOCK_AO_OCCLUDER[padded[nIdx + PADDED_SLICE]];
            const oDN = y > 0 ? BLOCK_AO_OCCLUDER[padded[nIdx - PADDED_SLICE - PADDED_W]] : 1;
            const oDS = y > 0 ? BLOCK_AO_OCCLUDER[padded[nIdx - PADDED_SLICE + PADDED_W]] : 1;
            const oUN = BLOCK_AO_OCCLUDER[padded[nIdx + PADDED_SLICE - PADDED_W]];
            const oUS = BLOCK_AO_OCCLUDER[padded[nIdx + PADDED_SLICE + PADDED_W]];

            if (face === FACE_PX) {
              // v0=(z+w,y)[DS], v1=(z,y)[DN], v2=(z,y+h)[UN], v3=(z+w,y+h)[US]
              ao0 = computeVertexAO(oS, oD, oDS);
              ao1 = computeVertexAO(oN, oD, oDN);
              ao2 = computeVertexAO(oN, oU, oUN);
              ao3 = computeVertexAO(oS, oU, oUS);
            } else {
              // v0=(z,y)[DN], v1=(z+w,y)[DS], v2=(z+w,y+h)[US], v3=(z,y+h)[UN]
              ao0 = computeVertexAO(oN, oD, oDN);
              ao1 = computeVertexAO(oS, oD, oDS);
              ao2 = computeVertexAO(oS, oU, oUS);
              ao3 = computeVertexAO(oN, oU, oUN);
            }
          }

          const fIdx = b * 6 + face;
          const isSideCap = (b === BLOCK.GRASS || b === BLOCK.SNOW_GRASS);
          const isUnder = y < undergroundCutoff ? 1 : 0;
          greedyMask[y * 16 + z] = packMaskKey(
            BLOCK_FACE_TILES[fIdx], sky, ao0, ao1, ao2, ao3,
            BLOCK_PASS[b], BLOCK_FLAGS[fIdx] | (isSideCap ? 4 : 0), isUnder
          );
          count++;
        }
      }

      if (count === 0) continue;
      const planeX = face === FACE_PX ? x + 1 : x;

      for (let v = 0; v < sliceH; v++) {
        for (let u = 0; u < 16; ) {
          const key = greedyMask[v * 16 + u];
          if (key === 0) { u++; continue; }

          let w = 1;
          while (u + w < 16 && greedyMask[v * 16 + u + w] === key) w++;

          const rawFlags = (key >>> 23) & 7;
          const noVertMerge = (rawFlags & 4) !== 0;
          let h = 1;
          if (!noVertMerge) {
            outerX: while (v + h < sliceH) {
              const rOff = (v + h) * 16 + u;
              for (let k = 0; k < w; k++) {
                if (greedyMask[rOff + k] !== key) break outerX;
              }
              h++;
            }
          }

          for (let dv = 0; dv < h; dv++) {
            const rOff = (v + dv) * 16 + u;
            for (let du = 0; du < w; du++) greedyMask[rOff + du] = 0;
          }

          if (v < minMeshY) minMeshY = v;
          if (v + h > maxMeshY) maxMeshY = v + h;

          const tileId = (key >>> 1) & 255;
          const sky = (key >>> 9) & 15;
          const ao0 = (key >>> 13) & 3;
          const ao1 = (key >>> 15) & 3;
          const ao2 = (key >>> 17) & 3;
          const ao3 = (key >>> 19) & 3;
          const pass = (key >>> 21) & 3;
          const flags = rawFlags & 3;
          const isUnder = (key >>> 26) & 1;
          const stream = selectStream(pass, isUnder);

          if (face === FACE_PX) {
            stream.emitQuad(
              planeX, v,     u + w, 0, h, ao0,
              planeX, v,     u,     w, h, ao1,
              planeX, v + h, u,     w, 0, ao2,
              planeX, v + h, u + w, 0, 0, ao3,
              FACE_PX, sky, tileId, 1, flags
            );
          } else {
            stream.emitQuad(
              planeX, v,     u,     0, h, ao0,
              planeX, v,     u + w, w, h, ao1,
              planeX, v + h, u + w, w, 0, ao2,
              planeX, v + h, u,     0, 0, ao3,
              FACE_NX, sky, tileId, 1, flags
            );
          }
          u += w;
        }
      }
    }
  }

  if (minMeshY > maxMeshY) {
    minMeshY = 0;
    maxMeshY = 1;
  }

  return {
    opaqueSurface: streamSurface.toTransferBuffer(),
    opaqueUnderground: streamUnderground.toTransferBuffer(),
    cutout: streamCutout.toTransferBuffer(),
    transparent: streamTrans.toTransferBuffer(),
    sectionMasks,
    minY: minMeshY,
    maxY: maxMeshY,
  };
}

/**
 * Fast far-distance LOD heightfield voxel mesher (`gridN x gridN` cells at `stride` blocks,
 * e.g. `16x16` cells at `stride = 4` covering a `4x4` merged 64x64 super-chunk in 1 draw call!).
 * Expects `heights` and `topBlocks` of size `(gridN + 2) * (gridN + 2)` (1-cell halo).
 */
export function buildHeightfieldLODMesh(heights, topBlocks, subBlocks, waterLevel, gridN, stride) {
  streamSurface.reset();
  streamTrans.reset();

  const pitch = gridN + 2;
  let minY = 255;
  let maxY = 1;
  let waterQuads = 0;

  // 1. Greedy-merge coplanar (height, topBlock) top faces across the gridN x gridN LOD cell
  for (let gz = 0; gz < gridN; gz++) {
    for (let gx = 0; gx < gridN; gx++) {
      const idx = (gz + 1) * pitch + (gx + 1);
      const h = heights[idx];
      const topB = topBlocks[idx];
      if (h < minY) minY = h;
      if (h + stride > maxY) maxY = h + stride;
      if (h < waterLevel) waterQuads++;
      greedyMask[gz * gridN + gx] = 1 | ((h & 511) << 1) | ((topB & 255) << 10);
    }
  }

  for (let v = 0; v < gridN; v++) {
    for (let u = 0; u < gridN; ) {
      const key = greedyMask[v * gridN + u];
      if (key === 0) { u++; continue; }
      let w = 1;
      while (u + w < gridN && greedyMask[v * gridN + u + w] === key) w++;
      let hSpan = 1;
      outerTop: while (v + hSpan < gridN) {
        for (let k = 0; k < w; k++) {
          if (greedyMask[(v + hSpan) * gridN + u + k] !== key) break outerTop;
        }
        hSpan++;
      }
      for (let dv = 0; dv < hSpan; dv++) {
        for (let du = 0; du < w; du++) greedyMask[(v + dv) * gridN + u + du] = 0;
      }

      const h = (key >>> 1) & 511;
      const topB = (key >>> 10) & 255;
      const topTile = BLOCK_FACE_TILES[topB * 6 + FACE_PY];
      const topFlag = BLOCK_FLAGS[topB * 6 + FACE_PY];
      const sky = h >= waterLevel - 1 ? 15 : Math.max(5, 15 - ((waterLevel - h) >> 1));

      streamSurface.emitQuad(
        u,     h, v + hSpan, 0,          hSpan * stride, 3,
        u + w, h, v + hSpan, w * stride, hSpan * stride, 3,
        u + w, h, v,         w * stride, 0,              3,
        u,     h, v,         0,          0,              3,
        FACE_PY, sky, topTile, stride, topFlag
      );
      u += w;
    }
  }

  // 2. Emit vertical step skirts only where adjacent cells differ in height
  for (let gz = 0; gz < gridN; gz++) {
    for (let gx = 0; gx < gridN; gx++) {
      const idx = (gz + 1) * pitch + (gx + 1);
      const h = heights[idx];
      const subB = subBlocks[idx];
      const x0 = gx;
      const x1 = gx + 1;
      const z0 = gz;
      const z1 = gz + 1;
      const sky = h >= waterLevel - 1 ? 15 : Math.max(5, 15 - ((waterLevel - h) >> 1));

      const hE = (gx === gridN - 1) ? Math.max(4, heights[idx + 1] - stride * 3) : heights[idx + 1];
      const hW = (gx === 0)         ? Math.max(4, heights[idx - 1] - stride * 3) : heights[idx - 1];
      const hS = (gz === gridN - 1) ? Math.max(4, heights[idx + pitch] - stride * 3) : heights[idx + pitch];
      const hN = (gz === 0)         ? Math.max(4, heights[idx - pitch] - stride * 3) : heights[idx - pitch];
      const sideTile = BLOCK_FACE_TILES[subB * 6 + FACE_PX];

      if (hE < h) {
        const dy = h - hE;
        streamSurface.emitQuad(
          x1, hE, z1, 0, dy, 2,
          x1, hE, z0, stride, dy, 2,
          x1, h,  z0, stride, 0, 3,
          x1, h,  z1, 0, 0, 3,
          FACE_PX, sky, sideTile, stride, 0
        );
      }
      if (hW < h) {
        const dy = h - hW;
        streamSurface.emitQuad(
          x0, hW, z0, 0, dy, 2,
          x0, hW, z1, stride, dy, 2,
          x0, h,  z1, stride, 0, 3,
          x0, h,  z0, 0, 0, 3,
          FACE_NX, sky, sideTile, stride, 0
        );
      }
      if (hS < h) {
        const dy = h - hS;
        streamSurface.emitQuad(
          x0, hS, z1, 0, dy, 2,
          x1, hS, z1, stride, dy, 2,
          x1, h,  z1, stride, 0, 3,
          x0, h,  z1, 0, 0, 3,
          FACE_PZ, sky, sideTile, stride, 0
        );
      }
      if (hN < h) {
        const dy = h - hN;
        streamSurface.emitQuad(
          x1, hN, z0, 0, dy, 2,
          x0, hN, z0, stride, dy, 2,
          x0, h,  z0, stride, 0, 3,
          x1, h,  z0, 0, 0, 3,
          FACE_NZ, sky, sideTile, stride, 0
        );
      }
    }
  }

  // Greedy-merge water cells across the LOD grid
  if (waterQuads > 0) {
    if (waterLevel > maxY) maxY = waterLevel + 1;
    const waterTile = BLOCK_FACE_TILES[BLOCK.WATER * 6 + FACE_PY];
    const iceTile = BLOCK_FACE_TILES[BLOCK.ICE * 6 + FACE_PY];

    for (let gz = 0; gz < gridN; gz++) {
      for (let gx = 0; gx < gridN; gx++) {
        const idx = (gz + 1) * pitch + (gx + 1);
        if (heights[idx] < waterLevel) {
          const isIce = topBlocks[idx] === BLOCK.ICE;
          greedyMask[gz * gridN + gx] = isIce ? 2 : 1;
        } else {
          greedyMask[gz * gridN + gx] = 0;
        }
      }
    }

    for (let v = 0; v < gridN; v++) {
      for (let u = 0; u < gridN; ) {
        const key = greedyMask[v * gridN + u];
        if (key === 0) { u++; continue; }
        let w = 1;
        while (u + w < gridN && greedyMask[v * gridN + u + w] === key) w++;
        let h = 1;
        outerW: while (v + h < gridN) {
          for (let k = 0; k < w; k++) {
            if (greedyMask[(v + h) * gridN + u + k] !== key) break outerW;
          }
          h++;
        }
        for (let dv = 0; dv < h; dv++) {
          for (let du = 0; du < w; du++) greedyMask[(v + dv) * gridN + u + du] = 0;
        }

        const isIce = key === 2;
        streamTrans.emitQuad(
          u,     waterLevel, v + h, 0, h * stride, 3,
          u + w, waterLevel, v + h, w * stride, h * stride, 3,
          u + w, waterLevel, v,     w * stride, 0, 3,
          u,     waterLevel, v,     0, 0, 3,
          FACE_PY, 15, isIce ? iceTile : waterTile, stride, isIce ? 0 : 2
        );
        u += w;
      }
    }
  }

  return {
    opaqueSurface: streamSurface.toTransferBuffer(),
    transparent: streamTrans.toTransferBuffer(),
    minY: Math.max(0, minY - 4),
    maxY: Math.min(255, maxY + 1),
  };
}
