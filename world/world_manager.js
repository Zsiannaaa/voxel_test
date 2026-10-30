// Infinite Streamed World Manager:
// - Pool of Terrain Workers (`terrain.worker.js`) + Pool of Mesher Workers (`mesher.worker.js`)
// - Quadtree 3-Tier LOD hierarchy (LOD0 16x16 full 3D voxel chunks -> LOD1 32x32 2x-stride
//   super-chunks -> LOD2 64x64 4x-stride super-chunks) enabling Render Distance 16..36+
// - View-direction & distance prioritized streaming + 2.0 ms/frame GPU upload budget
// - Frustum culling + 16x16x16 Section BFS cave connectivity graph (skips sealed underground caves)
// - Recycled 81KB halo transfer buffers & zero per-frame array allocations in the render loop.

import { BLOCK } from '../chunk/blocks.js';
import {
  CHUNK_H,
  CHUNK_SECTIONS,
  PADDED_SLICE,
  PADDED_VOLUME,
  PADDED_W,
  chunkKey,
  oppositeFace,
  sectionPairBit,
  voxelIndex,
} from '../chunk/chunk_data.js';
import { buildChunkMesh } from '../chunk/mesh_builder.js';
import { extractFrustumPlanes, frustumIntersectsAABB } from '../renderer/math3d.js';
import { BIOME_NAMES, BiomeField } from './biome.js';

const BFS_GRID_RADIUS = 18;
const BFS_GRID_SPAN = BFS_GRID_RADIUS * 2 + 1; // 37 x 37 x 16 = 21,904 sections
const BFS_MAX_NODES = BFS_GRID_SPAN * BFS_GRID_SPAN * CHUNK_SECTIONS;

// Packs (cx, cz) into a 30-bit non-negative V8 Small Integer (Smi) -> zero heap/string allocations!
function packChunkInt(cx, cz) {
  return ((cx + 16384) << 15) | ((cz + 16384) & 0x7fff);
}

export class WorldManager {
  constructor(renderer, storage, seed = 2026) {
    this.renderer = renderer;
    this.storage = storage;
    this.seed = seed;
    this.biomeField = new BiomeField(seed);
    this.biomeProbe = {};

    // Active LOD-0 full chunks: Map<"cx,cz", ChunkRecord> + Smi integer index Map<intKey, ChunkRecord>
    this.chunks = new Map();
    this.chunksByInt = new Map();
    // Pre-allocated 37x37 local chunk pointer cache for zero-allocation BFS & frustum pre-test
    this.bfsChunkGrid = new Array(BFS_GRID_SPAN * BFS_GRID_SPAN).fill(null);
    this.bfsColInFrustum = new Uint8Array(BFS_GRID_SPAN * BFS_GRID_SPAN);
    // Active LOD-1 (2x2 chunks = 32x32 blocks) & LOD-2 (4x4 chunks = 64x64 blocks) super-chunks
    this.lodMeshes = new Map();

    // Worker pools
    const cores = Math.max(2, Math.min(4, (navigator.hardwareConcurrency || 4) >> 1));
    this.terrainWorkers = [];
    this.mesherWorkers = [];
    this.terrainBusy = new Int32Array(cores);
    this.mesherBusy = new Int32Array(cores);

    for (let i = 0; i < cores; i++) {
      const tw = new Worker(new URL('./terrain.worker.js', import.meta.url), { type: 'module' });
      tw.onmessage = (ev) => this.onTerrainWorkerMessage(i, ev.data);
      this.terrainWorkers.push(tw);

      const mw = new Worker(new URL('./mesher.worker.js', import.meta.url), { type: 'module' });
      mw.onmessage = (ev) => this.onMesherWorkerMessage(i, ev.data);
      this.mesherWorkers.push(mw);
    }

    this.nextJobId = 1;
    this.pendingTerrain = new Set();
    this.pendingLOD = new Set();
    this.terrainQueue = [];
    this.lodQueue = [];
    this.meshQueue = [];
    this.uploadQueue = [];

    // Recycled 82,944-byte padded halo buffers returned by mesher workers
    this.haloPool = [];

    // Pre-allocated render draw lists & pooled draw-item structs (zero per-frame heap allocations!)
    this.opaqueDraws = [];
    this.cutoutDraws = [];
    this.transparentDraws = [];
    this.drawItemPool = [];
    this.drawPoolCursor = 0;
    this.wantedLOD0 = new Set();
    this.wantedLOD = new Set();
    this.frustumPlanes = new Float32Array(24);

    // Pre-allocated zero-clear stamp BFS buffers for 16x16x16 section cave visibility culling
    this.bfsVisited = new Uint32Array(BFS_MAX_NODES);
    this.bfsQueueNode = new Int32Array(BFS_MAX_NODES);
    this.bfsQueueEntryDir = new Int8Array(BFS_MAX_NODES); // -1 for start, 0..5 for entry face
    this.bfsQueueAxisMask = new Uint8Array(BFS_MAX_NODES); // committed outgoing directions bitmask
    this.bfsStamp = 1;

    this.lastQueueCX = 1e9;
    this.lastQueueCZ = 1e9;
    this.lastQueueRD = -1;
    this.lastFwdX = 0;
    this.lastFwdZ = -1;
    this.frameCounter = 0;

    // Telemetry stats for F3 overlay & profiler
    this.stats = {
      loadedChunks: 0,
      loadedLOD: 0,
      visibleChunks: 0,
      visibleLOD: 0,
      culledUnderground: 0,
      pendingJobs: 0,
      uploadMs: 0,
      cullMs: 0,
    };
  }

  getBiomeNameAt(wx, wz) {
    const s = this.biomeField.sampleColumn(Math.floor(wx), Math.floor(wz), this.biomeProbe);
    return BIOME_NAMES[s.biome] || 'Plains';
  }

  getSurfaceHeightAt(wx, wz) {
    const s = this.biomeField.sampleColumn(Math.floor(wx), Math.floor(wz), this.biomeProbe);
    return Math.max(s.height, s.waterLevel);
  }

  getBlock(wx, wy, wz) {
    if (wy < 0 || wy >= CHUNK_H) return BLOCK.AIR;
    const chunk = this.chunksByInt.get(packChunkInt(wx >> 4, wz >> 4));
    if (!chunk || !chunk.voxels) return BLOCK.AIR;
    return chunk.voxels[voxelIndex(wx & 15, wy, wz & 15)];
  }

  isChunkLoadedAt(wx, wz) {
    return this.chunksByInt.has(packChunkInt(wx >> 4, wz >> 4));
  }

  /**
   * Player block edit (`setBlock`): updates voxel immediately, records sparse delta to IndexedDB,
   * and performs synchronous (< 1.2 ms) immediate remesh of affected chunk (+ boundary neighbor)
   * so block breaking/placing responds on the very next frame with zero latency!
   */
  setBlock(wx, wy, wz, blockId) {
    if (wy < 0 || wy >= CHUNK_H) return false;
    const cx = wx >> 4;
    const cz = wz >> 4;
    const key = chunkKey(cx, cz);
    const chunk = this.chunks.get(key);
    if (!chunk || !chunk.voxels) return false;

    const lx = wx & 15;
    const lz = wz & 15;
    const vIdx = voxelIndex(lx, wy, lz);
    if (chunk.voxels[vIdx] === blockId) return false;

    chunk.voxels[vIdx] = blockId;
    if (wy > chunk.maxY) chunk.maxY = wy;
    if (blockId === BLOCK.AIR && wy < chunk.minSurfaceY) {
      chunk.minSurfaceY = Math.max(1, wy - 2);
    }
    chunk.rev++;

    if (this.storage) {
      this.storage.recordEdit(key, vIdx, blockId);
    }

    // Immediate next-frame remesh for snappy gameplay response
    this.remeshChunkImmediate(chunk);

    if (lx === 0)  this.remeshNeighborImmediate(cx - 1, cz);
    if (lx === 15) this.remeshNeighborImmediate(cx + 1, cz);
    if (lz === 0)  this.remeshNeighborImmediate(cx, cz - 1);
    if (lz === 15) this.remeshNeighborImmediate(cx, cz + 1);
    // Corner AO update when editing a corner voxel
    if (lx === 0  && lz === 0)  this.remeshNeighborImmediate(cx - 1, cz - 1);
    if (lx === 15 && lz === 0)  this.remeshNeighborImmediate(cx + 1, cz - 1);
    if (lx === 0  && lz === 15) this.remeshNeighborImmediate(cx - 1, cz + 1);
    if (lx === 15 && lz === 15) this.remeshNeighborImmediate(cx + 1, cz + 1);

    return true;
  }

  remeshNeighborImmediate(cx, cz) {
    const nb = this.chunks.get(chunkKey(cx, cz));
    if (nb && nb.voxels) {
      nb.rev++;
      this.remeshChunkImmediate(nb);
    }
  }

  remeshChunkImmediate(chunk) {
    const padded = this.acquireHaloBuffer();
    const scanMax = this.fillPaddedHalo(chunk, padded);
    const res = buildChunkMesh(padded, scanMax, chunk.minSurfaceY);
    this.releaseHaloBuffer(padded);

    chunk.meshSurface = this.renderer.createOrUpdateMeshBuffer(res.opaqueSurface, chunk.meshSurface);
    chunk.meshUnderground = this.renderer.createOrUpdateMeshBuffer(res.opaqueUnderground, chunk.meshUnderground);
    chunk.meshCutout = this.renderer.createOrUpdateMeshBuffer(res.cutout, chunk.meshCutout);
    chunk.meshTrans = this.renderer.createOrUpdateMeshBuffer(res.transparent, chunk.meshTrans);
    chunk.sectionMasks = res.sectionMasks;
    chunk.meshMinY = res.minY;
    chunk.meshMaxY = res.maxY;
    chunk.meshedRev = chunk.rev;
    chunk.hasMesh = true;
    chunk.fade = 1.0;
  }

  acquireHaloBuffer() {
    return this.haloPool.length > 0 ? this.haloPool.pop() : new Uint8Array(PADDED_VOLUME);
  }

  releaseHaloBuffer(buf) {
    if (buf && buf.byteLength === PADDED_VOLUME && this.haloPool.length < 24) {
      this.haloPool.push(buf);
    }
  }

  /**
   * Copies chunk's 16xH*16 voxels + 1-voxel halo from all 8 surrounding neighbors into `padded`.
   * Where a neighbor at the LOD0 outer border isn't loaded, extrudes the chunk's own underground
   * column up to the true 2D terrain height of that neighbor column so zero underground border walls
   * are generated!
   */
  fillPaddedHalo(chunk, padded) {
    const cx = chunk.cx;
    const cz = chunk.cz;
    const nW = this.chunks.get(chunkKey(cx - 1, cz));
    const nE = this.chunks.get(chunkKey(cx + 1, cz));
    const nN = this.chunks.get(chunkKey(cx, cz - 1));
    const nS = this.chunks.get(chunkKey(cx, cz + 1));
    const nNW = this.chunks.get(chunkKey(cx - 1, cz - 1));
    const nNE = this.chunks.get(chunkKey(cx + 1, cz - 1));
    const nSW = this.chunks.get(chunkKey(cx - 1, cz + 1));
    const nSE = this.chunks.get(chunkKey(cx + 1, cz + 1));

    let topY = chunk.maxY;
    if (nW && nW.maxY > topY) topY = nW.maxY;
    if (nE && nE.maxY > topY) topY = nE.maxY;
    if (nN && nN.maxY > topY) topY = nN.maxY;
    if (nS && nS.maxY > topY) topY = nS.maxY;
    topY = Math.min(CHUNK_H - 2, topY + 2);

    // Must clear through topY + 6 because buildChunkMesh scans skyLight from ceilY = topY + 3!
    padded.fill(0, 0, Math.min(PADDED_VOLUME, (topY + 6) * PADDED_SLICE));

    const vox = chunk.voxels;
    for (let y = 0; y <= topY; y++) {
      const srcY = y << 8;
      const dstY = y * PADDED_SLICE;

      // Center 16x16 rows (16-byte typed subarray copies!)
      for (let z = 0; z < 16; z++) {
        const sRow = srcY | (z << 4);
        const dRow = dstY + (z + 1) * PADDED_W + 1;
        padded.set(vox.subarray(sRow, sRow + 16), dRow);
      }

      // West (px = 0) & East (px = 17)
      if (nW && nW.voxels) {
        const nv = nW.voxels;
        for (let z = 0; z < 16; z++) padded[dstY + (z + 1) * PADDED_W] = nv[srcY | (z << 4) | 15];
      }
      if (nE && nE.voxels) {
        const nv = nE.voxels;
        for (let z = 0; z < 16; z++) padded[dstY + (z + 1) * PADDED_W + 17] = nv[srcY | (z << 4)];
      }

      // North (pz = 0) & South (pz = 17)
      if (nN && nN.voxels) {
        padded.set(nN.voxels.subarray(srcY | (15 << 4), (srcY | (15 << 4)) + 16), dstY + 1);
      }
      if (nS && nS.voxels) {
        padded.set(nS.voxels.subarray(srcY, srcY + 16), dstY + 17 * PADDED_W + 1);
      }

      // 4 diagonal corners for seamless cross-chunk corner AO
      if (nNW && nNW.voxels) padded[dstY] = nNW.voxels[srcY | (15 << 4) | 15];
      if (nNE && nNE.voxels) padded[dstY + 17] = nNE.voxels[srcY | (15 << 4)];
      if (nSW && nSW.voxels) padded[dstY + 17 * PADDED_W] = nSW.voxels[srcY | 15];
      if (nSE && nSE.voxels) padded[dstY + 17 * PADDED_W + 17] = nSE.voxels[srcY];
    }

    // Seal unloaded outer perimeter columns up to exact 2D neighbor surface height so outer LOD0
    // chunks never emit 64-block deep underground skirts at the LOD0 <-> LOD1 boundary!
    const wx0 = cx * 16;
    const wz0 = cz * 16;
    if (!nW || !nW.voxels) {
      for (let z = 0; z < 16; z++) {
        const s = this.biomeField.sampleColumn(wx0 - 1, wz0 + z, this.biomeProbe);
        const h = Math.min(topY, s.height);
        for (let y = 0; y <= h; y++) padded[y * PADDED_SLICE + (z + 1) * PADDED_W] = BLOCK.STONE;
        for (let y = h + 1; y <= Math.min(topY, s.waterLevel); y++) {
          padded[y * PADDED_SLICE + (z + 1) * PADDED_W] = BLOCK.WATER;
        }
      }
    }
    if (!nE || !nE.voxels) {
      for (let z = 0; z < 16; z++) {
        const s = this.biomeField.sampleColumn(wx0 + 16, wz0 + z, this.biomeProbe);
        const h = Math.min(topY, s.height);
        for (let y = 0; y <= h; y++) padded[y * PADDED_SLICE + (z + 1) * PADDED_W + 17] = BLOCK.STONE;
        for (let y = h + 1; y <= Math.min(topY, s.waterLevel); y++) {
          padded[y * PADDED_SLICE + (z + 1) * PADDED_W + 17] = BLOCK.WATER;
        }
      }
    }
    if (!nN || !nN.voxels) {
      for (let x = 0; x < 16; x++) {
        const s = this.biomeField.sampleColumn(wx0 + x, wz0 - 1, this.biomeProbe);
        const h = Math.min(topY, s.height);
        for (let y = 0; y <= h; y++) padded[y * PADDED_SLICE + (x + 1)] = BLOCK.STONE;
        for (let y = h + 1; y <= Math.min(topY, s.waterLevel); y++) {
          padded[y * PADDED_SLICE + (x + 1)] = BLOCK.WATER;
        }
      }
    }
    if (!nS || !nS.voxels) {
      for (let x = 0; x < 16; x++) {
        const s = this.biomeField.sampleColumn(wx0 + x, wz0 + 16, this.biomeProbe);
        const h = Math.min(topY, s.height);
        for (let y = 0; y <= h; y++) padded[y * PADDED_SLICE + 17 * PADDED_W + (x + 1)] = BLOCK.STONE;
        for (let y = h + 1; y <= Math.min(topY, s.waterLevel); y++) {
          padded[y * PADDED_SLICE + 17 * PADDED_W + (x + 1)] = BLOCK.WATER;
        }
      }
    }

    return topY;
  }

  /**
   * Computes Quadtree LOD tier limits aligned to 2x2 and 4x4 chunk cells:
   *  - LOD0 (1x1 full 16x256x16 voxel chunks): up to radius 12 chunks (192 blocks)
   *  - LOD1 (2x2 merged 32x32 stride-2 super-chunks): radius 12..20 chunks
   *  - LOD2 (4x4 merged 64x64 stride-4 super-chunks): radius 20..renderDistance (up to 36+ chunks)
   */
  rebuildStreamingTargets(camCX, camCZ, fwdX, fwdZ, renderDistance) {
    this.terrainQueue.length = 0;
    this.lodQueue.length = 0;

    const rdSq = renderDistance * renderDistance;
    const lod0Radius = Math.min(renderDistance, 12);
    const lod1Radius = Math.min(renderDistance, 20);

    const wantedLOD0 = this.wantedLOD0;
    const wantedLOD = this.wantedLOD;
    wantedLOD0.clear();
    wantedLOD.clear();

    const priorityScore = (dx, dz) => {
      const distSq = dx * dx + dz * dz;
      if (distSq < 2.5) return distSq * 0.25;
      const inv = 1.0 / Math.sqrt(distSq);
      const dot = (dx * inv) * fwdX + (dz * inv) * fwdZ;
      return distSq * (1.35 - 0.65 * Math.max(-0.3, dot));
    };

    // Walk 4x4 macro-cells covering [-renderDistance .. +renderDistance]
    const minMacro = Math.floor((camCX - renderDistance - 3) / 4);
    const maxMacro = Math.floor((camCX + renderDistance + 3) / 4);
    const minMacroZ = Math.floor((camCZ - renderDistance - 3) / 4);
    const maxMacroZ = Math.floor((camCZ + renderDistance + 3) / 4);

    for (let mz = minMacroZ; mz <= maxMacroZ; mz++) {
      for (let mx = minMacro; mx <= maxMacro; mx++) {
        const mCenterCX = mx * 4 + 2;
        const mCenterCZ = mz * 4 + 2;
        const mdx = mCenterCX - camCX;
        const mdz = mCenterCZ - camCZ;
        const mDist = Math.hypot(mdx, mdz);

        // Far horizon ring -> single 4x4 merged LOD-2 super-chunk (64x64 blocks, 16x16 cells @ stride 4)
        if (mDist > lod1Radius + 1.5 && mDist - 2.9 <= renderDistance) {
          const lKey = `L2:${mx},${mz}`;
          wantedLOD.add(lKey);
          if (!this.lodMeshes.has(lKey) && !this.pendingLOD.has(lKey)) {
            this.lodQueue.push({
              key: lKey,
              tier: 2,
              worldX0: mx * 64,
              worldZ0: mz * 64,
              gridN: 16,
              stride: 4,
              score: priorityScore(mdx, mdz) + 30,
            });
          }
          continue;
        }

        // Subdivide into four 2x2 sub-cells (32x32 blocks)
        for (let szOff = 0; szOff < 2; szOff++) {
          for (let sxOff = 0; sxOff < 2; sxOff++) {
            const sx = mx * 2 + sxOff;
            const sz = mz * 2 + szOff;
            const sCenterCX = sx * 2 + 1;
            const sCenterCZ = sz * 2 + 1;
            const sdx = sCenterCX - camCX;
            const sdz = sCenterCZ - camCZ;
            const sDist = Math.hypot(sdx, sdz);

            if (sDist > lod0Radius + 0.8 && sDist - 1.5 <= renderDistance) {
              const lKey = `L1:${sx},${sz}`;
              wantedLOD.add(lKey);
              if (!this.lodMeshes.has(lKey) && !this.pendingLOD.has(lKey)) {
                this.lodQueue.push({
                  key: lKey,
                  tier: 1,
                  worldX0: sx * 32,
                  worldZ0: sz * 32,
                  gridN: 16,
                  stride: 2,
                  score: priorityScore(sdx, sdz) + 12,
                });
              }
              continue;
            }

            // Subdivide into four 1x1 full LOD-0 voxel chunks (16x256x16)
            for (let czOff = 0; czOff < 2; czOff++) {
              for (let cxOff = 0; cxOff < 2; cxOff++) {
                const cx = sx * 2 + cxOff;
                const cz = sz * 2 + czOff;
                const cdx = cx + 0.5 - camCX;
                const cdz = cz + 0.5 - camCZ;
                if (cdx * cdx + cdz * cdz > rdSq + 2.0) continue;
                if (Math.hypot(cdx, cdz) > lod0Radius + 1.8) continue;

                const cKey = chunkKey(cx, cz);
                wantedLOD0.add(cKey);
                if (!this.chunks.has(cKey) && !this.pendingTerrain.has(cKey)) {
                  this.terrainQueue.push({
                    cx,
                    cz,
                    key: cKey,
                    score: priorityScore(cdx, cdz),
                  });
                } else {
                  const existing = this.chunks.get(cKey);
                  if (existing && existing.needsMesh && !existing.meshing) {
                    this.enqueueChunkMesh(existing, priorityScore(cdx, cdz));
                  }
                }
              }
            }
          }
        }
      }
    }

    // Sort highest priority LAST so `.pop()` is O(1)
    this.terrainQueue.sort((a, b) => b.score - a.score);
    this.lodQueue.sort((a, b) => b.score - a.score);

    // Unload distant LOD-0 chunks beyond hysteresis margin & free GPU VAOs/VBOs
    const keepMargin = lod0Radius + 4.0;
    for (const [cKey, chunk] of this.chunks.entries()) {
      if (!wantedLOD0.has(cKey)) {
        const dist = Math.hypot(chunk.cx + 0.5 - camCX, chunk.cz + 0.5 - camCZ);
        if (dist > keepMargin || dist > renderDistance + 2.5) {
          this.destroyChunk(chunk);
          this.chunks.delete(cKey);
          this.chunksByInt.delete(packChunkInt(chunk.cx, chunk.cz));
        }
      }
    }

    // Unload obsolete LOD-1/2 super-chunks (keep parent briefly if child LOD-0 chunks are still streaming!)
    for (const [lKey, lod] of this.lodMeshes.entries()) {
      lod.activeWanted = wantedLOD.has(lKey);
      if (!lod.activeWanted) {
        const centerCX = (lod.worldX0 + lod.spanBlocks * 0.5) / 16.0;
        const centerCZ = (lod.worldZ0 + lod.spanBlocks * 0.5) / 16.0;
        const dist = Math.hypot(centerCX - camCX, centerCZ - camCZ);
        const childrenReady = this.areSubChunksReady(lod.worldX0, lod.worldZ0, lod.spanBlocks);
        if (dist > renderDistance + 4.0 || childrenReady) {
          this.renderer.destroyMeshBuffer(lod.meshSurface);
          this.renderer.destroyMeshBuffer(lod.meshTrans);
          this.lodMeshes.delete(lKey);
        }
      }
    }
  }

  areSubChunksReady(worldX0, worldZ0, spanBlocks) {
    const cx0 = worldX0 >> 4;
    const cz0 = worldZ0 >> 4;
    const n = spanBlocks >> 4;
    for (let dz = 0; dz < n; dz++) {
      for (let dx = 0; dx < n; dx++) {
        const ch = this.chunks.get(chunkKey(cx0 + dx, cz0 + dz));
        if (!ch || !ch.hasMesh) return false;
      }
    }
    return true;
  }

  destroyChunk(chunk) {
    this.renderer.destroyMeshBuffer(chunk.meshSurface);
    this.renderer.destroyMeshBuffer(chunk.meshUnderground);
    this.renderer.destroyMeshBuffer(chunk.meshCutout);
    this.renderer.destroyMeshBuffer(chunk.meshTrans);
    chunk.voxels = null;
  }

  enqueueChunkMesh(chunk, score = 0) {
    if (chunk.meshing || chunk.queuedForMesh) return;
    chunk.queuedForMesh = true;
    chunk.priorityScore = score;
    this.meshQueue.push(chunk);
  }

  markNeighborDirty(cx, cz) {
    const nb = this.chunks.get(chunkKey(cx, cz));
    if (nb && nb.voxels) {
      nb.rev++;
      nb.needsMesh = true;
      if (!nb.meshing && !nb.queuedForMesh) {
        this.enqueueChunkMesh(nb, nb.priorityScore || 50);
      }
    }
  }

  dispatchWorkerJobs(camCX, camCZ) {
    // 1. Dispatch terrain & LOD jobs across terrain worker pool (max 3 in-flight per worker)
    for (let w = 0; w < this.terrainWorkers.length; w++) {
      while (this.terrainBusy[w] < 3) {
        // Interleave nearby LOD-0 chunks first, then horizon LOD super-chunks
        let job = null;
        const topChunk = this.terrainQueue[this.terrainQueue.length - 1];
        const topLod = this.lodQueue[this.lodQueue.length - 1];

        if (topChunk && (!topLod || topChunk.score <= topLod.score)) {
          job = this.terrainQueue.pop();
          this.pendingTerrain.add(job.key);
          this.terrainBusy[w]++;
          this.terrainWorkers[w].postMessage({
            type: 'chunk',
            id: this.nextJobId++,
            seed: this.seed,
            cx: job.cx,
            cz: job.cz,
          });
        } else if (topLod) {
          job = this.lodQueue.pop();
          this.pendingLOD.add(job.key);
          this.terrainBusy[w]++;
          this.terrainWorkers[w].postMessage({
            type: 'lod',
            id: this.nextJobId++,
            seed: this.seed,
            key: job.key,
            worldX0: job.worldX0,
            worldZ0: job.worldZ0,
            gridN: job.gridN,
            stride: job.stride,
          });
        } else {
          break;
        }
      }
    }

    // 2. Dispatch meshing jobs across mesher worker pool (max 2 in-flight per worker)
    if (this.meshQueue.length > 1) {
      for (let i = 0; i < this.meshQueue.length; i++) {
        const ch = this.meshQueue[i];
        const dx = ch.cx + 0.5 - camCX;
        const dz = ch.cz + 0.5 - camCZ;
        ch.priorityScore = dx * dx + dz * dz;
      }
      this.meshQueue.sort((a, b) => b.priorityScore - a.priorityScore);
    }

    for (let w = 0; w < this.mesherWorkers.length; w++) {
      while (this.mesherBusy[w] < 2 && this.meshQueue.length > 0) {
        const chunk = this.meshQueue.pop();
        chunk.queuedForMesh = false;
        if (!chunk.voxels || !this.chunks.has(chunk.key)) continue;

        chunk.meshing = true;
        chunk.needsMesh = false;
        const padded = this.acquireHaloBuffer();
        const scanMax = this.fillPaddedHalo(chunk, padded);

        this.mesherBusy[w]++;
        this.mesherWorkers[w].postMessage(
          {
            id: this.nextJobId++,
            cx: chunk.cx,
            cz: chunk.cz,
            rev: chunk.rev,
            padded,
            maxY: scanMax,
            minSurfaceY: chunk.minSurfaceY,
          },
          [padded.buffer]
        );
      }
    }
  }

  onTerrainWorkerMessage(workerIdx, msg) {
    this.terrainBusy[workerIdx] = Math.max(0, this.terrainBusy[workerIdx] - 1);

    if (msg.type === 'chunk') {
      const cKey = chunkKey(msg.cx, msg.cz);
      this.pendingTerrain.delete(cKey);

      let maxY = msg.maxY;
      if (this.storage) {
        const editedMaxY = this.storage.applyChunkDeltas(cKey, msg.voxels);
        if (editedMaxY > maxY) maxY = editedMaxY;
      }

      const chunk = {
        key: cKey,
        cx: msg.cx,
        cz: msg.cz,
        voxels: msg.voxels,
        minSurfaceY: msg.minSurfaceY,
        maxY,
        meshMinY: 0,
        meshMaxY: maxY,
        sectionMasks: null,
        meshSurface: null,
        meshUnderground: null,
        meshCutout: null,
        meshTrans: null,
        rev: 1,
        meshedRev: 0,
        hasMesh: false,
        needsMesh: true,
        meshing: false,
        queuedForMesh: false,
        priorityScore: 0,
        fade: 0.0,
        visStamp: 0,
        undergroundVisStamp: 0,
      };
      this.chunks.set(cKey, chunk);
      this.chunksByInt.set(packChunkInt(msg.cx, msg.cz), chunk);
      this.enqueueChunkMesh(chunk, 0);

      // Re-stitch existing meshed neighbors so cross-chunk AO and border faces are seamless
      this.markNeighborDirty(msg.cx - 1, msg.cz);
      this.markNeighborDirty(msg.cx + 1, msg.cz);
      this.markNeighborDirty(msg.cx, msg.cz - 1);
      this.markNeighborDirty(msg.cx, msg.cz + 1);
    } else if (msg.type === 'lod') {
      this.pendingLOD.delete(msg.key);
      this.uploadQueue.push({
        isLOD: true,
        key: msg.key,
        worldX0: msg.worldX0,
        worldZ0: msg.worldZ0,
        spanBlocks: msg.spanBlocks,
        opaqueSurface: msg.opaqueSurface,
        transparent: msg.transparent,
        minY: msg.minY,
        maxY: msg.maxY,
      });
    }
  }

  onMesherWorkerMessage(workerIdx, msg) {
    this.mesherBusy[workerIdx] = Math.max(0, this.mesherBusy[workerIdx] - 1);
    if (msg.recycledPadded) {
      this.releaseHaloBuffer(msg.recycledPadded);
    }

    const cKey = chunkKey(msg.cx, msg.cz);
    const chunk = this.chunks.get(cKey);
    if (!chunk) return;

    chunk.meshing = false;
    this.uploadQueue.push({
      isLOD: false,
      chunk,
      rev: msg.rev,
      opaqueSurface: msg.opaqueSurface,
      opaqueUnderground: msg.opaqueUnderground,
      cutout: msg.cutout,
      transparent: msg.transparent,
      sectionMasks: msg.sectionMasks,
      minY: msg.minY,
      maxY: msg.maxY,
    });

    if (chunk.needsMesh) {
      this.enqueueChunkMesh(chunk, chunk.priorityScore);
    }
  }

  /**
   * Time-sliced GPU VAO/VBO upload loop capped at `budgetMs` (default 2.0 ms/frame).
   */
  processGPUUploads(camX, camZ, budgetMs = 2.0) {
    if (this.uploadQueue.length === 0) {
      this.stats.uploadMs = 0;
      return;
    }

    const t0 = performance.now();
    // Upload closest meshes first
    if (this.uploadQueue.length > 2) {
      for (let i = 0; i < this.uploadQueue.length; i++) {
        const u = this.uploadQueue[i];
        const wx = u.isLOD ? (u.worldX0 + u.spanBlocks * 0.5) : (u.chunk.cx * 16 + 8);
        const wz = u.isLOD ? (u.worldZ0 + u.spanBlocks * 0.5) : (u.chunk.cz * 16 + 8);
        u.distSq = (wx - camX) * (wx - camX) + (wz - camZ) * (wz - camZ);
      }
      this.uploadQueue.sort((a, b) => b.distSq - a.distSq);
    }

    let uploaded = 0;
    while (this.uploadQueue.length > 0) {
      const item = this.uploadQueue.pop();

      if (item.isLOD) {
        const existing = this.lodMeshes.get(item.key);
        const meshSurface = this.renderer.createOrUpdateMeshBuffer(
          item.opaqueSurface,
          existing ? existing.meshSurface : null
        );
        const meshTrans = this.renderer.createOrUpdateMeshBuffer(
          item.transparent,
          existing ? existing.meshTrans : null
        );
        this.lodMeshes.set(item.key, {
          key: item.key,
          worldX0: item.worldX0,
          worldZ0: item.worldZ0,
          spanBlocks: item.spanBlocks,
          meshSurface,
          meshTrans,
          minY: item.minY,
          maxY: item.maxY,
          fade: existing ? existing.fade : 0.0,
          activeWanted: true,
        });
      } else {
        const chunk = item.chunk;
        if (!chunk.voxels || !this.chunks.has(chunk.key)) continue;
        const wasMeshed = chunk.hasMesh;
        chunk.meshSurface = this.renderer.createOrUpdateMeshBuffer(item.opaqueSurface, chunk.meshSurface);
        chunk.meshUnderground = this.renderer.createOrUpdateMeshBuffer(item.opaqueUnderground, chunk.meshUnderground);
        chunk.meshCutout = this.renderer.createOrUpdateMeshBuffer(item.cutout, chunk.meshCutout);
        chunk.meshTrans = this.renderer.createOrUpdateMeshBuffer(item.transparent, chunk.meshTrans);
        chunk.sectionMasks = item.sectionMasks;
        chunk.meshMinY = item.minY;
        chunk.meshMaxY = item.maxY;
        chunk.meshedRev = item.rev;
        chunk.hasMesh = true;
        if (!wasMeshed) {
          // Spawn vicinity chunks appear immediately on initial load; newly streamed distant chunks fade in smoothly
          chunk.fade = (item.distSq !== undefined && item.distSq < 48 * 48) ? 1.0 : 0.05;
        }
      }

      uploaded++;
      if (uploaded >= 1 && performance.now() - t0 >= budgetMs) {
        break;
      }
    }

    this.stats.uploadMs = +(performance.now() - t0).toFixed(2);
  }

  /**
   * Runs 16x16x16 Section BFS cave connectivity graph from the camera section,
   * determining which LOD-0 chunks and which underground cave sub-meshes are reachable!
   * Zero heap/string allocations: pre-populates a 37x37 chunk grid + frustum mask via Smi integer keys.
   */
  runSectionVisibilityBFS(camCX, camSY, camCZ, maxRadiusChunks) {
    const stamp = ++this.bfsStamp;
    const visited = this.bfsVisited;
    const qNode = this.bfsQueueNode;
    const qEntry = this.bfsQueueEntryDir;
    const qMask = this.bfsQueueAxisMask;
    const chunkGrid = this.bfsChunkGrid;
    const colFrustum = this.bfsColInFrustum;

    const r = Math.min(BFS_GRID_RADIUS, maxRadiusChunks + 1);
    const rLimitSq = (r + 1) * (r + 1);

    // Pre-populate 2D (dx, dz) chunk pointers & column frustum visibility once (at most 27x27 = 729 cols!)
    for (let dx = -r; dx <= r; dx++) {
      const rowBase = (dx + BFS_GRID_RADIUS) * BFS_GRID_SPAN;
      const wx0 = (camCX + dx) * 16;
      for (let dz = -r; dz <= r; dz++) {
        const gIdx = rowBase + (dz + BFS_GRID_RADIUS);
        if (dx * dx + dz * dz > rLimitSq) {
          chunkGrid[gIdx] = null;
          colFrustum[gIdx] = 0;
          continue;
        }
        chunkGrid[gIdx] = this.chunksByInt.get(packChunkInt(camCX + dx, camCZ + dz)) || null;
        if (dx * dx + dz * dz <= 2) {
          colFrustum[gIdx] = 1;
        } else {
          const wz0 = (camCZ + dz) * 16;
          colFrustum[gIdx] = frustumIntersectsAABB(this.frustumPlanes, wx0, 0, wz0, wx0 + 16, 256, wz0 + 16) ? 1 : 0;
        }
      }
    }

    const startSY = Math.max(0, Math.min(CHUNK_SECTIONS - 1, camSY));
    const encode = (dx, dz, sy) => ((dx + BFS_GRID_RADIUS) * BFS_GRID_SPAN + (dz + BFS_GRID_RADIUS)) * 16 + sy;

    const startNode = encode(0, 0, startSY);
    visited[startNode] = stamp;

    let head = 0;
    let tail = 0;
    qNode[tail] = startNode;
    qEntry[tail] = -1;
    qMask[tail] = 0;
    tail++;

    // Step vectors for 6 faces: 0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z
    const stepDX = [1, -1, 0, 0, 0, 0];
    const stepDY = [0, 0, 1, -1, 0, 0];
    const stepDZ = [0, 0, 0, 0, 1, -1];

    while (head < tail) {
      const node = qNode[head];
      const entryFace = qEntry[head];
      const usedMask = qMask[head];
      head++;

      const sy = node & 15;
      const gIdx = node >> 4;
      const dz = (gIdx % BFS_GRID_SPAN) - BFS_GRID_RADIUS;
      const dx = ((gIdx / BFS_GRID_SPAN) | 0) - BFS_GRID_RADIUS;

      const chunk = chunkGrid[gIdx];

      // Unloaded chunks are open air above sea level (sy >= 4) and solid earth underground (sy < 4)
      let secMask = sy >= 4 ? 0x7fff : 0;
      if (chunk) {
        chunk.visStamp = stamp;
        const underSection = (chunk.minSurfaceY - 6) >> 4;
        if (underSection >= 0 && sy <= underSection) {
          chunk.undergroundVisStamp = stamp;
        }
        if (chunk.sectionMasks) {
          secMask = chunk.sectionMasks[sy];
        }
      }

      if (secMask === 0) continue;

      for (let exitFace = 0; exitFace < 6; exitFace++) {
        const oppExit = oppositeFace(exitFace);
        if (usedMask & (1 << oppExit)) continue;

        if (entryFace !== -1) {
          if (entryFace === exitFace) continue;
          if ((secMask & sectionPairBit(entryFace, exitFace)) === 0) continue;
        }

        const ndx = dx + stepDX[exitFace];
        const ndz = dz + stepDZ[exitFace];
        const nsy = sy + stepDY[exitFace];

        if (nsy < 0 || nsy >= CHUNK_SECTIONS) continue;
        if (ndx < -r || ndx > r || ndz < -r || ndz > r) continue;

        const nGIdx = (ndx + BFS_GRID_RADIUS) * BFS_GRID_SPAN + (ndz + BFS_GRID_RADIUS);
        if (!colFrustum[nGIdx]) continue;

        const nNode = (nGIdx << 4) | nsy;
        if (visited[nNode] === stamp) continue;

        visited[nNode] = stamp;
        qNode[tail] = nNode;
        qEntry[tail] = oppExit;
        qMask[tail] = usedMask | (1 << exitFace);
        tail++;
      }
    }
  }

  pushDrawItem(targetList, mesh, ox, oz, fade, distSq) {
    let item;
    if (this.drawPoolCursor < this.drawItemPool.length) {
      item = this.drawItemPool[this.drawPoolCursor];
      item.mesh = mesh;
      item.ox = ox;
      item.oz = oz;
      item.fade = fade;
      item.distSq = distSq;
    } else {
      item = { mesh, ox, oz, fade, distSq };
      this.drawItemPool.push(item);
    }
    this.drawPoolCursor++;
    targetList.push(item);
  }

  /**
   * Main per-frame update:
   *  1. Updates streaming priority queues & dispatches worker jobs
   *  2. Uploads ready GPU VAOs within 2.0 ms budget
   *  3. Runs Frustum + Section BFS cave visibility culling and builds sorted 3-pass draw lists
   */
  update(dt, cameraPos, yaw, viewProj, renderDistance) {
    this.frameCounter++;
    const camX = cameraPos[0];
    const camY = cameraPos[1];
    const camZ = cameraPos[2];
    const camCX = Math.floor(camX / 16);
    const camCZ = Math.floor(camZ / 16);
    const camSY = Math.floor(camY / 16);

    const fwdX = Math.sin(yaw);
    const fwdZ = -Math.cos(yaw);

    const movedChunk = camCX !== this.lastQueueCX || camCZ !== this.lastQueueCZ;
    const turnedMuch = (fwdX * this.lastFwdX + fwdZ * this.lastFwdZ) < 0.90;
    const rdChanged = renderDistance !== this.lastQueueRD;
    const needsRetry = (this.stats.pendingJobs > 0 || this.chunks.size === 0) && (this.frameCounter % 90 === 0);

    if (movedChunk || turnedMuch || rdChanged || needsRetry) {
      this.lastQueueCX = camCX;
      this.lastQueueCZ = camCZ;
      this.lastQueueRD = renderDistance;
      this.lastFwdX = fwdX;
      this.lastFwdZ = fwdZ;
      this.rebuildStreamingTargets(camCX, camCZ, fwdX, fwdZ, renderDistance);
    }

    this.dispatchWorkerJobs(camCX, camCZ);
    this.processGPUUploads(camX, camZ, 2.0);

    // Frustum + Section BFS Cave Culling + Pass Sorting
    const tCull0 = performance.now();
    extractFrustumPlanes(this.frustumPlanes, viewProj);

    const lod0Radius = Math.min(renderDistance, 12);
    this.runSectionVisibilityBFS(camCX, camSY, camCZ, lod0Radius);
    const stamp = this.bfsStamp;

    this.opaqueDraws.length = 0;
    this.cutoutDraws.length = 0;
    this.transparentDraws.length = 0;
    this.drawPoolCursor = 0;

    let visChunks = 0;
    let visLOD = 0;
    let culledUnder = 0;
    const fadeStep = dt * 3.1;

    // 1. Collect visible LOD-0 chunks
    for (const chunk of this.chunks.values()) {
      if (!chunk.hasMesh) continue;
      if (chunk.fade < 1.0) {
        chunk.fade = Math.min(1.0, chunk.fade + fadeStep);
      }

      if (chunk.visStamp !== stamp) continue;

      const ox = chunk.cx * 16;
      const oz = chunk.cz * 16;
      if (!frustumIntersectsAABB(this.frustumPlanes, ox, chunk.meshMinY, oz, ox + 16, chunk.meshMaxY, oz + 16)) {
        continue;
      }

      const dx = ox + 8 - camX;
      const dz = oz + 8 - camZ;
      const distSq = dx * dx + dz * dz;
      if (distSq > (renderDistance + 1.5) * (renderDistance + 1.5) * 256) continue;

      visChunks++;

      if (chunk.meshSurface) {
        this.pushDrawItem(this.opaqueDraws, chunk.meshSurface, ox, oz, chunk.fade, distSq);
      }
      if (chunk.meshUnderground) {
        // Draw underground cave VAO if camera is underground/near-surface or BFS reached cave sections within 112 blocks
        const camBelow = camY < chunk.minSurfaceY + 4;
        if (camBelow || (chunk.undergroundVisStamp === stamp && distSq < 112 * 112)) {
          this.pushDrawItem(this.opaqueDraws, chunk.meshUnderground, ox, oz, chunk.fade, distSq);
        } else {
          culledUnder++;
        }
      }
      if (chunk.meshCutout) {
        this.pushDrawItem(this.cutoutDraws, chunk.meshCutout, ox, oz, chunk.fade, distSq);
      }
      if (chunk.meshTrans) {
        this.pushDrawItem(this.transparentDraws, chunk.meshTrans, ox, oz, chunk.fade, distSq);
      }
    }

    // 2. Collect visible LOD-1 / LOD-2 horizon super-chunks
    const maxWorldDistSq = (renderDistance + 3.0) * (renderDistance + 3.0) * 256;
    for (const lod of this.lodMeshes.values()) {
      if (!lod.activeWanted && this.areSubChunksReady(lod.worldX0, lod.worldZ0, lod.spanBlocks)) {
        continue;
      }
      if (lod.fade < 1.0) {
        lod.fade = Math.min(1.0, lod.fade + fadeStep);
      }

      const ox = lod.worldX0;
      const oz = lod.worldZ0;
      const span = lod.spanBlocks;
      if (!frustumIntersectsAABB(this.frustumPlanes, ox, lod.minY, oz, ox + span, lod.maxY, oz + span)) {
        continue;
      }

      const dx = ox + span * 0.5 - camX;
      const dz = oz + span * 0.5 - camZ;
      const distSq = dx * dx + dz * dz;
      if (distSq > maxWorldDistSq) continue;

      visLOD++;
      if (lod.meshSurface) {
        this.pushDrawItem(this.opaqueDraws, lod.meshSurface, ox, oz, lod.fade, distSq);
      }
      if (lod.meshTrans) {
        this.pushDrawItem(this.transparentDraws, lod.meshTrans, ox, oz, lod.fade, distSq);
      }
    }

    // Sort Pass 1 Opaque front-to-back (maximizes hardware Early-Z rejection)
    this.opaqueDraws.sort((a, b) => a.distSq - b.distSq);
    // Sort Pass 3 Transparent back-to-front (correct alpha compositing for water, ice, glass)
    this.transparentDraws.sort((a, b) => b.distSq - a.distSq);

    this.stats.cullMs = +(performance.now() - tCull0).toFixed(2);
    this.stats.loadedChunks = this.chunks.size;
    this.stats.loadedLOD = this.lodMeshes.size;
    this.stats.visibleChunks = visChunks;
    this.stats.visibleLOD = visLOD;
    this.stats.culledUnderground = culledUnder;
    this.stats.pendingJobs =
      this.terrainQueue.length +
      this.lodQueue.length +
      this.meshQueue.length +
      this.uploadQueue.length +
      this.pendingTerrain.size +
      this.pendingLOD.size;
  }
}
