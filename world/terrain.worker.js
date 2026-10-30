// Off-thread Terrain Generation Web Worker.
// Generates full 16x256x16 voxel chunks AND merged far-distance LOD heightfield meshes
// without stalling the main render thread. Transfers all TypedArray buffers zero-copy.

import { buildHeightfieldLODMesh } from '../chunk/mesh_builder.js';
import { TerrainGenerator } from './terrain_gen.js';

let generator = null;
let currentSeed = null;

function getGenerator(seed) {
  if (!generator || currentSeed !== seed) {
    currentSeed = seed;
    generator = new TerrainGenerator(seed);
  }
  return generator;
}

self.onmessage = (ev) => {
  const msg = ev.data;
  const gen = getGenerator(msg.seed);

  if (msg.type === 'chunk') {
    const { voxels, minSurfaceY, maxY } = gen.generateChunk(msg.cx, msg.cz);
    self.postMessage(
      {
        type: 'chunk',
        id: msg.id,
        cx: msg.cx,
        cz: msg.cz,
        voxels,
        minSurfaceY,
        maxY,
      },
      [voxels.buffer]
    );
  } else if (msg.type === 'lod') {
    const { heights, topBlocks, subBlocks, waterLevel } = gen.generateLODGrid(
      msg.worldX0,
      msg.worldZ0,
      msg.gridN,
      msg.stride
    );
    const mesh = buildHeightfieldLODMesh(
      heights,
      topBlocks,
      subBlocks,
      waterLevel,
      msg.gridN,
      msg.stride
    );
    self.postMessage(
      {
        type: 'lod',
        id: msg.id,
        key: msg.key,
        worldX0: msg.worldX0,
        worldZ0: msg.worldZ0,
        spanBlocks: msg.gridN * msg.stride,
        opaqueSurface: mesh.opaqueSurface,
        transparent: mesh.transparent,
        minY: mesh.minY,
        maxY: mesh.maxY,
      },
      [mesh.opaqueSurface.buffer, mesh.transparent.buffer]
    );
  }
};
