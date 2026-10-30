// Off-thread Chunk Meshing Web Worker.
// Runs sky-light flood propagation, 15-bit section cave visibility analysis, and
// 6-direction greedy meshing with smooth 0fps vertex AO.
// Returns the 81KB `padded` halo ArrayBuffer back to the main thread pool to prevent GC spikes.

import { buildChunkMesh } from '../chunk/mesh_builder.js';

self.onmessage = (ev) => {
  const { id, cx, cz, padded, maxY, minSurfaceY } = ev.data;
  const res = buildChunkMesh(padded, maxY, minSurfaceY);

  self.postMessage(
    {
      id,
      cx,
      cz,
      rev: ev.data.rev,
      opaqueSurface: res.opaqueSurface,
      opaqueUnderground: res.opaqueUnderground,
      cutout: res.cutout,
      transparent: res.transparent,
      sectionMasks: res.sectionMasks,
      minY: res.minY,
      maxY: res.maxY,
      recycledPadded: padded,
    },
    [
      res.opaqueSurface.buffer,
      res.opaqueUnderground.buffer,
      res.cutout.buffer,
      res.transparent.buffer,
      res.sectionMasks.buffer,
      padded.buffer,
    ]
  );
};
