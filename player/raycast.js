// Amanatides & Woo Fast 3D Voxel DDA Raycaster (`A Fast Voxel Traversal Algorithm for Ray Tracing`).
// Steps exact voxel boundaries up to `maxDist` (default 6.5 blocks), returning both the hit block
// coordinates `(x, y, z)` and the entry face normal `(nx, ny, nz)` for adjacent block placement.

import { BLOCK_RAYCASTABLE } from '../chunk/blocks.js';

export function raycastFromAngles(world, eyePos, yaw, pitch, maxDist = 6.5) {
  const cp = Math.cos(pitch);
  const dx = Math.sin(yaw) * cp;
  const dy = Math.sin(pitch);
  const dz = -Math.cos(yaw) * cp;
  return raycastDDA(world, eyePos[0], eyePos[1], eyePos[2], dx, dy, dz, maxDist);
}

export function raycastDDA(world, ox, oy, oz, dx, dy, dz, maxDist = 6.5) {
  const len = Math.hypot(dx, dy, dz);
  if (len < 1e-8) return null;
  dx /= len;
  dy /= len;
  dz /= len;

  let x = Math.floor(ox);
  let y = Math.floor(oy);
  let z = Math.floor(oz);

  const stepX = dx > 0 ? 1 : (dx < 0 ? -1 : 0);
  const stepY = dy > 0 ? 1 : (dy < 0 ? -1 : 0);
  const stepZ = dz > 0 ? 1 : (dz < 0 ? -1 : 0);

  const tDeltaX = stepX !== 0 ? Math.abs(1.0 / dx) : 1e30;
  const tDeltaY = stepY !== 0 ? Math.abs(1.0 / dy) : 1e30;
  const tDeltaZ = stepZ !== 0 ? Math.abs(1.0 / dz) : 1e30;

  let tMaxX = stepX > 0 ? (x + 1.0 - ox) * tDeltaX : (stepX < 0 ? (ox - x) * tDeltaX : 1e30);
  let tMaxY = stepY > 0 ? (y + 1.0 - oy) * tDeltaY : (stepY < 0 ? (oy - y) * tDeltaY : 1e30);
  let tMaxZ = stepZ > 0 ? (z + 1.0 - oz) * tDeltaZ : (stepZ < 0 ? (oz - z) * tDeltaZ : 1e30);

  let nx = 0;
  let ny = 0;
  let nz = 0;
  let t = 0.0;

  for (let iter = 0; iter < 64 && t <= maxDist; iter++) {
    const b = world.getBlock(x, y, z);
    if (BLOCK_RAYCASTABLE[b]) {
      return {
        x,
        y,
        z,
        nx,
        ny,
        nz,
        placeX: x + nx,
        placeY: y + ny,
        placeZ: z + nz,
        blockId: b,
        distance: t,
      };
    }

    if (tMaxX < tMaxY) {
      if (tMaxX < tMaxZ) {
        x += stepX;
        t = tMaxX;
        tMaxX += tDeltaX;
        nx = -stepX; ny = 0; nz = 0;
      } else {
        z += stepZ;
        t = tMaxZ;
        tMaxZ += tDeltaZ;
        nx = 0; ny = 0; nz = -stepZ;
      }
    } else {
      if (tMaxY < tMaxZ) {
        y += stepY;
        t = tMaxY;
        tMaxY += tDeltaY;
        nx = 0; ny = -stepY; nz = 0;
      } else {
        z += stepZ;
        t = tMaxZ;
        tMaxZ += tDeltaZ;
        nx = 0; ny = 0; nz = -stepZ;
      }
    }
  }

  return null;
}
