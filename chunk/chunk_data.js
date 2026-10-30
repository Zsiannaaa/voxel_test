// Chunk dimensions, Y-major voxel indexing, 8-byte vertex packer, and 15-bit section connectivity helpers.

export const CHUNK_W = 16;
export const CHUNK_H = 256;
export const CHUNK_D = 16;
export const SECTION_SIZE = 16;
export const CHUNK_SECTIONS = 16; // 16 vertical 16x16x16 sections = 256 height
export const CHUNK_VOLUME = CHUNK_W * CHUNK_H * CHUNK_D; // 65,536 bytes

export const PADDED_W = 18;
export const PADDED_D = 18;
export const PADDED_SLICE = PADDED_W * PADDED_D; // 324 bytes per Y slice
export const PADDED_VOLUME = PADDED_SLICE * CHUNK_H; // 82,944 bytes

export const SEA_LEVEL = 62;

// Y-major indexing inside a 16x256x16 chunk: horizontal slices are contiguous 256-byte pages
export function voxelIndex(x, y, z) {
  return (y << 8) | (z << 4) | x;
}

export function paddedIndex(px, y, pz) {
  return y * PADDED_SLICE + pz * PADDED_W + px;
}

export function chunkKey(cx, cz) {
  return `${cx},${cz}`;
}

/**
 * Packs one voxel vertex into 8 bytes (2 x uint32 in `out` at `ptr` and `ptr + 1`):
 *   word0: x(7b, 0..64) | y(9b, 0..256) | z(7b, 0..64) | normal(3b, 0..5) | ao(2b, 0..3) | sky(4b, 0..15)
 *   word1: tileId(8b)   | u(9b, 0..256) | v(9b, 0..256) | scale(3b, 1..4) | flags(3b, 0..7)
 */
export function packVertex(out, ptr, x, y, z, normal, ao, sky, tileId, u, v, scale, flags) {
  out[ptr] = (
    (x & 127) |
    ((y & 511) << 7) |
    ((z & 127) << 16) |
    ((normal & 7) << 23) |
    ((ao & 3) << 26) |
    ((sky & 15) << 28)
  ) >>> 0;

  out[ptr + 1] = (
    (tileId & 255) |
    ((u & 511) << 8) |
    ((v & 511) << 17) |
    ((scale & 7) << 26) |
    ((flags & 7) << 29)
  ) >>> 0;
}

// 15 unordered pairs of 6 section faces (0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z)
const PAIR_BIT_LUT = new Uint8Array(36);
{
  let bit = 0;
  for (let a = 0; a < 6; a++) {
    for (let b = a + 1; b < 6; b++) {
      PAIR_BIT_LUT[a * 6 + b] = bit;
      PAIR_BIT_LUT[b * 6 + a] = bit;
      bit++;
    }
  }
}

export function sectionPairBit(faceA, faceB) {
  if (faceA === faceB) return -1;
  return 1 << PAIR_BIT_LUT[faceA * 6 + faceB];
}

export function oppositeFace(face) {
  return face ^ 1;
}
