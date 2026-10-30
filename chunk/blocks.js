// Block registry, render pass types, physical properties, and atlas tile mappings.

export const PASS_OPAQUE = 0;
export const PASS_CUTOUT = 1;
export const PASS_TRANSPARENT = 2;

export const BLOCK = {
  AIR: 0,
  GRASS: 1,
  DIRT: 2,
  STONE: 3,
  COBBLESTONE: 4,
  SAND: 5,
  SNOW: 6,
  GRAVEL: 7,
  LOG: 8,
  PLANKS: 9,
  LEAVES: 10,
  WATER: 11,
  ICE: 12,
  COAL_ORE: 13,
  IRON_ORE: 14,
  GOLD_ORE: 15,
  BEDROCK: 16,
  GLASS: 17,
  SNOW_GRASS: 18,
  TALL_GRASS: 19,
};

export const TILE = {
  GRASS_TOP: 0,
  GRASS_SIDE: 1,
  DIRT: 2,
  STONE: 3,
  COBBLESTONE: 4,
  SAND: 5,
  SNOW: 6,
  GRAVEL: 7,
  LOG_TOP: 8,
  LOG_SIDE: 9,
  PLANKS: 10,
  LEAVES: 11,
  WATER: 12,
  ICE: 13,
  COAL_ORE: 14,
  IRON_ORE: 15,
  GOLD_ORE: 16,
  BEDROCK: 17,
  GLASS: 18,
  CLOUDS: 19,
  SNOW_GRASS_SIDE: 20,
  TALL_GRASS: 21,
  WATER_1: 22,
  WATER_2: 23,
  WATER_3: 24,
};

export const TILE_NAMES = [
  'grass_top',
  'grass_side',
  'dirt',
  'stone',
  'cobblestone',
  'sand',
  'snow',
  'gravel',
  'log_top',
  'log_side',
  'planks',
  'leaves',
  'water',
  'ice',
  'coal_ore',
  'iron_ore',
  'gold_ore',
  'bedrock',
  'glass',
  'clouds',
  'snow_grass_side',
  'tall_grass',
  'water_1',
  'water_2',
  'water_3',
];

// Face normal indices:
// 0: +X (East), 1: -X (West), 2: +Y (Top), 3: -Y (Bottom), 4: +Z (South), 5: -Z (North)
export const FACE_PX = 0;
export const FACE_NX = 1;
export const FACE_PY = 2;
export const FACE_NY = 3;
export const FACE_PZ = 4;
export const FACE_NZ = 5;

export const MAX_BLOCKS = 32;

// Typed lookup tables for zero-overhead inner loops in workers and physics
export const BLOCK_SOLID = new Uint8Array(MAX_BLOCKS);       // 1 if collides with player AABB
export const BLOCK_OPAQUE = new Uint8Array(MAX_BLOCKS);      // 1 if fully occludes neighbor faces & sky light
export const BLOCK_AO_OCCLUDER = new Uint8Array(MAX_BLOCKS); // 1 if casts vertex ambient occlusion
export const BLOCK_LIGHT_ATTEN = new Uint8Array(MAX_BLOCKS); // Sky light attenuation (0..15)
export const BLOCK_PASS = new Uint8Array(MAX_BLOCKS);        // PASS_OPAQUE / PASS_CUTOUT / PASS_TRANSPARENT
export const BLOCK_CROSS_QUAD = new Uint8Array(MAX_BLOCKS);  // 1 for X-shaped flora (tall_grass)
export const BLOCK_RAYCASTABLE = new Uint8Array(MAX_BLOCKS); // 1 if targeted by DDA block raycast
export const BLOCK_FLAGS = new Uint8Array(MAX_BLOCKS * 6);   // Per-face shader flags (1=foliage sway/tint, 2=water wave)
export const BLOCK_FACE_TILES = new Uint8Array(MAX_BLOCKS * 6);

export const BLOCK_DEFS = [
  { id: BLOCK.AIR, name: 'Air', solid: false, opaque: false, ao: false, atten: 0, pass: PASS_OPAQUE, ray: false, faces: [0, 0, 0, 0, 0, 0] },
  {
    id: BLOCK.GRASS, name: 'Grass Block', solid: true, opaque: true, ao: true, atten: 15, pass: PASS_OPAQUE, ray: true,
    faces: [TILE.GRASS_SIDE, TILE.GRASS_SIDE, TILE.GRASS_TOP, TILE.DIRT, TILE.GRASS_SIDE, TILE.GRASS_SIDE],
    faceFlags: [0, 0, 1, 0, 0, 0],
  },
  { id: BLOCK.DIRT, name: 'Dirt', solid: true, opaque: true, ao: true, atten: 15, pass: PASS_OPAQUE, ray: true, all: TILE.DIRT },
  { id: BLOCK.STONE, name: 'Stone', solid: true, opaque: true, ao: true, atten: 15, pass: PASS_OPAQUE, ray: true, all: TILE.STONE },
  { id: BLOCK.COBBLESTONE, name: 'Cobblestone', solid: true, opaque: true, ao: true, atten: 15, pass: PASS_OPAQUE, ray: true, all: TILE.COBBLESTONE },
  { id: BLOCK.SAND, name: 'Sand', solid: true, opaque: true, ao: true, atten: 15, pass: PASS_OPAQUE, ray: true, all: TILE.SAND },
  { id: BLOCK.SNOW, name: 'Snow', solid: true, opaque: true, ao: true, atten: 15, pass: PASS_OPAQUE, ray: true, all: TILE.SNOW },
  { id: BLOCK.GRAVEL, name: 'Gravel', solid: true, opaque: true, ao: true, atten: 15, pass: PASS_OPAQUE, ray: true, all: TILE.GRAVEL },
  {
    id: BLOCK.LOG, name: 'Oak Log', solid: true, opaque: true, ao: true, atten: 15, pass: PASS_OPAQUE, ray: true,
    faces: [TILE.LOG_SIDE, TILE.LOG_SIDE, TILE.LOG_TOP, TILE.LOG_TOP, TILE.LOG_SIDE, TILE.LOG_SIDE],
  },
  { id: BLOCK.PLANKS, name: 'Wood Planks', solid: true, opaque: true, ao: true, atten: 15, pass: PASS_OPAQUE, ray: true, all: TILE.PLANKS },
  {
    id: BLOCK.LEAVES, name: 'Leaves', solid: true, opaque: false, ao: true, atten: 2, pass: PASS_CUTOUT, ray: true,
    all: TILE.LEAVES, flagAll: 1,
  },
  {
    id: BLOCK.WATER, name: 'Water', solid: false, opaque: false, ao: false, atten: 2, pass: PASS_TRANSPARENT, ray: false,
    all: TILE.WATER, flagAll: 2,
  },
  { id: BLOCK.ICE, name: 'Ice', solid: true, opaque: false, ao: true, atten: 2, pass: PASS_TRANSPARENT, ray: true, all: TILE.ICE },
  { id: BLOCK.COAL_ORE, name: 'Coal Ore', solid: true, opaque: true, ao: true, atten: 15, pass: PASS_OPAQUE, ray: true, all: TILE.COAL_ORE },
  { id: BLOCK.IRON_ORE, name: 'Iron Ore', solid: true, opaque: true, ao: true, atten: 15, pass: PASS_OPAQUE, ray: true, all: TILE.IRON_ORE },
  { id: BLOCK.GOLD_ORE, name: 'Gold Ore', solid: true, opaque: true, ao: true, atten: 15, pass: PASS_OPAQUE, ray: true, all: TILE.GOLD_ORE },
  { id: BLOCK.BEDROCK, name: 'Bedrock', solid: true, opaque: true, ao: true, atten: 15, pass: PASS_OPAQUE, ray: true, all: TILE.BEDROCK },
  { id: BLOCK.GLASS, name: 'Glass', solid: true, opaque: false, ao: false, atten: 0, pass: PASS_TRANSPARENT, ray: true, all: TILE.GLASS },
  {
    id: BLOCK.SNOW_GRASS, name: 'Snowy Grass', solid: true, opaque: true, ao: true, atten: 15, pass: PASS_OPAQUE, ray: true,
    faces: [TILE.SNOW_GRASS_SIDE, TILE.SNOW_GRASS_SIDE, TILE.SNOW, TILE.DIRT, TILE.SNOW_GRASS_SIDE, TILE.SNOW_GRASS_SIDE],
  },
  {
    id: BLOCK.TALL_GRASS, name: 'Tall Grass', solid: false, opaque: false, ao: false, atten: 0, pass: PASS_CUTOUT, ray: true,
    crossQuad: true, all: TILE.TALL_GRASS, flagAll: 3,
  },
];

export const BLOCK_BY_ID = new Map();

for (const def of BLOCK_DEFS) {
  const id = def.id;
  BLOCK_BY_ID.set(id, def);
  BLOCK_SOLID[id] = def.solid ? 1 : 0;
  BLOCK_OPAQUE[id] = def.opaque ? 1 : 0;
  BLOCK_AO_OCCLUDER[id] = def.ao ? 1 : 0;
  BLOCK_LIGHT_ATTEN[id] = def.atten;
  BLOCK_PASS[id] = def.pass;
  BLOCK_CROSS_QUAD[id] = def.crossQuad ? 1 : 0;
  BLOCK_RAYCASTABLE[id] = def.ray ? 1 : 0;

  for (let f = 0; f < 6; f++) {
    BLOCK_FACE_TILES[id * 6 + f] = def.faces ? def.faces[f] : (def.all ?? 0);
    BLOCK_FLAGS[id * 6 + f] = def.faceFlags ? def.faceFlags[f] : (def.flagAll ?? 0);
  }
}

// 9-slot hotbar default blocks (plus full placeable list for scroll/palette)
export const HOTBAR_BLOCKS = [
  BLOCK.GRASS,
  BLOCK.COBBLESTONE,
  BLOCK.STONE,
  BLOCK.PLANKS,
  BLOCK.LOG,
  BLOCK.GLASS,
  BLOCK.SAND,
  BLOCK.LEAVES,
  BLOCK.GOLD_ORE,
];
