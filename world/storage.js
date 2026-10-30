// Asynchronous IndexedDB persistence for player-edited chunks.
// Stores compact sparse voxel deltas (`Uint32Array` of `(voxelIndex << 8) | blockId`)
// keyed by `${seed}:${cx},${cz}` so edited worlds save in < 1 ms and reload 100% offline.

const DB_NAME = 'argon_voxel_db';
const DB_VERSION = 1;
const STORE_NAME = 'chunk_deltas';

export class ChunkStorage {
  constructor(seed = 2026) {
    this.seed = seed;
    this.db = null;
    // In-memory cache of Map<chunkKey, Map<voxelIndex, blockId>> loaded at boot
    this.deltas = new Map();
    this.pendingSaves = new Set();
    this.saveTimer = null;
  }

  async init() {
    if (typeof indexedDB === 'undefined') return;
    await new Promise((resolve) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          db.createObjectStore(STORE_NAME);
        }
      };
      req.onsuccess = () => {
        this.db = req.result;
        resolve();
      };
      req.onerror = () => {
        console.warn('[Storage] IndexedDB unavailable; continuing with in-memory session edits.');
        resolve();
      };
    });

    if (!this.db) return;

    // Preload all saved sparse deltas for current seed into fast memory Map
    const prefix = `${this.seed}:`;
    await new Promise((resolve) => {
      const tx = this.db.transaction(STORE_NAME, 'readonly');
      const store = tx.objectStore(STORE_NAME);
      const cursorReq = store.openCursor();
      cursorReq.onsuccess = () => {
        const cursor = cursorReq.result;
        if (!cursor) {
          resolve();
          return;
        }
        const k = String(cursor.key);
        if (k.startsWith(prefix)) {
          const cKey = k.slice(prefix.length);
          const arr = cursor.value; // Uint32Array
          const m = new Map();
          if (arr && arr.length) {
            for (let i = 0; i < arr.length; i++) {
              const packed = arr[i];
              m.set(packed >>> 8, packed & 255);
            }
          }
          if (m.size > 0) this.deltas.set(cKey, m);
        }
        cursor.continue();
      };
      cursorReq.onerror = () => resolve();
    });
  }

  applyChunkDeltas(cKey, voxels) {
    const m = this.deltas.get(cKey);
    if (!m) return 0;
    let maxEditedY = 0;
    for (const [vIdx, blockId] of m.entries()) {
      voxels[vIdx] = blockId;
      const y = vIdx >> 8;
      if (y > maxEditedY) maxEditedY = y;
    }
    return maxEditedY;
  }

  recordEdit(cKey, vIdx, blockId) {
    let m = this.deltas.get(cKey);
    if (!m) {
      m = new Map();
      this.deltas.set(cKey, m);
    }
    m.set(vIdx, blockId);
    this.pendingSaves.add(cKey);
    this.scheduleFlush();
  }

  scheduleFlush() {
    if (!this.db || this.saveTimer !== null) return;
    this.saveTimer = setTimeout(() => {
      this.saveTimer = null;
      this.flushNow();
    }, 180);
  }

  flushNow() {
    if (!this.db || this.pendingSaves.size === 0) return Promise.resolve();
    const keys = Array.from(this.pendingSaves);
    this.pendingSaves.clear();

    return new Promise((resolve) => {
      const tx = this.db.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      for (const cKey of keys) {
        const m = this.deltas.get(cKey);
        if (!m || m.size === 0) continue;
        const packed = new Uint32Array(m.size);
        let i = 0;
        for (const [vIdx, blockId] of m.entries()) {
          packed[i++] = ((vIdx << 8) | (blockId & 255)) >>> 0;
        }
        store.put(packed, `${this.seed}:${cKey}`);
      }
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
    });
  }

  async clearAllEdits() {
    this.deltas.clear();
    this.pendingSaves.clear();
    if (!this.db) return;
    await new Promise((resolve) => {
      const tx = this.db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).clear();
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
    });
  }
}
