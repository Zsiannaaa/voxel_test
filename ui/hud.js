// HUD, 9-Slot Atlas-Rendered Hotbar, F3 Debug Telemetry Overlay & Settings Modal Manager.

import { BLOCK_BY_ID, BLOCK_FACE_TILES, FACE_PY, FACE_PZ } from '../chunk/blocks.js';
import { ATLAS_GRID, CELL_SIZE, TILE_PAD, TILE_SIZE } from '../textures/atlas.js';

function formatClock(timeOfDay) {
  // timeOfDay 0.00 = 06:00 sunrise, 0.25 = 12:00 noon, 0.50 = 18:00 sunset, 0.75 = 00:00 midnight
  const totalMinutes = Math.floor(((timeOfDay + 0.25) % 1.0) * 24 * 60);
  const hh = Math.floor(totalMinutes / 60);
  const mm = totalMinutes % 60;
  const phase = (hh >= 6 && hh < 19) ? 'Day' : 'Night';
  return `${phase} ${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}`;
}

function compassHeading(yaw) {
  // yaw = 0 -> -Z (North), +pi/2 -> +X (East), pi -> +Z (South), -pi/2 -> -X (West)
  const deg = (((yaw * 180) / Math.PI) % 360 + 360) % 360;
  if (deg >= 315 || deg < 45) return 'North (-Z)';
  if (deg >= 45 && deg < 135) return 'East (+X)';
  if (deg >= 135 && deg < 225) return 'South (+Z)';
  return 'West (-X)';
}

export class GameHUD {
  constructor(atlas, controller, settings, callbacks = {}) {
    this.atlas = atlas;
    this.controller = controller;
    this.settings = settings;
    this.callbacks = callbacks;

    this.statusBadge = document.getElementById('status-badge');
    this.timeBadge = document.getElementById('time-badge');
    this.hotbarEl = document.getElementById('hotbar');
    this.blockLabelEl = document.getElementById('selected-block-label');
    this.f3El = document.getElementById('f3-overlay');
    this.settingsPanel = document.getElementById('settings-panel');

    this.f3Visible = false;
    this.lastF3Update = 0;
    this.slotEls = [];

    this.buildHotbar(controller.hotbar, controller.selectedSlot);
    this.bindSettingsUI();
  }

  drawBlockIcon(canvas, blockId) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, TILE_SIZE, TILE_SIZE);
    ctx.imageSmoothingEnabled = false;

    // Show recognizable side face for grass/log/snow_grass, top face otherwise
    const sideTile = BLOCK_FACE_TILES[blockId * 6 + FACE_PZ];
    const topTile = BLOCK_FACE_TILES[blockId * 6 + FACE_PY];
    const tileIdx = sideTile !== topTile ? sideTile : topTile;

    const col = tileIdx % ATLAS_GRID;
    const row = Math.floor(tileIdx / ATLAS_GRID);
    const sx = col * CELL_SIZE + TILE_PAD;
    const sy = row * CELL_SIZE + TILE_PAD;

    if (this.atlas.source) {
      ctx.drawImage(this.atlas.source, sx, sy, TILE_SIZE, TILE_SIZE, 0, 0, TILE_SIZE, TILE_SIZE);
    }
  }

  buildHotbar(blocks, activeIdx) {
    this.hotbarEl.innerHTML = '';
    this.slotEls = [];

    for (let i = 0; i < blocks.length; i++) {
      const slot = document.createElement('div');
      slot.className = 'hotbar-slot' + (i === activeIdx ? ' active' : '');

      const keySpan = document.createElement('span');
      keySpan.className = 'hotbar-key';
      keySpan.textContent = String(i + 1);

      const icon = document.createElement('canvas');
      icon.width = TILE_SIZE;
      icon.height = TILE_SIZE;
      this.drawBlockIcon(icon, blocks[i]);

      slot.appendChild(keySpan);
      slot.appendChild(icon);
      slot.addEventListener('mousedown', (ev) => {
        ev.stopPropagation();
        this.controller.selectSlot(i);
      });

      this.hotbarEl.appendChild(slot);
      this.slotEls.push({ slot, icon });
    }

    this.updateHotbarSelection(activeIdx, blocks);
  }

  updateHotbarSelection(activeIdx, blocks) {
    for (let i = 0; i < this.slotEls.length; i++) {
      this.slotEls[i].slot.classList.toggle('active', i === activeIdx);
      this.drawBlockIcon(this.slotEls[i].icon, blocks[i]);
    }
    const def = BLOCK_BY_ID.get(blocks[activeIdx]);
    this.blockLabelEl.textContent = def ? def.name : 'Block';
  }

  toggleF3() {
    this.f3Visible = !this.f3Visible;
    this.f3El.classList.toggle('hidden', !this.f3Visible);
  }

  setSettingsOpen(open) {
    this.controller.settingsOpen = open;
    this.settingsPanel.classList.toggle('hidden', !open);
    if (open && document.pointerLockElement) {
      document.exitPointerLock();
    }
  }

  toggleSettings() {
    this.setSettingsOpen(!this.controller.settingsOpen);
  }

  bindSettingsUI() {
    const btnOpen = document.getElementById('btn-settings');
    const btnClose = document.getElementById('btn-close-settings');
    const btnResume = document.getElementById('btn-resume');
    const btnResetEdits = document.getElementById('btn-reset-edits');

    const inpRd = document.getElementById('inp-rd');
    const valRd = document.getElementById('val-rd');
    const inpFov = document.getElementById('inp-fov');
    const valFov = document.getElementById('val-fov');
    const inpRes = document.getElementById('inp-res');
    const valRes = document.getElementById('val-res');
    const inpTod = document.getElementById('inp-tod');
    const valTod = document.getElementById('val-tod');

    const chkAdaptive = document.getElementById('chk-adaptive');
    const chkFog = document.getElementById('chk-fog');
    const chkAo = document.getElementById('chk-ao');
    const chkCycle = document.getElementById('chk-cycle');

    // Sync initial control values
    inpRd.value = String(this.settings.renderDistance);
    valRd.textContent = String(this.settings.renderDistance);
    inpFov.value = String(this.settings.fov);
    valFov.textContent = `${this.settings.fov}°`;
    inpRes.value = String(this.settings.resolutionScale);
    valRes.textContent = `${Math.round(this.settings.resolutionScale * 100)}%`;
    inpTod.value = String(this.settings.timeOfDay);
    valTod.textContent = formatClock(this.settings.timeOfDay);

    chkAdaptive.checked = this.settings.adaptiveRes;
    chkFog.checked = this.settings.fog;
    chkAo.checked = this.settings.ao;
    chkCycle.checked = this.settings.dayNightCycle;

    const persist = () => {
      if (this.callbacks.onSettingsChanged) this.callbacks.onSettingsChanged(this.settings);
    };

    btnOpen.addEventListener('click', () => this.toggleSettings());
    btnClose.addEventListener('click', () => this.setSettingsOpen(false));
    btnResume.addEventListener('click', () => {
      this.setSettingsOpen(false);
      this.controller.requestLock();
    });

    inpRd.addEventListener('input', () => {
      this.settings.renderDistance = Number(inpRd.value);
      valRd.textContent = String(this.settings.renderDistance);
      persist();
    });
    inpFov.addEventListener('input', () => {
      this.settings.fov = Number(inpFov.value);
      valFov.textContent = `${this.settings.fov}°`;
      persist();
    });
    inpRes.addEventListener('input', () => {
      this.settings.resolutionScale = Number(inpRes.value);
      valRes.textContent = `${Math.round(this.settings.resolutionScale * 100)}%`;
      persist();
    });
    inpTod.addEventListener('input', () => {
      this.settings.timeOfDay = Number(inpTod.value);
      valTod.textContent = formatClock(this.settings.timeOfDay);
      persist();
    });

    chkAdaptive.addEventListener('change', () => {
      this.settings.adaptiveRes = chkAdaptive.checked;
      persist();
    });
    chkFog.addEventListener('change', () => {
      this.settings.fog = chkFog.checked;
      persist();
    });
    chkAo.addEventListener('change', () => {
      this.settings.ao = chkAo.checked;
      persist();
    });
    chkCycle.addEventListener('change', () => {
      this.settings.dayNightCycle = chkCycle.checked;
      persist();
    });

    btnResetEdits.addEventListener('click', () => {
      if (this.callbacks.onResetWorldEdits) this.callbacks.onResetWorldEdits();
    });

    this.inpTod = inpTod;
    this.valTod = valTod;
  }

  update(now, telemetry) {
    this.statusBadge.textContent = telemetry.mode;
    const clockText = formatClock(this.settings.timeOfDay);
    this.timeBadge.textContent = clockText;

    if (this.controller.settingsOpen && document.activeElement !== this.inpTod) {
      this.inpTod.value = String(this.settings.timeOfDay.toFixed(3));
      this.valTod.textContent = clockText;
    }

    if (!this.f3Visible || now - this.lastF3Update < 120) return;
    this.lastF3Update = now;

    const p = telemetry.playerPos;
    const cx = Math.floor(p[0] / 16);
    const cz = Math.floor(p[2] / 16);
    const hit = telemetry.targetHit;
    const hitName = hit ? (BLOCK_BY_ID.get(hit.blockId)?.name || hit.blockId) : 'None';
    const hitStr = hit ? `${hitName} @ (${hit.x}, ${hit.y}, ${hit.z})` : 'None';
    const jsHeapMB = performance.memory
      ? `${(performance.memory.usedJSHeapSize / (1024 * 1024)).toFixed(1)} MB`
      : 'N/A';

    this.f3El.textContent =
      `Argon Voxel Engine (Raw WebGL2 • Offline)\n` +
      `FPS: ${telemetry.fps} (1% low: ${telemetry.fps1Low}) | Frame: ${telemetry.frameMs.toFixed(2)} ms\n` +
      `GPU: ${telemetry.gpuName}\n` +
      `Resolution: ${telemetry.canvasW}x${telemetry.canvasH} (Scale: ${Math.round(telemetry.effectiveResScale * 100)}%) | RD: ${this.settings.renderDistance}\n` +
      `Draw Calls: ${telemetry.drawCalls} | Triangles: ${telemetry.triangles.toLocaleString()}\n` +
      `Chunks LOD0: ${telemetry.world.visibleChunks}/${telemetry.world.loadedChunks} | Horizon LOD1/2: ${telemetry.world.visibleLOD}/${telemetry.world.loadedLOD}\n` +
      `BFS Cave Culled: ${telemetry.world.culledUnderground} | Worker Jobs: ${telemetry.world.pendingJobs} | Cull: ${telemetry.world.cullMs}ms | Upload: ${telemetry.world.uploadMs}ms\n` +
      `VRAM Buffers: ${telemetry.gpuMB} MB | JS Heap: ${jsHeapMB} | Atlas: ${telemetry.atlasSource}\n` +
      `XYZ: ${p[0].toFixed(2)} / ${p[1].toFixed(2)} / ${p[2].toFixed(2)} | Chunk: (${cx}, ${cz})\n` +
      `Facing: ${compassHeading(telemetry.yaw)} | Biome: ${telemetry.biome} | Seed: ${telemetry.seed}\n` +
      `Looking at: ${hitStr}`;
  }
}
