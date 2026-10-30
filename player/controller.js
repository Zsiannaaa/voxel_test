// First-Person Input & Block Interaction Controller:
// Pointer lock, mouse look, WASD + Jump + Sprint + Fly toggle, 1-9 & scroll wheel hotbar,
// left-click block breaking, right-click block placement with AABB collision safety, and middle-click pick.

import { BLOCK, BLOCK_SOLID, HOTBAR_BLOCKS } from '../chunk/blocks.js';
import { CHUNK_H } from '../chunk/chunk_data.js';
import { raycastFromAngles } from './raycast.js';

const MAX_PITCH = (89.4 * Math.PI) / 180;
const MOUSE_SENSITIVITY = 0.0022;

export class PlayerController {
  constructor(canvas, player, world, callbacks = {}) {
    this.canvas = canvas;
    this.player = player;
    this.world = world;
    this.callbacks = callbacks;

    this.hotbar = [...HOTBAR_BLOCKS];
    this.selectedSlot = 0;
    this.targetHit = null;
    this.pointerLocked = false;
    this.settingsOpen = false;

    this.input = {
      forward: false,
      backward: false,
      left: false,
      right: false,
      jump: false,
      descend: false,
      sprint: false,
    };

    this.lastSpaceDownTime = 0;
    this.bindEvents();
  }

  getSelectedBlock() {
    return this.hotbar[this.selectedSlot];
  }

  selectSlot(idx) {
    this.selectedSlot = ((idx % this.hotbar.length) + this.hotbar.length) % this.hotbar.length;
    if (this.callbacks.onHotbarChange) {
      this.callbacks.onHotbarChange(this.selectedSlot, this.hotbar);
    }
  }

  requestLock() {
    if (this.settingsOpen) return;
    if (document.pointerLockElement !== this.canvas && this.canvas.requestPointerLock) {
      this.canvas.requestPointerLock();
    }
  }

  bindEvents() {
    document.addEventListener('pointerlockchange', () => {
      this.pointerLocked = (document.pointerLockElement === this.canvas);
      const promptEl = document.getElementById('pointer-prompt');
      if (promptEl) {
        promptEl.classList.toggle('locked', this.pointerLocked);
      }
      if (!this.pointerLocked) {
        this.clearInput();
      }
    });

    this.canvas.addEventListener('mousedown', (ev) => {
      if (this.settingsOpen) return;
      if (!this.pointerLocked) {
        this.requestLock();
        return;
      }

      if (ev.button === 0) {
        this.breakTargetedBlock();
      } else if (ev.button === 2) {
        this.placeTargetedBlock();
      } else if (ev.button === 1) {
        ev.preventDefault();
        this.pickTargetedBlock();
      }
    });

    window.addEventListener('contextmenu', (ev) => {
      if (ev.target === this.canvas || this.pointerLocked) {
        ev.preventDefault();
      }
    });

    window.addEventListener('mousemove', (ev) => {
      if (!this.pointerLocked || this.settingsOpen) return;
      this.player.yaw += ev.movementX * MOUSE_SENSITIVITY;
      this.player.pitch = Math.max(
        -MAX_PITCH,
        Math.min(MAX_PITCH, this.player.pitch - ev.movementY * MOUSE_SENSITIVITY)
      );
    });

    window.addEventListener('wheel', (ev) => {
      if (this.settingsOpen) return;
      if (Math.abs(ev.deltaY) < 1) return;
      this.selectSlot(this.selectedSlot + (ev.deltaY > 0 ? 1 : -1));
    }, { passive: true });

    window.addEventListener('keydown', (ev) => {
      if (ev.code === 'F3') {
        ev.preventDefault();
        if (this.callbacks.onToggleF3) this.callbacks.onToggleF3();
        return;
      }

      if (ev.code === 'KeyO' || (ev.code === 'Escape' && !this.pointerLocked)) {
        if (this.callbacks.onToggleSettings) this.callbacks.onToggleSettings();
        return;
      }

      if (this.settingsOpen) return;

      switch (ev.code) {
        case 'KeyW':
        case 'ArrowUp':
          this.input.forward = true;
          break;
        case 'KeyS':
        case 'ArrowDown':
          this.input.backward = true;
          break;
        case 'KeyA':
        case 'ArrowLeft':
          this.input.left = true;
          break;
        case 'KeyD':
        case 'ArrowRight':
          this.input.right = true;
          break;
        case 'Space': {
          ev.preventDefault();
          if (!ev.repeat) {
            const now = performance.now();
            if (now - this.lastSpaceDownTime < 280) {
              this.player.flying = !this.player.flying;
              this.player.vel[1] = 0;
              this.lastSpaceDownTime = 0;
            } else {
              this.lastSpaceDownTime = now;
            }
          }
          this.input.jump = true;
          break;
        }
        case 'ShiftLeft':
        case 'ShiftRight':
          this.input.descend = true;
          if (!this.player.flying) this.input.sprint = true;
          break;
        case 'ControlLeft':
        case 'ControlRight':
          this.input.sprint = true;
          break;
        case 'KeyC':
          this.input.descend = true;
          break;
        case 'KeyF':
          if (!ev.repeat) {
            this.player.flying = !this.player.flying;
            this.player.vel[1] = 0;
          }
          break;
        case 'BracketLeft':
          if (this.callbacks.onNudgeTime) this.callbacks.onNudgeTime(-0.03);
          break;
        case 'BracketRight':
          if (this.callbacks.onNudgeTime) this.callbacks.onNudgeTime(0.03);
          break;
        default: {
          if (ev.code.startsWith('Digit')) {
            const d = Number(ev.code.slice(5));
            if (d >= 1 && d <= 9) {
              this.selectSlot(d - 1);
            }
          }
          break;
        }
      }
    });

    window.addEventListener('keyup', (ev) => {
      switch (ev.code) {
        case 'KeyW':
        case 'ArrowUp':
          this.input.forward = false;
          break;
        case 'KeyS':
        case 'ArrowDown':
          this.input.backward = false;
          break;
        case 'KeyA':
        case 'ArrowLeft':
          this.input.left = false;
          break;
        case 'KeyD':
        case 'ArrowRight':
          this.input.right = false;
          break;
        case 'Space':
          this.input.jump = false;
          break;
        case 'ShiftLeft':
        case 'ShiftRight':
          this.input.descend = false;
          this.input.sprint = false;
          break;
        case 'ControlLeft':
        case 'ControlRight':
          this.input.sprint = false;
          break;
        case 'KeyC':
          this.input.descend = false;
          break;
      }
    });

    window.addEventListener('blur', () => this.clearInput());
  }

  clearInput() {
    this.input.forward = false;
    this.input.backward = false;
    this.input.left = false;
    this.input.right = false;
    this.input.jump = false;
    this.input.descend = false;
    this.input.sprint = false;
  }

  updateRaycast() {
    this.targetHit = raycastFromAngles(
      this.world,
      this.player.eyePos,
      this.player.yaw,
      this.player.pitch,
      6.5
    );
    return this.targetHit;
  }

  breakTargetedBlock() {
    const hit = this.updateRaycast();
    if (!hit) return false;
    // Prevent accidentally punching a hole through the bottom Y=0 world floor unless flying
    if (hit.y === 0 && hit.blockId === BLOCK.BEDROCK && !this.player.flying) {
      return false;
    }
    return this.world.setBlock(hit.x, hit.y, hit.z, BLOCK.AIR);
  }

  placeTargetedBlock() {
    const hit = this.updateRaycast();
    if (!hit || (hit.nx === 0 && hit.ny === 0 && hit.nz === 0)) return false;

    const px = hit.placeX;
    const py = hit.placeY;
    const pz = hit.placeZ;
    if (py < 0 || py >= CHUNK_H) return false;

    const existing = this.world.getBlock(px, py, pz);
    if (existing !== BLOCK.AIR && existing !== BLOCK.WATER && existing !== BLOCK.TALL_GRASS) {
      return false;
    }

    const placeBlock = this.getSelectedBlock();
    if (BLOCK_SOLID[placeBlock] && this.player.intersectsBlock(px, py, pz)) {
      return false;
    }

    return this.world.setBlock(px, py, pz, placeBlock);
  }

  pickTargetedBlock() {
    const hit = this.updateRaycast();
    if (!hit || hit.blockId === BLOCK.AIR) return;
    const existingIdx = this.hotbar.indexOf(hit.blockId);
    if (existingIdx !== -1) {
      this.selectSlot(existingIdx);
    } else {
      this.hotbar[this.selectedSlot] = hit.blockId;
      this.selectSlot(this.selectedSlot);
    }
  }
}
