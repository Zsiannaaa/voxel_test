// First-Person Player Physics:
// Swept axis-separated AABB voxel collision, gravity, jumping, sprinting, swimming & buoyancy,
// fly toggle, and spawn ground safety guard.

import { BLOCK, BLOCK_SOLID } from '../chunk/blocks.js';
import { CHUNK_H } from '../chunk/chunk_data.js';

export const PLAYER_HALF_W = 0.30;
export const PLAYER_HEIGHT = 1.80;
export const PLAYER_EYE_H = 1.62;

const GRAVITY = -28.0;
const WATER_GRAVITY = -5.2;
const JUMP_VEL = 9.0;
const WALK_SPEED = 4.8;
const SPRINT_SPEED = 7.6;
const SWIM_SPEED = 3.9;
const FLY_SPEED = 13.5;
const FLY_SPRINT_SPEED = 24.0;

export class PlayerPhysics {
  constructor(x = 8.5, y = 90.0, z = 8.5) {
    this.pos = new Float32Array([x, y, z]); // Feet center position (x, min_y, z)
    this.vel = new Float32Array([0, 0, 0]);
    this.eyePos = new Float32Array([x, y + PLAYER_EYE_H, z]);

    this.yaw = 2.25;   // Radians (0 = -Z, pi/2 = +X)
    this.pitch = -0.10;

    this.onGround = false;
    this.flying = false;
    this.sprinting = false;
    this.inWater = false;
    this.eyeUnderwater = false;
    this.spawnSettled = false;
  }

  syncEye() {
    this.eyePos[0] = this.pos[0];
    this.eyePos[1] = this.pos[1] + PLAYER_EYE_H;
    this.eyePos[2] = this.pos[2];
  }

  getModeLabel() {
    if (this.flying) return 'FLY';
    if (this.inWater) return 'SWIM';
    if (this.sprinting) return 'SPRINT';
    return 'WALK';
  }

  /**
   * Returns true if placing a solid 1x1x1 block at `(bx, by, bz)` would overlap the player AABB.
   */
  intersectsBlock(bx, by, bz) {
    const minX = this.pos[0] - PLAYER_HALF_W + 0.002;
    const maxX = this.pos[0] + PLAYER_HALF_W - 0.002;
    const minY = this.pos[1] + 0.002;
    const maxY = this.pos[1] + PLAYER_HEIGHT - 0.002;
    const minZ = this.pos[2] - PLAYER_HALF_W + 0.002;
    const maxZ = this.pos[2] + PLAYER_HALF_W - 0.002;

    return (
      maxX > bx && minX < bx + 1 &&
      maxY > by && minY < by + 1 &&
      maxZ > bz && minZ < bz + 1
    );
  }

  checkWaterState(world) {
    const bx = Math.floor(this.pos[0]);
    const bz = Math.floor(this.pos[2]);
    const feetBlock = world.getBlock(bx, Math.floor(this.pos[1] + 0.15), bz);
    const waistBlock = world.getBlock(bx, Math.floor(this.pos[1] + 0.95), bz);
    const eyeBlock = world.getBlock(
      Math.floor(this.eyePos[0]),
      Math.floor(this.eyePos[1]),
      Math.floor(this.eyePos[2])
    );

    this.inWater = (feetBlock === BLOCK.WATER || waistBlock === BLOCK.WATER);
    this.eyeUnderwater = (eyeBlock === BLOCK.WATER);
  }

  settleSpawnIfReady(world) {
    const wx = Math.floor(this.pos[0]);
    const wz = Math.floor(this.pos[2]);
    if (!world.isChunkLoadedAt(wx, wz)) return false;

    if (!this.spawnSettled) {
      // Find topmost solid or water surface block at spawn column so player never spawns underground
      for (let y = CHUNK_H - 2; y >= 1; y--) {
        const b = world.getBlock(wx, y, wz);
        if (BLOCK_SOLID[b] || b === BLOCK.WATER) {
          this.pos[1] = y + 1.02;
          this.vel[1] = 0;
          break;
        }
      }
      this.spawnSettled = true;
      this.syncEye();
    }
    return true;
  }

  update(dt, input, world) {
    if (!this.settleSpawnIfReady(world)) {
      this.syncEye();
      return;
    }

    // Clamp & sub-step physics so frame drops can never tunnel through 1-block geometry
    const clampedDt = Math.min(0.08, dt);
    const steps = Math.max(1, Math.ceil(clampedDt / 0.02));
    const stepDt = clampedDt / steps;

    for (let s = 0; s < steps; s++) {
      this.stepPhysics(stepDt, input, world);
    }

    this.syncEye();
    this.checkWaterState(world);
  }

  stepPhysics(dt, input, world) {
    this.checkWaterState(world);

    // Horizontal wish direction in world XZ from yaw
    let fwd = (input.forward ? 1 : 0) - (input.backward ? 1 : 0);
    let strafe = (input.right ? 1 : 0) - (input.left ? 1 : 0);
    const len = Math.hypot(fwd, strafe);
    if (len > 1.0) {
      fwd /= len;
      strafe /= len;
    }

    const sinY = Math.sin(this.yaw);
    const cosY = Math.cos(this.yaw);
    // Forward is (sinY, -cosY), Right is (cosY, sinY)
    const wishX = fwd * sinY + strafe * cosY;
    const wishZ = -fwd * cosY + strafe * sinY;

    this.sprinting = Boolean(input.sprint && fwd > 0.1 && !this.inWater);

    if (this.flying) {
      const speed = input.sprint ? FLY_SPRINT_SPEED : FLY_SPEED;
      const accel = Math.min(1.0, dt * 14.0);
      this.vel[0] += (wishX * speed - this.vel[0]) * accel;
      this.vel[2] += (wishZ * speed - this.vel[2]) * accel;

      let upMove = (input.jump ? 1 : 0) - (input.descend ? 1 : 0);
      this.vel[1] += (upMove * (speed * 0.78) - this.vel[1]) * accel;

      this.moveAxis(world, 1, this.vel[1] * dt);
      this.moveAxis(world, 0, this.vel[0] * dt);
      this.moveAxis(world, 2, this.vel[2] * dt);
      return;
    }

    if (this.inWater) {
      const targetVx = wishX * SWIM_SPEED;
      const targetVz = wishZ * SWIM_SPEED;
      const drag = Math.min(1.0, dt * 8.5);
      this.vel[0] += (targetVx - this.vel[0]) * drag;
      this.vel[2] += (targetVz - this.vel[2]) * drag;

      if (input.jump) {
        this.vel[1] += (4.3 - this.vel[1]) * Math.min(1.0, dt * 10.0);
      } else {
        this.vel[1] += WATER_GRAVITY * dt;
        this.vel[1] *= Math.max(0, 1.0 - dt * 3.2);
      }

      const prevX = this.pos[0];
      const prevZ = this.pos[2];
      this.moveAxis(world, 1, this.vel[1] * dt);
      const hitWallX = this.moveAxis(world, 0, this.vel[0] * dt);
      const hitWallZ = this.moveAxis(world, 2, this.vel[2] * dt);

      // Shoreline hop-out assist when swimming against a bank while holding jump
      if (input.jump && (hitWallX || hitWallZ) && (Math.abs(wishX) > 0.1 || Math.abs(wishZ) > 0.1)) {
        const probeY = Math.floor(this.pos[1] + 1.3);
        if (!BLOCK_SOLID[world.getBlock(Math.floor(prevX + wishX * 0.6), probeY, Math.floor(prevZ + wishZ * 0.6))]) {
          this.vel[1] = 6.6;
        }
      }
      return;
    }

    // Standard Ground / Air Walking & Jumping
    const moveSpeed = this.sprinting ? SPRINT_SPEED : WALK_SPEED;
    const response = this.onGround ? 18.0 : 5.5;
    const blend = Math.min(1.0, dt * response);
    this.vel[0] += (wishX * moveSpeed - this.vel[0]) * blend;
    this.vel[2] += (wishZ * moveSpeed - this.vel[2]) * blend;

    if (this.onGround && input.jump) {
      this.vel[1] = JUMP_VEL;
      this.onGround = false;
    }

    this.vel[1] = Math.max(-48.0, this.vel[1] + GRAVITY * dt);

    this.onGround = false;
    this.moveAxis(world, 1, this.vel[1] * dt);
    this.moveAxis(world, 0, this.vel[0] * dt);
    this.moveAxis(world, 2, this.vel[2] * dt);

    // Void safety net
    if (this.pos[1] < -20) {
      this.pos[1] = world.getSurfaceHeightAt(this.pos[0], this.pos[2]) + 3.0;
      this.vel[1] = 0;
    }
  }

  /**
   * Moves player AABB along `axis` (0=X, 1=Y, 2=Z) by `delta`, resolving solid voxel collisions.
   * Returns true if movement was clipped by a solid block.
   */
  moveAxis(world, axis, delta) {
    if (Math.abs(delta) < 1e-7) return false;
    this.pos[axis] += delta;

    const eps = 1e-4;
    const minX = Math.floor(this.pos[0] - PLAYER_HALF_W + eps);
    const maxX = Math.floor(this.pos[0] + PLAYER_HALF_W - eps);
    const minY = Math.floor(this.pos[1] + eps);
    const maxY = Math.floor(this.pos[1] + PLAYER_HEIGHT - eps);
    const minZ = Math.floor(this.pos[2] - PLAYER_HALF_W + eps);
    const maxZ = Math.floor(this.pos[2] + PLAYER_HALF_W - eps);

    for (let y = minY; y <= maxY; y++) {
      if (y < 0 || y >= CHUNK_H) continue;
      for (let z = minZ; z <= maxZ; z++) {
        for (let x = minX; x <= maxX; x++) {
          if (!BLOCK_SOLID[world.getBlock(x, y, z)]) continue;

          if (axis === 1) {
            if (delta < 0) {
              this.pos[1] = y + 1.0;
              this.onGround = true;
            } else {
              this.pos[1] = y - PLAYER_HEIGHT;
            }
            this.vel[1] = 0;
            return true;
          } else if (axis === 0) {
            if (delta > 0) this.pos[0] = x - PLAYER_HALF_W;
            else this.pos[0] = x + 1.0 + PLAYER_HALF_W;
            this.vel[0] = 0;
            return true;
          } else {
            if (delta > 0) this.pos[2] = z - PLAYER_HALF_W;
            else this.pos[2] = z + 1.0 + PLAYER_HALF_W;
            this.vel[2] = 0;
            return true;
          }
        }
      }
    }

    return false;
  }
}
