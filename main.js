// Argon Voxel — Complete Offline Open-World WebGL2 Voxel Engine Entry Point.
// Integrates:
//  - Offline padded texture atlas loader + deterministic procedural fallback (`generateAtlas`)
//  - Web Worker terrain & greedy AO meshing pools + Quadtree Multi-Tier Horizon LOD (RD 4..36+)
//  - Frustum culling + 16x16x16 Section BFS cave connectivity culling + 2.0 ms GPU upload budget
//  - First-person AABB voxel physics, swimming, fly mode, DDA raycast, block break/place & hotbar
//  - Day/night cycle (sun, moon, stars, clouds), F3 debug telemetry, Settings, Adaptive Resolution,
//    and IndexedDB sparse chunk-delta persistence.

import { loadAtlas } from './textures/atlas.js';
import { VoxelRenderer } from './renderer/gl_renderer.js';
import {
  createMat4,
  mat4Invert,
  mat4LookDir,
  mat4Multiply,
  mat4Perspective,
} from './renderer/math3d.js';
import { ChunkStorage } from './world/storage.js';
import { WorldManager } from './world/world_manager.js';
import { PlayerPhysics } from './player/physics.js';
import { PlayerController } from './player/controller.js';
import { GameHUD } from './ui/hud.js';

const SETTINGS_STORAGE_KEY = 'argon_voxel_settings_v1';
const DEFAULT_SEED = 2026;

function loadSavedSettings() {
  const defaults = {
    renderDistance: 16,
    fov: 75,
    resolutionScale: 1.0,
    adaptiveRes: true,
    fog: true,
    ao: true,
    dayNightCycle: true,
    timeOfDay: 0.21, // Crisp late-morning sunlight at boot (0=sunrise, 0.25=noon, 0.5=sunset, 0.75=midnight)
  };
  try {
    const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) return defaults;
    const parsed = JSON.parse(raw);
    return {
      ...defaults,
      renderDistance: Math.max(4, Math.min(36, Number(parsed.renderDistance) || 16)),
      fov: Math.max(60, Math.min(110, Number(parsed.fov) || 75)),
      resolutionScale: Math.max(0.5, Math.min(1.0, Number(parsed.resolutionScale) || 1.0)),
      adaptiveRes: parsed.adaptiveRes !== undefined ? Boolean(parsed.adaptiveRes) : true,
      fog: parsed.fog !== undefined ? Boolean(parsed.fog) : true,
      ao: parsed.ao !== undefined ? Boolean(parsed.ao) : true,
      dayNightCycle: parsed.dayNightCycle !== undefined ? Boolean(parsed.dayNightCycle) : true,
    };
  } catch (_e) {
    return defaults;
  }
}

function saveSettings(settings) {
  try {
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify({
      renderDistance: settings.renderDistance,
      fov: settings.fov,
      resolutionScale: settings.resolutionScale,
      adaptiveRes: settings.adaptiveRes,
      fog: settings.fog,
      ao: settings.ao,
      dayNightCycle: settings.dayNightCycle,
    }));
  } catch (_e) {}
}

async function boot() {
  const urlParams = new URLSearchParams(window.location.search);
  const seed = Number(urlParams.get('seed')) || DEFAULT_SEED;

  const canvas = document.getElementById('game-canvas');
  const renderer = new VoxelRenderer(canvas);

  const atlas = await loadAtlas(1337);
  renderer.uploadAtlas(atlas);

  const storage = new ChunkStorage(seed);
  await storage.init();

  const settings = loadSavedSettings();
  if (urlParams.has('rd')) {
    settings.renderDistance = Math.max(4, Math.min(36, Number(urlParams.get('rd')) || 16));
  }

  const world = new WorldManager(renderer, storage, seed);

  // Spawn at a scenic meadow overlook facing the forest valley, desert basin & 230-block alpine peaks
  const spawnX = 18.5;
  const spawnZ = -14.5;
  const approxSpawnY = world.getSurfaceHeightAt(spawnX, spawnZ) + 2.0;
  const player = new PlayerPhysics(spawnX, approxSpawnY, spawnZ);
  player.yaw = 2.26;
  player.pitch = -0.08;

  let hud = null;
  const controller = new PlayerController(canvas, player, world, {
    onHotbarChange: (slot, list) => {
      if (hud) hud.updateHotbarSelection(slot, list);
    },
    onToggleF3: () => {
      if (hud) hud.toggleF3();
    },
    onToggleSettings: () => {
      if (hud) hud.toggleSettings();
    },
    onNudgeTime: (delta) => {
      settings.timeOfDay = (((settings.timeOfDay + delta) % 1.0) + 1.0) % 1.0;
    },
  });

  hud = new GameHUD(atlas, controller, settings, {
    onSettingsChanged: (s) => saveSettings(s),
    onResetWorldEdits: async () => {
      await storage.clearAllEdits();
      window.location.reload();
    },
  });

  window.addEventListener('beforeunload', () => {
    storage.flushNow();
  });

  // Pre-allocated camera matrices
  const proj = createMat4();
  const view = createMat4();
  const viewProj = createMat4();
  const invViewProj = createMat4();

  // Rolling frame-time ring buffer for accurate FPS, 1% low, and adaptive resolution scaling
  const FRAME_HIST = 120;
  const frameTimesMs = new Float32Array(FRAME_HIST);
  const sortScratch = new Float32Array(FRAME_HIST);
  let histIdx = 0;
  let histCount = 0;

  let dynamicResFactor = 1.0;
  let adaptiveCooldown = 0;
  let smoothedFov = settings.fov;

  let fpsFrames = 0;
  let fpsAccum = 0;
  let displayFps = 60;
  let displayFrameMs = 16.67;
  let display1Low = 60;
  let lastNow = performance.now();

  const telemetry = {
    milestone: 'complete',
    fps: 60,
    fps1Low: 60,
    frameMs: 16.67,
    drawCalls: 0,
    triangles: 0,
    gpuMB: 0,
    canvasW: 1280,
    canvasH: 720,
    effectiveResScale: 1.0,
    mode: 'WALK',
    playerPos: player.pos,
    yaw: player.yaw,
    pitch: player.pitch,
    biome: 'Plains',
    seed,
    targetHit: null,
    world: world.stats,
    gpuName: renderer.gpuName,
    atlasSource: atlas.isFallback ? 'procedural-fallback' : (atlas.metadata?.generator || 'local-atlas'),
  };

  window.__ARGON_STATS__ = telemetry;
  window.__ARGON_ENGINE__ = {
    renderer,
    world,
    storage,
    player,
    controller,
    hud,
    settings,
  };

  function compute1PercentLow() {
    if (histCount < 10) return displayFps;
    for (let i = 0; i < histCount; i++) sortScratch[i] = frameTimesMs[i];
    const sub = sortScratch.subarray(0, histCount);
    sub.sort();
    const worstIdx = Math.min(histCount - 1, Math.floor(histCount * 0.98));
    const worstMs = Math.max(0.1, sub[worstIdx]);
    return Math.round(1000.0 / worstMs);
  }

  function frame(now) {
    const rawMs = Math.max(0.05, now - lastNow);
    const dt = Math.min(0.1, rawMs * 0.001);
    lastNow = now;

    frameTimesMs[histIdx] = rawMs;
    histIdx = (histIdx + 1) % FRAME_HIST;
    if (histCount < FRAME_HIST) histCount++;

    fpsFrames++;
    fpsAccum += dt;
    if (fpsAccum >= 0.25) {
      displayFps = Math.round(fpsFrames / fpsAccum);
      displayFrameMs = +((fpsAccum * 1000) / fpsFrames).toFixed(2);
      display1Low = compute1PercentLow();
      fpsFrames = 0;
      fpsAccum = 0;

      // Adaptive resolution scaler: lowers scale smoothly if FPS < 56, restores when FPS > 68
      if (settings.adaptiveRes) {
        if (now > adaptiveCooldown && world.stats.pendingJobs === 0) {
          if (displayFps < 56 && dynamicResFactor > 0.62) {
            dynamicResFactor = Math.max(0.60, +(dynamicResFactor - 0.06).toFixed(2));
            adaptiveCooldown = now + 750;
          } else if (displayFps > 68 && dynamicResFactor < 1.0) {
            dynamicResFactor = Math.min(1.0, +(dynamicResFactor + 0.05).toFixed(2));
            adaptiveCooldown = now + 950;
          }
        }
      } else {
        dynamicResFactor = 1.0;
      }
    }

    // Advance day/night cycle (~10 minute full day cycle by default)
    if (settings.dayNightCycle && !controller.settingsOpen) {
      settings.timeOfDay = (settings.timeOfDay + dt * (1.0 / 600.0)) % 1.0;
    }

    // 1. Player physics & voxel collision
    player.update(dt, controller.input, world);

    // 2. DDA block targeting raycast
    const hit = controller.updateRaycast();

    // 3. Smooth sprint FOV kick
    const targetFov = settings.fov + (player.sprinting ? 7.0 : 0.0);
    smoothedFov += (targetFov - smoothedFov) * Math.min(1.0, dt * 12.0);

    const effectiveScale = settings.resolutionScale * dynamicResFactor;
    const { width, height, aspect } = renderer.resize(1.5, effectiveScale);

    const farPlane = Math.max(650.0, settings.renderDistance * 16.0 + 220.0);
    mat4Perspective(proj, (smoothedFov * Math.PI) / 180.0, aspect, 0.1, farPlane);
    mat4LookDir(view, player.eyePos, player.yaw, player.pitch);
    mat4Multiply(viewProj, proj, view);
    mat4Invert(invViewProj, viewProj);

    // 4. World streaming, 2ms GPU upload budget, frustum + BFS cave culling
    world.update(dt, player.eyePos, player.yaw, viewProj, settings.renderDistance);

    // 5. Render WebGL2 passes
    renderer.renderFrame({
      viewProj,
      invViewProj,
      cameraPos: player.eyePos,
      timeSec: now * 0.001,
      timeOfDay: settings.timeOfDay,
      opaqueDraws: world.opaqueDraws,
      cutoutDraws: world.cutoutDraws,
      transparentDraws: world.transparentDraws,
      targetBlock: hit,
      settings,
      underwater: player.eyeUnderwater,
    });

    // 6. Telemetry & HUD update
    telemetry.fps = displayFps;
    telemetry.fps1Low = display1Low;
    telemetry.frameMs = displayFrameMs;
    telemetry.drawCalls = renderer.drawCalls;
    telemetry.triangles = renderer.triangles;
    telemetry.gpuMB = +(renderer.gpuBufferBytes / (1024 * 1024)).toFixed(2);
    telemetry.canvasW = width;
    telemetry.canvasH = height;
    telemetry.effectiveResScale = +effectiveScale.toFixed(2);
    telemetry.mode = player.getModeLabel();
    telemetry.yaw = player.yaw;
    telemetry.pitch = player.pitch;
    telemetry.biome = world.getBiomeNameAt(player.pos[0], player.pos[2]);
    telemetry.targetHit = hit;

    hud.update(now, telemetry);

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

boot().catch((err) => {
  console.error('[Argon Voxel Fatal Boot Error]', err);
});
