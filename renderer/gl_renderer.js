// Raw WebGL2 Renderer:
// - Compiles & caches minimal GLSL ES 3.00 shader programs
// - Manages one global static quad ELEMENT_ARRAY_BUFFER shared across all chunk VAOs
// - Uploads padded 512x512 atlas with NEAREST magnification and LINEAR_MIPMAP_LINEAR minification
//   clamped to TEXTURE_MAX_LEVEL = 4 (matching the 16px wrap bleed around every 32x32 tile)
// - Executes Sky/Sun/Moon/Stars -> Clouds -> Pass 1 Opaque (front-to-back) -> Pass 2 Cutout
//   -> Pass 3 Transparent (back-to-front) -> Target Block Outline with zero per-frame allocations.

import {
  CLOUD_FRAG,
  CLOUD_VERT,
  LINE_FRAG,
  LINE_VERT,
  SKY_FRAG,
  SKY_VERT,
  VOXEL_FRAG,
  VOXEL_VERT,
} from './shaders.js';

function compileShader(gl, type, source, label) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, source);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error(`[WebGL2 Shader Error: ${label}] ${log}`);
  }
  return sh;
}

function createProgram(gl, vsSrc, fsSrc, label) {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc, `${label}:VS`);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc, `${label}:FS`);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog);
    gl.deleteProgram(prog);
    throw new Error(`[WebGL2 Link Error: ${label}] ${log}`);
  }
  return prog;
}

function getUniforms(gl, prog, names) {
  const map = {};
  for (const name of names) {
    map[name] = gl.getUniformLocation(prog, name);
  }
  return map;
}

function lerp3(out, a, b, t) {
  out[0] = a[0] + (b[0] - a[0]) * t;
  out[1] = a[1] + (b[1] - a[1]) * t;
  out[2] = a[2] + (b[2] - a[2]) * t;
  return out;
}

export class VoxelRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: true,
      stencil: false,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: false,
    });
    if (!gl) {
      throw new Error('WebGL2 is required to run Argon Voxel.');
    }
    this.gl = gl;

    // Renderer telemetry counters reset every frame
    this.drawCalls = 0;
    this.triangles = 0;
    this.gpuBufferBytes = 0;

    // Compile shader programs
    this.voxelProg = createProgram(gl, VOXEL_VERT, VOXEL_FRAG, 'Voxel');
    this.voxelU = getUniforms(gl, this.voxelProg, [
      'uViewProj', 'uChunkOffset', 'uCameraPos', 'uTime', 'uAOStrength',
      'uDayFactor', 'uSunDir', 'uChunkFade', 'uAtlas', 'uSkyHorizonColor',
      'uSunlightColor', 'uFogStart', 'uFogEnd', 'uFogEnabled', 'uUnderwater', 'uPass',
    ]);

    this.skyProg = createProgram(gl, SKY_VERT, SKY_FRAG, 'Sky');
    this.skyU = getUniforms(gl, this.skyProg, [
      'uInvViewProj', 'uCameraPos', 'uSunDir', 'uSkyZenithColor',
      'uSkyHorizonColor', 'uDayFactor', 'uSunsetFactor', 'uTime', 'uUnderwater',
    ]);

    this.cloudProg = createProgram(gl, CLOUD_VERT, CLOUD_FRAG, 'Cloud');
    this.cloudU = getUniforms(gl, this.cloudProg, [
      'uViewProj', 'uCameraPos', 'uAtlas', 'uTime', 'uDayFactor',
      'uSkyHorizonColor', 'uSunlightColor',
    ]);

    this.lineProg = createProgram(gl, LINE_VERT, LINE_FRAG, 'Outline');
    this.lineU = getUniforms(gl, this.lineProg, ['uViewProj', 'uBlockPos']);

    // Empty VAO for vertex-ID fullscreen sky triangle
    this.skyVAO = gl.createVertexArray();

    // Cloud plane VAO
    this.cloudVAO = gl.createVertexArray();
    this.cloudVBO = gl.createBuffer();
    gl.bindVertexArray(this.cloudVAO);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.cloudVBO);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1]),
      gl.STATIC_DRAW
    );
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    // Target block 12-edge wireframe cube VAO (24 vertices)
    this.outlineVAO = gl.createVertexArray();
    this.outlineVBO = gl.createBuffer();
    gl.bindVertexArray(this.outlineVAO);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.outlineVBO);
    const edges = new Float32Array([
      0,0,0, 1,0,0,  1,0,0, 1,0,1,  1,0,1, 0,0,1,  0,0,1, 0,0,0,
      0,1,0, 1,1,0,  1,1,0, 1,1,1,  1,1,1, 0,1,1,  0,1,1, 0,1,0,
      0,0,0, 0,1,0,  1,0,0, 1,1,0,  1,0,1, 1,1,1,  0,0,1, 0,1,1,
    ]);
    gl.bufferData(gl.ARRAY_BUFFER, edges, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

    gl.bindVertexArray(null);

    // Global shared quad index buffer (supports up to 65,536 quads per single chunk VAO; auto-grows if needed)
    this.sharedQuadIBO = gl.createBuffer();
    this.maxQuadsCapacity = 0;
    this.ensureQuadIndexCapacity(32768);

    this.atlasTexture = null;

    // Pre-allocated atmosphere color vectors (zero per-frame allocations)
    this.sunDir = new Float32Array([0.4, 0.85, 0.35]);
    this.skyZenith = new Float32Array(3);
    this.skyHorizon = new Float32Array(3);
    this.sunlight = new Float32Array(3);
    this.dayFactor = 1.0;
    this.sunsetFactor = 0.0;

    // Hardware renderer string for F3 overlay
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    this.gpuName = dbg
      ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL)
      : gl.getParameter(gl.RENDERER);
  }

  ensureQuadIndexCapacity(quadCount) {
    if (quadCount <= this.maxQuadsCapacity) return;
    const gl = this.gl;
    let nextCap = Math.max(16384, this.maxQuadsCapacity || 16384);
    while (nextCap < quadCount) nextCap *= 2;

    const indices = new Uint32Array(nextCap * 6);
    for (let q = 0; q < nextCap; q++) {
      const v = q * 4;
      const i = q * 6;
      indices[i] = v;
      indices[i + 1] = v + 1;
      indices[i + 2] = v + 2;
      indices[i + 3] = v;
      indices[i + 4] = v + 2;
      indices[i + 5] = v + 3;
    }

    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.sharedQuadIBO);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
    this.maxQuadsCapacity = nextCap;
  }

  uploadAtlas(atlasObj) {
    const gl = this.gl;
    if (this.atlasTexture) gl.deleteTexture(this.atlasTexture);
    const tex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);

    if (atlasObj.source) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, atlasObj.source);
    } else {
      gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.RGBA8, atlasObj.width, atlasObj.height, 0,
        gl.RGBA, gl.UNSIGNED_BYTE, atlasObj.pixels
      );
    }

    // Clamp mipmap chain to level 4 (32x32 tile -> 2x2; 16px wrap bleed prevents cross-tile bleed!)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_BASE_LEVEL, 0);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAX_LEVEL, 4);
    gl.generateMipmap(gl.TEXTURE_2D);

    // Nearest-neighbor magnification for crisp voxel pixel art + trilinear mipmaps for shimmer-free distance
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    this.atlasTexture = tex;
  }

  /**
   * Creates or updates a compact 8-byte/vertex GPU mesh handle (`{ vao, vbo, indexCount, bytes }`).
   * Reuses existing VAO/VBO objects when pooled to avoid WebGL object churn.
   */
  createOrUpdateMeshBuffer(uint32Data, existing = null) {
    const gl = this.gl;
    if (!uint32Data || uint32Data.length === 0) {
      if (existing) this.destroyMeshBuffer(existing);
      return null;
    }

    const quadCount = uint32Data.length >> 3; // 8 uint32s = 4 vertices per quad
    this.ensureQuadIndexCapacity(quadCount);

    let handle = existing;
    if (!handle) {
      const vao = gl.createVertexArray();
      const vbo = gl.createBuffer();
      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribIPointer(0, 2, gl.UNSIGNED_INT, 8, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.sharedQuadIBO);
      gl.bindVertexArray(null);
      handle = { vao, vbo, indexCount: 0, quadCount: 0, bytes: 0 };
    } else {
      this.gpuBufferBytes -= handle.bytes;
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, handle.vbo);
    gl.bufferData(gl.ARRAY_BUFFER, uint32Data, gl.STATIC_DRAW);

    handle.quadCount = quadCount;
    handle.indexCount = quadCount * 6;
    handle.bytes = uint32Data.byteLength;
    this.gpuBufferBytes += handle.bytes;
    return handle;
  }

  destroyMeshBuffer(handle) {
    if (!handle) return;
    const gl = this.gl;
    this.gpuBufferBytes -= handle.bytes;
    gl.deleteBuffer(handle.vbo);
    gl.deleteVertexArray(handle.vao);
    handle.bytes = 0;
    handle.indexCount = 0;
  }

  resize(dprCap = 1.5, resScale = 1.0) {
    const dpr = Math.min(window.devicePixelRatio || 1, dprCap) * resScale;
    const w = Math.max(64, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(64, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.gl.viewport(0, 0, w, h);
    return { width: w, height: h, aspect: w / h };
  }

  /**
   * Computes smooth time-of-day atmosphere colors (`timeOfDay` in [0, 1):
   * 0.00 = sunrise, 0.25 = noon, 0.50 = sunset, 0.75 = midnight).
   */
  updateAtmosphere(timeOfDay) {
    const angle = timeOfDay * Math.PI * 2.0;
    const sy = Math.sin(angle);
    const sx = Math.cos(angle);
    const sz = 0.28;
    const invLen = 1.0 / Math.hypot(sx, sy, sz);
    this.sunDir[0] = sx * invLen;
    this.sunDir[1] = sy * invLen;
    this.sunDir[2] = sz * invLen;

    // Smooth day factor & golden-hour sunset factor
    this.dayFactor = Math.max(0, Math.min(1, (sy + 0.16) / 0.52));
    this.sunsetFactor = Math.max(0, 1.0 - Math.abs(sy - 0.03) / 0.28);

    const dayZenith = [0.24, 0.54, 0.92];
    const dayHorizon = [0.64, 0.81, 0.98];
    const sunsetZenith = [0.22, 0.28, 0.58];
    const sunsetHorizon = [0.94, 0.55, 0.34];
    const nightZenith = [0.02, 0.04, 0.10];
    const nightHorizon = [0.06, 0.10, 0.19];

    lerp3(this.skyZenith, nightZenith, dayZenith, this.dayFactor);
    lerp3(this.skyHorizon, nightHorizon, dayHorizon, this.dayFactor);

    if (this.sunsetFactor > 0.001) {
      lerp3(this.skyZenith, this.skyZenith, sunsetZenith, this.sunsetFactor * 0.55);
      lerp3(this.skyHorizon, this.skyHorizon, sunsetHorizon, this.sunsetFactor * 0.78);
    }

    const nightTint = [0.62, 0.72, 0.96];
    const noonTint = [1.02, 1.00, 0.96];
    const sunsetTint = [1.08, 0.80, 0.60];
    lerp3(this.sunlight, nightTint, noonTint, this.dayFactor);
    if (this.sunsetFactor > 0.001) {
      lerp3(this.sunlight, this.sunlight, sunsetTint, this.sunsetFactor * 0.65);
    }
  }

  /**
   * Renders a full frame:
   *   1. Sky dome, Sun, Moon & Stars
   *   2. Drifting Cloud plane
   *   3. Pass 1: Opaque terrain (front-to-back for Early-Z)
   *   4. Pass 2: Cutout foliage (leaves & tall grass)
   *   5. Pass 3: Transparent water/ice/glass (back-to-front)
   *   6. Target block wireframe outline
   */
  renderFrame({
    viewProj,
    invViewProj,
    cameraPos,
    timeSec,
    timeOfDay,
    opaqueDraws,
    cutoutDraws,
    transparentDraws,
    targetBlock,
    settings,
    underwater,
  }) {
    const gl = this.gl;
    this.drawCalls = 0;
    this.triangles = 0;

    this.updateAtmosphere(timeOfDay);

    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearDepth(1.0);
    gl.clear(gl.DEPTH_BUFFER_BIT);

    // 1. Sky Dome (fullscreen triangle, depth writes OFF)
    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(false);
    gl.disable(gl.BLEND);
    gl.disable(gl.CULL_FACE);

    gl.useProgram(this.skyProg);
    gl.uniformMatrix4fv(this.skyU.uInvViewProj, false, invViewProj);
    gl.uniform3fv(this.skyU.uCameraPos, cameraPos);
    gl.uniform3fv(this.skyU.uSunDir, this.sunDir);
    gl.uniform3fv(this.skyU.uSkyZenithColor, this.skyZenith);
    gl.uniform3fv(this.skyU.uSkyHorizonColor, this.skyHorizon);
    gl.uniform1f(this.skyU.uDayFactor, this.dayFactor);
    gl.uniform1f(this.skyU.uSunsetFactor, this.sunsetFactor);
    gl.uniform1f(this.skyU.uTime, timeSec);
    gl.uniform1f(this.skyU.uUnderwater, underwater ? 1.0 : 0.0);

    gl.bindVertexArray(this.skyVAO);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.drawCalls++;
    this.triangles += 1;

    // Bind voxel atlas on unit 0
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.atlasTexture);

    // Fog range derived from configured render distance (in blocks)
    const maxRadiusBlocks = settings.renderDistance * 16.0;
    const fogEnd = Math.max(48.0, maxRadiusBlocks - 10.0);
    const fogStart = fogEnd * 0.56;

    // 2. Voxel Passes (Opaque -> Cutout -> Clouds -> Transparent)
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.useProgram(this.voxelProg);

    gl.uniformMatrix4fv(this.voxelU.uViewProj, false, viewProj);
    gl.uniform3fv(this.voxelU.uCameraPos, cameraPos);
    gl.uniform1f(this.voxelU.uTime, timeSec);
    gl.uniform1f(this.voxelU.uAOStrength, settings.ao ? 1.0 : 0.0);
    gl.uniform1f(this.voxelU.uDayFactor, this.dayFactor);
    gl.uniform3fv(this.voxelU.uSunDir, this.sunDir);
    gl.uniform1i(this.voxelU.uAtlas, 0);
    gl.uniform3fv(this.voxelU.uSkyHorizonColor, this.skyHorizon);
    gl.uniform3fv(this.voxelU.uSunlightColor, this.sunlight);
    gl.uniform1f(this.voxelU.uFogStart, fogStart);
    gl.uniform1f(this.voxelU.uFogEnd, fogEnd);
    gl.uniform1f(this.voxelU.uFogEnabled, settings.fog ? 1.0 : 0.0);
    gl.uniform1f(this.voxelU.uUnderwater, underwater ? 1.0 : 0.0);

    // Pass 1: Opaque (depth write ON, backface culling ON)
    gl.depthMask(true);
    gl.disable(gl.BLEND);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.uniform1i(this.voxelU.uPass, 0);

    for (let i = 0; i < opaqueDraws.length; i++) {
      const item = opaqueDraws[i];
      gl.uniform3f(this.voxelU.uChunkOffset, item.ox, 0, item.oz);
      gl.uniform1f(this.voxelU.uChunkFade, item.fade);
      gl.bindVertexArray(item.mesh.vao);
      gl.drawElements(gl.TRIANGLES, item.mesh.indexCount, gl.UNSIGNED_INT, 0);
      this.drawCalls++;
      this.triangles += item.mesh.quadCount * 2;
    }

    // Pass 2: Cutout foliage (double-sided for cross-quad grass & lush leaf interiors)
    if (cutoutDraws.length > 0) {
      gl.disable(gl.CULL_FACE);
      gl.uniform1i(this.voxelU.uPass, 1);
      for (let i = 0; i < cutoutDraws.length; i++) {
        const item = cutoutDraws[i];
        gl.uniform3f(this.voxelU.uChunkOffset, item.ox, 0, item.oz);
        gl.uniform1f(this.voxelU.uChunkFade, item.fade);
        gl.bindVertexArray(item.mesh.vao);
        gl.drawElements(gl.TRIANGLES, item.mesh.indexCount, gl.UNSIGNED_INT, 0);
        this.drawCalls++;
        this.triangles += item.mesh.quadCount * 2;
      }
    }

    // Target block wireframe outline (drawn before transparent water/glass so it shows cleanly)
    if (targetBlock) {
      gl.useProgram(this.lineProg);
      gl.uniformMatrix4fv(this.lineU.uViewProj, false, viewProj);
      gl.uniform3f(this.lineU.uBlockPos, targetBlock.x, targetBlock.y, targetBlock.z);
      gl.bindVertexArray(this.outlineVAO);
      gl.drawArrays(gl.LINES, 0, 24);
      this.drawCalls++;
    }

    // Drifting Voxel Clouds at Y = 194 (depth-tested against tall 200+ mountain peaks!)
    if (!underwater) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);
      gl.disable(gl.CULL_FACE);

      gl.useProgram(this.cloudProg);
      gl.uniformMatrix4fv(this.cloudU.uViewProj, false, viewProj);
      gl.uniform3fv(this.cloudU.uCameraPos, cameraPos);
      gl.uniform1i(this.cloudU.uAtlas, 0);
      gl.uniform1f(this.cloudU.uTime, timeSec);
      gl.uniform1f(this.cloudU.uDayFactor, this.dayFactor);
      gl.uniform3fv(this.cloudU.uSkyHorizonColor, this.skyHorizon);
      gl.uniform3fv(this.cloudU.uSunlightColor, this.sunlight);

      gl.bindVertexArray(this.cloudVAO);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      this.drawCalls++;
      this.triangles += 2;
    }

    // Pass 3: Transparent water, ice, glass (sorted back-to-front, depth write OFF)
    if (transparentDraws.length > 0) {
      gl.useProgram(this.voxelProg);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);
      if (underwater) {
        // See water surface from below when swimming underwater
        gl.disable(gl.CULL_FACE);
      } else {
        gl.enable(gl.CULL_FACE);
        gl.cullFace(gl.BACK);
      }
      gl.uniform1i(this.voxelU.uPass, 2);

      for (let i = 0; i < transparentDraws.length; i++) {
        const item = transparentDraws[i];
        gl.uniform3f(this.voxelU.uChunkOffset, item.ox, 0, item.oz);
        gl.uniform1f(this.voxelU.uChunkFade, item.fade);
        gl.bindVertexArray(item.mesh.vao);
        gl.drawElements(gl.TRIANGLES, item.mesh.indexCount, gl.UNSIGNED_INT, 0);
        this.drawCalls++;
        this.triangles += item.mesh.quadCount * 2;
      }
    }

    gl.depthMask(true);
    gl.bindVertexArray(null);
  }
}
