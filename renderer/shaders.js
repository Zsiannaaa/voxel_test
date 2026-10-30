// Minimal, high-throughput GLSL ES 3.00 shaders for Raw WebGL2.
// - Voxel shader unpacks 8-byte `uvec2` vertices via `gl.vertexAttribIPointer`, computes padded
//   atlas UVs directly from `tileId`, uses pre-fract `dFdx/dFdy` + `textureGrad()` to eliminate
//   mipmap seam lines across greedy-meshed quads, animates water & foliage, and blends smooth fog.

export const VOXEL_VERT = `#version 300 es
precision highp float;
precision highp int;

layout(location = 0) in uvec2 aData;

uniform mat4 uViewProj;
uniform vec3 uChunkOffset;
uniform vec3 uCameraPos;
uniform float uTime;
uniform float uAOStrength;
uniform float uDayFactor;      // 0.0 at midnight .. 1.0 at noon
uniform vec3 uSunDir;
uniform float uChunkFade;      // 0..1 smooth chunk streaming fade-in

out vec2 vUV;
flat out uint vTileId;
flat out uint vFlags;
out float vShade;
out float vSkyLight;
out float vFogDist;
out vec3 vWorldPos;
out vec3 vNormal;

const vec3 NORMALS[6] = vec3[6](
  vec3( 1.0,  0.0,  0.0),
  vec3(-1.0,  0.0,  0.0),
  vec3( 0.0,  1.0,  0.0),
  vec3( 0.0, -1.0,  0.0),
  vec3( 0.0,  0.0,  1.0),
  vec3( 0.0,  0.0, -1.0)
);

// Classic crisp directional voxel face multipliers
const float FACE_SHADE[6] = float[6](0.76, 0.72, 1.00, 0.54, 0.86, 0.82);
const float AO_CURVE[4]   = float[4](0.40, 0.63, 0.82, 1.00);

void main() {
  uint w0 = aData.x;
  uint w1 = aData.y;

  float lx = float(w0 & 127u);
  float ly = float((w0 >> 7u) & 511u);
  float lz = float((w0 >> 16u) & 127u);
  uint normIdx = (w0 >> 23u) & 7u;
  uint aoIdx   = (w0 >> 26u) & 3u;
  uint skyRaw  = (w0 >> 28u) & 15u;

  uint tileId  = w1 & 255u;
  float uCoord = float((w1 >> 8u) & 511u);
  float vCoord = float((w1 >> 17u) & 511u);
  float scale  = float(max(1u, (w1 >> 26u) & 7u));
  uint flags   = (w1 >> 29u) & 7u;

  vec3 localPos = vec3(lx * scale, ly, lz * scale);

  // Pull diagonal cross-quad flora corners inward so width == 1.0 block and add wind sway at top
  if (flags == 3u) {
    float s = (uCoord < 0.5) ? 0.1464 : -0.1464;
    localPos.xz += vec2(s, s);
  } else if (flags == 4u) {
    float s = (uCoord < 0.5) ? 0.1464 : -0.1464;
    localPos.xz += vec2(s, -s);
  }

  vec3 worldPos = uChunkOffset + localPos;

  if ((flags == 3u || flags == 4u) && vCoord < 0.5) {
    float wind = sin(uTime * 2.3 + worldPos.x * 0.55 + worldPos.z * 0.45) * 0.085;
    worldPos.x += wind;
    worldPos.z += wind * 0.6;
  }

  // Smooth streaming chunk sink-up transition
  if (uChunkFade < 0.999) {
    worldPos.y -= (1.0 - uChunkFade) * 6.0;
  }

  vec3 nrm = NORMALS[min(normIdx, 5u)];
  float dirLight = FACE_SHADE[min(normIdx, 5u)];
  // Dynamic subtle sun warm directional boost on sun-facing slopes during daytime
  float sunAlign = max(0.0, dot(nrm, uSunDir));
  dirLight = mix(dirLight, min(1.08, dirLight * (0.88 + 0.28 * sunAlign)), uDayFactor * 0.65);

  float ao = mix(1.0, AO_CURVE[aoIdx], uAOStrength);

  vUV = vec2(uCoord, vCoord);
  vTileId = tileId;
  vFlags = flags;
  vShade = dirLight * ao;
  vSkyLight = float(skyRaw) / 15.0;
  vWorldPos = worldPos;
  vNormal = nrm;
  vFogDist = length(worldPos - uCameraPos);

  gl_Position = uViewProj * vec4(worldPos, 1.0);
}
`;

export const VOXEL_FRAG = `#version 300 es
precision highp float;
precision highp int;

uniform sampler2D uAtlas;
uniform float uTime;
uniform float uDayFactor;
uniform vec3 uSkyHorizonColor;
uniform vec3 uSunlightColor;
uniform float uFogStart;
uniform float uFogEnd;
uniform float uFogEnabled;
uniform float uChunkFade;
uniform float uUnderwater;
uniform int uPass; // 0: opaque, 1: cutout, 2: transparent
uniform vec3 uCameraPos;
uniform vec3 uSunDir;

in vec2 vUV;
flat in uint vTileId;
flat in uint vFlags;
in float vShade;
in float vSkyLight;
in float vFogDist;
in vec3 vWorldPos;
in vec3 vNormal;

out vec4 fragColor;

// Atlas layout: 512x512 total, 8x8 grid of 64x64 cells (32x32 inner tile at +16,+16)
const float INV_ATLAS = 1.0 / 512.0;
const float TILE_UV_SPAN = 32.0 * INV_ATLAS;
const float MAX_GRAD = 8.0 * INV_ATLAS;

vec4 sampleAtlasTile(uint tid, vec2 uvContinuous) {
  float col = float(tid & 7u);
  float row = float(tid >> 3u);
  vec2 origin = (vec2(col, row) * 64.0 + 16.0) * INV_ATLAS;

  // Compute screen-space UV derivatives BEFORE fract() so greedy quad wrap boundaries never spike mip level
  vec2 dx = clamp(dFdx(uvContinuous) * TILE_UV_SPAN, vec2(-MAX_GRAD), vec2(MAX_GRAD));
  vec2 dy = clamp(dFdy(uvContinuous) * TILE_UV_SPAN, vec2(-MAX_GRAD), vec2(MAX_GRAD));
  vec2 wrapped = fract(uvContinuous);

  return textureGrad(uAtlas, origin + wrapped * TILE_UV_SPAN, dx, dy);
}

void main() {
  vec2 uv = vUV;
  uint tid = vTileId;

  // Water flow UV distortion + 4-frame animated tile cycling (tiles 12, 22, 23, 24)
  vec4 tex;
  if (vFlags == 2u) {
    vec2 flow = uv + vec2(
      uTime * 0.14 + sin(vUV.y * 3.14159 + uTime * 1.6) * 0.045,
      uTime * 0.09 + cos(vUV.x * 3.14159 + uTime * 1.4) * 0.045
    );
    float framePos = mod(uTime * 2.2, 4.0);
    int f0 = int(floor(framePos));
    int f1 = (f0 + 1) & 3;
    uint t0 = (f0 == 0) ? 12u : uint(21 + f0);
    uint t1 = (f1 == 0) ? 12u : uint(21 + f1);
    tex = mix(sampleAtlasTile(t0, flow), sampleAtlasTile(t1, flow), fract(framePos));
  } else {
    tex = sampleAtlasTile(tid, uv);
  }

  if (uPass == 1 && tex.a < 0.5) {
    discard;
  }

  // Sky light nonlinear curve + night starlight floor + cave ambient floor
  float skyCurve = pow(vSkyLight, 1.35);
  float dayAmbient = mix(0.14, 1.0, uDayFactor);
  float lightLevel = max(0.095, skyCurve * dayAmbient);

  vec3 litColor = tex.rgb * uSunlightColor * (lightLevel * vShade);

  // Subtle biome variation on grass/foliage flags
  if (vFlags == 1u || vFlags == 3u || vFlags == 4u) {
    float lush = 0.5 + 0.5 * sin(vWorldPos.x * 0.018 + vWorldPos.z * 0.021);
    litColor *= mix(vec3(0.94, 1.02, 0.90), vec3(1.05, 0.98, 0.92), lush);
  }

  // Per-pixel wave normal perturbation, Fresnel sky reflection & sun glint on water surfaces
  if (vFlags == 2u) {
    float wx = cos(vWorldPos.x * 1.35 + uTime * 2.2) * 0.11 + sin(vWorldPos.z * 1.1 - uTime * 1.7) * 0.07;
    float wz = sin(vWorldPos.z * 1.45 + uTime * 1.9) * 0.11 + cos(vWorldPos.x * 0.9 + uTime * 1.5) * 0.07;
    vec3 waveNrm = (abs(vNormal.y) > 0.5) ? normalize(vec3(wx, 1.0, wz)) : normalize(vNormal);
    vec3 viewDir = normalize(uCameraPos - vWorldPos);
    float fresnel = pow(1.0 - max(0.0, dot(viewDir, waveNrm)), 2.8);
    litColor = mix(litColor, uSkyHorizonColor * 0.96, clamp(fresnel * 0.58, 0.0, 0.58));
    vec3 halfVec = normalize(viewDir + uSunDir);
    float spec = pow(max(0.0, dot(waveNrm, halfVec)), 42.0) * max(0.0, uDayFactor);
    litColor += uSunlightColor * spec * 0.52;
    tex.a = clamp(tex.a + fresnel * 0.18, 0.0, 0.94);
  }

  // Distance fog + smooth chunk fade-in blend into horizon sky
  float fogFactor = 0.0;
  if (uFogEnabled > 0.5) {
    float t = clamp((vFogDist - uFogStart) / max(1.0, uFogEnd - uFogStart), 0.0, 1.0);
    fogFactor = t * t * (3.0 - 2.0 * t);
  }
  if (uUnderwater > 0.5) {
    float uw = clamp(vFogDist / 26.0, 0.0, 0.92);
    litColor = mix(litColor * vec3(0.58, 0.84, 1.05), vec3(0.06, 0.28, 0.48) * max(0.25, uDayFactor), uw);
    fogFactor = 0.0;
  }

  fogFactor = max(fogFactor, 1.0 - uChunkFade);
  vec3 finalRgb = mix(litColor, uSkyHorizonColor, fogFactor);
  float outAlpha = (uPass == 2) ? mix(tex.a, 0.0, fogFactor * 0.85) : 1.0;

  fragColor = vec4(finalRgb, outAlpha);
}
`;

// Fullscreen triangle sky dome shader: dynamic day/night gradient + pixel sun & moon + stars
export const SKY_VERT = `#version 300 es
precision highp float;

out vec2 vNdc;

void main() {
  // Oversized fullscreen triangle from gl_VertexID (0, 1, 2) -> zero VBO needed!
  vec2 pos = vec2((gl_VertexID == 1) ? 3.0 : -1.0, (gl_VertexID == 2) ? 3.0 : -1.0);
  vNdc = pos;
  gl_Position = vec4(pos, 0.9999, 1.0);
}
`;

export const SKY_FRAG = `#version 300 es
precision highp float;

uniform mat4 uInvViewProj;
uniform vec3 uCameraPos;
uniform vec3 uSunDir;
uniform vec3 uSkyZenithColor;
uniform vec3 uSkyHorizonColor;
uniform float uDayFactor;
uniform float uSunsetFactor;
uniform float uTime;
uniform float uUnderwater;

in vec2 vNdc;
out vec4 fragColor;

float starHash(vec3 p) {
  p = fract(p * 0.3183099 + vec3(0.1, 0.2, 0.3));
  p *= 17.0;
  return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

void main() {
  if (uUnderwater > 0.5) {
    vec3 uwCol = vec3(0.06, 0.28, 0.48) * max(0.25, uDayFactor);
    fragColor = vec4(uwCol, 1.0);
    return;
  }

  vec4 farWorld = uInvViewProj * vec4(vNdc, 1.0, 1.0);
  vec3 dir = normalize(farWorld.xyz / farWorld.w - uCameraPos);

  float elev = clamp(dir.y, -0.25, 1.0);
  float t = pow(clamp(elev, 0.0, 1.0), 0.58);
  vec3 sky = mix(uSkyHorizonColor, uSkyZenithColor, t);

  // Below horizon fog skirt matches horizon smoothly
  if (elev < 0.0) {
    sky = mix(uSkyHorizonColor, uSkyHorizonColor * 0.78, clamp(-elev * 3.0, 0.0, 1.0));
  }

  // Warm sunrise/sunset directional glow band around sun azimuth
  float sunDot = dot(dir, uSunDir);
  if (uSunsetFactor > 0.01 && elev > -0.12) {
    float glow = pow(max(0.0, sunDot), 5.0) * (1.0 - clamp(abs(elev) * 1.4, 0.0, 1.0));
    sky += vec3(0.95, 0.42, 0.14) * glow * uSunsetFactor * 0.72;
  }

  // Crisp square voxel-styled Sun disc
  vec3 sunUp = abs(uSunDir.y) > 0.95 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
  vec3 sunRight = normalize(cross(sunUp, uSunDir));
  vec3 sunOrthoUp = cross(uSunDir, sunRight);
  if (sunDot > 0.96) {
    float sx = abs(dot(dir, sunRight));
    float sy = abs(dot(dir, sunOrthoUp));
    float box = max(sx, sy);
    if (box < 0.052) {
      vec3 core = box < 0.036 ? vec3(1.0, 0.98, 0.88) : vec3(1.0, 0.82, 0.45);
      sky = mix(sky, core, smoothstep(-0.08, 0.04, uSunDir.y));
    } else if (box < 0.10) {
      sky += vec3(1.0, 0.76, 0.40) * (1.0 - (box - 0.052) / 0.048) * 0.30;
    }
  }

  // Crisp square voxel-styled Moon disc (opposite sun)
  vec3 moonDir = -uSunDir;
  float moonDot = dot(dir, moonDir);
  if (moonDot > 0.97) {
    vec3 mRight = -sunRight;
    vec3 mUp = sunOrthoUp;
    float mx = dot(dir, mRight);
    float my = dot(dir, mUp);
    float box = max(abs(mx), abs(my));
    if (box < 0.042) {
      // Subtle pixel crater shading on moon face
      vec2 mPix = floor((vec2(mx, my) / 0.042) * 4.0);
      float crater = fract(sin(dot(mPix, vec2(12.9898, 78.233))) * 43758.5453);
      vec3 mCol = crater > 0.68 ? vec3(0.70, 0.77, 0.88) : vec3(0.90, 0.94, 1.0);
      sky = mix(sky, mCol, smoothstep(-0.08, 0.05, moonDir.y));
    }
  }

  // Twinkling night starfield above horizon
  float nightVis = clamp(1.0 - uDayFactor * 1.45, 0.0, 1.0) * smoothstep(0.02, 0.25, elev);
  if (nightVis > 0.01) {
    vec3 starCell = floor(dir * 190.0);
    float h = starHash(starCell);
    if (h > 0.992) {
      vec3 cellCenter = normalize(starCell + 0.5);
      float dist = length(dir - cellCenter) * 190.0;
      if (dist < 0.42) {
        float twinkle = 0.68 + 0.32 * sin(uTime * 2.6 + h * 120.0);
        sky += vec3(0.88, 0.93, 1.0) * nightVis * twinkle;
      }
    }
  }

  fragColor = vec4(sky, 1.0);
}
`;

// Drifting cloud layer at Y = 192 sampling the `clouds` tile (tile index 19) from the atlas
export const CLOUD_VERT = `#version 300 es
precision highp float;

layout(location = 0) in vec2 aQuad; // [-1..1, -1..1]

uniform mat4 uViewProj;
uniform vec3 uCameraPos;

out vec3 vWorldPos;

void main() {
  float extent = 620.0;
  vWorldPos = vec3(uCameraPos.x + aQuad.x * extent, 194.0, uCameraPos.z + aQuad.y * extent);
  gl_Position = uViewProj * vec4(vWorldPos, 1.0);
}
`;

export const CLOUD_FRAG = `#version 300 es
precision highp float;

uniform sampler2D uAtlas;
uniform vec3 uCameraPos;
uniform float uTime;
uniform float uDayFactor;
uniform vec3 uSkyHorizonColor;
uniform vec3 uSunlightColor;

in vec3 vWorldPos;
out vec4 fragColor;

void main() {
  vec2 cloudWorld = (vWorldPos.xz + vec2(uTime * 2.4, uTime * 0.7)) * 0.0055;
  // Crisp blocky voxel cloud cells (each texel = 5.6x5.6 world blocks)
  vec2 snapped = (floor(cloudWorld * 32.0) + 0.5) / 32.0;

  // Tile 19 (clouds) is at col = 3, row = 2 in 8x8 atlas
  vec2 origin = (vec2(3.0, 2.0) * 64.0 + 16.0) / 512.0;
  vec2 span = vec2(32.0 / 512.0);
  vec4 c1 = texture(uAtlas, origin + fract(snapped) * span);
  vec4 c2 = texture(uAtlas, origin + fract(snapped * 0.37 + vec2(0.21, 0.43)) * span);

  float macro = 0.5 + 0.5 * sin(snapped.x * 5.2 + 0.7) * cos(snapped.y * 4.6 - 0.4);
  float density = (c1.r * 0.55 + c2.r * 0.45) * (0.86 + 0.16 * macro);
  // Threshold into distinct fluffy voxel cloud islands with clear azure sky between them
  if (density < 0.946) discard;

  float dist = length(vWorldPos.xz - uCameraPos.xz);
  float horizonFade = 1.0 - smoothstep(260.0, 580.0, dist);
  if (horizonFade <= 0.01) discard;

  float puffShade = smoothstep(0.944, 0.985, density);
  vec3 baseCol = mix(vec3(0.80, 0.85, 0.93), vec3(1.0), puffShade);
  vec3 lit = baseCol * uSunlightColor * mix(0.24, 1.02, uDayFactor);
  lit = mix(uSkyHorizonColor, lit, horizonFade);

  fragColor = vec4(lit, 0.84 * horizonFade);
}
`;

// Crisp 3D wireframe target block outline shader
export const LINE_VERT = `#version 300 es
precision highp float;

layout(location = 0) in vec3 aPos;
uniform mat4 uViewProj;
uniform vec3 uBlockPos;

void main() {
  // Expand box by 0.0035 blocks around [0..1]^3 to prevent Z-fighting on solid faces
  vec3 world = uBlockPos + (aPos - 0.5) * 1.007 + 0.5;
  gl_Position = uViewProj * vec4(world, 1.0);
}
`;

export const LINE_FRAG = `#version 300 es
precision highp float;
out vec4 fragColor;

void main() {
  fragColor = vec4(0.06, 0.07, 0.09, 0.82);
}
`;
