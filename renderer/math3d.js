// Zero-allocation 4x4 column-major matrix math & 6-plane Gribb-Hartmann frustum culling.

export function createMat4() {
  const m = new Float32Array(16);
  m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1;
  return m;
}

export function mat4Perspective(out, fovRad, aspect, near, far) {
  const f = 1.0 / Math.tan(fovRad * 0.5);
  const nf = 1.0 / (near - far);
  out.fill(0);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (far + near) * nf;
  out[11] = -1.0;
  out[14] = (2.0 * far * near) * nf;
  return out;
}

export function mat4LookDir(out, eye, yaw, pitch) {
  // yaw = 0 looks along -Z; yaw = +pi/2 looks along +X
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);

  const fx = sy * cp;
  const fy = sp;
  const fz = -cy * cp;

  // right = normalize(cross(f, (0,1,0))) = (cy, 0, sy)
  const rx = cy;
  const ry = 0;
  const rz = sy;

  // up = cross(right, f)
  const ux = ry * fz - rz * fy;
  const uy = rz * fx - rx * fz;
  const uz = rx * fy - ry * fx;

  out[0] = rx;  out[4] = ry;  out[8]  = rz;  out[12] = -(rx * eye[0] + ry * eye[1] + rz * eye[2]);
  out[1] = ux;  out[5] = uy;  out[9]  = uz;  out[13] = -(ux * eye[0] + uy * eye[1] + uz * eye[2]);
  out[2] = -fx; out[6] = -fy; out[10] = -fz; out[14] = (fx * eye[0] + fy * eye[1] + fz * eye[2]);
  out[3] = 0;   out[7] = 0;   out[11] = 0;   out[15] = 1;
  return out;
}

export function mat4Multiply(out, a, b) {
  const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
  const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
  const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
  const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];

  for (let i = 0; i < 4; i++) {
    const o = i * 4;
    const b0 = b[o], b1 = b[o + 1], b2 = b[o + 2], b3 = b[o + 3];
    out[o]     = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    out[o + 1] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    out[o + 2] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    out[o + 3] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
  }
  return out;
}

export function mat4Invert(out, m) {
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return null;
  det = 1.0 / det;

  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return out;
}

/**
 * Extracts 6 normalized frustum planes (24 floats: [nx, ny, nz, d] * 6) from `vp`.
 */
export function extractFrustumPlanes(planesOut, vp) {
  // Left, Right, Bottom, Top, Near, Far
  const raw = [
    vp[3] + vp[0], vp[7] + vp[4], vp[11] + vp[8],  vp[15] + vp[12],
    vp[3] - vp[0], vp[7] - vp[4], vp[11] - vp[8],  vp[15] - vp[12],
    vp[3] + vp[1], vp[7] + vp[5], vp[11] + vp[9],  vp[15] + vp[13],
    vp[3] - vp[1], vp[7] - vp[5], vp[11] - vp[9],  vp[15] - vp[13],
    vp[3] + vp[2], vp[7] + vp[6], vp[11] + vp[10], vp[15] + vp[14],
    vp[3] - vp[2], vp[7] - vp[6], vp[11] - vp[10], vp[15] - vp[14],
  ];
  for (let p = 0; p < 6; p++) {
    const o = p * 4;
    const inv = 1.0 / Math.hypot(raw[o], raw[o + 1], raw[o + 2]);
    planesOut[o]     = raw[o] * inv;
    planesOut[o + 1] = raw[o + 1] * inv;
    planesOut[o + 2] = raw[o + 2] * inv;
    planesOut[o + 3] = raw[o + 3] * inv;
  }
  return planesOut;
}

/**
 * Fast P-vertex AABB vs 6-plane frustum test. Returns true if visible.
 */
export function frustumIntersectsAABB(planes, minX, minY, minZ, maxX, maxY, maxZ) {
  for (let p = 0; p < 24; p += 4) {
    const nx = planes[p];
    const ny = planes[p + 1];
    const nz = planes[p + 2];
    const px = nx >= 0 ? maxX : minX;
    const py = ny >= 0 ? maxY : minY;
    const pz = nz >= 0 ? maxZ : minZ;
    if (nx * px + ny * py + nz * pz + planes[p + 3] < 0) {
      return false;
    }
  }
  return true;
}
