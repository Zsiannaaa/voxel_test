#!/usr/bin/env python3
"""
Build-time AI texture generation & post-processing pipeline for Argon Voxel.

Calls Google's Nano Banana 2.1 image model (`gemini-nano-banana-2.1`) via the
Gemini REST API (`generateContent`), runs automated visual QA with up to 3
prompt-adjusted retries, enforces seamless toroidal tiling (verified numerically
on a 3x3 grid), downscales & palette-quantizes to 32x32 crisp pixel art,
derives cutout/translucent alpha channels, normalizes brightness/saturation,
synthesizes animated water flow frames, and packs a 512x512 padded texture atlas
(`assets/atlas.png` + `assets/atlas.json`).

Security: Reads `GEMINI_API_KEY` strictly from the environment and passes it via
the `x-goog-api-key` HTTP header. Never hardcodes or logs the key.
"""

from __future__ import annotations

import argparse
import base64
import colorsys
import io
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parent.parent
PROMPTS_PATH = ROOT_DIR / "tools" / "prompts.json"
CACHE_DIR = ROOT_DIR / "tools" / ".cache" / "raw"
ASSETS_DIR = ROOT_DIR / "assets"
ATLAS_PNG_PATH = ASSETS_DIR / "atlas.png"
ATLAS_JSON_PATH = ASSETS_DIR / "atlas.json"
PREVIEW_PATH = ASSETS_DIR / "atlas_preview_3x3.png"

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# Block ID & per-face mapping matching chunk/blocks.js so atlas.json is self-describing
BLOCK_SCHEMA = {
    "1": {"name": "grass", "faces": {"px": "grass_side", "nx": "grass_side", "py": "grass_top", "ny": "dirt", "pz": "grass_side", "nz": "grass_side"}},
    "2": {"name": "dirt", "all": "dirt"},
    "3": {"name": "stone", "all": "stone"},
    "4": {"name": "cobblestone", "all": "cobblestone"},
    "5": {"name": "sand", "all": "sand"},
    "6": {"name": "snow", "all": "snow"},
    "7": {"name": "gravel", "all": "gravel"},
    "8": {"name": "log", "faces": {"px": "log_side", "nx": "log_side", "py": "log_top", "ny": "log_top", "pz": "log_side", "nz": "log_side"}},
    "9": {"name": "planks", "all": "planks"},
    "10": {"name": "leaves", "all": "leaves"},
    "11": {"name": "water", "all": "water", "frames": ["water", "water_1", "water_2", "water_3"]},
    "12": {"name": "ice", "all": "ice"},
    "13": {"name": "coal_ore", "all": "coal_ore"},
    "14": {"name": "iron_ore", "all": "iron_ore"},
    "15": {"name": "gold_ore", "all": "gold_ore"},
    "16": {"name": "bedrock", "all": "bedrock"},
    "17": {"name": "glass", "all": "glass"},
    "18": {"name": "snow_grass", "faces": {"px": "snow_grass_side", "nx": "snow_grass_side", "py": "snow", "ny": "dirt", "pz": "snow_grass_side", "nz": "snow_grass_side"}},
    "19": {"name": "tall_grass", "all": "tall_grass"},
}


def build_prompt(cfg: Dict[str, Any], tile_name: str, tile_spec: Dict[str, Any], attempt: int) -> str:
    parts = [cfg["style_preamble"], tile_spec["prompt"]]
    wrap_mode = tile_spec.get("wrap_mode", "xy")
    if wrap_mode == "xy":
        parts.append(cfg["seamless_suffix"])
    elif wrap_mode == "x":
        parts.append("Must tile seamlessly left-to-right along the horizontal axis.")
    if attempt > 0:
        modifiers = cfg.get("retry_modifiers", [])
        mod = modifiers[min(attempt - 1, len(modifiers) - 1)] if modifiers else ""
        if mod:
            parts.append(mod)
    return " ".join(parts)


def call_nano_banana(api_key: str, model: str, prompt: str, image_size: str = "1K") -> bytes:
    """Calls Gemini API generateContent endpoint for Nano Banana 2.1 image output."""
    url = f"{GEMINI_API_BASE}/{model}:generateContent"
    payloads = [
        {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {"aspectRatio": "1:1", "imageSize": image_size},
            },
        },
        # Graceful fallback payload in case an endpoint revision omits imageSize
        {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["IMAGE", "TEXT"],
                "imageConfig": {"aspectRatio": "1:1"},
            },
        },
    ]

    last_err: Optional[Exception] = None
    for idx, body in enumerate(payloads):
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw_json = json.loads(resp.read().decode("utf-8"))
            candidates = raw_json.get("candidates") or []
            for cand in candidates:
                parts = (cand.get("content") or {}).get("parts") or []
                for part in parts:
                    inline = part.get("inlineData") or part.get("inline_data")
                    if inline and inline.get("data"):
                        return base64.b64decode(inline["data"])
            raise RuntimeError(f"API response contained no image inlineData: {json.dumps(raw_json)[:300]}")
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            last_err = RuntimeError(f"HTTP {exc.code} from {model}: {err_body[:320]}")
            # Retry with fallback schema only on HTTP 400 invalid argument
            if exc.code == 400 and idx == 0:
                continue
            break
        except Exception as exc:
            last_err = exc
            break

    raise RuntimeError(f"Gemini API call failed: {last_err}")


def qa_raw_image(img: Image.Image, tile_name: str, tile_spec: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Automated QA checks on raw model output BEFORE seamless synthesis:
      1. Aspect ratio check (rejects non-square outputs > 3% aspect deviation).
      2. Flat-color / degenerate output check (luma/chroma std-dev & color diversity).
      3. Visible border / frame check (outer 4% margin vs interior luminance jump).
      4. Text / watermark stroke anomaly check (isolated ultra-high-contrast sharp glyph edges in corners).
    """
    w, h = img.size
    if min(w, h) < 32:
        return False, f"Image too small ({w}x{h})"
    aspect_err = abs(w - h) / float(max(w, h))
    if aspect_err > 0.03:
        return False, f"Not square ({w}x{h}, aspect error {aspect_err:.3f})"

    # Analyze on a normalized 256x256 RGB buffer for consistent thresholds
    probe = img.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
    px = list(probe.getdata())
    n = len(px)

    mean_r = sum(p[0] for p in px) / n
    mean_g = sum(p[1] for p in px) / n
    mean_b = sum(p[2] for p in px) / n
    var = sum((p[0] - mean_r) ** 2 + (p[1] - mean_g) ** 2 + (p[2] - mean_b) ** 2 for p in px) / (3.0 * n)
    std_dev = math.sqrt(var)

    # Snow and clouds naturally have gentler contrast than cobblestone, so allow slightly lower floor there
    min_std = 4.5 if tile_name in ("snow", "clouds") else 8.5
    if std_dev < min_std:
        return False, f"Mostly flat color (RGB std_dev={std_dev:.2f} < {min_std})"

    # Border / frame check (skip for intentional framed glass or directional side tiles)
    wrap_mode = tile_spec.get("wrap_mode", "xy")
    if wrap_mode == "xy":
        margin = 10  # ~4% of 256
        border_luma: List[float] = []
        interior_luma: List[float] = []
        for y in range(256):
            row_off = y * 256
            for x in range(256):
                r, g, b = px[row_off + x]
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                if x < margin or x >= 256 - margin or y < margin or y >= 256 - margin:
                    border_luma.append(lum)
                elif margin * 2 <= x < 256 - margin * 2 and margin * 2 <= y < 256 - margin * 2:
                    interior_luma.append(lum)
        b_mean = sum(border_luma) / len(border_luma)
        i_mean = sum(interior_luma) / len(interior_luma)
        b_std = math.sqrt(sum((v - b_mean) ** 2 for v in border_luma) / len(border_luma))
        if abs(b_mean - i_mean) > max(34.0, std_dev * 1.45) or (b_std < 3.0 and std_dev > 14.0):
            return False, f"Visible border/frame detected (border_luma={b_mean:.1f}, interior_luma={i_mean:.1f}, border_std={b_std:.1f})"

        # Corner text / watermark stroke detector: checks if a corner patch has >2.8x edge density of center
        def patch_edge_energy(x0: int, y0: int, pw: int, ph: int) -> float:
            acc = 0.0
            cnt = 0
            for yy in range(y0, y0 + ph - 1):
                for xx in range(x0, x0 + pw - 1):
                    p0 = px[yy * 256 + xx]
                    p1 = px[yy * 256 + xx + 1]
                    p2 = px[(yy + 1) * 256 + xx]
                    acc += abs(p0[0] - p1[0]) + abs(p0[1] - p1[1]) + abs(p0[2] - p1[2])
                    acc += abs(p0[0] - p2[0]) + abs(p0[1] - p2[1]) + abs(p0[2] - p2[2])
                    cnt += 2
            return acc / max(1, cnt)

        center_energy = max(6.0, patch_edge_energy(88, 88, 80, 80))
        corners = [
            patch_edge_energy(6, 6, 48, 32),
            patch_edge_energy(202, 6, 48, 32),
            patch_edge_energy(6, 218, 48, 32),
            patch_edge_energy(202, 218, 48, 32),
        ]
        if max(corners) > center_energy * 2.85 and max(corners) > 52.0:
            return False, f"Possible text/watermark artifact in corner (corner_edge={max(corners):.1f} vs center={center_energy:.1f})"

    return True, f"raw_ok(std={std_dev:.1f})"


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def make_seamless(img: Image.Image, wrap_mode: str) -> Image.Image:
    """
    Synthesizes a mathematically continuous wrap-around tile using cosine
    toroidal cross-fading over a half-period offset buffer.
    - wrap_mode == 'xy': full 2D torus seamless synthesis
    - wrap_mode == 'x': horizontal seamless synthesis preserving top-to-bottom strata
    - wrap_mode in ('radial', 'frame', 'none'): gentle 10% edge symmetry blend or identity
    """
    rgb = img.convert("RGB")
    if wrap_mode in ("none", "frame"):
        return rgb

    w, h = rgb.size
    src = rgb.load()

    if wrap_mode == "radial":
        # Preserve centered concentric rings while matching opposite outer borders
        out = rgb.copy()
        dst = out.load()
        band = max(2, w // 10)
        for y in range(h):
            for x in range(band):
                t = 0.5 * (1.0 - x / float(band))
                r1, g1, b1 = src[x, y]
                r2, g2, b2 = src[w - 1 - x, y]
                avg = (int(r1 * (1 - t) + r2 * t), int(g1 * (1 - t) + g2 * t), int(b1 * (1 - t) + b2 * t))
                dst[x, y] = avg
                dst[w - 1 - x, y] = avg
        for x in range(w):
            cur = out.load()
            for y in range(band):
                t = 0.5 * (1.0 - y / float(band))
                r1, g1, b1 = cur[x, y]
                r2, g2, b2 = cur[x, h - 1 - y]
                avg = (int(r1 * (1 - t) + r2 * t), int(g1 * (1 - t) + g2 * t), int(b1 * (1 - t) + b2 * t))
                cur[x, y] = avg
                cur[x, h - 1 - y] = avg
        return out

    # 1D horizontal pass first (used by both 'x' and 'xy')
    horiz = Image.new("RGB", (w, h))
    h_px = horiz.load()
    hw = w // 2
    for y in range(h):
        for x in range(w):
            # Weight 1 at center of original image, 0 at original left/right seams
            ux = abs((x - hw) / float(hw))
            wx = 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, 1.0 - ux)))
            ox = (x + hw) % w
            r0, g0, b0 = src[x, y]
            r1, g1, b1 = src[ox, y]
            h_px[x, y] = (
                int(round(r0 * wx + r1 * (1.0 - wx))),
                int(round(g0 * wx + g1 * (1.0 - wx))),
                int(round(b0 * wx + b1 * (1.0 - wx))),
            )

    if wrap_mode == "x":
        return horiz

    # 2D vertical pass on top of horizontally-seamless image -> true 2D torus
    out = Image.new("RGB", (w, h))
    o_px = out.load()
    hh = h // 2
    for y in range(h):
        uy = abs((y - hh) / float(hh))
        wy = 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, 1.0 - uy)))
        oy = (y + hh) % h
        for x in range(w):
            r0, g0, b0 = h_px[x, y]
            r1, g1, b1 = h_px[x, oy]
            o_px[x, y] = (
                int(round(r0 * wy + r1 * (1.0 - wy))),
                int(round(g0 * wy + g1 * (1.0 - wy))),
                int(round(b0 * wy + b1 * (1.0 - wy))),
            )
    return out


def verify_3x3_seam_error(tile: Image.Image, wrap_mode: str) -> Tuple[bool, float, float]:
    """
    Builds an explicit 3x3 tiled grid and numerically compares the RGB gradient
    across the tile wrap seams against the mean interior adjacent-pixel gradient.
    """
    if wrap_mode in ("none", "frame"):
        return True, 0.0, 1.0

    rgb = tile.convert("RGB")
    w, h = rgb.size
    grid = Image.new("RGB", (w * 3, h * 3))
    for gy in range(3):
        for gx in range(3):
            grid.paste(rgb, (gx * w, gy * h))
    gpx = grid.load()

    seam_diffs: List[float] = []
    interior_diffs: List[float] = []

    def diff(p1: Tuple[int, int, int], p2: Tuple[int, int, int]) -> float:
        return (abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])) / 3.0

    # Measure vertical seams at x = w and x = 2*w across middle row
    for sx in (w, 2 * w):
        for y in range(h, 2 * h):
            seam_diffs.append(diff(gpx[sx - 1, y], gpx[sx, y]))

    for y in range(h, 2 * h):
        for x in range(w + 1, 2 * w - 1):
            interior_diffs.append(diff(gpx[x - 1, y], gpx[x, y]))

    if wrap_mode == "xy":
        for sy in (h, 2 * h):
            for x in range(w, 2 * w):
                seam_diffs.append(diff(gpx[x, sy - 1], gpx[x, sy]))
        for y in range(h + 1, 2 * h - 1):
            for x in range(w, 2 * w):
                interior_diffs.append(diff(gpx[x, y - 1], gpx[x, y]))

    seam_err = sum(seam_diffs) / max(1, len(seam_diffs))
    int_err = max(2.0, sum(interior_diffs) / max(1, len(interior_diffs)))
    passed = seam_err <= max(int_err * 1.45, 18.0)
    return passed, seam_err, int_err


def normalize_hsv_and_quantize(img32: Image.Image, tile_spec: Dict[str, Any], colors: int = 16) -> Image.Image:
    """Normalizes tile brightness & saturation into cohesive target window, then palette-quantizes."""
    rgb = img32.convert("RGB")
    w, h = rgb.size
    px = list(rgb.getdata())

    hsv_list: List[Tuple[float, float, float]] = [
        colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0) for (r, g, b) in px
    ]

    # Ignore near-black cutout background pixels when computing mean brightness/saturation
    alpha_mode = tile_spec.get("alpha_mode", "opaque")
    active = [hsv for hsv in hsv_list if not (alpha_mode in ("cutout", "glass") and hsv[2] < 0.14)]
    if not active:
        active = hsv_list

    mean_s = sum(x[1] for x in active) / len(active)
    mean_v = sum(x[2] for x in active) / len(active)

    min_v, max_v = tile_spec.get("target_brightness", [0.35, 0.70])
    min_s, max_s = tile_spec.get("target_saturation", [0.05, 0.65])
    target_v = min(max_v, max(min_v, mean_v))
    target_s = min(max_s, max(min_s, mean_s))

    v_scale = target_v / max(0.04, mean_v)
    s_scale = target_s / max(0.02, mean_s) if mean_s > 0.015 else 1.0

    adjusted: List[Tuple[int, int, int]] = []
    for hue, sat, val in hsv_list:
        if alpha_mode in ("cutout", "glass") and val < 0.12:
            adjusted.append((0, 0, 0))
            continue
        ns = max(0.0, min(1.0, sat * s_scale))
        nv = max(0.0, min(1.0, val * v_scale))
        rr, gg, bb = colorsys.hsv_to_rgb(hue, ns, nv)
        adjusted.append((int(round(rr * 255)), int(round(gg * 255)), int(round(bb * 255))))

    norm_img = Image.new("RGB", (w, h))
    norm_img.putdata(adjusted)
    quant = norm_img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    return quant.convert("RGB")


def apply_alpha_channel(img32: Image.Image, tile_name: str, tile_spec: Dict[str, Any]) -> Image.Image:
    """Derives cutout or translucent alpha channels for leaves, tall_grass, glass, water, ice, clouds."""
    rgba = img32.convert("RGBA")
    w, h = rgba.size
    px = rgba.load()
    alpha_mode = tile_spec.get("alpha_mode", "opaque")

    if alpha_mode == "opaque":
        return rgba

    if alpha_mode == "translucent":
        a_val = int(tile_spec.get("alpha_value", 190))
        for y in range(h):
            for x in range(w):
                r, g, b, _ = px[x, y]
                px[x, y] = (r, g, b, a_val)
        return rgba

    if alpha_mode == "cutout":
        # Derive cutout holes from darkest interstitial pixels (~22% gaps for leaves)
        lumas = sorted(0.299 * px[x, y][0] + 0.587 * px[x, y][1] + 0.114 * px[x, y][2] for y in range(h) for x in range(w))
        thresh = max(28.0, lumas[int(len(lumas) * (0.45 if tile_name == "tall_grass" else 0.20))])
        for y in range(h):
            for x in range(w):
                r, g, b, _ = px[x, y]
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                if lum <= thresh:
                    px[x, y] = (0, 0, 0, 0)
                else:
                    px[x, y] = (r, g, b, 255)
        return rgba

    if alpha_mode == "glass":
        for y in range(h):
            for x in range(w):
                r, g, b, _ = px[x, y]
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                is_frame = x < 2 or x >= w - 2 or y < 2 or y >= h - 2
                is_glint = lum > 175
                if is_frame or is_glint:
                    px[x, y] = (max(r, 195), max(g, 225), max(b, 238), 235)
                else:
                    px[x, y] = (180, 224, 242, 54)
        return rgba

    return rgba


def synthesize_water_frames(base_water: Image.Image) -> List[Image.Image]:
    """Generates 3 seamless phase-shifted flow frames from the base water tile."""
    w, h = base_water.size
    src = base_water.load()
    frames: List[Image.Image] = []
    for f in range(1, 4):
        frame = Image.new("RGBA", (w, h))
        dst = frame.load()
        shift_x = (f * w) // 4
        shift_y = (f * h) // 4
        phase = f * (math.pi * 0.5)
        for y in range(h):
            for x in range(w):
                wx = int(round(x + shift_x + 1.5 * math.sin((y / float(h)) * math.tau + phase))) % w
                wy = int(round(y + shift_y + 1.5 * math.cos((x / float(w)) * math.tau + phase))) % h
                dst[x, y] = src[wx, wy]
        frames.append(frame)
    return frames


def paste_tile_with_bleed(atlas: Image.Image, tile: Image.Image, cell_col: int, cell_row: int, tile_size: int, pad: int, wrap_mode: str) -> None:
    """
    Packs a 32x32 tile into its 64x64 cell with 16px bleed padding on all 4 sides
    so mipmapping up to level 4 never bleeds across neighboring atlas cells.
    """
    cell_size = tile_size + pad * 2
    ox = cell_col * cell_size
    oy = cell_row * cell_size
    src = tile.load()
    dst = atlas.load()

    wrap_x = wrap_mode in ("xy", "x")
    wrap_y = wrap_mode == "xy"

    for dy in range(-pad, tile_size + pad):
        sy = (dy % tile_size) if wrap_y else max(0, min(tile_size - 1, dy))
        for dx in range(-pad, tile_size + pad):
            sx = (dx % tile_size) if wrap_x else max(0, min(tile_size - 1, dx))
            dst[ox + pad + dx, oy + pad + dy] = src[sx, sy]


def render_3x3_contact_sheet(tiles: Dict[str, Image.Image], out_path: Path) -> None:
    """Renders a labeled 3x3 tiled visual QA preview sheet for human/agent inspection."""
    names = list(tiles.keys())
    cols = 5
    rows = math.ceil(len(names) / cols)
    tile_sz = 32
    preview_sz = tile_sz * 3  # 96x96 per 3x3 preview
    cell_w = preview_sz + 24
    cell_h = preview_sz + 34

    sheet = Image.new("RGBA", (cols * cell_w + 16, rows * cell_h + 16), (24, 26, 32, 255))
    draw = ImageDraw.Draw(sheet)

    for idx, name in enumerate(names):
        col = idx % cols
        row = idx // cols
        x0 = 12 + col * cell_w
        y0 = 12 + row * cell_h

        # Checkerboard behind alpha tiles
        for cy in range(0, preview_sz, 8):
            for cx in range(0, preview_sz, 8):
                c = (54, 58, 68, 255) if ((cx // 8 + cy // 8) % 2 == 0) else (38, 41, 49, 255)
                draw.rectangle([x0 + cx, y0 + cy, x0 + cx + 7, y0 + cy + 7], fill=c)

        tile = tiles[name]
        for gy in range(3):
            for gx in range(3):
                sheet.alpha_composite(tile, (x0 + gx * tile_sz, y0 + gy * tile_sz))

        draw.rectangle([x0 - 1, y0 - 1, x0 + preview_sz, y0 + preview_sz], outline=(95, 104, 122, 255))
        draw.text((x0, y0 + preview_sz + 5), f"{idx}: {name}", fill=(224, 230, 240, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def write_atlas_outputs(processed_tiles: Dict[str, Image.Image], cfg: Dict[str, Any], source_label: str) -> None:
    tile_size = int(cfg["tile_size"])
    pad = int(cfg["padding"])
    grid = int(cfg["atlas_grid"])
    cell_size = tile_size + pad * 2
    atlas_w = grid * cell_size
    atlas_h = grid * cell_size

    atlas = Image.new("RGBA", (atlas_w, atlas_h), (0, 0, 0, 0))
    tiles_meta: Dict[str, Any] = {}

    # Pack primary tiles from prompts.json
    for name, spec in cfg["tiles"].items():
        idx = int(spec["index"])
        col = idx % grid
        row = idx // grid
        tile_img = processed_tiles[name]
        paste_tile_with_bleed(atlas, tile_img, col, row, tile_size, pad, spec.get("wrap_mode", "xy"))
        u0 = (col * cell_size + pad) / float(atlas_w)
        v0 = (row * cell_size + pad) / float(atlas_h)
        u1 = u0 + tile_size / float(atlas_w)
        v1 = v0 + tile_size / float(atlas_h)
        tiles_meta[name] = {
            "index": idx,
            "uv": [round(u0, 6), round(v0, 6), round(u1, 6), round(v1, 6)],
            "alpha_mode": spec.get("alpha_mode", "opaque"),
        }

    # Pack synthesized water animation frames (indices 22..24)
    water_frames = synthesize_water_frames(processed_tiles["water"])
    for offset, frame_img in enumerate(water_frames, start=1):
        fname = f"water_{offset}"
        idx = 21 + offset
        col = idx % grid
        row = idx // grid
        paste_tile_with_bleed(atlas, frame_img, col, row, tile_size, pad, "xy")
        u0 = (col * cell_size + pad) / float(atlas_w)
        v0 = (row * cell_size + pad) / float(atlas_h)
        u1 = u0 + tile_size / float(atlas_w)
        v1 = v0 + tile_size / float(atlas_h)
        tiles_meta[fname] = {
            "index": idx,
            "uv": [round(u0, 6), round(v0, 6), round(u1, 6), round(v1, 6)],
            "alpha_mode": "translucent",
        }
        processed_tiles[fname] = frame_img

    # Resolve block ID -> face UV dictionary
    blocks_meta: Dict[str, Any] = {}
    for bid, binfo in BLOCK_SCHEMA.items():
        faces_out: Dict[str, Any] = {}
        for face_key in ("px", "nx", "py", "ny", "pz", "nz"):
            tname = binfo["faces"][face_key] if "faces" in binfo else binfo["all"]
            tentry = tiles_meta[tname]
            faces_out[face_key] = {
                "tile": tname,
                "index": tentry["index"],
                "uv": tentry["uv"],
            }
        entry: Dict[str, Any] = {"name": binfo["name"], "faces": faces_out}
        if "frames" in binfo:
            entry["frames"] = [tiles_meta[fn]["index"] for fn in binfo["frames"]]
        blocks_meta[bid] = entry

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    atlas.save(ATLAS_PNG_PATH)

    atlas_json = {
        "version": 1,
        "generator": source_label,
        "atlasWidth": atlas_w,
        "atlasHeight": atlas_h,
        "gridCols": grid,
        "gridRows": grid,
        "tileSize": tile_size,
        "padding": pad,
        "cellSize": cell_size,
        "tiles": tiles_meta,
        "blocks": blocks_meta,
    }
    ATLAS_JSON_PATH.write_text(json.dumps(atlas_json, indent=2), encoding="utf-8")
    render_3x3_contact_sheet(processed_tiles, PREVIEW_PATH)


def process_single_raw_image(raw_img: Image.Image, name: str, spec: Dict[str, Any]) -> Tuple[bool, str, Optional[Image.Image]]:
    ok, reason = qa_raw_image(raw_img, name, spec)
    if not ok:
        return False, reason, None

    cropped = center_crop_square(raw_img)
    wrap_mode = spec.get("wrap_mode", "xy")
    seamless_hi = make_seamless(cropped, wrap_mode)

    seam_ok, seam_err, int_err = verify_3x3_seam_error(seamless_hi, wrap_mode)
    if not seam_ok:
        return False, f"Failed 3x3 seam check (seam_err={seam_err:.2f} vs interior={int_err:.2f})", None

    downscaled = seamless_hi.resize((32, 32), Image.Resampling.BOX)
    quantized = normalize_hsv_and_quantize(downscaled, spec, colors=16)

    # Second 3x3 seam check at final 32x32 pixel resolution
    seam32_ok, seam32_err, int32_err = verify_3x3_seam_error(quantized, wrap_mode)
    if not seam32_ok:
        return False, f"Failed 32x32 3x3 seam check (seam_err={seam32_err:.2f} vs interior={int32_err:.2f})", None

    final_rgba = apply_alpha_channel(quantized, name, spec)
    return True, f"{reason}, seam32={seam32_err:.2f}/{int32_err:.2f}", final_rgba


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Argon Voxel texture atlas via Nano Banana 2.1 (Gemini API).")
    parser.add_argument("--model", default=None, help="Override Gemini image model ID (default from prompts.json: gemini-nano-banana-2.1)")
    parser.add_argument("--image-size", default="1K", choices=["1K", "2K", "4K"], help="Generation resolution tier (default: 1K)")
    parser.add_argument("--force", nargs="*", default=None, help="Force regenerate cached tiles: `--force` (all) or `--force stone dirt`")
    args = parser.parse_args()

    cfg = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    model = args.model or cfg.get("model", "gemini-nano-banana-2.1")

    force_all = args.force is not None and len(args.force) == 0
    force_set = set(args.force or [])

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Check if any tile needs a live API call
    missing_tiles = [
        name for name in cfg["tiles"]
        if force_all or (name in force_set) or not (CACHE_DIR / f"{name}.png").exists()
    ]

    if missing_tiles and not api_key:
        print("=" * 72, file=sys.stderr)
        print("[STOPPED AT AI TEXTURE STEP] GEMINI_API_KEY is not set in environment.", file=sys.stderr)
        print("", file=sys.stderr)
        print("What is needed to generate live Nano Banana 2.1 textures:", file=sys.stderr)
        print("  1. Obtain a Gemini API key with billing/image generation enabled from https://aistudio.google.com/apikey", file=sys.stderr)
        print("  2. Export it in your shell (never commit it):", file=sys.stderr)
        print('     PowerShell:  $env:GEMINI_API_KEY = "YOUR_API_KEY"', file=sys.stderr)
        print('     Bash/macOS:  export GEMINI_API_KEY="YOUR_API_KEY"', file=sys.stderr)
        print("  3. Run:         python tools/generate_textures.py", file=sys.stderr)
        print("", file=sys.stderr)
        print("Fallback status: The game includes a deterministic procedural fallback", file=sys.stderr)
        print("(`generateAtlas(seed)` in `textures/atlas.js` and `node tools/bake_fallback_atlas.mjs`)", file=sys.stderr)
        print("using the exact same 512x512 padded atlas.json layout so the game runs 100% offline.", file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        return 2

    processed_tiles: Dict[str, Image.Image] = {}
    api_calls = 0

    for name, spec in cfg["tiles"].items():
        cache_file = CACHE_DIR / f"{name}.png"
        need_regen = force_all or (name in force_set) or not cache_file.exists()

        if not need_regen:
            raw_img = Image.open(cache_file)
            ok, msg, final_tile = process_single_raw_image(raw_img, name, spec)
            if ok and final_tile is not None:
                print(f"[CACHE HIT] {name:16s} -> {msg}")
                processed_tiles[name] = final_tile
                continue
            print(f"[CACHE QA REJECT] {name:16s} -> {msg}; regenerating via API...")

        success = False
        for attempt in range(4):  # Initial try + up to 3 prompt-adjusted retries
            prompt = build_prompt(cfg, name, spec, attempt)
            label = "initial" if attempt == 0 else f"retry {attempt}/3"
            print(f"[API {model}] Generating '{name}' ({label})...")
            try:
                img_bytes = call_nano_banana(api_key, model, prompt, image_size=args.image_size)
                api_calls += 1
                raw_img = Image.open(io.BytesIO(img_bytes))
                raw_img.load()
            except Exception as exc:
                print(f"  [ERROR] {name} ({label}): {exc}")
                time.sleep(1.5 * (attempt + 1))
                continue

            ok, qa_msg, final_tile = process_single_raw_image(raw_img, name, spec)
            if ok and final_tile is not None:
                raw_img.save(cache_file)
                processed_tiles[name] = final_tile
                print(f"  [QA PASS] {name:16s} -> {qa_msg}")
                success = True
                break
            else:
                reject_path = CACHE_DIR / f"{name}.rejected_try{attempt}.png"
                raw_img.save(reject_path)
                print(f"  [QA REJECT] {name:16s} ({label}) -> {qa_msg} (saved {reject_path.name})")

        if not success:
            print(f"[FATAL] Tile '{name}' failed after 3 retries.", file=sys.stderr)
            return 1

    write_atlas_outputs(processed_tiles, cfg, source_label=f"nano-banana:{model}")
    est_cost = api_calls * (0.0336 if args.image_size == "1K" else (0.0504 if args.image_size == "2K" else 0.085))
    print(f"\n[DONE] Packed {len(processed_tiles)} tiles into {ATLAS_PNG_PATH}")
    print(f"       Metadata: {ATLAS_JSON_PATH}")
    print(f"       3x3 QA Sheet: {PREVIEW_PATH}")
    print(f"       API calls this run: {api_calls} (estimated cost: ~${est_cost:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
