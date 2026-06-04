#!/usr/bin/env python3
"""
image_to_stl.py — Image to STL Extrusion Converter
====================================================
Converts any image into a 3D-printable STL via height-map extrusion.
Launches a local web server with live Three.js preview.

Usage:
    python3 image_to_stl.py
    python3 image_to_stl.py --port 5050

Then open http://localhost:5000 in your browser.

Requirements:
    pip install flask pillow numpy
"""

import argparse
import base64
import io
import json
import math
import os
import struct
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps
try:
    from flask import Flask, jsonify, request, Response
except ImportError:
    Flask = None
    jsonify = request = Response = None

class _NoFlaskApp:
    def route(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def run(self, *args, **kwargs):
        raise RuntimeError("Flask is required for the web UI. Install it with: pip install flask")


app = Flask(__name__) if Flask else _NoFlaskApp()

# ─────────────────────────────────────────────
#  STL Generation
# ─────────────────────────────────────────────

UNIT_TO_MM = {
    "mm": 1.0,
    "cm": 10.0,
    "m":  1000.0,
    "in": 25.4,
    "ft": 304.8,
}

def image_to_heightmap(img: Image.Image, mode: str, detail: str) -> np.ndarray:
    """Convert image to a 2-D float32 heightmap in [0, 1]."""
    # Detail → resolution
    detail_sizes = {"low": 64, "medium": 128, "high": 256}
    size = detail_sizes.get(detail, 128)

    img = img.convert("RGBA")
    img = img.resize((size, size), Image.LANCZOS)

    r, g, b, a = img.split()
    r, g, b, a = np.array(r, float), np.array(g, float), np.array(b, float), np.array(a, float)

    if mode == "standard":
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    elif mode == "standard_color":
        gray = (r + g + b) / 3.0
    elif mode == "extrude":
        # Luminance-based: bright = high
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    elif mode == "extrude_color":
        # Saturation-based so colours stand out
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        gray = (maxc + minc) / 2.0
    else:
        gray = 0.299 * r + 0.587 * g + 0.114 * b

    # Alpha mask — transparent pixels sit at 0
    alpha_mask = a / 255.0
    gray = gray * alpha_mask

    gray = gray / 255.0  # normalise to [0,1]
    return gray.astype(np.float32)


def write_binary_stl(triangles: list, filename: str):
    """Write a list of (normal, v0, v1, v2) tuples to a binary STL."""
    with open(filename, "wb") as f:
        f.write(b"\0" * 80)  # header
        f.write(struct.pack("<I", len(triangles)))
        for normal, v0, v1, v2 in triangles:
            f.write(struct.pack("<fff", *normal))
            f.write(struct.pack("<fff", *v0))
            f.write(struct.pack("<fff", *v1))
            f.write(struct.pack("<fff", *v2))
            f.write(struct.pack("<H", 0))  # attribute byte count


def normal_of(v0, v1, v2):
    a = (v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2])
    b = (v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2])
    nx = a[1]*b[2] - a[2]*b[1]
    ny = a[2]*b[0] - a[0]*b[2]
    nz = a[0]*b[1] - a[1]*b[0]
    length = math.sqrt(nx*nx + ny*ny + nz*nz) or 1e-9
    return (nx/length, ny/length, nz/length)


def add_triangle(triangles: list, v0, v1, v2):
    triangles.append((normal_of(v0, v1, v2), v0, v1, v2))


def add_quad(triangles: list, v0, v1, v2, v3):
    add_triangle(triangles, v0, v1, v2)
    add_triangle(triangles, v1, v3, v2)


def sample_luminance(img: Image.Image, width: int, height: int) -> np.ndarray:
    """Return a resized luminance array in [0, 1]."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = img.resize((width, height), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def rounded_rect_mask(X, Y, cx, cy, w, h, r):
    qx = np.abs(X - cx) - (w / 2.0 - r)
    qy = np.abs(Y - cy) - (h / 2.0 - r)
    ox = np.maximum(qx, 0)
    oy = np.maximum(qy, 0)
    outside = np.sqrt(ox * ox + oy * oy)
    inside = np.minimum(np.maximum(qx, qy), 0)
    return outside + inside <= 0


def ellipse_mask(X, Y, cx, cy, rx, ry):
    return ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1.0


def line_mask(X, Y, x0, y0, x1, y1, half_width):
    px = X - x0
    py = Y - y0
    vx = x1 - x0
    vy = y1 - y0
    denom = vx * vx + vy * vy or 1e-9
    t = np.clip((px * vx + py * vy) / denom, 0, 1)
    dx = px - t * vx
    dy = py - t * vy
    return dx * dx + dy * dy <= half_width * half_width


def pfloat(params: dict, key: str, default: float) -> float:
    try:
        return float(params.get(key, default))
    except (TypeError, ValueError):
        return default


def pbool(params: dict, key: str, default: bool) -> bool:
    value = params.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() not in ("0", "false", "off", "no")
    return bool(value)


def crop_image_fractional(img: Image.Image, params: dict) -> Image.Image:
    """Crop source image by percent values supplied from the product UI."""
    img = ImageOps.exif_transpose(img)
    if pbool(params, "image_bg_remove", False) and img.mode in ("RGBA", "LA"):
        img = img.convert("RGBA")
    else:
        img = img.convert("RGB")
    w, h = img.size
    left = int(w * pfloat(params, "crop_left", 0) / 100.0)
    right = int(w * (100.0 - pfloat(params, "crop_right", 0)) / 100.0)
    top = int(h * pfloat(params, "crop_top", 0) / 100.0)
    bottom = int(h * (100.0 - pfloat(params, "crop_bottom", 0)) / 100.0)
    if right - left < 10 or bottom - top < 10:
        params["_crop_rect"] = (0.0, 0.0, 1.0, 1.0)
        return img
    params["_crop_rect"] = (left / w, top / h, right / w, bottom / h)
    return img.crop((left, top, right, bottom))


def resize_into_canvas(img: Image.Image, size: int, params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit the cropped image into the printable subject area and return RGB/luminance/saturation."""
    scale_x = pfloat(params, "subject_scale_x", pfloat(params, "car_scale_x", 1.0))
    scale_y = pfloat(params, "subject_scale_y", pfloat(params, "car_scale_y", 1.0))
    offset_y = pfloat(params, "subject_offset_y", pfloat(params, "car_offset_y", -0.03))

    target_w = max(8, min(size, int(size * 0.78 * scale_x)))
    target_h = max(8, min(size, int(size * 0.60 * scale_y)))
    fitted = ImageOps.contain(img, (target_w, target_h), Image.LANCZOS)

    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    present = np.zeros((size, size), dtype=bool)
    x = (size - fitted.width) // 2
    y = int(size * (0.50 - offset_y) - fitted.height / 2)
    y = max(0, min(size - fitted.height, y))
    params["_fit_rect"] = (x, y, fitted.width, fitted.height, size)
    use_alpha = pbool(params, "image_bg_remove", False) and fitted.mode in ("RGBA", "LA")
    if use_alpha:
        fitted = fitted.convert("RGBA")
        alpha = np.asarray(fitted.getchannel("A"), dtype=np.uint8)
        canvas.paste(fitted.convert("RGB"), (x, y), fitted.getchannel("A"))
        present[y:y + fitted.height, x:x + fitted.width] = alpha > 16
    else:
        canvas.paste(fitted.convert("RGB"), (x, y))
        present[y:y + fitted.height, x:x + fitted.width] = True

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    maxc = np.max(arr, axis=2)
    minc = np.min(arr, axis=2)
    sat = np.zeros_like(maxc)
    np.divide(maxc - minc, maxc, out=sat, where=maxc > 1e-6)
    return arr, lum, sat, present


def remove_subject_background(img: Image.Image, params: dict) -> Image.Image:
    """
    Build a conservative alpha cutout from border background color before product masking.

    This is intentionally local and deterministic: it removes background connected to the
    image edges, keeps the dominant centered subject, and leaves later product controls
    to refine small details.
    """
    src = ImageOps.exif_transpose(img).convert("RGB")
    w, h = src.size
    max_dim = 720
    scale = min(1.0, max_dim / max(w, h))
    work = src if scale >= 1.0 else src.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

    arr = np.asarray(work, dtype=np.float32) / 255.0
    rows, cols = arr.shape[:2]
    if rows < 8 or cols < 8:
        return src.convert("RGBA")

    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    maxc = np.max(arr, axis=2)
    minc = np.min(arr, axis=2)
    sat = np.zeros_like(maxc)
    np.divide(maxc - minc, maxc, out=sat, where=maxc > 1e-6)
    edge = edge_strength(lum)

    t = max(3, min(rows, cols) // 24)
    border = np.concatenate([
        arr[:t, :, :].reshape(-1, 3),
        arr[-t:, :, :].reshape(-1, 3),
        arr[:, :t, :].reshape(-1, 3),
        arr[:, -t:, :].reshape(-1, 3),
    ])
    border_lum = np.concatenate([lum[:t, :].ravel(), lum[-t:, :].ravel(), lum[:, :t].ravel(), lum[:, -t:].ravel()])
    border_sat = np.concatenate([sat[:t, :].ravel(), sat[-t:, :].ravel(), sat[:, :t].ravel(), sat[:, -t:].ravel()])

    bg = np.median(border, axis=0)
    bg_lum = float(np.median(border_lum))
    bg_sat = float(np.median(border_sat))
    bg_dist = np.linalg.norm(arr - bg, axis=2) / math.sqrt(3.0)
    border_dist = np.linalg.norm(border - bg, axis=1) / math.sqrt(3.0)
    noise = float(np.percentile(border_dist, 92))

    cut_strength = np.clip(pfloat(params, "image_bg_strength", 0.55), 0.0, 1.0)
    color_tol = np.clip(noise * 2.2 + 0.055 + cut_strength * 0.13, 0.07, 0.32)
    lum_tol = 0.12 + cut_strength * 0.14
    sat_tol = 0.14 + cut_strength * 0.16

    low_texture_bg = (
        (sat < max(0.18, bg_sat + sat_tol))
        & (np.abs(lum - bg_lum) < lum_tol * 1.25)
        & (edge < 0.42 + cut_strength * 0.25)
    )
    bg_like = (
        (bg_dist < color_tol)
        | low_texture_bg
        | ((np.abs(lum - bg_lum) < lum_tol) & (np.abs(sat - bg_sat) < sat_tol) & (edge < 0.55 + cut_strength * 0.25))
    )

    background = np.zeros((rows, cols), dtype=bool)
    stack = []
    for c in range(cols):
        for r in (0, rows - 1):
            if bg_like[r, c]:
                background[r, c] = True
                stack.append((r, c))
    for r in range(rows):
        for c in (0, cols - 1):
            if bg_like[r, c] and not background[r, c]:
                background[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < rows and 0 <= nc < cols and not background[nr, nc] and bg_like[nr, nc]:
                background[nr, nc] = True
                stack.append((nr, nc))

    subject = largest_center_component(~background)
    fill = float(subject.mean())
    if fill < 0.02 or fill > 0.92:
        subject = ~background

    subject = morph_mask(subject, 2, 0)
    alpha = Image.fromarray((subject.astype(np.uint8) * 255), "L")
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=0.7))
    if alpha.size != src.size:
        alpha = alpha.resize(src.size, Image.LANCZOS)

    out = src.convert("RGBA")
    out.putalpha(alpha)
    return out


def edge_strength(lum: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(lum)
    gy = np.zeros_like(lum)
    gx[:, 1:-1] = np.abs(lum[:, 2:] - lum[:, :-2]) * 0.5
    gy[1:-1, :] = np.abs(lum[2:, :] - lum[:-2, :]) * 0.5
    edge = np.sqrt(gx * gx + gy * gy)
    high = np.percentile(edge, 98) or 1.0
    return np.clip(edge / high, 0.0, 1.0)


def morph_mask(mask: np.ndarray, close_px: int, grow_px: int) -> np.ndarray:
    """Simple PIL-backed morphology for small binary product masks."""
    im = Image.fromarray((mask.astype(np.uint8) * 255), "L")
    for _ in range(max(0, close_px)):
        im = im.filter(ImageFilter.MaxFilter(3))
    for _ in range(max(0, close_px)):
        im = im.filter(ImageFilter.MinFilter(3))
    for _ in range(max(0, grow_px)):
        im = im.filter(ImageFilter.MaxFilter(3))
    return np.asarray(im) > 127


def largest_center_component(mask: np.ndarray) -> np.ndarray:
    """Keep the connected foreground part most likely to be the centered vehicle."""
    rows, cols = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best = []
    best_score = -1.0
    cx = (cols - 1) / 2.0
    cy = (rows - 1) * 0.50

    for sr in range(rows):
        for sc in range(cols):
            if not mask[sr, sc] or seen[sr, sc]:
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            pixels = []
            while stack:
                r, c = stack.pop()
                pixels.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if len(pixels) < 24:
                continue
            rr = np.array([p[0] for p in pixels], dtype=np.float32)
            cc = np.array([p[1] for p in pixels], dtype=np.float32)
            dist = math.hypot(float(cc.mean() - cx), float(rr.mean() - cy))
            score = len(pixels) - dist * 18.0
            if score > best_score:
                best_score = score
                best = pixels

    out = np.zeros_like(mask, dtype=bool)
    if best:
        rr, cc = zip(*best)
        out[np.array(rr), np.array(cc)] = True
        return out
    return mask


def flood_background_mask(present: np.ndarray, bg_like: np.ndarray, edge: np.ndarray, edge_barrier: float) -> np.ndarray:
    """Flood-fill removable background inward from the fitted image border."""
    rows, cols = present.shape
    removable = present & bg_like
    background = np.zeros_like(present, dtype=bool)
    stack = []

    pr, pc = np.where(present)
    if not len(pr):
        return background

    r0, r1 = int(pr.min()), int(pr.max())
    c0, c1 = int(pc.min()), int(pc.max())
    for c in range(c0, c1 + 1):
        for r in (r0, r1):
            if removable[r, c] and edge[r, c] < edge_barrier:
                background[r, c] = True
                stack.append((r, c))
    for r in range(r0, r1 + 1):
        for c in (c0, c1):
            if removable[r, c] and edge[r, c] < edge_barrier and not background[r, c]:
                background[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < rows and 0 <= nc < cols and not background[nr, nc]:
                if removable[nr, nc] and edge[nr, nc] < edge_barrier:
                    background[nr, nc] = True
                    stack.append((nr, nc))

    return background


def apply_eraser_strokes(mask: np.ndarray, params: dict, xx01: np.ndarray, yy01: np.ndarray) -> np.ndarray:
    """Remove user-painted regions from the extracted subject mask only."""
    strokes = params.get("erase_strokes") or []
    if not isinstance(strokes, list):
        return mask

    erased = mask.copy()
    crop = params.get("_crop_rect") or (0.0, 0.0, 1.0, 1.0)
    fit = params.get("_fit_rect")
    try:
        crop_l, crop_t, crop_r, crop_b = [float(v) for v in crop]
    except (TypeError, ValueError):
        crop_l, crop_t, crop_r, crop_b = 0.0, 0.0, 1.0, 1.0
    crop_w = max(crop_r - crop_l, 1e-6)
    crop_h = max(crop_b - crop_t, 1e-6)

    for stroke in strokes:
        if not isinstance(stroke, dict):
            continue
        try:
            x = float(stroke.get("x", -1))
            y = float(stroke.get("y", -1))
            r = float(stroke.get("r", 0.03))
        except (TypeError, ValueError):
            continue
        if r <= 0 or x < 0 or y < 0 or x > 1 or y > 1:
            continue

        if stroke.get("space") == "image" and fit:
            if not (crop_l <= x <= crop_r and crop_t <= y <= crop_b):
                continue
            try:
                fx, fy, fw, fh, canvas_size = [float(v) for v in fit]
            except (TypeError, ValueError):
                fx = fy = 0.0
                fw = fh = canvas_size = 1.0
            cx = (x - crop_l) / crop_w
            cy = (y - crop_t) / crop_h
            x = (fx + cx * fw) / max(canvas_size, 1.0)
            y = (fy + cy * fh) / max(canvas_size, 1.0)
            r = r * max(fw / crop_w, fh / crop_h) / max(canvas_size, 1.0)

        erased &= ((xx01 - x) ** 2 + (yy01 - y) ** 2) > r * r
    return erased


def smooth_subject_mask(mask: np.ndarray, smooth_px: float) -> np.ndarray:
    """Round jagged subject edges without changing the frame/backing mask."""
    if smooth_px <= 0:
        return mask
    img = Image.fromarray((mask.astype(np.uint8) * 255), "L")
    # A small close/open before blur removes staircase single-pixel bites.
    rounds = max(0, int(round(smooth_px * 0.35)))
    for _ in range(rounds):
        img = img.filter(ImageFilter.MaxFilter(3))
        img = img.filter(ImageFilter.MinFilter(3))
    img = img.filter(ImageFilter.GaussianBlur(radius=float(smooth_px)))
    return np.asarray(img) >= 128


def product_grid_size(detail: str, edge_smooth: float) -> int:
    """Choose a denser product mesh when smoothing is high enough to expose stair steps."""
    base_sizes = {"low": 160, "medium": 256, "high": 384}
    base = base_sizes.get(detail, 256)
    smooth_bonus = int(round(np.clip(edge_smooth, 0.0, 12.0) * 18.0))
    return int(min(560, base + smooth_bonus))


def smoothed_luminance_for_edges(lum: np.ndarray, smooth_px: float) -> np.ndarray:
    """Soften source edges before ridge extraction to avoid blocky raised line detail."""
    radius = max(0.0, min(float(smooth_px) * 0.28, 2.5))
    if radius <= 0.05:
        return lum
    img = Image.fromarray(np.clip(lum * 255.0, 0, 255).astype(np.uint8), "L")
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(img, dtype=np.float32) / 255.0


def make_product_height_and_mask(img: Image.Image, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a product-style mask and height field from the uploaded image.

    The subject is extracted from image contrast, color, darkness, and edges,
    then simplified into a printable relief. The circle/backing is optional.
    """
    detail = params.get("detail", "medium")
    edge_smooth = pfloat(params, "edge_smooth", 6.0)
    size = product_grid_size(detail, edge_smooth)

    yy, xx = np.mgrid[0:size, 0:size]
    X = (xx + 0.5) / size * 2.0 - 1.0
    Y = 1.0 - (yy + 0.5) / size * 2.0
    R = np.sqrt(X * X + Y * Y)

    frame_width = pfloat(params, "frame_width", 0.12)
    brace_width = pfloat(params, "brace_width", 0.085)
    hole_radius = pfloat(params, "hole_radius", 0.04)
    include_frame = pbool(params, "include_frame", True)

    outer_ring = R <= 1.0
    backing = np.zeros((size, size), dtype=bool)
    if include_frame:
        ring = outer_ring & ~(R < max(0.50, 1.0 - frame_width))
        brace_a = line_mask(X, Y, -0.86, -0.86, 0.86, 0.86, brace_width)
        brace_b = line_mask(X, Y, -0.86, 0.86, 0.86, -0.86, brace_width)
        backing = (ring | brace_a | brace_b) & outer_ring

        for hx, hy in [(-0.93, -0.93), (0.93, -0.93), (-0.93, 0.93), (0.93, 0.93)]:
            backing &= ~ellipse_mask(X, Y, hx, hy, hole_radius, hole_radius)

    if pbool(params, "image_bg_remove", False):
        img = remove_subject_background(img, params)
    cropped = crop_image_fractional(img, params)
    rgb, lum, sat, present = resize_into_canvas(cropped, size, params)
    edge = edge_strength(smoothed_luminance_for_edges(lum, edge_smooth))

    yy01 = (yy + 0.5) / size
    xx01 = (xx + 0.5) / size
    center_weight = np.clip(1.25 - np.abs(xx01 - 0.5) * 1.65 - np.abs(yy01 - 0.50) * 0.85, 0.0, 1.0)

    sat_threshold = pfloat(params, "sat_threshold", 0.22)
    dark_threshold = pfloat(params, "dark_threshold", 0.52)
    edge_threshold = pfloat(params, "edge_threshold", 0.28)
    mask_grow = int(round(pfloat(params, "mask_grow", 2)))
    mask_close = int(round(pfloat(params, "mask_close", 3)))
    min_fill = pfloat(params, "min_subject_fill", 0.05)
    max_fill = pfloat(params, "max_subject_fill", 0.34)
    bg_remove = pfloat(params, "bg_remove", 0.62)
    bg_tolerance = pfloat(params, "bg_tolerance", 0.26)
    bg_edge_barrier = pfloat(params, "bg_edge_barrier", 0.95)
    bg_border_trim = pfloat(params, "bg_border_trim", 0.04)

    pr, pc = np.where(present)
    if len(pr):
        r0, r1 = int(pr.min()), int(pr.max()) + 1
        c0, c1 = int(pc.min()), int(pc.max()) + 1
        t = max(1, min(r1 - r0, c1 - c0) // 18)
        border = np.concatenate([
            rgb[r0:r0 + t, c0:c1, :].reshape(-1, 3),
            rgb[r1 - t:r1, c0:c1, :].reshape(-1, 3),
            rgb[r0:r1, c0:c0 + t, :].reshape(-1, 3),
            rgb[r0:r1, c1 - t:c1, :].reshape(-1, 3),
        ])
    else:
        border = rgb.reshape(-1, 3)
        r0, r1, c0, c1 = 0, size, 0, size
    trim_px = max(0, int(round(min(r1 - r0, c1 - c0) * bg_border_trim)))
    subject_area = present.copy()
    if trim_px:
        subject_area[:r0 + trim_px, :] = False
        subject_area[r1 - trim_px:, :] = False
        subject_area[:, :c0 + trim_px] = False
        subject_area[:, c1 - trim_px:] = False
    bg = np.median(border, axis=0)
    bg_dist = np.linalg.norm(rgb - bg, axis=2) / math.sqrt(3.0)
    bg_threshold = pfloat(params, "bg_threshold", 0.30)
    color_subject = sat > sat_threshold
    dark_subject = lum < dark_threshold
    tolerance = bg_tolerance + bg_remove * 0.25
    bg_like = (
        (bg_dist < tolerance)
        | ((sat < sat_threshold * (1.35 + bg_remove)) & (lum > dark_threshold * (0.82 - bg_remove * 0.25)))
    )
    protected = (dark_subject | (color_subject & (bg_dist > bg_tolerance))) & (center_weight > 0.08)
    bg_like = bg_like | (present & ~protected)
    bg_like = bg_like & ~(protected & (bg_dist > bg_tolerance * (0.35 + bg_remove * 0.35)))
    background = flood_background_mask(present, bg_like, edge, bg_edge_barrier)
    for _ in range(max(0, int(round(bg_remove * 2)))):
        background = morph_mask(background, 1, 0) & present

    subject_seed = subject_area & ~background & (R < 0.92 if include_frame else R < 1.0)
    subject_seed &= (center_weight > max(0.0, 0.12 - bg_remove * 0.10))
    subject = largest_center_component(morph_mask(subject_seed, mask_close, mask_grow))

    core_seed = protected & subject_seed
    core = largest_center_component(morph_mask(core_seed, max(1, mask_close), max(1, mask_grow)))
    if subject.mean() > max_fill and core.mean() >= min_fill * 0.35:
        subject = core

    if subject.mean() < min_fill:
        fallback = (
            ((bg_dist > bg_threshold * 0.75) | color_subject | dark_subject | (edge > edge_threshold * 0.75))
            & subject_area
            & ~background
            & (center_weight > 0.08)
        )
        subject = largest_center_component(morph_mask(fallback, mask_close + 1, mask_grow + 1))

    subject = apply_eraser_strokes(subject, params, xx01, yy01)
    subject = smooth_subject_mask(subject, edge_smooth)
    if pbool(params, "image_bg_remove", False):
        subject &= present

    # Keep the detail level closer to the reference product: broad subject shape,
    # plus raised edge ridges, without reproducing every photographic texture.
    detail_level = pfloat(params, "detail_amount", 0.55)
    line_detail = (edge > edge_threshold * (1.05 - detail_level * 0.45)) & subject
    line_detail = morph_mask(line_detail, 0, max(0, int(round(pfloat(params, "edge_width", 1)))))
    line_detail = smooth_subject_mask(line_detail, min(edge_smooth * 0.45, 3.0)) & subject

    relief_strength = pfloat(params, "relief_strength", 0.30)
    edge_relief = pfloat(params, "edge_relief", 0.32)
    body_height = pfloat(params, "body_height", 0.72)
    frame_height = pfloat(params, "frame_height", 0.46)

    dark_relief = np.clip((dark_threshold - lum) / max(dark_threshold, 0.01), 0.0, 1.0)
    color_relief = np.clip((sat - sat_threshold) / max(1.0 - sat_threshold, 0.01), 0.0, 1.0)
    surface_relief = np.clip(
        dark_relief * relief_strength * 0.18 + color_relief * relief_strength * 0.10,
        0.0,
        0.14,
    )
    line_relief = np.clip(edge * edge_relief, 0.0, 0.35)
    trim = line_detail & subject
    mask = (backing | subject | trim) & outer_ring
    heights = np.zeros((size, size), dtype=np.float32)
    heights[backing] = frame_height
    heights[subject] = body_height + surface_relief[subject]
    heights[trim] = np.maximum(heights[trim], body_height + line_relief[trim] + 0.22)
    heights = np.clip(heights, 0.0, 1.0)
    heights *= mask
    return heights.astype(np.float32), mask


def masked_heightfield_to_stl(heights: np.ndarray, mask: np.ndarray, params: dict, out_path: str):
    """Convert an arbitrary 2-D mask + height field into a closed STL."""
    unit = params.get("unit", "mm")
    scale = UNIT_TO_MM[unit]
    W_phy = float(params.get("width", 120)) * scale
    H_phy = float(params.get("height", 120)) * scale
    D_phy = float(params.get("depth", 3)) * scale

    rows, cols = heights.shape
    dx = W_phy / cols
    dy = H_phy / rows
    x0 = -W_phy / 2.0
    y0 = -H_phy / 2.0
    Z = np.maximum(0.0, heights) * D_phy

    triangles = []

    def topv(rr, cc):
        zr = min(rr, rows - 1)
        zc = min(cc, cols - 1)
        return (x0 + cc * dx, y0 + rr * dy, float(Z[zr, zc]))

    def botv(rr, cc):
        return (x0 + cc * dx, y0 + rr * dy, 0.0)

    for r in range(rows):
        for c in range(cols):
            if not mask[r, c]:
                continue

            t00 = topv(r, c)
            t10 = topv(r, c + 1)
            t01 = topv(r + 1, c)
            t11 = topv(r + 1, c + 1)
            b00 = botv(r, c)
            b10 = botv(r, c + 1)
            b01 = botv(r + 1, c)
            b11 = botv(r + 1, c + 1)

            add_quad(triangles, t00, t10, t01, t11)
            add_quad(triangles, b10, b00, b11, b01)

            if r == 0 or not mask[r - 1, c]:
                add_quad(triangles, b00, b10, t00, t10)
            if r == rows - 1 or not mask[r + 1, c]:
                add_quad(triangles, b11, b01, t11, t01)
            if c == 0 or not mask[r, c - 1]:
                add_quad(triangles, b01, b00, t01, t00)
            if c == cols - 1 or not mask[r, c + 1]:
                add_quad(triangles, b10, b11, t10, t11)

    write_binary_stl(triangles, out_path)


def heightmap_to_stl(heightmap: np.ndarray, params: dict, out_path: str):
    """
    Convert a 2-D heightmap into a closed binary STL mesh.

    params keys:
        width, height, depth  – physical dimensions in chosen unit
        unit                  – "mm","cm","m","in","ft"
        base                  – base thickness in chosen unit
        tolerance             – how aggressively to simplify (0=none, currently unused)
    """
    unit   = params.get("unit", "mm")
    scale  = UNIT_TO_MM[unit]

    W_phy  = float(params.get("width",  50)) * scale   # mm
    H_phy  = float(params.get("height", 50)) * scale
    D_phy  = float(params.get("depth",  5))  * scale   # extrusion height above base
    base   = float(params.get("base",   1))  * scale

    rows, cols = heightmap.shape
    dx = W_phy / (cols - 1)
    dy = H_phy / (rows - 1)

    # Build vertex grid: z = base + heightmap * D_phy
    Z = base + heightmap * D_phy

    triangles = []

    # ── Top surface ──────────────────────────────────────────
    for r in range(rows - 1):
        for c in range(cols - 1):
            x0, y0 = c * dx, r * dy
            x1, y1 = (c+1)*dx, r*dy
            x2, y2 = c*dx,     (r+1)*dy
            x3, y3 = (c+1)*dx, (r+1)*dy

            v00 = (x0, y0, Z[r,   c  ])
            v10 = (x1, y0, Z[r,   c+1])
            v01 = (x2, y2, Z[r+1, c  ])
            v11 = (x3, y3, Z[r+1, c+1])

            triangles.append((normal_of(v00, v10, v01), v00, v10, v01))
            triangles.append((normal_of(v10, v11, v01), v10, v11, v01))

    # ── Bottom face (flat at z=0) ─────────────────────────────
    for r in range(rows - 1):
        for c in range(cols - 1):
            x0, y0 = c * dx, r * dy
            x1, y1 = (c+1)*dx, r*dy
            x2, y2 = c*dx,     (r+1)*dy
            x3, y3 = (c+1)*dx, (r+1)*dy

            v00 = (x0, y0, 0.0)
            v10 = (x1, y0, 0.0)
            v01 = (x2, y2, 0.0)
            v11 = (x3, y3, 0.0)

            # reversed winding for bottom (normal points down)
            triangles.append((normal_of(v00, v01, v10), v00, v01, v10))
            triangles.append((normal_of(v10, v01, v11), v10, v01, v11))

    # ── Side walls ───────────────────────────────────────────
    def add_wall_quad(a0, a1, b0, b1):
        triangles.append((normal_of(a0, a1, b0), a0, a1, b0))
        triangles.append((normal_of(a1, b1, b0), a1, b1, b0))

    # Front (r=0)
    for c in range(cols - 1):
        top0 = (c*dx,     0, Z[0, c  ])
        top1 = ((c+1)*dx, 0, Z[0, c+1])
        bot0 = (c*dx,     0, 0.0)
        bot1 = ((c+1)*dx, 0, 0.0)
        add_wall_quad(bot0, top0, bot1, top1)

    # Back (r=rows-1)
    for c in range(cols - 1):
        y = (rows-1)*dy
        top0 = (c*dx,     y, Z[-1, c  ])
        top1 = ((c+1)*dx, y, Z[-1, c+1])
        bot0 = (c*dx,     y, 0.0)
        bot1 = ((c+1)*dx, y, 0.0)
        add_wall_quad(bot1, top1, bot0, top0)

    # Left (c=0)
    for r in range(rows - 1):
        top0 = (0, r*dy,     Z[r,   0])
        top1 = (0, (r+1)*dy, Z[r+1, 0])
        bot0 = (0, r*dy,     0.0)
        bot1 = (0, (r+1)*dy, 0.0)
        add_wall_quad(bot1, top1, bot0, top0)

    # Right (c=cols-1)
    for r in range(rows - 1):
        x = (cols-1)*dx
        top0 = (x, r*dy,     Z[r,   -1])
        top1 = (x, (r+1)*dy, Z[r+1, -1])
        bot0 = (x, r*dy,     0.0)
        bot1 = (x, (r+1)*dy, 0.0)
        add_wall_quad(bot0, top0, bot1, top1)

    write_binary_stl(triangles, out_path)


# ─────────────────────────────────────────────
#  In-memory state (single user, local app)
# ─────────────────────────────────────────────

state = {
    "image_data": None,   # base64 PNG
    "params": {
        "filter":    "standard",
        "detail":    "medium",
        "base":      1,
        "tolerance": 0,
        "unit":      "mm",
        "width":     50,
        "height":    50,
        "depth":     5,
        "crop_left": 0,
        "crop_right": 0,
        "crop_top": 0,
        "crop_bottom": 0,
        "subject_scale_x": 1.0,
        "subject_scale_y": 1.0,
        "subject_offset_y": -0.03,
        "sat_threshold": 0.28,
        "dark_threshold": 0.45,
        "bg_threshold": 0.30,
        "bg_remove": 0.62,
        "image_bg_remove": False,
        "image_bg_strength": 0.55,
        "bg_tolerance": 0.26,
        "bg_edge_barrier": 0.95,
        "bg_border_trim": 0.04,
        "edge_threshold": 0.28,
        "detail_amount": 0.50,
        "edge_width": 1,
        "edge_smooth": 6.0,
        "mask_grow": 2,
        "mask_close": 3,
        "min_subject_fill": 0.05,
        "max_subject_fill": 0.34,
        "body_height": 0.72,
        "frame_height": 0.46,
        "include_frame": True,
        "frame_width": 0.12,
        "brace_width": 0.085,
        "hole_radius": 0.04,
        "relief_strength": 0.30,
        "edge_relief": 0.32,
        "erase_strokes": [],
    },
    "stl_ready": False,
    "stl_path":  None,
}


def generate_stl_bytes(image_b64: str, params: dict) -> bytes | None:
    """Generate STL bytes from a base64 image and params dict."""
    try:
        img_bytes = base64.b64decode(image_b64.split(",")[-1])
        img = Image.open(io.BytesIO(img_bytes))
        if params.get("filter") == "product":
            hmap, mask = make_product_height_and_mask(img, params)
            with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
                tmp = f.name
            masked_heightfield_to_stl(hmap, mask, params, tmp)
            with open(tmp, "rb") as f:
                data = f.read()
            os.unlink(tmp)
            return data

        hmap = image_to_heightmap(img, params["filter"], params["detail"])
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            tmp = f.name
        heightmap_to_stl(hmap, params, tmp)
        with open(tmp, "rb") as f:
            data = f.read()
        os.unlink(tmp)
        return data
    except Exception as e:
        print(f"[STL generation error] {e}", file=sys.stderr)
        return None


def generate_stl_file(image_path: str, out_path: str, params: dict):
    """Generate an STL directly from an image path."""
    img = Image.open(image_path)
    if params.get("filter") == "product":
        hmap, mask = make_product_height_and_mask(img, params)
        masked_heightfield_to_stl(hmap, mask, params, out_path)
    else:
        hmap = image_to_heightmap(img, params.get("filter", "standard"), params.get("detail", "medium"))
        heightmap_to_stl(hmap, params, out_path)


def heightmap_to_preview_json(heightmap: np.ndarray, params: dict) -> dict:
    """
    Return a compact JSON-serialisable mesh for Three.js BufferGeometry.
    Only the top surface + a flat base quad are included for speed.
    """
    rows, cols = heightmap.shape
    unit  = params.get("unit", "mm")
    scale = UNIT_TO_MM[unit]
    W = float(params.get("width",  50)) * scale
    H = float(params.get("height", 50)) * scale
    D = float(params.get("depth",  5))  * scale
    base = float(params.get("base", 1)) * scale

    dx = W / (cols - 1)
    dy = H / (rows - 1)
    Z = (base + heightmap * D).tolist()

    positions = []
    colors    = []

    def push_tri(v0, v1, v2, h0, h1, h2):
        for v, h in ((v0, h0), (v1, h1), (v2, h2)):
            positions.extend(v)
            t = min(1.0, h)
            colors.extend([0.2 + 0.6*t, 0.5 + 0.4*t, 0.8 - 0.3*t])

    for r in range(rows - 1):
        for c in range(cols - 1):
            x0, y0, z00 = c*dx,     r*dy,     Z[r  ][c  ]
            x1, y1, z10 = (c+1)*dx, r*dy,     Z[r  ][c+1]
            x2, y2, z01 = c*dx,     (r+1)*dy, Z[r+1][c  ]
            x3, y3, z11 = (c+1)*dx, (r+1)*dy, Z[r+1][c+1]

            h00 = heightmap[r,   c  ]
            h10 = heightmap[r,   c+1]
            h01 = heightmap[r+1, c  ]
            h11 = heightmap[r+1, c+1]

            push_tri((x0,z00,y0),(x1,z10,y1),(x2,z01,y2), h00,h10,h01)
            push_tri((x1,z10,y1),(x3,z11,y3),(x2,z01,y2), h10,h11,h01)

    # flat base at z=0
    base_positions = [
        0,0,0,     W,0,0,     0,0,H,
        W,0,0,     W,0,H,     0,0,H,
    ]
    base_colors = [0.15,0.15,0.2]*6

    return {
        "positions": positions + base_positions,
        "colors":    colors    + base_colors,
        "width": W, "height": H, "depth": D + base,
    }


def product_to_preview_json(heights: np.ndarray, mask: np.ndarray, params: dict) -> dict:
    """Preview mesh for the product mode, including its non-rectangular outline."""
    unit = params.get("unit", "mm")
    scale = UNIT_TO_MM[unit]
    W = float(params.get("width", 120)) * scale
    H = float(params.get("height", 120)) * scale
    D = float(params.get("depth", 3)) * scale

    rows, cols = heights.shape
    dx = W / cols
    dy = H / rows
    x0 = -W / 2.0
    y0 = -H / 2.0
    Z = np.maximum(0.0, heights) * D
    positions = []
    colors = []

    def push_tri(v0, v1, v2, h0, h1, h2):
        for v, h in ((v0, h0), (v1, h1), (v2, h2)):
            positions.extend(v)
            if h > 0.85:
                colors.extend([0.88, 0.82, 0.62])
            elif h > 0.60:
                colors.extend([0.48, 0.52, 0.57])
            else:
                colors.extend([0.30, 0.32, 0.34])

    for r in range(rows):
        for c in range(cols):
            if not mask[r, c]:
                continue
            z00 = float(Z[r, c])
            z10 = float(Z[r, min(c + 1, cols - 1)])
            z01 = float(Z[min(r + 1, rows - 1), c])
            z11 = float(Z[min(r + 1, rows - 1), min(c + 1, cols - 1)])
            x00, y00 = x0 + c * dx, y0 + r * dy
            x10, y10 = x0 + (c + 1) * dx, y00
            x01, y01 = x00, y0 + (r + 1) * dy
            x11, y11 = x10, y01
            push_tri((x00, z00, y00), (x10, z10, y10), (x01, z01, y01),
                     heights[r, c], heights[r, c], heights[r, c])
            push_tri((x10, z10, y10), (x11, z11, y11), (x01, z01, y01),
                     heights[r, c], heights[r, c], heights[r, c])

    return {
        "positions": positions,
        "colors": colors,
        "width": W,
        "height": H,
        "depth": D,
    }


# ─────────────────────────────────────────────
#  Flask routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")


@app.route("/api/upload", methods=["POST"])
def api_upload():
    data = request.get_json()
    state["image_data"] = data.get("image")
    return jsonify({"ok": True})


@app.route("/api/preview", methods=["POST"])
def api_preview():
    data    = request.get_json()
    img_b64 = data.get("image") or state.get("image_data")
    params  = data.get("params", state["params"])

    if not img_b64:
        return jsonify({"error": "no image"}), 400

    try:
        img_bytes = base64.b64decode(img_b64.split(",")[-1])
        img = Image.open(io.BytesIO(img_bytes))
        if params.get("filter") == "product":
            hmap, mask = make_product_height_and_mask(img, params)
            mesh = product_to_preview_json(hmap, mask, params)
        else:
            hmap = image_to_heightmap(img, params.get("filter","standard"), params.get("detail","medium"))
            mesh = heightmap_to_preview_json(hmap, params)
        return jsonify(mesh)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/export", methods=["POST"])
def api_export():
    data    = request.get_json()
    img_b64 = data.get("image") or state.get("image_data")
    params  = data.get("params", state["params"])

    if not img_b64:
        return jsonify({"error": "no image"}), 400

    stl_bytes = generate_stl_bytes(img_b64, params)
    if stl_bytes is None:
        return jsonify({"error": "STL generation failed"}), 500

    return Response(
        stl_bytes,
        mimetype="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="model.stl"'},
    )


# ─────────────────────────────────────────────
#  Embedded HTML / JS / CSS frontend
# ─────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Image → STL Converter</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #0d0f14;
    --panel:    #13161e;
    --border:   #252836;
    --accent:   #5b8af0;
    --accent2:  #a06ef7;
    --text:     #e2e4f0;
    --muted:    #6b6f82;
    --success:  #4ecb8c;
    --mono:     'Space Mono', monospace;
    --sans:     'Syne', sans-serif;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    display: flex;
    height: 100vh;
    overflow: hidden;
  }

  /* ── Sidebar ── */
  #sidebar {
    width: 320px;
    min-width: 320px;
    background: var(--panel);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    overflow-x: hidden;
  }

  #sidebar::-webkit-scrollbar { width: 4px; }
  #sidebar::-webkit-scrollbar-track { background: transparent; }
  #sidebar::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  .logo {
    padding: 20px 20px 12px;
    font-size: 18px;
    font-weight: 800;
    letter-spacing: -0.5px;
    border-bottom: 1px solid var(--border);
  }
  .logo span { color: var(--accent); }

  .section {
    padding: 16px 20px;
    border-bottom: 1px solid var(--border);
  }

  .section-title {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
  }

  /* Drop zone */
  #dropzone {
    border: 2px dashed var(--border);
    border-radius: 10px;
    padding: 28px 16px;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    position: relative;
  }
  #dropzone:hover, #dropzone.over {
    border-color: var(--accent);
    background: rgba(91,138,240,0.06);
  }
  #dropzone .dz-icon { font-size: 32px; margin-bottom: 8px; }
  #dropzone .dz-text { font-size: 12px; color: var(--muted); line-height: 1.6; }
  #dropzone input { position:absolute; inset:0; opacity:0; cursor:pointer; }

  #thumb-wrap {
    display: none;
    position: relative;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border);
  }
  #thumb-wrap img {
    width: 100%;
    display: block;
    max-height: 160px;
    object-fit: cover;
  }
  #thumb-clear {
    position: absolute; top: 6px; right: 6px;
    background: rgba(0,0,0,0.7); border: none; color: #fff;
    width: 24px; height: 24px; border-radius: 50%; cursor: pointer;
    font-size: 14px; display: flex; align-items: center; justify-content: center;
  }

  /* Pill selectors */
  .pill-group {
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
    margin-bottom: 14px;
  }
  .pill-group label {
    font-size: 11px;
    color: var(--muted);
    display: block;
    margin-bottom: 6px;
    font-weight: 700;
    letter-spacing: 0.5px;
  }
  .pill-row { display: flex; gap: 4px; flex-wrap: wrap; }
  .pill {
    padding: 5px 12px;
    border-radius: 20px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--muted);
    font-size: 11px;
    font-family: var(--mono);
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
  }
  .pill:hover { border-color: var(--accent); color: var(--text); }
  .pill.active {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
  }

  /* Sliders */
  .slider-row {
    margin-bottom: 14px;
  }
  .control-group {
    border-top: 1px solid var(--border);
    padding-top: 12px;
    margin-top: 12px;
  }
  .control-group:first-of-type {
    border-top: 0;
    padding-top: 0;
    margin-top: 0;
  }
  .control-group-title {
    color: var(--accent);
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 10px;
    text-transform: uppercase;
  }
  .slider-row .sr-top {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 6px;
  }
  .slider-row .sr-label {
    font-size: 11px; color: var(--muted); font-weight: 700; letter-spacing: 0.5px;
  }
  .slider-row .sr-val {
    font-size: 11px; font-family: var(--mono); color: var(--accent);
  }
  input[type=range] {
    -webkit-appearance: none;
    width: 100%; height: 4px;
    background: var(--border);
    border-radius: 2px; outline: none; cursor: pointer;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px; height: 14px;
    border-radius: 50%; background: var(--accent);
    box-shadow: 0 0 6px rgba(91,138,240,0.5);
  }

  /* Number inputs */
  .num-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 8px;
  }
  .num-field label {
    font-size: 10px; color: var(--muted); display: block;
    margin-bottom: 4px; font-weight: 700; letter-spacing: 0.5px;
  }
  .num-field input {
    width: 100%; background: var(--bg); border: 1px solid var(--border);
    color: var(--text); font-family: var(--mono); font-size: 12px;
    padding: 6px 8px; border-radius: 6px; outline: none;
    transition: border-color 0.2s;
  }
  .num-field input:focus { border-color: var(--accent); }

  #mask-editor {
    display: none;
    margin-top: 12px;
  }
  #eraser-canvas-wrap {
    width: 100%;
    aspect-ratio: 1 / 1;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    background: #090b10;
    position: relative;
  }
  #eraser-canvas {
    width: 100%;
    height: 100%;
    display: block;
    cursor: crosshair;
    touch-action: none;
  }
  .mini-actions {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 8px;
  }
  .mini-btn {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    cursor: pointer;
    font-family: var(--mono);
    font-size: 11px;
    padding: 7px 8px;
  }
  .mini-btn:hover { border-color: var(--accent); }

  /* Buttons */
  #btn-export {
    margin: 20px;
    width: calc(100% - 40px);
    padding: 12px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border: none; border-radius: 8px;
    color: #fff; font-family: var(--sans); font-size: 14px; font-weight: 700;
    cursor: pointer; transition: opacity 0.2s, transform 0.1s;
    letter-spacing: 0.5px;
  }
  #btn-export:hover { opacity: 0.9; }
  #btn-export:active { transform: scale(0.98); }
  #btn-export:disabled { opacity: 0.4; cursor: not-allowed; }

  /* Status bar */
  #status {
    margin: 0 20px 4px;
    font-size: 11px;
    font-family: var(--mono);
    color: var(--muted);
    min-height: 16px;
    transition: color 0.3s;
  }
  #status.ok  { color: var(--success); }
  #status.err { color: #f07070; }

  /* ── Viewport ── */
  #viewport {
    flex: 1;
    position: relative;
    overflow: hidden;
  }
  #canvas { width: 100%; height: 100%; display: block; }

  #overlay-hint {
    position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
    background: rgba(13,15,20,0.85); border: 1px solid var(--border);
    padding: 8px 16px; border-radius: 20px;
    font-size: 11px; color: var(--muted); font-family: var(--mono);
    pointer-events: none; white-space: nowrap;
  }

  #loading-overlay {
    position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
    background: rgba(13,15,20,0.7);
    font-size: 13px; color: var(--muted); font-family: var(--mono);
    opacity: 0; pointer-events: none;
    transition: opacity 0.2s;
  }
  #loading-overlay.visible { opacity: 1; pointer-events: all; }

  .spinner {
    width: 18px; height: 18px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    margin-right: 10px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  #no-image-msg {
    position: absolute; inset: 0;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 12px;
    color: var(--muted); font-family: var(--mono); font-size: 12px;
    pointer-events: none;
  }
  #no-image-msg .big { font-size: 48px; }
</style>
</head>
<body>

<!-- ──────────── SIDEBAR ──────────── -->
<div id="sidebar">
  <div class="logo">image<span>→</span>stl</div>

  <!-- Image Upload -->
  <div class="section">
    <div class="section-title">Source Image</div>
    <div id="dropzone">
      <div class="dz-icon">🖼</div>
      <div class="dz-text">Drop an image here<br>or click to browse</div>
      <input type="file" id="file-input" accept="image/*">
    </div>
    <div id="thumb-wrap">
      <img id="thumb" src="" alt="preview">
      <button id="thumb-clear">✕</button>
    </div>
    <div id="mask-editor">
      <div class="slider-row">
        <div class="sr-top">
          <span class="sr-label">MASK ERASER</span>
          <span class="sr-val" id="eraser-count">0 strokes</span>
        </div>
        <input type="range" id="sl-eraser" min="0.01" max="0.18" step="0.005" value="0.045">
      </div>
      <div id="eraser-canvas-wrap">
        <canvas id="eraser-canvas" width="512" height="512"></canvas>
      </div>
      <div class="mini-actions">
        <button class="mini-btn" id="btn-erase-undo" type="button">Undo</button>
        <button class="mini-btn" id="btn-erase-clear" type="button">Clear</button>
      </div>
    </div>
  </div>

  <!-- Filter -->
  <div class="section">
    <div class="section-title">Filter Mode</div>
    <div class="pill-row" id="filter-pills">
      <button class="pill active" data-v="standard">Standard</button>
      <button class="pill" data-v="standard_color">Standard (Color)</button>
      <button class="pill" data-v="extrude">Extrude</button>
      <button class="pill" data-v="extrude_color">Extrude (Color)</button>
      <button class="pill" data-v="product">Product</button>
    </div>
  </div>

  <!-- Detail -->
  <div class="section">
    <div class="section-title">Detail Level</div>
    <div class="pill-row" id="detail-pills">
      <button class="pill" data-v="low">Low</button>
      <button class="pill active" data-v="medium">Medium</button>
      <button class="pill" data-v="high">High</button>
    </div>
  </div>

  <!-- Base & Tolerance -->
  <div class="section">
    <div class="section-title">Base &amp; Tolerance</div>
    <div class="slider-row">
      <div class="sr-top">
        <span class="sr-label">BASE THICKNESS</span>
        <span class="sr-val" id="base-val">1.0</span>
      </div>
      <input type="range" id="sl-base" min="0.5" max="10" step="0.5" value="1">
    </div>
    <div class="slider-row">
      <div class="sr-top">
        <span class="sr-label">TOLERANCE</span>
        <span class="sr-val" id="tol-val">0</span>
      </div>
      <input type="range" id="sl-tol" min="0" max="5" step="0.5" value="0">
    </div>
  </div>

  <!-- Units -->
  <div class="section">
    <div class="section-title">Units</div>
    <div class="pill-row" id="unit-pills">
      <button class="pill active" data-v="mm">Millimeters</button>
      <button class="pill" data-v="cm">Centimeters</button>
      <button class="pill" data-v="m">Meters</button>
      <button class="pill" data-v="in">Inches</button>
      <button class="pill" data-v="ft">Feet</button>
    </div>
  </div>

  <!-- Dimensions -->
  <div class="section">
    <div class="section-title">Dimensions</div>
    <div class="num-grid">
      <div class="num-field">
        <label>WIDTH</label>
        <input type="number" id="inp-w" value="50" min="1" step="1">
      </div>
      <div class="num-field">
        <label>HEIGHT</label>
        <input type="number" id="inp-h" value="50" min="1" step="1">
      </div>
      <div class="num-field">
        <label>DEPTH</label>
        <input type="number" id="inp-d" value="5" min="0.1" step="0.5">
      </div>
    </div>
  </div>

  <!-- Product tuning -->
  <div class="section" id="product-tuning">
    <div class="section-title">Product Tuning</div>
    <div class="control-group">
      <div class="control-group-title">Subject Shape</div>
      <div class="slider-row">
        <div class="sr-top"><span class="sr-label">IMAGE BG REMOVE</span><span class="sr-val" id="image-bg-remove-val">off</span></div>
        <input type="checkbox" class="product-check" data-param="image_bg_remove" data-label="image-bg-remove-val">
      </div>
      <div class="slider-row">
        <div class="sr-top"><span class="sr-label">IMAGE BG STRENGTH</span><span class="sr-val" id="image-bg-strength-val">0.55</span></div>
        <input type="range" class="product-slider" data-param="image_bg_strength" data-label="image-bg-strength-val" min="0.00" max="1.00" step="0.01" value="0.55">
      </div>
      <div class="slider-row">
        <div class="sr-top"><span class="sr-label">EDGE SMOOTH</span><span class="sr-val" id="edge-smooth-val">6</span></div>
        <input type="range" class="product-slider" data-param="edge_smooth" data-label="edge-smooth-val" min="0" max="12" step="0.25" value="6">
      </div>
      <div class="slider-row">
        <div class="sr-top"><span class="sr-label">MASK GROW</span><span class="sr-val" id="mask-grow-val">2</span></div>
        <input type="range" class="product-slider" data-param="mask_grow" data-label="mask-grow-val" min="0" max="7" step="1" value="2">
      </div>
      <div class="slider-row">
        <div class="sr-top"><span class="sr-label">MASK CLOSE</span><span class="sr-val" id="mask-close-val">3</span></div>
        <input type="range" class="product-slider" data-param="mask_close" data-label="mask-close-val" min="0" max="7" step="1" value="3">
      </div>
      <div class="slider-row">
        <div class="sr-top"><span class="sr-label">MAX SUBJECT FILL</span><span class="sr-val" id="max-subject-fill-val">0.34</span></div>
        <input type="range" class="product-slider" data-param="max_subject_fill" data-label="max-subject-fill-val" min="0.10" max="0.80" step="0.01" value="0.34">
      </div>
    </div>

    <div class="control-group">
      <div class="control-group-title">Place & Crop</div>
    <div class="slider-row">
      <div class="sr-top"><span class="sr-label">CROP TOP</span><span class="sr-val" id="crop-top-val">0</span></div>
      <input type="range" class="product-slider" data-param="crop_top" data-label="crop-top-val" min="0" max="40" step="1" value="0">
    </div>
    <div class="slider-row">
      <div class="sr-top"><span class="sr-label">CROP BOTTOM</span><span class="sr-val" id="crop-bottom-val">0</span></div>
      <input type="range" class="product-slider" data-param="crop_bottom" data-label="crop-bottom-val" min="0" max="40" step="1" value="0">
    </div>
    <div class="slider-row">
      <div class="sr-top"><span class="sr-label">CROP SIDES</span><span class="sr-val" id="crop-sides-val">0</span></div>
      <input type="range" class="product-slider" data-param="crop_sides" data-label="crop-sides-val" min="0" max="30" step="1" value="0">
    </div>
    <div class="slider-row">
      <div class="sr-top"><span class="sr-label">SUBJECT WIDTH</span><span class="sr-val" id="subject-scale-x-val">1.00</span></div>
      <input type="range" class="product-slider" data-param="subject_scale_x" data-label="subject-scale-x-val" min="0.65" max="1.25" step="0.01" value="1.00">
    </div>
    <div class="slider-row">
      <div class="sr-top"><span class="sr-label">SUBJECT HEIGHT</span><span class="sr-val" id="subject-scale-y-val">1.00</span></div>
      <input type="range" class="product-slider" data-param="subject_scale_y" data-label="subject-scale-y-val" min="0.65" max="1.35" step="0.01" value="1.00">
    </div>
    <div class="slider-row">
      <div class="sr-top"><span class="sr-label">SUBJECT VERTICAL</span><span class="sr-val" id="subject-offset-y-val">-0.03</span></div>
      <input type="range" class="product-slider" data-param="subject_offset_y" data-label="subject-offset-y-val" min="-0.25" max="0.25" step="0.01" value="-0.03">
    </div>
    </div>

    <div class="control-group">
      <div class="control-group-title">Pickup Thresholds</div>
    <div class="slider-row">
      <div class="sr-top"><span class="sr-label">COLOR PICKUP</span><span class="sr-val" id="sat-threshold-val">0.28</span></div>
      <input type="range" class="product-slider" data-param="sat_threshold" data-label="sat-threshold-val" min="0.02" max="0.70" step="0.01" value="0.28">
    </div>
    <div class="slider-row">
      <div class="sr-top"><span class="sr-label">DARK PICKUP</span><span class="sr-val" id="dark-threshold-val">0.45</span></div>
      <input type="range" class="product-slider" data-param="dark_threshold" data-label="dark-threshold-val" min="0.10" max="0.90" step="0.01" value="0.45">
    </div>
    <div class="slider-row">
      <div class="sr-top"><span class="sr-label">BACKGROUND PICKUP</span><span class="sr-val" id="bg-threshold-val">0.30</span></div>
      <input type="range" class="product-slider" data-param="bg_threshold" data-label="bg-threshold-val" min="0.03" max="0.60" step="0.01" value="0.30">
    </div>
    <div class="slider-row">
      <div class="sr-top"><span class="sr-label">BG REMOVE</span><span class="sr-val" id="bg-remove-val">0.62</span></div>
      <input type="range" class="product-slider" data-param="bg_remove" data-label="bg-remove-val" min="0.00" max="1.00" step="0.01" value="0.62">
    </div>
    <div class="slider-row">
      <div class="sr-top"><span class="sr-label">BG TOLERANCE</span><span class="sr-val" id="bg-tolerance-val">0.26</span></div>
      <input type="range" class="product-slider" data-param="bg_tolerance" data-label="bg-tolerance-val" min="0.02" max="0.70" step="0.01" value="0.26">
    </div>
    <div class="slider-row">
      <div class="sr-top"><span class="sr-label">EDGE PROTECT</span><span class="sr-val" id="bg-edge-barrier-val">0.95</span></div>
      <input type="range" class="product-slider" data-param="bg_edge_barrier" data-label="bg-edge-barrier-val" min="0.05" max="1.00" step="0.01" value="0.95">
    </div>
    <div class="slider-row">
      <div class="sr-top"><span class="sr-label">BORDER TRIM</span><span class="sr-val" id="bg-border-trim-val">0.04</span></div>
      <input type="range" class="product-slider" data-param="bg_border_trim" data-label="bg-border-trim-val" min="0.00" max="0.20" step="0.005" value="0.04">
    </div>
    </div>

    <div class="control-group">
      <div class="control-group-title">Relief & Detail</div>
    <div class="slider-row">
      <div class="sr-top"><span class="sr-label">EDGE PICKUP</span><span class="sr-val" id="edge-threshold-val">0.28</span></div>
      <input type="range" class="product-slider" data-param="edge_threshold" data-label="edge-threshold-val" min="0.05" max="0.80" step="0.01" value="0.28">
    </div>
    <div class="slider-row">
      <div class="sr-top"><span class="sr-label">DETAIL AMOUNT</span><span class="sr-val" id="detail-amount-val">0.50</span></div>
      <input type="range" class="product-slider" data-param="detail_amount" data-label="detail-amount-val" min="0.10" max="1.00" step="0.01" value="0.50">
    </div>
    <div class="slider-row">
      <div class="sr-top"><span class="sr-label">EDGE WIDTH</span><span class="sr-val" id="edge-width-val">1</span></div>
      <input type="range" class="product-slider" data-param="edge_width" data-label="edge-width-val" min="0" max="4" step="1" value="1">
    </div>
    <div class="slider-row">
      <div class="sr-top"><span class="sr-label">BODY HEIGHT</span><span class="sr-val" id="body-height-val">0.72</span></div>
      <input type="range" class="product-slider" data-param="body_height" data-label="body-height-val" min="0.35" max="0.95" step="0.01" value="0.72">
    </div>
    <div class="slider-row">
      <div class="sr-top"><span class="sr-label">PHOTO RELIEF</span><span class="sr-val" id="relief-strength-val">0.30</span></div>
      <input type="range" class="product-slider" data-param="relief_strength" data-label="relief-strength-val" min="0.00" max="0.70" step="0.01" value="0.30">
    </div>
    <div class="slider-row">
      <div class="sr-top"><span class="sr-label">EDGE RELIEF</span><span class="sr-val" id="edge-relief-val">0.32</span></div>
      <input type="range" class="product-slider" data-param="edge_relief" data-label="edge-relief-val" min="0.00" max="0.80" step="0.01" value="0.32">
    </div>
    </div>

    <div class="control-group">
      <div class="control-group-title">Frame</div>
    <div class="slider-row">
      <div class="sr-top"><span class="sr-label">CIRCLE FRAME</span><span class="sr-val" id="include-frame-val">on</span></div>
      <input type="checkbox" class="product-check" data-param="include_frame" data-label="include-frame-val" checked>
    </div>
    <div class="slider-row">
      <div class="sr-top"><span class="sr-label">FRAME WIDTH</span><span class="sr-val" id="frame-width-val">0.12</span></div>
      <input type="range" class="product-slider" data-param="frame_width" data-label="frame-width-val" min="0.06" max="0.25" step="0.005" value="0.12">
    </div>
    <div class="slider-row">
      <div class="sr-top"><span class="sr-label">BRACE WIDTH</span><span class="sr-val" id="brace-width-val">0.085</span></div>
      <input type="range" class="product-slider" data-param="brace_width" data-label="brace-width-val" min="0.03" max="0.18" step="0.005" value="0.085">
    </div>
    <div class="slider-row">
      <div class="sr-top"><span class="sr-label">HOLE SIZE</span><span class="sr-val" id="hole-radius-val">0.04</span></div>
      <input type="range" class="product-slider" data-param="hole_radius" data-label="hole-radius-val" min="0.00" max="0.09" step="0.005" value="0.04">
    </div>
    </div>
  </div>

  <div id="status"></div>
  <button id="btn-export" disabled>Export STL</button>
</div>

<!-- ──────────── VIEWPORT ──────────── -->
<div id="viewport">
  <canvas id="canvas"></canvas>
  <div id="no-image-msg">
    <div class="big">🧊</div>
    <div>Upload an image to preview your 3D model</div>
  </div>
  <div id="loading-overlay">
    <div class="spinner"></div> Generating preview…
  </div>
  <div id="overlay-hint">🖱 Drag to rotate · Scroll to zoom · Right-drag to pan</div>
</div>

<!-- Three.js r148 via CDN -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>

<script>
// ═══════════════════════════════════════════════
//  State
// ═══════════════════════════════════════════════
const state = {
  image: null,
  params: {
    filter: 'standard', detail: 'medium',
    base: 1, tolerance: 0, unit: 'mm',
    width: 50, height: 50, depth: 5,
    crop_left: 0, crop_right: 0, crop_top: 0, crop_bottom: 0,
    subject_scale_x: 1.0, subject_scale_y: 1.0, subject_offset_y: -0.03,
    sat_threshold: 0.28, dark_threshold: 0.45, bg_threshold: 0.30,
    bg_remove: 0.62, image_bg_remove: false, image_bg_strength: 0.55, bg_tolerance: 0.26, bg_edge_barrier: 0.95, bg_border_trim: 0.04,
    edge_threshold: 0.28, detail_amount: 0.50,
    edge_width: 1, edge_smooth: 6.0, mask_grow: 2, mask_close: 3, min_subject_fill: 0.05, max_subject_fill: 0.34,
    body_height: 0.72, frame_height: 0.46, include_frame: true,
    frame_width: 0.12, brace_width: 0.085, hole_radius: 0.04,
    relief_strength: 0.30, edge_relief: 0.32,
    erase_strokes: [],
  }
};

let eraserImage = null;
let eraserDrawing = false;

let debounceTimer = null;
function schedulePreview(delay = 400) {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(fetchPreview, delay);
}

// ═══════════════════════════════════════════════
//  Three.js Setup
// ═══════════════════════════════════════════════
const canvas   = document.getElementById('canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x0d0f14);

const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10000);
camera.position.set(80, 80, 80);

// Lights
scene.add(new THREE.AmbientLight(0xffffff, 0.4));
const dirL = new THREE.DirectionalLight(0xffffff, 0.9);
dirL.position.set(100, 200, 100);
scene.add(dirL);
const dirL2 = new THREE.DirectionalLight(0x8ab4f8, 0.4);
dirL2.position.set(-100, -50, -100);
scene.add(dirL2);

// Grid helper
const grid = new THREE.GridHelper(200, 20, 0x252836, 0x1a1d28);
scene.add(grid);

let meshObj = null;

function resize() {
  const vp = document.getElementById('viewport');
  const w = vp.clientWidth, h = vp.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
resize();
window.addEventListener('resize', resize);

// ── Orbit controls (manual) ──────────────────
let isDragging  = false;
let isRightDrag = false;
let lastX = 0, lastY = 0;
let theta = 0.8, phi = 0.9, radius = 160;
const target = new THREE.Vector3(25, 5, 25);

function updateCamera() {
  camera.position.set(
    target.x + radius * Math.sin(phi) * Math.sin(theta),
    target.y + radius * Math.cos(phi),
    target.z + radius * Math.sin(phi) * Math.cos(theta)
  );
  camera.lookAt(target);
}
updateCamera();

canvas.addEventListener('mousedown', e => {
  isDragging  = true;
  isRightDrag = e.button === 2;
  lastX = e.clientX; lastY = e.clientY;
});
canvas.addEventListener('contextmenu', e => e.preventDefault());
window.addEventListener('mouseup', () => { isDragging = false; });
window.addEventListener('mousemove', e => {
  if (!isDragging) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  if (isRightDrag) {
    const right = new THREE.Vector3();
    right.crossVectors(camera.getWorldDirection(new THREE.Vector3()), camera.up).normalize();
    target.addScaledVector(right, -dx * 0.15);
    target.y += dy * 0.15;
  } else {
    theta -= dx * 0.008;
    phi = Math.max(0.05, Math.min(Math.PI - 0.05, phi + dy * 0.008));
  }
  updateCamera();
});
canvas.addEventListener('wheel', e => {
  radius = Math.max(10, Math.min(2000, radius + e.deltaY * 0.4));
  updateCamera();
});

// Render loop
(function loop() {
  requestAnimationFrame(loop);
  renderer.render(scene, camera);
})();

// ═══════════════════════════════════════════════
//  Mesh update
// ═══════════════════════════════════════════════
function applyMeshData(data) {
  if (meshObj) { scene.remove(meshObj); meshObj.geometry.dispose(); meshObj.material.dispose(); }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(data.positions, 3));
  geo.setAttribute('color',    new THREE.Float32BufferAttribute(data.colors,    3));
  geo.computeVertexNormals();

  const mat = new THREE.MeshLambertMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
  });

  meshObj = new THREE.Mesh(geo, mat);
  scene.add(meshObj);

  // Centre model
  geo.computeBoundingBox();
  const box = geo.boundingBox;
  const cx = (box.min.x + box.max.x) / 2;
  const cz = (box.min.z + box.max.z) / 2;
  target.set(cx, 0, cz);
  const maxDim = Math.max(data.width, data.height, data.depth);
  radius = maxDim * 2.5;
  phi = 0.85; theta = 0.6;
  updateCamera();

  // Update grid
  grid.scale.setScalar(maxDim / 100);
  grid.position.set(cx, -0.5, cz);

  document.getElementById('no-image-msg').style.display = 'none';
}

// ═══════════════════════════════════════════════
//  API calls
// ═══════════════════════════════════════════════
async function fetchPreview() {
  if (!state.image) return;
  const overlay = document.getElementById('loading-overlay');
  overlay.classList.add('visible');
  setStatus('Generating preview…', '');
  try {
    const res = await fetch('/api/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: state.image, params: state.params }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    applyMeshData(data);
    setStatus('Preview ready', 'ok');
  } catch (e) {
    setStatus('Preview error: ' + e.message, 'err');
  } finally {
    overlay.classList.remove('visible');
  }
}

async function exportSTL() {
  if (!state.image) return;
  setStatus('Generating STL…', '');
  document.getElementById('btn-export').disabled = true;
  try {
    const res = await fetch('/api/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: state.image, params: state.params }),
    });
    if (!res.ok) throw new Error(await res.text());
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = 'model.stl'; a.click();
    URL.revokeObjectURL(url);
    setStatus('STL exported ✓', 'ok');
  } catch (e) {
    setStatus('Export error: ' + e.message, 'err');
  } finally {
    document.getElementById('btn-export').disabled = false;
  }
}

function setStatus(msg, cls) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = cls;
}

function eraserImageRect() {
  const canvas = document.getElementById('eraser-canvas');
  if (!eraserImage) return { x: 0, y: 0, w: canvas.width, h: canvas.height };
  const scale = Math.min(canvas.width / eraserImage.width, canvas.height / eraserImage.height);
  const w = eraserImage.width * scale;
  const h = eraserImage.height * scale;
  return { x: (canvas.width - w) / 2, y: (canvas.height - h) / 2, w, h };
}

function updateEraserCount() {
  const count = state.params.erase_strokes.length;
  document.getElementById('eraser-count').textContent = count + (count === 1 ? ' stroke' : ' strokes');
}

function drawEraserCanvas() {
  const canvas = document.getElementById('eraser-canvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#090b10';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  if (eraserImage) {
    const imgRect = eraserImageRect();
    ctx.drawImage(eraserImage, imgRect.x, imgRect.y, imgRect.w, imgRect.h);
  }

  ctx.fillStyle = 'rgba(240,80,80,0.42)';
  ctx.strokeStyle = 'rgba(255,235,235,0.85)';
  ctx.lineWidth = 1;
  const imgRect = eraserImageRect();
  state.params.erase_strokes.forEach(stroke => {
    const imageSpace = stroke.space === 'image';
    const x = imageSpace ? imgRect.x + stroke.x * imgRect.w : stroke.x * canvas.width;
    const y = imageSpace ? imgRect.y + stroke.y * imgRect.h : stroke.y * canvas.height;
    const r = imageSpace ? stroke.r * imgRect.w : stroke.r * canvas.width;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  });
  updateEraserCount();
}

function addEraserStroke(e) {
  const canvas = document.getElementById('eraser-canvas');
  const rect = canvas.getBoundingClientRect();
  const imgRect = eraserImageRect();
  const px = (e.clientX - rect.left) / rect.width * canvas.width;
  const py = (e.clientY - rect.top) / rect.height * canvas.height;
  if (px < imgRect.x || py < imgRect.y || px > imgRect.x + imgRect.w || py > imgRect.y + imgRect.h) return;
  const x = Math.max(0, Math.min(1, (px - imgRect.x) / imgRect.w));
  const y = Math.max(0, Math.min(1, (py - imgRect.y) / imgRect.h));
  const brush = parseFloat(document.getElementById('sl-eraser').value);
  const r = brush * canvas.width / Math.max(imgRect.w, 1);
  const strokes = state.params.erase_strokes;
  const last = strokes[strokes.length - 1];
  if (last && Math.hypot(last.x - x, last.y - y) < r * 0.35) return;
  strokes.push({ x, y, r, space: 'image' });
  drawEraserCanvas();
  schedulePreview(250);
}

function resetEraserForImage() {
  state.params.erase_strokes = [];
  eraserImage = new Image();
  eraserImage.onload = drawEraserCanvas;
  eraserImage.src = state.image;
  document.getElementById('mask-editor').style.display = 'block';
  updateEraserCount();
}

// ═══════════════════════════════════════════════
//  UI Wiring
// ═══════════════════════════════════════════════

// File input / drop
function loadFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const reader = new FileReader();
  reader.onload = e => {
    state.image = e.target.result;
    document.getElementById('thumb').src = state.image;
    document.getElementById('dropzone').style.display = 'none';
    document.getElementById('thumb-wrap').style.display = 'block';
    document.getElementById('btn-export').disabled = false;
    resetEraserForImage();
    schedulePreview(100);
  };
  reader.readAsDataURL(file);
  document.getElementById('file-input').value = '';
}

document.getElementById('file-input').addEventListener('change', e => loadFile(e.target.files[0]));
const dz = document.getElementById('dropzone');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('over'); });
dz.addEventListener('dragleave', () => dz.classList.remove('over'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('over');
  loadFile(e.dataTransfer.files[0]);
});

document.getElementById('thumb-clear').addEventListener('click', () => {
  state.image = null;
  document.getElementById('dropzone').style.display = '';
  document.getElementById('thumb-wrap').style.display = 'none';
  document.getElementById('mask-editor').style.display = 'none';
  state.params.erase_strokes = [];
  eraserImage = null;
  document.getElementById('btn-export').disabled = true;
  document.getElementById('no-image-msg').style.display = 'flex';
  if (meshObj) { scene.remove(meshObj); meshObj.geometry.dispose(); meshObj = null; }
});

const eraserCanvas = document.getElementById('eraser-canvas');
eraserCanvas.addEventListener('pointerdown', e => {
  if (!state.image) return;
  eraserDrawing = true;
  eraserCanvas.setPointerCapture(e.pointerId);
  addEraserStroke(e);
});
eraserCanvas.addEventListener('pointermove', e => {
  if (eraserDrawing) addEraserStroke(e);
});
eraserCanvas.addEventListener('pointerup', e => {
  eraserDrawing = false;
  eraserCanvas.releasePointerCapture(e.pointerId);
});
eraserCanvas.addEventListener('pointercancel', () => { eraserDrawing = false; });
document.getElementById('sl-eraser').addEventListener('input', drawEraserCanvas);
document.getElementById('btn-erase-undo').addEventListener('click', () => {
  state.params.erase_strokes.pop();
  drawEraserCanvas();
  schedulePreview(100);
});
document.getElementById('btn-erase-clear').addEventListener('click', () => {
  state.params.erase_strokes = [];
  drawEraserCanvas();
  schedulePreview(100);
});

// Pill groups
function wirePills(groupId, paramKey) {
  const group = document.getElementById(groupId);
  group.querySelectorAll('.pill').forEach(btn => {
    btn.addEventListener('click', () => {
      group.querySelectorAll('.pill').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.params[paramKey] = btn.dataset.v;
      if (paramKey === 'filter' && btn.dataset.v === 'product') {
        state.params.width = 120;
        state.params.height = 120;
        state.params.depth = 3;
        state.params.base = 0.5;
        document.getElementById('inp-w').value = 120;
        document.getElementById('inp-h').value = 120;
        document.getElementById('inp-d').value = 3;
        document.getElementById('sl-base').value = 0.5;
        document.getElementById('base-val').textContent = '0.5';
      }
      schedulePreview();
    });
  });
}
wirePills('filter-pills', 'filter');
wirePills('detail-pills', 'detail');
wirePills('unit-pills',   'unit');

// Sliders
document.getElementById('sl-base').addEventListener('input', e => {
  state.params.base = parseFloat(e.target.value);
  document.getElementById('base-val').textContent = e.target.value;
  schedulePreview();
});
document.getElementById('sl-tol').addEventListener('input', e => {
  state.params.tolerance = parseFloat(e.target.value);
  document.getElementById('tol-val').textContent = e.target.value;
  schedulePreview();
});

document.querySelectorAll('.product-slider').forEach(slider => {
  slider.addEventListener('input', e => {
    const key = e.target.dataset.param;
    const value = parseFloat(e.target.value);
    if (key === 'crop_sides') {
      state.params.crop_left = value;
      state.params.crop_right = value;
    } else {
      state.params[key] = value;
    }
    const label = document.getElementById(e.target.dataset.label);
    if (label) label.textContent = e.target.value;
    schedulePreview();
  });
});

document.querySelectorAll('.product-check').forEach(check => {
  check.addEventListener('change', e => {
    const key = e.target.dataset.param;
    state.params[key] = e.target.checked;
    const label = document.getElementById(e.target.dataset.label);
    if (label) label.textContent = e.target.checked ? 'on' : 'off';
    schedulePreview();
  });
});

// Number inputs
['w','h','d'].forEach((k, i) => {
  const paramKey = ['width','height','depth'][i];
  document.getElementById('inp-'+k).addEventListener('input', e => {
    state.params[paramKey] = parseFloat(e.target.value) || 0;
    schedulePreview();
  });
});

// Export
document.getElementById('btn-export').addEventListener('click', exportSTL);
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Image → STL Converter with live preview")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on (default: 5000)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--input", help="Input image path for direct STL export")
    parser.add_argument("--output", help="Output STL path for direct STL export")
    parser.add_argument("--filter", default="standard",
                        choices=["standard", "standard_color", "extrude", "extrude_color", "product"],
                        help="Conversion mode for direct export")
    parser.add_argument("--detail", default="medium", choices=["low", "medium", "high"],
                        help="Mesh detail for direct export")
    parser.add_argument("--unit", default="mm", choices=list(UNIT_TO_MM.keys()),
                        help="Unit for direct export dimensions")
    parser.add_argument("--width", type=float, help="Output width for direct export")
    parser.add_argument("--height", type=float, help="Output height for direct export")
    parser.add_argument("--depth", type=float, help="Output depth/thickness for direct export")
    parser.add_argument("--base", type=float, default=1, help="Base thickness for heightmap modes")
    parser.add_argument("--crop-left", type=float, default=0, help="Product crop percent from left")
    parser.add_argument("--crop-right", type=float, default=0, help="Product crop percent from right")
    parser.add_argument("--crop-top", type=float, default=0, help="Product crop percent from top")
    parser.add_argument("--crop-bottom", type=float, default=0, help="Product crop percent from bottom")
    parser.add_argument("--subject-scale-x", "--car-scale-x", dest="subject_scale_x",
                        type=float, default=1.0, help="Product subject width multiplier")
    parser.add_argument("--subject-scale-y", "--car-scale-y", dest="subject_scale_y",
                        type=float, default=1.0, help="Product subject height multiplier")
    parser.add_argument("--subject-offset-y", "--car-offset-y", dest="subject_offset_y",
                        type=float, default=-0.03, help="Product subject vertical offset")
    parser.add_argument("--sat-threshold", type=float, default=0.28, help="Product color pickup threshold")
    parser.add_argument("--dark-threshold", type=float, default=0.45, help="Product dark pickup threshold")
    parser.add_argument("--bg-threshold", type=float, default=0.30, help="Product background difference threshold")
    parser.add_argument("--bg-remove", type=float, default=0.62, help="Background removal aggression")
    parser.add_argument("--image-bg-remove", dest="image_bg_remove", action="store_true", default=False,
                        help="Cut out the image background before product mask generation")
    parser.add_argument("--no-image-bg-remove", dest="image_bg_remove", action="store_false",
                        help="Disable the image background cutout")
    parser.add_argument("--image-bg-strength", type=float, default=0.55, help="Image background cutout strength")
    parser.add_argument("--bg-tolerance", type=float, default=0.26, help="How similar pixels can be to the border background and still be removed")
    parser.add_argument("--bg-edge-barrier", type=float, default=0.95, help="How strongly edges protect the subject during background flood removal")
    parser.add_argument("--bg-border-trim", type=float, default=0.04, help="Trim this fraction from the fitted image border before subject extraction")
    parser.add_argument("--edge-threshold", type=float, default=0.28, help="Product edge pickup threshold")
    parser.add_argument("--detail-amount", type=float, default=0.50, help="Product relief detail amount")
    parser.add_argument("--edge-width", type=float, default=1, help="Product raised edge width")
    parser.add_argument("--edge-smooth", type=float, default=6.0, help="Smooth jagged subject mask edges before export")
    parser.add_argument("--mask-grow", type=float, default=2, help="Product subject mask growth")
    parser.add_argument("--mask-close", type=float, default=3, help="Product subject mask smoothing")
    parser.add_argument("--min-subject-fill", type=float, default=0.05, help="Minimum subject fill before fallback")
    parser.add_argument("--max-subject-fill", type=float, default=0.34, help="Maximum subject fill before core fallback")
    parser.add_argument("--body-height", type=float, default=0.72, help="Product subject body height ratio")
    parser.add_argument("--frame-height", type=float, default=0.46, help="Product frame height ratio")
    parser.add_argument("--frame", action="store_true", help="Enable circular frame and diagonal braces")
    parser.add_argument("--no-frame", action="store_true", help="Disable circular frame and diagonal braces")
    parser.add_argument("--frame-width", type=float, default=0.12, help="Product frame width ratio")
    parser.add_argument("--brace-width", type=float, default=0.085, help="Product diagonal brace width ratio")
    parser.add_argument("--hole-radius", type=float, default=0.04, help="Product corner hole radius ratio")
    parser.add_argument("--relief-strength", type=float, default=0.30, help="Product broad photo relief strength")
    parser.add_argument("--edge-relief", type=float, default=0.32, help="Product edge relief strength")
    args = parser.parse_args()

    if args.input or args.output:
        if not args.input or not args.output:
            parser.error("--input and --output must be provided together")
        product_defaults = args.filter == "product"
        params = {
            "filter": args.filter,
            "detail": args.detail,
            "unit": args.unit,
            "width": args.width if args.width is not None else (120 if product_defaults else 50),
            "height": args.height if args.height is not None else (120 if product_defaults else 50),
            "depth": args.depth if args.depth is not None else (3 if product_defaults else 5),
            "base": args.base,
            "tolerance": 0,
            "crop_left": args.crop_left,
            "crop_right": args.crop_right,
            "crop_top": args.crop_top,
            "crop_bottom": args.crop_bottom,
            "subject_scale_x": args.subject_scale_x,
            "subject_scale_y": args.subject_scale_y,
            "subject_offset_y": args.subject_offset_y,
            "sat_threshold": args.sat_threshold,
            "dark_threshold": args.dark_threshold,
            "bg_threshold": args.bg_threshold,
            "bg_remove": args.bg_remove,
            "image_bg_remove": args.image_bg_remove,
            "image_bg_strength": args.image_bg_strength,
            "bg_tolerance": args.bg_tolerance,
            "bg_edge_barrier": args.bg_edge_barrier,
            "bg_border_trim": args.bg_border_trim,
            "edge_threshold": args.edge_threshold,
            "detail_amount": args.detail_amount,
            "edge_width": args.edge_width,
            "edge_smooth": args.edge_smooth,
            "mask_grow": args.mask_grow,
            "mask_close": args.mask_close,
            "min_subject_fill": args.min_subject_fill,
            "max_subject_fill": args.max_subject_fill,
            "body_height": args.body_height,
            "frame_height": args.frame_height,
            "include_frame": not args.no_frame,
            "frame_width": args.frame_width,
            "brace_width": args.brace_width,
            "hole_radius": args.hole_radius,
            "relief_strength": args.relief_strength,
            "edge_relief": args.edge_relief,
        }
        generate_stl_file(args.input, args.output, params)
        print(f"Created STL: {args.output}")
        return

    url = f"http://localhost:{args.port}"
    print(f"\n  🧊  Image → STL Converter")
    print(f"  ─────────────────────────")
    print(f"  Open: {url}\n")

    if not args.no_browser:
        def open_later():
            time.sleep(1.2)
            webbrowser.open(url)
        threading.Thread(target=open_later, daemon=True).start()

    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
