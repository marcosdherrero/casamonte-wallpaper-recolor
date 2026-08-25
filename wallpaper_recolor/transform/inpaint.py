# -*- coding: utf-8 -*-
"""
wallpaper_recolor.transform.inpaint
-----------------------------------
Fill masked holes on **crops around each detection** at source resolution.

Mask (EasyOCR quads, or AABB fallback):
fillPoly → MORPH_CLOSE (letter gaps) → dilate (antialias / shadow).
Geometric / stripe wallpapers (default, 122-LA4): wide horizontal close
15×3, 5×5 dilate, extra +x pad, downward shadow pad. Floral / damask:
tighter close and a single small dilate.

Fill order: LaMa ONNX (cached, never downloaded here) at native crop size
padded to a multiple of 8 (no downscale) → cv2.inpaint NS → numpy
period/Hilbert. Optional extras: ``pip install -r requirements-ocr.txt``.

Tile Build also uses this stack on wrap seams: roll the motif so opposite
edges meet in the middle, inpaint that plus-shaped band with LaMa (or
period/Hilbert fallback), then roll back. That uses both sides of the
repeat as context instead of copying one corner onto the other.

Empty mask is identity. Preview and Save share ``inpaint_image``.

Class references (code + name only):
- CAP3321C Data Wrangling
- CAP4631C Machine Learning
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from wallpaper_recolor.color.color_ranges import luma_channel
from wallpaper_recolor.labels.boxes import aabb_quad
from wallpaper_recolor.transform.lama_onnx import lama_inpaint_crop, lama_inpaint_windows
from wallpaper_recolor.transform.tessellate import estimate_axis_period, hilbert_xy_to_d

Box = tuple[int, int, int, int]
Quad = Sequence[tuple[float, float]]
STYLE_GEOMETRIC = "geometric"
STYLE_FLORAL = "floral"
STYLE_DEFAULT = STYLE_GEOMETRIC
_LAST_BACKEND = "numpy"
_NEIGH = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
_PAD_DEFAULT = 2
_CROP_MARGIN = 16
_CROP_MARGIN_MAX = 48
_SEAM_MIN = 8
_SEAM_MAX = 48
_SEAM_CONTEXT = 96
_LAMA_TILE = 512
_LAMA_OVERLAP = 64
_GLYPH_RING = 14
_INK_DELTA = 14.0


@dataclass(frozen=True)
class InpaintStyle:
    """Wallpaper-optimized Remove mask. Kernels are OpenCV MORPH_RECT (w, h)."""

    key: str
    close_kernel: tuple[int, int]
    dilate_kernel: tuple[int, int]
    dilate_iterations: int
    pad_x: int
    pad_y_down: int
    crop_pad: int


# Geometric / stripe linen (122-LA4): close letter gaps into a horizontal bar
# so LaMa can bridge stripes L→R; dilate for antialias; extra +y for drop-shadow.
STYLE_GEOMETRIC_SPEC = InpaintStyle(
    key=STYLE_GEOMETRIC,
    close_kernel=(15, 3),
    dilate_kernel=(5, 5),
    dilate_iterations=1,
    pad_x=8,
    pad_y_down=6,
    crop_pad=32,
)
# Floral / damask: keep the hole tight so weave/flowers are not eaten.
STYLE_FLORAL_SPEC = InpaintStyle(
    key=STYLE_FLORAL,
    close_kernel=(7, 3),
    dilate_kernel=(3, 3),
    dilate_iterations=1,
    pad_x=2,
    pad_y_down=2,
    crop_pad=16,
)


def inpaint_style(name: str | None) -> InpaintStyle:
    key = str(name or STYLE_DEFAULT).strip().lower()
    if key in ("floral", "damask", "tight"):
        return STYLE_FLORAL_SPEC
    return STYLE_GEOMETRIC_SPEC


def inpaint_backend() -> str:
    """``lama``, ``cv2``, ``numpy``, or ``identity`` from the last ``inpaint_image``."""
    return _LAST_BACKEND


def _set_backend(name: str) -> None:
    global _LAST_BACKEND
    _LAST_BACKEND = str(name)


def _hilbert_order(span: int) -> int:
    span = max(2, int(span))
    return max(1, int(np.ceil(np.log2(span))))


def _dilate8(mask: np.ndarray, times: int = 1) -> np.ndarray:
    """8-connected dilate (numpy, no scipy / OpenCV)."""
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(times))):
        acc = out.copy()
        acc[1:, :] |= out[:-1, :]
        acc[:-1, :] |= out[1:, :]
        acc[:, 1:] |= out[:, :-1]
        acc[:, :-1] |= out[:, 1:]
        acc[1:, 1:] |= out[:-1, :-1]
        acc[1:, :-1] |= out[:-1, 1:]
        acc[:-1, 1:] |= out[1:, :-1]
        acc[:-1, :-1] |= out[1:, 1:]
        out = acc
    return out


def mask_from_boxes(
    height: int,
    width: int,
    boxes: Sequence[Box],
    *,
    pad: int = _PAD_DEFAULT,
) -> np.ndarray:
    """True inside inclusive-exclusive boxes, optionally dilated by ``pad``."""
    h = max(0, int(height))
    w = max(0, int(width))
    mask = np.zeros((h, w), dtype=bool)
    if h == 0 or w == 0:
        return mask
    grow = max(0, int(pad))
    for raw in boxes:
        if raw is None or len(raw) != 4:
            continue
        x0, y0, x1, y1 = (int(round(float(v))) for v in raw)
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        x0 = max(0, x0 - grow)
        y0 = max(0, y0 - grow)
        x1 = min(w, x1 + grow)
        y1 = min(h, y1 + grow)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def _raise_if_cancelled(cancel) -> None:
    if cancel is not None and cancel.is_set():
        raise InterruptedError("cancelled")


def _try_cv2():
    try:
        import cv2
    except ImportError:
        return None
    return cv2


def _inflate_quad(quad: Quad, pad_x: int, pad_y_down: int) -> list[tuple[tuple[int, int], ...]]:
    """Horizontal expand (stripe bridging) plus a downward copy (drop-shadow)."""
    pts = [(int(round(float(x))), int(round(float(y)))) for x, y in quad]
    if len(pts) < 3:
        return []
    xs = [p[0] for p in pts]
    mid = (min(xs) + max(xs)) * 0.5
    px = max(0, int(pad_x))
    grown = tuple((p[0] - px if p[0] <= mid else p[0] + px, p[1]) for p in pts)
    out = [grown]
    dy = max(0, int(pad_y_down))
    if dy:
        out.append(tuple((p[0], p[1] + dy) for p in grown))
    return out


def _fill_poly_mask(height: int, width: int, polygons: Sequence[Sequence[tuple[int, int]]]) -> np.ndarray:
    cv2 = _try_cv2()
    mask = np.zeros((max(0, int(height)), max(0, int(width))), dtype=np.uint8)
    if mask.size == 0:
        return mask
    if cv2 is not None:
        for poly in polygons:
            pts = np.array(list(poly), dtype=np.int32)
            if pts.shape[0] >= 3:
                cv2.fillPoly(mask, [pts], 255)
        return mask
    img = Image.fromarray(mask, mode="L")
    draw = ImageDraw.Draw(img)
    for poly in polygons:
        pts = [(int(p[0]), int(p[1])) for p in poly]
        if len(pts) >= 3:
            draw.polygon(pts, fill=255)
    return np.asarray(img, dtype=np.uint8)


def _rank_rect(mask: np.ndarray, kx: int, ky: int, *, maximize: bool) -> np.ndarray:
    src = np.asarray(mask, dtype=np.uint8)
    h, w = src.shape
    kx = max(1, int(kx))
    ky = max(1, int(ky))
    out = np.zeros((h, w), dtype=np.uint8) if maximize else np.full((h, w), 255, dtype=np.uint8)
    py, px = ky // 2, kx // 2
    for dy in range(-py, ky - py):
        for dx in range(-px, kx - px):
            y0s, y1s = max(0, dy), h + min(0, dy)
            x0s, x1s = max(0, dx), w + min(0, dx)
            y0k, y1k = max(0, -dy), h - max(0, dy)
            x0k, x1k = max(0, -dx), w - max(0, dx)
            if maximize:
                out[y0s:y1s, x0s:x1s] = np.maximum(out[y0s:y1s, x0s:x1s], src[y0k:y1k, x0k:x1k])
            else:
                out[y0s:y1s, x0s:x1s] = np.minimum(out[y0s:y1s, x0s:x1s], src[y0k:y1k, x0k:x1k])
    return out


def _morph_close_dilate(mask: np.ndarray, style: InpaintStyle) -> np.ndarray:
    """Close letter gaps, then dilate antialiased edges — at this crop's native pixels."""
    cv2 = _try_cv2()
    u8 = np.asarray(mask, dtype=np.uint8)
    if u8.max() <= 1:
        u8 = (u8 > 0).astype(np.uint8) * 255
    ck, dk = style.close_kernel, style.dilate_kernel
    if cv2 is not None:
        closed = cv2.morphologyEx(
            u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, ck)
        )
        if style.dilate_iterations > 0:
            closed = cv2.dilate(
                closed,
                cv2.getStructuringElement(cv2.MORPH_RECT, dk),
                iterations=int(style.dilate_iterations),
            )
        return closed
    closed = _rank_rect(_rank_rect(u8, ck[0], ck[1], maximize=True), ck[0], ck[1], maximize=False)
    for _ in range(max(0, int(style.dilate_iterations))):
        closed = _rank_rect(closed, dk[0], dk[1], maximize=True)
    return closed


def text_mask_from_quads(
    height: int,
    width: int,
    quads: Sequence[Quad],
    *,
    style: InpaintStyle | None = None,
    cancel=None,
) -> np.ndarray:
    """uint8 0/255 mask: fillPoly quads, close, dilate. Coordinates are this crop."""
    spec = style if style is not None else STYLE_GEOMETRIC_SPEC
    _raise_if_cancelled(cancel)
    polygons: list[tuple[tuple[int, int], ...]] = []
    for quad in quads:
        _raise_if_cancelled(cancel)
        polygons.extend(_inflate_quad(quad, spec.pad_x, spec.pad_y_down))
    raw = _fill_poly_mask(height, width, polygons)
    if not np.any(raw):
        return raw
    return _morph_close_dilate(raw, spec)


def _cv2_inpaint_crop(rgb: np.ndarray, hole: np.ndarray) -> np.ndarray | None:
    cv2 = _try_cv2()
    if cv2 is None:
        return None
    src = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
    mask = np.asarray(hole, dtype=np.uint8)
    if mask.max() <= 1:
        mask = (mask > 0).astype(np.uint8) * 255
    if not np.any(mask):
        return src
    try:
        return cv2.inpaint(src, mask, 7, cv2.INPAINT_NS)
    except cv2.error:
        return None


def _fill_crop_backend(rgb: np.ndarray, hole: np.ndarray, *, wrap: bool, cancel=None) -> np.ndarray:
    """LaMa (native crop, pad×8) → OpenCV NS → numpy Hilbert. Never downscales."""
    _raise_if_cancelled(cancel)
    if not np.any(hole):
        return np.array(rgb, copy=True)
    filled = lama_inpaint_crop(rgb, hole)
    if filled is not None:
        _set_backend("lama")
        out = np.array(rgb, copy=True)
        hole_b = hole.astype(bool)
        out[hole_b] = filled[hole_b]
        return out
    filled = _cv2_inpaint_crop(rgb, hole)
    if filled is not None:
        _set_backend("cv2")
        return filled
    _set_backend("numpy")
    return _pattern_fill(rgb, np.asarray(hole, dtype=bool), wrap=wrap, cancel=cancel)


def _seam_band(span: int) -> int:
    return max(_SEAM_MIN, min(_SEAM_MAX, int(round(int(span) / 16.0))))


def wrap_seam_mask(height: int, width: int, *, wrap_h: bool, wrap_v: bool) -> np.ndarray:
    """Plus-shaped hole at the rolled midlines (horizontal and/or vertical wrap)."""
    h, w = max(1, int(height)), max(1, int(width))
    mask = np.zeros((h, w), dtype=bool)
    if wrap_h:
        bw = _seam_band(w)
        x0 = max(0, w // 2 - bw // 2)
        mask[:, x0 : x0 + bw] = True
    if wrap_v:
        bh = _seam_band(h)
        y0 = max(0, h // 2 - bh // 2)
        mask[y0 : y0 + bh, :] = True
    return mask


def _seam_windows(
    height: int,
    width: int,
    *,
    wrap_h: bool,
    wrap_v: bool,
) -> list[tuple[int, int, int, int]]:
    """Overlapping LaMa crops along the rolled wrap seams (not the full frame)."""
    h, w = max(1, int(height)), max(1, int(width))
    tile = _LAMA_TILE
    overlap = _LAMA_OVERLAP
    context = _SEAM_CONTEXT
    windows: list[tuple[int, int, int, int]] = []

    def _along(vertical: bool) -> None:
        if vertical:
            cx = w // 2
            half = _seam_band(w) // 2 + context
            x0 = max(0, cx - half)
            x1 = min(w, cx + half)
            if x1 - x0 > tile:
                mid = (x0 + x1) // 2
                x0 = max(0, mid - tile // 2)
                x1 = min(w, x0 + tile)
            y = 0
            while y < h:
                y1 = min(h, y + tile)
                windows.append((y, x0, y1, x1))
                if y1 >= h:
                    break
                nxt = y1 - overlap
                if nxt <= y:
                    break
                y = nxt
            return
        cy = h // 2
        half = _seam_band(h) // 2 + context
        y0 = max(0, cy - half)
        y1 = min(h, cy + half)
        if y1 - y0 > tile:
            mid = (y0 + y1) // 2
            y0 = max(0, mid - tile // 2)
            y1 = min(h, y0 + tile)
        x = 0
        while x < w:
            x1 = min(w, x + tile)
            windows.append((y0, x, y1, x1))
            if x1 >= w:
                break
            nxt = x1 - overlap
            if nxt <= x:
                break
            x = nxt

    if wrap_h:
        _along(True)
    if wrap_v:
        _along(False)
    return windows


def inpaint_wrap_seams(
    arr: np.ndarray,
    *,
    wrap_h: bool,
    wrap_v: bool,
    cancel=None,
) -> np.ndarray:
    """Make opposite edges meet by inpainting them together in the rolled canvas.

    Rolls the image so wrap edges sit at the midlines, fills a plus-shaped
    seam with LaMa (then OpenCV, then period/Hilbert copies from this image),
    and rolls back. Identity when both wraps are off or the frame is tiny.
    """
    _raise_if_cancelled(cancel)
    src = np.asarray(arr)
    if src.size == 0 or src.ndim < 2:
        return np.array(src, copy=True)
    if not wrap_h and not wrap_v:
        return np.array(src, copy=True)
    h, w = int(src.shape[0]), int(src.shape[1])
    if h < 16 or w < 16:
        return np.array(src, copy=True)
    dy = h // 2 if wrap_v else 0
    dx = w // 2 if wrap_h else 0
    rolled = np.roll(np.roll(src, dy, axis=0), dx, axis=1)
    hole = wrap_seam_mask(h, w, wrap_h=wrap_h, wrap_v=wrap_v)
    if not np.any(hole):
        return np.array(src, copy=True)
    if rolled.ndim == 3 and rolled.shape[-1] >= 3:
        rgb = np.ascontiguousarray(rolled[..., :3], dtype=np.uint8)
        out = np.array(rolled, copy=True)
        filled_rgb = _fill_wrap_seam_rgb(rgb, hole, wrap_h=wrap_h, wrap_v=wrap_v, cancel=cancel)
        out[..., :3] = filled_rgb
        return np.roll(np.roll(out, -dy, axis=0), -dx, axis=1)
    filled = _pattern_fill(rolled, hole, wrap=True, cancel=cancel)
    return np.roll(np.roll(filled, -dy, axis=0), -dx, axis=1)


def _fill_wrap_seam_rgb(
    rgb: np.ndarray,
    hole: np.ndarray,
    *,
    wrap_h: bool,
    wrap_v: bool,
    cancel=None,
) -> np.ndarray:
    """LaMa along seam windows, then OpenCV, then wrap-aware period/Hilbert."""
    _raise_if_cancelled(cancel)
    left = np.asarray(hole, dtype=bool)
    out = np.array(rgb, copy=True)
    windows = _seam_windows(out.shape[0], out.shape[1], wrap_h=wrap_h, wrap_v=wrap_v)
    lama, covered = lama_inpaint_windows(out, left, windows, overlap=_LAMA_OVERLAP)
    if lama is not None and covered is not None:
        _set_backend("lama")
        out = np.where(covered[..., None], lama, out)
        left = left & ~covered
    if np.any(left):
        for y0, x0, y1, x1 in windows:
            _raise_if_cancelled(cancel)
            sub = left[y0:y1, x0:x1]
            if not np.any(sub):
                continue
            filled = _cv2_inpaint_crop(out[y0:y1, x0:x1], sub)
            if filled is None:
                continue
            crop = np.array(out[y0:y1, x0:x1], copy=True)
            crop[sub] = filled[sub]
            out[y0:y1, x0:x1] = crop
            left[y0:y1, x0:x1] = False
            _set_backend("cv2")
    if np.any(left):
        _set_backend("numpy")
        out = _pattern_fill(out, left, wrap=True, cancel=cancel)
    return out


def _map_boxes_to_image(
    boxes: Sequence[Box],
    image_size: tuple[int, int],
    src_size: tuple[int, int] | None,
) -> list[Box]:
    iw, ih = image_size
    mapped = list(boxes)
    if src_size is None:
        return mapped
    sw, sh = max(1, int(src_size[0])), max(1, int(src_size[1]))
    if (sw, sh) == (iw, ih):
        return mapped
    return [
        (
            int(round(x0 * iw / sw)),
            int(round(y0 * ih / sh)),
            int(round(x1 * iw / sw)),
            int(round(y1 * ih / sh)),
        )
        for x0, y0, x1, y1 in boxes
    ]


def _map_quads_to_image(
    quads: Sequence[Quad],
    image_size: tuple[int, int],
    src_size: tuple[int, int] | None,
) -> list[tuple[tuple[int, int], ...]]:
    iw, ih = image_size
    if src_size is None:
        return [tuple((int(round(float(x))), int(round(float(y)))) for x, y in q) for q in quads]
    sw, sh = max(1, int(src_size[0])), max(1, int(src_size[1]))
    if (sw, sh) == (iw, ih):
        return [tuple((int(round(float(x))), int(round(float(y)))) for x, y in q) for q in quads]
    return [
        tuple(
            (
                int(round(float(x) * iw / sw)),
                int(round(float(y) * ih / sh)),
            )
            for x, y in q
        )
        for q in quads
    ]


def _crop_around_box(
    height: int, width: int, box: Box, spec: InpaintStyle
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(v) for v in box)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    pad = max(0, int(spec.crop_pad))
    px = pad + max(0, int(spec.pad_x))
    y1 = y1 + max(0, int(spec.pad_y_down))
    return (
        max(0, x0 - px),
        max(0, y0 - pad),
        min(width, x1 + px),
        min(height, y1 + pad),
    )


def _erode8(mask: np.ndarray, times: int = 1) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(times))):
        acc = out.copy()
        acc[1:, :] &= out[:-1, :]
        acc[:-1, :] &= out[1:, :]
        acc[:, 1:] &= out[:, :-1]
        acc[:, :-1] &= out[:, 1:]
        acc[1:, 1:] &= out[:-1, :-1]
        acc[1:, :-1] &= out[:-1, 1:]
        acc[:-1, 1:] &= out[1:, :-1]
        acc[:-1, :-1] &= out[1:, 1:]
        out = acc
    return out


def glyph_mask_from_boxes(
    rgb: np.ndarray,
    boxes: Sequence[Box],
    *,
    cancel=None,
) -> np.ndarray:
    """True on high-contrast ink strokes inside each box, not the whole rectangle."""
    src = np.asarray(rgb)
    h, w = src.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    if h == 0 or w == 0 or not boxes:
        return mask
    gray = luma_channel(src[..., :3] if src.ndim == 3 else src)
    ring = max(4, int(_GLYPH_RING))
    for raw in boxes:
        _raise_if_cancelled(cancel)
        if raw is None or len(raw) != 4:
            continue
        x0, y0, x1, y1 = (int(round(float(v))) for v in raw)
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w, x1)
        y1 = min(h, y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        rx0 = max(0, x0 - ring)
        ry0 = max(0, y0 - ring)
        rx1 = min(w, x1 + ring)
        ry1 = min(h, y1 + ring)
        ring_m = np.ones((ry1 - ry0, rx1 - rx0), dtype=bool)
        iy0, iy1 = y0 - ry0, y1 - ry0
        ix0, ix1 = x0 - rx0, x1 - rx0
        ring_m[iy0:iy1, ix0:ix1] = False
        ring_vals = gray[ry0:ry1, rx0:rx1][ring_m]
        if ring_vals.size < 8:
            ring_vals = gray[ry0:ry1, rx0:rx1].ravel()
        if ring_vals.size < 4:
            continue
        med = float(np.median(ring_vals))
        mad = float(np.median(np.abs(ring_vals.astype(np.float32) - med))) + 1.0
        delta = max(_INK_DELTA, 2.2 * mad)
        inner = gray[y0:y1, x0:x1]
        dark = inner < (med - delta)
        light = inner > (med + delta + 4.0)
        n_dark = int(dark.sum())
        n_light = int(light.sum())
        area = max(1, inner.size)
        if n_dark >= n_light and n_dark >= max(4, area // 400):
            ink = dark
        elif n_light >= max(4, area // 400):
            ink = light
        else:
            ink = dark | light
        if int(ink.sum()) < 3:
            continue
        ink = _dilate8(_erode8(ink, 1), 2)
        mask[y0:y1, x0:x1] |= ink
    return mask


def _pattern_fill(
    rgb: np.ndarray,
    hole: np.ndarray,
    *,
    wrap: bool = False,
    cancel=None,
) -> np.ndarray:
    """Copy weave from a motif-period offset, then Hilbert-fill leftover strokes."""
    if not np.any(hole):
        return np.array(rgb, copy=True)
    h, w = hole.shape
    out = np.array(rgb, copy=True)
    remaining = hole.copy()
    px = int(estimate_axis_period(out, 1))
    py = int(estimate_axis_period(out, 0))
    offsets: list[tuple[int, int]] = []
    if 4 <= px < w // 2:
        offsets.extend(((px, 0), (-px, 0), (2 * px, 0), (-2 * px, 0)))
    if 4 <= py < h // 2:
        offsets.extend(((0, py), (0, -py)))
    if 4 <= px < w // 2 and 4 <= py < h // 2:
        offsets.extend(((px, py), (-px, py), (px, -py), (-px, -py)))
    known = ~hole
    nd = 3 if out.ndim == 3 else 0
    for ox, oy in offsets:
        _raise_if_cancelled(cancel)
        if not np.any(remaining):
            break
        ys, xs = np.nonzero(remaining)
        sy = ys + oy
        sx = xs + ox
        if wrap:
            sy %= h
            sx %= w
            valid = np.ones(ys.shape, dtype=bool)
        else:
            valid = (sy >= 0) & (sy < h) & (sx >= 0) & (sx < w)
            sy = np.where(valid, sy, 0)
            sx = np.where(valid, sx, 0)
        ok = valid & known[sy, sx]
        if not np.any(ok):
            continue
        out[ys[ok], xs[ok]] = out[sy[ok], sx[ok]]
        remaining[ys[ok], xs[ok]] = False
    if np.any(remaining):
        filled = _fill_crop(out, remaining, wrap=wrap, cancel=cancel)
        if nd:
            out = np.where(remaining[..., None], filled, out)
        else:
            out = np.where(remaining, filled, out)
    return out


def _fill_crop(rgb: np.ndarray, hole: np.ndarray, *, wrap: bool = False, cancel=None) -> np.ndarray:
    """Grow from known pixels into ``hole`` with Hilbert-local neighbor copies."""
    if not np.any(hole):
        return rgb
    known = ~hole
    if not np.any(known):
        return rgb
    h, w = hole.shape
    ch = 1 if rgb.ndim == 2 else int(rgb.shape[-1])
    out = rgb.astype(np.float32, copy=True)
    if out.ndim == 2:
        out = out[..., None]
    order = _hilbert_order(max(h, w))
    yy, xx = np.indices((h, w), dtype=np.int64)
    hid = hilbert_xy_to_d(xx, yy, order).astype(np.float32)
    hid_span = float(max((1 << order) ** 2 - 1, 1))
    max_iter = int(h + w + 4)
    for _ in range(max_iter):
        neigh_known = np.zeros((h, w), dtype=bool)
        if wrap:
            for dy, dx in _NEIGH:
                neigh_known |= np.roll(np.roll(known, dy, axis=0), dx, axis=1)
        else:
            for dy, dx in _NEIGH:
                y0s, y1s = max(0, dy), h + min(0, dy)
                x0s, x1s = max(0, dx), w + min(0, dx)
                y0k, y1k = max(0, -dy), h - max(0, dy)
                x0k, x1k = max(0, -dx), w - max(0, dx)
                neigh_known[y0s:y1s, x0s:x1s] |= known[y0k:y1k, x0k:x1k]
        front = (~known) & neigh_known
        if not np.any(front):
            break
        _raise_if_cancelled(cancel)
        best_w = np.full((h, w), -1.0, dtype=np.float32)
        best = np.zeros((h, w, ch), dtype=np.float32)
        mean_acc = np.zeros((h, w, ch), dtype=np.float32)
        mean_n = np.zeros((h, w), dtype=np.float32)
        for dy, dx in _NEIGH:
            if wrap:
                ny = (yy + dy) % h
                nx = (xx + dx) % w
            else:
                ny = np.clip(yy + dy, 0, h - 1)
                nx = np.clip(xx + dx, 0, w - 1)
            ok = known[ny, nx] & front
            if not np.any(ok):
                continue
            dist = float(dy * dy + dx * dx) ** 0.5
            hd = np.abs(hid - hid[ny, nx]) / hid_span
            wt = ok.astype(np.float32) / (dist + 0.35) / (1.0 + 6.0 * hd)
            src = out[ny, nx]
            take = wt > best_w
            best_w = np.where(take, wt, best_w)
            best = np.where(take[..., None], src, best)
            mean_acc += src * wt[..., None]
            mean_n += wt
        filled = (best_w >= 0.0) & front
        if not np.any(filled):
            break
        n = np.maximum(mean_n, 1e-6)
        blended = 0.86 * best + 0.14 * (mean_acc / n[..., None])
        out = np.where(filled[..., None], blended, out)
        known |= filled
    if rgb.ndim == 2:
        return np.clip(np.rint(out[..., 0]), 0, 255).astype(np.uint8)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def inpaint_array(arr: np.ndarray, mask: np.ndarray, *, wrap: bool = False, cancel=None) -> np.ndarray:
    """Fill ``mask`` True pixels. Empty mask returns a copy (identity)."""
    _raise_if_cancelled(cancel)
    src = np.asarray(arr)
    hole = np.asarray(mask, dtype=bool)
    if hole.shape[:2] != src.shape[:2]:
        raise ValueError("mask shape must match image height/width")
    if not np.any(hole):
        return np.array(src, copy=True)
    if src.ndim == 3 and src.shape[-1] >= 3:
        rgb = np.ascontiguousarray(src[..., :3])
        filled = np.array(src, copy=True)
        filled[..., :3] = _inpaint_rgb(rgb, hole, wrap=wrap, cancel=cancel)
        return filled
    return _inpaint_rgb(src, hole, wrap=wrap, cancel=cancel)


def _inpaint_rgb(rgb: np.ndarray, hole: np.ndarray, *, wrap: bool = False, cancel=None) -> np.ndarray:
    ys, xs = np.nonzero(hole)
    if ys.size == 0:
        return np.array(rgb, copy=True)
    h, w = hole.shape
    hole_h = int(ys.max()) - int(ys.min()) + 1
    hole_w = int(xs.max()) - int(xs.min()) + 1
    margin = max(_CROP_MARGIN, min(_CROP_MARGIN_MAX, max(hole_h, hole_w)))
    y0 = max(0, int(ys.min()) - margin)
    x0 = max(0, int(xs.min()) - margin)
    y1 = min(h, int(ys.max()) + 1 + margin)
    x1 = min(w, int(xs.max()) + 1 + margin)
    crop = np.array(rgb[y0:y1, x0:x1], copy=True)
    crop_hole = hole[y0:y1, x0:x1]
    filled = _pattern_fill(crop, crop_hole, wrap=wrap, cancel=cancel)
    out = np.array(rgb, copy=True)
    out[y0:y1, x0:x1] = filled
    return out


def inpaint_image(
    image: Image.Image,
    boxes: Sequence[Box] | None,
    *,
    src_size: tuple[int, int] | None = None,
    pad: int = _PAD_DEFAULT,
    wrap: bool = False,
    glyph: bool = True,
    cancel=None,
    quads: Sequence[Quad] | None = None,
    style: str | None = None,
) -> Image.Image:
    """Inpaint OCR glyphs on source-resolution crops (not a 12k full-frame pass).

    ``boxes`` / ``quads`` are ``src_size`` pixels (or this image). Mask is
    fillPoly + wallpaper morphology. Fill is LaMa, else cv2 NS, else numpy.
    ``glyph`` is kept for callers; the wallpaper mask replaces bbox diffusion.
    """
    del glyph, pad  # wallpaper mask replaces bbox Telea / Hilbert-on-rect
    _raise_if_cancelled(cancel)
    if image is None:
        raise ValueError("image is required")
    if not boxes:
        _set_backend("identity")
        return image.copy()
    spec = inpaint_style(style)
    iw, ih = image.size
    mapped = _map_boxes_to_image(list(boxes), (iw, ih), src_size)
    mapped_quads: list[Quad]
    if quads:
        mapped_quads = list(_map_quads_to_image(quads, (iw, ih), src_size))
        while len(mapped_quads) < len(mapped):
            mapped_quads.append(aabb_quad(mapped[len(mapped_quads)]))
    else:
        mapped_quads = [aabb_quad(box) for box in mapped]
    rgba = image.mode == "RGBA"
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    out = np.array(rgb, copy=True)
    used = False
    for box, quad in zip(mapped, mapped_quads):
        _raise_if_cancelled(cancel)
        x0, y0, x1, y1 = _crop_around_box(ih, iw, box, spec)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        crop = out[y0:y1, x0:x1]
        local = [
            tuple((int(px) - x0, int(py) - y0) for px, py in quad),
        ]
        hole = text_mask_from_quads(
            y1 - y0, x1 - x0, local, style=spec, cancel=cancel
        )
        if not np.any(hole):
            continue
        filled = _fill_crop_backend(crop, hole, wrap=wrap, cancel=cancel)
        out[y0:y1, x0:x1] = filled
        used = True
    if not used:
        _set_backend("identity")
        return image.copy()
    result = Image.fromarray(out, mode="RGB")
    if rgba:
        alpha = image.split()[-1]
        result = result.convert("RGBA")
        result.putalpha(alpha)
    elif image.mode != "RGB":
        result = result.convert(image.mode)
    return result
