# -*- coding: utf-8 -*-
"""
wallpaper_recolor.labels.boxes
------------------------------
Axis-aligned text boxes in **source pixels**, mapped through Crop onto the
preview / work image. View-zoom is display-only and is applied by the UI
after these helpers.

Class references (code + name only):
- CAP3321C Data Wrangling
"""

from __future__ import annotations

from collections.abc import Sequence

from wallpaper_recolor.transform.crop import clamp_zoom, paste_top_left

Box = tuple[int, int, int, int]
Quad = tuple[tuple[int, int], ...]


def aabb_quad(box: Box) -> Quad:
    """Four corners of an axis-aligned box (TL, TR, BR, BL)."""
    x0, y0, x1, y1 = box
    return ((int(x0), int(y0)), (int(x1), int(y0)), (int(x1), int(y1)), (int(x0), int(y1)))


def normalize_box(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: int,
    height: int,
) -> Box | None:
    """Inclusive-exclusive box clipped to ``width`` × ``height``, or None if empty."""
    w = max(1, int(width))
    h = max(1, int(height))
    a = int(round(min(float(x0), float(x1))))
    b = int(round(max(float(x0), float(x1))))
    c = int(round(min(float(y0), float(y1))))
    d = int(round(max(float(y0), float(y1))))
    a = max(0, min(a, w))
    b = max(0, min(b, w))
    c = max(0, min(c, h))
    d = max(0, min(d, h))
    if b - a < 2 or d - c < 2:
        return None
    return (a, c, b, d)


def scale_box(box: Box, src_size: tuple[int, int], dest_size: tuple[int, int]) -> Box:
    """Map a box from ``src_size`` onto ``dest_size`` (nearest)."""
    sw, sh = max(1, int(src_size[0])), max(1, int(src_size[1]))
    dw, dh = max(1, int(dest_size[0])), max(1, int(dest_size[1]))
    x0, y0, x1, y1 = box
    nx0 = int(round(x0 * dw / sw))
    ny0 = int(round(y0 * dh / sh))
    nx1 = int(round(x1 * dw / sw))
    ny1 = int(round(y1 * dh / sh))
    out = normalize_box(nx0, ny0, nx1, ny1, dw, dh)
    return out if out is not None else (0, 0, min(2, dw), min(2, dh))


def scale_boxes(
    boxes: Sequence[Box],
    src_size: tuple[int, int],
    dest_size: tuple[int, int],
) -> list[Box]:
    return [scale_box(b, src_size, dest_size) for b in boxes]


def _frame_paste(
    source_size: tuple[int, int],
    crop_x: float,
    crop_y: float,
    crop_zoom: float,
) -> tuple[int, int, float, int, int]:
    """Scaled-image top-left in the frame, zoom, and frame size (source pixels)."""
    sw, sh = max(1, int(source_size[0])), max(1, int(source_size[1]))
    z = clamp_zoom(crop_zoom)
    left, top, _nw, _nh = paste_top_left(sw, sh, crop_x, crop_y, z)
    return left, top, z, sw, sh


def display_xy_to_source(
    px: float,
    py: float,
    display_size: tuple[int, int],
    source_size: tuple[int, int],
    crop_x: float = 0.0,
    crop_y: float = 0.0,
    crop_zoom: float = 1.0,
) -> tuple[int, int]:
    """Map a pixel on the framed preview onto source coordinates.

    Display maps onto the full frame (same size as the source). Source is
    recovered with the same paste+zoom affine as ``apply_crop``: zoom about
    the frame center, X/Y = image-center offset from that center.
    """
    dw, dh = max(1, int(display_size[0])), max(1, int(display_size[1]))
    left, top, z, sw, sh = _frame_paste(source_size, crop_x, crop_y, crop_zoom)
    fx = float(px) * sw / dw
    fy = float(py) * sh / dh
    sx = (fx - left) / z
    sy = (fy - top) / z
    return (
        int(max(0, min(sw - 1, round(sx)))),
        int(max(0, min(sh - 1, round(sy)))),
    )


def display_box_to_source(
    box: Box,
    display_size: tuple[int, int],
    source_size: tuple[int, int],
    crop_x: float = 0.0,
    crop_y: float = 0.0,
    crop_zoom: float = 1.0,
) -> Box | None:
    """Drag-rect on the preview → source-pixel box (Crop X/Y/zoom applied)."""
    x0, y0, x1, y1 = box
    sx0, sy0 = display_xy_to_source(
        x0, y0, display_size, source_size, crop_x, crop_y, crop_zoom
    )
    sx1, sy1 = display_xy_to_source(
        x1, y1, display_size, source_size, crop_x, crop_y, crop_zoom
    )
    sw, sh = source_size
    return normalize_box(sx0, sy0, sx1, sy1, sw, sh)


def source_box_to_display(
    box: Box,
    display_size: tuple[int, int],
    source_size: tuple[int, int],
    crop_x: float = 0.0,
    crop_y: float = 0.0,
    crop_zoom: float = 1.0,
) -> Box | None:
    """Source box onto the framed preview; None if fully outside the frame."""
    dw, dh = max(1, int(display_size[0])), max(1, int(display_size[1]))
    left, top, z, sw, sh = _frame_paste(source_size, crop_x, crop_y, crop_zoom)
    x0, y0, x1, y1 = box
    dx0 = (float(x0) * z + left) * dw / sw
    dy0 = (float(y0) * z + top) * dh / sh
    dx1 = (float(x1) * z + left) * dw / sw
    dy1 = (float(y1) * z + top) * dh / sh
    return normalize_box(dx0, dy0, dx1, dy1, dw, dh)


def source_xy_to_display(
    x: float,
    y: float,
    display_size: tuple[int, int],
    source_size: tuple[int, int],
    crop_x: float = 0.0,
    crop_y: float = 0.0,
    crop_zoom: float = 1.0,
) -> tuple[float, float, float]:
    """Source point + scale (display px per source px) on the framed preview."""
    dw, dh = max(1, int(display_size[0])), max(1, int(display_size[1]))
    left, top, z, sw, sh = _frame_paste(source_size, crop_x, crop_y, crop_zoom)
    scale = float(dw) / float(sw) * z
    dx = (float(x) * z + left) * float(dw) / float(sw)
    dy = (float(y) * z + top) * float(dh) / float(sh)
    return dx, dy, scale


def box_contains(box: Box, x: float, y: float, slop: int = 2) -> bool:
    x0, y0, x1, y1 = box
    s = max(0, int(slop))
    return (x0 - s) <= float(x) < (x1 + s) and (y0 - s) <= float(y) < (y1 + s)
