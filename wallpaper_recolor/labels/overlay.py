# -*- coding: utf-8 -*-
"""
wallpaper_recolor.labels.overlay
--------------------------------
Dashed / solid detection boxes on Original and Result, plus compositing
the editable label on the Result preview.

Class references (code + name only):
- CAP3321C Data Wrangling
"""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image, ImageDraw

from wallpaper_recolor.labels.boxes import Box, source_box_to_display
from wallpaper_recolor.labels.layer import LabelSpec, composite_label

_HALO = (248, 244, 236)
_STROKE = (28, 22, 18)
_SELECTED = (74, 144, 217)
_DASH = 6
_GAP = 4


def _dashed_line(
    draw: ImageDraw.ImageDraw,
    a: tuple[float, float],
    b: tuple[float, float],
    fill: tuple[int, int, int],
    width: int,
) -> None:
    x0, y0 = a
    x1, y1 = b
    dx = float(x1) - float(x0)
    dy = float(y1) - float(y0)
    seg = (dx * dx + dy * dy) ** 0.5
    if seg < 1e-6:
        return
    ux, uy = dx / seg, dy / seg
    dist = 0.0
    on = True
    remain = float(_DASH)
    cx, cy = float(x0), float(y0)
    while dist < seg:
        step = min(remain, seg - dist)
        nx, ny = cx + ux * step, cy + uy * step
        if on:
            draw.line([(cx, cy), (nx, ny)], fill=fill, width=width)
        dist += step
        cx, cy = nx, ny
        remain -= step
        if remain <= 1e-6:
            on = not on
            remain = float(_DASH if on else _GAP)


def _rect_outline(
    draw: ImageDraw.ImageDraw,
    box: Box,
    fill: tuple[int, int, int],
    *,
    dashed: bool,
    width: int,
) -> None:
    x0, y0, x1, y1 = box
    pts = (
        (x0, y0),
        (x1 - 1, y0),
        (x1 - 1, y1 - 1),
        (x0, y1 - 1),
    )
    closed = pts + (pts[0],)
    if dashed:
        for a, b in zip(closed, closed[1:]):
            _dashed_line(draw, a, b, fill, width)
        return
    draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=fill, width=width)


def _preview_draw_copy(image: Image.Image) -> Image.Image:
    """Keep alpha so knockout holes survive box overlays."""
    if image.mode == "RGBA":
        return image.copy()
    if image.mode in ("LA", "PA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        return image.convert("RGBA").copy()
    return image.convert("RGB").copy()


def draw_label_boxes(
    image: Image.Image,
    boxes: Sequence[Box],
    selected: Sequence[int] | None,
    source_size: tuple[int, int],
    *,
    crop_x: float = 0.0,
    crop_y: float = 0.0,
    crop_zoom: float = 1.0,
) -> Image.Image:
    """Overlay detection boxes. Selected = solid; others dashed."""
    if not boxes:
        return image
    out = _preview_draw_copy(image)
    draw = ImageDraw.Draw(out)
    chosen = set(int(i) for i in (selected or ()))
    for i, box in enumerate(boxes):
        mapped = source_box_to_display(
            box, out.size, source_size, crop_x, crop_y, crop_zoom
        )
        if mapped is None:
            continue
        sel = i in chosen
        color = _SELECTED if sel else _STROKE
        _rect_outline(draw, mapped, _HALO, dashed=not sel, width=3)
        _rect_outline(draw, mapped, color, dashed=not sel, width=1)
    return out


def draw_roi_box(
    image: Image.Image,
    roi: Box | None,
    source_size: tuple[int, int],
    *,
    crop_x: float = 0.0,
    crop_y: float = 0.0,
    crop_zoom: float = 1.0,
) -> Image.Image:
    """Dashed search-area rectangle. Same size as ``image`` (copy if drawing)."""
    if roi is None:
        return image
    mapped = source_box_to_display(
        roi, image.size, source_size, crop_x, crop_y, crop_zoom
    )
    if mapped is None:
        return image
    out = _preview_draw_copy(image)
    draw = ImageDraw.Draw(out)
    _rect_outline(draw, mapped, _HALO, dashed=True, width=3)
    _rect_outline(draw, mapped, _SELECTED, dashed=True, width=1)
    return out


def decorate_preview(
    image: Image.Image,
    boxes: Sequence[Box],
    selected: Sequence[int] | None,
    source_size: tuple[int, int],
    *,
    crop_x: float = 0.0,
    crop_y: float = 0.0,
    crop_zoom: float = 1.0,
    label: LabelSpec | None = None,
    show_label: bool = False,
    roi: Box | None = None,
) -> Image.Image:
    """Boxes / ROI on a same-size copy; optional label plate on Result."""
    out = draw_roi_box(
        image,
        roi,
        source_size,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_zoom=crop_zoom,
    )
    out = draw_label_boxes(
        out,
        boxes,
        selected,
        source_size,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_zoom=crop_zoom,
    )
    if show_label and label is not None and label.is_set():
        out = composite_label(
            out,
            label,
            source_size,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_zoom=crop_zoom,
        )
    return out
