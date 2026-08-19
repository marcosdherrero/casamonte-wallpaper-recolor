# -*- coding: utf-8 -*-
"""
wallpaper_recolor.transform.crop
----------------------
Position & Zoom places the image in a **frame the size of the source**.

X/Y are offsets of the **image center** from the **frame center**, in source
pixels. (0, 0) is centered. Zoom scales about the frame center (not a corner).
Empty frame is transparent (preview checker; save keeps alpha). Offsets are
**not** clamped — the image can sit fully off-canvas.

``window_size`` / ``max_origin`` / ``crop_array`` remain the tight in-image
slice Tessellate Build uses while searching for a wrap window.

Class references (code + name only):
- CAP3321C Data Wrangling
"""

from __future__ import annotations

import numpy as np
from PIL import Image

ZOOM_MIN = 1.0
ZOOM_MAX = 8.0
ZOOM_DEFAULT = 1.0
CROP_XY_DEFAULT = 0


def clamp_zoom(zoom: float) -> float:
    """Keep zoom in [1, ZOOM_MAX]."""
    try:
        z = float(zoom)
    except (TypeError, ValueError):
        return ZOOM_DEFAULT
    if z != z:  # NaN
        return ZOOM_DEFAULT
    return min(ZOOM_MAX, max(ZOOM_MIN, z))


def window_size(src_w: int, src_h: int, zoom: float) -> tuple[int, int]:
    """Tight wrap-search window (never larger than the source). Tessellate only."""
    z = clamp_zoom(zoom)
    src_w = max(1, int(src_w))
    src_h = max(1, int(src_h))
    cw = max(1, min(src_w, int(round(src_w / z))))
    ch = max(1, min(src_h, int(round(src_h / z))))
    return cw, ch


def max_origin(src_w: int, src_h: int, zoom: float) -> tuple[int, int]:
    """Largest legal top-left so a Tessellate wrap-search window still fits."""
    cw, ch = window_size(src_w, src_h, zoom)
    return max(0, int(src_w) - cw), max(0, int(src_h) - ch)


def offset_slider_limit(src_w: int, src_h: int, zoom: float) -> tuple[int, int]:
    """Slider half-range (px) so the image can leave the frame; values are not clamped."""
    z = clamp_zoom(zoom)
    src_w = max(1, int(src_w))
    src_h = max(1, int(src_h))
    return (
        max(1, int(round(src_w * z))),
        max(1, int(round(src_h * z))),
    )


def paste_top_left(
    frame_w: int,
    frame_h: int,
    x: float,
    y: float,
    zoom: float,
) -> tuple[int, int, int, int]:
    """Scaled-image top-left in the frame, plus scaled width/height.

    ``(x, y)`` is the image-center offset from the frame center, in frame pixels.
    """
    frame_w = max(1, int(frame_w))
    frame_h = max(1, int(frame_h))
    z = clamp_zoom(zoom)
    nw = max(1, int(round(frame_w * z)))
    nh = max(1, int(round(frame_h * z)))
    left = int(round(frame_w * 0.5 + float(x) - nw * 0.5))
    top = int(round(frame_h * 0.5 + float(y) - nh * 0.5))
    return left, top, nw, nh


def top_left_to_center_offset(
    src_w: int,
    src_h: int,
    left: float,
    top: float,
    zoom: float,
) -> tuple[float, float]:
    """Map a Tessellate tight-window origin onto center-offset X/Y."""
    src_w = max(1, int(src_w))
    src_h = max(1, int(src_h))
    z = clamp_zoom(zoom)
    cw, ch = window_size(src_w, src_h, z)
    region_cx = float(left) + cw * 0.5
    region_cy = float(top) + ch * 0.5
    ox = -(region_cx - src_w * 0.5) * z
    oy = -(region_cy - src_h * 0.5) * z
    return ox, oy


def crop_box(
    src_w: int,
    src_h: int,
    x: float,
    y: float,
    zoom: float,
) -> tuple[int, int, int, int]:
    """Source-space rectangle covering the output frame (may extend outside the image)."""
    src_w = max(1, int(src_w))
    src_h = max(1, int(src_h))
    z = clamp_zoom(zoom)
    left, top, _nw, _nh = paste_top_left(src_w, src_h, x, y, z)
    s0x = (0.0 - left) / z
    s0y = (0.0 - top) / z
    s1x = (float(src_w) - left) / z
    s1y = (float(src_h) - top) / z
    return (
        int(round(s0x)),
        int(round(s0y)),
        int(round(s1x)),
        int(round(s1y)),
    )


def is_identity_crop(src_w: int, src_h: int, x: float, y: float, zoom: float) -> bool:
    """True when the image is centered at zoom 1 (full frame, no empty margin)."""
    del src_w, src_h
    z = clamp_zoom(zoom)
    if abs(z - 1.0) > 1e-6:
        return False
    return abs(float(x)) < 0.5 and abs(float(y)) < 0.5


def apply_crop(
    image: Image.Image,
    x: float,
    y: float,
    zoom: float,
    *,
    src_size: tuple[int, int] | None = None,
) -> Image.Image:
    """Place ``image`` in a same-size frame using center-offset X/Y and zoom.

    When ``src_size`` is the full-res size and ``image`` is a downscaled
    work copy, X/Y (source px) are mapped into this image's pixel space.
    Zoom is scale-invariant. Off-frame pixels are transparent (RGBA).
    """
    iw, ih = image.size
    if src_size is None:
        mx, my = float(x), float(y)
    else:
        sw, sh = max(1, int(src_size[0])), max(1, int(src_size[1]))
        mx = float(x) * iw / sw
        my = float(y) * ih / sh
    if is_identity_crop(iw, ih, mx, my, zoom):
        return image
    left, top, nw, nh = paste_top_left(iw, ih, mx, my, zoom)
    src = image.convert("RGBA")
    if src.size != (nw, nh):
        src = src.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGBA", (iw, ih), (0, 0, 0, 0))
    canvas.paste(src, (left, top), src)
    return canvas


def crop_array(arr, x: float, y: float, zoom: float):
    """Tight in-image slice (Tessellate wrap search). Not the Position & Zoom frame."""
    h, w = int(arr.shape[0]), int(arr.shape[1])
    cw, ch = window_size(w, h, zoom)
    max_x, max_y = max_origin(w, h, zoom)
    left = int(max(0, min(int(round(float(x))), max_x)))
    top = int(max(0, min(int(round(float(y))), max_y)))
    right, bottom = left + cw, top + ch
    if (left, top, right, bottom) == (0, 0, w, h):
        return arr
    return arr[top:bottom, left:right]


def apply_crop_array(arr, x: float, y: float, zoom: float, *, fill=0):
    """Same center-frame placement as ``apply_crop``, for label / alpha arrays."""
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if is_identity_crop(w, h, x, y, zoom):
        return arr
    left, top, nw, nh = paste_top_left(w, h, x, y, zoom)
    src = np.asarray(arr)
    if (nh, nw) != (h, w):
        im = Image.fromarray(src)
        im = im.resize((nw, nh), Image.Resampling.NEAREST)
        src = np.asarray(im)
    if src.ndim == 2:
        out = np.full((h, w), fill, dtype=src.dtype)
    else:
        out = np.full((h, w) + src.shape[2:], fill, dtype=src.dtype)
    x0 = max(0, left)
    y0 = max(0, top)
    x1 = min(w, left + nw)
    y1 = min(h, top + nh)
    if x1 <= x0 or y1 <= y0:
        return out
    sx0 = x0 - left
    sy0 = y0 - top
    out[y0:y1, x0:x1] = src[sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)]
    return out
