# -*- coding: utf-8 -*-
"""
wallpaper_recolor.color.color_math
----------------------------
GUI-free HSL / Lab helpers shared by the range remap and the color wheel.

Vectorized HLS → RGB and sRGB → CIE Lab are used on print-size arrays
(preview is downscaled; save can be ~207MP), so this stays numpy-only —
no tkinter, no per-pixel Python.

Class references (code + name only):
- CAP3321C Data Wrangling
- CAP4631C Machine Learning
"""

from __future__ import annotations  # tuple[int, int, int] hints on 3.9-style runtimes

import colorsys

import numpy as np


def rgb_to_hsl(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """Return H, S, L in 0–1 from 8-bit RGB (colorsys uses HLS order)."""
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h, s, l


def hsl_to_rgb(h: float, s: float, l: float) -> tuple[int, int, int]:
    """8-bit RGB from H, S, L in 0–1."""
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def hsl_lightness(rgb: np.ndarray) -> np.ndarray:
    """HSL L in 0–1: (max(R,G,B) + min(R,G,B)) / 2 per pixel.

    This is the lightness channel a Color-blend keeps — grain and weave
    live here, not in hue.
    """
    rgb_f = rgb.astype(np.float32) * (1.0 / 255.0)
    maxc = rgb_f.max(axis=-1)
    minc = rgb_f.min(axis=-1)
    return (maxc + minc) * 0.5


def hls_array_to_rgb(h: np.ndarray, l: np.ndarray, s: np.ndarray) -> np.ndarray:
    """colorsys.hls_to_rgb over numpy arrays → N×3 uint8.

    One-pixel Python loop is too slow on wallpaper; use the HLS formula
    vectorized (https://en.wikipedia.org/wiki/HSL_and_HSV#HSL_to_RGB).
    """
    h = np.asarray(h, dtype=np.float32)
    l = np.clip(np.asarray(l, dtype=np.float32), 0.0, 1.0)
    s = np.clip(np.asarray(s, dtype=np.float32), 0.0, 1.0)
    c = (1.0 - np.abs(2.0 * l - 1.0)) * s
    hp = (h % 1.0) * 6.0
    x = c * (1.0 - np.abs((hp % 2.0) - 1.0))
    m = l - c / 2.0
    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)
    sextant = np.floor(hp).astype(np.int32) % 6

    def _take(src, mask: np.ndarray) -> np.ndarray | float:
        # 0.0 channels are scalars; c/x are per-pixel arrays
        return src if isinstance(src, float) else src[mask]

    for i, (cr, cg, cb) in enumerate(
        (
            (c, x, 0.0),
            (x, c, 0.0),
            (0.0, c, x),
            (0.0, x, c),
            (x, 0.0, c),
            (c, 0.0, x),
        )
    ):
        mask = sextant == i
        r[mask] = _take(cr, mask)
        g[mask] = _take(cg, mask)
        b[mask] = _take(cb, mask)
    out = np.stack((r + m, g + m, b + m), axis=-1)
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


# sRGB D65 → XYZ (IEC 61966-2-1). Rows are X, Y, Z.
_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)
# D65 white point for CIE Lab
_XN, _YN, _ZN = 0.95047, 1.0, 1.08883
_LAB_DELTA = 6.0 / 29.0
_LAB_DELTA3 = _LAB_DELTA ** 3
_LAB_LIN_SCALE = 1.0 / (3.0 * _LAB_DELTA * _LAB_DELTA)


def rgb_to_lab_array(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 (…×3) → CIE Lab float32 (D65). Vectorized — no Python pixels.

    Lab is the space for color-closeness k-means / 1-NN (CAP4631C): Euclidean
    distance there tracks perceived color better than RGB.
    """
    srgb = np.clip(rgb.astype(np.float32) * (1.0 / 255.0), 0.0, 1.0)
    linear = np.where(
        srgb <= 0.04045,
        srgb * (1.0 / 12.92),
        ((srgb + 0.055) * (1.0 / 1.055)) ** 2.4,
    )
    xyz = linear @ _SRGB_TO_XYZ.T
    xyz[..., 0] /= _XN
    xyz[..., 1] /= _YN
    xyz[..., 2] /= _ZN
    xyz = np.clip(xyz, 0.0, None)
    f = np.where(xyz > _LAB_DELTA3, np.cbrt(xyz), xyz * _LAB_LIN_SCALE + 4.0 / 29.0)
    lab = np.empty(rgb.shape, dtype=np.float32)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


def rgb_tuple_to_lab(rgb: tuple[int, int, int]) -> np.ndarray:
    """Lab (3,) float32 for one 8-bit sRGB swatch (palette targets / V6-N)."""
    arr = np.array(((rgb[0], rgb[1], rgb[2]),), dtype=np.uint8).reshape(1, 1, 3)
    return rgb_to_lab_array(arr)[0, 0]


def lab_tuple_to_rgb(lab: np.ndarray) -> tuple[int, int, int]:
    """8-bit sRGB for one Lab center (k-means / match-from swatch)."""
    arr = np.asarray(lab, dtype=np.float32).reshape(1, 1, 3)
    rgb = lab_to_rgb_array(arr)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


# Inverse of _SRGB_TO_XYZ so Lab → RGB uses the same D65 matrix as clustering.
_XYZ_TO_SRGB = np.linalg.inv(_SRGB_TO_XYZ).astype(np.float32)


def lab_to_rgb_array(lab: np.ndarray) -> np.ndarray:
    """CIE Lab float32 (D65) → sRGB uint8. Inverse of rgb_to_lab_array.

    Used for Color/Luminosity remap: keep original L*, write replacement a*, b*.
    """
    fy = (lab[..., 0] + 16.0) * (1.0 / 116.0)
    fx = fy + lab[..., 1] * (1.0 / 500.0)
    fz = fy - lab[..., 2] * (1.0 / 200.0)
    f = np.stack((fx, fy, fz), axis=-1)
    xyz = np.where(f > _LAB_DELTA, f ** 3, (f - 4.0 / 29.0) / _LAB_LIN_SCALE)
    xyz[..., 0] *= _XN
    xyz[..., 1] *= _YN
    xyz[..., 2] *= _ZN
    linear = xyz @ _XYZ_TO_SRGB.T
    linear = np.clip(linear, 0.0, None)
    srgb = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    return np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)
