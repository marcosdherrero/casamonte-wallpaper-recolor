# -*- coding: utf-8 -*-
"""
wallpaper_recolor.color.swatch_match
------------------------------------
Fit Color & lighting Temperature / Tint / Exposure so a photographed
Pantone chip matches the official table RGB.

Same von Kries + EV model as Gray World / White Patch (tone.py): channel
gains from sampled RGB → target RGB become Temperature / Tint after
mean-normalizing; leftover luma is Exposure (one stop at ±1).

No tkinter — numpy only.

Class references (code + name only):
- CAP3321C Data Wrangling
"""

from __future__ import annotations

import math

import numpy as np

from wallpaper_recolor.color.color_ranges import LUMA_B, LUMA_G, LUMA_R
from wallpaper_recolor.color.pantone import lookup_pantone_rgb, pantone_code_for_rgb
from wallpaper_recolor.color.tone import (
    EXPOSURE_EV,
    illuminant_to_temp_tint,
    temperature_tint_gains,
)

SWATCH_SAMPLE_RADIUS = 4
_GAIN_FLOOR = 1.0


def mean_rgb_region(
    rgb: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> tuple[int, int, int]:
    """Mean 8-bit RGB in an inclusive pixel box (clipped to the image)."""
    pix = np.asarray(rgb[..., :3])
    h, w = pix.shape[:2]
    if h < 1 or w < 1:
        return (0, 0, 0)
    xa = max(0, min(w, min(int(x0), int(x1))))
    xb = max(0, min(w, max(int(x0), int(x1)) + 1))
    ya = max(0, min(h, min(int(y0), int(y1))))
    yb = max(0, min(h, max(int(y0), int(y1)) + 1))
    if xb <= xa or yb <= ya:
        return (0, 0, 0)
    mean = pix[ya:yb, xa:xb].reshape(-1, 3).mean(axis=0)
    return (
        int(round(float(mean[0]))),
        int(round(float(mean[1]))),
        int(round(float(mean[2]))),
    )


def mean_rgb_at(
    rgb: np.ndarray,
    x: int,
    y: int,
    radius: int = SWATCH_SAMPLE_RADIUS,
) -> tuple[int, int, int]:
    """Mean RGB in a square neighborhood around ``(x, y)``."""
    r = max(0, int(radius))
    return mean_rgb_region(rgb, int(x) - r, int(y) - r, int(x) + r, int(y) + r)


def swatch_match_gains(
    sampled_rgb: tuple[int, int, int] | np.ndarray,
    target_rgb: tuple[int, int, int] | np.ndarray,
) -> np.ndarray:
    """Per-channel multiply that maps ``sampled_rgb`` onto ``target_rgb``."""
    sampled = np.asarray(sampled_rgb, dtype=np.float64).reshape(-1)[:3]
    target = np.asarray(target_rgb, dtype=np.float64).reshape(-1)[:3]
    return (target / np.maximum(sampled, _GAIN_FLOOR)).astype(np.float32)


def swatch_match_temp_tint_exposure(
    sampled_rgb: tuple[int, int, int] | np.ndarray,
    target_rgb: tuple[int, int, int] | np.ndarray,
) -> tuple[float, float, float]:
    """Temperature / Tint / Exposure (−1…+1) so ``apply_tone_rgb`` hits the chip.

    Temperature / Tint take the chromaticity of ``target / sampled`` the same
    way White Patch does. Exposure is the Rec. 709 luma ratio after that WB
    (``luma *= 2 ** (exposure * EV)``). Amounts are clipped to the slider range.
    """
    sampled = np.asarray(sampled_rgb, dtype=np.float32).reshape(-1)[:3]
    target = np.asarray(target_rgb, dtype=np.float32).reshape(-1)[:3]
    temp, tint = illuminant_to_temp_tint(swatch_match_gains(sampled, target))
    after_wb = sampled * temperature_tint_gains(temp, tint)
    luma_s = float(LUMA_R * after_wb[0] + LUMA_G * after_wb[1] + LUMA_B * after_wb[2])
    luma_t = float(LUMA_R * target[0] + LUMA_G * target[1] + LUMA_B * target[2])
    if luma_s < 1e-3:
        exposure = 0.0
    else:
        exposure = math.log2(max(luma_t, 1e-3) / max(luma_s, 1e-3)) / float(EXPOSURE_EV)
    return (
        float(max(-1.0, min(1.0, temp))),
        float(max(-1.0, min(1.0, tint))),
        float(max(-1.0, min(1.0, exposure))),
    )


def correction_for_pantone(
    sampled_rgb: tuple[int, int, int] | np.ndarray,
    pantone_code: str,
) -> tuple[float, float, float] | None:
    """``swatch_match_temp_tint_exposure`` for an official Pantone code, or None."""
    target = lookup_pantone_rgb(pantone_code)
    if target is None:
        return None
    return swatch_match_temp_tint_exposure(sampled_rgb, target)


def guess_swatch_pantone(sampled_rgb: tuple[int, int, int]) -> str:
    """Nearest table code to the photographed chip (empty if the table is down)."""
    return pantone_code_for_rgb(
        (int(sampled_rgb[0]), int(sampled_rgb[1]), int(sampled_rgb[2])),
        closest=True,
    ) or ""
