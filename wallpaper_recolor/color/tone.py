# -*- coding: utf-8 -*-
"""
wallpaper_recolor.color.tone
----------------------
After-remap color & lighting: white balance, exposure, contrast, darks/lights,
print CMY↔RGB color balance, and saturation.

Applied **after** exact or grain recolor (not before labeling) so range
assignment stays on the original wallpaper and weave still reads.

Photographer/print order (ON1 color-correction edit, PyImageSearch WB,
ML_ColorCorrection_tool gray-world / simple WB, EUDL color-cast / CCM):
1. Temperature / Tint (global von Kries) plus Lights RGB in highlights
2. Exposure / brightness / contrast / darks / lights (luma curve)
3. Color balance as Cyan↔Red, Magenta↔Green, Yellow↔Blue (subtractive ink)
4. Saturation (luma-preserving chroma)

Gray-world and white-patch estimate gains and **set** Temperature / Tint
(operator can still nudge). Normalize lighting still sets Darks / Lights.

Amounts −1…+1 (UI −100…+100). All zeros is identity.

No tkinter — numpy only (same rule as color_math / layers).

Class references (code + name only):
- CAP3321C Data Wrangling
"""

from __future__ import annotations

import numpy as np

from wallpaper_recolor.color.color_ranges import LUMA_B, LUMA_G, LUMA_R

# UI Scale from_/to — 0 is neutral (no change)
TONE_SLIDER_MIN = -100.0
TONE_SLIDER_MAX = 100.0
TONE_NEUTRAL = 0.0

# Rec. 709 luma 0–1: shadows below this, highlights above
SHADOW_PIVOT = 0.40
HIGHLIGHT_PIVOT = 0.60
# Brightness: additive luma offset only (±1 is a clear lift). Exposure owns
# the multiplicative EV gain so the two sliders are not the same global lift.
BRIGHTNESS_ADD = 0.35
# Exposure: luma *= 2 ** (amount * EV). ±1 slider → ±1 stop.
EXPOSURE_EV = 1.0
# Contrast: expand/compress around Rec. 709 mid-gray. ±1 → 2× / flat.
CONTRAST_PIVOT = 0.5
# Lights RGB: von Kries / illuminant scale in highlights. ±1 → ×1.5 / ×0.5
# on fully-weighted lights (White Patch region). 0 is identity.
LIGHTS_WB_SPAN = 0.5
# CMY ink: same span on the complementary RGB channel. +Cyan pulls R
# (adds cyan ink); −Cyan boosts R. Same for Magenta↔G and Yellow↔B.
CMY_INK_SPAN = LIGHTS_WB_SPAN
# Saturation: luma-preserving chroma scale. ±1 → 0× (gray) / 2× chroma.
SATURATION_SPAN = 1.0
# White Patch: skip clipped 255s the way the article uses a percentile < 100
WHITE_PATCH_PERCENTILE = 99.0

_STRIP_ROWS = 128
_FULL_VECTOR_MAX = 4_000_000
_NEUTRAL_EPS = 1e-12


def is_neutral_tone(
    darks: float,
    lights: float,
    brightness: float,
    lights_reds: float = TONE_NEUTRAL,
    lights_greens: float = TONE_NEUTRAL,
    lights_blues: float = TONE_NEUTRAL,
    contrast: float = TONE_NEUTRAL,
    exposure: float = TONE_NEUTRAL,
    lights_cyan: float = TONE_NEUTRAL,
    lights_magenta: float = TONE_NEUTRAL,
    lights_yellow: float = TONE_NEUTRAL,
    darks_cyan: float = TONE_NEUTRAL,
    darks_magenta: float = TONE_NEUTRAL,
    darks_yellow: float = TONE_NEUTRAL,
    temperature: float = TONE_NEUTRAL,
    tint: float = TONE_NEUTRAL,
    saturation: float = TONE_NEUTRAL,
    balance_cyan: float = TONE_NEUTRAL,
    balance_magenta: float = TONE_NEUTRAL,
    balance_yellow: float = TONE_NEUTRAL,
) -> bool:
    """True when the curve is a no-op (slider midpoints)."""
    return (
        abs(float(darks)) < _NEUTRAL_EPS
        and abs(float(lights)) < _NEUTRAL_EPS
        and abs(float(brightness)) < _NEUTRAL_EPS
        and abs(float(lights_reds)) < _NEUTRAL_EPS
        and abs(float(lights_greens)) < _NEUTRAL_EPS
        and abs(float(lights_blues)) < _NEUTRAL_EPS
        and abs(float(contrast)) < _NEUTRAL_EPS
        and abs(float(exposure)) < _NEUTRAL_EPS
        and abs(float(lights_cyan)) < _NEUTRAL_EPS
        and abs(float(lights_magenta)) < _NEUTRAL_EPS
        and abs(float(lights_yellow)) < _NEUTRAL_EPS
        and abs(float(darks_cyan)) < _NEUTRAL_EPS
        and abs(float(darks_magenta)) < _NEUTRAL_EPS
        and abs(float(darks_yellow)) < _NEUTRAL_EPS
        and abs(float(temperature)) < _NEUTRAL_EPS
        and abs(float(tint)) < _NEUTRAL_EPS
        and abs(float(saturation)) < _NEUTRAL_EPS
        and abs(float(balance_cyan)) < _NEUTRAL_EPS
        and abs(float(balance_magenta)) < _NEUTRAL_EPS
        and abs(float(balance_yellow)) < _NEUTRAL_EPS
    )


def slider_to_amount(pct: float) -> float:
    """Map a −100…+100 slider onto −1…+1 for apply_tone_rgb."""
    return max(-1.0, min(1.0, float(pct) / 100.0))


def highlight_weight(luma_n: np.ndarray) -> np.ndarray:
    """Lights mask: Rec. 709 luma 0–1, same ramp as the Lights slider."""
    hi_span = max(1e-6, 1.0 - HIGHLIGHT_PIVOT)
    return np.clip((luma_n - HIGHLIGHT_PIVOT) / hi_span, 0.0, 1.0)


def shadow_weight(luma_n: np.ndarray) -> np.ndarray:
    """Darks mask: Rec. 709 luma 0–1, same ramp as the Darks slider."""
    return np.clip((SHADOW_PIVOT - luma_n) / SHADOW_PIVOT, 0.0, 1.0)


def illuminant_gains(
    lights_reds: float,
    lights_greens: float,
    lights_blues: float,
) -> np.ndarray:
    """Von Kries diagonal from Lights RGB sliders (−1…+1). Neutral is ones.

    +Reds boosts red (warmer highlights); −Reds pulls red (Gray World–style
    channel balance). Gains are 1 ± LIGHTS_WB_SPAN at the slider ends.
    """
    return np.array(
        [
            1.0 + float(lights_reds) * LIGHTS_WB_SPAN,
            1.0 + float(lights_greens) * LIGHTS_WB_SPAN,
            1.0 + float(lights_blues) * LIGHTS_WB_SPAN,
        ],
        dtype=np.float32,
    )


def cmy_gains(
    cyan: float,
    magenta: float,
    yellow: float,
) -> np.ndarray:
    """Subtractive ink diagonal: +C pulls R, +M pulls G, +Y pulls B.

    Complementary to Lights RGB (which *adds* that channel). ±1 → ×0.5 / ×1.5
    on the complementary RGB channel; 0 is identity.
    """
    return np.array(
        [
            1.0 - float(cyan) * CMY_INK_SPAN,
            1.0 - float(magenta) * CMY_INK_SPAN,
            1.0 - float(yellow) * CMY_INK_SPAN,
        ],
        dtype=np.float32,
    )


def temperature_tint_gains(temperature: float, tint: float) -> np.ndarray:
    """Global von Kries from Temperature / Tint (−1…+1). Neutral is ones.

    +Temperature warms (boost R, pull B). +Tint is magenta (pull G).
    """
    return np.array(
        [
            1.0 + float(temperature) * LIGHTS_WB_SPAN,
            1.0 - float(tint) * LIGHTS_WB_SPAN,
            1.0 - float(temperature) * LIGHTS_WB_SPAN,
        ],
        dtype=np.float32,
    )


def illuminant_to_temp_tint(gains: np.ndarray) -> tuple[float, float]:
    """Map a 3-channel gain vector onto Temperature / Tint (−1…+1).

    Mean-normalizes so overall exposure stays on the Exposure control.
    """
    g = np.asarray(gains, dtype=np.float64).reshape(-1)[:3]
    g = g / max(float(np.mean(g)), 1e-6)
    temp = float((g[0] - g[2]) / (2.0 * LIGHTS_WB_SPAN))
    tint = float((1.0 - g[1]) / LIGHTS_WB_SPAN)
    return (
        max(-1.0, min(1.0, temp)),
        max(-1.0, min(1.0, tint)),
    )


def _mix_channel_gains(out: np.ndarray, mix: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """``out * (1 + mix * (gains - 1))`` — numpy-wide, print-size safe."""
    w = mix.astype(np.float32)
    return out * (1.0 + w[..., np.newaxis] * (gains - 1.0))


def _channel_means(rgb: np.ndarray, weight: np.ndarray | None) -> np.ndarray:
    """Per-channel mean, optionally weighted (highlight mask). rgb is H×W×3."""
    pix = np.asarray(rgb[..., :3], dtype=np.float64)
    if weight is None:
        return pix.mean(axis=(0, 1))
    w = np.asarray(weight, dtype=np.float64)
    wsum = float(w.sum())
    if wsum < 1e-12:
        return pix.mean(axis=(0, 1))
    return (pix * w[..., np.newaxis]).sum(axis=(0, 1)) / wsum


def gray_world_gains(
    rgb: np.ndarray,
    weight: np.ndarray | None = None,
) -> np.ndarray:
    """Gray World: scale so each channel’s (weighted) mean equals the mean of means.

    ``gain_c = gray / mean_c`` — a red cast (high mean R) gets gain_r < 1.
    Optional ``weight`` limits the estimate to Lights (or any mask).
    """
    means = _channel_means(rgb, weight)
    gray = float(np.mean(means))
    return (gray / np.maximum(means, 1e-6)).astype(np.float32)


def white_patch_gains(
    rgb: np.ndarray,
    weight: np.ndarray | None = None,
    percentile: float = WHITE_PATCH_PERCENTILE,
) -> np.ndarray:
    """White Patch / max-RGB: scale so each channel’s high percentile meets the max of those.

    Article form: ``image / percentile(image, p, axis=(0, 1))``. Using the max
    of the three percentiles as the shared white point keeps overall exposure
    closer to identity than stretching every channel to 255. ``weight`` > 0
    pixels only (Lights region); no weight → whole image.
    """
    pix = np.asarray(rgb[..., :3], dtype=np.float64)
    if weight is not None:
        sel = np.asarray(weight) > 1e-6
        if np.any(sel):
            pix = pix[sel]
        else:
            pix = pix.reshape(-1, 3)
    else:
        pix = pix.reshape(-1, 3)
    peaks = np.percentile(pix, float(percentile), axis=0)
    white = float(np.max(peaks))
    return (white / np.maximum(peaks, 1e-6)).astype(np.float32)


def apply_tone_rgb(
    rgb: np.ndarray,
    darks: float = TONE_NEUTRAL,
    lights: float = TONE_NEUTRAL,
    brightness: float = TONE_NEUTRAL,
    lights_reds: float = TONE_NEUTRAL,
    lights_greens: float = TONE_NEUTRAL,
    lights_blues: float = TONE_NEUTRAL,
    contrast: float = TONE_NEUTRAL,
    exposure: float = TONE_NEUTRAL,
    lights_cyan: float = TONE_NEUTRAL,
    lights_magenta: float = TONE_NEUTRAL,
    lights_yellow: float = TONE_NEUTRAL,
    darks_cyan: float = TONE_NEUTRAL,
    darks_magenta: float = TONE_NEUTRAL,
    darks_yellow: float = TONE_NEUTRAL,
    temperature: float = TONE_NEUTRAL,
    tint: float = TONE_NEUTRAL,
    saturation: float = TONE_NEUTRAL,
    balance_cyan: float = TONE_NEUTRAL,
    balance_magenta: float = TONE_NEUTRAL,
    balance_yellow: float = TONE_NEUTRAL,
) -> np.ndarray:
    """Luma curve, white balance, print color balance, then saturation.

    Amounts in −1…+1. 0 is identity.
    """
    if is_neutral_tone(
        darks,
        lights,
        brightness,
        lights_reds,
        lights_greens,
        lights_blues,
        contrast,
        exposure,
        lights_cyan,
        lights_magenta,
        lights_yellow,
        darks_cyan,
        darks_magenta,
        darks_yellow,
        temperature,
        tint,
        saturation,
        balance_cyan,
        balance_magenta,
        balance_yellow,
    ):
        return rgb
    h, w = rgb.shape[:2]
    args = (
        float(darks),
        float(lights),
        float(brightness),
        float(lights_reds),
        float(lights_greens),
        float(lights_blues),
        float(contrast),
        float(exposure),
        float(lights_cyan),
        float(lights_magenta),
        float(lights_yellow),
        float(darks_cyan),
        float(darks_magenta),
        float(darks_yellow),
        float(temperature),
        float(tint),
        float(saturation),
        float(balance_cyan),
        float(balance_magenta),
        float(balance_yellow),
    )
    if h * w <= _FULL_VECTOR_MAX:
        return _apply_tone_block(rgb, *args)
    out = np.empty_like(rgb)
    for y0 in range(0, h, _STRIP_ROWS):
        y1 = min(h, y0 + _STRIP_ROWS)
        out[y0:y1] = _apply_tone_block(rgb[y0:y1], *args)
    return out


def apply_tone_from_map(rgb: np.ndarray, range_map) -> np.ndarray:
    """apply_tone_rgb from ColorRangeMap tone_* fields (preview/save lockstep)."""
    bal_c = float(getattr(range_map, "tone_balance_cyan", 0.0))
    bal_m = float(getattr(range_map, "tone_balance_magenta", 0.0))
    bal_y = float(getattr(range_map, "tone_balance_yellow", 0.0))
    lights_c = 0.0
    lights_m = 0.0
    lights_y = 0.0
    if abs(bal_c) + abs(bal_m) + abs(bal_y) < _NEUTRAL_EPS:
        # Pre-rebuild wpedit: Lights CMY was highlight-only, not global pairs.
        lights_c = float(getattr(range_map, "tone_lights_cyan", 0.0))
        lights_m = float(getattr(range_map, "tone_lights_magenta", 0.0))
        lights_y = float(getattr(range_map, "tone_lights_yellow", 0.0))
    return apply_tone_rgb(
        rgb,
        range_map.tone_darks,
        range_map.tone_lights,
        range_map.tone_brightness,
        range_map.tone_lights_reds,
        range_map.tone_lights_greens,
        range_map.tone_lights_blues,
        float(getattr(range_map, "tone_contrast", 0.0)),
        float(getattr(range_map, "tone_exposure", 0.0)),
        lights_c,
        lights_m,
        lights_y,
        float(getattr(range_map, "tone_darks_cyan", 0.0)),
        float(getattr(range_map, "tone_darks_magenta", 0.0)),
        float(getattr(range_map, "tone_darks_yellow", 0.0)),
        float(getattr(range_map, "tone_temperature", 0.0)),
        float(getattr(range_map, "tone_tint", 0.0)),
        float(getattr(range_map, "tone_saturation", 0.0)),
        bal_c,
        bal_m,
        bal_y,
    )


def _apply_tone_block(
    rgb: np.ndarray,
    darks: float,
    lights: float,
    brightness: float,
    lights_reds: float,
    lights_greens: float,
    lights_blues: float,
    contrast: float,
    exposure: float,
    lights_cyan: float,
    lights_magenta: float,
    lights_yellow: float,
    darks_cyan: float,
    darks_magenta: float,
    darks_yellow: float,
    temperature: float,
    tint: float,
    saturation: float,
    balance_cyan: float,
    balance_magenta: float,
    balance_yellow: float,
) -> np.ndarray:
    """One numpy pass. Order matches Color & lighting UI (preview/save lockstep).

    1. Temperature / Tint (global von Kries)
    2. Darks / Lights / Brightness / Exposure / Contrast (luma, original weights)
    3. Lights RGB in highlight_w
    4. Global Cyan↔Red / Magenta↔Green / Yellow↔Blue, then optional region CMY
    5. Saturation (luma-preserving chroma)
    """
    rgb_f = rgb.astype(np.float32)
    luma = LUMA_R * rgb_f[..., 0] + LUMA_G * rgb_f[..., 1] + LUMA_B * rgb_f[..., 2]
    luma_n = luma * (1.0 / 255.0)
    shadow_w = np.clip((SHADOW_PIVOT - luma_n) / SHADOW_PIVOT, 0.0, 1.0)
    highlight_w = highlight_weight(luma_n)

    out = rgb_f
    if abs(temperature) + abs(tint) >= _NEUTRAL_EPS:
        out = out * temperature_tint_gains(temperature, tint)

    luma_src = LUMA_R * out[..., 0] + LUMA_G * out[..., 1] + LUMA_B * out[..., 2]
    luma_n_src = luma_src * (1.0 / 255.0)
    luma_out = luma_n_src
    shadow_mix = shadow_w * abs(darks)
    shadow_target = 0.0 if darks >= 0.0 else 1.0
    luma_out = luma_out * (1.0 - shadow_mix) + shadow_target * shadow_mix
    highlight_mix = highlight_w * abs(lights)
    highlight_target = 1.0 if lights >= 0.0 else 0.0
    luma_out = luma_out * (1.0 - highlight_mix) + highlight_target * highlight_mix
    luma_out = luma_out + brightness * BRIGHTNESS_ADD
    if abs(exposure) >= _NEUTRAL_EPS:
        luma_out = luma_out * np.float32(2.0 ** (exposure * EXPOSURE_EV))
    if abs(contrast) >= _NEUTRAL_EPS:
        luma_out = (luma_out - CONTRAST_PIVOT) * (1.0 + contrast) + CONTRAST_PIVOT
    luma_out = np.clip(luma_out, 0.0, 1.0)
    new_luma = luma_out * 255.0
    scale = new_luma / np.maximum(luma_src, 1e-3)
    out = out * scale[..., np.newaxis]
    black = luma_src < 1e-3
    if np.any(black):
        g = new_luma[black]
        out[black] = np.stack((g, g, g), axis=-1)

    if not is_neutral_tone(0.0, 0.0, 0.0, lights_reds, lights_greens, lights_blues):
        out = _mix_channel_gains(
            out, highlight_w, illuminant_gains(lights_reds, lights_greens, lights_blues)
        )
    if abs(balance_cyan) + abs(balance_magenta) + abs(balance_yellow) >= _NEUTRAL_EPS:
        ones = np.ones(highlight_w.shape, dtype=np.float32)
        out = _mix_channel_gains(
            out, ones, cmy_gains(balance_cyan, balance_magenta, balance_yellow)
        )
    if abs(lights_cyan) + abs(lights_magenta) + abs(lights_yellow) >= _NEUTRAL_EPS:
        out = _mix_channel_gains(
            out, highlight_w, cmy_gains(lights_cyan, lights_magenta, lights_yellow)
        )
    if abs(darks_cyan) + abs(darks_magenta) + abs(darks_yellow) >= _NEUTRAL_EPS:
        out = _mix_channel_gains(
            out, shadow_w, cmy_gains(darks_cyan, darks_magenta, darks_yellow)
        )
    if abs(saturation) >= _NEUTRAL_EPS:
        luma_now = LUMA_R * out[..., 0] + LUMA_G * out[..., 1] + LUMA_B * out[..., 2]
        chroma = out - luma_now[..., np.newaxis]
        out = luma_now[..., np.newaxis] + chroma * np.float32(
            1.0 + saturation * SATURATION_SPAN
        )
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def estimate_white_patch_temp_tint(rgb: np.ndarray) -> tuple[float, float]:
    """White Patch / max-RGB in Lights → Temperature / Tint (−1…+1)."""
    pix = np.asarray(rgb[..., :3], dtype=np.float32)
    luma_n = (LUMA_R * pix[..., 0] + LUMA_G * pix[..., 1] + LUMA_B * pix[..., 2]) * (1.0 / 255.0)
    return illuminant_to_temp_tint(white_patch_gains(pix, weight=highlight_weight(luma_n)))


def estimate_gray_world_temp_tint(rgb: np.ndarray) -> tuple[float, float]:
    """Gray World illuminant → Temperature / Tint (−1…+1)."""
    return illuminant_to_temp_tint(gray_world_gains(rgb))


def estimate_tone_amounts(
    src_rgb: np.ndarray,
    dst_rgb: np.ndarray,
) -> tuple[float, float]:
    """Fit Darks / Lights (−1…+1) so Tone's luma mix approximates ``src`` → ``dst``.

    Same Rec. 709 luma and SHADOW_PIVOT / HIGHLIGHT_PIVOT as ``apply_tone_rgb``.
    Darks uses the inverted mix sign (+ = crush / richer shadows, − = lift)
    so a flatten that lifts shadows writes a negative Darks amount — the
    slider still matches the visual. Lights stays + = lift, − = crush.
    Brightness, Exposure, Contrast, Lights RGB, and Lights/Darks CMY are not
    fitted (a mean-preserving flatten has no global lift, contrast, WB, or ink term).
    """
    src = np.asarray(src_rgb[..., :3], dtype=np.float32)
    dst = np.asarray(dst_rgb[..., :3], dtype=np.float32)
    luma_s = (LUMA_R * src[..., 0] + LUMA_G * src[..., 1] + LUMA_B * src[..., 2]) * (1.0 / 255.0)
    luma_d = (LUMA_R * dst[..., 0] + LUMA_G * dst[..., 1] + LUMA_B * dst[..., 2]) * (1.0 / 255.0)
    delta = luma_d - luma_s
    shadow_w = np.clip((SHADOW_PIVOT - luma_s) / SHADOW_PIVOT, 0.0, 1.0)
    highlight_w = highlight_weight(luma_s)
    shadow_mix = _fit_tone_mix(delta, luma_s, shadow_w)
    highlight_mix = _fit_tone_mix(delta, luma_s, highlight_w)
    # _fit_tone_mix is white (+) / black (−). Darks slider is the inverse.
    return (-shadow_mix, highlight_mix)


def _fit_tone_mix(delta: np.ndarray, luma_n: np.ndarray, weight: np.ndarray) -> float:
    """Least-squares mix toward white (+) or black (−) in −1…+1.

    Lights uses this sign as-is. Darks negates it in ``estimate_tone_amounts``.
    """
    w = np.maximum(weight.astype(np.float64, copy=False), 0.0)
    if float(np.mean(w)) < 1e-6:
        return 0.0
    dlt = delta.astype(np.float64, copy=False)
    luma = luma_n.astype(np.float64, copy=False)

    def _proj(basis: np.ndarray) -> float:
        b = basis.ravel()
        y = dlt.ravel()
        n = float(np.dot(b, b))
        if n < 1e-12:
            return 0.0
        return float(np.dot(y, b) / n)

    lift = max(0.0, _proj(w * (1.0 - luma)))
    crush = max(0.0, _proj(-(w * luma)))
    if lift >= crush:
        return float(min(1.0, lift))
    return float(max(-1.0, -crush))
