# -*- coding: utf-8 -*-
"""
wallpaper_recolor.color.layers
------------------------
Range-separation layers: non-overlapping masks, an exact N-color master,
and a texture presentation composite.

Exact master = solid fill through each mask (optional production extra).
Texture / grain = Color/Luminosity remap: keep the original image's Lab L*
    (weave / shadows / grain) and write the picked color's a*, b* per range.
    No grayscale Overlay plate — that looked like a colored varnish.

Solid fill is a label→RGB lookup table (one gather, no per-range Python
masks). Grain is the same LUT in Lab plus original L*, numpy-wide, in row
strips on print-size files so we never keep extra 207MP float copies around.

Tone (white balance, exposure, contrast, darks/lights, print CMY balance,
saturation) runs after that remap on both masters so the weave still reads
and save matches the live preview. Range eyes knock hidden pixels out of
Result (alpha 0) after tone — not original RGB.

Class references (code + name only):
- CAP3321C Data Wrangling
- CAP4631C Machine Learning
"""

from __future__ import annotations  # list[np.ndarray] hints

from collections.abc import Sequence

import numpy as np
from PIL import Image

from wallpaper_recolor.color.color_math import lab_to_rgb_array, rgb_to_lab_array
from wallpaper_recolor.color.color_ranges import (
    TEXTURE_DEFAULT_STRENGTH,
    ColorRange,
    ColorRangeMap,
    _image_from_rgb,
    band_masks,
    label_image,
    luma_channel,
    replacement_lab_lut,
    replacement_lut,
)
from wallpaper_recolor.color.tone import apply_tone_from_map

# ~207MP RGB is ~600MB; float32 Lab is ~2.4GB. Strip so Color blend stays
# numpy-wide without holding several full-res temporaries (Windows
# "Not Responding" was the Tk thread, but peak RAM still matters).
_STRIP_ROWS = 128
_FULL_VECTOR_MAX = 4_000_000


def texture_detail_gray(rgb: np.ndarray) -> np.ndarray:
    """8-bit Rec. 709 luma — reference plate (not composited as Overlay)."""
    return np.clip(np.round(luma_channel(rgb)), 0, 255).astype(np.uint8)


def _solid_from_lut(rgb: np.ndarray, labels: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Vectorized exact fill: labels index the replacement LUT."""
    last = lut.shape[0] - 1
    painted = lut[np.clip(labels, 0, last)]
    valid = labels >= 0
    if bool(valid.all()):
        return painted
    out = rgb.copy()
    out[valid] = painted[valid]
    return out


def exact_rgb(
    rgb: np.ndarray,
    labels: np.ndarray,
    ranges: Sequence[ColorRange],
) -> np.ndarray:
    """Solid replacement_rgb per band. Transparent labels stay source RGB."""
    lut = replacement_lut(ranges)
    h, w = labels.shape
    if h * w <= _FULL_VECTOR_MAX:
        return _solid_from_lut(rgb, labels, lut)
    out = np.empty_like(rgb)
    for y0 in range(0, h, _STRIP_ROWS):
        y1 = min(h, y0 + _STRIP_ROWS)
        out[y0:y1] = _solid_from_lut(rgb[y0:y1], labels[y0:y1], lut)
    return out


def _color_luminosity_block(
    rgb: np.ndarray,
    labels: np.ndarray,
    lab_lut: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Keep original L* mixed toward the hex L*; a*, b* come from the hex.

    strength 1 = original luminosity (grain) + replacement chroma.
    strength 0 is not used here — callers take the exact RGB LUT instead
    so a Lab round-trip cannot shift a flat fill.
    """
    mix = max(0.0, min(1.0, float(strength)))
    lab = rgb_to_lab_array(rgb)
    last = lab_lut.shape[0] - 1
    idx = np.clip(labels, 0, last)
    out_lab = np.empty_like(lab)
    L_rep = lab_lut[idx, 0]
    out_lab[..., 0] = L_rep * (1.0 - mix) + lab[..., 0] * mix
    out_lab[..., 1] = lab_lut[idx, 1]
    out_lab[..., 2] = lab_lut[idx, 2]
    textured = lab_to_rgb_array(out_lab)
    valid = labels >= 0
    if bool(valid.all()):
        return textured
    out = rgb.copy()
    out[valid] = textured[valid]
    return out


def presentation_rgb(
    rgb: np.ndarray,
    labels: np.ndarray,
    ranges: Sequence[ColorRange],
    strength: float = TEXTURE_DEFAULT_STRENGTH,
) -> np.ndarray:
    """Color/Luminosity grain — one vectorized pass (strips if huge).

    strength 0 matches exact_rgb; 1 keeps original L* with the picked a*, b*.
    """
    mix = max(0.0, min(1.0, float(strength)))
    if mix <= 0.0:
        return exact_rgb(rgb, labels, ranges)
    lab_lut = replacement_lab_lut(ranges)
    h, w = labels.shape
    if h * w <= _FULL_VECTOR_MAX:
        return _color_luminosity_block(rgb, labels, lab_lut, mix)
    out = np.empty_like(rgb)
    for y0 in range(0, h, _STRIP_ROWS):
        y1 = min(h, y0 + _STRIP_ROWS)
        out[y0:y1] = _color_luminosity_block(
            rgb[y0:y1], labels[y0:y1], lab_lut, mix
        )
    return out


def effective_texture_strength(range_map: ColorRangeMap) -> float:
    """Slider mix, or 0 when the texture eye is off (flat fills)."""
    if not range_map.texture_enabled:
        return 0.0
    return max(0.0, min(1.0, float(range_map.texture_strength)))


def hidden_range_mask(labels: np.ndarray, ranges: Sequence[ColorRange]) -> np.ndarray:
    hidden = [band.index for band in ranges if not band.visible]
    if not hidden:
        return np.zeros(labels.shape, dtype=bool)
    return np.isin(labels, np.asarray(hidden, dtype=labels.dtype))


def knockout_alpha(
    labels: np.ndarray,
    ranges: Sequence[ColorRange],
    alpha: np.ndarray | None,
) -> np.ndarray | None:
    """Zero alpha on eye-off ranges. Opaque RGB stays RGB when nothing is hidden."""
    mask = hidden_range_mask(labels, ranges)
    if not bool(mask.any()):
        return None if alpha is None else np.asarray(alpha, dtype=np.uint8)
    if alpha is None:
        out = np.full(labels.shape, 255, dtype=np.uint8)
    else:
        out = np.asarray(alpha, dtype=np.uint8).copy()
    out[mask] = 0
    return out


def _pack_composite(
    rgb: np.ndarray,
    labels: np.ndarray,
    ranges: Sequence[ColorRange],
    alpha: np.ndarray | None,
) -> Image.Image:
    """RGB plus knockout / source alpha (RGBA when any hole exists)."""
    holes = hidden_range_mask(labels, ranges)
    a = knockout_alpha(labels, ranges, alpha)
    if bool(holes.any()):
        rgb = np.array(rgb, copy=True, dtype=np.uint8)
        rgb[holes] = 0
    return _image_from_rgb(rgb, a)


def _finish_composite(painted: np.ndarray, range_map: ColorRangeMap) -> np.ndarray:
    """Tone after remap. Knockout is alpha, applied when packing."""
    return _tone_after_remap(painted, range_map)


def _tone_after_remap(rgb: np.ndarray, range_map: ColorRangeMap) -> np.ndarray:
    """Color & lighting after remap (identity if all tone fields are 0)."""
    return apply_tone_from_map(rgb, range_map)


def live_composite_from_map(range_map: ColorRangeMap) -> Image.Image:
    """Preview/grain result: texture, tone, then knockout of hidden ranges."""
    if range_map.rgb is None or range_map.labels is None:
        raise ValueError("ColorRangeMap has no pixel data")
    painted = presentation_rgb(
        range_map.rgb,
        range_map.labels,
        range_map.ranges,
        strength=effective_texture_strength(range_map),
    )
    out = _finish_composite(painted, range_map)
    return _pack_composite(out, range_map.labels, range_map.ranges, range_map.alpha)


def composites_from_map(range_map: ColorRangeMap) -> tuple[Image.Image, Image.Image]:
    """(exact, texture presentation) for the working copy already on the map."""
    if range_map.rgb is None or range_map.labels is None:
        raise ValueError("ColorRangeMap has no pixel data")
    exact = _finish_composite(
        exact_rgb(range_map.rgb, range_map.labels, range_map.ranges),
        range_map,
    )
    tex = _finish_composite(
        presentation_rgb(
            range_map.rgb,
            range_map.labels,
            range_map.ranges,
            strength=effective_texture_strength(range_map),
        ),
        range_map,
    )
    return (
        _pack_composite(exact, range_map.labels, range_map.ranges, range_map.alpha),
        _pack_composite(tex, range_map.labels, range_map.ranges, range_map.alpha),
    )


def labeled_composite_for_image(
    image: Image.Image,
    range_map: ColorRangeMap,
    *,
    grain: bool,
) -> tuple[Image.Image, np.ndarray, np.ndarray | None, np.ndarray]:
    """One full-res master plus (rgb, alpha, labels) — no unused twin remap."""
    rgb, alpha, labels = label_image(image, range_map)
    if grain:
        out = presentation_rgb(
            rgb, labels, range_map.ranges, strength=effective_texture_strength(range_map)
        )
    else:
        out = exact_rgb(rgb, labels, range_map.ranges)
    master = _pack_composite(
        _finish_composite(out, range_map), labels, range_map.ranges, alpha
    )
    return master, rgb, alpha, labels


def composite_for_image(
    image: Image.Image,
    range_map: ColorRangeMap,
    *,
    grain: bool,
) -> Image.Image:
    """Full-res exact or grain composite (save path — one output, not both)."""
    master, _rgb, _alpha, _labels = labeled_composite_for_image(
        image, range_map, grain=grain
    )
    return master


def composites_for_image(
    image: Image.Image,
    range_map: ColorRangeMap,
) -> tuple[Image.Image, Image.Image, np.ndarray, np.ndarray | None, np.ndarray]:
    """Full-res exact + presentation, plus (rgb, alpha, labels) for mask export."""
    rgb, alpha, labels = label_image(image, range_map)
    exact = _finish_composite(exact_rgb(rgb, labels, range_map.ranges), range_map)
    tex = _finish_composite(
        presentation_rgb(
            rgb, labels, range_map.ranges, strength=effective_texture_strength(range_map)
        ),
        range_map,
    )
    return (
        _pack_composite(exact, labels, range_map.ranges, alpha),
        _pack_composite(tex, labels, range_map.ranges, alpha),
        rgb,
        alpha,
        labels,
    )


def masks_for_image(labels: np.ndarray, range_map: ColorRangeMap) -> list[np.ndarray]:
    """One 8-bit mask per range."""
    return band_masks(labels, len(range_map.ranges))
