# -*- coding: utf-8 -*-
"""
wallpaper_recolor.transform.tessellate
----------------------------
Seamless tile for wallpaper print. Four Build modes; all are identity until
Build (except Detail mosaic, which still needs Build).

**Tile (Repeating Design)** (default): estimate the motif period from the
image (1D autocorrelation on a probe), crop to a whole number of repeats,
and fill leftover edge strips by wrapping/copying that motif. Stretching
the core to the frame would break the period so a 3×3 / Offset seam
fails. Then roll the tile so opposite edges meet in the middle and
inpaint that plus-shaped seam with LaMa (OpenCV / period-Hilbert fallback)
using the surrounding motif as context — not just cloning one corner onto
the other. Result left↔right and top↔bottom match. Already-periodic crops
(whole repeats whose cells match) are a no-op. Matching opposite *edges*
alone is not enough — a studio bowl makes both edges dark.

**Tessellation**: Hilbert (crinkly) diffuse from the chosen H/V sides;
opposite side is the model. Mix weights follow a Hilbert front — locality-
preserving quadrant recursion (Hayes), not a linear wipe. If opposite edges
already match (structure + luma), Build is a no-op.

**Mesh**: expanding grid + periodic cut (older Escher-style warp). Build
applies it at full strength.

**Detail mosaic** (Voronoi) places seeds by Sobel density (numpy, not OpenCV),
Lloyd-relaxes them, and fills each cell with the mean color of a small crop
at the seed. Computed on a work-sized grid and nearest-upsampled so print
files never allocate a full-frame mask per tile. Build still writes a wrap
crop and Hilbert-diffuses (Tessellation) unless opposite edges already
tile — mosaic alone is not a seamless repeat.

Horizontal and vertical are independent (Left+Top is a combo). Pipeline:
remap/tone → Crop → optional lighting flatten → tessellate if Built →
output Scale. Normalize lighting divides out a Lab bowl + bilinear ramp
and matches opposite edges so a 3×3 / Offset tile is even. It does not
require Build; Build does not imply flatten. Vectorized numpy (strip
processing) so print-size images stay off the Python pixel loop.

Class references (code + name only):
- CAP3321C Data Wrangling
- CAP4631C Machine Learning
- CAP4633C Machine Learning 2
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from wallpaper_recolor.color.color_math import lab_to_rgb_array, rgb_to_lab_array
from wallpaper_recolor.transform.crop import (
    apply_crop,
    clamp_zoom,
    crop_array,
    max_origin,
    window_size,
)

# Axis choices — independent, so Left+Top is a combo without extra modes
SIDE_OFF = "off"
SIDE_LEFT = "left"
SIDE_RIGHT = "right"
SIDE_TOP = "top"
SIDE_BOTTOM = "bottom"
H_SIDES = (SIDE_OFF, SIDE_LEFT, SIDE_RIGHT)
V_SIDES = (SIDE_OFF, SIDE_TOP, SIDE_BOTTOM)

# Tile = period crop + motif wrap (default). Tessellation = Hilbert diffuse.
# Mesh = geometric warp. Voronoi = detail mosaic (density seeds + Lloyd).
MODE_TILE = "tile"
MODE_TESSELLATE = "tessellate"
MODE_MESH = "mesh"
MODE_VORONOI = "voronoi"
MODE_DEFAULT = MODE_TILE
MODE_LABEL_TILE = "Tile (Repeating Design)"
MODE_LABEL_TESSELLATE = "Tessellation"
MODE_LABEL_MESH = "Mesh"
MODE_LABEL_VORONOI = "Detail mosaic"
MODE_CHOICES = (
    (MODE_TILE, MODE_LABEL_TILE),
    (MODE_TESSELLATE, MODE_LABEL_TESSELLATE),
    (MODE_MESH, MODE_LABEL_MESH),
    (MODE_VORONOI, MODE_LABEL_VORONOI),
)
MODE_LABELS = tuple(label for _key, label in MODE_CHOICES)
MODES = tuple(key for key, _label in MODE_CHOICES)

TILES_MIN = 16
TILES_MAX = 20000
TILES_DEFAULT = 9000
LLOYD_MIN = 0
LLOYD_MAX = 6
LLOYD_DEFAULT = 2
# Raster Voronoi on a work grid, then nearest-upsample (no 207MP per-cell mask)
_VORONOI_WORK_EDGE = 1200
_VORONOI_LLOYD_EDGE = 280
_VORONOI_ASSIGN_EDGE = 480
_VORONOI_ASSIGN_BATCH = 1024

# Mesh warp only (Build applies mesh at full strength)
STRENGTH_MIN = 0.0
STRENGTH_MAX = 1.0
STRENGTH_DEFAULT = 0.0
_STRENGTH_EPS = 1e-6

# How far the interlocking cut swings, as a fraction of width/height at s=1
_WAVE_AMP = 0.12
# Power-map extra exponent at full strength (1 + this) — expanding cells
_STRETCH_K = 1.5
# Strip remap so 207MP does not allocate two full-image float32 maps
_STRIP_ROWS = 128

# Crop search — modest zoom so a window is easier to wrap, not a tiny stamp
_SEARCH_ZOOMS = (1.0, 1.08, 1.16, 1.25)
_SEARCH_ORIGINS = 7
_PROBE_MAX_EDGE = 96
_SHIFT_FRAC = 0.12
_SHIFT_MAX = 40
_ALIGN_BAND = 8
_ALIGN_STRUCT_WEIGHT = 1.0
_ALIGN_COLOR_WEIGHT = 0.15
_BLEND_FRAC = 0.45
_ZOOM_COST = 0.02
# Wrap-identical edges score 0. Cap is ~0 so a tiled synthetic stays a no-op.
_IDENTITY_EDGE_MSE = 0.25
# Autocorr period: first peak above this (and 45% of the strongest peak).
_PERIOD_MIN = 4
_PERIOD_AC_MIN = 0.18
_PERIOD_AC_REL = 0.45
_PERIOD_PROBE = 160

# Illumination flatten — bowl (Gaussian) + linear/bilinear ramp, not weave grain
_ILLUM_WORK_EDGE = 32
_ILLUM_PLANE_EDGE = 48
_ILLUM_BLUR_FRAC = 0.35
_ILLUM_L_EPS = 1.0
_ILLUM_CHROMA_MIX = 0.35


def coerce_built(value: object) -> bool:
    """True when Build is on. Accepts bool or legacy strength 0–1 / 0–100."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, bytes)):
        raw = str(value).strip().lower()
        if raw in ("1", "true", "yes", "on", "built"):
            return True
        if raw in ("0", "false", "no", "off", "", "none"):
            return False
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return False
    try:
        s = float(value)
    except (TypeError, ValueError):
        return bool(value)
    if s != s:  # NaN
        return False
    # Legacy strength slider: 0 = identity, anything else = built
    return s > _STRENGTH_EPS


def clamp_strength(strength: float) -> float:
    """Keep mesh strength in [0, 1]."""
    try:
        s = float(strength)
    except (TypeError, ValueError):
        return STRENGTH_DEFAULT
    if s != s:  # NaN
        return STRENGTH_DEFAULT
    return min(STRENGTH_MAX, max(STRENGTH_MIN, s))


def clamp_tiles(count: object, height: int | None = None, width: int | None = None) -> int:
    """Tile count in [TILES_MIN, TILES_MAX], and never denser than the frame."""
    try:
        n = int(round(float(count)))
    except (TypeError, ValueError):
        n = TILES_DEFAULT
    n = min(TILES_MAX, max(TILES_MIN, n))
    if height is not None and width is not None:
        cap = max(TILES_MIN, (max(1, int(height)) * max(1, int(width))) // 4)
        n = min(n, cap)
    return n


def clamp_lloyd(count: object) -> int:
    """Lloyd iterations in [LLOYD_MIN, LLOYD_MAX]."""
    try:
        n = int(round(float(count)))
    except (TypeError, ValueError):
        n = LLOYD_DEFAULT
    return min(LLOYD_MAX, max(LLOYD_MIN, n))


def normalize_h_side(side: str | None) -> str:
    raw = str(side or SIDE_OFF).strip().lower()
    if raw in ("left", "l"):
        return SIDE_LEFT
    if raw in ("right", "r"):
        return SIDE_RIGHT
    return SIDE_OFF


def normalize_v_side(side: str | None) -> str:
    raw = str(side or SIDE_OFF).strip().lower()
    if raw in ("top", "t", "up"):
        return SIDE_TOP
    if raw in ("bottom", "b", "down"):
        return SIDE_BOTTOM
    return SIDE_OFF


def normalize_tess_mode(mode: str | None) -> str:
    raw = str(mode or MODE_DEFAULT).strip().lower()
    raw = raw.replace("(", "").replace(")", "").replace(" ", "_").replace("-", "_")
    raw = "_".join(p for p in raw.split("_") if p)
    if raw in ("mesh", "mesh_warp", "warp", "escher"):
        return MODE_MESH
    if raw in ("voronoi", "detail", "detail_mosaic", "mosaic", "cells"):
        return MODE_VORONOI
    if raw in ("tessellate", "tessellation", "hilbert", "crinkly", "diffuse"):
        return MODE_TESSELLATE
    if raw in (
        "tile",
        "tile_repeating_design",
        "repeating",
        "repeat",
        "repeat_tile",
        "pattern",
        "pattern_tile",
        "repeating_design",
    ):
        return MODE_TILE
    return MODE_DEFAULT


def tess_mode_label(mode: str | None) -> str:
    """Dropdown label for a stored mode key (or a label / alias)."""
    key = normalize_tess_mode(mode)
    for stored, label in MODE_CHOICES:
        if stored == key:
            return label
    return MODE_LABEL_TILE


def coerce_normalize_lighting(value: object) -> bool:
    """True when spatial lighting flatten (bowl / ramp) is on."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, bytes)):
        raw = str(value).strip().lower()
        if raw in ("1", "true", "yes", "on"):
            return True
        if raw in ("0", "false", "no", "off", "", "none"):
            return False
        return False
    return bool(value)


def is_identity_tessellate(
    h_side: str | None,
    v_side: str | None,
    built: object = False,
    mode: str | None = None,
    normalize_lighting: object = False,
) -> bool:
    """True when wrap / mesh / mosaic / tile should skip.

    Tile, Tessellation, and Mesh still need a chosen side. Detail mosaic
    runs with sides Off (density-only, like the source script).
    ``normalize_lighting`` is ignored here — flatten is a separate
    pipeline step and never implies wrap.
    """
    del normalize_lighting
    if not coerce_built(built):
        return True
    if normalize_tess_mode(mode) == MODE_VORONOI:
        return False
    return (
        normalize_h_side(h_side) == SIDE_OFF
        and normalize_v_side(v_side) == SIDE_OFF
    )


def edges_already_match(
    arr: np.ndarray,
    h_side: str | None,
    v_side: str | None,
) -> bool:
    """True when opposite wrap edges already agree (structure + luma).

    ``edge_mismatch`` is wrap-strip MSE (structure first, a little raw
    color). A perfectly tiled / wrap-identical uint8 array scores 0.0.
    Cap ``_IDENTITY_EDGE_MSE`` is ~0 so Build stays a no-op.
    """
    h_side = normalize_h_side(h_side)
    v_side = normalize_v_side(v_side)
    if h_side == SIDE_OFF and v_side == SIDE_OFF:
        return True
    return float(edge_mismatch(arr, h_side, v_side)) <= _IDENTITY_EDGE_MSE


def _low_frequency_field(ch: np.ndarray) -> np.ndarray:
    """Estimate the light field: downsample, blur, upsample (not the weave)."""
    src = np.ascontiguousarray(ch, dtype=np.float32)
    h, w = int(src.shape[0]), int(src.shape[1])
    if max(h, w) <= 1:
        return src
    lo = float(np.min(src))
    hi = float(np.max(src))
    span = hi - lo
    if span < 1e-6:
        return src
    u8 = np.clip(np.round((src - lo) * (255.0 / span)), 0, 255).astype(np.uint8)
    im = Image.fromarray(u8, mode="L")
    long_edge = max(h, w)
    if long_edge > _ILLUM_WORK_EDGE:
        scale = _ILLUM_WORK_EDGE / float(long_edge)
        ww = max(1, int(round(w * scale)))
        wh = max(1, int(round(h * scale)))
        im = im.resize((ww, wh), Image.Resampling.BILINEAR)
    radius = max(1.2, _ILLUM_BLUR_FRAC * float(min(im.size)))
    im = im.filter(ImageFilter.GaussianBlur(radius=radius))
    if im.size != (w, h):
        im = im.resize((w, h), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32) * (span / 255.0) + lo


def _stride_for_fit(ch: np.ndarray, max_edge: int) -> np.ndarray:
    src = np.ascontiguousarray(ch, dtype=np.float32)
    h, w = int(src.shape[0]), int(src.shape[1])
    long_edge = max(h, w)
    if long_edge <= max_edge:
        return src
    step = max(1, int(round(long_edge / float(max_edge))))
    return src[::step, ::step]


def _bilinear_field(ch: np.ndarray) -> np.ndarray:
    """Least-squares a + bx + cy + dxy evaluated at full resolution.

    A huge Gaussian of a linear ramp is nearly constant, so the bowl
    divide leaves the ramp. Fitting this plane (bilinear) kills it.
    """
    src = np.ascontiguousarray(ch, dtype=np.float32)
    h, w = int(src.shape[0]), int(src.shape[1])
    if h < 2 or w < 2:
        return src
    work = _stride_for_fit(src, _ILLUM_PLANE_EDGE)
    wh, ww = int(work.shape[0]), int(work.shape[1])
    ys = np.linspace(0.0, 1.0, wh, dtype=np.float64)
    xs = np.linspace(0.0, 1.0, ww, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    a_mat = np.stack(
        (np.ones(xx.size, dtype=np.float64), xx.ravel(), yy.ravel(), (xx * yy).ravel()),
        axis=1,
    )
    coeffs, *_ = np.linalg.lstsq(a_mat, work.ravel().astype(np.float64), rcond=None)
    fy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    fx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    a0, b0, c0, d0 = (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]), float(coeffs[3]))
    return a0 + b0 * fx + c0 * fy + d0 * fx * fy


def _divide_by_field(src: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Multiplicative flatten; keep the source mean."""
    src_f = src.astype(np.float32, copy=False)
    field_f = field.astype(np.float32, copy=False)
    ref = float(np.mean(field_f))
    if abs(ref) < _ILLUM_L_EPS:
        ref = _ILLUM_L_EPS if ref >= 0.0 else -_ILLUM_L_EPS
    gain = ref / np.maximum(field_f, _ILLUM_L_EPS)
    out = src_f * gain
    src_mean = float(np.mean(src_f))
    out_mean = float(np.mean(out))
    if abs(out_mean) > 1e-6:
        out *= src_mean / out_mean
    return out


def _add_opposite_edge_ramp(plane: np.ndarray, axis: int, mix: float = 1.0) -> np.ndarray:
    """1D ramp so opposite-edge means meet at their midpoint."""
    out = plane.astype(np.float32, copy=True)
    n = int(out.shape[axis])
    if n < 2 or mix == 0.0:
        return out
    if axis == 0:
        lo = float(np.mean(out[0]))
        hi = float(np.mean(out[-1]))
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    else:
        lo = float(np.mean(out[:, 0]))
        hi = float(np.mean(out[:, -1]))
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)[None, :]
    mid = 0.5 * (lo + hi)
    out += float(mix) * ((1.0 - t) * (mid - lo) + t * (mid - hi))
    return out


def _match_opposite_lab_ramps(lab: np.ndarray) -> np.ndarray:
    """Both-axis 1D ramps: L* fully, a*b* mildly. Mean L* is restored."""
    out = lab.astype(np.float32, copy=True)
    mean_l = float(np.mean(out[..., 0]))
    out[..., 0] = _add_opposite_edge_ramp(out[..., 0], 0, 1.0)
    out[..., 0] = _add_opposite_edge_ramp(out[..., 0], 1, 1.0)
    out[..., 1] = _add_opposite_edge_ramp(out[..., 1], 0, _ILLUM_CHROMA_MIX)
    out[..., 1] = _add_opposite_edge_ramp(out[..., 1], 1, _ILLUM_CHROMA_MIX)
    out[..., 2] = _add_opposite_edge_ramp(out[..., 2], 0, _ILLUM_CHROMA_MIX)
    out[..., 2] = _add_opposite_edge_ramp(out[..., 2], 1, _ILLUM_CHROMA_MIX)
    out[..., 0] += mean_l - float(np.mean(out[..., 0]))
    out[..., 0] = np.clip(out[..., 0], 0.0, 100.0)
    return out


def _flatten_luma_plane(plane: np.ndarray) -> np.ndarray:
    """Divide out a luma bowl and a bilinear ramp; then match opposite edges."""
    src = plane.astype(np.float32, copy=False)
    mean_l = float(np.mean(src))
    out = _divide_by_field(src, _low_frequency_field(src))
    out = _divide_by_field(out, _bilinear_field(out))
    out = _add_opposite_edge_ramp(out, 0, 1.0)
    out = _add_opposite_edge_ramp(out, 1, 1.0)
    out += mean_l - float(np.mean(out))
    return out


def _match_wrap_edges_array(arr: np.ndarray) -> np.ndarray:
    """Lab opposite-edge ramps on RGB (or luma ramps on a single plane)."""
    if arr.ndim == 2:
        out = _add_opposite_edge_ramp(arr.astype(np.float32, copy=False), 0, 1.0)
        out = _add_opposite_edge_ramp(out, 1, 1.0)
        return _cast_like(out, arr.dtype)
    rgb = arr[..., :3]
    lab = rgb_to_lab_array(rgb)
    lab = _match_opposite_lab_ramps(lab)
    flat_rgb = lab_to_rgb_array(lab)
    if arr.shape[-1] == 3:
        return flat_rgb if flat_rgb.dtype == arr.dtype else _cast_like(flat_rgb, arr.dtype)
    out = np.array(arr, copy=True)
    out[..., :3] = flat_rgb
    return out


def flatten_illumination_array(arr: np.ndarray) -> np.ndarray:
    """Remove large-scale lighting bowls *and* linear/bilinear ramps.

    Lab L* is divided by a very blurred copy, then by a fitted bilinear
    plane (gain referenced to mean L*) so a studio pillow *or* a
    top-dark/bottom-bright ramp dies while weave grain stays. Opposite
    edges then get a 1D L* ramp (mild a*b*) so wrap is already continuous.
    """
    if arr.ndim == 2:
        out = _flatten_luma_plane(arr)
        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            return np.clip(np.round(out), info.min, info.max).astype(arr.dtype, copy=False)
        return out.astype(arr.dtype, copy=False)
    rgb = arr[..., :3]
    lab = rgb_to_lab_array(rgb)
    lab[..., 0] = np.clip(_flatten_luma_plane(lab[..., 0]), 0.0, 100.0)
    a_illum = _low_frequency_field(lab[..., 1])
    b_illum = _low_frequency_field(lab[..., 2])
    lab[..., 1] = lab[..., 1] - _ILLUM_CHROMA_MIX * (a_illum - float(np.mean(a_illum)))
    lab[..., 2] = lab[..., 2] - _ILLUM_CHROMA_MIX * (b_illum - float(np.mean(b_illum)))
    lab = _match_opposite_lab_ramps(lab)
    flat_rgb = lab_to_rgb_array(lab)
    if arr.shape[-1] == 3:
        return flat_rgb
    out = np.array(arr, copy=True)
    out[..., :3] = flat_rgb
    return out


def apply_normalize_lighting(image: Image.Image) -> Image.Image:
    """Standalone Lab bowl + bilinear + opposite-edge flatten. No tessellate wrap."""
    arr = np.asarray(image)
    out = flatten_illumination_array(arr)
    if out is arr:
        return image
    return Image.fromarray(out, mode=image.mode)


def estimate_normalize_tone(image: Image.Image) -> tuple[float, float]:
    """Darks / Lights (−1…+1) that approximate flatten via existing Tone math.

    Preview/save apply ``apply_normalize_lighting`` as a spatial step;
    this estimate is only for callers that still map flatten onto sliders.
    A flatten that lifts shadows writes negative Darks (left = pull darks
    back) so the slider matches the inverted Darks mix.
    """
    from wallpaper_recolor.color.tone import estimate_tone_amounts

    rgb = np.asarray(image.convert("RGB"))
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    long_edge = max(h, w)
    if long_edge > 256:
        scale = 256.0 / float(long_edge)
        ww = max(1, int(round(w * scale)))
        wh = max(1, int(round(h * scale)))
        rgb = np.asarray(
            Image.fromarray(rgb).resize((ww, wh), Image.Resampling.BILINEAR)
        )
    flat = flatten_illumination_array(rgb)
    return estimate_tone_amounts(rgb, flat)


# ---------------------------------------------------------------------------
# Probe / mismatch — any image, not a textile period finder
# ---------------------------------------------------------------------------

def _probe(arr: np.ndarray) -> np.ndarray:
    """Stride-downscale for search; keeps edge structure without a full copy."""
    h, w = int(arr.shape[0]), int(arr.shape[1])
    long_edge = max(h, w)
    if long_edge <= _PROBE_MAX_EDGE:
        return arr
    step = max(1, int(round(long_edge / _PROBE_MAX_EDGE)))
    return arr[::step, ::step]


def _as_float(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32, copy=False)


def _zero_mean_spatial(arr: np.ndarray) -> np.ndarray:
    a = _as_float(arr)
    if a.ndim >= 3:
        return a - np.mean(a, axis=(0, 1), keepdims=True)
    return a - float(np.mean(a))


def _strip_mismatch(left: np.ndarray, right: np.ndarray) -> float:
    """Wrap cost: structure first (weave), a little raw color (lighting)."""
    a = _as_float(left)
    b = _as_float(right)
    raw = float(np.mean((a - b) ** 2))
    struct = float(np.mean((_zero_mean_spatial(a) - _zero_mean_spatial(b)) ** 2))
    return _ALIGN_STRUCT_WEIGHT * struct + _ALIGN_COLOR_WEIGHT * raw


def _align_band_size(n: int) -> int:
    return max(1, min(_ALIGN_BAND, max(1, int(n) // 4)))


def edge_mismatch(arr: np.ndarray, h_side: str, v_side: str) -> float:
    """MSE of wrap-edge *strips* (and four corners when both axes are on)."""
    a = _as_float(arr)
    height, width = int(a.shape[0]), int(a.shape[1])
    cost = 0.0
    n = 0
    if h_side != SIDE_OFF:
        bw = _align_band_size(width)
        cost += _strip_mismatch(a[:, :bw], a[:, -bw:])
        n += 1
    if v_side != SIDE_OFF:
        bh = _align_band_size(height)
        cost += _strip_mismatch(a[:bh], a[-bh:])
        n += 1
    if h_side != SIDE_OFF and v_side != SIDE_OFF:
        corners = np.stack(
            (a[0, 0], a[0, -1], a[-1, 0], a[-1, -1]), axis=0
        )
        mean_c = np.mean(corners, axis=0, keepdims=True)
        cost += float(np.mean((corners - mean_c) ** 2))
        n += 1
    if n == 0:
        return 0.0
    return cost / float(n)


def _origin_samples(max_o: int, count: int) -> list[int]:
    max_o = max(0, int(max_o))
    if max_o <= 0:
        return [0]
    n = min(max(2, int(count)), max_o + 1)
    return [int(round(v)) for v in np.linspace(0, max_o, n)]


def find_match_crop(
    arr: np.ndarray,
    h_side: str,
    v_side: str,
) -> tuple[int, int, float]:
    """Zoom / top-left that minimize wrap-edge MSE. Prefers less zoom.

    Origins grow from the chosen side: Left/Top stay near 0; Right/Bottom
    sit near the far origin so the window keeps that edge as the model.
    """
    h_side = normalize_h_side(h_side)
    v_side = normalize_v_side(v_side)
    height, width = int(arr.shape[0]), int(arr.shape[1])
    probe = _probe(arr)
    ph, pw = int(probe.shape[0]), int(probe.shape[1])
    best_z = 1.0
    best_x = 0
    best_y = 0
    best_cost = float("inf")
    for zoom in _SEARCH_ZOOMS:
        cw, ch = window_size(pw, ph, zoom)
        max_x, max_y = max_origin(pw, ph, zoom)
        if h_side == SIDE_OFF:
            xs = [0]
        elif h_side == SIDE_RIGHT:
            xs = list(reversed(_origin_samples(max_x, _SEARCH_ORIGINS)))
        else:
            xs = _origin_samples(max_x, _SEARCH_ORIGINS)
        if v_side == SIDE_OFF:
            ys = [0]
        elif v_side == SIDE_BOTTOM:
            ys = list(reversed(_origin_samples(max_y, _SEARCH_ORIGINS)))
        else:
            ys = _origin_samples(max_y, _SEARCH_ORIGINS)
        for y in ys:
            for x in xs:
                win = probe[y : y + ch, x : x + cw]
                if win.shape[0] < 2 or win.shape[1] < 2:
                    continue
                cost = edge_mismatch(win, h_side, v_side)
                cost += _ZOOM_COST * (float(zoom) - 1.0) * 255.0 * 255.0
                if cost < best_cost:
                    best_cost = cost
                    best_z = float(zoom)
                    sx = (float(x) / float(pw)) * float(width) if pw else 0.0
                    sy = (float(y) / float(ph)) * float(height) if ph else 0.0
                    best_x = int(round(sx))
                    best_y = int(round(sy))
    max_x, max_y = max_origin(width, height, best_z)
    return (
        max(0, min(best_x, max_x)),
        max(0, min(best_y, max_y)),
        float(best_z),
    )


def _shift_candidates(n: int) -> list[int]:
    n = max(1, int(n))
    span = max(1, min(_SHIFT_MAX, int(round(n * _SHIFT_FRAC))))
    if span <= 1:
        return [0, 1, -1]
    step = max(1, span // 6)
    out = list(range(0, span + 1, step)) + list(range(-step, -span - 1, -step))
    if 1 not in out:
        out.append(1)
    if -1 not in out:
        out.append(-1)
    # unique, 0 first
    seen = set()
    ordered: list[int] = []
    for s in out:
        if s not in seen:
            seen.add(s)
            ordered.append(int(s))
    return ordered


def _axis_wrap_cost_shifted(arr: np.ndarray, shift: int, axis: int) -> float:
    """Wrap-strip cost of ``np.roll(arr, shift, axis)`` without rolling the frame."""
    n = int(arr.shape[axis])
    if n < 2:
        return 0.0
    band = _align_band_size(n)
    s = int(shift) % n
    lo = (np.arange(band, dtype=np.int64) - s) % n
    hi = (np.arange(n - band, n, dtype=np.int64) - s) % n
    if axis == 1:
        return _strip_mismatch(arr[:, lo], arr[:, hi])
    return _strip_mismatch(arr[lo], arr[hi])


def _best_axis_shift(arr: np.ndarray, axis: int) -> int:
    """Integer roll (full-res pixels) that minimizes opposite-edge structure MSE."""
    probe = _probe(arr)
    pn = int(probe.shape[axis])
    fn = int(arr.shape[axis])
    best_s = 0
    best_cost = float("inf")
    for s in _shift_candidates(pn):
        err = _axis_wrap_cost_shifted(probe, s, axis)
        if err < best_cost:
            best_cost = err
            best_s = int(s)
    scale = float(fn) / float(max(pn, 1))
    full_s = int(round(best_s * scale)) % max(fn, 1)
    step = max(1, int(round(scale)))
    best_full = full_s
    best_full_cost = _axis_wrap_cost_shifted(arr, full_s, axis)
    for s in range(full_s - step, full_s + step + 1):
        err = _axis_wrap_cost_shifted(arr, s, axis)
        if err < best_full_cost:
            best_full_cost = err
            best_full = int(s)
    if fn:
        best_full %= fn
        if best_full > fn // 2:
            best_full -= fn
    return int(best_full)


def _autocorr_period_1d(sig: np.ndarray) -> int:
    """First strong autocorrelation peak; ``len(sig)`` if none."""
    n = int(sig.size)
    if n < _PERIOD_MIN * 2:
        return n
    work = np.asarray(sig, dtype=np.float64).reshape(-1)
    work = work - float(np.mean(work))
    energy = float(np.dot(work, work))
    if energy < 1e-6:
        return n
    nfft = int(1 << int(np.ceil(np.log2(max(8, n * 2)))))
    spec = np.fft.rfft(work, n=nfft)
    ac = np.fft.irfft(spec * np.conj(spec), n=nfft)[:n]
    ac /= ac[0] + 1e-12
    lo = _PERIOD_MIN
    hi = max(lo + 1, n // 2)
    region = ac[lo:hi]
    if region.size < 3:
        return n
    peak_h = float(np.max(region))
    if peak_h < _PERIOD_AC_MIN:
        return n
    thresh = max(_PERIOD_AC_MIN, _PERIOD_AC_REL * peak_h)
    chosen = 0
    for i in range(lo + 1, hi - 1):
        if ac[i] >= ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] >= thresh:
            chosen = int(i)
            break
    if chosen <= 0:
        chosen = int(np.argmax(region)) + lo
    # Prefer the fundamental when 2×period is the strongest peak.
    half = chosen // 2
    if half >= lo and ac[half] >= thresh * 0.85:
        chosen = half
    return max(_PERIOD_MIN, min(n, chosen))


def estimate_axis_period(arr: np.ndarray, axis: int) -> int:
    """Motif period in pixels along ``axis`` (0=rows/Y, 1=cols/X).

    Mean-projects the other axis on a stride probe, then takes the first
    strong 1D autocorrelation peak. Returns the full span when the axis
    is not periodic.
    """
    axis = 1 if int(axis) else 0
    n_full = int(arr.shape[axis])
    if n_full < _PERIOD_MIN * 2:
        return n_full
    probe = _probe(arr) if max(int(arr.shape[0]), int(arr.shape[1])) > _PERIOD_PROBE else arr
    gray = _gray32(probe)
    sig = np.mean(gray, axis=0 if axis == 1 else 1)
    k = _autocorr_period_1d(sig)
    n_p = int(sig.size)
    if k >= n_p // 2:
        return n_full
    period = int(round(k * float(n_full) / float(max(n_p, 1))))
    return max(_PERIOD_MIN, min(n_full, period))


def _axis_cells_match(arr: np.ndarray, period: int, axis: int) -> bool:
    """True when two adjacent period cells agree (image is already repeating)."""
    n = int(arr.shape[axis])
    p = int(period)
    if p < 2 or p > n // 2 or n < p * 2:
        return False
    if axis == 1:
        return _strip_mismatch(arr[:, :p], arr[:, p : p * 2]) <= _IDENTITY_EDGE_MSE
    return _strip_mismatch(arr[:p], arr[p : p * 2]) <= _IDENTITY_EDGE_MSE


def _axis_whole_repeats(n: int, period: int) -> bool:
    """True when ``n`` is an integer number of ``period`` cells (no leftover)."""
    n = int(n)
    p = int(period)
    if p < 2 or p > n // 2:
        return False
    return n % p == 0


def image_already_periodic(
    arr: np.ndarray,
    h_side: str | None,
    v_side: str | None,
) -> bool:
    """True when chosen wrap axes are already a whole number of repeats.

    Opposite-edge agreement alone is not enough: a studio lighting bowl
    makes left≈right (both dark) while the weave still fails to wrap.
    """
    h_side = normalize_h_side(h_side)
    v_side = normalize_v_side(v_side)
    if h_side == SIDE_OFF and v_side == SIDE_OFF:
        return True
    height, width = int(arr.shape[0]), int(arr.shape[1])
    if h_side != SIDE_OFF:
        px = estimate_axis_period(arr, 1)
        if not _axis_whole_repeats(width, px) or not _axis_cells_match(arr, px, 1):
            return False
    if v_side != SIDE_OFF:
        py = estimate_axis_period(arr, 0)
        if not _axis_whole_repeats(height, py) or not _axis_cells_match(arr, py, 0):
            return False
    return True


def _repeat_origin_samples(max_o: int, period: int, from_low: bool) -> list[int]:
    max_o = max(0, int(max_o))
    p = max(1, int(period))
    if max_o <= 0:
        return [0]
    aligned = list(range(0, max_o + 1, p))
    if max_o not in aligned:
        aligned.append(max_o)
    if from_low:
        ordered = aligned
    else:
        ordered = list(reversed(aligned))
    if 0 not in ordered:
        ordered.append(0)
    seen: set[int] = set()
    out: list[int] = []
    for x in ordered:
        if x not in seen:
            seen.add(x)
            out.append(int(x))
        if len(out) >= _SEARCH_ORIGINS:
            break
    return out


def _repeat_zoom_candidates(src_w: int, src_h: int, px: int, py: int, h_on: bool, v_on: bool) -> list[float]:
    zooms = [float(z) for z in _SEARCH_ZOOMS]
    if h_on and px >= _PERIOD_MIN:
        for nrep in range(max(1, src_w // px), 0, -1):
            z = float(src_w) / float(max(1, nrep * px))
            if 1.0 <= z <= 8.0:
                zooms.append(z)
    if v_on and py >= _PERIOD_MIN:
        for nrep in range(max(1, src_h // py), 0, -1):
            z = float(src_h) / float(max(1, nrep * py))
            if 1.0 <= z <= 8.0:
                zooms.append(z)
    out: list[float] = []
    seen: set[float] = set()
    for z in sorted(zooms):
        zc = float(clamp_zoom(z))
        key = round(zc, 4)
        if key not in seen:
            seen.add(key)
            out.append(zc)
    return out


def find_repeat_crop(
    arr: np.ndarray,
    h_side: str,
    v_side: str,
) -> tuple[int, int, float]:
    """Zoom / origin whose window is a whole number of motif repeats."""
    h_side = normalize_h_side(h_side)
    v_side = normalize_v_side(v_side)
    height, width = int(arr.shape[0]), int(arr.shape[1])
    probe = _probe(arr)
    ph, pw = int(probe.shape[0]), int(probe.shape[1])
    px = estimate_axis_period(probe, 1) if h_side != SIDE_OFF else pw
    py = estimate_axis_period(probe, 0) if v_side != SIDE_OFF else ph
    best_z = 1.0
    best_x = 0
    best_y = 0
    best_cost = float("inf")
    h_on = h_side != SIDE_OFF
    v_on = v_side != SIDE_OFF
    for zoom in _repeat_zoom_candidates(pw, ph, px, py, h_on, v_on):
        cw, ch = window_size(pw, ph, zoom)
        max_x, max_y = max_origin(pw, ph, zoom)
        if not h_on:
            xs = [0]
        else:
            xs = _repeat_origin_samples(max_x, px, from_low=h_side != SIDE_RIGHT)
        if not v_on:
            ys = [0]
        else:
            ys = _repeat_origin_samples(max_y, py, from_low=v_side != SIDE_BOTTOM)
        leftover_w = 0 if not h_on or px < 2 else min(cw % px, px - (cw % px) if cw % px else 0)
        leftover_h = 0 if not v_on or py < 2 else min(ch % py, py - (ch % py) if ch % py else 0)
        zoom_pen = _ZOOM_COST * (float(zoom) - 1.0) * 255.0 * 255.0
        for y in ys:
            for x in xs:
                win = probe[y : y + ch, x : x + cw]
                if win.shape[0] < 2 or win.shape[1] < 2:
                    continue
                cost = float(leftover_w + leftover_h) * 80.0
                cost += edge_mismatch(win, h_side, v_side)
                cost += zoom_pen
                if cost < best_cost:
                    best_cost = cost
                    best_z = float(zoom)
                    sx = (float(x) / float(pw)) * float(width) if pw else 0.0
                    sy = (float(y) / float(ph)) * float(height) if ph else 0.0
                    best_x = int(round(sx))
                    best_y = int(round(sy))
    max_x, max_y = max_origin(width, height, best_z)
    return (
        max(0, min(best_x, max_x)),
        max(0, min(best_y, max_y)),
        float(best_z),
    )


def _core_origin_span(
    n: int,
    period: int,
    from_low: bool,
) -> tuple[int, int]:
    """Origin and length of the largest whole-repeat strip along one axis."""
    n = max(1, int(n))
    p = max(1, int(period))
    if p >= n:
        return 0, n
    count = max(1, n // p)
    span = count * p
    if span > n:
        span = n
    if from_low:
        return 0, span
    return n - span, span


def _copy_fill_leftover(
    arr: np.ndarray,
    ox: int,
    cw: int,
    oy: int,
    ch: int,
    h_on: bool,
    v_on: bool,
) -> np.ndarray:
    """Paste motif wrap into leftover strips; keep the integer-repeat core.

    Never resize a leftover band (or repeat one column). Stretching a 1-wide
    source turns horizontal linen grain into vertical streaks along the edge.
    """
    out = np.array(arr, copy=True)
    height, width = int(arr.shape[0]), int(arr.shape[1])
    ox, cw = int(ox), max(1, int(cw))
    oy, ch = int(oy), max(1, int(ch))
    if h_on and cw >= 2 and cw < width:
        xs = np.arange(width, dtype=np.int64)
        hole = (xs < ox) | (xs >= ox + cw)
        if np.any(hole):
            src = ox + np.mod(xs - ox, cw)
            src = np.clip(src, 0, width - 1)
            # One unique source column would smear leftover into vertical streaks.
            if int(np.unique(src[hole]).size) >= 2 or int(np.count_nonzero(hole)) <= 2:
                out[:, hole] = arr[:, src[hole]]
    if v_on and ch >= 2 and ch < height:
        ys = np.arange(height, dtype=np.int64)
        hole = (ys < oy) | (ys >= oy + ch)
        if np.any(hole):
            src = oy + np.mod(ys - oy, ch)
            src = np.clip(src, 0, height - 1)
            if int(np.unique(src[hole]).size) >= 2 or int(np.count_nonzero(hole)) <= 2:
                out[hole] = out[src[hole]]
    return out


def _fit_frame_by_wrap(
    arr: np.ndarray,
    src_h: int,
    src_w: int,
    h_side: str,
    v_side: str,
) -> np.ndarray:
    """Restore the source frame by wrapping the motif — never bilinear leftover."""
    height, width = int(arr.shape[0]), int(arr.shape[1])
    src_h, src_w = max(1, int(src_h)), max(1, int(src_w))
    if height == src_h and width == src_w:
        return arr
    if height >= src_h and width >= src_w:
        y0 = height - src_h if v_side == SIDE_BOTTOM else 0
        x0 = width - src_w if h_side == SIDE_RIGHT else 0
        y0 = max(0, min(y0, height - src_h))
        x0 = max(0, min(x0, width - src_w))
        return np.array(arr[y0 : y0 + src_h, x0 : x0 + src_w], copy=True)
    if arr.ndim == 2:
        canvas = np.zeros((src_h, src_w), dtype=arr.dtype)
    else:
        canvas = np.zeros((src_h, src_w) + arr.shape[2:], dtype=arr.dtype)
    paste_h = min(height, src_h)
    paste_w = min(width, src_w)
    ox = src_w - paste_w if h_side == SIDE_RIGHT else 0
    oy = src_h - paste_h if v_side == SIDE_BOTTOM else 0
    ox = max(0, min(ox, src_w - paste_w))
    oy = max(0, min(oy, src_h - paste_h))
    canvas[oy : oy + paste_h, ox : ox + paste_w] = arr[:paste_h, :paste_w]
    return _copy_fill_leftover(
        canvas,
        ox,
        paste_w,
        oy,
        paste_h,
        h_side != SIDE_OFF and paste_w < src_w,
        v_side != SIDE_OFF and paste_h < src_h,
    )


def _tile_from_pattern(
    arr: np.ndarray,
    h_side: str,
    v_side: str,
    *,
    nearest: bool,
) -> np.ndarray:
    """Crop to whole repeats and copy leftover from the motif (do not stretch)."""
    height, width = int(arr.shape[0]), int(arr.shape[1])
    px = estimate_axis_period(arr, 1) if h_side != SIDE_OFF else width
    py = estimate_axis_period(arr, 0) if v_side != SIDE_OFF else height
    ox, cw = (
        _core_origin_span(width, px, from_low=h_side != SIDE_RIGHT)
        if h_side != SIDE_OFF
        else (0, width)
    )
    oy, ch = (
        _core_origin_span(height, py, from_low=v_side != SIDE_BOTTOM)
        if v_side != SIDE_OFF
        else (0, height)
    )
    cw = max(1, min(cw, width - ox))
    ch = max(1, min(ch, height - oy))
    out = _copy_fill_leftover(
        arr, ox, cw, oy, ch, h_side != SIDE_OFF, v_side != SIDE_OFF
    )
    if not nearest:
        out = _seal_wrap_axes(out, h_side, v_side)
        out = _refine_tile_seams_ai(out, h_side, v_side)
    return _pin_wrap_edges(out, h_side, v_side)


def _refine_tile_seams_ai(arr: np.ndarray, h_side: str, v_side: str) -> np.ndarray:
    """LaMa (or period/Hilbert) on rolled wrap seams — both edges as context.

    Leftover copy only sees one corner of this scan. Rolling the tile puts
    opposite edges together so inpaint can synthesize the join from the
    surrounding motif instead of cloning that corner.
    """
    if h_side == SIDE_OFF and v_side == SIDE_OFF:
        return arr
    try:
        from wallpaper_recolor.transform.inpaint import inpaint_wrap_seams
    except ImportError:
        return arr
    return inpaint_wrap_seams(
        arr,
        wrap_h=h_side != SIDE_OFF,
        wrap_v=v_side != SIDE_OFF,
    )


def _tile_wrap_if_needed(
    arr: np.ndarray,
    h_side: str,
    v_side: str,
    *,
    nearest: bool,
) -> np.ndarray:
    """Period crop/fill unless sides are Off or the frame already repeats."""
    if h_side == SIDE_OFF and v_side == SIDE_OFF:
        return arr
    if image_already_periodic(arr, h_side, v_side):
        return arr
    return _tile_from_pattern(arr, h_side, v_side, nearest=nearest)


def plan_tessellate_crop(
    arr: np.ndarray,
    h_side: str | None,
    v_side: str | None,
    mode: str | None = None,
) -> tuple[int, int, float]:
    """Crop window (x, y, zoom) Build should write to the Crop panel.

    Tile mode aligns to whole motif repeats. Tessellation / mosaic search
    wrap-edge MSE, then bake residual rolls into origin when they still fit.
    Identity tiles return ``(0, 0, 1.0)`` so Crop can stay put.
    """
    h_side = normalize_h_side(h_side)
    v_side = normalize_v_side(v_side)
    mode = normalize_tess_mode(mode)
    height, width = int(arr.shape[0]), int(arr.shape[1])
    if mode == MODE_TILE:
        if image_already_periodic(arr, h_side, v_side):
            return 0, 0, 1.0
        return find_repeat_crop(arr, h_side, v_side)
    if edges_already_match(arr, h_side, v_side):
        return 0, 0, 1.0
    x, y, zoom = find_match_crop(arr, h_side, v_side)
    work = crop_array(arr, x, y, zoom)
    if h_side != SIDE_OFF:
        x -= _best_axis_shift(work, 1)
    if v_side != SIDE_OFF:
        y -= _best_axis_shift(work, 0)
    max_x, max_y = max_origin(width, height, zoom)
    return (
        max(0, min(int(x), max_x)),
        max(0, min(int(y), max_y)),
        float(zoom),
    )


# ---------------------------------------------------------------------------
# Tessellate — crop + Hilbert (crinkly) diffuse; opposite side is the model
# ---------------------------------------------------------------------------
# Brian Hayes, "Crinkly Curves", American Scientist 101(3), 2013:
# Hilbert/Peano keep nearby plane points nearby on a 1D path (locality).
# A lawn-mower / linear ramp does not — it smears structure. Blend along a
# Hilbert-folded front that still grows from the chosen side toward the model.

def _cast_like(out: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(out), info.min, info.max).astype(dtype, copy=False)
    return out.astype(dtype, copy=False)


def _hilbert_order(span: int) -> int:
    """Smallest Hilbert order whose 2^n grid covers ``span`` pixels."""
    span = max(2, int(span))
    return max(1, int(np.ceil(np.log2(span))))


def hilbert_xy_to_d(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    """Hilbert index of grid points — vectorized Butz / Wikipedia walk.

    ``order`` is n in 2^n. Quadrant recursion (Hayes: divide the square into
    four, rotate/reflect so the path stays continuous) maps (x, y) → d.
    Nearby (x, y) usually have nearby d — the locality tessellate blending uses.
    """
    order = max(1, int(order))
    n = 1 << order
    x_b, y_b = np.broadcast_arrays(
        np.asarray(x, dtype=np.int64),
        np.asarray(y, dtype=np.int64),
    )
    x_w = np.clip(x_b, 0, n - 1).copy()
    y_w = np.clip(y_b, 0, n - 1).copy()
    d = np.zeros(x_w.shape, dtype=np.int64)
    s = n >> 1
    while s > 0:
        rx = ((x_w & s) > 0).astype(np.int64)
        ry = ((y_w & s) > 0).astype(np.int64)
        d += (s * s) * ((3 * rx) ^ ry)
        # Rotate/reflect the next subsquare so the U-motifs join (Hayes).
        swap = ry == 0
        flip = swap & (rx == 1)
        if np.any(flip):
            x_w = np.where(flip, (n - 1) - x_w, x_w)
            y_w = np.where(flip, (n - 1) - y_w, y_w)
        if np.any(swap):
            x_new = np.where(swap, y_w, x_w)
            y_new = np.where(swap, x_w, y_w)
            x_w, y_w = x_new, y_new
        s >>= 1
    return d


def _crinkly_front_alpha(
    height: int,
    width: int,
    y0: int,
    y1: int,
    *,
    from_low: bool,
    axis: int,
) -> np.ndarray:
    """Mix weights for rows ``y0:y1``: 1 at the grow-from side, 0 at the model.

    Linear progress along the chosen→opposite axis is folded with the Hilbert
    index in the interior (crinkly front). Edges stay linear so wrap pins are
    exact. Not a straight wipe: mid-span weights vary along the perpendicular.
    """
    height = max(1, int(height))
    width = max(1, int(width))
    y0 = max(0, int(y0))
    y1 = min(height, max(y0 + 1, int(y1)))
    rows = y1 - y0
    ys = np.arange(y0, y1, dtype=np.int64)[:, None]
    xs = np.arange(width, dtype=np.int64)[None, :]
    span = max(height, width)
    order = _hilbert_order(span)
    n = 1 << order
    ix = np.minimum((xs * n) // width, n - 1)
    iy = np.minimum((ys * n) // height, n - 1)
    if axis == 1:
        hx, hy = (ix, iy) if from_low else (n - 1 - ix, iy)
        u = xs.astype(np.float32) / float(max(width - 1, 1))
    else:
        hx, hy = (iy, ix) if from_low else (n - 1 - iy, ix)
        u = ys.astype(np.float32) / float(max(height - 1, 1))
    if not from_low:
        u = 1.0 - u
    d = hilbert_xy_to_d(hx, hy, order)
    tmax = float(max(n * n - 1, 1))
    t = d.astype(np.float32) / tmax
    # g=0 at both edges (keep u); g=1 at mid-span (use Hilbert locality)
    g = 4.0 * u * (1.0 - u)
    t_front = u + (t - u) * g
    if t_front.shape != (rows, width):
        t_front = np.broadcast_to(t_front, (rows, width)).copy()
    band = max(_BLEND_FRAC, 1e-6)
    alpha = np.clip((band - t_front) / band, 0.0, 1.0)
    return alpha * alpha


def _mix_crinkly(
    arr: np.ndarray,
    other: np.ndarray,
    *,
    from_low: bool,
    axis: int,
    nearest: bool,
) -> np.ndarray:
    """``out = alpha * other + (1-alpha) * arr`` with a Hilbert front, by row strips."""
    height = int(arr.shape[0])
    width = int(arr.shape[1])
    strips: list[np.ndarray] = []
    for y0 in range(0, height, _STRIP_ROWS):
        y1 = min(height, y0 + _STRIP_ROWS)
        alpha = _crinkly_front_alpha(
            height, width, y0, y1, from_low=from_low, axis=axis
        )
        weight = alpha[..., None] if arr.ndim == 3 else alpha
        chunk = arr[y0:y1].astype(np.float32)
        alt = other[y0:y1].astype(np.float32)
        if nearest:
            blended = np.where(weight >= 0.5, alt, chunk)
        else:
            blended = weight * alt + (1.0 - weight) * chunk
        strips.append(_cast_like(blended, arr.dtype))
    return np.concatenate(strips, axis=0)


def _pin_horizontal(arr: np.ndarray, from_left: bool) -> np.ndarray:
    """Force left and right columns equal (mean of both wrap edges)."""
    del from_left
    out = np.array(arr, copy=True)
    left = _as_float(out[:, 0])
    right = _as_float(out[:, -1])
    mid = _cast_like(0.5 * (left + right), arr.dtype)
    out[:, 0] = mid
    out[:, -1] = mid
    return out


def _pin_vertical(arr: np.ndarray, from_top: bool) -> np.ndarray:
    """Force top and bottom rows equal (mean of both wrap edges)."""
    del from_top
    out = np.array(arr, copy=True)
    top = _as_float(out[0])
    bot = _as_float(out[-1])
    mid = _cast_like(0.5 * (top + bot), arr.dtype)
    out[0] = mid
    out[-1] = mid
    return out


def _pin_corners(arr: np.ndarray) -> np.ndarray:
    """All four wrap corners share the mean so a 3×3 tile has no corner crack."""
    a = _as_float(arr)
    mean_c = (
        a[0, 0].astype(np.float32)
        + a[0, -1].astype(np.float32)
        + a[-1, 0].astype(np.float32)
        + a[-1, -1].astype(np.float32)
    ) / 4.0
    out = np.array(arr, copy=True)
    pinned = _cast_like(mean_c[None, ...], arr.dtype)[0]
    out[0, 0] = pinned
    out[0, -1] = pinned
    out[-1, 0] = pinned
    out[-1, -1] = pinned
    return out


def _diffuse_horizontal(arr: np.ndarray, from_left: bool, *, nearest: bool) -> np.ndarray:
    """Grow from Left or Right along a Hilbert front; opposite column is the model."""
    wrap = np.roll(arr, 1 if from_left else -1, axis=1)
    out = _mix_crinkly(arr, wrap, from_low=from_left, axis=1, nearest=nearest)
    return _pin_horizontal(out, from_left)


def _diffuse_vertical(arr: np.ndarray, from_top: bool, *, nearest: bool) -> np.ndarray:
    """Grow from Top or Bottom along a Hilbert front; opposite row is the model."""
    wrap = np.roll(arr, 1 if from_top else -1, axis=0)
    out = _mix_crinkly(arr, wrap, from_low=from_top, axis=0, nearest=nearest)
    return _pin_vertical(out, from_top)


def _pin_wrap_edges(arr: np.ndarray, h_side: str, v_side: str) -> np.ndarray:
    """Chosen/opposite edges (and corners) agree after crop or scale-back."""
    if h_side == SIDE_LEFT:
        arr = _pin_horizontal(arr, True)
    elif h_side == SIDE_RIGHT:
        arr = _pin_horizontal(arr, False)
    if v_side == SIDE_TOP:
        arr = _pin_vertical(arr, True)
    elif v_side == SIDE_BOTTOM:
        arr = _pin_vertical(arr, False)
    if h_side != SIDE_OFF and v_side != SIDE_OFF:
        arr = _pin_corners(arr)
    return arr


def _seal_wrap_axes(arr: np.ndarray, h_side: str, v_side: str) -> np.ndarray:
    """1D opposite-edge ramps on the chosen wrap axes so tiles match in lighting."""
    if h_side == SIDE_OFF and v_side == SIDE_OFF:
        return arr
    if arr.ndim == 2:
        out = arr.astype(np.float32, copy=False)
        if v_side != SIDE_OFF:
            out = _add_opposite_edge_ramp(out, 0, 1.0)
        if h_side != SIDE_OFF:
            out = _add_opposite_edge_ramp(out, 1, 1.0)
        return _cast_like(out, arr.dtype)
    rgb = arr[..., :3]
    lab = rgb_to_lab_array(rgb)
    mean_l = float(np.mean(lab[..., 0]))
    if v_side != SIDE_OFF:
        lab[..., 0] = _add_opposite_edge_ramp(lab[..., 0], 0, 1.0)
        lab[..., 1] = _add_opposite_edge_ramp(lab[..., 1], 0, _ILLUM_CHROMA_MIX)
        lab[..., 2] = _add_opposite_edge_ramp(lab[..., 2], 0, _ILLUM_CHROMA_MIX)
    if h_side != SIDE_OFF:
        lab[..., 0] = _add_opposite_edge_ramp(lab[..., 0], 1, 1.0)
        lab[..., 1] = _add_opposite_edge_ramp(lab[..., 1], 1, _ILLUM_CHROMA_MIX)
        lab[..., 2] = _add_opposite_edge_ramp(lab[..., 2], 1, _ILLUM_CHROMA_MIX)
    lab[..., 0] += mean_l - float(np.mean(lab[..., 0]))
    lab[..., 0] = np.clip(lab[..., 0], 0.0, 100.0)
    flat_rgb = lab_to_rgb_array(lab)
    if arr.shape[-1] == 3:
        return flat_rgb if flat_rgb.dtype == arr.dtype else _cast_like(flat_rgb, arr.dtype)
    out = np.array(arr, copy=True)
    out[..., :3] = flat_rgb
    return out


def _hilbert_wrap_if_needed(
    arr: np.ndarray,
    h_side: str,
    v_side: str,
    *,
    nearest: bool,
) -> np.ndarray:
    """Diffuse + pin unless sides are Off or the frame already tiles."""
    if h_side == SIDE_OFF and v_side == SIDE_OFF:
        return arr
    if edges_already_match(arr, h_side, v_side):
        return arr
    return _diffuse_to_model(arr, h_side, v_side, nearest=nearest)


def _diffuse_to_model(
    arr: np.ndarray,
    h_side: str,
    v_side: str,
    *,
    nearest: bool,
) -> np.ndarray:
    """Diffuse toward the opposite side as the model. Crop is a separate step."""
    work = arr
    if h_side == SIDE_LEFT:
        work = _diffuse_horizontal(work, True, nearest=nearest)
    elif h_side == SIDE_RIGHT:
        work = _diffuse_horizontal(work, False, nearest=nearest)
    if v_side == SIDE_TOP:
        work = _diffuse_vertical(work, True, nearest=nearest)
    elif v_side == SIDE_BOTTOM:
        work = _diffuse_vertical(work, False, nearest=nearest)
    if not nearest:
        work = _seal_wrap_axes(work, h_side, v_side)
    return _pin_wrap_edges(work, h_side, v_side)


# ---------------------------------------------------------------------------
# Mesh warp — expanding cells + periodic cut (optional mode)
# ---------------------------------------------------------------------------

def _axis_map(
    t: np.ndarray,
    grow_from_zero: bool,
    strength: float,
) -> np.ndarray:
    """Identity → power stretch along one axis; ``t`` is 0…1 from the min edge.

    grow_from_zero True: chosen side is t=0 (Left / Top).
    False: chosen side is t=1 (Right / Bottom).
    """
    s = float(strength)
    p = 1.0 + _STRETCH_K * s
    if grow_from_zero:
        stretched = np.power(np.clip(t, 0.0, 1.0), p)
    else:
        stretched = 1.0 - np.power(np.clip(1.0 - t, 0.0, 1.0), p)
    return (1.0 - s) * t + s * stretched


def _source_maps(
    y0: int,
    y1: int,
    width: int,
    height: int,
    h_side: str,
    v_side: str,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Float source x/y for dest rows [y0:y1]. Period-w wrap matches u=0 with u=1."""
    w = max(1, int(width))
    h = max(1, int(height))
    rows = max(1, int(y1) - int(y0))
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(int(y0), int(y1), dtype=np.float32)
    u = xs[None, :] / max(w - 1, 1)
    v = ys[:, None] / max(h - 1, 1)
    u = np.broadcast_to(u, (rows, w)).copy()
    v = np.broadcast_to(v, (rows, w)).copy()
    s = float(strength)
    amp = _WAVE_AMP * s
    u_src = u
    v_src = v
    if h_side == SIDE_LEFT:
        u_src = _axis_map(u, True, s) + amp * np.sin(2.0 * np.pi * v)
    elif h_side == SIDE_RIGHT:
        u_src = _axis_map(u, False, s) + amp * np.sin(2.0 * np.pi * v)
    if v_side == SIDE_TOP:
        v_src = _axis_map(v, True, s) + amp * np.sin(2.0 * np.pi * u)
    elif v_side == SIDE_BOTTOM:
        v_src = _axis_map(v, False, s) + amp * np.sin(2.0 * np.pi * u)
    map_x = u_src * float(w)
    map_y = v_src * float(h)
    return map_x, map_y


def _remap_bilinear_wrap(
    arr: np.ndarray,
    map_y: np.ndarray,
    map_x: np.ndarray,
) -> np.ndarray:
    """Vectorized bilinear sample with periodic wrap (seamless tile)."""
    h, w = int(arr.shape[0]), int(arr.shape[1])
    x = np.mod(map_x.astype(np.float32, copy=False), float(w))
    y = np.mod(map_y.astype(np.float32, copy=False), float(h))
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    wx = x - x0.astype(np.float32)
    wy = y - y0.astype(np.float32)
    x1 = x0 + 1
    y1 = y0 + 1
    np.remainder(x0, w, out=x0)
    np.remainder(x1, w, out=x1)
    np.remainder(y0, h, out=y0)
    np.remainder(y1, h, out=y1)
    if arr.ndim == 2:
        ia = arr[y0, x0].astype(np.float32)
        ib = arr[y0, x1].astype(np.float32)
        ic = arr[y1, x0].astype(np.float32)
        id_ = arr[y1, x1].astype(np.float32)
        out = (ia * (1.0 - wx) + ib * wx) * (1.0 - wy) + (
            ic * (1.0 - wx) + id_ * wx
        ) * wy
    else:
        wx3 = wx[..., None]
        wy3 = wy[..., None]
        ia = arr[y0, x0].astype(np.float32)
        ib = arr[y0, x1].astype(np.float32)
        ic = arr[y1, x0].astype(np.float32)
        id_ = arr[y1, x1].astype(np.float32)
        out = (ia * (1.0 - wx3) + ib * wx3) * (1.0 - wy3) + (
            ic * (1.0 - wx3) + id_ * wx3
        ) * wy3
    return _cast_like(out, arr.dtype)


def _remap_nearest_wrap(
    arr: np.ndarray,
    map_y: np.ndarray,
    map_x: np.ndarray,
) -> np.ndarray:
    """Nearest sample with wrap — for integer label maps."""
    h, w = int(arr.shape[0]), int(arr.shape[1])
    x = np.mod(np.rint(map_x), float(w)).astype(np.int32)
    y = np.mod(np.rint(map_y), float(h)).astype(np.int32)
    return arr[y, x]


def _mesh_warp_array(
    arr: np.ndarray,
    h_side: str,
    v_side: str,
    strength: float,
    *,
    nearest: bool,
) -> np.ndarray:
    """Expanding-cell remap with wrap. ``nearest`` keeps labels as integers."""
    height, width = int(arr.shape[0]), int(arr.shape[1])
    remap = _remap_nearest_wrap if nearest else _remap_bilinear_wrap
    strips: list[np.ndarray] = []
    for y0 in range(0, height, _STRIP_ROWS):
        y1 = min(height, y0 + _STRIP_ROWS)
        map_x, map_y = _source_maps(y0, y1, width, height, h_side, v_side, strength)
        strips.append(remap(arr, map_y, map_x))
    return np.concatenate(strips, axis=0)


# ---------------------------------------------------------------------------
# Detail mosaic — raster Voronoi (numpy Sobel + nearest site, no OpenCV)
# ---------------------------------------------------------------------------

def _gray32(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)
    rgb = arr[..., :3].astype(np.float32)
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """3×3 Sobel magnitude — same role as cv2.Sobel, numpy only."""
    g = np.pad(gray.astype(np.float32, copy=False), 1, mode="edge")
    gx = (
        -g[:-2, :-2]
        + g[:-2, 2:]
        - 2.0 * g[1:-1, :-2]
        + 2.0 * g[1:-1, 2:]
        - g[2:, :-2]
        + g[2:, 2:]
    )
    gy = (
        -g[:-2, :-2]
        - 2.0 * g[:-2, 1:-1]
        - g[:-2, 2:]
        + g[2:, :-2]
        + 2.0 * g[2:, 1:-1]
        + g[2:, 2:]
    )
    return np.sqrt(gx * gx + gy * gy)


def _smooth_density(mag: np.ndarray) -> np.ndarray:
    """Light Gaussian so seed density follows structure, not pixel noise."""
    m = np.maximum(mag.astype(np.float32, copy=False), 0.0)
    peak = float(m.max()) if m.size else 0.0
    if peak <= 1e-6:
        return m
    u8 = np.clip(m * (255.0 / peak), 0.0, 255.0).astype(np.uint8)
    blurred = Image.fromarray(u8, mode="L").filter(ImageFilter.GaussianBlur(radius=1.2))
    return np.asarray(blurred, dtype=np.float32) * (peak / 255.0)


def _side_density_bias(height: int, width: int, h_side: str, v_side: str) -> np.ndarray:
    """Ramp that puts more seeds on the chosen side (grow toward the opposite)."""
    bias = np.ones((height, width), dtype=np.float32)
    if h_side == SIDE_OFF and v_side == SIDE_OFF:
        return bias
    ys = np.linspace(0.0, 1.0, num=max(1, height), dtype=np.float32)[:, None]
    xs = np.linspace(0.0, 1.0, num=max(1, width), dtype=np.float32)[None, :]
    if h_side == SIDE_LEFT:
        bias *= 1.0 + 1.6 * (1.0 - xs)
    elif h_side == SIDE_RIGHT:
        bias *= 1.0 + 1.6 * xs
    if v_side == SIDE_TOP:
        bias *= 1.0 + 1.6 * (1.0 - ys)
    elif v_side == SIDE_BOTTOM:
        bias *= 1.0 + 1.6 * ys
    return bias


def _edge_anchors(
    height: int,
    width: int,
    h_side: str,
    v_side: str,
    n_edge: int,
) -> np.ndarray:
    """Sites on the rectangle border so cells do not collapse at the frame."""
    h = max(1, int(height))
    w = max(1, int(width))
    n_edge = max(4, int(n_edge))
    xs = np.linspace(0.0, float(w - 1), n_edge, dtype=np.float32)
    ys = np.linspace(0.0, float(h - 1), n_edge, dtype=np.float32)
    top = np.stack([xs, np.zeros(n_edge, dtype=np.float32)], axis=1)
    bot = np.stack([xs, np.full(n_edge, float(h - 1), dtype=np.float32)], axis=1)
    left = np.stack([np.zeros(n_edge, dtype=np.float32), ys], axis=1)
    right = np.stack([np.full(n_edge, float(w - 1), dtype=np.float32), ys], axis=1)
    parts = [top, bot, left, right]
    extra = max(4, n_edge)
    if h_side == SIDE_LEFT:
        parts.append(np.stack([np.zeros(extra, dtype=np.float32), np.linspace(0.0, float(h - 1), extra, dtype=np.float32)], axis=1))
    elif h_side == SIDE_RIGHT:
        parts.append(np.stack([np.full(extra, float(w - 1), dtype=np.float32), np.linspace(0.0, float(h - 1), extra, dtype=np.float32)], axis=1))
    if v_side == SIDE_TOP:
        parts.append(np.stack([np.linspace(0.0, float(w - 1), extra, dtype=np.float32), np.zeros(extra, dtype=np.float32)], axis=1))
    elif v_side == SIDE_BOTTOM:
        parts.append(np.stack([np.linspace(0.0, float(w - 1), extra, dtype=np.float32), np.full(extra, float(h - 1), dtype=np.float32)], axis=1))
    return np.concatenate(parts, axis=0)


def _place_sites(
    arr: np.ndarray,
    h_side: str,
    v_side: str,
    tiles: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Density-weighted interior seeds + clipped edge anchors. Sites are (x, y)."""
    height, width = int(arr.shape[0]), int(arr.shape[1])
    tiles = clamp_tiles(tiles, height, width)
    dens = _smooth_density(_sobel_magnitude(_gray32(arr)))
    dens = dens * _side_density_bias(height, width, h_side, v_side)
    dens = dens + 1e-3 * float(np.max(dens) + 1.0)
    n_edge = max(6, int(round(np.sqrt(float(tiles)))))
    anchors = _edge_anchors(height, width, h_side, v_side, n_edge)
    n_anchor = int(anchors.shape[0])
    n_rand = max(0, tiles - n_anchor)
    if n_rand > 0:
        p = dens.ravel().astype(np.float64)
        p = p / p.sum()
        idx = rng.choice(p.size, size=min(n_rand, p.size), replace=False, p=p)
        ys, xs = np.divmod(idx, width)
        interior = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        sites = np.concatenate([anchors, interior], axis=0)
    else:
        sites = anchors
    sites[:, 0] = np.clip(sites[:, 0], 0.0, float(width - 1))
    sites[:, 1] = np.clip(sites[:, 1], 0.0, float(height - 1))
    return sites, n_anchor


def _resize_hw(arr: np.ndarray, height: int, width: int, *, nearest: bool) -> np.ndarray:
    if int(arr.shape[0]) == height and int(arr.shape[1]) == width:
        return arr
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    if arr.ndim == 2:
        if np.issubdtype(arr.dtype, np.integer) and arr.dtype != np.uint8:
            im = Image.fromarray(arr.astype(np.int32), mode="I")
            out = im.resize((width, height), resample)
            return np.asarray(out, dtype=arr.dtype)
        im = Image.fromarray(arr)
        return np.asarray(im.resize((width, height), resample), dtype=arr.dtype)
    im = Image.fromarray(arr)
    return np.asarray(im.resize((width, height), resample), dtype=arr.dtype)


def _fit_work(arr: np.ndarray, max_edge: int) -> np.ndarray:
    h, w = int(arr.shape[0]), int(arr.shape[1])
    long_edge = max(h, w)
    if long_edge <= max_edge:
        return arr
    scale = float(max_edge) / float(long_edge)
    return _resize_hw(
        arr,
        max(1, int(round(h * scale))),
        max(1, int(round(w * scale))),
        nearest=False,
    )


def _assign_nearest(height: int, width: int, sites: np.ndarray) -> np.ndarray:
    """Raster Voronoi: each pixel's nearest site. Batched so N×H×W is never allocated."""
    n = int(sites.shape[0])
    sx = sites[:, 0].astype(np.float32, copy=False)
    sy = sites[:, 1].astype(np.float32, copy=False)
    labels = np.empty((height, width), dtype=np.int32)
    batch = max(32, min(_VORONOI_ASSIGN_BATCH, height))
    xs = np.arange(width, dtype=np.float32)
    for y0 in range(0, height, batch):
        y1 = min(height, y0 + batch)
        ys = np.arange(y0, y1, dtype=np.float32)
        # (rows, cols, 1) minus (N,) → (rows, cols, N) would be huge; do rows×N then cols.
        # Per row: (W, N) = (xs - sx)^2 + (y - sy)^2
        for i, y in enumerate(ys):
            dx = xs[:, None] - sx[None, :]
            dy = y - sy
            d2 = dx * dx + dy * dy
            labels[y0 + i] = np.argmin(d2, axis=1)
    return labels


def _lloyd_relax(
    height: int,
    width: int,
    sites: np.ndarray,
    n_anchor: int,
    iterations: int,
) -> np.ndarray:
    """Move free sites to their cell centroid; edge anchors stay put. Clip each round."""
    out = sites.astype(np.float32, copy=True)
    n_anchor = max(0, min(int(n_anchor), int(out.shape[0])))
    it = clamp_lloyd(iterations)
    if it <= 0 or out.shape[0] <= n_anchor:
        return out
    probe_h, probe_w = height, width
    long_edge = max(height, width)
    if long_edge > _VORONOI_LLOYD_EDGE:
        scale = float(_VORONOI_LLOYD_EDGE) / float(long_edge)
        probe_h = max(2, int(round(height * scale)))
        probe_w = max(2, int(round(width * scale)))
        probe_sites = out.copy()
        probe_sites[:, 0] *= float(probe_w - 1) / max(width - 1, 1)
        probe_sites[:, 1] *= float(probe_h - 1) / max(height - 1, 1)
    else:
        probe_sites = out
    for _ in range(it):
        labels = _assign_nearest(probe_h, probe_w, probe_sites)
        n = int(probe_sites.shape[0])
        flat = labels.ravel()
        counts = np.bincount(flat, minlength=n).astype(np.float64)
        ys, xs = np.divmod(np.arange(probe_h * probe_w, dtype=np.int32), probe_w)
        cx = np.bincount(flat, weights=xs.astype(np.float64), minlength=n)
        cy = np.bincount(flat, weights=ys.astype(np.float64), minlength=n)
        alive = counts > 0
        cx[alive] /= counts[alive]
        cy[alive] /= counts[alive]
        probe_sites[n_anchor:, 0] = np.clip(cx[n_anchor:], 0.0, float(probe_w - 1))
        probe_sites[n_anchor:, 1] = np.clip(cy[n_anchor:], 0.0, float(probe_h - 1))
        # empty free cells: leave at last position (already clipped)
    if probe_h != height or probe_w != width:
        out[:, 0] = probe_sites[:, 0] * float(width - 1) / max(probe_w - 1, 1)
        out[:, 1] = probe_sites[:, 1] * float(height - 1) / max(probe_h - 1, 1)
    else:
        out = probe_sites
    out[:, 0] = np.clip(out[:, 0], 0.0, float(width - 1))
    out[:, 1] = np.clip(out[:, 1], 0.0, float(height - 1))
    return out


def _fill_cell_means(arr: np.ndarray, labels: np.ndarray, sites: np.ndarray) -> np.ndarray:
    """Mean color per cell via bincount — sample the seed pixel if a cell is empty."""
    n = int(sites.shape[0])
    height, width = int(arr.shape[0]), int(arr.shape[1])
    sx = np.clip(np.rint(sites[:, 0]), 0, width - 1).astype(np.int32)
    sy = np.clip(np.rint(sites[:, 1]), 0, height - 1).astype(np.int32)
    flat = labels.ravel()
    counts = np.bincount(flat, minlength=n).astype(np.float64)
    if arr.ndim == 2:
        sums = np.bincount(
            flat, weights=arr.ravel().astype(np.float64), minlength=n
        )
        means = np.zeros(n, dtype=np.float64)
        alive = counts > 0
        means[alive] = sums[alive] / counts[alive]
        means[~alive] = arr[sy[~alive], sx[~alive]].astype(np.float64)
        out = means[labels]
        return _cast_like(out, arr.dtype)
    ch = int(arr.shape[2])
    means = np.zeros((n, ch), dtype=np.float64)
    for c in range(ch):
        sums = np.bincount(
            flat, weights=arr[..., c].ravel().astype(np.float64), minlength=n
        )
        alive = counts > 0
        means[alive, c] = sums[alive] / counts[alive]
        means[~alive, c] = arr[sy[~alive], sx[~alive], c].astype(np.float64)
    out = means[labels]
    return _cast_like(out, arr.dtype)


def _voronoi_mosaic(
    arr: np.ndarray,
    h_side: str,
    v_side: str,
    tiles: int,
    lloyd: int,
    *,
    nearest: bool,
) -> np.ndarray:
    """Detail mosaic on a work grid, nearest-upsampled to the input H×W."""
    src_h, src_w = int(arr.shape[0]), int(arr.shape[1])
    work = _fit_work(arr, _VORONOI_WORK_EDGE)
    wh, ww = int(work.shape[0]), int(work.shape[1])
    rng = np.random.default_rng(0)
    sites, n_anchor = _place_sites(work, h_side, v_side, tiles, rng)
    sites = _lloyd_relax(wh, ww, sites, n_anchor, lloyd)
    labels = _assign_nearest(wh, ww, sites)
    if nearest:
        sx = np.clip(np.rint(sites[:, 0]), 0, ww - 1).astype(np.int32)
        sy = np.clip(np.rint(sites[:, 1]), 0, wh - 1).astype(np.int32)
        seed_vals = work[sy, sx]
        mosaic = seed_vals[labels]
    else:
        mosaic = _fill_cell_means(work, labels, sites)
    return _resize_hw(mosaic, src_h, src_w, nearest=True)


# ---------------------------------------------------------------------------
# Public apply
# ---------------------------------------------------------------------------

def tessellate_array(
    arr: np.ndarray,
    h_side: str | None = SIDE_OFF,
    v_side: str | None = SIDE_OFF,
    built: object = False,
    *,
    nearest: bool = False,
    mode: str | None = MODE_DEFAULT,
    tiles: object = TILES_DEFAULT,
    lloyd: object = LLOYD_DEFAULT,
) -> np.ndarray:
    """Seamless-tile a H×W or H×W×C array. ``nearest`` keeps labels as integers."""
    h_side = normalize_h_side(h_side)
    v_side = normalize_v_side(v_side)
    mode = normalize_tess_mode(mode)
    if is_identity_tessellate(h_side, v_side, built, mode=mode):
        return arr
    if mode == MODE_MESH:
        return _mesh_warp_array(arr, h_side, v_side, 1.0, nearest=nearest)
    if mode == MODE_VORONOI:
        out = _voronoi_mosaic(
            arr, h_side, v_side, clamp_tiles(tiles), clamp_lloyd(lloyd), nearest=nearest
        )
        return _hilbert_wrap_if_needed(out, h_side, v_side, nearest=nearest)
    if mode == MODE_TESSELLATE:
        return _hilbert_wrap_if_needed(arr, h_side, v_side, nearest=nearest)
    return _tile_wrap_if_needed(arr, h_side, v_side, nearest=nearest)


def apply_tessellate(
    image: Image.Image,
    h_side: str | None = SIDE_OFF,
    v_side: str | None = SIDE_OFF,
    built: object = False,
    mode: str | None = MODE_DEFAULT,
    tiles: object = TILES_DEFAULT,
    lloyd: object = LLOYD_DEFAULT,
) -> Image.Image:
    """Seamless-tile ``image``; output size always equals input size.

    Does not flatten lighting — that is ``apply_normalize_lighting``, a
    separate pipeline step. Tile mode wraps by repeating the detected
    motif, then inpaints the wrap seam (LaMa when cached). Tessellation
    Hilbert-diffuses toward the opposite side. Already-matching /
    already-periodic edges are a no-op.
    """
    h_side = normalize_h_side(h_side)
    v_side = normalize_v_side(v_side)
    mode = normalize_tess_mode(mode)
    if is_identity_tessellate(h_side, v_side, built, mode=mode):
        return image
    src_w, src_h = image.size
    arr = np.asarray(image)
    out = tessellate_array(
        arr,
        h_side,
        v_side,
        True,
        nearest=False,
        mode=mode,
        tiles=tiles,
        lloyd=lloyd,
    )
    if out is arr:
        return image
    if int(out.shape[0]) != src_h or int(out.shape[1]) != src_w:
        out = _fit_frame_by_wrap(out, src_h, src_w, h_side, v_side)
        out = _pin_wrap_edges(out, h_side, v_side)
    return Image.fromarray(out, mode=image.mode)


def apply_crop_lighting_tessellate(
    image: Image.Image,
    crop_x: float = 0.0,
    crop_y: float = 0.0,
    crop_zoom: float = 1.0,
    h_side: str | None = SIDE_OFF,
    v_side: str | None = SIDE_OFF,
    built: object = False,
    *,
    mode: str | None = MODE_DEFAULT,
    tiles: object = TILES_DEFAULT,
    lloyd: object = LLOYD_DEFAULT,
    normalize_lighting: object = False,
    src_size: tuple[int, int] | None = None,
) -> Image.Image:
    """Crop → optional lighting flatten → tessellate.

    Flatten removes the studio bowl so 3×3 / Offset tiles stay even.
    Darks/Lights remain a separate grade and are not written here.
    """
    image = apply_crop(image, crop_x, crop_y, crop_zoom, src_size=src_size)
    if coerce_normalize_lighting(normalize_lighting):
        image = apply_normalize_lighting(image)
    return apply_tessellate(
        image,
        h_side,
        v_side,
        built,
        mode=mode,
        tiles=tiles,
        lloyd=lloyd,
    )
