# -*- coding: utf-8 -*-
"""
wallpaper_recolor.color.color_ranges
------------------------------
Subdivide an image into N ranges and remap each range to a replacement color.

Default split is **color closeness**: k-means (or a fixed palette) in CIE Lab
so similar hues cluster together — not Rec. 709 lightness bands. Adding a
range inserts a new cluster and shifts coverage; existing match-from /
change-to swatches stay (Pantone change-to survives). Histogram splits:
Rec. 709 luma, or CIE Lab L* / a* / b* bins (equal steps or even pixels).

Two composites:
- Texture / grain: Color/Luminosity — original L* (weave / shadows) plus the
  picked color's a*, b* (default save). texture_strength mixes that grain
  against flat fills (0 = exact solids, 1 = full original luminosity).
- Exact master: solid replacement_rgb through non-overlapping masks

Tone (darks / lights / brightness / Lights RGB) lives on the map as −1…+1
and is applied in layers after the remap so preview and save stay in lockstep.

HSL keep-L remains available via preserve_texture=True on the paint helper.

Class references (code + name only):
- CAP3321C Data Wrangling
- CAP4631C Machine Learning
- CAP4633C Machine Learning 2
"""

from __future__ import annotations  # tuple[int, int, int] hints on 3.9-style runtimes

from collections.abc import Sequence
from dataclasses import dataclass, field
import math

import numpy as np
from PIL import Image

from wallpaper_recolor.color.color_math import (
    hls_array_to_rgb,
    hsl_lightness,
    lab_tuple_to_rgb,
    rgb_to_hsl,
    rgb_to_lab_array,
    rgb_tuple_to_lab,
)

# Rec. 709 luma weights — G dominates because the eye is most sensitive there
LUMA_R = 0.2126
LUMA_G = 0.7152
LUMA_B = 0.0722

# Split methods passed to build_range_map (Range by: maps onto these)
SPLIT_COLOR_CLOSENESS = "color_closeness"  # k-means / nearest Lab (default)
SPLIT_EQUAL_LIGHTNESS = "equal_lightness"  # Rec. 709 even steps black → white
SPLIT_EQUAL_PIXELS = "equal_pixels"  # Rec. 709 even pixel counts (percentiles)
SPLIT_LAB_L_EQUAL = "lab_l_equal"
SPLIT_LAB_L_PIXELS = "lab_l_pixels"
SPLIT_LAB_A_EQUAL = "lab_a_equal"
SPLIT_LAB_A_PIXELS = "lab_a_pixels"
SPLIT_LAB_B_EQUAL = "lab_b_equal"
SPLIT_LAB_B_PIXELS = "lab_b_pixels"
SPLIT_LAB_C_EQUAL = "lab_c_equal"
SPLIT_LAB_C_PIXELS = "lab_c_pixels"

SPLIT_COLOR_CLOSENESS_LABEL = "Color closeness"
SPLIT_EQUAL_LIGHTNESS_LABEL = "Equal lightness"
SPLIT_EQUAL_PIXELS_LABEL = "Even pixel split"

# Primary Range by: dropdown
RANGE_BY_COLOR_LABEL = "Color closeness"  # k-means / nearest Lab
RANGE_BY_LUMA_LABEL = "Luma (Rec. 709)"  # light / mid / dark Rec. 709 bands
RANGE_BY_LAB_L_LABEL = "L*"
RANGE_BY_LAB_A_LABEL = "a* (green–red)"  # CIELAB a* opponent axis
RANGE_BY_LAB_B_LABEL = "b* (blue–yellow)"  # CIELAB b* opponent axis
RANGE_BY_LAB_C_LABEL = "C* (chroma)"  # C*ab = hypot(a*, b*) — 45° in the a*–b* plane

# Lab-axis histogram splits → channel index (3 = C* chroma, not a Lab slice)
_LAB_C_CHANNEL = 3
_CSTAR_MAX = 180.0  # CIE C*ab span for sRGB (0 = gray)
_LAB_CHANNEL_FOR = {
    SPLIT_LAB_L_EQUAL: 0,
    SPLIT_LAB_L_PIXELS: 0,
    SPLIT_LAB_A_EQUAL: 1,
    SPLIT_LAB_A_PIXELS: 1,
    SPLIT_LAB_B_EQUAL: 2,
    SPLIT_LAB_B_PIXELS: 2,
    SPLIT_LAB_C_EQUAL: _LAB_C_CHANNEL,
    SPLIT_LAB_C_PIXELS: _LAB_C_CHANNEL,
}
_PIXEL_BIN_METHODS = frozenset(
    {
        SPLIT_EQUAL_PIXELS,
        SPLIT_LAB_L_PIXELS,
        SPLIT_LAB_A_PIXELS,
        SPLIT_LAB_B_PIXELS,
        SPLIT_LAB_C_PIXELS,
    }
)
ALLOWED_SPLIT_METHODS = frozenset(
    {
        SPLIT_COLOR_CLOSENESS,
        SPLIT_EQUAL_LIGHTNESS,
        SPLIT_EQUAL_PIXELS,
        SPLIT_LAB_L_EQUAL,
        SPLIT_LAB_L_PIXELS,
        SPLIT_LAB_A_EQUAL,
        SPLIT_LAB_A_PIXELS,
        SPLIT_LAB_B_EQUAL,
        SPLIT_LAB_B_PIXELS,
        SPLIT_LAB_C_EQUAL,
        SPLIT_LAB_C_PIXELS,
    }
)
# Saved L* / "L split" jobs are Rec. 709 luma in the UI (duplicate of Range by Luma).
_LUMA_SPLIT_ALIASES = {
    "l_split": SPLIT_EQUAL_LIGHTNESS,
    "l": SPLIT_EQUAL_LIGHTNESS,
    "L": SPLIT_EQUAL_LIGHTNESS,
    "L*": SPLIT_EQUAL_LIGHTNESS,
    "l*": SPLIT_EQUAL_LIGHTNESS,
    "lab_l": SPLIT_EQUAL_LIGHTNESS,
    RANGE_BY_LAB_L_LABEL: SPLIT_EQUAL_LIGHTNESS,
    SPLIT_LAB_L_EQUAL: SPLIT_EQUAL_LIGHTNESS,
    SPLIT_LAB_L_PIXELS: SPLIT_EQUAL_PIXELS,
}
_CHROMA_SPLIT_ALIASES = {
    "C*": SPLIT_LAB_C_EQUAL,
    "c*": SPLIT_LAB_C_EQUAL,
    "chroma": SPLIT_LAB_C_EQUAL,
    RANGE_BY_LAB_C_LABEL: SPLIT_LAB_C_EQUAL,
}

# Color-closeness assignment: k-means from the scan (default) vs nearest palette hex
ASSIGN_KMEANS = "kmeans"
ASSIGN_PALETTE = "palette"
ASSIGN_KMEANS_LABEL = "Cluster from image"
ASSIGN_PALETTE_LABEL = "Snap to palette hexes"

# k-means on the work image (CAP4631C / CAP4633C). Subsample so rebuild stays
# snappy; Lloyd then assigns every opaque pixel to the nearest center.
KMEANS_SAMPLES = 32768
KMEANS_ITERS = 14
KMEANS_SEED = 0
# Auto-k search on import: 2…8, not MAX_RANGES (24). Silhouette on a further
# 4096-point draw — full 32k pairwise distances are O(n²).
AUTO_K_MIN = 2
AUTO_K_MAX = 8
SILHOUETTE_SAMPLES = 4096
# Tight silhouette window for the usual pick (RGB patches, empty-k ties).
_AUTO_K_SIL_NEAR = 0.02
# Analogous L* inks: silhouette often scores k=2 a bit higher even when a
# third dark/mid/light ink is real (V6-N teal linen, Δsil ≈ 0.08).
_AUTO_K_SIL_LSTAR_SLACK = 0.12
_AUTO_K_LSTAR_GAP = 10.0
_AUTO_K_REL_DROP_2_3 = 0.25
_AUTO_K_HUE_ANALOGOUS_DEG = 55.0
_AUTO_K_CHROMA_NEUTRAL = 8.0
# Full-res Lab as one array is ~2.5GB at 207MP — assign in row strips instead.
ASSIGN_STRIP_ROWS = 128
FULL_VECTOR_MAX_PIXELS = 4_000_000  # preview / work image: one numpy pass

MIN_COVERAGE = 0.03  # default floor so a range cannot vanish
MIN_COVERAGE_MAX = 0.40  # spinbox cap (0–40%)
# Texture slider 0–100% → 0–1. Default is full original luminosity (new hues,
# no gray Overlay plate). 0 is exact solids.
TEXTURE_DEFAULT_STRENGTH = 1.0


def clamp_min_coverage(value: object) -> float:
    """Coverage steal floor 0…0.40 (UI Min %)."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        n = MIN_COVERAGE
    if n != n:  # NaN
        n = MIN_COVERAGE
    return max(0.0, min(MIN_COVERAGE_MAX, n))


def canonicalize_split_method(method: str) -> str:
    """Map legacy L* / l_split aliases onto Rec. 709 luma; keep known methods."""
    key = str(method or "").strip()
    if not key:
        return SPLIT_COLOR_CLOSENESS
    mapped = _LUMA_SPLIT_ALIASES.get(key)
    if mapped is None:
        mapped = _LUMA_SPLIT_ALIASES.get(key.lower())
    if mapped is not None:
        return mapped
    chroma = _CHROMA_SPLIT_ALIASES.get(key)
    if chroma is None:
        chroma = _CHROMA_SPLIT_ALIASES.get(key.lower())
    if chroma is not None:
        return chroma
    if key in ALLOWED_SPLIT_METHODS:
        return key
    return SPLIT_COLOR_CLOSENESS


def is_color_split(method: str) -> bool:
    """True when ranges are nearest-Lab clusters, not 1D histogram bins."""
    return method == SPLIT_COLOR_CLOSENESS


def is_lab_channel_split(method: str) -> bool:
    """True when ranges are L* / a* / b* / C* histogram bins (not Rec. 709 luma)."""
    return method in _LAB_CHANNEL_FOR


def is_pixel_bin_split(method: str) -> bool:
    """True for even-pixel (percentile) bins on luma or a Lab axis."""
    return method in _PIXEL_BIN_METHODS


def split_axis_channel(method: str) -> int | None:
    """0=L*, 1=a*, 2=b*, 3=C* for Lab-axis bins; None for Rec. 709 luma or k-means."""
    return _LAB_CHANNEL_FOR.get(method)


def split_axis_caption(method: str) -> str:
    """Short axis name for captions (Rec. 709 luma stays ``L``)."""
    ch = split_axis_channel(method)
    if ch == 0:
        return "L*"
    if ch == 1:
        return "a*"
    if ch == 2:
        return "b*"
    if ch == _LAB_C_CHANNEL:
        return "C*"
    return "L"


def _lab_axis_values(lab: np.ndarray, method: str) -> np.ndarray:
    """1D samples along L*, a*, b*, or C* = hypot(a*, b*) (45° in a*–b*)."""
    ch = split_axis_channel(method)
    if ch is None:
        raise ValueError("not a Lab-axis split")
    if int(ch) == _LAB_C_CHANNEL:
        a = lab[..., 1].astype(np.float32, copy=False)
        b = lab[..., 2].astype(np.float32, copy=False)
        return np.hypot(a, b)
    return lab[..., int(ch)]


def bin_display_key(low: float, high: float, method: str) -> float:
    """0–1 key for the coverage-bar gray (luma / L* / a* / b* / C* mid-bin)."""
    mid = (float(low) + float(high)) * 0.5
    ch = split_axis_channel(method)
    if ch is None:
        return max(0.0, min(1.0, mid / 255.0))
    if ch == 0:
        return max(0.0, min(1.0, mid / 100.0))
    if ch == _LAB_C_CHANNEL:
        return max(0.0, min(1.0, mid / _CSTAR_MAX))
    return max(0.0, min(1.0, (mid + 128.0) / 255.0))


def _channel_span_defaults(method: str) -> tuple[float, float]:
    ch = split_axis_channel(method)
    if ch is None:
        return (0.0, 255.0)
    if ch == 0:
        return (0.0, 100.0)
    if ch == _LAB_C_CHANNEL:
        return (0.0, _CSTAR_MAX)
    return (-128.0, 127.0)


def is_palette_assign(mode: str) -> bool:
    """True when pixels snap to preset hexes instead of image k-means."""
    return mode == ASSIGN_PALETTE


@dataclass
class ColorRange:
    """One range: match-from (closeness target) and change-to (paint) colors.

    luma_low / luma_high are Rec. 709 or Lab-axis bin edges, or the
    cluster center's L* (repeated) for color closeness.
    weight is bin share (luma) or cluster influence (closeness Voronoi).
    """

    index: int  # 0 = first range (darkest luma, or palette / k-means order)
    luma_low: float  # inclusive lower luma (0–255) or cluster L*
    luma_high: float  # exclusive upper luma except the last band
    mean_rgb: tuple[int, int, int]  # average original color in this range
    match_rgb: tuple[int, int, int]  # top swatch — Lab closeness target
    replacement_rgb: tuple[int, int, int]  # bottom swatch — pixels become this
    pixel_count: int  # how many pixels landed in this range
    total_pixels: int  # RGB pixel count (alpha ignored)
    weight: float = 0.0  # luma/percentile span, or cluster influence (sums to 1)
    name: str = ""  # optional label (V6-N: "Dark / Garden Grove", …)
    visible: bool = True  # eye off → knock those pixels out of Result (alpha 0)

    @property
    def share(self) -> float:
        """Fraction of the image covered by this range (0–1)."""
        if self.total_pixels <= 0:
            return 0.0
        return self.pixel_count / self.total_pixels

    @property
    def luma_key(self) -> float:
        """Mid-bin Rec. 709 luma 0–1. Not coverage weight."""
        mid = (float(self.luma_low) + float(self.luma_high)) * 0.5
        return max(0.0, min(1.0, mid / 255.0))


@dataclass
class ColorRangeMap:
    """N ranges plus the per-pixel range index for one (usually work-size) image."""

    range_count: int
    split_method: str
    ranges: list[ColorRange] = field(default_factory=list)
    # edges: N+1 luma cuts used by digitize — reused on the full-res save
    edges: np.ndarray | None = None
    # centers: N×3 CIE Lab — k-means or palette targets; full-res save uses these
    centers: np.ndarray | None = None
    # labels: H×W int32, -1 = transparent / unused, 0..N-1 = range index
    labels: np.ndarray | None = None
    rgb: np.ndarray | None = None  # H×W×3 uint8 working copy
    alpha: np.ndarray | None = None  # H×W uint8 or None
    lab: np.ndarray | None = None  # work-image Lab cache (not used on 207MP save)
    # Texture slider 0–1: 0 = solid fills, 1 = keep original L* (Color blend)
    texture_strength: float = TEXTURE_DEFAULT_STRENGTH
    # Texture eye: False = skip grain (flat fills) without moving the slider
    texture_enabled: bool = True
    # Tone sliders −1…+1 (UI −100…+100). All zeros = identity after remap.
    tone_darks: float = 0.0
    tone_lights: float = 0.0
    tone_brightness: float = 0.0
    tone_contrast: float = 0.0
    tone_exposure: float = 0.0
    # Lights Reds / Greens / Blues — white-balance in highlights only.
    tone_lights_reds: float = 0.0
    tone_lights_greens: float = 0.0
    tone_lights_blues: float = 0.0
    # Print color balance (Cyan↔Red, Magenta↔Green, Yellow↔Blue) plus WB / sat.
    tone_temperature: float = 0.0
    tone_tint: float = 0.0
    tone_saturation: float = 0.0
    tone_balance_cyan: float = 0.0
    tone_balance_magenta: float = 0.0
    tone_balance_yellow: float = 0.0
    # Legacy Lights/Darks CMY — still snapshotted; UI uses tone_balance_*.
    tone_lights_cyan: float = 0.0
    tone_lights_magenta: float = 0.0
    tone_lights_yellow: float = 0.0
    tone_darks_cyan: float = 0.0
    tone_darks_magenta: float = 0.0
    tone_darks_yellow: float = 0.0
    # Histogram Start (Rec. 709 0–255 or Lab units). None = channel default low.
    bin_start: float | None = None
    # Steal / bin floor 0–1 (luma Min %). Color closeness uses this too.
    min_coverage: float = MIN_COVERAGE

    def set_replacement(self, index: int, rgb: tuple[int, int, int]) -> None:
        """Store the change-to RGB on range ``index`` (bottom swatch)."""
        self.ranges[index].replacement_rgb = rgb

    def set_match(self, index: int, rgb: tuple[int, int, int]) -> None:
        """Store the match-from RGB and sync that Lab center (top swatch)."""
        self.ranges[index].match_rgb = rgb
        if self.centers is not None and 0 <= index < int(self.centers.shape[0]):
            self.centers[index] = rgb_tuple_to_lab(rgb)

    def weights(self) -> list[float]:
        """Coverage / cluster-influence weights in range order."""
        return [band.weight for band in self.ranges]

    def match_colors(self) -> list[tuple[int, int, int]]:
        return [band.match_rgb for band in self.ranges]

    def replacement_colors(self) -> list[tuple[int, int, int]]:
        return [band.replacement_rgb for band in self.ranges]


def rgb_array(image: Image.Image, *, copy: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (H×W×3 uint8 RGB, optional H×W uint8 alpha).

    ``copy=False`` keeps a view of the Pillow buffer — save/export must not
    mutate it (each extra 207MP RGB copy is ~600MB).
    """
    if image.mode == "RGBA":
        arr = np.asarray(image, dtype=np.uint8)
        rgb, alpha = arr[..., :3], arr[..., 3]
        if copy:
            return rgb.copy(), alpha.copy()
        return rgb, alpha
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if copy:
        return rgb.copy(), None
    return rgb, None


def luma_channel(rgb: np.ndarray) -> np.ndarray:
    """Rec. 709 luma as float32 in 0–255 (same scale as 8-bit channels)."""
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    return LUMA_R * r + LUMA_G * g + LUMA_B * b


def resolved_bin_start(split_method: str, bin_start: float | None) -> float:
    """User Start clamped to the channel's legal span (luma 0–255, L* 0–100, a*/b* ±128)."""
    lo_d, hi_d = _channel_span_defaults(split_method)
    if bin_start is None:
        return lo_d
    try:
        n = float(bin_start)
    except (TypeError, ValueError):
        return lo_d
    if n != n:
        return lo_d
    return max(lo_d, min(hi_d, n))


def _bin_edges(
    values: np.ndarray,
    n: int,
    split_method: str,
    valid: np.ndarray,
    *,
    start: float | None = None,
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Return N+1 edges for digitize on a 1D channel (luma or Lab axis).

    First edge is max(user Start, image min) when Start is at/below the min;
    if Start is above the image min, bins run Start → max and darker/lower
    samples are unlabeled (see ``_label_pixels``).
    """
    if n < 1:
        raise ValueError("range_count must be at least 1")
    lo_d, hi_d = _channel_span_defaults(split_method)
    user = resolved_bin_start(split_method, start)
    samples = values[valid]
    if samples.size == 0:
        hi = hi_d if hi_d > user else user + 1.0
        return np.linspace(user, hi, n + 1)

    img_min = float(samples.min())
    img_max = float(samples.max())
    lo = max(user, img_min)
    hi = img_max
    in_span = samples[samples >= lo]
    if weights is not None and len(list(weights)) == n:
        w = np.asarray(list(weights), dtype=np.float64)
        w = np.clip(w, 1e-6, None)
        w = w / w.sum()
        cum = np.concatenate(([0.0], np.cumsum(w)))
        cum[-1] = 1.0
        if is_pixel_bin_split(split_method) and in_span.size > 0:
            percents = np.clip(cum * 100.0, 0.0, 100.0)
            edges = np.percentile(in_span, percents).astype(np.float64)
            edges[0] = lo
            edges[-1] = float(in_span.max())
            return edges
        span_hi = hi if hi > lo else lo + 1e-3
        return lo + cum * (span_hi - lo)

    if in_span.size == 0:
        span_hi = lo + (1.0 if split_axis_channel(split_method) is None else 0.25)
        return np.linspace(lo, span_hi, n + 1)

    if is_pixel_bin_split(split_method):
        percents = np.linspace(0.0, 100.0, n + 1)
        edges = np.percentile(in_span, percents).astype(np.float64)
        edges = np.unique(np.round(edges, 6))
        if edges.size < 2:
            return np.linspace(lo, float(in_span.max()), n + 1)
        edges[0] = lo
        edges[-1] = float(in_span.max())
        return edges

    if hi - lo < 1e-3:
        pad = 0.5 if split_axis_channel(split_method) is None else 0.25
        return np.array([lo - pad, hi + pad], dtype=np.float64)
    return np.linspace(lo, hi, n + 1)


def _valid_mask(alpha: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    """Opaque pixels (alpha > 0), or all-true when the image has no alpha."""
    if alpha is None:
        return np.ones(shape, dtype=bool)
    return alpha > 0


def _label_pixels(values: np.ndarray, edges: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """digitize a 1D channel into 0..N-1; transparent or below Start get -1."""
    labels = np.digitize(values, edges[1:-1], right=True).astype(np.int32)
    labels[~valid] = -1
    if edges.size >= 1:
        labels[valid & (values < float(edges[0]))] = -1
    return labels


def _kmeans_pp_init(samples: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeds (CAP4633C): new centers prefer points far from existing ones."""
    m = samples.shape[0]
    centers = np.empty((n, samples.shape[1]), dtype=np.float64)
    centers[0] = samples[int(rng.integers(0, m))]
    closest = np.full(m, np.inf, dtype=np.float64)
    for j in range(1, n):
        delta = samples - centers[j - 1]
        d2 = np.einsum("ij,ij->i", delta, delta)
        closest = np.minimum(closest, d2)
        total = float(closest.sum())
        if total <= 1e-12:
            extra = rng.integers(0, m, size=n - j)
            centers[j:] = samples[extra]
            break
        pick = rng.random() * total
        idx = int(np.searchsorted(np.cumsum(closest), pick, side="right"))
        centers[j] = samples[min(idx, m - 1)]
    return centers


def _kmeans_centers_from_image(
    rgb: np.ndarray,
    valid: np.ndarray,
    n: int,
    lab: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """N Lab centers (darkest L* first) plus a Lab cube for the work image.

    k-means++ then Lloyd (CAP4631C / CAP4633C). Same seed as build_range_map
    so Reset snaps back to the N most matching colors.
    """
    if lab is None:
        lab = rgb_to_lab_array(rgb)
    n = max(1, int(n))
    samples = lab[valid]
    if samples.size == 0:
        return np.zeros((n, 3), dtype=np.float32), lab
    rng = np.random.default_rng(KMEANS_SEED)
    pts = samples.reshape(-1, 3)
    if pts.shape[0] > KMEANS_SAMPLES:
        idx = rng.choice(pts.shape[0], KMEANS_SAMPLES, replace=False)
        pts = pts[idx]
    n = min(n, int(pts.shape[0]))
    centers = _kmeans_lab(pts.astype(np.float64), n, rng)
    order = np.argsort(centers[:, 0], kind="stable")
    return centers[order], lab


def _kmeans_lab(samples: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Lloyd k-means on M×3 Lab samples → n×3 centers (CAP4631C)."""
    m = samples.shape[0]
    n = max(1, min(int(n), m))
    centers = _kmeans_pp_init(samples, n, rng)
    labels = np.zeros(m, dtype=np.int32)
    for _ in range(KMEANS_ITERS):
        # M×N distances are fine here (M ≤ KMEANS_SAMPLES)
        delta = samples[:, None, :] - centers[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", delta, delta)
        labels = d2.argmin(axis=1).astype(np.int32)
        moved = False
        for i in range(n):
            members = samples[labels == i]
            if members.shape[0] == 0:
                # Empty cluster: steal the sample farthest from all centers
                far = d2.min(axis=1).argmax()
                new_c = samples[far]
            else:
                new_c = members.mean(axis=0)
            if np.abs(new_c - centers[i]).max() > 1e-4:
                moved = True
            centers[i] = new_c
        if not moved:
            break
    return centers.astype(np.float32)


def _kmeans_subsample(lab: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Same Lab draw ``_kmeans_centers_from_image`` uses (KMEANS_SEED / KMEANS_SAMPLES)."""
    samples = lab[valid]
    if samples.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    rng = np.random.default_rng(KMEANS_SEED)
    pts = samples.reshape(-1, 3).astype(np.float64, copy=False)
    if pts.shape[0] > KMEANS_SAMPLES:
        idx = rng.choice(pts.shape[0], KMEANS_SAMPLES, replace=False)
        pts = pts[idx]
    return pts


def _cluster_inertia(samples: np.ndarray, centers: np.ndarray) -> tuple[float, np.ndarray]:
    """Within-cluster sum of squares and nearest-center labels."""
    delta = samples[:, None, :] - centers[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", delta, delta)
    labels = d2.argmin(axis=1).astype(np.int32)
    inertia = float(d2[np.arange(samples.shape[0]), labels].sum())
    return inertia, labels


def _mean_silhouette(samples: np.ndarray, labels: np.ndarray, n: int) -> float:
    """Mean silhouette on M×3 samples (vectorized per cluster; M is small)."""
    m = int(samples.shape[0])
    n = int(n)
    if m < 2 or n < 2:
        return 0.0
    mean_d = np.zeros((m, n), dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    for c in range(n):
        members = samples[labels == c]
        mc = int(members.shape[0])
        counts[c] = mc
        if mc == 0:
            mean_d[:, c] = np.inf
            continue
        delta = samples[:, None, :] - members[None, :, :]
        d = np.sqrt(np.maximum(np.einsum("ijk,ijk->ij", delta, delta), 0.0))
        mean_d[:, c] = d.mean(axis=1)
    a = np.zeros(m, dtype=np.float64)
    b = np.full(m, np.inf, dtype=np.float64)
    for c in range(n):
        mask = labels == c
        if not np.any(mask):
            continue
        mc = counts[c]
        if mc <= 1.0:
            a[mask] = 0.0
        else:
            a[mask] = mean_d[mask, c] * mc / (mc - 1.0)
        others = [j for j in range(n) if j != c and counts[j] > 0]
        if others:
            b[mask] = mean_d[mask][:, others].min(axis=1)
    finite = np.isfinite(b) & (np.maximum(a, b) > 1e-12)
    if not np.any(finite):
        return 0.0
    s = (b[finite] - a[finite]) / np.maximum(a[finite], b[finite])
    return float(np.mean(s))


def _elbow_k(ks: np.ndarray, inertias: np.ndarray) -> int:
    """kneedle-style: k farthest from the line between first and last inertia."""
    ks = np.asarray(ks, dtype=np.float64)
    inertias = np.asarray(inertias, dtype=np.float64)
    if ks.size == 1:
        return int(ks[0])
    span_k = float(ks[-1] - ks[0]) or 1.0
    span_i = float(inertias[0] - inertias[-1])
    if abs(span_i) < 1e-12:
        return int(ks[0])
    x = (ks - ks[0]) / span_k
    y = (inertias - inertias[-1]) / span_i
    # Line from (0, 1) to (1, 0): y = 1 - x
    dist = np.abs(y - (1.0 - x)) / math.sqrt(2.0)
    return int(ks[int(dist.argmax())])


def _circular_hue_spread_deg(
    centers: np.ndarray,
    chroma_min: float = _AUTO_K_CHROMA_NEUTRAL,
) -> float:
    """Smallest arc (degrees) covering Lab hues of chromatic centers."""
    a = centers[:, 1]
    b = centers[:, 2]
    chroma = np.hypot(a, b)
    keep = chroma >= chroma_min
    if int(np.count_nonzero(keep)) < 2:
        return 0.0
    hues = np.sort(np.degrees(np.arctan2(b[keep], a[keep])))
    diffs = np.diff(hues)
    wrap = float(hues[0] + 360.0 - hues[-1])
    max_gap = wrap if diffs.size == 0 else max(float(diffs.max()), wrap)
    return float(360.0 - max_gap)


def _lstar_ink_stats(
    centers: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Min L* gap between centers and smallest cluster share."""
    n = int(centers.shape[0])
    shares = np.bincount(labels, minlength=n).astype(np.float64)
    shares /= max(int(labels.size), 1)
    lstar = np.sort(centers[:, 0].astype(np.float64))
    gaps = np.diff(lstar)
    min_gap = float(gaps.min()) if gaps.size else 0.0
    min_share = float(shares.min()) if shares.size else 0.0
    return min_gap, min_share


def choose_kmeans_k(
    image: Image.Image,
    *,
    k_min: int = AUTO_K_MIN,
    k_max: int = AUTO_K_MAX,
) -> int:
    """Best cluster count from silhouette + inertia elbow (Generic import).

    Fits k-means on the usual Lab subsample (KMEANS_SAMPLES, KMEANS_SEED).
    Silhouette is scored on a further SILHOUETTE_SAMPLES draw. Prefers the
    highest silhouette; among scores within 0.02 of that peak, pick the k
    closest to the inertia elbow (smaller k on a tie) so over-clustering
    loses when silhouettes are close.

    Analogous / near-neutral scans are different: silhouette often prefers
    k=2 even when a third ink is a real L* split (dark / mid / highlight).
    If k=2 wins that way, promote to 3 when inertia still drops a lot 2→3,
    L* centers are well separated, and hues stay in a tight analog (do not
    promote on to 4–8 linen grain).

    Search is ``k_min``…``k_max`` (default 2…8), not the UI max of 24.
    """
    rgb, alpha = rgb_array(image, copy=False)
    valid = _valid_mask(alpha, rgb.shape[:2])
    lab = rgb_to_lab_array(rgb)
    pts = _kmeans_subsample(lab, valid)
    m = int(pts.shape[0])
    if m < 2:
        return max(1, m)
    k_min = max(2, int(k_min))
    k_max = max(k_min, min(int(k_max), m))
    rng_sil = np.random.default_rng(KMEANS_SEED + 1)
    if m > SILHOUETTE_SAMPLES:
        sil_idx = rng_sil.choice(m, SILHOUETTE_SAMPLES, replace=False)
        sil_pts = pts[sil_idx]
    else:
        sil_pts = pts
    ks = np.arange(k_min, k_max + 1, dtype=np.int32)
    sils = np.zeros(ks.size, dtype=np.float64)
    inertias = np.zeros(ks.size, dtype=np.float64)
    lstar_gap_k3 = 0.0
    min_share_k3 = 0.0
    hue_spread_k3 = 0.0
    med_chroma_k3 = 0.0
    for i, k in enumerate(ks):
        rng = np.random.default_rng(KMEANS_SEED)
        centers = _kmeans_lab(pts, int(k), rng)
        n = int(centers.shape[0])
        inertias[i], labels = _cluster_inertia(pts, centers)
        _sil_inertia, sil_labels = _cluster_inertia(sil_pts, centers)
        sils[i] = _mean_silhouette(sil_pts, sil_labels, n)
        if int(k) == 3:
            lstar_gap_k3, min_share_k3 = _lstar_ink_stats(centers, labels)
            hue_spread_k3 = _circular_hue_spread_deg(centers)
            med_chroma_k3 = float(
                np.median(np.hypot(centers[:, 1], centers[:, 2]))
            )
    best_sil = float(sils.max())
    close = sils >= (best_sil - _AUTO_K_SIL_NEAR)
    candidates = ks[close]
    elbow = _elbow_k(ks, inertias)
    order = np.lexsort((candidates, np.abs(candidates - elbow)))
    chosen = int(candidates[order[0]])

    # Wallpaper ink count: three L* bands of one analogous hue beat two
    # when silhouette's 0.02 window + smaller-k tie-break would lock onto 2.
    idx = {int(k): i for i, k in enumerate(ks)}
    if (
        chosen == 2
        and 2 in idx
        and 3 in idx
        and sils[idx[3]] >= sils[idx[2]] - _AUTO_K_SIL_LSTAR_SLACK
        and inertias[idx[2]] > 1e-12
        and ((inertias[idx[2]] - inertias[idx[3]]) / inertias[idx[2]])
        >= _AUTO_K_REL_DROP_2_3
        and lstar_gap_k3 >= _AUTO_K_LSTAR_GAP
        and min_share_k3 >= MIN_COVERAGE
        and (
            hue_spread_k3 <= _AUTO_K_HUE_ANALOGOUS_DEG
            or med_chroma_k3 < _AUTO_K_CHROMA_NEUTRAL
        )
    ):
        return 3
    return chosen


def _assign_lab_block(
    lab: np.ndarray,
    centers: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """1-NN in Lab over a H×W×3 block. ``scale[i]`` multiplies distance to center i.

    Loop is over N centers (≤24), not pixels — each step is numpy-wide.
    Distance / weight (scale = 1/weight) lets a cluster steal more pixels.
    """
    h, w = lab.shape[:2]
    n = int(centers.shape[0])
    labels = np.zeros((h, w), dtype=np.int32)
    best = None
    for i in range(n):
        delta = lab - centers[i]
        d2 = delta[..., 0] * delta[..., 0] + delta[..., 1] * delta[..., 1] + delta[..., 2] * delta[..., 2]
        s = float(scale[i])
        if s != 1.0:
            d2 = d2 * (s * s)
        if best is None:
            best = d2
        else:
            closer = d2 < best
            labels[closer] = i
            best = np.where(closer, d2, best)
    return labels


def assign_nearest_lab(
    lab: np.ndarray | None,
    rgb: np.ndarray,
    centers: np.ndarray,
    valid: np.ndarray,
    weights: Sequence[float] | None = None,
    *,
    freeze_pair: tuple[int, int] | None = None,
    prev_labels: np.ndarray | None = None,
) -> np.ndarray:
    """Label each opaque pixel with the nearest Lab center (optionally weighted).

    ``freeze_pair``: only those two clusters exchange pixels (other labels stay).
    Used after a pair-steal so a third cluster cannot grow from 1/weight Voronoi.
    """
    n = int(centers.shape[0])
    scale = np.ones(n, dtype=np.float64)
    if weights is not None:
        w = np.clip(np.asarray(list(weights), dtype=np.float64), 1e-4, None)
        w = w / w.sum()
        scale = 1.0 / w
    h, w_px = rgb.shape[:2]
    pixels = h * w_px
    pair = None
    if freeze_pair is not None and prev_labels is not None and prev_labels.shape == (h, w_px):
        a, b = int(freeze_pair[0]), int(freeze_pair[1])
        if a != b and 0 <= a < n and 0 <= b < n:
            pair = (a, b)

    if pixels <= FULL_VECTOR_MAX_PIXELS:
        block = lab if lab is not None else rgb_to_lab_array(rgb)
        if pair is None:
            labels = _assign_lab_block(block, centers, scale)
        else:
            labels = _reassign_lab_pair(block, centers, scale, prev_labels, pair)
        labels[~valid] = -1
        return labels

    labels = np.full((h, w_px), -1, dtype=np.int32)
    strip = ASSIGN_STRIP_ROWS
    for y0 in range(0, h, strip):
        y1 = min(h, y0 + strip)
        block = rgb_to_lab_array(rgb[y0:y1])
        if pair is None:
            sl = _assign_lab_block(block, centers, scale)
        else:
            sl = _reassign_lab_pair(block, centers, scale, prev_labels[y0:y1], pair)
        sl[~valid[y0:y1]] = -1
        labels[y0:y1] = sl
    return labels


def _reassign_lab_pair(
    lab: np.ndarray,
    centers: np.ndarray,
    scale: np.ndarray,
    prev_labels: np.ndarray,
    pair: tuple[int, int],
) -> np.ndarray:
    """Weighted 1-NN between ``pair`` only, on pixels already labeled as either."""
    a, b = pair
    labels = np.array(prev_labels, copy=True, dtype=np.int32)
    mask = (labels == a) | (labels == b)
    if not bool(mask.any()):
        return labels

    def _d2(index: int) -> np.ndarray:
        delta = lab - centers[index]
        d2 = (
            delta[..., 0] * delta[..., 0]
            + delta[..., 1] * delta[..., 1]
            + delta[..., 2] * delta[..., 2]
        )
        s = float(scale[index])
        if s != 1.0:
            d2 = d2 * (s * s)
        return d2

    pick_b = _d2(b) < _d2(a)
    labels[mask] = np.where(pick_b[mask], b, a)
    return labels


def _ranges_from_labels(
    rgb: np.ndarray,
    labels: np.ndarray,
    n: int,
    valid: np.ndarray,
    weights: np.ndarray,
    centers: np.ndarray | None,
    edges: np.ndarray | None,
) -> list[ColorRange]:
    """Fill ColorRange stats from a label field (work image)."""
    total = int(valid.sum())
    ranges: list[ColorRange] = []
    for i in range(n):
        mask = labels == i
        count = int(mask.sum())
        if count > 0:
            mean = rgb[mask].mean(axis=0)
            mean_rgb = (int(round(mean[0])), int(round(mean[1])), int(round(mean[2])))
        else:
            mean_rgb = (128, 128, 128)
        if edges is not None:
            low = float(edges[i])
            high = float(edges[i + 1])
        elif centers is not None:
            low = high = float(centers[i, 0])
        else:
            low = high = 0.0
        # Cluster center → match-from; identity recolor until they pick a change-to
        if centers is not None:
            match_rgb = lab_tuple_to_rgb(centers[i])
        else:
            match_rgb = mean_rgb
        ranges.append(
            ColorRange(
                index=i,
                luma_low=low,
                luma_high=high,
                mean_rgb=mean_rgb,
                match_rgb=match_rgb,
                replacement_rgb=match_rgb,
                pixel_count=count,
                total_pixels=total,
                weight=float(weights[i]) if i < weights.size else 1.0 / n,
            )
        )
    return ranges


def build_range_map(
    image: Image.Image,
    range_count: int,
    split_method: str = SPLIT_COLOR_CLOSENESS,
    *,
    palette_rgb: Sequence[tuple[int, int, int]] | None = None,
    bin_start: float | None = None,
    min_coverage: float | None = None,
) -> ColorRangeMap:
    """Assign every opaque pixel to one of ``range_count`` ranges.

    Color closeness (default): k-means in Lab, or nearest of ``palette_rgb``
    when a preset supplies palette targets (V6-N greens as match-from).
    Match-from and change-to start as the same N colors (identity recolor).
    Histogram splits: Rec. 709 luma or CIE Lab L* / a* / b* (equal or pixels).
    """
    rgb, alpha = rgb_array(image)
    valid = _valid_mask(alpha, rgb.shape[:2])
    mc = clamp_min_coverage(MIN_COVERAGE if min_coverage is None else min_coverage)

    if is_color_split(split_method):
        if palette_rgb:
            centers = np.stack([rgb_tuple_to_lab(c) for c in palette_rgb]).astype(np.float32)
            n = int(centers.shape[0])
            lab = rgb_to_lab_array(rgb)
        else:
            n = max(1, int(range_count))
            centers, lab = _kmeans_centers_from_image(rgb, valid, n)
            n = int(centers.shape[0])
        weights = np.full(n, 1.0 / n, dtype=np.float64)
        labels = assign_nearest_lab(lab, rgb, centers, valid, weights)
        ranges = _ranges_from_labels(rgb, labels, n, valid, weights, centers, None)
        if palette_rgb:
            # Keep the preset hexes (Lab round-trip would drift a few codes)
            for i, spec in enumerate(palette_rgb):
                if i >= len(ranges):
                    break
                ranges[i].match_rgb = spec
                ranges[i].replacement_rgb = spec
        return ColorRangeMap(
            range_count=n,
            split_method=SPLIT_COLOR_CLOSENESS,
            ranges=ranges,
            edges=None,
            centers=centers,
            labels=labels,
            rgb=rgb,
            alpha=alpha,
            lab=lab,
            bin_start=bin_start,
            min_coverage=mc,
        )

    lab = None
    if is_lab_channel_split(split_method):
        lab = rgb_to_lab_array(rgb)
        values = _lab_axis_values(lab, split_method)
    else:
        values = luma_channel(rgb)
    edges = _bin_edges(values, range_count, split_method, valid, start=bin_start)
    labels = _label_pixels(values, edges, valid)
    actual_n = edges.size - 1  # may be < range_count on flat images
    weights = np.full(actual_n, 1.0 / actual_n, dtype=np.float64)
    ranges = _ranges_from_labels(rgb, labels, actual_n, valid, weights, None, edges)
    return ColorRangeMap(
        range_count=actual_n,
        split_method=split_method,
        ranges=ranges,
        edges=edges,
        centers=None,
        labels=labels,
        rgb=rgb,
        alpha=alpha,
        lab=lab,
        bin_start=bin_start,
        min_coverage=mc,
    )


def _histogram_channel(range_map: ColorRangeMap, rgb: np.ndarray) -> np.ndarray:
    """1D samples for luma or Lab-axis bins (caches Lab on the work map)."""
    ch = split_axis_channel(range_map.split_method)
    if ch is None:
        return luma_channel(rgb)
    lab = range_map.lab
    if lab is None:
        lab = rgb_to_lab_array(rgb)
        range_map.lab = lab
    return _lab_axis_values(lab, range_map.split_method)


def apply_weights(
    range_map: ColorRangeMap,
    weights: Sequence[float],
    *,
    freeze_pair: tuple[int, int] | None = None,
) -> None:
    """Rebuild membership from coverage weights; keep replacement colors.

    Luma / Lab-axis splits: re-cut histogram bins (CAP3321C weighted bins).
    Color closeness: scale Lab distance by 1/weight so a heavier cluster
    steals more pixels (same centers — preview/save stay in sync).
    ``freeze_pair`` limits k-means reassignment to those two labels so a
    third cluster cannot grow from a pair-steal.
    """
    if range_map.rgb is None:
        raise ValueError("ColorRangeMap has no pixel data")
    n = len(range_map.ranges)
    w = np.asarray(list(weights), dtype=np.float64)
    if w.size != n:
        raise ValueError(f"expected {n} weights, got {w.size}")
    w = np.clip(w, 1e-4, None)
    floor = clamp_min_coverage(getattr(range_map, "min_coverage", MIN_COVERAGE))
    clip_lo = floor if floor > 0 else 1e-6
    w = np.clip(w, clip_lo, None)
    w = w / w.sum()

    valid = _valid_mask(range_map.alpha, range_map.rgb.shape[:2])
    rgb = range_map.rgb

    if is_color_split(range_map.split_method):
        if range_map.centers is None:
            raise ValueError("ColorRangeMap has no cluster centers")
        lab = range_map.lab
        if lab is None:
            lab = rgb_to_lab_array(rgb)
            range_map.lab = lab
        range_map.labels = assign_nearest_lab(
            lab,
            rgb,
            range_map.centers,
            valid,
            w,
            freeze_pair=freeze_pair,
            prev_labels=range_map.labels,
        )
        total = int(valid.sum())
        for i, band in enumerate(range_map.ranges):
            mask = range_map.labels == i
            count = int(mask.sum())
            band.weight = float(w[i])
            band.pixel_count = count
            band.total_pixels = total
            band.luma_low = band.luma_high = float(range_map.centers[i, 0])
            if count > 0:
                mean = rgb[mask].mean(axis=0)
                band.mean_rgb = (int(round(mean[0])), int(round(mean[1])), int(round(mean[2])))
        return

    values = _histogram_channel(range_map, rgb)
    edges = _bin_edges(
        values,
        n,
        range_map.split_method,
        valid,
        start=getattr(range_map, "bin_start", None),
        weights=w,
    )

    range_map.edges = edges
    range_map.labels = _label_pixels(values, edges, valid)
    total = int(valid.sum())
    for i, band in enumerate(range_map.ranges):
        mask = range_map.labels == i
        count = int(mask.sum())
        band.weight = float(w[i])
        band.luma_low = float(edges[i])
        band.luma_high = float(edges[i + 1])
        band.pixel_count = count
        band.total_pixels = total
        if count > 0:
            mean = rgb[mask].mean(axis=0)
            band.mean_rgb = (int(round(mean[0])), int(round(mean[1])), int(round(mean[2])))


def apply_bin_limits(
    range_map: ColorRangeMap,
    *,
    min_coverage: float | None = None,
    bin_start: float | None = None,
) -> None:
    """Re-cut histogram from Start / Min % without wiping match/change-to colors."""
    if min_coverage is not None:
        range_map.min_coverage = clamp_min_coverage(min_coverage)
    if bin_start is not None:
        range_map.bin_start = float(bin_start)
    if is_color_split(range_map.split_method) or range_map.rgb is None:
        return
    matches = [band.match_rgb for band in range_map.ranges]
    repls = [band.replacement_rgb for band in range_map.ranges]
    names = [band.name for band in range_map.ranges]
    vis = [bool(band.visible) for band in range_map.ranges]
    apply_weights(range_map, range_map.weights())
    for band, m, r, name, v in zip(range_map.ranges, matches, repls, names, vis):
        band.match_rgb = m
        band.replacement_rgb = r
        band.name = name
        band.visible = v


def steal_from_pair(
    weights: Sequence[float],
    index: int,
    neighbor: int,
    target: float,
    floor: float = MIN_COVERAGE,
) -> list[float]:
    """Give ``index`` ``target`` mass; take/give only from ``neighbor``."""
    w = [float(x) for x in weights]
    n = len(w)
    if n <= 1:
        return [1.0]
    index = max(0, min(int(index), n - 1))
    neighbor = max(0, min(int(neighbor), n - 1))
    if neighbor == index:
        neighbor = index + 1 if index < n - 1 else index - 1
    pair = w[index] + w[neighbor]
    cap = max(floor, pair - floor)
    t = min(max(float(target), floor), cap)
    w[index] = t
    w[neighbor] = pair - t
    total = sum(w) or 1.0
    return [x / total for x in w]


def steal_from_adjacent(
    weights: Sequence[float],
    index: int,
    target: float,
    floor: float = MIN_COVERAGE,
) -> list[float]:
    """Give ``index`` ``target`` mass; take/give only from the bar-adjacent range.

    Typed % (luma / L* / a* / b*) and divider drags: middle/first ranges steal
    from the **right** neighbor; the last range steals from the left. Other
    weights stay put. ``floor`` (MIN_COVERAGE) keeps both sides of the pair
    visible.
    """
    n = len(list(weights))
    if n <= 1:
        return [1.0]
    index = max(0, min(int(index), n - 1))
    neighbor = index + 1 if index < n - 1 else index - 1
    return steal_from_pair(weights, index, neighbor, target, floor)


def lab_nearest_other(centers: np.ndarray, index: int) -> int:
    """Index of the other Lab center nearest to ``centers[index]`` (Euclidean)."""
    n = int(centers.shape[0])
    if n <= 1:
        return 0
    index = max(0, min(int(index), n - 1))
    c = centers[index].astype(np.float64)
    best_i = 0 if index != 0 else 1
    best_d = float("inf")
    for i in range(n):
        if i == index:
            continue
        delta = centers[i].astype(np.float64) - c
        d = float(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2])
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def set_range_weight(range_map: ColorRangeMap, index: int, weight: float) -> None:
    """Give range ``index`` this coverage; take/give from one neighbor only.

    Color closeness: Lab-nearest other center, then freeze those two labels.
    Luma / L* / a* / b*: bar-adjacent (right neighbor, last steals left).
    """
    n = len(range_map.ranges)
    if n <= 1:
        apply_weights(range_map, [1.0])
        return
    index = max(0, min(int(index), n - 1))
    floor = clamp_min_coverage(getattr(range_map, "min_coverage", MIN_COVERAGE))
    if is_color_split(range_map.split_method) and range_map.centers is not None:
        neighbor = lab_nearest_other(range_map.centers, index)
        new_weights = steal_from_pair(
            range_map.weights(), index, neighbor, float(weight), floor=floor
        )
        apply_weights(range_map, new_weights, freeze_pair=(index, neighbor))
        return
    new_weights = steal_from_adjacent(
        range_map.weights(), index, float(weight), floor=floor
    )
    apply_weights(range_map, new_weights)


def sync_centers_from_match(range_map: ColorRangeMap) -> None:
    """Rebuild Lab centers from each range's match-from RGB (eyedrop / wheel)."""
    n = len(range_map.ranges)
    if n == 0:
        return
    centers = np.stack([rgb_tuple_to_lab(band.match_rgb) for band in range_map.ranges]).astype(
        np.float32
    )
    range_map.centers = centers


def reset_to_image_colors(range_map: ColorRangeMap) -> None:
    """Snap match-from and change-to to the N most matching image colors.

    Color closeness: re-run k-means (same seed as build) and even weights.
    Luma splits: keep bins; both swatches go back to each band's mean RGB.
    """
    n = len(range_map.ranges)
    if n == 0 or range_map.rgb is None:
        return
    for band in range_map.ranges:
        band.name = ""
    if is_color_split(range_map.split_method):
        valid = _valid_mask(range_map.alpha, range_map.rgb.shape[:2])
        centers, lab = _kmeans_centers_from_image(range_map.rgb, valid, n, range_map.lab)
        range_map.lab = lab
        range_map.centers = centers
        n = int(centers.shape[0])
        even = [1.0 / n] * n
        for i, band in enumerate(range_map.ranges[:n]):
            rgb = lab_tuple_to_rgb(centers[i])
            band.match_rgb = rgb
            band.replacement_rgb = rgb
        apply_weights(range_map, even)
        return
    for band in range_map.ranges:
        band.match_rgb = band.mean_rgb
        band.replacement_rgb = band.mean_rgb


def _swatch_snapshot(
    ranges: Sequence[ColorRange],
) -> list[tuple[tuple[int, int, int], tuple[int, int, int], str, bool]]:
    """match-from, change-to, name, visible — restored after a coverage shift."""
    return [
        (band.match_rgb, band.replacement_rgb, band.name, bool(band.visible))
        for band in ranges
    ]


def _restore_swatches(
    ranges: Sequence[ColorRange],
    saved: Sequence[tuple[tuple[int, int, int], tuple[int, int, int], str, bool]],
) -> None:
    """Write saved swatches back; extra ranges (the new one) are left alone."""
    for band, (match, replace, name, visible) in zip(ranges, saved):
        band.match_rgb = match
        band.replacement_rgb = replace
        band.name = name
        band.visible = visible


def _farthest_lab_sample(
    lab: np.ndarray,
    valid: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """k-means++ next seed: Lab sample with the largest nearest-center distance."""
    samples = lab[valid].reshape(-1, 3)
    if samples.size == 0:
        return np.zeros(3, dtype=np.float32)
    rng = np.random.default_rng(KMEANS_SEED)
    pts = samples.astype(np.float64)
    if pts.shape[0] > KMEANS_SAMPLES:
        idx = rng.choice(pts.shape[0], KMEANS_SAMPLES, replace=False)
        pts = pts[idx]
    closest = np.full(pts.shape[0], np.inf, dtype=np.float64)
    for i in range(int(centers.shape[0])):
        delta = pts - centers[i]
        d2 = np.einsum("ij,ij->i", delta, delta)
        closest = np.minimum(closest, d2)
    return pts[int(closest.argmax())].astype(np.float32)


def _settle_new_lab_center(
    range_map: ColorRangeMap,
    new_index: int,
    weights: Sequence[float],
) -> None:
    """Lloyd-update only the new center; existing match-from centers stay put."""
    if range_map.rgb is None or range_map.centers is None:
        return
    valid = _valid_mask(range_map.alpha, range_map.rgb.shape[:2])
    lab = range_map.lab
    if lab is None:
        lab = rgb_to_lab_array(range_map.rgb)
        range_map.lab = lab
    centers = range_map.centers
    for _ in range(KMEANS_ITERS):
        labels = assign_nearest_lab(lab, range_map.rgb, centers, valid, weights)
        mask = (labels == new_index) & valid
        if not bool(mask.any()):
            break
        new_c = lab[mask].reshape(-1, 3).mean(axis=0)
        if np.abs(new_c - centers[new_index]).max() <= 1e-4:
            centers[new_index] = new_c
            break
        centers[new_index] = new_c


def insert_color_range(range_map: ColorRangeMap) -> int:
    """Append one cluster/range. Keep existing match-from / change-to / names / eyes.

    Color closeness: seed a new Lab center (farthest sample) and let it steal
    pixels; old centers stay so Pantone change-to on the other ranges survives.
    Texture slider and eye on the map are not touched.
    Returns the new range index.
    """
    if range_map.rgb is None:
        raise ValueError("ColorRangeMap has no pixel data")
    n = len(range_map.ranges)
    if n < 1:
        raise ValueError("ColorRangeMap has no ranges to shift")
    saved = _swatch_snapshot(range_map.ranges)
    old_w = [float(band.weight) for band in range_map.ranges]
    new_w = 1.0 / (n + 1)
    scale = 1.0 - new_w
    weights = [max(w * scale, 1e-4) for w in old_w] + [new_w]
    total = sum(weights) or 1.0
    weights = [w / total for w in weights]
    total_px = int(range_map.ranges[0].total_pixels) if range_map.ranges else 0

    if is_color_split(range_map.split_method):
        sync_centers_from_match(range_map)
        if range_map.centers is None:
            raise ValueError("ColorRangeMap has no cluster centers")
        valid = _valid_mask(range_map.alpha, range_map.rgb.shape[:2])
        lab = range_map.lab
        if lab is None:
            lab = rgb_to_lab_array(range_map.rgb)
            range_map.lab = lab
        seed = _farthest_lab_sample(lab, valid, range_map.centers)
        range_map.centers = np.vstack((range_map.centers, seed.reshape(1, 3)))
        rgb = lab_tuple_to_rgb(range_map.centers[-1])
        range_map.ranges.append(
            ColorRange(
                index=n,
                luma_low=float(range_map.centers[-1, 0]),
                luma_high=float(range_map.centers[-1, 0]),
                mean_rgb=rgb,
                match_rgb=rgb,
                replacement_rgb=rgb,
                pixel_count=0,
                total_pixels=total_px,
                weight=new_w,
            )
        )
        range_map.range_count = n + 1
        _settle_new_lab_center(range_map, n, weights)
        apply_weights(range_map, weights)
        _restore_swatches(range_map.ranges, saved)
        new_band = range_map.ranges[n]
        new_rgb = lab_tuple_to_rgb(range_map.centers[n])
        new_band.match_rgb = new_rgb
        new_band.replacement_rgb = new_rgb
        new_band.name = ""
        new_band.visible = True
        return n

    range_map.ranges.append(
        ColorRange(
            index=n,
            luma_low=0.0,
            luma_high=0.0,
            mean_rgb=(128, 128, 128),
            match_rgb=(128, 128, 128),
            replacement_rgb=(128, 128, 128),
            pixel_count=0,
            total_pixels=total_px,
            weight=new_w,
        )
    )
    range_map.range_count = n + 1
    apply_weights(range_map, weights)
    _restore_swatches(range_map.ranges, saved)
    new_band = range_map.ranges[n]
    new_band.match_rgb = new_band.mean_rgb
    new_band.replacement_rgb = new_band.mean_rgb
    new_band.name = ""
    new_band.visible = True
    return n


def drop_color_range(range_map: ColorRangeMap, index: int) -> None:
    """Remove range ``index`` and shift coverage onto the rest.

    Surviving match-from / change-to / names / eyes stay. Texture is unchanged.
    """
    if range_map.rgb is None:
        raise ValueError("ColorRangeMap has no pixel data")
    n = len(range_map.ranges)
    if n <= 1:
        return
    index = max(0, min(int(index), n - 1))
    saved = [
        item
        for i, item in enumerate(_swatch_snapshot(range_map.ranges))
        if i != index
    ]
    weights = [float(band.weight) for i, band in enumerate(range_map.ranges) if i != index]
    total = sum(weights) or 1.0
    weights = [w / total for w in weights]

    del range_map.ranges[index]
    for i, band in enumerate(range_map.ranges):
        band.index = i
    range_map.range_count = n - 1
    if range_map.centers is not None and range_map.centers.shape[0] > index:
        range_map.centers = np.delete(range_map.centers, index, axis=0)

    _restore_swatches(range_map.ranges, saved)
    if is_color_split(range_map.split_method):
        sync_centers_from_match(range_map)
    apply_weights(range_map, weights)
    _restore_swatches(range_map.ranges, saved)
    if is_color_split(range_map.split_method):
        sync_centers_from_match(range_map)


def replacement_lut(ranges: Sequence[ColorRange]) -> np.ndarray:
    """N×3 uint8 table: label index → replacement RGB (one vectorized gather)."""
    if not ranges:
        return np.zeros((1, 3), dtype=np.uint8)
    n = max(band.index for band in ranges) + 1
    lut = np.zeros((n, 3), dtype=np.uint8)
    for band in ranges:
        lut[band.index] = band.replacement_rgb
    return lut


def replacement_lab_lut(ranges: Sequence[ColorRange]) -> np.ndarray:
    """N×3 float32 Lab of each range's replacement RGB (Color/Luminosity LUT)."""
    if not ranges:
        return np.zeros((1, 3), dtype=np.float32)
    n = max(band.index for band in ranges) + 1
    lut = np.zeros((n, 3), dtype=np.float32)
    for band in ranges:
        lut[band.index] = rgb_tuple_to_lab(band.replacement_rgb)
    return lut


def _paint_ranges(
    rgb: np.ndarray,
    labels: np.ndarray,
    ranges: Sequence[ColorRange],
    *,
    preserve_texture: bool,
) -> np.ndarray:
    """Remap labeled pixels. ``labels == -1`` (transparent) stay untouched.

    preserve_texture True: HSL keep-L — original per-pixel lightness stays so
    linen grain / weave / shadows survive; H and S come from the picked color.

    preserve_texture False: solid ``replacement_rgb`` via a label LUT
    (no Python per-range mask loop on 207MP).
    """
    if not preserve_texture:
        lut = replacement_lut(ranges)
        last = lut.shape[0] - 1
        idx = np.clip(labels, 0, last)
        painted = lut[idx]
        valid = labels >= 0
        if bool(valid.all()):
            return painted
        out = rgb.copy()
        out[valid] = painted[valid]
        return out

    out = rgb.copy()
    n = len(ranges)
    if n == 0:
        return out
    # LUT: one H/S pair per range — vectorized over all opaque pixels
    h_lut = np.zeros(n, dtype=np.float32)
    s_lut = np.zeros(n, dtype=np.float32)
    for band in ranges:
        h, s, _l = rgb_to_hsl(band.replacement_rgb)
        h_lut[band.index] = h
        s_lut[band.index] = s

    valid = labels >= 0
    if not np.any(valid):
        return out

    # Keep HSL lightness of the source pixel (texture); Rec. 709 luma
    # tracks this closely on near-neutral wallpaper and is the bin axis.
    l_pix = hsl_lightness(rgb)[valid]
    idx = labels[valid]
    out[valid] = hls_array_to_rgb(h_lut[idx], l_pix, s_lut[idx])
    return out


def _image_from_rgb(rgb: np.ndarray, alpha: np.ndarray | None) -> Image.Image:
    """Wrap RGB (and optional alpha) as a PIL image."""
    if alpha is None:
        return Image.fromarray(rgb, mode="RGB")
    rgba = np.dstack((rgb, alpha))
    return Image.fromarray(rgba, mode="RGBA")


def apply_replacements(
    range_map: ColorRangeMap,
    *,
    preserve_texture: bool = True,
) -> Image.Image:
    """Remap the working copy: keep texture by default, or solid-fill."""
    if range_map.rgb is None or range_map.labels is None:
        raise ValueError("ColorRangeMap has no pixel data")

    out = _paint_ranges(
        range_map.rgb,
        range_map.labels,
        range_map.ranges,
        preserve_texture=preserve_texture,
    )
    return _image_from_rgb(out, range_map.alpha)


def label_image(
    image: Image.Image,
    range_map: ColorRangeMap,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Full-res (or any) pixels labeled with this map's centers or bin edges.

    Preview maps are built on a downscaled work image; export reuses the
    stored Lab centers (or luma / L* / a* / b* cuts) so print files match
    the UI (CAP4631C same quantizer). ``copy=False`` — callers must not mutate rgb.
    """
    rgb, alpha = rgb_array(image, copy=False)
    valid = _valid_mask(alpha, rgb.shape[:2])
    if is_color_split(range_map.split_method):
        if range_map.centers is None:
            raise ValueError("ColorRangeMap has no cluster centers")
        labels = assign_nearest_lab(
            None,
            rgb,
            range_map.centers,
            valid,
            range_map.weights(),
        )
        return rgb, alpha, labels
    if range_map.edges is None:
        raise ValueError("ColorRangeMap has no bin edges")
    ch = split_axis_channel(range_map.split_method)
    if ch is None:
        values = luma_channel(rgb)
        labels = _label_pixels(values, range_map.edges, valid)
        return rgb, alpha, labels
    h, w_px = rgb.shape[:2]
    if h * w_px <= FULL_VECTOR_MAX_PIXELS:
        lab = rgb_to_lab_array(rgb)
        labels = _label_pixels(_lab_axis_values(lab, range_map.split_method), range_map.edges, valid)
        return rgb, alpha, labels
    labels = np.full((h, w_px), -1, dtype=np.int32)
    strip = ASSIGN_STRIP_ROWS
    for y0 in range(0, h, strip):
        y1 = min(h, y0 + strip)
        block = rgb_to_lab_array(rgb[y0:y1])
        labels[y0:y1] = _label_pixels(
            _lab_axis_values(block, range_map.split_method),
            range_map.edges,
            valid[y0:y1],
        )
    return rgb, alpha, labels


def apply_to_image(
    image: Image.Image,
    range_map: ColorRangeMap,
    *,
    preserve_texture: bool = True,
) -> Image.Image:
    """Remap ``image`` with this map's centers/edges and replacement colors.

    Preview maps may be built on a downscaled copy; save uses the same
    assignment and the same texture-preserving remap on the original pixels.
    """
    rgb, alpha, labels = label_image(image, range_map)
    out = _paint_ranges(
        rgb,
        labels,
        range_map.ranges,
        preserve_texture=preserve_texture,
    )
    return _image_from_rgb(out, alpha)


def snapshot_assignment(range_map: ColorRangeMap) -> ColorRangeMap:
    """Copy assignment params for a background save (no work-image pixel buffers)."""
    centers = None if range_map.centers is None else np.array(range_map.centers, copy=True)
    edges = None if range_map.edges is None else np.array(range_map.edges, copy=True)
    ranges = [
        ColorRange(
            index=band.index,
            luma_low=band.luma_low,
            luma_high=band.luma_high,
            mean_rgb=band.mean_rgb,
            match_rgb=(
                int(band.match_rgb[0]),
                int(band.match_rgb[1]),
                int(band.match_rgb[2]),
            ),
            replacement_rgb=(
                int(band.replacement_rgb[0]),
                int(band.replacement_rgb[1]),
                int(band.replacement_rgb[2]),
            ),
            pixel_count=band.pixel_count,
            total_pixels=band.total_pixels,
            weight=band.weight,
            name=band.name,
            visible=bool(band.visible),
        )
        for band in range_map.ranges
    ]
    return ColorRangeMap(
        range_count=range_map.range_count,
        split_method=range_map.split_method,
        ranges=ranges,
        edges=edges,
        centers=centers,
        labels=None,
        rgb=None,
        alpha=None,
        lab=None,
        texture_strength=float(range_map.texture_strength),
        texture_enabled=bool(range_map.texture_enabled),
        tone_darks=float(range_map.tone_darks),
        tone_lights=float(range_map.tone_lights),
        tone_brightness=float(range_map.tone_brightness),
        tone_contrast=float(getattr(range_map, "tone_contrast", 0.0)),
        tone_exposure=float(getattr(range_map, "tone_exposure", 0.0)),
        tone_lights_reds=float(range_map.tone_lights_reds),
        tone_lights_greens=float(range_map.tone_lights_greens),
        tone_lights_blues=float(range_map.tone_lights_blues),
        tone_temperature=float(getattr(range_map, "tone_temperature", 0.0)),
        tone_tint=float(getattr(range_map, "tone_tint", 0.0)),
        tone_saturation=float(getattr(range_map, "tone_saturation", 0.0)),
        tone_balance_cyan=float(getattr(range_map, "tone_balance_cyan", 0.0)),
        tone_balance_magenta=float(getattr(range_map, "tone_balance_magenta", 0.0)),
        tone_balance_yellow=float(getattr(range_map, "tone_balance_yellow", 0.0)),
        tone_lights_cyan=float(getattr(range_map, "tone_lights_cyan", 0.0)),
        tone_lights_magenta=float(getattr(range_map, "tone_lights_magenta", 0.0)),
        tone_lights_yellow=float(getattr(range_map, "tone_lights_yellow", 0.0)),
        tone_darks_cyan=float(getattr(range_map, "tone_darks_cyan", 0.0)),
        tone_darks_magenta=float(getattr(range_map, "tone_darks_magenta", 0.0)),
        tone_darks_yellow=float(getattr(range_map, "tone_darks_yellow", 0.0)),
        bin_start=getattr(range_map, "bin_start", None),
        min_coverage=clamp_min_coverage(getattr(range_map, "min_coverage", MIN_COVERAGE)),
    )


def band_masks(labels: np.ndarray, range_count: int) -> list[np.ndarray]:
    """8-bit grayscale masks, one per band: 255 in-range, 0 elsewhere.

    Non-overlapping by construction (each pixel has one label). Transparent
    samples (label -1) stay 0 on every mask.
    """
    return [(labels == i).astype(np.uint8) * 255 for i in range(range_count)]


def range_mask_preview(range_map: ColorRangeMap, index: int, size: int = 48) -> Image.Image:
    """Small swatch: original pixels in this range on a dark gray field."""
    if range_map.rgb is None or range_map.labels is None:
        raise ValueError("ColorRangeMap has no pixel data")

    rgb = range_map.rgb
    mask = range_map.labels == index
    canvas = np.full(rgb.shape, 40, dtype=np.uint8)  # dark field so light inks read
    canvas[mask] = rgb[mask]
    preview = Image.fromarray(canvas, mode="RGB")
    preview.thumbnail((size, size), Image.Resampling.NEAREST)
    return preview
