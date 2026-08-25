# -*- coding: utf-8 -*-
"""
wallpaper_recolor.ui.snapshot
------------------------------
Undo ticks (EditSnapshot) and small JSON / work-image helpers.

An undo tick stores colors, weights, eyes, texture, and tone — not pixel
buffers — so Ctrl+Z stays cheap on print-size TIFs.

Class references (code + name only):
- CAP3321C Data Wrangling
- CAP4631C Machine Learning
- CAP4633C Machine Learning 2
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from wallpaper_recolor.color.color_ranges import ASSIGN_KMEANS, MIN_COVERAGE
from wallpaper_recolor.color.presets import default_presets_path
from wallpaper_recolor.ui.constants import LAYOUT_PROFILES_FILENAME

def default_layout_profiles_path() -> Path:
    """layout_profiles.json beside presets.json (project folder)."""
    return default_presets_path().with_name(LAYOUT_PROFILES_FILENAME)


def _history_menu_label(verb: str, count: int) -> str:
    """Undo/Redo label; remaining steps sit after a tab (Tk accelerator column, gray)."""
    if count:
        return f"{verb}\t{count}"
    return verb


def _json_rgb(value) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return (
            max(0, min(255, int(value[0]))),
            max(0, min(255, int(value[1]))),
            max(0, min(255, int(value[2]))),
        )
    return (0, 0, 0)


def _json_box(value) -> tuple[int, int, int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return (int(value[0]), int(value[1]), int(value[2]), int(value[3]))
    return None


def _fit(image: Image.Image, max_edge: int) -> Image.Image:
    """Downscale so the long side is at most ``max_edge`` (nearest for crisp edges)."""
    w, h = image.size
    long_edge = max(w, h)
    if long_edge <= max_edge:
        return image.copy()
    scale = max_edge / long_edge
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    # BILINEAR for photos; wallpaper patterns stay readable at this size
    return image.resize(size, Image.Resampling.BILINEAR)


@dataclass(frozen=True)
class EditSnapshot:
    """Compact undo tick: colors, weights, eyes, texture, tone — not pixel buffers."""

    replacements: tuple[tuple[int, int, int], ...]
    match_rgbs: tuple[tuple[int, int, int], ...]
    weights: tuple[float, ...]
    visibilities: tuple[bool, ...]
    names: tuple[str, ...]
    texture_strength: float
    texture_enabled: bool
    tone_darks: float
    tone_lights: float
    tone_brightness: float
    tone_contrast: float
    tone_exposure: float
    tone_lights_reds: float
    tone_lights_greens: float
    tone_lights_blues: float
    split_method: str
    range_count: int
    selected_index: int
    selected_half: str
    preset_id: str | None
    crop_x: int
    crop_y: int
    crop_zoom: float
    tess_h: str
    tess_v: str
    tess_built: bool
    tess_mode: str
    tess_tiles: int
    tess_lloyd: int
    tess_normalize: bool
    lighting_auto_darks: float
    lighting_auto_lights: float
    inpaint_boxes: tuple[tuple[int, int, int, int], ...]
    label_text: str
    label_size: int
    label_color: str
    label_x: int
    label_y: int
    label_font: str = ""
    inpaint_layer_id: str = ""
    inpaint_quads: tuple = ()
    layers: tuple[dict, ...] = ()
    selected_layer_ids: tuple[str, ...] = ()
    assignment_mode: str = ASSIGN_KMEANS
    tone_lights_cyan: float = 0.0
    tone_lights_magenta: float = 0.0
    tone_lights_yellow: float = 0.0
    tone_darks_cyan: float = 0.0
    tone_darks_magenta: float = 0.0
    tone_darks_yellow: float = 0.0
    tone_temperature: float = 0.0
    tone_tint: float = 0.0
    tone_saturation: float = 0.0
    tone_balance_cyan: float = 0.0
    tone_balance_magenta: float = 0.0
    tone_balance_yellow: float = 0.0
    bin_start: float | None = None
    min_coverage: float = MIN_COVERAGE
    icc_path: str | None = None
