# -*- coding: utf-8 -*-
"""
wallpaper_recolor.ui.constants
------------------------------
Shared UI constants for Wallpaper Recolor (preview, dock, tools).

Class references (code + name only):
- CAP3321C Data Wrangling
- CAP4631C Machine Learning
- CAP4633C Machine Learning 2
"""

from __future__ import annotations

from pathlib import Path

from wallpaper_recolor.color.color_ranges import (
    ASSIGN_KMEANS,
    ASSIGN_KMEANS_LABEL,
    ASSIGN_PALETTE,
    ASSIGN_PALETTE_LABEL,
    RANGE_BY_COLOR_LABEL,
    RANGE_BY_LAB_A_LABEL,
    RANGE_BY_LAB_B_LABEL,
    RANGE_BY_LAB_L_LABEL,
    RANGE_BY_LUMA_LABEL,
    SPLIT_EQUAL_LIGHTNESS,
    SPLIT_EQUAL_LIGHTNESS_LABEL,
    SPLIT_EQUAL_PIXELS,
    SPLIT_EQUAL_PIXELS_LABEL,
    SPLIT_LAB_A_EQUAL,
    SPLIT_LAB_A_PIXELS,
    SPLIT_LAB_B_EQUAL,
    SPLIT_LAB_B_PIXELS,
    SPLIT_LAB_L_EQUAL,
    SPLIT_LAB_L_PIXELS,
)
from wallpaper_recolor.transform.tessellate import (
    SIDE_BOTTOM,
    SIDE_LEFT,
    SIDE_OFF,
    SIDE_RIGHT,
    SIDE_TOP,
)

# ---------------------------------------------------------------------------
# Preview Fit / view-zoom (100% = contain in pane, not crop)
# ---------------------------------------------------------------------------
# Longest preview edge in pixels — fallback 100% box when pane size is unknown
PREVIEW_MAX_EDGE = 560
# Fallback 100% long edge for tool tabs when that tab's pane is unmapped
TILE_VIEW_MAX_EDGE = 420 * 3  # tile_repeat cell_max_edge × grid
SEAM_VIEW_MAX_EDGE = 720
MOCKUP_VIEW_MAX_EDGE = 900
# Preview *view* zoom: 100% = fit-to-pane (whole image visible); 800% = 8× that
# fitted size. NEAREST from work-res. Crop zoom still crops the wallpaper.
VIEW_ZOOM_PCT_MIN = 100.0
VIEW_ZOOM_PCT_MAX = 800.0
VIEW_ZOOM_PCT_DEFAULT = 100.0
VIEW_ZOOM_PCT_STEP = 25.0
# Compact Color & lighting knobs: needle is a 270° readout; drag is relative
# (up/right +, down/left −). Gain grows gently with distance from the knob.
TONE_KNOB_PX = 22
TONE_KNOB_SWEEP_DEG = 270.0
TONE_KNOB_MIN_DEG = 225.0  # 7:30 — needle at −100
TONE_KNOB_BASE_RATE = 0.04  # units per pixel at the knob (fraction of a unit)
TONE_KNOB_GROWTH = 2.2
TONE_KNOB_REF_PX = 110.0
TONE_KNOB_STEP = 0.1
_MIN_PANE_FOR_FIT = 16  # ignore unmapped / 1×1 widgets when fitting
PREVIEW_PANE_BG = "#2a2a2a"  # letterbox / pillarbox (same on Original and Result)
_PREVIEW_PAN_DRAG_PX = 4
# Longest edge used to build the interactive range map (not the saved file)
WORK_MAX_EDGE = 1600
MIN_RANGES = 2
MAX_RANGES = 24
DEFAULT_RANGES = 4
# Coalesce wheel / coverage-bar drags so the preview stays live without jank
PREVIEW_DEBOUNCE_MS = 40
CLUSTER_DEBOUNCE_MS = 120  # Lab scatter; keep wheel/coverage drags off the Tk thread
_LAYER_SWATCH_PX = 16
_LAYER_TWISTY_OPEN = "▾"
_LAYER_TWISTY_SHUT = "▸"
CHIP_COLUMNS = 6  # compact range chips wrap under the coverage bar
DEFAULT_MOCKUP_REPEATS = 4.0
DEFAULT_MOCKUP_COVER = "full"
WALLPAPER_GEOMETRIC_LABEL = "Geometric (stripes)"
WALLPAPER_FLORAL_LABEL = "Floral (damask)"
WALLPAPER_STYLE_LABELS = (WALLPAPER_GEOMETRIC_LABEL, WALLPAPER_FLORAL_LABEL)
# Title-bar drag past this many pixels starts snap-rearrange between columns
_UNDOCK_DRAG_PX = 10
# Vertical gap under each packed pane (must match _repack pady)
_PANEL_PACK_PADY = 6
_PANEL_BAR_BG = "#dedede"
_PANEL_BAR_FG = "#222222"
_SNAP_HIGHLIGHT = "#4a90d9"
_INSERT_MARKER_BG = "#2d6bb3"
_COL_IDLE_BORDER = "#c8c8c8"
# Widget bind + bind_all: ttk.Scale/Notebook eat MouseWheel before bind_all (last bindtag)
_COLUMN_SCROLL_TAG = "WallpaperColScroll"
# Replace (do not add=+) these class binds so the Scale default cannot nudge the value
_SCALE_WHEEL_CLASSES = ("TScale", "Scale")
_WHEEL_CLASS_TAGS = (
    "TNotebook",
    "TCombobox",
    "TCheckbutton",
    "TLabel",
    "TFrame",
    "TButton",
    "TEntry",
    "TSpinbox",
    "Canvas",
    "Frame",
    "Label",
    "Button",
    "Entry",
    "Checkbutton",
    "Text",
)
# FA icons live in wallpaper_recolor/icons/ (SVG source + cached PNG)
_ICONS_DIR = Path(__file__).resolve().parents[1] / "icons"
# FA rotate-left (rotate-left-solid-full.svg) rasterized for slider reset icons
_RESET_ICON_PX = 18
_RESET_ICON_FG = (34, 34, 34, 255)
_RESET_SVG_NAME = "rotate-left-solid-full.svg"
_RESET_PNG_NAME = "rotate-left-solid-full.png"
# FA eye-dropper (eye-dropper-solid-full.svg) — Original follow-cursor + swatch button
_EYEDROP_ICON_PX = 22
_EYEDROP_ICON_FG = (250, 250, 250, 255)
_EYEDROP_ICON_HALO = (20, 20, 20, 255)
_EYEDROP_SVG_NAME = "eye-dropper-solid-full.svg"
_EYEDROP_PNG_NAME = "eye-dropper-solid-full.png"
# FA eye / eye-slash — visible vs hidden (texture + each color range)
_EYE_ICON_PX = 16
_EYE_ICON_FG = (34, 34, 34, 255)
_EYE_ON_SVG_NAME = "eye-solid-full.svg"
_EYE_ON_PNG_NAME = "eye-solid-full.png"
_EYE_OFF_SVG_NAME = "eye-slash-solid-full.svg"
_EYE_OFF_PNG_NAME = "eye-slash-solid-full.png"
# FA magnifying-glass minus / plus — zoom-out (left) and zoom-in (right)
_ZOOM_ICON_PX = 16
_ZOOM_OUT_SVG_NAME = "magnifying-glass-minus-solid-full.svg"
_ZOOM_OUT_PNG_NAME = "magnifying-glass-minus-solid-full.png"
_ZOOM_IN_SVG_NAME = "magnifying-glass-plus-solid-full.svg"
_ZOOM_IN_PNG_NAME = "magnifying-glass-plus-solid-full.png"
# Eyedrop loupe: 10× an 11px neighborhood of the displayed Original (110px circle)
_LOUPE_SRC_PX = 11
_LOUPE_ZOOM = 10
_LOUPE_PX = _LOUPE_SRC_PX * _LOUPE_ZOOM
_LOUPE_GAP = 16  # top-right of the pipette tip; not under the sample point
# Hide a slider's Reset when it is within half a unit of that control's default
_RESET_EPS = 0.5
_ZOOM_RESET_EPS = 0.01  # zoom is 1–8, so 0.5 would hide a 1.4× zoom reset
# Horizontal paned split: left stays larger by default; user drags the sash
LEFT_COL_WEIGHT = 3
RIGHT_COL_WEIGHT = 1
LEFT_COL_MINSIZE = 360
RIGHT_COL_MINSIZE = 220
LEFT_SASH_FRACTION = 0.72
RIGHT_SPLIT_FRACTION = 0.5
RIGHT_SPLIT_MINSIZE = 80
_RIGHT_SASH_BAND_PX = 10
# Ctrl+Z / Ctrl+Y keep this many past edits (oldest drops off)
HISTORY_LIMIT = 20
HISTORY_PANEL_TITLE = "History"
# Session JSON next to presets.json (dock order, hidden panes, sash)
LAYOUT_PROFILES_FILENAME = "layout_profiles.json"
# Pane title / View menu (X/Y then zoom). Saved layouts may still say "Crop".
CROP_PANEL_TITLE = "Position & Zoom"
TONE_PANEL_TITLE = "Color & lighting"
_LAYOUT_TITLE_ALIASES = {"Crop": CROP_PANEL_TITLE, "Tone": TONE_PANEL_TITLE}
# Wallpaper Recolor edit-state files (JSON; .wpedit or *_edit.json)
EDIT_STATE_FORMAT = "wpedit"
EDIT_STATE_VERSION = 2
EDIT_STATE_FILETYPES = [
    ("Wallpaper edit state", "*.wpedit *.json"),
    ("JSON", "*.json"),
    ("All files", "*.*"),
]
_TESS_H_TIPS = {
    SIDE_OFF: "Skip horizontal wrap.",
    SIDE_LEFT: "Use the left edge as the source; the opposite edge is the model.",
    SIDE_RIGHT: "Use the right edge as the source; the opposite edge is the model.",
}
_TESS_V_TIPS = {
    SIDE_OFF: "Skip vertical wrap.",
    SIDE_TOP: "Use the top edge as the source; the opposite edge is the model.",
    SIDE_BOTTOM: "Use the bottom edge as the source; the opposite edge is the model.",
}

# Range by: k-means, Rec. 709 luma, or CIE Lab axis bins
RANGE_BY_LABELS = (
    RANGE_BY_COLOR_LABEL,  # k-means / nearest Lab — similar colors
    RANGE_BY_LUMA_LABEL,  # Rec. 709 light / mid / dark
    RANGE_BY_LAB_L_LABEL,
    RANGE_BY_LAB_A_LABEL,
    RANGE_BY_LAB_B_LABEL,
)
# Shown when Range by is Color closeness — k-means vs nearest palette hex
ASSIGN_LABELS = (ASSIGN_KMEANS_LABEL, ASSIGN_PALETTE_LABEL)
ASSIGN_BY_LABEL = {
    ASSIGN_KMEANS_LABEL: ASSIGN_KMEANS,
    ASSIGN_PALETTE_LABEL: ASSIGN_PALETTE,
}
ASSIGN_LABEL_FOR = {
    ASSIGN_KMEANS: ASSIGN_KMEANS_LABEL,
    ASSIGN_PALETTE: ASSIGN_PALETTE_LABEL,
}
# Equal steps vs even pixels — luma and L* / a* / b* histogram splits
LUMA_SPLIT_LABELS = {
    SPLIT_EQUAL_PIXELS_LABEL: SPLIT_EQUAL_PIXELS,  # histogram thirds / even pixel count
    SPLIT_EQUAL_LIGHTNESS_LABEL: SPLIT_EQUAL_LIGHTNESS,  # even steps along the axis
}
_RANGE_BY_BIN_METHODS = {
    RANGE_BY_LUMA_LABEL: (SPLIT_EQUAL_LIGHTNESS, SPLIT_EQUAL_PIXELS),
    RANGE_BY_LAB_L_LABEL: (SPLIT_LAB_L_EQUAL, SPLIT_LAB_L_PIXELS),
    RANGE_BY_LAB_A_LABEL: (SPLIT_LAB_A_EQUAL, SPLIT_LAB_A_PIXELS),
    RANGE_BY_LAB_B_LABEL: (SPLIT_LAB_B_EQUAL, SPLIT_LAB_B_PIXELS),
}
# Default dock: every pane visible. View can still hide; Reset layout restores this.
JOB_HIDDEN_PANEL_TITLES = frozenset()
INSPECTION_TAB_TITLES = ("3×3 tile", "Seam offset", "Room mockup")

ICC_FILETYPES = [
    ("ICC profiles", "*.icc *.icm"),
    ("All files", "*.*"),
]
# Pointer tools — View Move pans the camera; Grab Move offsets the selected image
# ---------------------------------------------------------------------------
# Pointer tools (toolbar + Tools menu stay in sync)
# ---------------------------------------------------------------------------
TOOL_VIEW_MOVE = "view_move"
TOOL_GRAB_MOVE = "grab_move"
TOOL_GRAB = TOOL_GRAB_MOVE  # old name
TOOL_VIEW_MOVE_LABEL = "View Move"
TOOL_GRAB_MOVE_LABEL = "Grab Move"
POINTER_TOOL_LABELS = (TOOL_VIEW_MOVE_LABEL, TOOL_GRAB_MOVE_LABEL)
POINTER_TOOL_BY_LABEL = {
    TOOL_VIEW_MOVE_LABEL: TOOL_VIEW_MOVE,
    TOOL_GRAB_MOVE_LABEL: TOOL_GRAB_MOVE,
}
POINTER_LABEL_FOR = {
    TOOL_VIEW_MOVE: TOOL_VIEW_MOVE_LABEL,
    TOOL_GRAB_MOVE: TOOL_GRAB_MOVE_LABEL,
}


__all__ = [n for n in globals() if not n.startswith("__")]
