# -*- coding: utf-8 -*-
"""
wallpaper_recolor.ui.app
------------------------------
WallpaperRecolorApp shell: holds document state and ``__init__``, then
delegates feature work to mixins under ``wallpaper_recolor.ui.mixins``.

Where to look
-------------
- chrome: menubar, View Move / Grab Move, close-save, busy bar
- layout: paned right column, dock, panel builders, View menu
- preview: Fit/contain, checker blit, eyedrop, Clusters glue
- ranges: coverage %, presets, hide-eye knockout
- adjust: Color & lighting, texture, Position & Zoom, scale, tessellate
- layers_labels: Layers tree, Labels / OCR
- session: open/export/job pack, ``.wpedit``, undo

Launch: ``python run_app.py`` → ``ui.run`` (lazy-imports this class).
Tests import helpers from this module so ``patch.object(ui.app, ...)`` works.

Class references (code + name only):
- CAP3321C Data Wrangling
- CAP4631C Machine Learning
- CAP4633C Machine Learning 2
"""

from __future__ import annotations

from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from io import BytesIO
from types import SimpleNamespace
import json
import math
import numpy as np
import threading
import tkinter as tk

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageTk

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
    SPLIT_COLOR_CLOSENESS,
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
    ColorRangeMap,
    MIN_COVERAGE,
    MIN_COVERAGE_MAX,
    apply_bin_limits,
    apply_weights,
    bin_display_key,
    build_range_map,
    clamp_min_coverage,
    drop_color_range,
    insert_color_range,
    is_color_split,
    is_lab_channel_split,
    is_palette_assign,
    resolved_bin_start,
    split_axis_caption,
    split_axis_channel,
    choose_kmeans_k,
    AUTO_K_MAX,
    reset_to_image_colors,
    set_range_weight,
    snapshot_assignment,
    sync_centers_from_match,
)
from wallpaper_recolor.ui.color_wheel import ColorWheel, rgb_to_hex
from wallpaper_recolor.ui.cluster_view import (
    CLUSTER_ZOOM_PCT_MAX,
    ClusterPlot,
    clamp_cluster_zoom_pct,
    cluster_scatter_data,
)
from wallpaper_recolor.ui.coverage_bar import HALF_MATCH, HALF_REPLACE, CoverageBar
from wallpaper_recolor.ui.tooltip import bind_tooltip
from wallpaper_recolor.io.export_layers_zip import export_layers_zip as write_layers_zip
from wallpaper_recolor.io.export_pack import export_job_pack
from wallpaper_recolor.transform.crop import (
    CROP_XY_DEFAULT,
    ZOOM_DEFAULT,
    ZOOM_MAX,
    ZOOM_MIN,
    apply_crop,
    clamp_zoom,
    offset_slider_limit,
    top_left_to_center_offset,
)
from wallpaper_recolor.transform.tessellate import (
    LLOYD_DEFAULT,
    LLOYD_MAX,
    LLOYD_MIN,
    MODE_DEFAULT,
    MODE_LABELS,
    MODE_MESH,
    MODE_TILE,
    MODE_VORONOI,
    SIDE_BOTTOM,
    SIDE_LEFT,
    SIDE_OFF,
    SIDE_RIGHT,
    SIDE_TOP,
    TILES_DEFAULT,
    TILES_MAX,
    TILES_MIN,
    apply_crop_lighting_tessellate,
    apply_tessellate,
    clamp_lloyd,
    clamp_tiles,
    coerce_built,
    coerce_normalize_lighting,
    edges_already_match,
    estimate_normalize_tone,
    normalize_h_side,
    normalize_tess_mode,
    normalize_v_side,
    plan_tessellate_crop,
    tess_mode_label,
)
from wallpaper_recolor.transform.inpaint import (
    STYLE_FLORAL,
    STYLE_GEOMETRIC,
    inpaint_backend,
    inpaint_image,
)
from wallpaper_recolor.transform.lama_onnx import lama_onnx_available
from wallpaper_recolor.io.image_io import OPEN_FILETYPES, SAVE_FILETYPES, load_image, save_image
from wallpaper_recolor.labels.boxes import (
    box_contains,
    display_box_to_source,
    display_xy_to_source,
    source_box_to_display,
)
from wallpaper_recolor.labels.detect import (
    aabb_quad,
    detect_text_boxes,
    detect_text_regions,
    tesseract_status_text,
)
from wallpaper_recolor.labels.layer import (
    LABEL_COLOR_DEFAULT,
    LABEL_FONT_DEFAULT,
    LABEL_SIZE_DEFAULT,
    LABEL_SIZE_MAX,
    LABEL_SIZE_MIN,
    LabelSpec,
    clamp_label_size,
    composite_label,
    list_font_families,
    parse_label_color,
    rgb_to_hex as label_rgb_to_hex,
)
from wallpaper_recolor.layers.stack import (
    ROLE_BASE,
    LayerStack,
    StackLayer,
    composite_stack,
    composite_over_checker,
    correction_target_ids,
    flatten_rgb_or_keep_alpha,
    inpaint_target_layer,
    primary_is_label,
)
from wallpaper_recolor.labels.overlay import decorate_preview
from wallpaper_recolor.color.layers import (
    TEXTURE_DEFAULT_STRENGTH,
    composite_for_image,
    live_composite_from_map,
)
from wallpaper_recolor.color.presets import (
    GENERIC_LABEL,
    apply_preset_palette,
    default_presets_path,
    delete_user_preset,
    ensure_default_presets,
    get_preset,
    is_generic_label,
    list_presets,
    range_by_label_for,
    save_user_preset,
    snapshot_preset,
    split_label_for,
)
from wallpaper_recolor.preview.preview_tools import (
    MOCKUP_COVER_LABELS,
    cover_frac_from_key,
    offset_seam,
    room_mockup,
    tile_repeat,
)
from wallpaper_recolor.transform.scale import (
    DEFAULT_RESAMPLE,
    DPI_CHOICES,
    DPI_CUSTOM_LABEL,
    DPI_DEFAULT,
    RESAMPLE_LABELS,
    UNIT_INCHES,
    UNIT_PIXELS,
    UNITS,
    format_dim,
    from_pixels,
    is_physical_unit,
    parse_dim,
    parse_dpi_choice,
    resolve_output_size,
    scale_image,
    to_pixels,
)
from wallpaper_recolor.color.tone import (
    TONE_NEUTRAL,
    TONE_SLIDER_MAX,
    TONE_SLIDER_MIN,
    apply_tone_rgb,
    estimate_gray_world_temp_tint,
    estimate_white_patch_temp_tint,
    is_neutral_tone,
    slider_to_amount,
)
from wallpaper_recolor.ui.constants import *
from wallpaper_recolor.ui.icons import (
    _EYEDROP_ICON_FG,
    _EYEDROP_ICON_HALO,
    _LOUPE_GAP,
    _LOUPE_PX,
    _LOUPE_SRC_PX,
    _LOUPE_ZOOM,
    _eyedrop_icon_image,
    _eyedrop_icon_photo,
    _eye_icon_photos,
    _fa_icon_image,
    _glyph_hotspot,
    _icon_has_opaque_plate,
    _make_eyedrop_loupe_image,
    _paste_rgba,
    _rasterize_fa_svg,
    _reset_icon_photo,
    _tk_photo_png,
    _zoom_icon_photos,
)
from wallpaper_recolor.ui.preview_fit import (
    PreviewZoomHost,
    _format_zoom_text,
    _preview_base_size,
    _scale_view_zoom,
    _view_zoom_size,
    _wheel_zoom_pct_delta,
    _widget_contains_root,
    clamp_view_zoom_pct,
    contain_size,
    fit_max_edge,
    letterbox_xy,
    pane_fit_factor,
    pane_usable_for_fit,
    shared_fit_factor,
    shared_pane_size,
    view_zoom_factor,
)
from wallpaper_recolor.ui.widgets import (
    EyeToggle,
    RangeChip,
    ToneKnob,
    _bind_smooth_scale,
    _bind_tree,
    _bind_wheel_tree,
    tone_knob_gain,
)
from wallpaper_recolor.ui.dock import DockablePanel, ScrollColumn, _ClipCanvas, _SashSplit
from wallpaper_recolor.ui.snapshot import (
    EditSnapshot,
    _fit,
    _history_menu_label,
    _json_box,
    _json_rgb,
    default_layout_profiles_path,
)
from wallpaper_recolor.ui.launch import (
    _monitor_work_area,
    _place_maximized_on_launch_monitor,
    _pointer_screen_xy,
    run,
)
from wallpaper_recolor.ui.mixins.adjust import AppAdjustMixin
from wallpaper_recolor.ui.mixins.chrome import AppChromeMixin
from wallpaper_recolor.ui.mixins.layers_labels import AppLayersLabelsMixin
from wallpaper_recolor.ui.mixins.layout import AppLayoutMixin
from wallpaper_recolor.ui.mixins.preview import AppPreviewMixin
from wallpaper_recolor.ui.mixins.ranges import AppRangesMixin
from wallpaper_recolor.ui.mixins.session import AppSessionMixin

# Re-export names tests import from wallpaper_recolor.ui.app
from wallpaper_recolor.io.export_layers_zip import export_layers_zip as write_layers_zip  # noqa: F401


class WallpaperRecolorApp(
    AppChromeMixin,
    AppLayoutMixin,
    AppLayersLabelsMixin,
    AppPreviewMixin,
    AppAdjustMixin,
    AppRangesMixin,
    AppSessionMixin,
):
    """Main window for wallpaper tonal-separation remapping."""

    def __init__(self, root: tk.Tk) -> None:
        # root: the Tk application window (withdrawn in tests; zoomed in run())
        self.root = root
        self.root.title("Wallpaper Recolor")
        self.root.minsize(1180, 760)

        # ---------------------------------------------------------------------------
        # Document buffers: full-res source vs work-size map (≤ WORK_MAX_EDGE)
        # ---------------------------------------------------------------------------
        self.source_path: Path | None = None
        self.source_image: Image.Image | None = None  # full-resolution original
        self.work_image: Image.Image | None = None  # downscaled map / preview
        self.range_map: ColorRangeMap | None = None
        self.range_chips: list[RangeChip] = []
        self.selected_index = 0
        self.selected_half = HALF_REPLACE  # wheel / eyedrop edit change-to by default
        self._scratch_rgb: tuple[int, int, int] = (255, 0, 0)
        self.preset_id: str | None = None
        self.icc_path: Path | None = None
        self._orig_photo: ImageTk.PhotoImage | None = None
        self._orig_pil: Image.Image | None = None
        self._tex_photo: ImageTk.PhotoImage | None = None
        self._tex_pil: Image.Image | None = None
        self._tool_photo: ImageTk.PhotoImage | None = None
        self._tool_pil: Image.Image | None = None
        self._tool_zoom_max_edge: int = PREVIEW_MAX_EDGE
        self._work_live: Image.Image | None = None
        self._preview_job: str | None = None  # after() id while dragging
        self._cluster_job: str | None = None
        self._layer_rows: list[tk.Misc] = []
        self._layer_range_rows: dict[int, dict] = {}
        self._layer_pct_mute = False
        self._fit_job: str | None = None  # after() id while the preview pane resizes
        self._fit_panes_applied: tuple[tuple[int, int], ...] | None = None
        self._preview_pan_x = 0
        self._preview_pan_y = 0
        self._pan_syncing = False
        self._mute_ui = False  # skip slider/wheel feedback while we load a range
        self._busy = False  # full-res save/export running on a worker thread
        self._job_cancel = threading.Event()
        self._job_cancellable = False
        self._job_cancel_status = "Cancelled."
        self._tone_spins: list[ttk.Spinbox] = []
        self._tone_focus_spin: ttk.Spinbox | None = None
        self._opening = False  # skip undo push while Open replaces the file
        self._history_lock = False
        self._column_scroll_bound = False
        self._chrome_after: str | None = None
        self._snap: tuple[ScrollColumn, int] | None = None
        self._snap_moving: DockablePanel | None = None
        self._undo_stack: list[EditSnapshot] = []
        self._redo_stack: list[EditSnapshot] = []
        # FA glyphs: keep PhotoImage refs so Tk does not GC the pixels
        self._reset_photo = _reset_icon_photo(self.root)
        self._eyedrop_pil = _eyedrop_icon_image()
        self._eyedrop_photo = _tk_photo_png(self._eyedrop_pil, self.root)
        self._eyedrop_hotspot = _glyph_hotspot(self._eyedrop_pil)
        self._eye_on_photo, self._eye_off_photo = _eye_icon_photos(self.root)
        self._eye_photos = (self._eye_on_photo, self._eye_off_photo)
        self._zoom_out_photo, self._zoom_in_photo = _zoom_icon_photos(self.root)
        self._eyedrop_overlay: tk.Canvas | None = None
        self._orig_eyedrop_photo: ImageTk.PhotoImage | None = None
        self._wheel_before: EditSnapshot | None = None
        self._slider_before: EditSnapshot | None = None
        self._generic_match: tuple[tuple[int, int, int], ...] | None = None
        self._generic_replace: tuple[tuple[int, int, int], ...] | None = None
        self._layout_profile_name: str | None = None
        self._layout_profiles_path = default_layout_profiles_path()
        self._edit_state_path: Path | None = None

        self.range_count = tk.IntVar(value=DEFAULT_RANGES)
        self.range_by = tk.StringVar(value=RANGE_BY_COLOR_LABEL)  # Color closeness default
        self.assign_label = tk.StringVar(value=ASSIGN_KMEANS_LABEL)  # Cluster from image
        self.luma_split_label = tk.StringVar(value=SPLIT_EQUAL_PIXELS_LABEL)  # luma default
        self.bin_min_pct = tk.DoubleVar(value=round(MIN_COVERAGE * 100.0))
        self.bin_start = tk.DoubleVar(value=0.0)
        self._bin_limits_mute = False
        self.status = tk.StringVar(value="Open a TIF, PNG, or JPEG — or pick a preset, then open.")
        self.busy_caption = tk.StringVar(value="")
        self.edit_caption = tk.StringVar(value="Open an image, then click a range to pick its color.")
        self.preset_choice = tk.StringVar(value=GENERIC_LABEL)
        self.mockup_repeats = tk.DoubleVar(value=DEFAULT_MOCKUP_REPEATS)
        self.mockup_cover = tk.StringVar(value=DEFAULT_MOCKUP_COVER)
        self.mockup_caption = tk.StringVar(value="Repeat scale: 4.0 tiles across the wall")
        # 0–100% original luminosity vs flat fill (default = full grain, new hues)
        self.texture_pct = tk.DoubleVar(value=round(TEXTURE_DEFAULT_STRENGTH * 100.0))
        self.texture_label = tk.StringVar(
            value=f"Texture: {round(TEXTURE_DEFAULT_STRENGTH * 100.0):.0f}%"
        )
        self.texture_enabled = tk.BooleanVar(value=True)
        self.darks_pct = tk.DoubleVar(value=TONE_NEUTRAL)
        self.lights_pct = tk.DoubleVar(value=TONE_NEUTRAL)
        self.brightness_pct = tk.DoubleVar(value=TONE_NEUTRAL)
        self.contrast_pct = tk.DoubleVar(value=TONE_NEUTRAL)
        self.exposure_pct = tk.DoubleVar(value=TONE_NEUTRAL)
        self.lights_reds_pct = tk.DoubleVar(value=TONE_NEUTRAL)
        self.lights_greens_pct = tk.DoubleVar(value=TONE_NEUTRAL)
        self.lights_blues_pct = tk.DoubleVar(value=TONE_NEUTRAL)
        self.temperature_pct = tk.DoubleVar(value=TONE_NEUTRAL)
        self.tint_pct = tk.DoubleVar(value=TONE_NEUTRAL)
        self.saturation_pct = tk.DoubleVar(value=TONE_NEUTRAL)
        self.balance_cyan_pct = tk.DoubleVar(value=TONE_NEUTRAL)
        self.balance_magenta_pct = tk.DoubleVar(value=TONE_NEUTRAL)
        self.balance_yellow_pct = tk.DoubleVar(value=TONE_NEUTRAL)
        self._tone_updating = False
        # Scale panel — empty size = original pixels; DPI always tags the saved file
        self.scale_unit = tk.StringVar(value=UNIT_PIXELS)
        self.scale_width = tk.StringVar(value="")
        self.scale_height = tk.StringVar(value="")
        self.scale_lock = tk.BooleanVar(value=True)
        self.scale_dpi_choice = tk.StringVar(value=str(DPI_DEFAULT))
        self.scale_dpi_custom = tk.StringVar(value="")
        self.scale_resample = tk.StringVar(value=DEFAULT_RESAMPLE)
        self.scale_source_note = tk.StringVar(value="Source: —")
        self.scale_equiv_note = tk.StringVar(value="")
        self.scale_save_note = tk.StringVar(value="Save at original size")
        self._scale_unit_prev = UNIT_PIXELS
        self._scale_dpi_prev = str(DPI_DEFAULT)
        self._scale_updating = False
        # Crop — top-left x/y in source pixels; zoom 1 = full image
        self.crop_x = tk.DoubleVar(value=float(CROP_XY_DEFAULT))
        self.crop_y = tk.DoubleVar(value=float(CROP_XY_DEFAULT))
        self.crop_zoom = tk.DoubleVar(value=float(ZOOM_DEFAULT))
        self.crop_x_text = tk.StringVar(value="0")
        self.crop_y_text = tk.StringVar(value="0")
        self.crop_zoom_text = tk.StringVar(value="1")
        self._crop_updating = False
        # Preview view-zoom — display only (100% fit–800%); independent of Crop zoom.
        # Composite and Clusters each keep their own percent; the header shows the active tab.
        self.preview_zoom = tk.DoubleVar(value=float(VIEW_ZOOM_PCT_DEFAULT))
        self.preview_zoom_caption = tk.StringVar(value="100%")
        self.view_zoom_title = tk.StringVar(value="View zoom")
        self._composite_zoom_pct = float(VIEW_ZOOM_PCT_DEFAULT)
        self._cluster_zoom_pct = float(VIEW_ZOOM_PCT_DEFAULT)
        self._preview_zoom_updating = False
        self.orig_title = tk.StringVar(value="Original")
        # Tessellate — Tile (Repeating Design) default; Tessellation / Mesh / mosaic
        self.tess_mode = tk.StringVar(value=MODE_DEFAULT)
        self.tess_mode_label = tk.StringVar(value=tess_mode_label(MODE_DEFAULT))
        self.tess_h = tk.StringVar(value=SIDE_OFF)
        self.tess_v = tk.StringVar(value=SIDE_OFF)
        self.tess_built = tk.BooleanVar(value=False)
        self.tess_tiles = tk.DoubleVar(value=float(TILES_DEFAULT))
        self.tess_lloyd = tk.DoubleVar(value=float(LLOYD_DEFAULT))
        self.tess_tiles_text = tk.StringVar(value=str(TILES_DEFAULT))
        self.tess_lloyd_text = tk.StringVar(value=str(LLOYD_DEFAULT))
        self.tess_normalize = tk.BooleanVar(value=False)
        self._lighting_auto_darks = 0.0
        self._lighting_auto_lights = 0.0
        self._tess_updating = False
        self._tess_committed = (SIDE_OFF, SIDE_OFF, False, MODE_DEFAULT)
        self._closing = False
        # Labels — detect / inpaint boxes in source pixels; editable overlay
        self._inpaint_boxes: list[tuple[int, int, int, int]] = []
        self._inpaint_quads: list[tuple[tuple[int, int], ...]] = []
        self._inpaint_layer_id: str = ""
        self._detect_boxes: list[tuple[int, int, int, int]] = []
        self._detect_quads: list[tuple[tuple[int, int], ...]] = []
        self._detect_roi: tuple[int, int, int, int] | None = None
        self._selected_detect: set[int] = set()
        self._label_mark_mode = False
        self._label_place_mode = False
        self._label_updating = False
        self.label_text = tk.StringVar(value="")
        self.label_size = tk.DoubleVar(value=float(LABEL_SIZE_DEFAULT))
        self.label_size_text = tk.StringVar(value=str(LABEL_SIZE_DEFAULT))
        self.label_color = tk.StringVar(value=LABEL_COLOR_DEFAULT)
        self.label_x = tk.StringVar(value="0")
        self.label_y = tk.StringVar(value="0")
        self.label_ocr_status = tk.StringVar(value=tesseract_status_text())
        self.label_font = tk.StringVar(value=LABEL_FONT_DEFAULT)
        self.wallpaper_style = tk.StringVar(value=WALLPAPER_GEOMETRIC_LABEL)
        self.layer_stack = LayerStack()
        self._layer_rows: list = []
        self._layer_drag_before: EditSnapshot | None = None
        self.pointer_tool = tk.StringVar(value=TOOL_VIEW_MOVE)
        self.pointer_tool_label = tk.StringVar(value=TOOL_VIEW_MOVE_LABEL)

        # layout mixin: paned columns, dock, Composite/Clusters notebook
        self._build_layout()
        self._apply_job_layout_defaults()
        self.root.bind("<Control-o>", lambda _e: self.open_image())
        self.root.bind("<Control-s>", lambda _e: self.save_image_as())
        self.root.bind("<Control-e>", lambda _e: self.export_pack())
        self.root.bind("<Escape>", self._on_escape_deselect, add="+")
        # bind_all so Ctrl+Z/Y work from a popped-out preview; skip while typing in an Entry
        self.root.bind_all("<Control-z>", self._on_undo_key)
        self.root.bind_all("<Control-Z>", self._on_undo_key)
        self.root.bind_all("<Control-y>", self._on_redo_key)
        self.root.bind_all("<Control-Y>", self._on_redo_key)
        self.root.bind_all("<Control-Shift-Z>", self._on_redo_key)
        self.root.bind_all("<Control-Shift-z>", self._on_redo_key)
        # Pointer-hover column scroll: bind_all is last, so also bind ttk classes.
        # TScale/Scale class binds are *replaced* so the slider cannot eat the wheel.
        self.root.bind_all("<MouseWheel>", self._on_column_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self._on_column_mousewheel, add="+")
        self.root.bind_all("<Button-5>", self._on_column_mousewheel, add="+")
        for cls_name in _SCALE_WHEEL_CLASSES:
            self.root.bind_class(cls_name, "<MouseWheel>", self._on_column_mousewheel)
            self.root.bind_class(cls_name, "<Button-4>", self._on_column_mousewheel)
            self.root.bind_class(cls_name, "<Button-5>", self._on_column_mousewheel)
        for cls_name in _WHEEL_CLASS_TAGS:
            self.root.bind_class(cls_name, "<MouseWheel>", self._on_column_mousewheel, add="+")
            self.root.bind_class(cls_name, "<Button-4>", self._on_column_mousewheel, add="+")
            self.root.bind_class(cls_name, "<Button-5>", self._on_column_mousewheel, add="+")
        self.left_column._bind_column_wheel_widgets()
        self.right_top_column._bind_column_wheel_widgets()
        self.right_bottom_column._bind_column_wheel_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self._on_app_close)


# Mixins bind these names at import time. Tests patch wallpaper_recolor.ui.app,
# so mixin call sites must look the names up here (not their own globals).
# ---------------------------------------------------------------------------
# Patch bridge: unittest patches this module; mixins imported the originals
# ---------------------------------------------------------------------------
import sys as _sys

import wallpaper_recolor.ui.mixins.adjust as _mix_adjust
import wallpaper_recolor.ui.mixins.chrome as _mix_chrome
import wallpaper_recolor.ui.mixins.layers_labels as _mix_layers
import wallpaper_recolor.ui.mixins.layout as _mix_layout
import wallpaper_recolor.ui.mixins.preview as _mix_preview
import wallpaper_recolor.ui.mixins.ranges as _mix_ranges
import wallpaper_recolor.ui.mixins.session as _mix_session


class _AppNameProxy:
    """Forward calls to this module so ``patch.object(ui.app, name)`` is visible."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __call__(self, *args, **kwargs):
        return getattr(_sys.modules[__name__], self._name)(*args, **kwargs)


for _mix in (
    _mix_adjust,
    _mix_chrome,
    _mix_layers,
    _mix_layout,
    _mix_preview,
    _mix_ranges,
    _mix_session,
):
    for _name in (
        "build_range_map",
        "composite_for_image",
        "save_image",
        "load_image",
        "_make_eyedrop_loupe_image",
    ):
        if hasattr(_mix, _name):
            setattr(_mix, _name, _AppNameProxy(_name))
del _mix, _name

