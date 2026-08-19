# -*- coding: utf-8 -*-
"""
wallpaper_recolor.ui.mixins.layout
------------------------------
Right-column paned layout, dock, sash, panel builders, View menu.

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


class AppLayoutMixin:
    """Right-column paned layout, dock, sash, panel builders, View menu."""

    # ---------------------------------------------------------------------------
    # Window body: left preview + two clipped right columns (true pack children)
    # ---------------------------------------------------------------------------
    def _build_layout(self) -> None:
        self._build_menubar()
        self.toolbar = ttk.Frame(self.root, padding=8)
        self.toolbar.pack(fill="x")
        toolbar = self.toolbar

        self.open_btn = ttk.Button(toolbar, text="Open image…", command=self.open_image)
        self.open_btn.pack(side="left", anchor="s")
        bind_tooltip(self.open_btn, "Open a TIF, PNG, or JPEG wallpaper.")

        self.tool_strip = ttk.Frame(toolbar)
        self.tool_strip.pack(side="left", padx=(10, 12), anchor="s")
        ttk.Label(self.tool_strip, text="Tools:").pack(side="left")
        self.tools_combo = ttk.Combobox(
            self.tool_strip,
            textvariable=self.pointer_tool_label,
            values=list(POINTER_TOOL_LABELS),
            state="readonly",
            width=12,
        )
        self.tools_combo.pack(side="left", padx=(4, 0))
        self._bind_readonly_combo(self.tools_combo, self._on_tools_combo)
        bind_tooltip(
            self.tools_combo,
            "View Move: pan and wheel-zoom the preview camera. "
            "Grab Move: drag the selected image inside the output frame (Position & Zoom X/Y).",
        )

        ttk.Label(toolbar, text="Presets:").pack(side="left")
        self.preset_combo = ttk.Combobox(
            toolbar,
            textvariable=self.preset_choice,
            values=[],
            state="readonly",
            width=18,
        )
        self.preset_combo.pack(side="left", padx=(4, 4))
        self._bind_readonly_combo(self.preset_combo, self.apply_selected_preset)
        ttk.Button(toolbar, text="Save preset…", command=self.save_preset).pack(side="left", padx=(0, 4))
        self.delete_preset_btn = ttk.Button(
            toolbar,
            text="Delete",
            command=self.delete_selected_preset,
            state="disabled",
        )
        self.delete_preset_btn.pack(side="left", padx=(0, 12))
        ensure_default_presets()
        self._reload_preset_combo()

        ttk.Label(toolbar, text="Ranges:").pack(side="left")
        spin = ttk.Spinbox(
            toolbar,
            from_=MIN_RANGES,
            to=MAX_RANGES,
            textvariable=self.range_count,
            width=4,
            command=self._on_range_count,
        )
        spin.pack(side="left", padx=(4, 12))
        spin.bind("<Return>", lambda _e: self._on_range_count())
        spin.bind("<FocusOut>", lambda _e: self._on_range_count())
        self.range_spin = spin
        bind_tooltip(
            spin,
            "Add or remove a color range. Existing match-from / change-to colors stay; "
            "the new range can take coverage, then pick a Pantone change-to.",
        )

        ttk.Label(toolbar, text="Range by:").pack(side="left")
        self.range_by_combo = ttk.Combobox(
            toolbar,
            textvariable=self.range_by,
            values=list(RANGE_BY_LABELS),
            state="readonly",
            width=20,
        )
        self.range_by_combo.pack(side="left", padx=(4, 12))
        self._bind_readonly_combo(self.range_by_combo, self._on_range_by)

        self.assign_caption = ttk.Label(toolbar, text="Assign:")
        self.assign_combo = ttk.Combobox(
            toolbar,
            textvariable=self.assign_label,
            values=list(ASSIGN_LABELS),
            state="readonly",
            width=20,
        )
        self._bind_readonly_combo(self.assign_combo, self._on_assign_mode)
        bind_tooltip(
            self.assign_combo,
            "Cluster from image: k-means on the scan (default). "
            "Snap to palette hexes: nearest named-preset Lab target.",
        )

        self.luma_kind_caption = ttk.Label(toolbar, text="Split:")
        self.luma_kind_combo = ttk.Combobox(
            toolbar,
            textvariable=self.luma_split_label,
            values=list(LUMA_SPLIT_LABELS.keys()),
            state="readonly",
            width=16,
        )
        self._bind_readonly_combo(self.luma_kind_combo, self.rebuild_ranges)

        self.bin_min_caption = ttk.Label(toolbar, text="Min %:")
        self.bin_min_spin = ttk.Spinbox(
            toolbar,
            from_=0,
            to=int(round(MIN_COVERAGE_MAX * 100.0)),
            increment=1,
            width=4,
            textvariable=self.bin_min_pct,
            command=self._on_bin_limits,
        )
        self.bin_min_spin.bind("<Return>", lambda _e: self._on_bin_limits())
        self.bin_min_spin.bind("<FocusOut>", lambda _e: self._on_bin_limits())
        bind_tooltip(
            self.bin_min_spin,
            "Luma coverage floor (0–40%). Steal / split will not shrink a range below this.",
        )
        self.bin_start_caption = ttk.Label(toolbar, text="Start:")
        self.bin_start_spin = ttk.Spinbox(
            toolbar,
            from_=0,
            to=255,
            increment=1,
            width=6,
            textvariable=self.bin_start,
            command=self._on_bin_limits,
        )
        self.bin_start_spin.bind("<Return>", lambda _e: self._on_bin_limits())
        self.bin_start_spin.bind("<FocusOut>", lambda _e: self._on_bin_limits())
        bind_tooltip(
            self.bin_start_spin,
            "Lower edge of range 1. Rec. 709 luma 0–255, L* 0–100, a*/b* −128–127. "
            "Pixels below Start stay original (unlabeled).",
        )

        self.reset_btn = ttk.Button(toolbar, text="Reset colors", command=self.reset_colors)
        self.reset_btn.pack(side="left")
        ttk.Button(toolbar, text="ICC profile…", command=self.pick_icc).pack(side="left", padx=(12, 0))
        self._sync_range_by_controls()

        # Two columns + sash: left = preview/coverage; right = vertical split (filters / layers)
        self.body_paned = ttk.Panedwindow(self.root, orient="horizontal")
        self.body_paned.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        self.left_column = ScrollColumn(self.body_paned, self, "left")
        self.right_host = _SashSplit(
            self.body_paned,
            orient="vertical",
            sashwidth=8,
            sashrelief="raised",
            bd=0,
            sashpad=2,
        )
        self.right_top_column = ScrollColumn(self.right_host, self, "right_top")
        self.right_bottom_column = ScrollColumn(self.right_host, self, "right_bottom")
        self.right_column = self.right_top_column  # alias: wheel + filter stack

        self.preview_panel = DockablePanel(
            self.left_column,
            self,
            "Preview",
            pop_label="Pop out preview",
            dock_label="Dock preview",
            pane_weight=4,
            pane_minsize=180,
            float_size="960x720",
            allow_pop_out=True,
            flex=True,
        )
        self.texture_panel = DockablePanel(
            self.right_top_column,
            self,
            "Texture",
            pane_weight=0,
            pane_minsize=56,
            float_size="520x110",
        )
        self.coverage_panel = DockablePanel(
            self.left_column,
            self,
            "Coverage",
            pane_weight=2,
            pane_minsize=140,
            float_size="720x340",
        )
        self.tone_panel = DockablePanel(
            self.right_top_column,
            self,
            TONE_PANEL_TITLE,
            pane_weight=0,
            pane_minsize=180,
            float_size="520x560",
        )
        self.scale_panel = DockablePanel(
            self.right_top_column,
            self,
            "Scale",
            pane_weight=0,
            pane_minsize=150,
            float_size="520x240",
        )
        self.crop_panel = DockablePanel(
            self.right_top_column,
            self,
            CROP_PANEL_TITLE,
            pane_weight=0,
            pane_minsize=140,
            float_size="520x220",
        )
        self.tess_panel = DockablePanel(
            self.right_bottom_column,
            self,
            "Tessellate",
            pane_weight=0,
            pane_minsize=160,
            float_size="520x280",
        )
        self.layers_panel = DockablePanel(
            self.right_bottom_column,
            self,
            "Layers",
            pane_weight=0,
            pane_minsize=160,
            float_size="520x280",
        )
        self.labels_panel = DockablePanel(
            self.right_bottom_column,
            self,
            "Labels",
            pane_weight=0,
            pane_minsize=160,
            float_size="520x340",
        )
        self.history_panel = DockablePanel(
            self.right_bottom_column,
            self,
            HISTORY_PANEL_TITLE,
            pane_weight=0,
            pane_minsize=80,
            float_size="520x220",
        )
        self.wheel_panel = DockablePanel(
            self.right_top_column,
            self,
            "Color wheel",
            pane_weight=1,
            pane_minsize=220,
            float_size="360x520",
        )
        self._all_panels = (
            self.preview_panel,
            self.texture_panel,
            self.coverage_panel,
            self.tone_panel,
            self.scale_panel,
            self.crop_panel,
            self.tess_panel,
            self.layers_panel,
            self.labels_panel,
            self.history_panel,
            self.wheel_panel,
        )
        # Default dock: all panes visible + expanded
        self.left_column.attach(self.preview_panel)
        self.left_column.attach(self.coverage_panel)
        self.right_top_column.attach(self.wheel_panel)
        self.right_top_column.attach(self.texture_panel)
        self.right_top_column.attach(self.tone_panel)
        self.right_top_column.attach(self.scale_panel)
        self.right_top_column.attach(self.crop_panel)
        self.right_bottom_column.attach(self.layers_panel)
        self.right_bottom_column.attach(self.labels_panel)
        self.right_bottom_column.attach(self.tess_panel)
        self.right_bottom_column.attach(self.history_panel)
        self._view_checks: dict[DockablePanel, tk.BooleanVar] = {}
        for panel in self._all_panels:
            self._view_checks[panel] = tk.BooleanVar(value=True)
        self._tab_checks: dict[str, tk.BooleanVar] = {}
        for title in INSPECTION_TAB_TITLES:
            self._tab_checks[title] = tk.BooleanVar(value=True)
        self._rebuild_view_menu()
        self._rebuild_edit_menu()
        self.right_host.add(
            self.right_top_column,
            minsize=RIGHT_SPLIT_MINSIZE,
            stretch="always",
            height=1,
        )
        self.right_host.add(
            self.right_bottom_column,
            minsize=RIGHT_SPLIT_MINSIZE,
            stretch="always",
            height=1,
        )
        try:
            self.right_host.pane(self.right_top_column, minsize=RIGHT_SPLIT_MINSIZE, weight=1)
            self.right_host.pane(self.right_bottom_column, minsize=RIGHT_SPLIT_MINSIZE, weight=1)
        except tk.TclError:
            pass
        self.body_paned.add(self.left_column, weight=LEFT_COL_WEIGHT)
        self.body_paned.add(self.right_host, weight=RIGHT_COL_WEIGHT)
        try:
            self.body_paned.pane(self.left_column, minsize=LEFT_COL_MINSIZE, weight=LEFT_COL_WEIGHT)
            self.body_paned.pane(self.right_host, minsize=RIGHT_COL_MINSIZE, weight=RIGHT_COL_WEIGHT)
        except tk.TclError:
            pass
        self._right_sash_init = False
        self.right_host.bind("<Configure>", self._on_right_host_configure)
        self._sash_after = self.root.after_idle(self._set_default_sash)
        self.root.bind("<Destroy>", self._cancel_idle_jobs, add="+")

        preview_body = self.preview_panel.body
        preview_body.columnconfigure(0, weight=1)
        preview_body.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(preview_body)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # --- Composite: original + live result (texture eye + slider; no separate flat pane) ---
        comp = ttk.Frame(self.notebook, padding=4)
        self._composite_tab = comp
        self.notebook.add(comp, text="Composite")
        for col in range(2):
            # uniform: 50/50 even when "Original (file.tif)" is wider than "Result"
            comp.columnconfigure(col, weight=1, uniform="preview")
        comp.rowconfigure(1, weight=1)
        self.orig_title_label = ttk.Label(comp, textvariable=self.orig_title)
        self.orig_title_label.grid(row=0, column=0, sticky="ew")
        self.result_title_label = ttk.Label(comp, text="Result")
        self.result_title_label.grid(row=0, column=1, sticky="ew")
        bind_tooltip(
            self.result_title_label,
            "Live recode: Texture slider plus range eyes.",
        )
        self.orig_host = tk.Frame(comp, bg=PREVIEW_PANE_BG)
        self.orig_host.grid(row=1, column=0, sticky="nsew", padx=(0, 3))
        self.orig_zoom_host = PreviewZoomHost(
            self.orig_host,
            self,
            on_click=self._on_original_click,
            on_rect=self._on_preview_mark_rect,
            share_pan=True,
        )
        self.orig_zoom_host.pack(fill="both", expand=True)
        self.orig_label = self.orig_zoom_host.image_label
        host_bg = str(self.orig_host.cget("bg"))
        self._eyedrop_overlay = tk.Canvas(
            self.orig_host,
            width=_EYEDROP_ICON_PX,
            height=_EYEDROP_ICON_PX,
            bd=0,
            highlightthickness=0,
            highlightbackground=host_bg,
            relief="flat",
            bg=host_bg,
            takefocus=0,
        )
        self._eyedrop_overlay.create_image(0, 0, image=self._eyedrop_photo, anchor="nw")
        self._eyedrop_overlay.image = self._eyedrop_photo  # type: ignore[attr-defined]
        tk.Misc.lift(self._eyedrop_overlay)
        self.tex_host = tk.Frame(comp, bg=PREVIEW_PANE_BG)
        self.tex_host.grid(row=1, column=1, sticky="nsew", padx=(3, 0))
        self.tex_zoom_host = PreviewZoomHost(
            self.tex_host,
            self,
            on_tap=self._on_result_click,
            on_rect=self._on_preview_mark_rect,
            share_pan=True,
        )
        self.tex_zoom_host.pack(fill="both", expand=True)
        self.tex_label = self.tex_zoom_host.image_label
        for widget in (
            self.orig_host,
            self.orig_zoom_host.viewport,
            self.orig_label,
        ):
            widget.bind("<Enter>", self._on_orig_eyedrop_move, add="+")
            widget.bind("<Motion>", self._on_orig_eyedrop_move, add="+")
            widget.bind("<Leave>", self._on_orig_eyedrop_leave, add="+")
        self._eyedrop_overlay.bind("<Button-1>", self._on_eyedrop_overlay_click)
        self._eyedrop_overlay.bind("<Motion>", self._on_eyedrop_overlay_move)
        self._eyedrop_overlay.bind("<Leave>", self._on_orig_eyedrop_leave)
        ttk.Label(comp, textvariable=self.scale_save_note).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

        self._build_clusters_tab()

        self.tile_zoom_host = PreviewZoomHost(self.notebook, self)
        self.tile_label = self.tile_zoom_host.image_label
        self.notebook.add(self.tile_zoom_host, text="3×3 tile")

        self.seam_zoom_host = PreviewZoomHost(self.notebook, self)
        self.seam_label = self.seam_zoom_host.image_label
        self.notebook.add(self.seam_zoom_host, text="Seam offset")

        mock = ttk.Frame(self.notebook, padding=4)
        self.mock_tab = mock
        self.notebook.add(mock, text="Room mockup")
        mock.rowconfigure(1, weight=1)
        mock.columnconfigure(0, weight=1)
        mock_ctrl = ttk.Frame(mock)
        mock_ctrl.grid(row=0, column=0, sticky="ew")
        mock_head = ttk.Frame(mock_ctrl)
        mock_head.pack(fill="x")
        ttk.Label(mock_head, textvariable=self.mockup_caption).pack(side="left")
        self.mockup_reset = self._make_slider_reset(mock_head, self._reset_mockup)
        self.mockup_scale = ttk.Scale(
            mock_ctrl,
            from_=1.0,
            to=12.0,
            variable=self.mockup_repeats,
            orient="horizontal",
            command=self._on_mockup_scale,
        )
        self.mockup_scale.pack(fill="x", pady=(2, 6))
        bind_tooltip(
            self.mockup_scale,
            "How many motif repeats across the back wall.",
        )
        cover_row = ttk.Frame(mock_ctrl)
        cover_row.pack(fill="x", pady=(0, 4))
        ttk.Label(cover_row, text="Wall cover:").pack(side="left")
        self.mockup_cover_radios = []
        for key in ("full", "half", "third", "quarter"):
            btn = ttk.Radiobutton(
                cover_row,
                text=MOCKUP_COVER_LABELS[key],
                value=key,
                variable=self.mockup_cover,
                command=self._on_mockup_cover,
            )
            btn.pack(side="left", padx=(4, 0))
            bind_tooltip(
                btn,
                "How much of the back wall is papered (from floor).",
            )
            self.mockup_cover_radios.append(btn)
        self.mock_zoom_host = PreviewZoomHost(mock, self)
        self.mock_zoom_host.grid(row=1, column=0, sticky="nsew")
        self.mock_label = self.mock_zoom_host.image_label
        self._build_preview_zoom_header()
        self._bind_preview_fit_resize()

        tex_head = ttk.Frame(self.texture_panel.body)
        tex_head.pack(fill="x")
        tex_bg = ttk.Style().lookup("TFrame", "background") or "#f0f0f0"
        self.texture_eye = EyeToggle(
            tex_head,
            self._eye_photos,
            self._on_texture_eye,
            bg=tex_bg,
            tooltip="On keeps original grain; off uses flat fills.",
        )
        self.texture_eye.pack(side="left")
        ttk.Label(tex_head, textvariable=self.texture_label).pack(side="left", padx=(8, 0))
        self.texture_reset = self._make_slider_reset(tex_head, self._reset_texture)
        self.texture_scale = ttk.Scale(
            self.texture_panel.body,
            from_=0.0,
            to=100.0,
            variable=self.texture_pct,
            orient="horizontal",
            command=self._on_texture_slider,
        )
        self.texture_scale.pack(fill="x", pady=(2, 0))
        bind_tooltip(
            self.texture_scale,
            "0% flat fills; 100% original grain with the new hues.",
        )

        cover_host = self.coverage_panel.body
        self.coverage = CoverageBar(
            cover_host,
            on_weights=self._on_coverage_weights,
            on_select=lambda i, h=None: self.select_range(i, h, toggle=True),
            on_toggle_visible=self._on_bar_toggle_visible,
            on_percent_commit=self.apply_typed_percent,
            on_edit_begin=self._mark_slider_begin,
            on_edit_end=self._mark_slider_end,
            on_eyedrop=self._on_eyedrop_button,
            eyedrop_photo=self._eyedrop_photo,
            eye_on_photo=self._eye_on_photo,
            eye_off_photo=self._eye_off_photo,
        )
        self.coverage.pack(fill="x", pady=(0, 0))

        cover_head = ttk.Frame(cover_host)
        cover_head.pack(fill="x", pady=(4, 0))
        self.cover_hint = tk.StringVar(value="")
        self.cover_reset = self._make_slider_reset(cover_head, self.reset_colors)

        self.chips_host = ttk.Frame(cover_host)
        # Range chips used to wrap under Coverage; ranges now live as sublayers.

        tone_host = self.tone_panel.body
        wb = ttk.LabelFrame(tone_host, text="White balance")
        wb.pack(fill="x", pady=(0, 6))
        (
            self.temperature_spin,
            self.temperature_knob,
            self.temperature_reset,
        ) = self._add_tone_spin_row(
            wb,
            "Temperature",
            self.temperature_pct,
            self._reset_temperature,
            "Warm (+) / cool (−). Gray World and White Patch set this. 0 is identity.",
        )
        (
            self.tint_spin,
            self.tint_knob,
            self.tint_reset,
        ) = self._add_tone_spin_row(
            wb,
            "Tint",
            self.tint_pct,
            self._reset_tint,
            "Magenta (+) / green (−). 0 is identity.",
        )
        lights_rgb = ttk.Frame(wb)
        lights_rgb.pack(fill="x", padx=(8, 0), pady=(2, 0))
        ttk.Label(lights_rgb, text="Highlight RGB").pack(anchor="w")
        (
            self.lights_reds_spin,
            self.lights_reds_knob,
            self.lights_reds_reset,
        ) = self._add_tone_spin_row(
            lights_rgb,
            "Reds",
            self.lights_reds_pct,
            lambda: self._reset_lights_rgb("reds"),
            "Highlight red multiply (White Patch region). Independent of Temperature.",
        )
        (
            self.lights_greens_spin,
            self.lights_greens_knob,
            self.lights_greens_reset,
        ) = self._add_tone_spin_row(
            lights_rgb,
            "Greens",
            self.lights_greens_pct,
            lambda: self._reset_lights_rgb("greens"),
            "Highlight green multiply. Independent of Tint.",
        )
        (
            self.lights_blues_spin,
            self.lights_blues_knob,
            self.lights_blues_reset,
        ) = self._add_tone_spin_row(
            lights_rgb,
            "Blues",
            self.lights_blues_pct,
            lambda: self._reset_lights_rgb("blues"),
            "Highlight blue multiply. Independent of Temperature.",
        )
        auto_wb = ttk.Frame(wb)
        auto_wb.pack(fill="x", pady=(4, 2))
        self.gray_world_btn = ttk.Button(
            auto_wb, text="Gray world", command=self._on_gray_world
        )
        self.gray_world_btn.pack(side="left")
        bind_tooltip(
            self.gray_world_btn,
            "Estimates Gray World white balance and sets Temperature / Tint. "
            "You can still nudge the numbers.",
        )
        self.white_patch_btn = ttk.Button(
            auto_wb, text="White patch", command=self._on_white_patch
        )
        self.white_patch_btn.pack(side="left", padx=(6, 0))
        bind_tooltip(
            self.white_patch_btn,
            "Estimates White Patch / max-RGB in highlights and sets Temperature / Tint.",
        )

        expo = ttk.LabelFrame(tone_host, text="Exposure")
        expo.pack(fill="x", pady=(0, 6))
        (
            self.exposure_spin,
            self.exposure_knob,
            self.exposure_reset,
        ) = self._add_tone_spin_row(
            expo,
            "Exposure",
            self.exposure_pct,
            self._reset_exposure,
            "Stops of exposure. +100 is +1 EV. 0 is identity.",
        )
        (
            self.brightness_spin,
            self.brightness_knob,
            self.brightness_reset,
        ) = self._add_tone_spin_row(
            expo,
            "Brightness",
            self.brightness_pct,
            self._reset_brightness,
            "Additive lift. 0 is identity.",
        )

        contra = ttk.LabelFrame(tone_host, text="Contrast")
        contra.pack(fill="x", pady=(0, 6))
        (
            self.contrast_spin,
            self.contrast_knob,
            self.contrast_reset,
        ) = self._add_tone_spin_row(
            contra,
            "Contrast",
            self.contrast_pct,
            self._reset_contrast,
            "Expand or flatten around mid-gray. 0 is identity.",
        )

        tone_luma = ttk.LabelFrame(tone_host, text="Darks / Lights")
        tone_luma.pack(fill="x", pady=(0, 6))
        (
            self.darks_spin,
            self.darks_knob,
            self.darks_reset,
        ) = self._add_tone_spin_row(
            tone_luma,
            "Darks",
            self.darks_pct,
            self._reset_darks,
            "Lift or crush the darks. 0 is identity.",
        )
        (
            self.lights_spin,
            self.lights_knob,
            self.lights_reset,
        ) = self._add_tone_spin_row(
            tone_luma,
            "Lights",
            self.lights_pct,
            self._reset_lights,
            "Lift or crush the lights. 0 is identity.",
        )
        norm_row = ttk.Frame(tone_luma)
        norm_row.pack(fill="x", pady=(4, 2))
        self.tess_normalize_btn = ttk.Button(
            norm_row, text="Normalize lighting", command=self._on_tess_normalize
        )
        self.tess_normalize_btn.pack(side="left")
        bind_tooltip(
            self.tess_normalize_btn,
            "Estimates a studio flatten, then sets Darks / Lights so the grade "
            "is visible and editable. Does not wrap or tessellate.",
        )
        self.tess_normalize_reset = self._make_slider_reset(
            norm_row, self._reset_tess_normalize
        )

        balance = ttk.LabelFrame(tone_host, text="Color balance (print CMY)")
        balance.pack(fill="x", pady=(0, 6))
        (
            self.balance_cyan_spin,
            self.balance_cyan_knob,
            self.balance_cyan_reset,
        ) = self._add_tone_spin_row(
            balance,
            "Cyan ↔ Red",
            self.balance_cyan_pct,
            lambda: self._reset_balance("cyan"),
            "Print pair: +Cyan adds C ink / pulls R; −Cyan adds red. "
            "Corrects ink vs screen RGB. 0 is identity.",
        )
        (
            self.balance_magenta_spin,
            self.balance_magenta_knob,
            self.balance_magenta_reset,
        ) = self._add_tone_spin_row(
            balance,
            "Magenta ↔ Green",
            self.balance_magenta_pct,
            lambda: self._reset_balance("magenta"),
            "Print pair: +Magenta adds M ink / pulls G; −Magenta adds green. 0 is identity.",
        )
        (
            self.balance_yellow_spin,
            self.balance_yellow_knob,
            self.balance_yellow_reset,
        ) = self._add_tone_spin_row(
            balance,
            "Yellow ↔ Blue",
            self.balance_yellow_pct,
            lambda: self._reset_balance("yellow"),
            "Print pair: +Yellow adds Y ink / pulls B; −Yellow adds blue. 0 is identity.",
        )

        sat = ttk.LabelFrame(tone_host, text="Saturation")
        sat.pack(fill="x", pady=(0, 4))
        (
            self.saturation_spin,
            self.saturation_knob,
            self.saturation_reset,
        ) = self._add_tone_spin_row(
            sat,
            "Saturation",
            self.saturation_pct,
            self._reset_saturation,
            "Luma-preserving chroma. −100 is gray; 0 is identity. "
            "Does not replace the Texture grain mix.",
        )

        # Trough-safe drag + arrows; MouseWheel stays page-scroll via _bind_wheel_tree
        self._wire_smooth_scale(self.mockup_scale, self.mockup_repeats, step=0.1, from_=1.0, to_=12.0)
        self._wire_smooth_scale(self.texture_scale, self.texture_pct, step=1, from_=0.0, to_=100.0)

        self._build_scale_panel()
        self._build_crop_panel()
        self._build_tessellate_panel()
        self._build_layers_panel()
        self._build_labels_panel()
        self._build_history_panel()

        # htmlcolorcodes-style wheel — edits whichever range is selected
        # Natural height (not expand) so sliders snapped under the wheel stay in the scroll stack
        self.wheel = ColorWheel(
            self.wheel_panel.body,
            on_color=self._on_wheel_color,
            on_color_commit=self._on_wheel_commit,
        )
        self.wheel.pack(anchor="n")
        region_tip = self.wheel._wheel_canvas_tip
        bind_tooltip(
            self.wheel.canvas,
            lambda e=None, region=region_tip: (
                f"{self.edit_caption.get().strip()}\n{region(e)}"
                if self.edit_caption.get().strip()
                else region(e)
            ).strip(),
        )

        # Footer: compact 200px busy slot + status caption. The slot stays packed
        # so start/stop never changes window height (unlike packing in the toolbar).
        # The Progressbar itself is unmapped while idle — Windows vista/xpnative
        # still paints a green stub at value 0 even after stop()/disabled.
        self.footer = ttk.Frame(self.root)
        self.footer.pack(fill="x", side="bottom")
        self.busy_bar = ttk.Frame(self.footer, width=200)
        self.busy_bar.pack(side="right", padx=(4, 8), pady=2)
        self.busy_progress = ttk.Progressbar(
            self.busy_bar, mode="indeterminate", length=200
        )
        try:
            slot_h = max(int(self.busy_progress.winfo_reqheight()), 16)
        except (tk.TclError, TypeError, ValueError):
            slot_h = 16
        self.busy_bar.configure(width=200, height=slot_h)
        self.busy_bar.pack_propagate(False)
        try:
            self.busy_progress.configure(value=0, state="disabled")
        except tk.TclError:
            pass
        self.busy_cancel = ttk.Button(
            self.footer, text="Cancel", command=self._on_busy_cancel, width=8
        )
        bind_tooltip(self.busy_cancel, "Stop Detect or Remove and keep the image unchanged.")
        self.status_bar = ttk.Label(
            self.footer, textvariable=self.status, padding=(8, 4), relief="sunken", anchor="w"
        )
        self.status_bar.pack(side="left", fill="x", expand=True)
        self._sync_slider_resets()
        self._select_composite_preview_tab()
        self._raise_window_chrome()

    def _wire_smooth_scale(self, scale, var, step=1, from_=None, to_=None) -> None:
        """Trough-safe drag + arrow ±step; wheel still page-scrolls the column."""
        _bind_smooth_scale(
            scale,
            var,
            step=step,
            from_=from_,
            to_=to_,
            on_begin=self._mark_slider_begin,
            on_end=self._mark_slider_end,
        )
        _bind_wheel_tree(scale, self._on_column_mousewheel)

    def _add_tone_spin_row(
        self,
        host: ttk.Frame,
        name: str,
        var: tk.DoubleVar,
        on_reset,
        tooltip: str,
    ) -> tuple[ttk.Spinbox, ToneKnob, tk.Label]:
        """Label + numeric stepper (−100…+100, ±1) + relative drag knob + hide-when-neutral Reset."""
        row = ttk.Frame(host)
        row.pack(fill="x", pady=(2, 0))
        ttk.Label(row, text=name).pack(side="left")
        spin = ttk.Spinbox(
            row,
            from_=TONE_SLIDER_MIN,
            to=TONE_SLIDER_MAX,
            increment=1,
            textvariable=var,
            width=7,
            wrap=False,
            command=lambda: self._on_tone_slider(""),
        )
        try:
            spin.configure(format="%.1f")
        except tk.TclError:
            pass
        spin.pack(side="left", padx=(8, 0))
        knob = ToneKnob(
            row,
            var,
            on_change=lambda: self._on_tone_slider(""),
            on_begin=self._mark_slider_begin,
            on_end=self._mark_slider_end,
            from_=TONE_SLIDER_MIN,
            to_=TONE_SLIDER_MAX,
            tooltip="Arrows ±1. Hold knob: up/right +, down/left −; farther = faster.",
        )
        knob.pack(side="left", padx=(6, 0))
        _bind_wheel_tree(knob, self._on_column_mousewheel)
        if not hasattr(self, "_tone_knobs"):
            self._tone_knobs = []
        self._tone_knobs.append(knob)
        if not hasattr(self, "_tone_spins"):
            self._tone_spins = []
        self._tone_spins.append(spin)
        reset = self._make_slider_reset(row, on_reset)
        spin.bind("<Return>", lambda _e, v=var, s=spin: self._commit_tone_spin(v, s))
        spin.bind("<FocusOut>", lambda _e, v=var, s=spin: self._commit_tone_spin(v, s))
        spin.bind("<FocusIn>", lambda e, s=spin: self._on_tone_spin_focus(e, s))
        spin.bind("<ButtonPress-1>", self._mark_slider_begin)
        spin.bind("<ButtonRelease-1>", self._mark_slider_end)
        spin.bind("<KeyPress-Up>", self._mark_slider_begin)
        spin.bind("<KeyPress-Down>", self._mark_slider_begin)
        spin.bind("<KeyRelease-Up>", self._mark_slider_end)
        spin.bind("<KeyRelease-Down>", self._mark_slider_end)
        bind_tooltip(spin, tooltip)
        _bind_wheel_tree(spin, self._on_column_mousewheel)
        return spin, knob, reset

    def _on_tone_spin_focus(self, event=None, spin=None) -> None:
        self._tone_focus_spin = spin
        self._mark_slider_begin(event)

    def _build_scale_panel(self) -> None:
        """Output size + DPI + resample — dockable like Tone; applies on Save / Export."""
        host = self.scale_panel.body
        ttk.Label(host, textvariable=self.scale_source_note).pack(anchor="w")

        unit_row = ttk.Frame(host)
        unit_row.pack(fill="x", pady=(4, 0))
        ttk.Label(unit_row, text="Units:").pack(side="left")
        unit_combo = ttk.Combobox(
            unit_row,
            textvariable=self.scale_unit,
            values=list(UNITS),
            state="readonly",
            width=14,
        )
        unit_combo.pack(side="left", padx=(4, 0))
        self._bind_readonly_combo(unit_combo, self._on_scale_unit)
        bind_tooltip(
            unit_combo,
            "Output units for W×H. Empty size keeps original pixels. Independent of preview zoom.",
        )

        size_row = ttk.Frame(host)
        size_row.pack(fill="x", pady=(4, 0))
        ttk.Label(size_row, text="W").pack(side="left")
        self.scale_width_entry = ttk.Entry(size_row, textvariable=self.scale_width, width=8)
        self.scale_width_entry.pack(side="left", padx=(4, 8))
        ttk.Label(size_row, text="×  H").pack(side="left")
        self.scale_height_entry = ttk.Entry(size_row, textvariable=self.scale_height, width=8)
        self.scale_height_entry.pack(side="left", padx=(4, 8))
        self.scale_lock_chk = ttk.Checkbutton(
            size_row,
            text="Lock aspect",
            variable=self.scale_lock,
            command=self._on_scale_lock,
        )
        self.scale_lock_chk.pack(side="left")
        bind_tooltip(self.scale_width_entry, "Output width. Empty keeps the source width.")
        bind_tooltip(self.scale_height_entry, "Output height. Empty keeps the source height.")
        bind_tooltip(self.scale_lock_chk, "Keep source proportions when typing W or H.")
        self.scale_width_entry.bind("<KeyRelease>", lambda _e: self._on_scale_width())
        self.scale_height_entry.bind("<KeyRelease>", lambda _e: self._on_scale_height())
        self.scale_width_entry.bind("<FocusOut>", lambda _e: self._on_scale_width())
        self.scale_height_entry.bind("<FocusOut>", lambda _e: self._on_scale_height())
        self.scale_width_entry.bind("<Return>", lambda _e: self._on_scale_width())
        self.scale_height_entry.bind("<Return>", lambda _e: self._on_scale_height())

        dpi_row = ttk.Frame(host)
        dpi_row.pack(fill="x", pady=(4, 0))
        ttk.Label(dpi_row, text="DPI:").pack(side="left")
        self.scale_dpi_combo = ttk.Combobox(
            dpi_row,
            textvariable=self.scale_dpi_choice,
            values=list(DPI_CHOICES),
            state="readonly",
            width=10,
        )
        self.scale_dpi_combo.pack(side="left", padx=(4, 8))
        self._bind_readonly_combo(self.scale_dpi_combo, self._on_scale_dpi_choice)
        self.scale_dpi_custom_entry = ttk.Entry(
            dpi_row, textvariable=self.scale_dpi_custom, width=7
        )
        self.scale_dpi_custom_entry.bind("<KeyRelease>", lambda _e: self._on_scale_dpi_custom())
        self.scale_dpi_custom_entry.bind("<FocusOut>", lambda _e: self._on_scale_dpi_custom())
        self.scale_dpi_custom_entry.bind("<Return>", lambda _e: self._on_scale_dpi_custom())
        bind_tooltip(
            self.scale_dpi_combo,
            "Tags the saved file. Used to convert inches/cm to pixels.",
        )
        bind_tooltip(self.scale_dpi_custom_entry, "Custom DPI when the list is set to Custom…")

        ttk.Label(host, textvariable=self.scale_equiv_note).pack(anchor="w", pady=(2, 0))

        filt_row = ttk.Frame(host)
        filt_row.pack(fill="x", pady=(4, 0))
        ttk.Label(filt_row, text="Resample:").pack(side="left")
        filt_combo = ttk.Combobox(
            filt_row,
            textvariable=self.scale_resample,
            values=list(RESAMPLE_LABELS),
            state="readonly",
            width=42,
        )
        filt_combo.pack(side="left", padx=(4, 0), fill="x", expand=True)
        self._bind_readonly_combo(filt_combo, self._refresh_scale_labels)
        bind_tooltip(filt_combo, "Resample filter used on Save / Export — not preview zoom.")
        ttk.Label(host, textvariable=self.scale_save_note).pack(anchor="w", pady=(4, 0))
        self._sync_dpi_custom_row()
        self._refresh_scale_labels()

    def _build_preview_zoom_header(self) -> None:
        """Compact Fit / − / slider / + / % / reset on the Preview title bar."""
        bar = self.preview_panel.bar
        ctrl = tk.Frame(bar, bg=_PANEL_BAR_BG, cursor="arrow")
        ctrl.pack(side="right", padx=(0, 4), pady=1)
        ttk.Label(ctrl, textvariable=self.view_zoom_title).pack(side="left", padx=(0, 4))
        self.preview_zoom_reset = self._make_slider_reset(
            ctrl, self._reset_preview_zoom, tip="Fit"
        )
        try:
            self.preview_zoom_reset.configure(bg=_PANEL_BAR_BG)
        except tk.TclError:
            pass
        minus = tk.Label(
            ctrl,
            image=self._zoom_out_photo,
            bd=0,
            highlightthickness=0,
            bg=_PANEL_BAR_BG,
            cursor="hand2",
            takefocus=0,
        )
        minus.image = self._zoom_out_photo  # type: ignore[attr-defined]
        minus.pack(side="left")
        minus.bind("<Button-1>", lambda _e: self._nudge_preview_zoom(-VIEW_ZOOM_PCT_STEP))
        plus = tk.Label(
            ctrl,
            image=self._zoom_in_photo,
            bd=0,
            highlightthickness=0,
            bg=_PANEL_BAR_BG,
            cursor="hand2",
            takefocus=0,
        )
        plus.image = self._zoom_in_photo  # type: ignore[attr-defined]
        plus.bind("<Button-1>", lambda _e: self._nudge_preview_zoom(VIEW_ZOOM_PCT_STEP))
        self.preview_zoom_scale = ttk.Scale(
            ctrl,
            from_=VIEW_ZOOM_PCT_MIN,
            to=VIEW_ZOOM_PCT_MAX,
            variable=self.preview_zoom,
            orient="horizontal",
            length=88,
            command=self._on_preview_zoom_slider,
        )
        self.preview_zoom_scale.pack(side="left", padx=4)
        plus.pack(side="left")
        ttk.Label(ctrl, textvariable=self.preview_zoom_caption, width=5).pack(side="left")
        fit = tk.Label(
            ctrl,
            text="Fit",
            bd=0,
            highlightthickness=0,
            bg=_PANEL_BAR_BG,
            fg=_PANEL_BAR_FG,
            cursor="hand2",
            takefocus=0,
            font=("Segoe UI", 8),
            padx=4,
        )
        fit.pack(side="left", padx=(4, 0))
        fit.bind("<Button-1>", lambda _e: self._reset_preview_zoom())
        self.preview_zoom_fit = fit
        self._preview_zoom_minus = minus
        self._preview_zoom_plus = plus
        self._view_zoom_tip = (
            "Composite: wallpaper fit. Clusters: Lab camera. Not Position & Zoom."
        )
        bind_tooltip(self.preview_zoom_scale, self._view_zoom_tip)
        bind_tooltip(minus, "Zoom the active Preview tab out. Display only.")
        bind_tooltip(plus, "Zoom the active Preview tab in. Display only.")
        bind_tooltip(fit, "Fit the whole image in the pane (contain). Display only.")
        self._wire_smooth_scale(
            self.preview_zoom_scale,
            self.preview_zoom,
            step=1,
            from_=VIEW_ZOOM_PCT_MIN,
            to_=VIEW_ZOOM_PCT_MAX,
        )

    def _build_crop_panel(self) -> None:
        """Center-offset X/Y (px) + zoom about the frame center."""
        host = self.crop_panel.body
        self.crop_x_scale, self.crop_x_entry, self.crop_x_reset = self._add_crop_xy_row(
            host, "X", self.crop_x, self.crop_x_text, "x"
        )
        self.crop_y_scale, self.crop_y_entry, self.crop_y_reset = self._add_crop_xy_row(
            host, "Y", self.crop_y, self.crop_y_text, "y"
        )

        zoom_head = ttk.Frame(host)
        zoom_head.pack(fill="x", pady=(8, 0))
        ttk.Label(zoom_head, text="Zoom").pack(side="left")
        self.crop_zoom_entry = ttk.Entry(zoom_head, textvariable=self.crop_zoom_text, width=8)
        self.crop_zoom_entry.pack(side="left", padx=(8, 0))
        self.crop_zoom_reset = self._make_slider_reset(zoom_head, self._reset_crop_zoom)
        self.crop_zoom_entry.bind("<Return>", lambda _e: self._commit_crop_entry("zoom"))
        self.crop_zoom_entry.bind("<FocusOut>", lambda _e: self._commit_crop_entry("zoom"))

        zoom_row = ttk.Frame(host)
        zoom_row.pack(fill="x", pady=(2, 0))
        bg = ttk.Style().lookup("TFrame", "background") or "#f0f0f0"
        minus = tk.Label(
            zoom_row,
            image=self._zoom_out_photo,
            bd=0,
            highlightthickness=0,
            bg=bg,
            takefocus=0,
        )
        minus.image = self._zoom_out_photo  # type: ignore[attr-defined]
        minus.pack(side="left")
        plus = tk.Label(
            zoom_row,
            image=self._zoom_in_photo,
            bd=0,
            highlightthickness=0,
            bg=bg,
            takefocus=0,
        )
        plus.image = self._zoom_in_photo  # type: ignore[attr-defined]
        plus.pack(side="right")
        self.crop_zoom_scale = ttk.Scale(
            zoom_row,
            from_=ZOOM_MIN,
            to=ZOOM_MAX,
            variable=self.crop_zoom,
            orient="horizontal",
            command=self._on_crop_zoom_slider,
        )
        self.crop_zoom_scale.pack(side="left", fill="x", expand=True, padx=4)
        self._zoom_minus_icon = minus
        self._zoom_plus_icon = plus
        bind_tooltip(
            self.crop_zoom_scale,
            "Crop zoom about the frame center. 1 is the full image — not Preview zoom.",
        )
        bind_tooltip(
            self.crop_zoom_entry,
            "Typed crop zoom about the frame center. 1 is the full image — not Preview zoom.",
        )
        bind_tooltip(minus, "Zoom out (image smaller in the frame).")
        bind_tooltip(plus, "Zoom in (image larger about the frame center).")

        lim_x, lim_y = offset_slider_limit(1, 1, ZOOM_DEFAULT)
        self._wire_smooth_scale(self.crop_x_scale, self.crop_x, step=1, from_=float(-lim_x))
        self._wire_smooth_scale(self.crop_y_scale, self.crop_y, step=1, from_=float(-lim_y))
        self._wire_smooth_scale(
            self.crop_zoom_scale, self.crop_zoom, step=0.1, from_=ZOOM_MIN, to_=ZOOM_MAX
        )
        self._sync_crop_bounds(clamp=True)
        self._sync_crop_entry_text()

    def _add_crop_xy_row(
        self,
        host: ttk.Frame,
        label: str,
        var: tk.DoubleVar,
        text_var: tk.StringVar,
        which: str,
    ) -> tuple[ttk.Scale, ttk.Entry, tk.Label]:
        """Label + typed px + reset on one row, slider under it."""
        head = ttk.Frame(host)
        head.pack(fill="x", pady=(6, 0))
        ttk.Label(head, text=f"{label} (px from center)").pack(side="left")
        entry = ttk.Entry(head, textvariable=text_var, width=8)
        entry.pack(side="left", padx=(8, 0))
        reset = self._make_slider_reset(
            head, self._reset_crop_x if which == "x" else self._reset_crop_y
        )
        entry.bind("<Return>", lambda _e, w=which: self._commit_crop_entry(w))
        entry.bind("<FocusOut>", lambda _e, w=which: self._commit_crop_entry(w))
        scale = ttk.Scale(
            host,
            from_=-1.0,
            to=1.0,
            variable=var,
            orient="horizontal",
            command=self._on_crop_xy_slider,
        )
        scale.pack(fill="x")
        axis = "X" if which == "x" else "Y"
        bind_tooltip(
            scale,
            f"{axis} offset of the image center from the frame center, in source pixels. "
            "0 is centered. Empty frame is checker (preview) / transparent (save).",
        )
        bind_tooltip(
            entry,
            f"Typed {axis} offset from the frame center, in source pixels (may be negative).",
        )
        return scale, entry, reset

    def _build_tessellate_panel(self) -> None:
        """Mode dropdown + side radios + Build — tile, tessellation, mesh, mosaic."""
        host = self.tess_panel.body
        mode_head = ttk.Frame(host)
        mode_head.pack(fill="x")
        ttk.Label(mode_head, text="Mode").pack(side="left")
        self.tess_mode_reset = self._make_slider_reset(mode_head, self._reset_tess_mode)
        mode_row = ttk.Frame(host)
        mode_row.pack(fill="x")
        self.tess_mode_combo = ttk.Combobox(
            mode_row,
            textvariable=self.tess_mode_label,
            values=list(MODE_LABELS),
            state="readonly",
            width=28,
        )
        self.tess_mode_combo.pack(side="left", fill="x", expand=True)
        self._bind_readonly_combo(self.tess_mode_combo, self._on_tess_mode_combo)
        bind_tooltip(
            self.tess_mode_combo,
            "Tile (Repeating Design): crop to whole repeats, resize to the tile, "
            "and fill leftover edges from the pattern so it wraps. "
            "Tessellation: Hilbert (crinkly) diffuse. Mesh: warp / grid stretch. "
            "Detail mosaic: Voronoi fill by image detail.",
        )

        h_head = ttk.Frame(host)
        h_head.pack(fill="x", pady=(8, 0))
        ttk.Label(h_head, text="Horizontal").pack(side="left")
        self.tess_h_reset = self._make_slider_reset(h_head, self._reset_tess_h)
        h_row = ttk.Frame(host)
        h_row.pack(fill="x")
        self.tess_h_radios = []
        for value, label in (
            (SIDE_OFF, "Off"),
            (SIDE_LEFT, "Left"),
            (SIDE_RIGHT, "Right"),
        ):
            btn = ttk.Radiobutton(
                h_row,
                text=label,
                value=value,
                variable=self.tess_h,
                command=self._on_tess_side,
            )
            btn.pack(side="left", padx=(0, 8))
            bind_tooltip(btn, _TESS_H_TIPS[value])
            self.tess_h_radios.append(btn)

        v_head = ttk.Frame(host)
        v_head.pack(fill="x", pady=(8, 0))
        ttk.Label(v_head, text="Vertical").pack(side="left")
        self.tess_v_reset = self._make_slider_reset(v_head, self._reset_tess_v)
        v_row = ttk.Frame(host)
        v_row.pack(fill="x")
        self.tess_v_radios = []
        for value, label in (
            (SIDE_OFF, "Off"),
            (SIDE_TOP, "Top"),
            (SIDE_BOTTOM, "Bottom"),
        ):
            btn = ttk.Radiobutton(
                v_row,
                text=label,
                value=value,
                variable=self.tess_v,
                command=self._on_tess_side,
            )
            btn.pack(side="left", padx=(0, 8))
            bind_tooltip(btn, _TESS_V_TIPS[value])
            self.tess_v_radios.append(btn)

        self.tess_mosaic_host = ttk.Frame(host)
        tiles_head = ttk.Frame(self.tess_mosaic_host)
        tiles_head.pack(fill="x", pady=(8, 0))
        ttk.Label(tiles_head, text="Tiles").pack(side="left")
        self.tess_tiles_entry = ttk.Entry(
            tiles_head, textvariable=self.tess_tiles_text, width=8
        )
        self.tess_tiles_entry.pack(side="left", padx=(8, 0))
        self.tess_tiles_reset = self._make_slider_reset(tiles_head, self._reset_tess_tiles)
        self.tess_tiles_entry.bind("<Return>", lambda _e: self._commit_tess_tiles())
        self.tess_tiles_entry.bind("<FocusOut>", lambda _e: self._commit_tess_tiles())
        self.tess_tiles_scale = ttk.Scale(
            self.tess_mosaic_host,
            from_=float(TILES_MIN),
            to=float(TILES_MAX),
            variable=self.tess_tiles,
            orient="horizontal",
            command=self._on_tess_tiles_slider,
        )
        self.tess_tiles_scale.pack(fill="x", pady=(2, 0))
        bind_tooltip(
            self.tess_tiles_scale,
            "Voronoi cell count for Detail mosaic.",
        )
        bind_tooltip(self.tess_tiles_entry, "Typed Voronoi cell count for Detail mosaic.")
        lloyd_head = ttk.Frame(self.tess_mosaic_host)
        lloyd_head.pack(fill="x", pady=(8, 0))
        ttk.Label(lloyd_head, text="Lloyd").pack(side="left")
        self.tess_lloyd_entry = ttk.Entry(
            lloyd_head, textvariable=self.tess_lloyd_text, width=8
        )
        self.tess_lloyd_entry.pack(side="left", padx=(8, 0))
        self.tess_lloyd_reset = self._make_slider_reset(lloyd_head, self._reset_tess_lloyd)
        self.tess_lloyd_entry.bind("<Return>", lambda _e: self._commit_tess_lloyd())
        self.tess_lloyd_entry.bind("<FocusOut>", lambda _e: self._commit_tess_lloyd())
        self.tess_lloyd_scale = ttk.Scale(
            self.tess_mosaic_host,
            from_=float(LLOYD_MIN),
            to=float(LLOYD_MAX),
            variable=self.tess_lloyd,
            orient="horizontal",
            command=self._on_tess_lloyd_slider,
        )
        self.tess_lloyd_scale.pack(fill="x", pady=(2, 0))
        bind_tooltip(
            self.tess_lloyd_scale,
            "Lloyd relax iterations that even out mosaic cells.",
        )
        bind_tooltip(
            self.tess_lloyd_entry,
            "Typed Lloyd iterations that even out mosaic cells.",
        )
        self._wire_smooth_scale(
            self.tess_tiles_scale,
            self.tess_tiles,
            step=1,
            from_=float(TILES_MIN),
            to_=float(TILES_MAX),
        )
        self._wire_smooth_scale(
            self.tess_lloyd_scale,
            self.tess_lloyd,
            step=1,
            from_=float(LLOYD_MIN),
            to_=float(LLOYD_MAX),
        )

        build_row = ttk.Frame(host)
        build_row.pack(fill="x", pady=(8, 0))
        self.tess_build_btn = ttk.Button(
            build_row, text="Build", command=self._on_tess_build
        )
        self.tess_build_btn.pack(side="left")
        bind_tooltip(
            self.tess_build_btn,
            "Run the selected mode and write the wrap crop to Position & Zoom.",
        )
        self.tess_build_reset = self._make_slider_reset(build_row, self._reset_tess_build)
        self._sync_tess_mosaic_controls()

    def _build_history_panel(self) -> None:
        """Undo stack as a clickable list (most recent toward the top)."""
        host = self.history_panel.body
        cap = ttk.Label(host, text="Click a row to jump. Same snapshots as Ctrl+Z.")
        cap.pack(anchor="w")
        self.history_list = tk.Listbox(host, height=6, exportselection=False, activestyle="none")
        self.history_list.pack(fill="both", expand=True, pady=(4, 0))
        self.history_list.bind("<<ListboxSelect>>", self._on_history_list_select)
        self._history_rows: list[tuple[str, int]] = []
        self._history_list_mute = False
        self._refresh_history_list()

    def _build_layers_panel(self) -> None:
        """Stack list: eye, name, type. Higher row paints on top."""
        host = self.layers_panel.body
        btn_row = ttk.Frame(host)
        btn_row.pack(fill="x", pady=(0, 4))
        self.layers_add_image_btn = ttk.Button(
            btn_row, text="Add image", command=self._on_add_image_layer
        )
        self.layers_add_image_btn.pack(side="left")
        bind_tooltip(
            self.layers_add_image_btn,
            "Open another image as an overlay layer (motif, logo).",
        )
        self.layers_add_label_btn = ttk.Button(
            btn_row, text="Add label", command=self._on_add_label_layer
        )
        self.layers_add_label_btn.pack(side="left", padx=(6, 0))
        bind_tooltip(
            self.layers_add_label_btn,
            "Create a text layer. Place it on Original or Result.",
        )
        self.layers_up_btn = ttk.Button(btn_row, text="Up", command=self._on_layer_up)
        self.layers_up_btn.pack(side="left", padx=(6, 0))
        bind_tooltip(self.layers_up_btn, "Move the selected layer toward the front.")
        self.layers_down_btn = ttk.Button(
            btn_row, text="Down", command=self._on_layer_down
        )
        self.layers_down_btn.pack(side="left", padx=(6, 0))
        bind_tooltip(self.layers_down_btn, "Move the selected layer toward the back.")
        self.layers_list = ttk.Frame(host)
        self.layers_list.pack(fill="x")
        host.bind("<Button-1>", self._on_layers_bg_click)
        self.layers_list.bind("<Button-1>", self._on_layers_bg_click)
        self._layer_rows = []
        self._layer_range_rows = {}
        self._refresh_layers_panel()

    def _build_labels_panel(self) -> None:
        """Detect / remove wallpaper text, then an editable label layer."""
        host = self.labels_panel.body
        self.labels_ocr_label = ttk.Label(
            host, textvariable=self.label_ocr_status, wraplength=320
        )
        if self.label_ocr_status.get():
            self.labels_ocr_label.pack(anchor="w", pady=(0, 0))

        btn_row = ttk.Frame(host)
        btn_row.pack(fill="x", pady=(0, 0))
        self.labels_detect_btn = ttk.Button(
            btn_row, text="Detect", command=self._on_label_detect
        )
        self.labels_detect_btn.pack(side="left")
        bind_tooltip(
            self.labels_detect_btn,
            "Find text painted into the wallpaper (EasyOCR when installed, else "
            "Tesseract/contrast). Runs on the preview or Select area, not the 12k file. "
            "Not Label layers you added.",
        )
        self.labels_remove_btn = ttk.Button(
            btn_row, text="Remove", command=self._on_label_remove
        )
        self.labels_remove_btn.pack(side="left", padx=(6, 0))
        bind_tooltip(
            self.labels_remove_btn,
            "Fill EasyOCR polygons (closed into a line bar, dilated) on a padded "
            "source-resolution crop: LaMa ONNX if cached, else OpenCV NS, else numpy. "
            "Geometric spans stripes; Floral stays tight. Label layers stay on top.",
        )
        self.labels_clear_btn = ttk.Button(
            btn_row, text="Clear", command=self._on_label_clear
        )
        self.labels_clear_btn.pack(side="left", padx=(6, 0))
        bind_tooltip(self.labels_clear_btn, "Clear detections and the selection.")
        self.labels_mark_btn = ttk.Button(
            btn_row, text="Select area", command=self._on_label_mark_toggle
        )
        self.labels_mark_btn.pack(side="left", padx=(6, 0))
        bind_tooltip(
            self.labels_mark_btn,
            "Drag a rectangle on Original or Result to limit Detect.",
        )

        style_row = ttk.Frame(host)
        style_row.pack(fill="x", pady=(6, 0))
        ttk.Label(style_row, text="Wallpaper").pack(side="left")
        self.wallpaper_style_combo = ttk.Combobox(
            style_row,
            textvariable=self.wallpaper_style,
            values=list(WALLPAPER_STYLE_LABELS),
            state="readonly",
            width=20,
        )
        self.wallpaper_style_combo.pack(side="left", padx=(8, 0))
        self._bind_readonly_combo(self.wallpaper_style_combo, lambda: None)
        bind_tooltip(
            self.wallpaper_style_combo,
            "Geometric (default, 122-LA4): horizontal 15×3 close, 5×5 dilate, "
            "drop-shadow +6 px. Floral: tighter 7×3 close, 3×3 dilate.",
        )

        ttk.Label(host, text="Label layer", font=("Segoe UI", 9, "bold")).pack(
            anchor="w", pady=(10, 0)
        )
        ttk.Label(host, text="Text").pack(anchor="w", pady=(4, 0))
        self.label_text_entry = ttk.Entry(host, textvariable=self.label_text)
        self.label_text_entry.pack(fill="x")
        self.label_text_entry.bind("<Return>", lambda _e: self._commit_label_fields())
        self.label_text_entry.bind("<FocusOut>", lambda _e: self._commit_label_fields())
        bind_tooltip(self.label_text_entry, "Text drawn on the label layer.")

        size_head = ttk.Frame(host)
        size_head.pack(fill="x", pady=(8, 0))
        ttk.Label(size_head, text="Font size").pack(side="left")
        self.label_size_entry = ttk.Entry(
            size_head, textvariable=self.label_size_text, width=6
        )
        self.label_size_entry.pack(side="left", padx=(8, 0))
        self.label_size_reset = self._make_slider_reset(size_head, self._reset_label_size)
        self.label_size_entry.bind("<Return>", lambda _e: self._commit_label_size())
        self.label_size_entry.bind("<FocusOut>", lambda _e: self._commit_label_size())
        self.label_size_scale = ttk.Scale(
            host,
            from_=float(LABEL_SIZE_MIN),
            to=float(LABEL_SIZE_MAX),
            variable=self.label_size,
            orient="horizontal",
            command=self._on_label_size_slider,
        )
        self.label_size_scale.pack(fill="x", pady=(2, 0))
        bind_tooltip(self.label_size_scale, "Font size of the label layer.")
        bind_tooltip(self.label_size_entry, "Typed font size of the label layer.")
        self._wire_smooth_scale(
            self.label_size_scale,
            self.label_size,
            step=1,
            from_=float(LABEL_SIZE_MIN),
            to_=float(LABEL_SIZE_MAX),
        )

        font_row = ttk.Frame(host)
        font_row.pack(fill="x", pady=(8, 0))
        ttk.Label(font_row, text="Font").pack(side="left")
        families = list_font_families(self.root)
        if LABEL_FONT_DEFAULT not in families:
            families = [LABEL_FONT_DEFAULT] + families
        self.label_font_combo = ttk.Combobox(
            font_row,
            textvariable=self.label_font,
            values=families,
            state="readonly",
            width=18,
        )
        self.label_font_combo.pack(side="left", padx=(8, 0), fill="x", expand=True)
        self._bind_readonly_combo(self.label_font_combo, self._on_label_font)
        bind_tooltip(
            self.label_font_combo,
            "Font family for the selected label. Preview and export rasterize with PIL.",
        )

        color_row = ttk.Frame(host)
        color_row.pack(fill="x", pady=(8, 0))
        ttk.Label(color_row, text="Color").pack(side="left")
        self.label_color_entry = ttk.Entry(
            color_row, textvariable=self.label_color, width=10
        )
        self.label_color_entry.pack(side="left", padx=(8, 0))
        self.label_color_entry.bind("<Return>", lambda _e: self._commit_label_fields())
        self.label_color_entry.bind("<FocusOut>", lambda _e: self._commit_label_fields())
        self.label_use_change_btn = ttk.Button(
            color_row, text="Change-to", command=self._label_use_change_to
        )
        self.label_use_change_btn.pack(side="left", padx=(6, 0))
        bind_tooltip(self.label_color_entry, "Label color as #RRGGBB.")
        bind_tooltip(
            self.label_use_change_btn,
            "Use the selected range’s change-to color.",
        )

        pos_row = ttk.Frame(host)
        pos_row.pack(fill="x", pady=(8, 0))
        ttk.Label(pos_row, text="X").pack(side="left")
        self.label_x_entry = ttk.Entry(pos_row, textvariable=self.label_x, width=7)
        self.label_x_entry.pack(side="left", padx=(4, 8))
        ttk.Label(pos_row, text="Y").pack(side="left")
        self.label_y_entry = ttk.Entry(pos_row, textvariable=self.label_y, width=7)
        self.label_y_entry.pack(side="left", padx=(4, 0))
        self.label_place_btn = ttk.Button(
            pos_row, text="Place", command=self._on_label_place_toggle
        )
        self.label_place_btn.pack(side="left", padx=(8, 0))
        bind_tooltip(self.label_x_entry, "Label X in source pixels.")
        bind_tooltip(self.label_y_entry, "Label Y in source pixels.")
        bind_tooltip(self.label_place_btn, "Click Original or Result to set X/Y.")
        self.label_x_entry.bind("<Return>", lambda _e: self._commit_label_fields())
        self.label_x_entry.bind("<FocusOut>", lambda _e: self._commit_label_fields())
        self.label_y_entry.bind("<Return>", lambda _e: self._commit_label_fields())
        self.label_y_entry.bind("<FocusOut>", lambda _e: self._commit_label_fields())
        self._sync_label_modes()

    def _build_clusters_tab(self) -> None:
        """Preview notebook tab: Lab k-means scatter (matplotlib 3D or 2D fallback)."""
        self.cluster_plot = ClusterPlot(self.notebook)
        self.notebook.add(self.cluster_plot, text="Clusters")
        self._bind_readonly_combo(self.cluster_plot.mode_combo, self._schedule_cluster_view)
        self.cluster_plot.on_pick = self._on_cluster_pick
        self.cluster_plot.on_zoom = self._on_cluster_camera_zoom
        self.cluster_plot.on_selected_rgb = self._cluster_selected_rgb
        self.cluster_plot.on_move_start = self._on_cluster_move_start
        self.cluster_plot.on_move = self._on_cluster_move
        self.cluster_plot.on_move_end = self._on_cluster_move_end

    def _cancel_idle_jobs(self, event=None) -> None:
        if event is not None and event.widget is not self.root:
            return
        job = getattr(self, "_sash_after", None)
        if job is not None:
            try:
                self.root.after_cancel(job)
            except tk.TclError:
                pass
            self._sash_after = None
        fit_job = getattr(self, "_fit_job", None)
        if fit_job is not None:
            try:
                self.root.after_cancel(fit_job)
            except tk.TclError:
                pass
            self._fit_job = None
        self._cancel_preview_job()
        cluster_job = getattr(self, "_cluster_job", None)
        if cluster_job is not None:
            try:
                self.root.after_cancel(cluster_job)
            except tk.TclError:
                pass
            self._cluster_job = None

    def _set_default_sash(self) -> None:
        """Left column starts wider; right split is half / half."""
        try:
            if not self.body_paned.winfo_exists():
                return
            self.body_paned.update_idletasks()
            width = int(self.body_paned.winfo_width())
            if width < 80:
                return
            self.body_paned.sashpos(0, int(width * LEFT_SASH_FRACTION))
            if self._apply_right_sash_fraction(RIGHT_SPLIT_FRACTION):
                self._right_sash_init = True
            for col in self._dock_columns():
                col._sync_layout()
        except tk.TclError:
            pass

    def _apply_right_sash_fraction(self, fraction: float) -> bool:
        """Place the vertical right sash; True when ``right_host`` is tall enough."""
        host = getattr(self, "right_host", None)
        if host is None:
            return False
        try:
            host.update_idletasks()
            height = int(host.winfo_height())
            if height < 80:
                return False
            minsz = min(RIGHT_SPLIT_MINSIZE, max(24, height // 4))
            pos = int(round(height * float(fraction)))
            pos = max(minsz, min(height - minsz, pos))
            host.sashpos(0, pos)
            return True
        except (tk.TclError, TypeError, ValueError):
            return False

    def _on_right_host_configure(self, event) -> None:
        """First time the right split is tall enough, lock the sash at 50%."""
        if event.widget is not getattr(self, "right_host", None):
            return
        if getattr(self, "_right_sash_init", False):
            return
        try:
            if int(event.height) < 80:
                return
        except (TypeError, ValueError):
            return
        if self._apply_right_sash_fraction(RIGHT_SPLIT_FRACTION):
            self._right_sash_init = True
            for col in self._dock_columns():
                try:
                    col._sync_layout()
                except tk.TclError:
                    pass

    def _on_column_mousewheel(self, event) -> str | None:
        """Scroll the column under the pointer — never the slider under it.

        Windows sends MouseWheel to the *focused* widget; bind_all is last so
        ttk.Scale / Notebook can swallow it unless we replace the TScale class
        bind and return break. bind_tree + class binds run first.
        Docked panes are true children of each column canvas, so
        ``winfo_containing`` / pack-in / ``contains_root`` still hit-test the
        pointer; combobox dropdowns live in another Toplevel — leave those alone.
        """
        if self._wheel_over_combobox_popdown(event):
            return None
        if self._on_preview_ctrl_wheel(event):
            return "break"
        if self._on_tone_spin_wheel(event):
            return "break"
        for x, y in self._wheel_event_xy(event):
            col = None
            for candidate in self._dock_columns():
                if candidate.contains_root(x, y):
                    col = candidate
                    break
            if col is None:
                target = None
                try:
                    target = self.root.winfo_containing(x, y)
                except tk.TclError:
                    target = None
                col = self._column_from_widget(target) if target is not None else None
            if col is None:
                widget = getattr(event, "widget", None)
                if isinstance(widget, str):
                    try:
                        widget = self.root.nametowidget(widget)
                    except (KeyError, tk.TclError):
                        widget = None
                if widget is not None and isinstance(widget, tk.Misc):
                    col = self._column_from_widget(widget)
            if col is not None:
                return col._on_mousewheel(event)
        return "break"

    def _on_tone_spin_wheel(self, event) -> str | None:
        """Focused Color & lighting spin: wheel ±1 when the pointer is over that row."""
        spin = getattr(self, "_tone_focus_spin", None)
        if spin is None:
            return None
        try:
            focused = self.root.focus_get()
        except tk.TclError:
            focused = None
        if focused is not None and focused is not spin:
            return None
        row = getattr(spin, "master", None)
        over = False
        for x, y in self._wheel_event_xy(event):
            if row is not None and _widget_contains_root(row, x, y):
                over = True
                break
            if _widget_contains_root(spin, x, y):
                over = True
                break
        if not over:
            return None
        delta = getattr(event, "delta", 0) or 0
        try:
            delta = int(delta)
        except (TypeError, ValueError):
            delta = 0
        num = getattr(event, "num", 0)
        try:
            num = int(num)
        except (TypeError, ValueError):
            num = 0
        step = 0
        if delta:
            step = 1 if delta > 0 else -1
        elif num == 4:
            step = 1
        elif num == 5:
            step = -1
        if step == 0:
            return "break"
        try:
            val = float(spin.get())
        except (tk.TclError, TypeError, ValueError):
            val = 0.0
        val = max(TONE_SLIDER_MIN, min(TONE_SLIDER_MAX, val + step))
        spin.set(val)
        self._on_tone_slider("")
        return "break"

    def _wheel_event_xy(self, event) -> list[tuple[int, int]]:
        """Screen coords: event.x_root first (tests), then the OS pointer (Windows)."""
        out: list[tuple[int, int]] = []
        try:
            out.append((int(event.x_root), int(event.y_root)))
        except (tk.TclError, AttributeError, TypeError, ValueError):
            pass
        try:
            out.append((int(self.root.winfo_pointerx()), int(self.root.winfo_pointery())))
        except tk.TclError:
            pass
        uniq: list[tuple[int, int]] = []
        for pair in out:
            if pair not in uniq:
                uniq.append(pair)
        return uniq

    def _wheel_over_combobox_popdown(self, event) -> bool:
        """True when the pointer is over a ttk Combobox list (separate Toplevel)."""
        widgets: list[tk.Misc] = []
        w = getattr(event, "widget", None)
        if isinstance(w, str):
            try:
                w = self.root.nametowidget(w)
            except (KeyError, tk.TclError):
                w = None
        if w is not None and isinstance(w, tk.Misc):
            widgets.append(w)
        try:
            hit = self.root.winfo_containing(int(self.root.winfo_pointerx()), int(self.root.winfo_pointery()))
            if hit is not None:
                widgets.append(hit)
        except tk.TclError:
            pass
        for widget in widgets:
            try:
                top = widget.winfo_toplevel()
                cls = str(top.winfo_class())
            except (tk.TclError, AttributeError):
                continue
            if cls == "ComboboxPopdown" or "popdown" in cls.lower():
                return True
        return False

    def _column_from_widget(self, widget: tk.Misc) -> ScrollColumn | None:
        """Walk parent (panels are true children of each scroller inner)."""
        current: tk.Misc | None = widget
        seen: set[str] = set()
        while current is not None:
            key = str(current)
            if key in seen:
                break
            seen.add(key)
            for col in self._dock_columns():
                if current is col or current is col.inner or current is col.canvas:
                    return col
            col = getattr(current, "column", None)
            if isinstance(col, ScrollColumn) and not bool(getattr(current, "is_floating", False)):
                return col
            nxt: tk.Misc | None = None
            try:
                info = current.pack_info()
                geom = info.get("in")
                if isinstance(geom, str):
                    geom = current.nametowidget(geom)
                for col in self._dock_columns():
                    if geom is col.inner:
                        return col
                if geom is not None and isinstance(geom, tk.Misc):
                    nxt = geom
            except (tk.TclError, KeyError, AttributeError):
                pass
            try:
                parent_path = current.winfo_parent()
                parent = current.nametowidget(parent_path) if parent_path else None
            except (tk.TclError, KeyError):
                parent = None
            current = parent if parent is not None else nxt
        return None

    def _dock_columns(self) -> tuple[ScrollColumn, ...]:
        cols: list[ScrollColumn] = [self.left_column]
        top = getattr(self, "right_top_column", None) or getattr(self, "right_column", None)
        bottom = getattr(self, "right_bottom_column", None)
        if top is not None:
            cols.append(top)
        if bottom is not None and bottom is not top:
            cols.append(bottom)
        return tuple(cols)

    def _raise_dock_stacks(self) -> None:
        """Lift packed panes inside each scroller — never the column over the other half."""
        if getattr(self, "_raising_docks", False):
            return
        self._raising_docks = True
        try:
            for col in self._dock_columns():
                for panel in col._docked_panels():
                    try:
                        panel.lift()
                    except tk.TclError:
                        pass
        finally:
            self._raising_docks = False

    def _hit_column(self, x_root: int, y_root: int) -> tuple[ScrollColumn, int] | None:
        """Column under the pointer. Right host is split by the sash (not overlapping geometry)."""
        moving = self._snap_moving
        left = getattr(self, "left_column", None)
        if left is not None and left.contains_root(x_root, y_root):
            return left, left.insert_index_at(y_root, moving=moving)
        host = getattr(self, "right_host", None)
        top = getattr(self, "right_top_column", None)
        bottom = getattr(self, "right_bottom_column", None)
        if host is not None and top is not None and bottom is not None:
            try:
                hx, hy = int(host.winfo_rootx()), int(host.winfo_rooty())
                hw = max(1, int(host.winfo_width()))
                hh = max(1, int(host.winfo_height()))
                if hx <= int(x_root) < hx + hw and hy <= int(y_root) < hy + hh:
                    sash = int(host.sashpos(0))
                    local_y = int(y_root) - hy
                    if abs(local_y - sash) <= _RIGHT_SASH_BAND_PX:
                        return None
                    col = top if local_y < sash else bottom
                    return col, col.insert_index_at(y_root, moving=moving)
            except tk.TclError:
                pass
        for col in self._dock_columns():
            if col.contains_root(x_root, y_root):
                return col, col.insert_index_at(y_root, moving=moving)
        return None

    def _update_snap_target(self, x_root: int, y_root: int, moving: DockablePanel) -> None:
        """Highlight the column under the pointer and show an insert slot."""
        self._snap_moving = moving
        hit = self._hit_column(x_root, y_root)
        self._snap = hit
        for col in self._dock_columns():
            if hit is not None and col is hit[0]:
                col.set_drop_highlight(True)
                col.show_insert_marker(hit[1], moving)
            else:
                col.set_drop_highlight(False)
                col.hide_insert_marker()

    def _clear_snap(self) -> None:
        self._snap = None
        self._snap_moving = None
        for col in self._dock_columns():
            col.set_drop_highlight(False)
            col.hide_insert_marker()

    def _place_panel(self, panel: DockablePanel, column: ScrollColumn, index: int) -> None:
        """Snap-dock ``panel`` into ``column`` at ``index`` (above/below siblings)."""
        old = panel.column
        if old is not None and old is not column and panel in old.panels:
            old.panels.remove(panel)
            old._repack()
        elif old is column and panel in column.panels:
            old_i = column.panels.index(panel)
            column.panels.remove(panel)
            if old_i < index:
                index -= 1
        column.attach(panel, index)

    def _default_dock(self) -> tuple[tuple[DockablePanel, ScrollColumn], ...]:
        """Factory dock: Preview + Coverage left; wheel/filters top; layers/history bottom."""
        return (
            (self.preview_panel, self.left_column),
            (self.coverage_panel, self.left_column),
            (self.wheel_panel, self.right_top_column),
            (self.texture_panel, self.right_top_column),
            (self.tone_panel, self.right_top_column),
            (self.scale_panel, self.right_top_column),
            (self.crop_panel, self.right_top_column),
            (self.layers_panel, self.right_bottom_column),
            (self.labels_panel, self.right_bottom_column),
            (self.tess_panel, self.right_bottom_column),
            (self.history_panel, self.right_bottom_column),
        )

    def _rebuild_view_menu(self) -> None:
        """Preview zoom, layout profiles, Reset layout, pane checks, inspection tabs."""
        menu = getattr(self, "view_menu", None)
        if menu is None:
            return
        menu.delete(0, "end")
        menu.add_command(label="Fit", command=self._reset_preview_zoom)
        menu.add_command(
            label="Zoom in",
            command=lambda: self._nudge_preview_zoom(VIEW_ZOOM_PCT_STEP),
        )
        menu.add_command(
            label="Zoom out",
            command=lambda: self._nudge_preview_zoom(-VIEW_ZOOM_PCT_STEP),
        )
        menu.add_command(label="Reset preview", command=self._reset_preview_zoom)
        menu.add_separator()
        self._rebuild_layout_profiles_menu()
        menu.add_cascade(label="Layout profiles", menu=self.layout_profiles_menu)
        menu.add_command(label="Reset layout", command=self.reset_layout)
        menu.add_separator()
        for panel in getattr(self, "_all_panels", ()):
            var = self._view_checks.get(panel)
            if var is None:
                var = tk.BooleanVar(value=not panel.hidden)
                self._view_checks[panel] = var
            var.set(not panel.hidden)
            menu.add_checkbutton(
                label=panel.panel_title,
                variable=var,
                command=lambda p=panel: self._on_view_toggle(p),
            )
        if getattr(self, "_tab_checks", None):
            menu.add_separator()
            for title in INSPECTION_TAB_TITLES:
                var = self._tab_checks.get(title)
                if var is None:
                    var = tk.BooleanVar(value=True)
                    self._tab_checks[title] = var
                menu.add_checkbutton(
                    label=title,
                    variable=var,
                    command=lambda t=title: self._on_inspection_tab_toggle(t),
                )

    def _rebuild_layout_profiles_menu(self) -> None:
        menu = getattr(self, "layout_profiles_menu", None)
        if menu is None:
            return
        menu.delete(0, "end")
        menu.add_command(label="Save layout profile…", command=self._save_layout_profile_dialog)
        names = sorted(self._load_layout_profiles())
        if not names:
            return
        menu.add_separator()
        for name in names:
            menu.add_command(
                label=name,
                command=lambda n=name: self._apply_layout_profile(n),
            )
        del_menu = getattr(self, "layout_delete_menu", None)
        if del_menu is None:
            return
        del_menu.delete(0, "end")
        for name in names:
            del_menu.add_command(
                label=name,
                command=lambda n=name: self._delete_layout_profile(n),
            )
        menu.add_separator()
        menu.add_cascade(label="Delete", menu=del_menu)

    def _on_view_toggle(self, panel: DockablePanel) -> None:
        var = self._view_checks.get(panel)
        show = True if var is None else bool(var.get())
        if show:
            self.show_panel(panel)
        else:
            self.hide_panel(panel)

    def hide_panel(self, panel: DockablePanel) -> None:
        """Pack-forget the pane; widgets stay alive so View can show it again."""
        if panel.hidden:
            return
        if panel.is_floating:
            panel.dock()
        col = panel.column
        if col is not None and panel in col.panels:
            panel._last_column = col
            panel._last_index = col.panels.index(panel)
            col.detach(panel)
        else:
            panel.pack_forget()
        panel.hidden = True
        var = self._view_checks.get(panel)
        if var is not None:
            var.set(False)

    def show_panel(self, panel: DockablePanel) -> None:
        """Re-attach to the last (or default) column without destroying widgets."""
        if not panel.hidden and panel.column is not None and panel in panel.column.panels:
            return
        panel.hidden = False
        col = panel._last_column or panel._home_column or self.right_column
        index = panel._last_index
        if index is None:
            col.attach(panel)
        else:
            col.attach(panel, index)
        var = self._view_checks.get(panel)
        if var is not None:
            var.set(True)

    def _inspection_tab_widget(self, title: str) -> tk.Misc | None:
        if title == "3×3 tile":
            return getattr(self, "tile_zoom_host", None)
        if title == "Seam offset":
            return getattr(self, "seam_zoom_host", None)
        if title == "Room mockup":
            return getattr(self, "mock_tab", None)
        return None

    def _selected_inspection_kind(self) -> str | None:
        """'tile' / 'seam' / 'mockup' when that View tab is selected, else None."""
        try:
            selected = str(self.notebook.select())
        except tk.TclError:
            return None
        tile = getattr(self, "tile_zoom_host", None)
        seam = getattr(self, "seam_zoom_host", None)
        mock = getattr(self, "mock_tab", None)
        if tile is not None and selected == str(tile):
            return "tile"
        if seam is not None and selected == str(seam):
            return "seam"
        if mock is not None and selected == str(mock):
            return "mockup"
        return None

    def _tab_is_mapped(self, widget: tk.Misc) -> bool:
        try:
            return str(widget) in self.notebook.tabs()
        except tk.TclError:
            return False

    def _hide_inspection_tabs(self) -> None:
        notebook = getattr(self, "notebook", None)
        if notebook is None:
            return
        for title in INSPECTION_TAB_TITLES:
            widget = self._inspection_tab_widget(title)
            if widget is None:
                continue
            try:
                notebook.hide(widget)
            except tk.TclError:
                try:
                    notebook.forget(widget)
                except tk.TclError:
                    pass
            var = self._tab_checks.get(title)
            if var is not None:
                var.set(False)

    def _on_inspection_tab_toggle(self, title: str) -> None:
        widget = self._inspection_tab_widget(title)
        var = self._tab_checks.get(title)
        if widget is None:
            return
        show = True if var is None else bool(var.get())
        notebook = self.notebook
        if show:
            if not self._tab_is_mapped(widget):
                try:
                    notebook.add(widget, text=title)
                except tk.TclError:
                    try:
                        notebook.tab(widget, state="normal")
                    except tk.TclError:
                        pass
            try:
                notebook.select(widget)
            except tk.TclError:
                pass
            self._refresh_tool_tab()
        else:
            try:
                notebook.hide(widget)
            except tk.TclError:
                try:
                    notebook.forget(widget)
                except tk.TclError:
                    pass
        if var is not None:
            var.set(show)

    def _show_inspection_tabs(self) -> None:
        notebook = getattr(self, "notebook", None)
        if notebook is None:
            return
        for title in INSPECTION_TAB_TITLES:
            var = self._tab_checks.get(title)
            if var is not None:
                var.set(True)
            self._on_inspection_tab_toggle(title)

    def _apply_job_layout_defaults(self) -> None:
        """Fresh / reset layout: every pane visible and expanded; inspection tabs shown."""
        for panel in getattr(self, "_all_panels", ()):
            panel.hidden = False
            panel.set_expanded(True)
            var = self._view_checks.get(panel)
            if var is not None:
                var.set(True)
        self._show_inspection_tabs()
        self._sync_range_by_controls()
        self._select_composite_preview_tab()

    def reset_layout(self) -> None:
        """All panes visible and expanded: Preview + Coverage left; filters top; layers bottom."""
        for panel in self._all_panels:
            if panel.is_floating:
                try:
                    panel.dock()
                except tk.TclError:
                    pass
            panel.hidden = False
            panel.pack_forget()
        for col in self._dock_columns():
            col.panels.clear()
        for panel, col in self._default_dock():
            col.attach(panel)
        self._set_default_sash()
        self._layout_profile_name = None
        self._apply_job_layout_defaults()
        for panel, var in self._view_checks.items():
            var.set(not panel.hidden)

    def _layout_profiles_file(self) -> Path:
        path = getattr(self, "_layout_profiles_path", None)
        return Path(path) if path else default_layout_profiles_path()

    def _load_layout_profiles(self) -> dict[str, dict]:
        path = self._layout_profiles_file()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        profiles = raw.get("profiles", raw)
        if not isinstance(profiles, dict):
            return {}
        return {
            str(name).strip(): spec
            for name, spec in profiles.items()
            if str(name).strip() and isinstance(spec, dict)
        }

    def _write_layout_profiles(self, profiles: dict[str, dict]) -> None:
        path = self._layout_profiles_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"profiles": profiles}, indent=2) + "\n",
            encoding="utf-8",
        )

    def _sash_fraction(self) -> float:
        try:
            self.body_paned.update_idletasks()
            width = int(self.body_paned.winfo_width())
            if width >= 80:
                return max(0.05, min(0.95, float(self.body_paned.sashpos(0)) / width))
        except tk.TclError:
            pass
        return float(LEFT_SASH_FRACTION)

    def _right_sash_fraction(self) -> float:
        try:
            host = getattr(self, "right_host", None)
            if host is None:
                return float(RIGHT_SPLIT_FRACTION)
            host.update_idletasks()
            height = int(host.winfo_height())
            if height >= 80:
                return max(0.15, min(0.85, float(host.sashpos(0)) / height))
        except tk.TclError:
            pass
        return float(RIGHT_SPLIT_FRACTION)

    def _capture_layout_spec(self) -> dict:
        """Dock order, hidden/collapsed panes, and sash fractions for layout_profiles.json."""
        left = [p.panel_title for p in self.left_column.panels]
        right_top = [p.panel_title for p in self.right_top_column.panels]
        right_bottom = [p.panel_title for p in self.right_bottom_column.panels]
        hidden = [p.panel_title for p in self._all_panels if p.hidden]
        collapsed = [p.panel_title for p in self._all_panels if not getattr(p, "expanded", True)]
        hidden_tabs = [
            title
            for title in INSPECTION_TAB_TITLES
            if not bool((self._tab_checks.get(title) or tk.BooleanVar(value=True)).get())
        ]
        placed = set(left) | set(right_top) | set(right_bottom)
        for panel in self._all_panels:
            if not panel.is_floating:
                continue
            if panel.panel_title in placed or panel.panel_title in hidden:
                continue
            idx = panel._last_index if panel._last_index is not None else 0
            left.insert(max(0, min(idx, len(left))), panel.panel_title)
        return {
            "left": left,
            "right_top": right_top,
            "right_bottom": right_bottom,
            "right": right_top + right_bottom,
            "hidden": hidden,
            "collapsed": collapsed,
            "hidden_tabs": hidden_tabs,
            "sash_fraction": self._sash_fraction(),
            "right_sash_fraction": self._right_sash_fraction(),
        }

    def _apply_layout_spec(self, spec: dict) -> None:
        by_title = {p.panel_title: p for p in self._all_panels}
        for panel in self._all_panels:
            if panel.is_floating:
                try:
                    panel.dock()
                except tk.TclError:
                    pass
            panel.hidden = False
            panel.pack_forget()
        for col in self._dock_columns():
            col.panels.clear()

        def _fill(titles, column: ScrollColumn) -> None:
            for title in titles:
                panel = by_title.get(str(title))
                if panel is None or panel in column.panels:
                    continue
                panel.column = column
                panel._last_column = column
                panel._last_index = len(column.panels)
                panel.hidden = False
                column.panels.append(panel)
                column._adopt(panel)

        def _title(name) -> str:
            key = str(name)
            return _LAYOUT_TITLE_ALIASES.get(key, key)

        left = [_title(t) for t in (spec.get("left") or [])]
        right_top = [_title(t) for t in (spec.get("right_top") or [])]
        right_bottom = [_title(t) for t in (spec.get("right_bottom") or [])]
        if not right_top and not right_bottom:
            combined = [_title(t) for t in (spec.get("right") or [])]
            bottom_names = {"Layers", "Labels", "Tessellate", HISTORY_PANEL_TITLE}
            right_top = [t for t in combined if t not in bottom_names]
            right_bottom = [t for t in combined if t in bottom_names]
        hidden = {_title(t) for t in (spec.get("hidden") or [])}
        collapsed = {_title(t) for t in (spec.get("collapsed") or [])}
        _fill(left, self.left_column)
        _fill(right_top, self.right_top_column)
        _fill(right_bottom, self.right_bottom_column)
        placed = set(left) | set(right_top) | set(right_bottom)
        for panel, col in self._default_dock():
            if panel.panel_title in hidden or panel.panel_title in placed:
                continue
            if panel in col.panels:
                continue
            panel.column = col
            panel._last_column = col
            col.panels.append(panel)
            col._adopt(panel)
            placed.add(panel.panel_title)
        for col in self._dock_columns():
            col._repack()
        for title in hidden:
            panel = by_title.get(title)
            if panel is not None:
                self.hide_panel(panel)
        for panel in self._all_panels:
            panel.set_expanded(panel.panel_title not in collapsed)
        if "hidden_tabs" in spec:
            hidden_tabs = {_title(t) for t in (spec.get("hidden_tabs") or [])}
            for title in INSPECTION_TAB_TITLES:
                var = self._tab_checks.get(title)
                if var is not None:
                    var.set(title not in hidden_tabs)
                self._on_inspection_tab_toggle(title)
        else:
            self._show_inspection_tabs()
        frac = spec.get("sash_fraction")
        try:
            frac_f = float(frac)
        except (TypeError, ValueError):
            frac_f = float(LEFT_SASH_FRACTION)
        if 0.05 < frac_f < 0.95:
            try:
                self.body_paned.update_idletasks()
                width = int(self.body_paned.winfo_width())
                if width >= 80:
                    self.body_paned.sashpos(0, int(width * frac_f))
            except tk.TclError:
                pass
        rfrac = spec.get("right_sash_fraction")
        try:
            rfrac_f = float(rfrac)
        except (TypeError, ValueError):
            rfrac_f = float(RIGHT_SPLIT_FRACTION)
        if 0.15 < rfrac_f < 0.85:
            if self._apply_right_sash_fraction(rfrac_f):
                self._right_sash_init = True
        for col in self._dock_columns():
            try:
                col._sync_layout()
            except tk.TclError:
                pass
        for panel, var in self._view_checks.items():
            var.set(not panel.hidden)

    def _save_layout_profile_dialog(self) -> None:
        name = simpledialog.askstring(
            "Save layout profile",
            "Name this dock layout:",
            parent=self.root,
            initialvalue=self._layout_profile_name or "",
        )
        if name is None:
            return
        name = " ".join(str(name).split())
        if not name:
            return
        profiles = self._load_layout_profiles()
        profiles[name] = self._capture_layout_spec()
        try:
            self._write_layout_profiles(profiles)
        except OSError as exc:
            messagebox.showerror("Could not save layout profile", str(exc), parent=self.root)
            return
        self._layout_profile_name = name
        self.status.set(f"Saved layout profile “{name}”")

    def _apply_layout_profile(self, name: str) -> None:
        profiles = self._load_layout_profiles()
        spec = profiles.get(name)
        if not isinstance(spec, dict):
            messagebox.showerror(
                "Layout profile missing",
                f"“{name}” was not found in layout_profiles.json.",
                parent=self.root,
            )
            return
        self._apply_layout_spec(spec)
        self._layout_profile_name = name
        self.status.set(f"Applied layout profile “{name}”")

    def _delete_layout_profile(self, name: str) -> None:
        if not messagebox.askyesno(
            "Delete layout profile",
            f"Delete “{name}”?",
            parent=self.root,
        ):
            return
        profiles = self._load_layout_profiles()
        profiles.pop(name, None)
        try:
            self._write_layout_profiles(profiles)
        except OSError as exc:
            messagebox.showerror("Could not delete layout profile", str(exc), parent=self.root)
            return
        if self._layout_profile_name == name:
            self._layout_profile_name = None
        self.status.set(f"Deleted layout profile “{name}”")
