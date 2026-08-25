# -*- coding: utf-8 -*-
"""
wallpaper_recolor.ui.mixins.layers_labels
------------------------------
Layers tree, color-range rows, Labels / OCR UI.

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


class AppLayersLabelsMixin:
    """Layers tree, color-range rows, Labels / OCR UI."""

    # ---------------------------------------------------------------------------
    # Labels / inpaint state
    # ---------------------------------------------------------------------------
    def _reset_labels_state(self) -> None:
        """Clear detections, inpaints, and the typed label (new file)."""
        self._inpaint_boxes = []
        self._inpaint_quads = []
        self._inpaint_layer_id = ""
        self._detect_boxes = []
        self._detect_quads = []
        self._detect_roi = None
        self._selected_detect = set()
        self._label_mark_mode = False
        self._label_place_mode = False
        prev = self._label_updating
        self._label_updating = True
        try:
            self.label_text.set("")
            self.label_size.set(float(LABEL_SIZE_DEFAULT))
            self.label_size_text.set(str(LABEL_SIZE_DEFAULT))
            self.label_color.set(LABEL_COLOR_DEFAULT)
            self.label_x.set("0")
            self.label_y.set("0")
            self.label_font.set(LABEL_FONT_DEFAULT)
            self.label_ocr_status.set(tesseract_status_text())
        finally:
            self._label_updating = prev
        self._prune_label_layers()
        self._sync_label_modes()
        self._refresh_layers_panel()

    def _label_spec(self) -> LabelSpec:
        try:
            x = int(round(float(self.label_x.get() or 0)))
        except (TypeError, ValueError, tk.TclError):
            x = 0
        try:
            y = int(round(float(self.label_y.get() or 0)))
        except (TypeError, ValueError, tk.TclError):
            y = 0
        return LabelSpec(
            text=str(self.label_text.get() or ""),
            size=clamp_label_size(self.label_size.get()),
            color=parse_label_color(self.label_color.get()),
            x=x,
            y=y,
            font=str(self.label_font.get() or LABEL_FONT_DEFAULT),
        )

    def _inpaint_tuple(self) -> tuple[tuple[int, int, int, int], ...]:
        return tuple(self._inpaint_boxes)

    def _apply_inpaint_to_image(self, image: Image.Image) -> Image.Image:
        if not self._inpaint_boxes:
            return image
        return inpaint_image(
            image,
            self._inpaint_boxes,
            src_size=self._crop_src_size(),
            wrap=normalize_tess_mode(self.tess_mode.get()) == MODE_TILE,
            quads=self._inpaint_quads or None,
            style=self._wallpaper_style_key(),
            cancel=getattr(self, "_job_cancel", None),
        )

    def _wallpaper_style_key(self) -> str:
        label = str(self.wallpaper_style.get() if hasattr(self, "wallpaper_style") else "")
        if label == WALLPAPER_FLORAL_LABEL or "floral" in label.lower():
            return STYLE_FLORAL
        return STYLE_GEOMETRIC

    def _sync_label_modes(self) -> None:
        mark = bool(self._label_mark_mode)
        for host in (
            getattr(self, "orig_zoom_host", None),
            getattr(self, "tex_zoom_host", None),
        ):
            if host is not None:
                host.rect_mode = mark
                host._sync_host_cursor()
        if hasattr(self, "labels_mark_btn"):
            self.labels_mark_btn.configure(
                text="Selecting…" if mark else "Select area"
            )
        if hasattr(self, "label_place_btn"):
            self.label_place_btn.configure(
                text="Click preview" if self._label_place_mode else "Place"
            )
        self._sync_eyedrop_cursor()

    def _selected_label_layer(self) -> StackLayer | None:
        for ly in self.layer_stack.selected():
            if ly.is_label():
                return ly
        return None

    def _prune_label_layers(self) -> None:
        self.layer_stack.layers = [ly for ly in self.layer_stack.layers if not ly.is_label()]
        ids = {ly.id for ly in self.layer_stack.layers}
        self.layer_stack.selected_ids = [i for i in self.layer_stack.selected_ids if i in ids]
        if not self.layer_stack.selected_ids and self.layer_stack.layers:
            self.layer_stack.selected_ids = [self.layer_stack.layers[0].id]

    def _ensure_base_layer(self) -> None:
        if self.work_image is None:
            return
        base = self.layer_stack.base_layer()
        name = self.source_path.stem if self.source_path is not None else "Image"
        path = str(self.source_path) if self.source_path is not None else ""
        if base is None:
            self.layer_stack.add_image(
                name=name,
                path=path,
                raster=self.work_image,
                role=ROLE_BASE,
            )
            self._sync_range_layers()
            return
        base.raster = self.work_image
        if path:
            base.path = path
        if name and (not base.name or base.name == "Image"):
            base.name = name
        self._sync_range_layers()

    def _sync_range_layers(self) -> None:
        """Keep Color ranges + Range N as children of the opened Image layer."""
        base = self.layer_stack.base_layer()
        if base is None:
            return
        if self.range_map is None:
            self.layer_stack.sync_range_children(base.id, 0)
            return
        vis = [bool(band.visible) for band in self.range_map.ranges]
        names = [
            str(band.name or f"Range {i + 1}")
            for i, band in enumerate(self.range_map.ranges)
        ]
        self.layer_stack.sync_range_children(
            base.id, len(self.range_map.ranges), vis, names
        )

    def _restore_layer_stack(
        self,
        records,
        selected_ids,
    ) -> None:
        rasters = {ly.id: ly.raster for ly in self.layer_stack.layers if ly.raster is not None}
        recs = [rec for rec in records if isinstance(rec, dict)]
        if recs:
            self.layer_stack.replace_from_records(recs, selected_ids, rasters=rasters)
            for ly in self.layer_stack.layers:
                if ly.raster is not None:
                    continue
                if ly.is_base() and self.work_image is not None:
                    ly.raster = self.work_image
                elif ly.is_image() and ly.path:
                    path = Path(ly.path)
                    if path.is_file():
                        try:
                            ly.raster = _fit(load_image(path), WORK_MAX_EDGE)
                        except (OSError, ValueError):
                            ly.raster = None
        else:
            self._ensure_base_layer()
            spec = self._label_spec()
            if spec.is_set() and not any(ly.is_label() for ly in self.layer_stack.layers):
                self.layer_stack.add_label(
                    name="Label",
                    text=spec.text,
                    size=spec.size,
                    color=spec.color,
                    font=spec.font,
                    x=spec.x,
                    y=spec.y,
                    select=False,
                )
        self._ensure_base_layer()
        ly = self._selected_label_layer()
        if ly is not None:
            self._sync_label_fields_from_layer(ly)

    def _sync_label_fields_from_layer(self, ly: StackLayer | None = None) -> None:
        row = ly if ly is not None else self._selected_label_layer()
        if row is None or not row.is_label():
            return
        prev = self._label_updating
        self._label_updating = True
        try:
            self.label_text.set(str(row.text or ""))
            size = clamp_label_size(row.size)
            self.label_size.set(float(size))
            self.label_size_text.set(str(size))
            self.label_color.set(label_rgb_to_hex(row.color))
            self.label_x.set(str(int(row.x)))
            self.label_y.set(str(int(row.y)))
            self.label_font.set(str(row.font or LABEL_FONT_DEFAULT))
        finally:
            self._label_updating = prev

    def _write_label_fields_to_layer(self) -> StackLayer | None:
        spec = self._label_spec()
        ly = self._selected_label_layer()
        if ly is None:
            if not spec.is_set():
                return None
            n = 1 + sum(1 for row in self.layer_stack.layers if row.is_label())
            ly = self.layer_stack.add_label(
                name=f"Label {n}" if n > 1 else "Label",
                text=spec.text,
                size=spec.size,
                color=spec.color,
                font=spec.font,
                x=spec.x,
                y=spec.y,
            )
        else:
            ly.apply_label_spec(spec)
        return ly

    def _layer_kind_label(self, ly) -> str:
        if ly.is_range():
            return "Range"
        if ly.is_range_group():
            return "Color ranges"
        if ly.is_label():
            return "Label"
        if ly.is_group():
            return "Group"
        return "Image"

    # ---------------------------------------------------------------------------
    # Layers tree (range rows are children of the Image layer)
    # ---------------------------------------------------------------------------
    def _refresh_layers_panel(self) -> None:
        host = getattr(self, "layers_list", None)
        if host is None:
            return
        for child in host.winfo_children():
            child.destroy()
        self._layer_rows = []
        self._layer_range_rows = {}
        chosen = set(self.layer_stack.selected_ids)
        for ly, depth in self.layer_stack.walk_visible_tree(""):
            bg = "#dbe8f6" if ly.id in chosen else "#f5f5f5"
            row = tk.Frame(host, bg=bg, padx=4, pady=2, cursor="hand2")
            row.pack(fill="x", pady=1)
            if depth:
                spacer = tk.Frame(row, bg=bg, width=14 * int(depth), height=1)
                spacer.pack(side="left")
                spacer.pack_propagate(False)
            if ly.is_group():
                twisty = tk.Label(
                    row,
                    text=_LAYER_TWISTY_OPEN if ly.expanded else _LAYER_TWISTY_SHUT,
                    bg=bg,
                    fg="#333333",
                    width=2,
                    cursor="hand2",
                )
                twisty.pack(side="left")
                bind_tooltip(twisty, "Show or hide Color ranges.")
                twisty.bind(
                    "<Button-1>",
                    lambda _e, lid=ly.id, was=bool(ly.expanded): self._on_layer_twisty(
                        lid, not was
                    ),
                )
            eye = EyeToggle(
                row,
                self._eye_photos,
                lambda shown, lid=ly.id: self._on_layer_eye(lid, shown),
                bg=bg,
                tooltip="Show or hide this layer. Hidden layers skip composite, Build, and color.",
            )
            eye.set_shown(bool(ly.visible))
            eye.pack(side="left")
            kind = self._layer_kind_label(ly)
            if ly.is_range():
                self._pack_range_layer_row(row, ly, bg)
            else:
                title = tk.Label(
                    row,
                    text=ly.name or kind,
                    bg=bg,
                    anchor="w",
                    cursor="hand2",
                )
                title.pack(side="left", fill="x", expand=True, padx=(4, 0))
                meta = tk.Label(row, text=kind, bg=bg, fg="#555555", cursor="hand2")
                meta.pack(side="right")
                bind_tooltip(row, f"{kind} layer. Click to select; drag on the preview to move.")
                bind_tooltip(title, f"{kind} layer. Click to select; drag on the preview to move.")
                for widget in (row, title, meta):
                    widget.bind("<Button-1>", lambda _e, lid=ly.id: self._on_select_layer(lid))
            self._layer_rows.append(row)
        col = getattr(self.layers_panel, "column", None)
        if col is not None:
            col._tag_column_widgets()

    def _pack_range_layer_row(self, row: tk.Misc, ly, bg: str) -> None:
        """Eye (already packed) + match / change swatches + coverage Spinbox + kebab."""
        index = int(ly.range_index)
        band = None
        if self.range_map is not None and 0 <= index < len(self.range_map.ranges):
            band = self.range_map.ranges[index]
        match_rgb = band.match_rgb if band is not None else (180, 180, 180)
        repl_rgb = band.replacement_rgb if band is not None else (180, 180, 180)
        pct = int(round((band.weight if band is not None else 0.0) * 100.0))
        sel = index == int(self.selected_index)
        n_ranges = len(self.range_map.ranges) if self.range_map is not None else 0
        kebab = tk.Label(
            row,
            text=_LAYER_KEBAB,
            bg=bg,
            fg="#333333",
            width=2,
            cursor="hand2",
        )
        kebab.pack(side="right")
        menu = tk.Menu(kebab, tearoff=0)
        can_remove = n_ranges > 1
        menu.add_command(
            label="Remove",
            command=lambda i=index: self._on_remove_layer_range(i),
            state="normal" if can_remove else "disabled",
        )
        kebab.bind("<Button-1>", lambda e, i=index: self._on_range_kebab(e, i))
        bind_tooltip(kebab, "More options for this color range.")
        match_cv = self._make_layer_swatch(
            row, match_rgb, index, HALF_MATCH, selected=sel and self.selected_half == HALF_MATCH
        )
        match_cv.pack(side="left", padx=(4, 0))
        slash = tk.Label(row, text="/", bg=bg, fg="#333333", cursor="hand2")
        slash.pack(side="left", padx=3)
        repl_cv = self._make_layer_swatch(
            row,
            repl_rgb,
            index,
            HALF_REPLACE,
            selected=sel and self.selected_half == HALF_REPLACE,
        )
        repl_cv.pack(side="left")
        floor_pct = max(0, int(round(clamp_min_coverage(
            getattr(self.range_map, "min_coverage", MIN_COVERAGE) if self.range_map is not None else MIN_COVERAGE
        ) * 100.0)))
        if floor_pct < 1:
            floor_pct = 0
        var = tk.StringVar(value=str(pct))
        spin = ttk.Spinbox(
            row,
            from_=floor_pct,
            to=100 - floor_pct,
            increment=1,
            width=4,
            textvariable=var,
            command=lambda i=index: self._on_layer_range_pct(i),
        )
        spin.pack(side="left", padx=(8, 0))
        spin.bind("<Return>", lambda _e, i=index: self._on_layer_range_pct(i))
        spin.bind("<FocusOut>", lambda _e, i=index: self._on_layer_range_pct(i))
        bind_tooltip(match_cv, "Match-from — click to select and load the wheel.")
        bind_tooltip(slash, "Match-from / change-to.")
        bind_tooltip(repl_cv, "Change-to — click to select and load the wheel.")
        bind_tooltip(
            spin,
            "Coverage weight (same as the Coverage bar). Steals from the adjacent range.",
        )
        for widget in (row, slash):
            widget.bind("<Button-1>", lambda _e, lid=ly.id: self._on_select_layer(lid))
        self._layer_range_rows[index] = {
            "match": match_cv,
            "replace": repl_cv,
            "slash": slash,
            "pct": var,
            "spin": spin,
            "row": row,
            "kebab": kebab,
            "menu": menu,
        }

    def _make_layer_swatch(
        self,
        parent: tk.Misc,
        rgb: tuple[int, int, int],
        index: int,
        half: str,
        *,
        selected: bool,
    ) -> tk.Canvas:
        size = _LAYER_SWATCH_PX
        outline = "#ffcc33" if selected else "#888888"
        width = 2 if selected else 1
        canvas = tk.Canvas(
            parent,
            width=size,
            height=size,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        fill = rgb_to_hex(rgb)
        canvas.create_rectangle(
            1, 1, size, size, fill=fill, outline=outline, width=width, tags="swatch"
        )
        canvas.bind(
            "<Button-1>",
            lambda _e, i=index, h=half: self._on_layer_range_swatch(i, h),
        )
        return canvas

    def _paint_layer_swatch(
        self, canvas: tk.Canvas, rgb: tuple[int, int, int], *, selected: bool
    ) -> None:
        size = _LAYER_SWATCH_PX
        outline = "#ffcc33" if selected else "#888888"
        width = 2 if selected else 1
        canvas.delete("swatch")
        canvas.create_rectangle(
            1,
            1,
            size,
            size,
            fill=rgb_to_hex(rgb),
            outline=outline,
            width=width,
            tags="swatch",
        )

    def _on_layer_twisty(self, layer_id: str, expanded: bool) -> None:
        if self.layer_stack.set_expanded(layer_id, expanded):
            self._refresh_layers_panel()

    def _on_layer_range_swatch(self, index: int, half: str) -> None:
        if self.range_map is None:
            return
        for ly in self.layer_stack.layers:
            if ly.is_range() and int(ly.range_index) == int(index):
                self.layer_stack.select(ly.id)
                break
        self.select_range(index, half, toggle=True)
        self._refresh_layers_panel()

    def _on_layer_range_pct(self, index: int) -> None:
        if self._layer_pct_mute or self._mute_ui or self.range_map is None:
            return
        row = self._layer_range_rows.get(int(index))
        if row is None:
            return
        raw = str(row["pct"].get() or "").strip().rstrip("%")
        try:
            pct = float(raw)
        except ValueError:
            return
        self.apply_typed_percent(index, pct)

    def _on_range_kebab(self, event, index: int) -> str:
        """Post the range-row overflow menu at the ⋯ button."""
        row = self._layer_range_rows.get(int(index))
        if row is None:
            return "break"
        menu = row.get("menu")
        if menu is None:
            return "break"
        widget = event.widget
        try:
            x = int(widget.winfo_rootx())
            y = int(widget.winfo_rooty() + widget.winfo_height())
        except tk.TclError:
            x, y = int(event.x_root), int(event.y_root)
        try:
            menu.tk_popup(x, y)
        finally:
            try:
                menu.grab_release()
            except tk.TclError:
                pass
        return "break"

    def _on_remove_layer_range(self, index: int) -> None:
        """Delete a color range (not hide). Surviving match-from / change-to stay."""
        if self._mute_ui or self._history_lock or self.range_map is None:
            return
        n = len(self.range_map.ranges)
        if n <= 1:
            return
        index = max(0, min(int(index), n - 1))
        before = self._capture_edit()
        drop_color_range(self.range_map, index)
        new_n = len(self.range_map.ranges)
        if new_n >= n:
            return
        self.range_count.set(new_n)
        sel = int(self.selected_index)
        if sel < 0:
            self.selected_index = -1
        else:
            if sel > index:
                sel -= 1
            self.selected_index = max(0, min(sel, new_n - 1))
        self.status.set("Removed a color range — remaining match-from / change-to kept.")
        preset = get_preset(self.preset_id) if self.preset_id else None
        if preset is not None and preset.range_count != new_n:
            self._clear_preset_selection()
        self._push_undo_state(before)
        self._sync_texture_to_map()
        self._rebuild_chips()
        self._load_selected_onto_wheel()
        self._sync_range_widgets(update_bar=True)
        self._sync_range_layers()
        self._refresh_layers_panel()
        self._refresh_now()

    def _sync_layer_range_rows(self) -> None:
        """Update swatches and % without rebuilding rows (keeps Spinbox focus)."""
        if self.range_map is None or not self._layer_range_rows:
            return
        self._layer_pct_mute = True
        try:
            for index, widgets in self._layer_range_rows.items():
                if index < 0 or index >= len(self.range_map.ranges):
                    continue
                band = self.range_map.ranges[index]
                sel = index == int(self.selected_index)
                self._paint_layer_swatch(
                    widgets["match"],
                    band.match_rgb,
                    selected=sel and self.selected_half == HALF_MATCH,
                )
                self._paint_layer_swatch(
                    widgets["replace"],
                    band.replacement_rgb,
                    selected=sel and self.selected_half == HALF_REPLACE,
                )
                want = str(int(round(band.weight * 100.0)))
                if str(widgets["pct"].get()) != want:
                    widgets["pct"].set(want)
        finally:
            self._layer_pct_mute = False

    def _on_select_layer(self, layer_id: str) -> None:
        self.layer_stack.select(layer_id)
        ly = self.layer_stack.get(layer_id)
        if ly is not None and ly.is_label():
            self._sync_label_fields_from_layer(ly)
        if ly is not None and ly.is_range() and int(ly.range_index) >= 0:
            self.select_range(ly.range_index, toggle=True)
        self._refresh_layers_panel()
        for host in (
            getattr(self, "orig_zoom_host", None),
            getattr(self, "tex_zoom_host", None),
        ):
            if host is not None:
                host._sync_host_cursor()
        self._refresh_now()

    def _on_layer_eye(self, layer_id: str, shown: bool) -> None:
        ly = self.layer_stack.get(layer_id)
        if ly is not None and ly.is_range() and int(ly.range_index) >= 0:
            self.set_range_visible(ly.range_index, bool(shown))
            self._refresh_layers_panel()
            return
        before = self._capture_edit()
        self.layer_stack.set_visible(layer_id, bool(shown))
        self._push_undo_state(before)
        self._refresh_layers_panel()
        self._refresh_now()

    def _on_layer_up(self) -> None:
        ly = self.layer_stack.primary()
        if ly is None or ly.is_range() or ly.is_range_group():
            return
        before = self._capture_edit()
        if self.layer_stack.move_up(ly.id):
            self._push_undo_state(before)
            self._refresh_layers_panel()
            self._refresh_now()

    def _on_layer_down(self) -> None:
        ly = self.layer_stack.primary()
        if ly is None or ly.is_range() or ly.is_range_group():
            return
        before = self._capture_edit()
        if self.layer_stack.move_down(ly.id):
            self._push_undo_state(before)
            self._refresh_layers_panel()
            self._refresh_now()

    def _on_add_image_layer(self) -> None:
        if self.work_image is None:
            self.status.set("Open a wallpaper first.")
            return
        path = filedialog.askopenfilename(title="Add image layer", filetypes=OPEN_FILETYPES)
        if not path:
            return
        try:
            image = load_image(path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not open image", str(exc), parent=self.root)
            return
        before = self._capture_edit()
        src = Path(path)
        self.layer_stack.add_image(
            name=src.stem or "Image",
            path=str(src),
            raster=_fit(image, WORK_MAX_EDGE),
        )
        self._push_undo_state(before)
        self._refresh_layers_panel()
        self.status.set(f"Added image layer {src.name}")
        self._refresh_now()

    def _on_add_label_layer(self) -> None:
        before = self._capture_edit()
        n = 1 + sum(1 for row in self.layer_stack.layers if row.is_label())
        ly = self.layer_stack.add_label(
            name=f"Label {n}" if n > 1 else "Label",
            text=str(self.label_text.get() or "Label"),
            size=clamp_label_size(self.label_size.get()),
            color=parse_label_color(self.label_color.get()),
            font=str(self.label_font.get() or LABEL_FONT_DEFAULT),
        )
        if not str(self.label_text.get() or "").strip():
            ly.text = "Label"
        self._sync_label_fields_from_layer(ly)
        self._label_place_mode = True
        self._label_mark_mode = False
        self._sync_label_modes()
        self._push_undo_state(before)
        self._refresh_layers_panel()
        self.status.set("Click Original or Result to place the label.")
        self._refresh_now()

    def _on_label_font(self) -> None:
        if self._mute_ui or self._label_updating:
            return
        before = self._capture_edit()
        ly = self._write_label_fields_to_layer()
        if ly is not None:
            ly.font = str(self.label_font.get() or LABEL_FONT_DEFAULT)
        self._push_undo_state(before)
        self._refresh_now()

    def _on_label_detect(self) -> None:
        """Find baked-in glyphs on the wallpaper raster (busy bar if OCR is slow)."""
        if self._busy:
            return
        if self.work_image is None:
            self.status.set("Open an image first.")
            return
        view = self._apply_view_crop(self.work_image).copy()
        cx, cy, cz = self._crop_xy_zoom()
        src = self._crop_src_size()
        roi_view = None
        if self._detect_roi is not None:
            roi_view = source_box_to_display(
                self._detect_roi, view.size, src, cx, cy, cz
            )
        roi_src = self._detect_roi
        view_size = view.size
        cancel = self._job_cancel

        def work():
            return detect_text_regions(view, roi=roi_view, cancel=cancel)

        def on_ok(found) -> None:
            mapped: list[tuple[int, int, int, int]] = []
            quads: list[tuple[tuple[int, int], ...]] = []
            for region in found or ():
                box = getattr(region, "box", region)
                src_box = display_box_to_source(box, view_size, src, cx, cy, cz)
                if src_box is None:
                    continue
                mapped.append(src_box)
                raw_quad = getattr(region, "quad", None)
                if raw_quad:
                    pts = tuple(
                        display_xy_to_source(px, py, view_size, src, cx, cy, cz)
                        for px, py in raw_quad
                    )
                    quads.append(pts)
                else:
                    quads.append(aabb_quad(src_box))
            self._detect_boxes = mapped
            self._detect_quads = quads
            self._selected_detect = set()
            self._label_place_mode = False
            if not mapped:
                if roi_src is not None:
                    self._detect_boxes = [roi_src]
                    self._detect_quads = [aabb_quad(roi_src)]
                    self._selected_detect = {0}
                    self._label_mark_mode = False
                    self.status.set(
                        "No printed text in the area — selected the rectangle. "
                        "Remove to fill the wallpaper pattern."
                    )
                else:
                    self._label_mark_mode = True
                    self.status.set(
                        "No printed text found — drag a rectangle on Original or Result, then Detect."
                    )
            else:
                self._label_mark_mode = False
                where = " in the selected area" if roi_view is not None else ""
                n = len(mapped)
                self.status.set(
                    f"Found {n} printed-text region{'s' if n != 1 else ''}{where}. "
                    "Remove fills the wallpaper pattern through them."
                )
            self._sync_label_modes()
            self._refresh_now()

        def on_err(exc: BaseException) -> None:
            messagebox.showerror("Could not detect text", str(exc), parent=self.root)
            self.status.set("Detect failed")

        self._run_background(
            "Detecting…",
            work,
            on_ok,
            on_err,
            cancellable=True,
            cancel_status="Detect cancelled.",
        )

    def _boxes_for_remove(self) -> list[tuple[int, int, int, int]]:
        """Selected detect boxes, else all detections, else the Mark rectangle."""
        if self._selected_detect:
            chosen = sorted(
                i for i in self._selected_detect if 0 <= i < len(self._detect_boxes)
            )
            return [self._detect_boxes[i] for i in chosen]
        if self._detect_boxes:
            return list(self._detect_boxes)
        if self._detect_roi is not None:
            return [self._detect_roi]
        return []

    def _quads_for_remove(self, boxes: list[tuple[int, int, int, int]]) -> list[tuple[tuple[int, int], ...]]:
        n = len(self._detect_quads)
        if self._selected_detect:
            chosen = sorted(
                i for i in self._selected_detect if 0 <= i < len(self._detect_boxes)
            )
            return [
                self._detect_quads[i] if i < n else aabb_quad(self._detect_boxes[i])
                for i in chosen
            ]
        if self._detect_boxes and n:
            return list(self._detect_quads[: len(self._detect_boxes)])
        return [aabb_quad(box) for box in boxes]

    def _on_label_remove(self) -> None:
        """Inpaint OCR polygons on source-resolution crops (LaMa / cv2 / numpy)."""
        if self._busy:
            return
        if self.work_image is None:
            self.status.set("Open an image first.")
            return
        boxes = self._boxes_for_remove()
        if not boxes:
            self.status.set("Detect printed text or select an area, then Remove.")
            return
        quads = self._quads_for_remove(boxes)
        target = inpaint_target_layer(self.layer_stack)
        hole_id = target.id if target is not None else ""
        pending_boxes = list(boxes)
        pending_quads = list(quads)
        used_detect = set()
        if self._selected_detect:
            used_detect = {i for i in self._selected_detect if 0 <= i < len(self._detect_boxes)}
        clear_all_detect = bool(self._detect_boxes) and not used_detect
        clear_roi = not self._detect_boxes and self._detect_roi is not None
        self._cancel_preview_job()
        n = len(boxes)
        snap = self._preview_snapshot()
        if snap is not None:
            snap["holes"] = tuple(list(self._inpaint_boxes) + pending_boxes)
            snap["quads"] = tuple(list(self._inpaint_quads) + pending_quads)
            snap["inpaint_layer_id"] = hole_id
            snap["cancel"] = self._job_cancel
            snap["style"] = self._wallpaper_style_key()

        def work():
            if snap is None:
                return None
            return self._preview_pils(snap)

        def on_ok(computed) -> None:
            before = self._capture_edit()
            self._inpaint_layer_id = hole_id
            self._inpaint_boxes.extend(pending_boxes)
            self._inpaint_quads.extend(pending_quads)
            if used_detect:
                self._detect_boxes = [
                    box for i, box in enumerate(self._detect_boxes) if i not in used_detect
                ]
                self._detect_quads = [
                    q for i, q in enumerate(self._detect_quads) if i not in used_detect
                ]
            elif clear_all_detect:
                self._detect_boxes = []
                self._detect_quads = []
            if clear_roi:
                self._detect_roi = None
            self._selected_detect = set()
            self._push_undo_state(before)
            if computed is not None:
                orig_disp, live_preview, live = computed
                self._apply_preview_pils(orig_disp, live_preview, live)
            backend = inpaint_backend()
            extra = ""
            if backend == "cv2" and not lama_onnx_available():
                extra = " OpenCV inpaint (cache LaMa ONNX for better fills)."
            elif backend == "numpy":
                extra = " numpy fill (install requirements-ocr.txt for EasyOCR + LaMa)."
            elif backend == "lama":
                extra = " LaMa."
            self.status.set(
                f"Filled {n} printed-text region{'s' if n != 1 else ''} "
                f"with the wallpaper pattern{extra} (Ctrl+Z undoes)."
            )

        def on_err(exc: BaseException) -> None:
            messagebox.showerror("Could not remove text", str(exc), parent=self.root)
            self.status.set("Remove failed")

        self._run_background(
            "Removing…",
            work,
            on_ok,
            on_err,
            cancellable=True,
            cancel_status="Remove cancelled.",
        )

    def _on_label_clear(self) -> None:
        self._detect_boxes = []
        self._detect_quads = []
        self._detect_roi = None
        self._selected_detect = set()
        self._label_mark_mode = False
        self._label_place_mode = False
        self._sync_label_modes()
        self._refresh_now()

    def _on_label_mark_toggle(self) -> None:
        self._label_mark_mode = not self._label_mark_mode
        if self._label_mark_mode:
            self._label_place_mode = False
            self.status.set("Drag on Original or Result to select the Detect area.")
        self._sync_label_modes()

    def _on_label_place_toggle(self) -> None:
        self._label_place_mode = not self._label_place_mode
        if self._label_place_mode:
            self._label_mark_mode = False
            self.status.set("Click Original or Result to place the label.")
        self._sync_label_modes()

    def _on_label_size_slider(self, _value: str = "") -> None:
        if self._mute_ui or self._label_updating:
            return
        size = clamp_label_size(self.label_size.get())
        self._label_updating = True
        try:
            self.label_size.set(float(size))
            self.label_size_text.set(str(size))
        finally:
            self._label_updating = False
        ly = self._selected_label_layer()
        if ly is not None:
            ly.size = size
        self._sync_slider_resets()
        self._schedule_preview()

    def _commit_label_size(self, _event=None) -> None:
        if self._mute_ui or self._label_updating:
            return
        before = self._capture_edit()
        size = clamp_label_size(self.label_size_text.get())
        self._label_updating = True
        try:
            self.label_size.set(float(size))
            self.label_size_text.set(str(size))
        finally:
            self._label_updating = False
        self._push_undo_state(before)
        self._write_label_fields_to_layer()
        self._sync_slider_resets()
        self._refresh_now()

    def _reset_label_size(self) -> None:
        before = self._capture_edit()
        self._label_updating = True
        try:
            self.label_size.set(float(LABEL_SIZE_DEFAULT))
            self.label_size_text.set(str(LABEL_SIZE_DEFAULT))
        finally:
            self._label_updating = False
        self._push_undo_state(before)
        self._write_label_fields_to_layer()
        self._sync_slider_resets()
        self._refresh_now()

    def _commit_label_fields(self, _event=None) -> None:
        if self._mute_ui or self._label_updating:
            return
        before = self._capture_edit()
        rgb = parse_label_color(self.label_color.get())
        spec = self._label_spec()
        self._label_updating = True
        try:
            self.label_color.set(f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}")
            self.label_x.set(str(spec.x))
            self.label_y.set(str(spec.y))
        finally:
            self._label_updating = False
        self._push_undo_state(before)
        self._write_label_fields_to_layer()
        self._refresh_layers_panel()
        self._refresh_now()

    def _label_use_change_to(self) -> None:
        if self.range_map is None or not self.range_map.ranges:
            return
        if self._has_range_selection():
            idx = int(self.selected_index)
        else:
            idx = 0
        rgb = self.range_map.ranges[idx].replacement_rgb
        before = self._capture_edit()
        self.label_color.set(rgb_to_hex(rgb))
        self._write_label_fields_to_layer()
        self._push_undo_state(before)
        self._refresh_layers_panel()
        self._refresh_now()

    def _commit_drag_rect(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        *,
        display_size: tuple[int, int] | None = None,
    ) -> tuple[int, int, int, int] | None:
        """Store a preview drag-rect as the Detect ROI in source pixels."""
        if self.work_image is None:
            return None
        disp = display_size
        if disp is None and self._orig_pil is not None:
            disp = self._orig_pil.size
        if disp is None:
            cropped = self._apply_view_crop(self.work_image)
            disp = cropped.size
        cx, cy, cz = self._crop_xy_zoom()
        box = display_box_to_source(
            (x0, y0, x1, y1), disp, self._crop_src_size(), cx, cy, cz
        )
        if box is None:
            return None
        self._detect_roi = box
        self._label_mark_mode = False
        self._sync_label_modes()
        self.status.set("Area selected. Detect searches this region.")
        self._refresh_now()
        return box

    def _on_preview_mark_rect(self, x0: int, y0: int, x1: int, y1: int) -> None:
        if self._orig_pil is None:
            return
        iw, ih = self._orig_pil.size
        photo = self._orig_photo
        dw, dh = iw, ih
        if photo is not None:
            try:
                dw, dh = int(photo.width()), int(photo.height())
            except tk.TclError:
                pass

        def _map(x: int, y: int) -> tuple[int, int]:
            px = int(round(x * iw / max(dw, 1)))
            py = int(round(y * ih / max(dh, 1)))
            return max(0, min(iw - 1, px)), max(0, min(ih - 1, py))

        d0 = _map(x0, y0)
        d1 = _map(x1, y1)
        self._commit_drag_rect(d0[0], d0[1], d1[0], d1[1], display_size=(iw, ih))

    def _place_label_at_display(self, px: int, py: int) -> None:
        if self._orig_pil is None:
            return
        cx, cy, cz = self._crop_xy_zoom()
        sx, sy = display_xy_to_source(
            px, py, self._orig_pil.size, self._crop_src_size(), cx, cy, cz
        )
        before = self._capture_edit()
        self.label_x.set(str(sx))
        self.label_y.set(str(sy))
        if self._selected_label_layer() is None:
            spec = self._label_spec()
            n = 1 + sum(1 for row in self.layer_stack.layers if row.is_label())
            self.layer_stack.add_label(
                name=f"Label {n}" if n > 1 else "Label",
                text=spec.text or "Label",
                size=spec.size,
                color=spec.color,
                font=spec.font,
                x=sx,
                y=sy,
            )
        self._write_label_fields_to_layer()
        self._label_place_mode = False
        self._sync_label_modes()
        self._push_undo_state(before)
        self._refresh_layers_panel()
        self.status.set(f"Label at {sx}, {sy} px (source).")
        self._refresh_now()

    def _try_select_detect_at_display(self, px: int, py: int) -> bool:
        if not self._detect_boxes or self._orig_pil is None:
            return False
        cx, cy, cz = self._crop_xy_zoom()
        sx, sy = display_xy_to_source(
            px, py, self._orig_pil.size, self._crop_src_size(), cx, cy, cz
        )
        hit = None
        for i, box in enumerate(self._detect_boxes):
            if box_contains(box, sx, sy, slop=3):
                hit = i
        if hit is None:
            return False
        if hit in self._selected_detect:
            self._selected_detect.discard(hit)
        else:
            self._selected_detect.add(hit)
        n = len(self._selected_detect)
        self.status.set(f"{n} region{'s' if n != 1 else ''} selected.")
        self._refresh_now()
        return True
