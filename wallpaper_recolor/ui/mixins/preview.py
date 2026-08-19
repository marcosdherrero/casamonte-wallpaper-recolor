# -*- coding: utf-8 -*-
"""
wallpaper_recolor.ui.mixins.preview
------------------------------
Original/Result Fit, checker composite, eyedrop, clusters glue.

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


class AppPreviewMixin:
    """Original/Result Fit, checker composite, eyedrop, clusters glue."""

    # ---------------------------------------------------------------------------
    # Composite Fit / letterbox / view-zoom (independent of Clusters Lab zoom)
    # ---------------------------------------------------------------------------
    def _preview_zoom_hosts(self) -> tuple[PreviewZoomHost, ...]:
        hosts = []
        for name in (
            "orig_zoom_host",
            "tex_zoom_host",
            "tile_zoom_host",
            "seam_zoom_host",
            "mock_zoom_host",
        ):
            host = getattr(self, name, None)
            if host is not None:
                hosts.append(host)
        return tuple(hosts)

    def _composite_hosts(self) -> tuple[PreviewZoomHost, PreviewZoomHost] | None:
        orig = getattr(self, "orig_zoom_host", None)
        tex = getattr(self, "tex_zoom_host", None)
        if orig is None or tex is None:
            return None
        return orig, tex

    def _mirror_composite_pan(self, source: PreviewZoomHost) -> None:
        """Copy Original/Result pan so both panes share one scroll origin."""
        pair = self._composite_hosts()
        if pair is None or getattr(self, "_pan_syncing", False):
            return
        orig, tex = pair
        other = tex if source is orig else orig if source is tex else None
        if other is None:
            return
        self._pan_syncing = True
        try:
            other._pan_x = source._pan_x
            other._pan_y = source._pan_y
            other._layout(propagate=False)
            if other._pan_x != source._pan_x or other._pan_y != source._pan_y:
                source._pan_x = other._pan_x
                source._pan_y = other._pan_y
                source._layout(propagate=False)
            self._preview_pan_x = source._pan_x
            self._preview_pan_y = source._pan_y
            self._sync_composite_letterbox()
        finally:
            self._pan_syncing = False

    def _fit_pane_size(self, host: PreviewZoomHost) -> tuple[int, int]:
        """Dest rectangle for contain-fit: host size with overflow bars hidden.

        Scrollbars sit inside the host, so measuring the viewport while they
        are mapped under-counts the pane and 100% still crops. Hide them, then
        use the stable host box so Original and Result share one contain size.
        """
        try:
            host._sync_scrollbars(False, False)
        except (tk.TclError, AttributeError):
            pass
        try:
            return max(1, int(host.winfo_width())), max(1, int(host.winfo_height()))
        except (tk.TclError, AttributeError):
            return 1, 1

    def _bind_preview_fit_resize(self) -> None:
        """Refit 100% when Original / Result / tool panes change size."""
        for host in self._preview_zoom_hosts():
            host.bind("<Configure>", self._on_preview_fit_configure, add="+")

    def _on_preview_fit_configure(self, event) -> None:
        if getattr(event, "widget", None) not in self._preview_zoom_hosts():
            return
        self._schedule_preview_fit()

    def _schedule_preview_fit(self) -> None:
        if self._orig_pil is None and self._tool_pil is None:
            return
        job = getattr(self, "_fit_job", None)
        if job is not None:
            try:
                self.root.after_cancel(job)
            except tk.TclError:
                pass
        try:
            self._fit_job = self.root.after(PREVIEW_DEBOUNCE_MS, self._flush_preview_fit)
        except tk.TclError:
            self._fit_job = None

    def _flush_preview_fit(self) -> None:
        self._fit_job = None
        if self._orig_pil is None and self._tool_pil is None:
            return
        key = tuple(self._fit_pane_size(host) for host in self._preview_zoom_hosts())
        if key == self._fit_panes_applied:
            return
        self._fit_panes_applied = key
        self._apply_preview_zoom()

    def _composite_shared_pane(self) -> tuple[int, int] | None:
        """Equal-width destination used to fit Original and Result together."""
        panes: list[tuple[int, int]] = []
        for name in ("orig_zoom_host", "tex_zoom_host"):
            host = getattr(self, name, None)
            if host is not None:
                panes.append(self._fit_pane_size(host))
        return shared_pane_size(tuple(panes))

    def _composite_zoom_max_edge(
        self, src_w: int | None = None, src_h: int | None = None
    ) -> int:
        """Shared 100% long-edge cap for Original and Result (one destination pane)."""
        if src_w is None or src_h is None:
            if self._orig_pil is not None:
                src_w, src_h = self._orig_pil.size
            elif self._tex_pil is not None:
                src_w, src_h = self._tex_pil.size
            else:
                src_w, src_h = 1, 1
        pane = self._composite_shared_pane()
        panes = (pane, pane) if pane is not None else ((1, 1), (1, 1))
        return fit_max_edge(
            int(src_w), int(src_h), panes, fallback=PREVIEW_MAX_EDGE
        )

    def _sync_composite_letterbox(self) -> None:
        """Same dest rectangle and pan in both Composite panes (centered letterbox)."""
        pair = self._composite_hosts()
        if pair is None:
            return
        orig, tex = pair
        try:
            vw = min(orig._vp_size()[0], tex._vp_size()[0])
            vh = min(orig._vp_size()[1], tex._vp_size()[1])
        except (tk.TclError, AttributeError):
            orig._layout(propagate=False)
            tex._layout(propagate=False)
            return
        if not pane_usable_for_fit(vw, vh):
            orig._layout(propagate=False)
            tex._layout(propagate=False)
            return
        iw, ih = orig._img_size()
        if iw <= 0 or ih <= 0:
            iw, ih = tex._img_size()
        lx, ly = letterbox_xy(iw, ih, vw, vh)
        overflow_x = iw > vw
        overflow_y = ih > vh
        if self._preview_zoom_factor() <= 1.0 + 1e-6:
            overflow_x = False
            overflow_y = False
            orig._pan_x = tex._pan_x = 0
            orig._pan_y = tex._pan_y = 0
        if overflow_x:
            orig._pan_x = tex._pan_x = min(max(0, orig._pan_x), max(0, iw - vw))
        else:
            orig._pan_x = tex._pan_x = 0
        if overflow_y:
            orig._pan_y = tex._pan_y = min(max(0, orig._pan_y), max(0, ih - vh))
        else:
            orig._pan_y = tex._pan_y = 0
        x = -orig._pan_x if overflow_x else lx
        y = -orig._pan_y if overflow_y else ly
        for host in pair:
            try:
                host.viewport.configure(bg=host._bg)
                if host._photo is not None and iw > 0 and ih > 0:
                    host.image_label.place(x=x, y=y, width=iw, height=ih)
                host._sync_scrollbars(overflow_x, overflow_y)
                host._update_scrollbar_values()
            except tk.TclError:
                pass
        self._preview_pan_x = orig._pan_x
        self._preview_pan_y = orig._pan_y

    def _composite_view_size(self) -> tuple[int, int]:
        """Shared on-screen pixel size for Original and Result at the current zoom."""
        z = self._preview_zoom_factor()
        if self._orig_pil is not None:
            sw, sh = self._orig_pil.size
        elif self._tex_pil is not None:
            sw, sh = self._tex_pil.size
        else:
            sw, sh = 1, 1
        pane = self._composite_shared_pane()
        if pane is not None:
            fitted_w, fitted_h = contain_size(sw, sh, *pane)
            if z <= 1.0 + 1e-6:
                return fitted_w, fitted_h
            return (
                max(1, int(round(fitted_w * z))),
                max(1, int(round(fitted_h * z))),
            )
        max_edge = self._composite_zoom_max_edge(sw, sh)
        return _view_zoom_size(max(1, sw), max(1, sh), z, max_edge)

    def _current_tool_fit_max_edge(self) -> int:
        """100% long-edge cap for the active 3×3 / seam / mockup pane."""
        img = self._tool_pil
        kind = self._selected_inspection_kind()
        host = None
        fallback = PREVIEW_MAX_EDGE
        if kind == "tile":
            host = getattr(self, "tile_zoom_host", None)
            fallback = TILE_VIEW_MAX_EDGE
        elif kind == "seam":
            host = getattr(self, "seam_zoom_host", None)
            fallback = SEAM_VIEW_MAX_EDGE
        elif kind == "mockup":
            host = getattr(self, "mock_zoom_host", None)
            fallback = MOCKUP_VIEW_MAX_EDGE
        if img is None:
            return int(self._tool_zoom_max_edge or fallback)
        if host is None:
            edge = max(1, int(fallback))
        else:
            edge = fit_max_edge(
                *img.size, (self._fit_pane_size(host),), fallback=fallback
            )
        self._tool_zoom_max_edge = edge
        return edge

    def _preview_image_labels(self) -> tuple[tk.Misc, ...]:
        labels = []
        for name in ("orig_label", "tex_label", "tile_label", "seam_label", "mock_label"):
            widget = getattr(self, name, None)
            if widget is not None:
                labels.append(widget)
        return tuple(labels)

    def _widget_is_preview_image(self, widget: tk.Misc) -> bool:
        """True if ``widget`` is a preview photo label (not letterbox / header)."""
        labels = self._preview_image_labels()
        current: tk.Misc | None = widget
        seen: set[str] = set()
        while current is not None:
            key = str(current)
            if key in seen:
                break
            seen.add(key)
            if current in labels:
                return True
            current = getattr(current, "master", None)
        return False

    def _pointer_over_preview_image(self, event) -> bool:
        """True when the pointer is over Original / Result / tile / seam / mock photo."""
        w = getattr(event, "widget", None)
        if isinstance(w, str):
            try:
                w = self.root.nametowidget(w)
            except (KeyError, tk.TclError):
                w = None
        if w is not None and isinstance(w, tk.Misc) and self._widget_is_preview_image(w):
            return True
        for x, y in self._wheel_event_xy(event):
            for lab in self._preview_image_labels():
                if _widget_contains_root(lab, x, y):
                    return True
            try:
                hit = self.root.winfo_containing(x, y)
            except tk.TclError:
                hit = None
            if hit is not None and self._widget_is_preview_image(hit):
                return True
        return False

    def _pointer_over_clusters(self, event) -> bool:
        plot = getattr(self, "cluster_plot", None)
        if plot is None or not self._clusters_tab_selected():
            return False
        for x, y in self._wheel_event_xy(event):
            if plot.contains_root(x, y):
                return True
        w = getattr(event, "widget", None)
        if isinstance(w, str):
            try:
                w = self.root.nametowidget(w)
            except (KeyError, tk.TclError):
                w = None
        current = w if isinstance(w, tk.Misc) else None
        while current is not None:
            if current is plot:
                return True
            current = getattr(current, "master", None)
        return False

    def _on_preview_ctrl_wheel(self, event) -> str | None:
        """Wheel over a preview page: view-zoom, do not page-scroll.

        Composite: Original / Result / tool photo. Clusters: Lab camera only.
        Wheel over the rest of the column still page-scrolls. Ctrl is not required.
        Windows ``delta > 0`` (wheel up) zooms in; ``'??'`` num is ignored.
        """
        over_clusters = self._pointer_over_clusters(event)
        over_image = self._pointer_over_preview_image(event)
        if not over_clusters and not over_image:
            return None
        if over_clusters and not over_image and not self._clusters_tab_selected():
            return None
        delta = _wheel_zoom_pct_delta(event)
        if delta == 0.0:
            return "break"
        self._nudge_preview_zoom(delta, event=event)
        return "break"

    def _preview_zoom_factor(self) -> float:
        """Wallpaper Composite zoom (never the Clusters camera)."""
        try:
            pct = float(self._composite_zoom_pct)
        except (TypeError, ValueError):
            pct = VIEW_ZOOM_PCT_DEFAULT
        return view_zoom_factor(pct)

    def _nudge_preview_zoom(self, delta_pct: float, event=None) -> None:
        try:
            current = float(self.preview_zoom.get())
        except (tk.TclError, ValueError, TypeError):
            current = VIEW_ZOOM_PCT_DEFAULT
        self._set_preview_zoom_pct(current + float(delta_pct), event=event)

    def _on_preview_zoom_slider(self, _value: str = "") -> None:
        if self._mute_ui or self._preview_zoom_updating:
            return
        self._set_preview_zoom_pct(self.preview_zoom.get())

    def _reset_preview_zoom(self) -> None:
        """Fit the active Preview tab (Composite wallpaper or Clusters camera)."""
        self._set_preview_zoom_pct(VIEW_ZOOM_PCT_DEFAULT, reset_pan=True)

    def _sync_view_zoom_header(self) -> None:
        """Show the selected tab's zoom on the slider without applying the other page."""
        clusters = self._clusters_tab_selected()
        pct = self._cluster_zoom_pct if clusters else self._composite_zoom_pct
        self.view_zoom_title.set("View zoom")
        try:
            self.preview_zoom_scale.configure(
                to=CLUSTER_ZOOM_PCT_MAX if clusters else VIEW_ZOOM_PCT_MAX
            )
        except (tk.TclError, AttributeError):
            pass
        prev = self._preview_zoom_updating
        self._preview_zoom_updating = True
        try:
            self.preview_zoom.set(float(pct))
            self.preview_zoom_caption.set(f"{int(round(pct))}%")
        finally:
            self._preview_zoom_updating = prev

    def _on_cluster_camera_zoom(self, pct: float) -> None:
        self._cluster_zoom_pct = clamp_cluster_zoom_pct(pct)
        if self._clusters_tab_selected():
            self._sync_view_zoom_header()

    def _on_cluster_pick(self, rgb: tuple[int, int, int], _y: int, _x: int) -> None:
        """Double-click a scatter point: sample source RGB into the selected half."""
        self._apply_eyedrop_rgb((int(rgb[0]), int(rgb[1]), int(rgb[2])))

    def _cluster_selected_rgb(self) -> tuple[int, int, int] | None:
        if self._has_range_selection() and self.range_map is not None:
            try:
                band = self.range_map.ranges[self.selected_index]
            except (IndexError, AttributeError):
                band = None
            else:
                if self.selected_half == HALF_MATCH:
                    return tuple(int(c) for c in band.match_rgb)
                return tuple(int(c) for c in band.replacement_rgb)
        rgb = getattr(self, "_scratch_rgb", None)
        if rgb is None:
            return None
        return (int(rgb[0]), int(rgb[1]), int(rgb[2]))

    def _on_cluster_move_start(self) -> None:
        """One undo snapshot at MMB press; live Lab drags commit on release."""
        if not self._has_range_selection():
            return
        if self._wheel_before is None:
            self._wheel_before = self._capture_edit()

    def _on_cluster_move(self, rgb: tuple[int, int, int]) -> None:
        """Live MMB Lab drag — debounce preview like the color wheel."""
        color = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        if not self._has_range_selection():
            self._set_scratch_rgb(color)
            return
        if self.range_map is None:
            return
        if self._wheel_before is None:
            self._wheel_before = self._capture_edit()
        self.set_range_color(self.selected_index, color)
        self._load_selected_onto_wheel()

    def _on_cluster_move_end(self, rgb: tuple[int, int, int]) -> None:
        self._on_wheel_commit((int(rgb[0]), int(rgb[1]), int(rgb[2])))

    def _set_preview_zoom_pct(
        self,
        pct: float,
        *,
        event=None,
        reset_pan: bool = False,
    ) -> None:
        new_pct = (
            clamp_cluster_zoom_pct(pct)
            if self._clusters_tab_selected()
            else clamp_view_zoom_pct(pct)
        )
        if self._clusters_tab_selected():
            self._cluster_zoom_pct = float(new_pct)
            prev = self._preview_zoom_updating
            self._preview_zoom_updating = True
            try:
                self.preview_zoom.set(float(new_pct))
                self.preview_zoom_caption.set(f"{int(round(new_pct))}%")
            finally:
                self._preview_zoom_updating = prev
            plot = getattr(self, "cluster_plot", None)
            if plot is not None:
                if reset_pan:
                    plot.center_view()
                plot.set_zoom_pct(new_pct, notify=False)
            return
        old_z = self._preview_zoom_factor()
        prev = self._preview_zoom_updating
        self._preview_zoom_updating = True
        try:
            self._composite_zoom_pct = float(new_pct)
            self.preview_zoom.set(float(new_pct))
            self.preview_zoom_caption.set(f"{int(round(new_pct))}%")
        finally:
            self._preview_zoom_updating = prev
        new_z = view_zoom_factor(new_pct)
        self._apply_preview_zoom()
        pair = self._composite_hosts()
        if reset_pan:
            self._preview_pan_x = 0
            self._preview_pan_y = 0
            if pair is not None:
                orig, tex = pair
                orig._pan_x = orig._pan_y = 0
                tex._pan_x = tex._pan_y = 0
                orig._layout(propagate=False)
                tex._layout(propagate=False)
            for host in self._preview_zoom_hosts():
                if pair is not None and host in pair:
                    continue
                host.reset_pan()
        elif event is not None and abs(new_z - old_z) > 1e-6:
            try:
                x_root = int(event.x_root)
                y_root = int(event.y_root)
            except (TypeError, ValueError, AttributeError):
                x_root = y_root = None  # type: ignore[assignment]
            if x_root is not None:
                for host in self._preview_zoom_hosts():
                    try:
                        if (
                            _widget_contains_root(host, x_root, y_root)
                            or _widget_contains_root(host.viewport, x_root, y_root)
                            or _widget_contains_root(host.image_label, x_root, y_root)
                        ):
                            host.zoom_anchor(old_z, new_z, x_root, y_root)
                            break
                    except tk.TclError:
                        continue
        elif abs(new_z - old_z) > 1e-6 and old_z > 0:
            ratio = new_z / old_z
            if pair is not None:
                orig, _tex = pair
                orig._pan_x = int(round(orig._pan_x * ratio))
                orig._pan_y = int(round(orig._pan_y * ratio))
                orig._layout()
            for host in self._preview_zoom_hosts():
                if pair is not None and host in pair:
                    continue
                host._pan_x = int(round(host._pan_x * ratio))
                host._pan_y = int(round(host._pan_y * ratio))
                host._layout()
        if hasattr(self, "texture_reset"):
            self._sync_slider_resets()

    def _apply_preview_zoom(self) -> None:
        """Rebuild displayed preview photos from stored PILs. Does not recrop.

        Original and Result share one on-screen size (contain-fit × zoom).
        """
        for host in self._preview_zoom_hosts():
            try:
                host._sync_scrollbars(False, False)
            except tk.TclError:
                pass
        z = self._preview_zoom_factor()
        max_edge = self._composite_zoom_max_edge()
        shared = None
        if self._orig_pil is not None or self._tex_pil is not None:
            shared = self._composite_view_size()
        if self._eyedrop_overlay is not None:
            self._hide_eyedrop_overlay()
        if self._orig_pil is not None:
            shown = _scale_view_zoom(
                self._orig_pil, z, max_edge=max_edge, size=shared
            )
            shown = composite_over_checker(shown)
            self._orig_photo = ImageTk.PhotoImage(shown, master=self.root)
            if hasattr(self, "orig_zoom_host"):
                self.orig_zoom_host.set_photo(self._orig_photo)
            else:
                self.orig_label.configure(image=self._orig_photo)
        if self._tex_pil is not None:
            shown = _scale_view_zoom(
                self._tex_pil, z, max_edge=max_edge, size=shared
            )
            shown = composite_over_checker(shown)
            self._tex_photo = ImageTk.PhotoImage(shown, master=self.root)
            if hasattr(self, "tex_zoom_host"):
                self.tex_zoom_host.set_photo(self._tex_photo)
            else:
                self.tex_label.configure(image=self._tex_photo)
        if self._tool_pil is not None:
            shown = _scale_view_zoom(
                self._tool_pil, z, max_edge=self._current_tool_fit_max_edge()
            )
            self._tool_photo = ImageTk.PhotoImage(shown, master=self.root)
            kind = self._selected_inspection_kind()
            if kind == "tile" and hasattr(self, "tile_zoom_host"):
                self.tile_zoom_host.set_photo(self._tool_photo)
            elif kind == "seam" and hasattr(self, "seam_zoom_host"):
                self.seam_zoom_host.set_photo(self._tool_photo)
            elif kind == "mockup" and hasattr(self, "mock_zoom_host"):
                self.mock_zoom_host.set_photo(self._tool_photo)
        self._fit_panes_applied = tuple(
            self._fit_pane_size(host) for host in self._preview_zoom_hosts()
        )
        self._sync_composite_letterbox()
        self._sync_eyedrop_cursor()

    # ---------------------------------------------------------------------------
    # Eyedrop on Original; Result click places / selects labels
    # ---------------------------------------------------------------------------
    def _on_original_click(self, event) -> None:
        """Eyedrop the Original preview, or select / place a label."""
        if getattr(self, "orig_zoom_host", None) is not None and (
            self.orig_zoom_host.panning or self.orig_zoom_host.moving_layer
        ):
            return
        x, y = self._event_to_orig_label_xy(event)
        mapped = self._orig_click_to_display(x, y)
        if mapped is None:
            return
        if self._label_place_mode:
            self._place_label_at_display(*mapped)
            return
        if self._try_select_detect_at_display(*mapped):
            return
        rgb = self._sample_original_rgb(x, y)
        if rgb is None:
            return
        self._apply_eyedrop_rgb(rgb)

    def _on_result_click(self, event) -> None:
        """Select detection boxes or place a label on the Result preview."""
        if getattr(self, "tex_zoom_host", None) is not None and (
            self.tex_zoom_host.panning or self.tex_zoom_host.moving_layer
        ):
            return
        host = self.tex_zoom_host
        try:
            x = int(event.x_root) - int(host.image_label.winfo_rootx())
            y = int(event.y_root) - int(host.image_label.winfo_rooty())
        except (tk.TclError, TypeError, ValueError, AttributeError):
            x, y = int(getattr(event, "x", 0)), int(getattr(event, "y", 0))
        mapped = self._orig_click_to_display(x, y)
        if mapped is None:
            return
        if self._label_place_mode:
            self._place_label_at_display(*mapped)
            return
        self._try_select_detect_at_display(*mapped)

    def _apply_eyedrop_rgb(self, rgb: tuple[int, int, int]) -> None:
        """Set the active swatch from a sampled Original pixel (undo as one wheel-like tick)."""
        if primary_is_label(self.layer_stack):
            if self._wheel_before is None:
                self._wheel_before = self._capture_edit()
            ly = self._selected_label_layer()
            if ly is not None:
                ly.color = tuple(int(c) for c in rgb)
                self._sync_label_fields_from_layer(ly)
                self._schedule_preview()
            self._on_wheel_commit(rgb)
            return
        if self.range_map is None:
            return
        if not self._has_range_selection():
            self._set_scratch_rgb(rgb)
            self._on_wheel_commit(rgb)
            return
        if self._wheel_before is None:
            self._wheel_before = self._capture_edit()
        self.set_range_color(self.selected_index, rgb)
        self._load_selected_onto_wheel()
        self._on_wheel_commit(rgb)

    def _sample_original_rgb(self, click_x: int, click_y: int) -> tuple[int, int, int] | None:
        """Map pipette-tip coords on the Original label to a cropped work-image RGB."""
        mapped = self._orig_click_to_display(click_x, click_y)
        if mapped is None or self.work_image is None or self._orig_pil is None:
            return None
        px, py = mapped
        iw, ih = self._orig_pil.size
        view = self._apply_view_crop(self.work_image)
        ww, wh = view.size
        sx = min(ww - 1, max(0, int(px * ww / max(iw, 1))))
        sy = min(wh - 1, max(0, int(py * wh / max(ih, 1))))
        pix = view.getpixel((sx, sy))
        if isinstance(pix, int):
            return (pix, pix, pix)
        return (int(pix[0]), int(pix[1]), int(pix[2]))

    def _event_to_orig_label_xy(self, event) -> tuple[int, int]:
        """Pointer → orig_label coords (maps through pan; zoom applied in display map)."""
        x_root = getattr(event, "x_root", None)
        y_root = getattr(event, "y_root", None)
        if x_root is not None and y_root is not None:
            try:
                return (
                    int(x_root) - int(self.orig_label.winfo_rootx()),
                    int(y_root) - int(self.orig_label.winfo_rooty()),
                )
            except (tk.TclError, TypeError, ValueError):
                pass
        return int(getattr(event, "x", 0)), int(getattr(event, "y", 0))

    def _orig_click_to_display(self, click_x: int, click_y: int) -> tuple[int, int] | None:
        """Map Original-label coords through view pan/zoom to an ``_orig_pil`` pixel.

        ``click_x`` / ``click_y`` are orig_label space (the NEAREST-scaled photo).
        Scale by displayed size vs ``_orig_pil`` so the pipette hits the same
        work pixel the loupe samples — not ``click / zoom`` of a 560px preview.
        """
        if self._orig_pil is None:
            return None
        z = self._preview_zoom_factor()
        iw, ih = self._orig_pil.size
        max_edge = self._composite_zoom_max_edge(iw, ih)
        photo = self._orig_photo
        if photo is not None:
            try:
                dw = int(photo.width())
                dh = int(photo.height())
            except tk.TclError:
                dw, dh = _view_zoom_size(iw, ih, z, max_edge)
            try:
                lw = max(int(self.orig_label.winfo_width()), 1)
                lh = max(int(self.orig_label.winfo_height()), 1)
            except tk.TclError:
                lw, lh = dw, dh
            x0 = (lw - dw) // 2
            y0 = (lh - dh) // 2
            zx = int(click_x) - x0
            zy = int(click_y) - y0
            if zx < 0 or zy < 0 or zx >= dw or zy >= dh:
                return None
        else:
            dw, dh = _view_zoom_size(iw, ih, z, max_edge)
            zx, zy = int(click_x), int(click_y)
            if zx < 0 or zy < 0 or zx >= dw or zy >= dh:
                return None
        px = int(zx * iw / max(dw, 1))
        py = int(zy * ih / max(dh, 1))
        if px < 0 or py < 0 or px >= iw or py >= ih:
            return None
        return px, py

    def _eyedrop_enabled(self) -> bool:
        """Original click samples while a match-from / change-to swatch can receive it."""
        if self._label_mark_mode or self._label_place_mode:
            return False
        return self.range_map is not None

    def _on_eyedrop_button(self) -> None:
        """Coverage dropper — remind that Original click fills the active swatch."""
        if self.range_map is None:
            self.status.set("Open an image, then click Original to sample a color.")
            return
        self._sync_eyedrop_cursor()
        half = "match-from" if self.selected_half == HALF_MATCH else "change-to"
        self.status.set(f"Click Original to sample into {half}")

    def _set_eyedrop_widget_cursor(self, cursor: str) -> None:
        widgets = [self.orig_host, self.orig_label, self._eyedrop_overlay]
        if getattr(self, "orig_zoom_host", None) is not None:
            widgets.extend([self.orig_zoom_host, self.orig_zoom_host.viewport])
        for widget in widgets:
            if widget is None:
                continue
            try:
                widget.configure(cursor=cursor)
            except tk.TclError:
                if cursor == "none":
                    try:
                        widget.configure(cursor="crosshair")
                    except tk.TclError:
                        pass

    def _sync_eyedrop_cursor(self) -> None:
        """Hide the system cursor on Original; the FA dropper overlay follows the pointer."""
        if not self._eyedrop_enabled():
            self._set_eyedrop_widget_cursor("")
            self._hide_eyedrop_overlay()
            return
        self._set_eyedrop_widget_cursor("none")

    def _label_to_host_xy(self, x: int, y: int) -> tuple[int, int]:
        """orig_label coords → orig_host place coords (through pan)."""
        try:
            return (
                x + int(self.orig_label.winfo_rootx()) - int(self.orig_host.winfo_rootx()),
                y + int(self.orig_label.winfo_rooty()) - int(self.orig_host.winfo_rooty()),
            )
        except tk.TclError:
            return x, y

    def _host_to_label_xy(self, x: int, y: int) -> tuple[int, int]:
        """orig_host coords → orig_label coords (pipette tip / sample space)."""
        try:
            return (
                x + int(self.orig_host.winfo_rootx()) - int(self.orig_label.winfo_rootx()),
                y + int(self.orig_host.winfo_rooty()) - int(self.orig_label.winfo_rooty()),
            )
        except tk.TclError:
            return x, y

    def _host_pointer_xy(self) -> tuple[int, int] | None:
        """Mouse position in orig_host, or None if the pointer query fails."""
        try:
            return (
                int(self.orig_host.winfo_pointerx()) - int(self.orig_host.winfo_rootx()),
                int(self.orig_host.winfo_pointery()) - int(self.orig_host.winfo_rooty()),
            )
        except tk.TclError:
            return None

    def _overlay_tip_label_xy(self) -> tuple[int, int] | None:
        """Pipette tip in orig_label coords from the placed overlay + hotspot."""
        overlay = self._eyedrop_overlay
        if overlay is None:
            return None
        hx, hy = self._eyedrop_hotspot
        try:
            if str(overlay.winfo_manager()) != "place":
                return None
            host_x = int(overlay.winfo_x()) + hx
            host_y = int(overlay.winfo_y()) + hy
        except tk.TclError:
            return None
        return self._host_to_label_xy(host_x, host_y)

    def _loupe_anchor(self, x: int, y: int, width: int, height: int) -> tuple[int, int]:
        """Top-right of the sample point; flip/clamp so the circle stays in the preview."""
        size = _LOUPE_PX
        gap = _LOUPE_GAP
        hw = max(int(width), 1)
        hh = max(int(height), 1)
        lx = x + gap
        ly = y - size - gap
        if lx + size > hw:
            lx = x - size - gap
        if ly < 0:
            ly = y + gap
        lx = min(max(0, lx), max(0, hw - size))
        ly = min(max(0, ly), max(0, hh - size))
        return lx, ly

    def _restore_orig_preview(self) -> None:
        """Undo a stamped dropper so Original shows the clean preview again."""
        if self._orig_photo is None:
            self._orig_eyedrop_photo = None
            return
        try:
            self.orig_label.configure(image=self._orig_photo)
        except tk.TclError:
            pass
        self._orig_eyedrop_photo = None

    def _stamp_eyedrop_at_display(self, px: int, py: int) -> None:
        """Composite dropper + loupe onto the preview so transparent corners show wallpaper.

        ``px, py`` are ``_orig_pil`` pixels (same as ``_sample_original_rgb``).
        The loupe still samples that neighborhood; the glyph is placed on the zoomed photo.
        """
        if self._orig_pil is None:
            return
        z = self._preview_zoom_factor()
        shown = _scale_view_zoom(
            self._orig_pil,
            z,
            max_edge=self._composite_zoom_max_edge(),
            size=self._composite_view_size(),
        )
        hx, hy = self._eyedrop_hotspot
        base = shown.convert("RGBA")
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        loupe = _make_eyedrop_loupe_image(self._orig_pil, px, py)
        iw, ih = self._orig_pil.size
        dw, dh = shown.size
        sx = dw / max(iw, 1)
        sy = dh / max(ih, 1)
        zx = int(round(px * sx + (sx - 1.0) * 0.5))
        zy = int(round(py * sy + (sy - 1.0) * 0.5))
        zx = min(max(0, zx), max(0, base.size[0] - 1))
        zy = min(max(0, zy), max(0, base.size[1] - 1))
        lx, ly = self._loupe_anchor(zx, zy, *base.size)
        _paste_rgba(layer, loupe, (lx, ly))
        _paste_rgba(layer, self._eyedrop_pil, (zx - hx, zy - hy))
        stamped = Image.alpha_composite(base, layer)
        self._orig_eyedrop_photo = ImageTk.PhotoImage(stamped.convert("RGB"), master=self.root)
        try:
            self.orig_label.configure(image=self._orig_eyedrop_photo)
        except tk.TclError:
            pass

    def _place_eyedrop_overlay(self, x: int, y: int) -> None:
        """Pin the FA dropper so its pipette tip sits on the sample point ``(x, y)``.

        ``(x, y)`` is orig_label space — the same coords ``_sample_original_rgb`` uses.
        Over the image the glyph and loupe are alpha-composited (no widget rectangle).
        In the letterbox a chrome-less Canvas follows the pointer with host-matching bg.
        """
        overlay = self._eyedrop_overlay
        if overlay is None:
            return
        hx, hy = self._eyedrop_hotspot
        mapped = None
        if self._orig_pil is not None:
            mapped = self._orig_click_to_display(x, y)
        if mapped is not None:
            self._stamp_eyedrop_at_display(mapped[0], mapped[1])
            try:
                if str(overlay.winfo_manager()) == "place":
                    overlay.place_forget()
            except tk.TclError:
                pass
            return
        self._restore_orig_preview()
        host_x, host_y = self._label_to_host_xy(x, y)
        overlay.place(x=host_x - hx, y=host_y - hy)
        tk.Misc.lift(overlay)

    def _hide_eyedrop_overlay(self) -> None:
        overlay = self._eyedrop_overlay
        if overlay is not None:
            try:
                if str(overlay.winfo_manager()) == "place":
                    overlay.place_forget()
            except tk.TclError:
                pass
        self._restore_orig_preview()

    def _on_orig_eyedrop_move(self, event) -> None:
        if not self._eyedrop_enabled():
            return
        x, y = self._event_to_orig_label_xy(event)
        self._place_eyedrop_overlay(x, y)

    def _on_eyedrop_overlay_move(self, event) -> None:
        if not self._eyedrop_enabled():
            return
        pointer = self._host_pointer_xy()
        if pointer is not None:
            x, y = self._host_to_label_xy(*pointer)
        else:
            overlay = self._eyedrop_overlay
            if overlay is None:
                return
            try:
                x, y = self._host_to_label_xy(
                    int(overlay.winfo_x()) + int(event.x),
                    int(overlay.winfo_y()) + int(event.y),
                )
            except tk.TclError:
                return
        self._place_eyedrop_overlay(x, y)

    def _on_eyedrop_overlay_click(self, event) -> None:
        tip = self._overlay_tip_label_xy()
        if tip is None:
            pointer = self._host_pointer_xy()
            if pointer is None:
                try:
                    overlay = self._eyedrop_overlay
                    hx, hy = self._eyedrop_hotspot
                    tip = self._host_to_label_xy(
                        int(overlay.winfo_x()) + hx,
                        int(overlay.winfo_y()) + hy,
                    )
                except (tk.TclError, AttributeError):
                    return
            else:
                tip = self._host_to_label_xy(*pointer)
        self._on_original_click(SimpleNamespace(x=tip[0], y=tip[1]))

    def _on_orig_eyedrop_leave(self, _event) -> None:
        self.root.after_idle(self._maybe_hide_eyedrop_overlay)

    def _maybe_hide_eyedrop_overlay(self) -> None:
        """Leave the pane (not just orig_label → overlay) before hiding the dropper."""
        if not self._eyedrop_enabled():
            self._hide_eyedrop_overlay()
            return
        try:
            px = int(self.orig_host.winfo_pointerx())
            py = int(self.orig_host.winfo_pointery())
            x0 = int(self.orig_host.winfo_rootx())
            y0 = int(self.orig_host.winfo_rooty())
            x1 = x0 + int(self.orig_host.winfo_width())
            y1 = y0 + int(self.orig_host.winfo_height())
        except tk.TclError:
            self._hide_eyedrop_overlay()
            return
        if x0 <= px < x1 and y0 <= py < y1:
            return
        overlay = self._eyedrop_overlay
        if overlay is not None:
            try:
                if str(overlay.winfo_manager()) == "place":
                    ox = int(overlay.winfo_rootx())
                    oy = int(overlay.winfo_rooty())
                    ox1 = ox + int(overlay.winfo_width())
                    oy1 = oy + int(overlay.winfo_height())
                    if ox <= px < ox1 and oy <= py < oy1:
                        return
            except tk.TclError:
                pass
        self._hide_eyedrop_overlay()

    def _mockup_cover_frac(self) -> float:
        """Floor-up fraction of the back wall covered by wallpaper (1 = full)."""
        try:
            key = str(self.mockup_cover.get())
        except (tk.TclError, ValueError):
            key = DEFAULT_MOCKUP_COVER
        return cover_frac_from_key(key)

    def _on_mockup_cover(self) -> None:
        if self._mute_ui:
            return
        self._schedule_preview()

    def _on_mockup_scale(self, _value: str) -> None:
        if self._mute_ui:
            return
        self.mockup_caption.set(
            f"Repeat scale: {float(self.mockup_repeats.get()):.1f} tiles across the wall"
        )
        self._sync_slider_resets()
        self._schedule_preview()

    def _cancel_preview_job(self) -> None:
        if self._preview_job is None:
            return
        try:
            self.root.after_cancel(self._preview_job)
        except tk.TclError:
            pass
        self._preview_job = None

    def _schedule_preview(self) -> None:
        """Debounce image remap while the wheel or coverage bar is dragging."""
        self._cancel_preview_job()
        self._preview_job = self.root.after(PREVIEW_DEBOUNCE_MS, self._flush_preview)
        self._schedule_cluster_view()

    def _flush_preview(self) -> None:
        self._preview_job = None
        self._refresh_previews()
        self._schedule_cluster_view()

    def _clusters_tab_selected(self) -> bool:
        plot = getattr(self, "cluster_plot", None)
        if plot is None:
            return False
        try:
            return str(self.notebook.select()) == str(plot)
        except tk.TclError:
            return False

    def _schedule_cluster_view(self) -> None:
        job = getattr(self, "_cluster_job", None)
        if job is not None:
            try:
                self.root.after_cancel(job)
            except tk.TclError:
                pass
        if not hasattr(self, "root"):
            return
        try:
            self._cluster_job = self.root.after(CLUSTER_DEBOUNCE_MS, self._flush_cluster_view)
        except tk.TclError:
            self._cluster_job = None

    def _flush_cluster_view(self) -> None:
        self._cluster_job = None
        plot = getattr(self, "cluster_plot", None)
        if plot is None or not self._clusters_tab_selected():
            return
        if self.range_map is None:
            plot.set_data(None)
            return
        data = cluster_scatter_data(
            self.range_map,
            self.work_image,
            mode=plot.mode_key(),
        )
        plot.set_data(data)

    def _refresh_now(self) -> None:
        """Immediate preview (open / rebuild / reset — not a drag)."""
        self._cancel_preview_job()
        self._refresh_previews()
        self._schedule_cluster_view()

    def _select_composite_preview_tab(self) -> None:
        """First open / reset: Composite, not Clusters."""
        tab = getattr(self, "_composite_tab", None)
        notebook = getattr(self, "notebook", None)
        if tab is None or notebook is None:
            return
        try:
            notebook.select(tab)
        except tk.TclError:
            try:
                notebook.select(0)
            except tk.TclError:
                pass

    def _on_tab_changed(self, _event=None) -> None:
        if self._eyedrop_overlay is not None:
            self._hide_eyedrop_overlay()
        self._sync_view_zoom_header()
        if self._clusters_tab_selected():
            self._flush_cluster_view()
            plot = getattr(self, "cluster_plot", None)
            if plot is not None:
                plot.set_zoom_pct(self._cluster_zoom_pct, notify=False)
        if self._work_live is None:
            return
        self._refresh_tool_tab()

    def _master_work(self) -> Image.Image | None:
        """Tile / seam / mockup follow the live result (eyes + texture slider)."""
        return self._work_live

    def _refresh_tool_tab(self) -> None:
        """Fill 3×3 / seam / mockup from the live composite (work-image size)."""
        src = self._master_work()
        if src is None:
            return
        kind = self._selected_inspection_kind()
        if kind is None:
            return
        z = self._preview_zoom_factor()
        src_edge = max(src.size)
        if kind == "tile":
            img = tile_repeat(src, cell_max_edge=src_edge)
            self._tool_pil = img
            max_edge = self._current_tool_fit_max_edge()
            self._tool_photo = ImageTk.PhotoImage(
                _scale_view_zoom(img, z, max_edge=max_edge), master=self.root
            )
            if hasattr(self, "tile_zoom_host"):
                self.tile_zoom_host.set_photo(self._tool_photo)
            else:
                self.tile_label.configure(image=self._tool_photo)
        elif kind == "seam":
            img = offset_seam(src, cell_max_edge=src_edge)
            self._tool_pil = img
            max_edge = self._current_tool_fit_max_edge()
            self._tool_photo = ImageTk.PhotoImage(
                _scale_view_zoom(img, z, max_edge=max_edge), master=self.root
            )
            if hasattr(self, "seam_zoom_host"):
                self.seam_zoom_host.set_photo(self._tool_photo)
            else:
                self.seam_label.configure(image=self._tool_photo)
        elif kind == "mockup":
            img = room_mockup(
                src,
                repeats_x=float(self.mockup_repeats.get()),
                cover_frac=self._mockup_cover_frac(),
            )
            self._tool_pil = img
            max_edge = self._current_tool_fit_max_edge()
            self._tool_photo = ImageTk.PhotoImage(
                _scale_view_zoom(img, z, max_edge=max_edge), master=self.root
            )
            if hasattr(self, "mock_zoom_host"):
                self.mock_zoom_host.set_photo(self._tool_photo)
            else:
                self.mock_label.configure(image=self._tool_photo)

    def _preview_snapshot(self) -> dict | None:
        """Tk-thread copy of preview inputs. Safe to hand to a worker."""
        if self.work_image is None or self.range_map is None:
            return None
        self._ensure_base_layer()
        return {
            "work_image": self.work_image,
            "range_map": self.range_map,
            "holes": tuple(self._inpaint_boxes),
            "quads": tuple(self._inpaint_quads),
            "style": self._wallpaper_style_key(),
            "cancel": getattr(self, "_job_cancel", None),
            "inpaint_layer_id": str(self._inpaint_layer_id or ""),
            "crop_src": self._crop_src_size(),
            "crop": self._crop_xy_zoom(),
            "tess": self._tess_params(),
            "tiles": self._tess_tiles_value(),
            "lloyd": self._tess_lloyd_value(),
            "tone": self._tone_apply_kwargs(),
            "detect_boxes": list(self._detect_boxes),
            "selected_detect": self._selected_detect,
            "detect_roi": self._detect_roi,
            "label_spec": self._label_spec(),
            "stack": self.layer_stack,
            "selected_ids": tuple(self.layer_stack.selected_ids),
            "grain": self._save_uses_grain(),
        }

    def _preview_pils(self, snap: dict) -> tuple[Image.Image, Image.Image, Image.Image]:
        """Crop / tessellate / overlay PIL frames. No Tk (worker-safe)."""
        work_image = snap["work_image"]
        range_map = snap["range_map"]
        holes = snap["holes"]
        crop_src = snap["crop_src"]
        cx, cy, cz = snap["crop"]
        h_side, v_side, built, mode = snap["tess"]
        wrap_holes = normalize_tess_mode(mode) == MODE_TILE
        quads = snap.get("quads")
        style = snap.get("style") or self._wallpaper_style_key()
        cancel = snap.get("cancel")
        stack = snap.get("stack")
        if stack is None:
            stack = LayerStack()
        selected_ids = snap.get("selected_ids", tuple(stack.selected_ids))
        targets = correction_target_ids(stack, selected_ids)
        orig_src = work_image
        primary = stack.primary()
        hole_id = str(snap.get("inpaint_layer_id") or "")
        base_ly = stack.base_layer()
        if not hole_id and base_ly is not None:
            hole_id = base_ly.id
        if (
            primary is not None
            and primary.is_image()
            and not primary.is_base()
            and primary.raster is not None
        ):
            orig_src = primary.raster
        if holes and (
            orig_src is work_image
            or (primary is not None and primary.id == hole_id)
        ):
            orig_src = inpaint_image(
                orig_src,
                holes,
                src_size=crop_src,
                wrap=wrap_holes,
                quads=quads,
                style=style,
                cancel=cancel,
            )
        if orig_src is work_image or orig_src.size == work_image.size:
            orig = apply_crop(orig_src, cx, cy, cz, src_size=crop_src)
        else:
            orig = orig_src.copy()
        # Original is the source crop (inpaint if any). Tone / Normalize lighting
        # live on the Result stack via remap, same as recolor and Texture.
        processed: dict[str, Image.Image] = {}
        for ly in stack.layers:
            if not ly.visible or not ly.is_image():
                continue
            img = work_image if ly.is_base() else ly.raster
            if img is None:
                continue
            selected = ly.id in targets
            fill_holes = bool(holes) and ly.id == hole_id
            if ly.is_base():
                if selected:
                    img = live_composite_from_map(range_map)
                if fill_holes:
                    img = inpaint_image(
                        img,
                        holes,
                        src_size=crop_src,
                        wrap=wrap_holes,
                        quads=quads,
                        style=style,
                        cancel=cancel,
                    )
                img = apply_crop(img, cx, cy, cz, src_size=crop_src)
                if selected:
                    img = apply_tessellate(
                        img,
                        h_side,
                        v_side,
                        built,
                        mode=mode,
                        tiles=snap["tiles"],
                        lloyd=snap["lloyd"],
                    )
            else:
                if selected:
                    img = composite_for_image(
                        img, range_map, grain=bool(snap.get("grain", True))
                    )
                if fill_holes:
                    img = inpaint_image(
                        img,
                        holes,
                        src_size=crop_src,
                        wrap=wrap_holes,
                        quads=quads,
                        style=style,
                        cancel=cancel,
                    )
                if selected:
                    img = apply_tessellate(
                        img,
                        h_side,
                        v_side,
                        built,
                        mode=mode,
                        tiles=snap["tiles"],
                        lloyd=snap["lloyd"],
                    )
            processed[ly.id] = img
        if processed:
            live_rgba = composite_stack(
                stack,
                orig.size,
                crop_src,
                processed=processed,
                crop_x=cx,
                crop_y=cy,
                crop_zoom=cz,
            )
            live = live_rgba
        else:
            live = orig.copy()
        if live.size != orig.size:
            live = live.resize(orig.size, Image.Resampling.NEAREST)
        has_label_layer = any(ly.is_label() and ly.visible for ly in stack.layers)
        orig_disp = decorate_preview(
            orig,
            snap["detect_boxes"],
            snap["selected_detect"],
            crop_src,
            crop_x=cx,
            crop_y=cy,
            crop_zoom=cz,
            roi=snap["detect_roi"],
        )
        live_preview = decorate_preview(
            live,
            snap["detect_boxes"],
            snap["selected_detect"],
            crop_src,
            crop_x=cx,
            crop_y=cy,
            crop_zoom=cz,
            label=None if has_label_layer else snap["label_spec"],
            show_label=not has_label_layer,
            roi=snap["detect_roi"],
        )
        if orig_disp.size != orig.size:
            orig_disp = orig_disp.resize(orig.size, Image.Resampling.NEAREST)
        if live_preview.size != orig.size:
            live_preview = live_preview.resize(orig.size, Image.Resampling.NEAREST)
        return orig_disp, live_preview, live

    def _apply_preview_pils(
        self,
        orig_disp: Image.Image,
        live_preview: Image.Image,
        live: Image.Image,
    ) -> None:
        """Push computed preview frames onto the Tk widgets (Tk thread)."""
        self._work_live = live
        self._orig_pil = orig_disp
        self._tex_pil = live_preview
        self._apply_preview_zoom()
        self._refresh_tool_tab()

    def _refresh_previews(self) -> None:
        snap = self._preview_snapshot()
        if snap is None:
            return
        orig_disp, live_preview, live = self._preview_pils(snap)
        self._apply_preview_pils(orig_disp, live_preview, live)
