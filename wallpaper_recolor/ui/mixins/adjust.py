# -*- coding: utf-8 -*-
"""
wallpaper_recolor.ui.mixins.adjust
------------------------------
Color & lighting, ToneKnob wiring, texture, Position & Zoom, scale, tessellate.

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
    apply_normalize_lighting,
    apply_tessellate,
    clamp_lloyd,
    clamp_tiles,
    coerce_built,
    coerce_normalize_lighting,
    edges_already_match,
    estimate_normalize_tone,
    image_already_periodic,
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


class AppAdjustMixin:
    """Color & lighting, ToneKnob wiring, texture, Position & Zoom, scale, tessellate."""

    # ---------------------------------------------------------------------------
    # Output scale (px / in, DPI) — independent of preview Fit
    # ---------------------------------------------------------------------------
    def _scale_dpi(self) -> float:
        """Selected print DPI from the dropdown (or Custom… entry)."""
        return parse_dpi_choice(self.scale_dpi_choice.get(), self.scale_dpi_custom.get())

    def _source_wh(self) -> tuple[int, int] | None:
        """Full-resolution source pixels — never the view-zoomed preview bitmap."""
        if self.source_image is not None:
            return self.source_image.size
        return None

    def _sync_dpi_custom_row(self) -> None:
        """Custom… reveals a small DPI entry; presets hide it."""
        if self.scale_dpi_choice.get() == DPI_CUSTOM_LABEL:
            self.scale_dpi_custom_entry.pack(side="left")
        else:
            self.scale_dpi_custom_entry.pack_forget()

    def _on_scale_dpi_choice(self) -> None:
        """Preset DPI, or Custom… (prefill from the last preset so they can tweak)."""
        choice = self.scale_dpi_choice.get()
        if choice == DPI_CUSTOM_LABEL and self._scale_dpi_prev != DPI_CUSTOM_LABEL:
            prev = parse_dpi_choice(self._scale_dpi_prev, "")
            self.scale_dpi_custom.set(
                str(int(prev)) if abs(prev - int(prev)) < 1e-6 else f"{prev:g}"
            )
        self._scale_dpi_prev = choice
        self._sync_dpi_custom_row()
        if self.scale_lock.get():
            if parse_dim(self.scale_width.get()) is not None:
                self._fill_locked_other(from_width=True)
            elif parse_dim(self.scale_height.get()) is not None:
                self._fill_locked_other(from_width=False)
        self._refresh_scale_labels()

    def _on_scale_dpi_custom(self) -> None:
        if self._scale_updating or self.scale_dpi_choice.get() != DPI_CUSTOM_LABEL:
            return
        if self.scale_lock.get():
            if parse_dim(self.scale_width.get()) is not None:
                self._fill_locked_other(from_width=True)
            elif parse_dim(self.scale_height.get()) is not None:
                self._fill_locked_other(from_width=False)
        self._refresh_scale_labels()

    def _on_scale_unit(self) -> None:
        """Convert W×H into the new unit so 10 in becomes 25.4 cm, not a raw 10."""
        new = self.scale_unit.get()
        old = self._scale_unit_prev
        if new != old:
            dpi = self._scale_dpi()
            self._scale_updating = True
            try:
                w = parse_dim(self.scale_width.get())
                h = parse_dim(self.scale_height.get())
                if w is not None:
                    self.scale_width.set(format_dim(from_pixels(to_pixels(w, old, dpi), new, dpi), new))
                if h is not None:
                    self.scale_height.set(format_dim(from_pixels(to_pixels(h, old, dpi), new, dpi), new))
            finally:
                self._scale_updating = False
            self._scale_unit_prev = new
        self._refresh_scale_labels()

    def _on_scale_lock(self) -> None:
        if self.scale_lock.get():
            if parse_dim(self.scale_width.get()) is not None:
                self._fill_locked_other(from_width=True)
            elif parse_dim(self.scale_height.get()) is not None:
                self._fill_locked_other(from_width=False)
        self._refresh_scale_labels()

    def _on_scale_width(self) -> None:
        if self._scale_updating:
            return
        if self.scale_lock.get():
            self._fill_locked_other(from_width=True)
        self._refresh_scale_labels()

    def _on_scale_height(self) -> None:
        if self._scale_updating:
            return
        if self.scale_lock.get():
            self._fill_locked_other(from_width=False)
        self._refresh_scale_labels()

    def _fill_locked_other(self, *, from_width: bool) -> None:
        """Keep the other side in source aspect when Lock aspect is on."""
        src = self._source_wh()
        if src is None:
            return
        src_w, src_h = src
        unit = self.scale_unit.get()
        dpi = self._scale_dpi()
        aspect = src_w / float(src_h)
        self._scale_updating = True
        try:
            if from_width:
                w = parse_dim(self.scale_width.get())
                if w is None:
                    return
                pw = to_pixels(w, unit, dpi)
                ph = max(1, int(round(pw / aspect)))
                self.scale_height.set(format_dim(from_pixels(ph, unit, dpi), unit))
            else:
                h = parse_dim(self.scale_height.get())
                if h is None:
                    return
                ph = to_pixels(h, unit, dpi)
                pw = max(1, int(round(ph * aspect)))
                self.scale_width.set(format_dim(from_pixels(pw, unit, dpi), unit))
        finally:
            self._scale_updating = False

    def _output_scale_args(self) -> tuple[tuple[int, int] | None, str, float]:
        """Snapshot size / filter / DPI for the save worker (no Tk on that thread).

        View zoom is display-only — Save / Export must not read it or the
        on-screen PhotoImage size. Empty Scale W/H → ``None`` → original
        source pixels after crop/tessellate.
        """
        src = self._source_wh()
        src_w, src_h = src if src is not None else (1, 1)
        unit = self.scale_unit.get()
        dpi = self._scale_dpi()
        size = resolve_output_size(
            src_w,
            src_h,
            parse_dim(self.scale_width.get()),
            parse_dim(self.scale_height.get()),
            unit,
            dpi,
            bool(self.scale_lock.get()),
        )
        return size, self.scale_resample.get(), dpi

    def _refresh_scale_labels(self) -> None:
        """Source size, pixel equivalent, and 'Save at W×H px' note."""
        src = self._source_wh()
        dpi = self._scale_dpi()
        if src is None:
            self.scale_source_note.set("Source: —")
            self.scale_equiv_note.set("")
            self.scale_save_note.set("Save at original size")
            return
        src_w, src_h = src
        self.scale_source_note.set(f"Source: {src_w} × {src_h} px")
        size, resample, _dpi = self._output_scale_args()
        out_w, out_h = size if size is not None else (src_w, src_h)
        unit = self.scale_unit.get()
        if is_physical_unit(unit):
            self.scale_equiv_note.set(f"= {out_w} × {out_h} px at {dpi:g} DPI")
        else:
            inches_w = out_w / dpi
            inches_h = out_h / dpi
            self.scale_equiv_note.set(
                f"Print size {format_dim(inches_w, UNIT_INCHES)} × {format_dim(inches_h, UNIT_INCHES)} in at {dpi:g} DPI"
            )
        if size is None:
            self.scale_save_note.set(f"Save at original size ({src_w}×{src_h} px, {dpi:g} DPI)")
        else:
            filt = resample.split(" (")[0]
            self.scale_save_note.set(f"Save at {out_w}×{out_h} px · {dpi:g} DPI · {filt}")

    def _texture_strength(self) -> float:
        """Texture mix 0–1 from the slider (0 = solids, 1 = original luminosity)."""
        try:
            pct = float(self.texture_pct.get())
        except (tk.TclError, ValueError):
            pct = TEXTURE_DEFAULT_STRENGTH * 100.0
        return max(0.0, min(1.0, pct / 100.0))

    def _sync_texture_to_map(self) -> None:
        """Keep the map's texture mix and eye in lockstep with the slider (save matches preview)."""
        if self.range_map is None:
            return
        self.range_map.texture_strength = self._texture_strength()
        self.range_map.texture_enabled = bool(self.texture_enabled.get())

    # ---------------------------------------------------------------------------
    # Texture slider / eye — must not reset tone sliders
    # ---------------------------------------------------------------------------
    def _on_texture_slider(self, _value: str) -> None:
        """Texture / grain strength — live preview via the same debounce as the wheel."""
        if self._mute_ui:
            return
        strength = self._texture_strength()
        self.texture_label.set(f"Texture: {strength * 100.0:.0f}%")
        self._sync_texture_to_map()
        self._sync_slider_resets()
        self._schedule_preview()

    def _on_texture_eye(self, shown: bool) -> None:
        """FA slash-eye = flat fills; solid eye = slider mix. One history tick."""
        if self._mute_ui:
            return
        self.texture_enabled.set(bool(shown))
        if self.range_map is None:
            return
        before = self._capture_edit()
        self._sync_texture_to_map()
        self._push_undo_state(before)
        self._schedule_preview()

    def _crop_src_size(self) -> tuple[int, int]:
        """Full-res size for X/Y; work-image size if nothing is open yet."""
        if self.source_image is not None:
            return self.source_image.size
        if self.work_image is not None:
            return self.work_image.size
        return (1, 1)

    def _crop_xy_zoom(self) -> tuple[float, float, float]:
        """Current center-offset X/Y (source px) and zoom. Offsets are not clamped."""
        try:
            x = float(self.crop_x.get())
        except (tk.TclError, ValueError):
            x = float(CROP_XY_DEFAULT)
        try:
            y = float(self.crop_y.get())
        except (tk.TclError, ValueError):
            y = float(CROP_XY_DEFAULT)
        try:
            z = float(self.crop_zoom.get())
        except (tk.TclError, ValueError):
            z = float(ZOOM_DEFAULT)
        z = clamp_zoom(z)
        return float(int(round(x))), float(int(round(y))), z

    def _apply_view_crop(self, image: Image.Image) -> Image.Image:
        """Map source-pixel X/Y + zoom onto this (possibly downscaled) image."""
        x, y, z = self._crop_xy_zoom()
        return apply_crop(image, x, y, z, src_size=self._crop_src_size())

    def _sync_crop_entry_text(self) -> None:
        x, y, z = self._crop_xy_zoom()
        prev = self._crop_updating
        self._crop_updating = True
        try:
            self.crop_x_text.set(str(int(round(x))))
            self.crop_y_text.set(str(int(round(y))))
            self.crop_zoom_text.set(_format_zoom_text(z))
        finally:
            self._crop_updating = prev

    def _sync_crop_bounds(self, *, clamp: bool = True) -> None:
        """Slider range follows zoom so X/Y can push the image off the frame."""
        if not hasattr(self, "crop_x_scale") or not hasattr(self, "crop_y_scale"):
            return
        sw, sh = self._crop_src_size()
        try:
            z = clamp_zoom(float(self.crop_zoom.get()))
        except (tk.TclError, ValueError):
            z = float(ZOOM_DEFAULT)
        lim_x, lim_y = offset_slider_limit(sw, sh, z)
        prev = self._crop_updating
        self._crop_updating = True
        try:
            self.crop_x_scale.configure(from_=float(-lim_x), to=float(lim_x))
            self.crop_y_scale.configure(from_=float(-lim_y), to=float(lim_y))
            if clamp:
                self.crop_zoom.set(float(z))
        finally:
            self._crop_updating = prev

    def _on_crop_xy_slider(self, _value: str = "") -> None:
        if self._mute_ui or self._crop_updating:
            return
        self._sync_crop_bounds(clamp=True)
        self._sync_crop_entry_text()
        self._sync_slider_resets()
        self._schedule_preview()

    def _on_crop_zoom_slider(self, _value: str = "") -> None:
        if self._mute_ui or self._crop_updating:
            return
        self._sync_crop_bounds(clamp=True)
        self._sync_crop_entry_text()
        self._sync_slider_resets()
        self._schedule_preview()

    def _commit_crop_entry(self, which: str, _event=None) -> None:
        """Typed X, Y, or zoom — one undo tick, live preview. X/Y are not clamped."""
        if self._mute_ui or self._crop_updating:
            return
        text_var = {
            "x": self.crop_x_text,
            "y": self.crop_y_text,
            "zoom": self.crop_zoom_text,
        }[which]
        try:
            val = float(text_var.get().strip())
        except (TypeError, ValueError):
            self._sync_crop_entry_text()
            return
        before = self._capture_edit()
        prev = self._crop_updating
        self._crop_updating = True
        try:
            if which == "x":
                self.crop_x.set(val)
            elif which == "y":
                self.crop_y.set(val)
            else:
                self.crop_zoom.set(clamp_zoom(val))
        finally:
            self._crop_updating = prev
        self._sync_crop_bounds(clamp=True)
        self._sync_crop_entry_text()
        self._sync_slider_resets()
        self._push_undo_state(before)
        self._schedule_preview()

    def _reset_crop_x(self) -> None:
        self._reset_crop_var("x")

    def _reset_crop_y(self) -> None:
        self._reset_crop_var("y")

    def _reset_crop_zoom(self) -> None:
        self._reset_crop_var("zoom")

    def _reset_crop_var(self, which: str) -> None:
        x, y, z = self._crop_xy_zoom()
        if which == "x" and abs(x - CROP_XY_DEFAULT) < _RESET_EPS:
            return
        if which == "y" and abs(y - CROP_XY_DEFAULT) < _RESET_EPS:
            return
        if which == "zoom" and abs(z - ZOOM_DEFAULT) < _ZOOM_RESET_EPS:
            return
        before = self._capture_edit()
        prev = self._crop_updating
        self._crop_updating = True
        try:
            if which == "x":
                self.crop_x.set(float(CROP_XY_DEFAULT))
            elif which == "y":
                self.crop_y.set(float(CROP_XY_DEFAULT))
            else:
                self.crop_zoom.set(float(ZOOM_DEFAULT))
        finally:
            self._crop_updating = prev
        self._sync_crop_bounds(clamp=True)
        self._sync_crop_entry_text()
        self._sync_slider_resets()
        self._push_undo_state(before)
        self._schedule_preview()

    def _tess_tiles_value(self) -> int:
        try:
            return clamp_tiles(self.tess_tiles.get())
        except (tk.TclError, ValueError, TypeError):
            return TILES_DEFAULT

    def _tess_lloyd_value(self) -> int:
        try:
            return clamp_lloyd(self.tess_lloyd.get())
        except (tk.TclError, ValueError, TypeError):
            return LLOYD_DEFAULT

    def _tess_params(self) -> tuple[str, str, bool, str]:
        """Normalized sides + whether Build has run + mode."""
        return (
            normalize_h_side(self.tess_h.get()),
            normalize_v_side(self.tess_v.get()),
            bool(self.tess_built.get()),
            normalize_tess_mode(self.tess_mode.get()),
        )

    def _tess_normalize_on(self) -> bool:
        try:
            return coerce_normalize_lighting(self.tess_normalize.get())
        except (tk.TclError, ValueError, TypeError):
            return False

    def _apply_view_tone(self, image: Image.Image) -> Image.Image:
        """Same Color & lighting numbers as Result, on this RGB frame."""
        kwargs = self._tone_apply_kwargs()
        if is_neutral_tone(**kwargs):
            return image
        rgb = np.asarray(image.convert("RGB"))
        out = apply_tone_rgb(rgb, **kwargs)
        result = Image.fromarray(out, mode="RGB")
        if image.mode != "RGB":
            return result.convert(image.mode)
        return result

    def _apply_view_tessellate(self, image: Image.Image) -> Image.Image:
        """Same tessellate as Save, on this (possibly downscaled / cropped) image."""
        h_side, v_side, built, mode = self._tess_params()
        return apply_tessellate(
            image,
            h_side,
            v_side,
            built,
            mode=mode,
            tiles=self._tess_tiles_value(),
            lloyd=self._tess_lloyd_value(),
        )

    def _apply_view_transform(self, image: Image.Image, *, include_tone: bool = False) -> Image.Image:
        """Crop → optional Tone + lighting flatten → tessellate.

        Flatten rides with Tone so Original (include_tone=False) stays the
        source crop. Result preview uses ``_preview_pils`` for the same steps.
        """
        x, y, z = self._crop_xy_zoom()
        h_side, v_side, built, mode = self._tess_params()
        image = apply_crop(
            image, x, y, z, src_size=self._crop_src_size()
        )
        if include_tone:
            image = self._apply_view_tone(image)
            if self._tess_normalize_on():
                image = apply_normalize_lighting(image)
        return apply_tessellate(
            image,
            h_side,
            v_side,
            built,
            mode=mode,
            tiles=self._tess_tiles_value(),
            lloyd=self._tess_lloyd_value(),
        )

    def _set_crop_xy_zoom(self, x: float, y: float, zoom: float) -> None:
        """Write Crop sliders + entries without scheduling a preview."""
        prev = self._crop_updating
        self._crop_updating = True
        try:
            self.crop_zoom.set(float(clamp_zoom(zoom)))
            self.crop_x.set(float(int(round(x))))
            self.crop_y.set(float(int(round(y))))
        finally:
            self._crop_updating = prev
        self._sync_crop_bounds(clamp=False)
        self._sync_crop_entry_text()

    def _plan_build_crop(self, h_side: str, v_side: str) -> tuple[float, float, float] | None:
        """Center-offset X/Y + zoom Tessellate Build should write, or None if identity."""
        if self.work_image is None:
            return None
        if normalize_tess_mode(self.tess_mode.get()) == MODE_MESH:
            return None
        rgb = self.work_image.convert("RGB")
        if self._tess_normalize_on():
            rgb = apply_normalize_lighting(rgb)
        arr = np.asarray(rgb)
        mode = self._tess_params()[3]
        if normalize_tess_mode(mode) == MODE_TILE:
            if image_already_periodic(arr, h_side, v_side):
                return None
        elif edges_already_match(arr, h_side, v_side):
            return None
        lx, ly, lz = plan_tessellate_crop(arr, h_side, v_side, mode=mode)
        if lx == 0 and ly == 0 and abs(float(lz) - 1.0) < 1e-6:
            return None
        wh, ww = int(arr.shape[0]), int(arr.shape[1])
        ox, oy = top_left_to_center_offset(ww, wh, lx, ly, lz)
        sw, sh = self._crop_src_size()
        return (
            float(int(round(ox * sw / float(max(ww, 1))))),
            float(int(round(oy * sh / float(max(wh, 1))))),
            float(clamp_zoom(lz)),
        )

    def _on_tess_side(self) -> None:
        """Left/Right and Top/Bottom radios — one undo tick per click.

        ttk.Radiobutton already wrote the StringVar before ``command`` runs, so
        we snap back to the last committed sides long enough to capture 'before'.
        Changing sides clears a previous Build (identity until they Build again).
        """
        if self._mute_ui or self._tess_updating:
            return
        new_h = normalize_h_side(self.tess_h.get())
        new_v = normalize_v_side(self.tess_v.get())
        old_h, old_v, old_built, old_mode = self._tess_committed
        if new_h == old_h and new_v == old_v:
            return
        self._tess_updating = True
        try:
            self.tess_h.set(old_h)
            self.tess_v.set(old_v)
            before = self._capture_edit()
            self.tess_h.set(new_h)
            self.tess_v.set(new_v)
            self.tess_built.set(False)
        finally:
            self._tess_updating = False
        self._tess_committed = (new_h, new_v, False, old_mode)
        self._sync_slider_resets()
        self._push_undo_state(before)
        self._schedule_preview()

    def _on_tess_mode_combo(self) -> None:
        """Dropdown label → stored mode key, then the usual mode undo tick."""
        self.tess_mode.set(normalize_tess_mode(self.tess_mode_label.get()))
        self._on_tess_mode()

    def _sync_tess_mode_combo(self) -> None:
        """Keep the Mode dropdown label aligned with the stored mode key."""
        if not hasattr(self, "tess_mode_label"):
            return
        label = tess_mode_label(self.tess_mode.get())
        if self.tess_mode_label.get() == label:
            return
        prev = self._tess_updating
        self._tess_updating = True
        try:
            self.tess_mode_label.set(label)
        finally:
            self._tess_updating = prev

    def _on_tess_mode(self) -> None:
        """Tile / Tessellation / Mesh / mosaic — one undo tick per change."""
        if self._mute_ui or self._tess_updating:
            return
        new_mode = normalize_tess_mode(self.tess_mode.get())
        old_h, old_v, old_built, old_mode = self._tess_committed
        if new_mode == old_mode:
            self._sync_tess_mode_combo()
            return
        self._tess_updating = True
        try:
            self.tess_mode.set(old_mode)
            before = self._capture_edit()
            self.tess_mode.set(new_mode)
            self.tess_built.set(False)
        finally:
            self._tess_updating = False
        self._sync_tess_mode_combo()
        self._tess_committed = (old_h, old_v, False, new_mode)
        self._sync_tess_mosaic_controls()
        self._sync_slider_resets()
        self._push_undo_state(before)
        self._schedule_preview()

    def _on_tess_build(self) -> None:
        """Search wrap crop, write Crop panel, diffuse — one undo tick."""
        if self._mute_ui or self._tess_updating or self._busy:
            return
        new_h = normalize_h_side(self.tess_h.get())
        new_v = normalize_v_side(self.tess_v.get())
        new_mode = normalize_tess_mode(self.tess_mode.get())
        # Tile keeps Off as Off — forcing Left/Top wrapped leftover and smeared
        # the left column even when Horizontal/Vertical were Off.
        if (
            new_mode != MODE_TILE
            and new_h == SIDE_OFF
            and new_v == SIDE_OFF
        ):
            new_h, new_v = SIDE_LEFT, SIDE_TOP
        planned = None
        if new_mode != MODE_MESH:
            planned = self._plan_build_crop(new_h, new_v)
        if (
            bool(self.tess_built.get())
            and (new_h, new_v, True, new_mode) == self._tess_committed
            and planned is None
        ):
            return
        before = self._capture_edit()
        self._tess_updating = True
        try:
            self.tess_h.set(new_h)
            self.tess_v.set(new_v)
            self.tess_built.set(True)
        finally:
            self._tess_updating = False
        if planned is not None:
            self._set_crop_xy_zoom(*planned)
        self._tess_committed = (new_h, new_v, True, new_mode)
        self._sync_slider_resets()
        self._push_undo_state(before)
        self._cancel_preview_job()
        prev_status = self.status.get()
        snap = self._preview_snapshot()

        def work():
            if snap is None:
                return None
            return self._preview_pils(snap)

        def on_ok(computed) -> None:
            if computed is not None:
                orig_disp, live_preview, live = computed
                self._apply_preview_pils(orig_disp, live_preview, live)
            col = self.tess_panel.column
            if col is not None:
                col._sync_layout()
            self.status.set(prev_status)

        def on_err(exc: BaseException) -> None:
            messagebox.showerror("Could not tessellate", str(exc), parent=self.root)
            self.status.set("Build failed")

        self._run_background("Building…", work, on_ok, on_err)

    def _set_tone_sliders(self, darks: float, lights: float) -> None:
        """Write Darks / Lights (−1…+1) onto steppers and the range map.

        White balance, Exposure, Contrast, Highlight RGB, CMY pairs, and
        Saturation stay as they are.
        """
        prev = self._mute_ui
        self._mute_ui = True
        try:
            self.darks_pct.set(float(darks) * 100.0)
            self.lights_pct.set(float(lights) * 100.0)
        finally:
            self._mute_ui = prev
        self._sync_tone_labels()
        self._sync_tone_to_map()

    def _apply_normalize_to_tone(self) -> None:
        """Run flatten off-screen, map the luma change into Darks / Lights.

        Estimates from the cropped, un-toned work image so a second click
        overwrites rather than stacking on already-graded sliders.
        White balance / Exposure / Contrast / Highlight RGB / CMY / Saturation stay as they are.
        """
        img = self.work_image
        if img is None:
            self._lighting_auto_darks = 0.0
            self._lighting_auto_lights = 0.0
            return
        img = self._apply_view_crop(img)
        darks, lights = estimate_normalize_tone(img)
        self._lighting_auto_darks = float(darks)
        self._lighting_auto_lights = float(lights)
        self._set_tone_sliders(darks, lights)

    def _on_tess_normalize(self) -> None:
        """Turn on spatial lighting flatten (bowl / ramp). Darks/Lights stay independent."""
        if self._mute_ui or self._tess_updating:
            return
        before = self._capture_edit()
        self._tess_updating = True
        try:
            self.tess_normalize.set(True)
        finally:
            self._tess_updating = False
        self._sync_slider_resets()
        self._push_undo_state(before)
        self._refresh_now()

    def _tone_zero_attrs(self) -> tuple[str, ...]:
        return (
            "tone_darks",
            "tone_lights",
            "tone_brightness",
            "tone_contrast",
            "tone_exposure",
            "tone_lights_reds",
            "tone_lights_greens",
            "tone_lights_blues",
            "tone_temperature",
            "tone_tint",
            "tone_saturation",
            "tone_balance_cyan",
            "tone_balance_magenta",
            "tone_balance_yellow",
            "tone_lights_cyan",
            "tone_lights_magenta",
            "tone_lights_yellow",
            "tone_darks_cyan",
            "tone_darks_magenta",
            "tone_darks_yellow",
        )

    def _remap_rgb_without_tone(self) -> np.ndarray | None:
        """Recolored RGB before Color & lighting, for Gray World / White Patch."""
        rm = self.range_map
        if rm is not None and rm.rgb is not None and rm.labels is not None:
            saved = {name: float(getattr(rm, name, 0.0)) for name in self._tone_zero_attrs()}
            try:
                for name in saved:
                    setattr(rm, name, 0.0)
                img = live_composite_from_map(rm)
            finally:
                for name, value in saved.items():
                    setattr(rm, name, value)
            return np.asarray(img.convert("RGB"))
        if self.work_image is None:
            return None
        return np.asarray(self._apply_view_crop(self.work_image).convert("RGB"))

    def _set_temp_tint_from_estimate(self, temp: float, tint: float) -> None:
        prev = self._mute_ui
        self._mute_ui = True
        try:
            self.temperature_pct.set(float(round(float(temp) * 100.0)))
            self.tint_pct.set(float(round(float(tint) * 100.0)))
        finally:
            self._mute_ui = prev
        self._on_tone_slider("")

    def _on_gray_world(self) -> None:
        """Gray World estimate → Temperature / Tint (still editable)."""
        if self._mute_ui:
            return
        rgb = self._remap_rgb_without_tone()
        if rgb is None:
            return
        before = self._capture_edit()
        temp, tint = estimate_gray_world_temp_tint(rgb)
        self._set_temp_tint_from_estimate(temp, tint)
        self._push_undo_state(before)

    def _on_white_patch(self) -> None:
        """White Patch / max-RGB in highlights → Temperature / Tint."""
        if self._mute_ui:
            return
        rgb = self._remap_rgb_without_tone()
        if rgb is None:
            return
        before = self._capture_edit()
        temp, tint = estimate_white_patch_temp_tint(rgb)
        self._set_temp_tint_from_estimate(temp, tint)
        self._push_undo_state(before)

    def _normalize_lighting_dirty(self) -> bool:
        """Reset when Normalize lighting is on."""
        return self._tess_normalize_on()

    def _reset_tess_normalize(self) -> None:
        """Clear the spatial flatten flag. Darks/Lights keep their own reset."""
        if not self._normalize_lighting_dirty():
            return
        before = self._capture_edit()
        self._tess_updating = True
        try:
            self.tess_normalize.set(False)
        finally:
            self._tess_updating = False
        self._lighting_auto_darks = 0.0
        self._lighting_auto_lights = 0.0
        self._sync_slider_resets()
        self._push_undo_state(before)
        self._schedule_preview()

    def _reset_tess_mode(self) -> None:
        if normalize_tess_mode(self.tess_mode.get()) == MODE_DEFAULT:
            return
        before = self._capture_edit()
        self._tess_updating = True
        try:
            self.tess_mode.set(MODE_DEFAULT)
            self.tess_built.set(False)
        finally:
            self._tess_updating = False
        self._sync_tess_mode_combo()
        self._tess_committed = self._tess_params()
        self._sync_tess_mosaic_controls()
        self._sync_slider_resets()
        self._push_undo_state(before)
        self._schedule_preview()

    def _reset_tess_h(self) -> None:
        self._reset_tess_side("h")

    def _reset_tess_v(self) -> None:
        self._reset_tess_side("v")

    def _reset_tess_side(self, which: str) -> None:
        current = normalize_h_side(self.tess_h.get()) if which == "h" else normalize_v_side(
            self.tess_v.get()
        )
        if current == SIDE_OFF:
            return
        before = self._capture_edit()
        self._tess_updating = True
        try:
            if which == "h":
                self.tess_h.set(SIDE_OFF)
            else:
                self.tess_v.set(SIDE_OFF)
            self.tess_built.set(False)
        finally:
            self._tess_updating = False
        self._tess_committed = self._tess_params()
        self._sync_slider_resets()
        self._push_undo_state(before)
        self._schedule_preview()

    def _reset_tess_build(self) -> None:
        """Undo Build — identity until they click Build again."""
        if not bool(self.tess_built.get()):
            return
        before = self._capture_edit()
        self._tess_updating = True
        try:
            self.tess_built.set(False)
        finally:
            self._tess_updating = False
        self._tess_committed = self._tess_params()
        self._sync_slider_resets()
        self._push_undo_state(before)
        self._schedule_preview()

    def _sync_tess_mosaic_controls(self) -> None:
        """Tiles / Lloyd only apply to Detail mosaic."""
        host = getattr(self, "tess_mosaic_host", None)
        if host is None:
            return
        if normalize_tess_mode(self.tess_mode.get()) == MODE_VORONOI:
            if str(host.winfo_manager()) != "pack":
                host.pack(fill="x", before=self.tess_build_btn.master)
        else:
            host.pack_forget()
        col = getattr(self.tess_panel, "column", None)
        if col is not None:
            col._sync_layout()

    def _sync_tess_mosaic_text(self) -> None:
        prev = self._tess_updating
        self._tess_updating = True
        try:
            self.tess_tiles_text.set(str(self._tess_tiles_value()))
            self.tess_lloyd_text.set(str(self._tess_lloyd_value()))
        except tk.TclError:
            pass
        finally:
            self._tess_updating = prev

    def _on_tess_tiles_slider(self, _value: str = "") -> None:
        if self._mute_ui or self._tess_updating:
            return
        self._sync_tess_mosaic_text()
        self._sync_slider_resets()
        if bool(self.tess_built.get()):
            self._schedule_preview()

    def _on_tess_lloyd_slider(self, _value: str = "") -> None:
        if self._mute_ui or self._tess_updating:
            return
        self._sync_tess_mosaic_text()
        self._sync_slider_resets()
        if bool(self.tess_built.get()):
            self._schedule_preview()

    def _commit_tess_tiles(self, _event=None) -> None:
        if self._mute_ui or self._tess_updating:
            return
        try:
            val = clamp_tiles(self.tess_tiles_text.get().strip())
        except (TypeError, ValueError):
            self._sync_tess_mosaic_text()
            return
        before = self._capture_edit()
        self.tess_tiles.set(float(val))
        self._sync_tess_mosaic_text()
        self._sync_slider_resets()
        self._push_undo_state(before)
        if bool(self.tess_built.get()):
            self._schedule_preview()

    def _commit_tess_lloyd(self, _event=None) -> None:
        if self._mute_ui or self._tess_updating:
            return
        try:
            val = clamp_lloyd(self.tess_lloyd_text.get().strip())
        except (TypeError, ValueError):
            self._sync_tess_mosaic_text()
            return
        before = self._capture_edit()
        self.tess_lloyd.set(float(val))
        self._sync_tess_mosaic_text()
        self._sync_slider_resets()
        self._push_undo_state(before)
        if bool(self.tess_built.get()):
            self._schedule_preview()

    def _reset_tess_tiles(self) -> None:
        if abs(self._tess_tiles_value() - TILES_DEFAULT) < 1:
            return
        before = self._capture_edit()
        self.tess_tiles.set(float(TILES_DEFAULT))
        self._sync_tess_mosaic_text()
        self._sync_slider_resets()
        self._push_undo_state(before)
        if bool(self.tess_built.get()):
            self._schedule_preview()

    def _reset_tess_lloyd(self) -> None:
        if self._tess_lloyd_value() == LLOYD_DEFAULT:
            return
        before = self._capture_edit()
        self.tess_lloyd.set(float(LLOYD_DEFAULT))
        self._sync_tess_mosaic_text()
        self._sync_slider_resets()
        self._push_undo_state(before)
        if bool(self.tess_built.get()):
            self._schedule_preview()

    def _tone_amounts(self) -> tuple[float, float, float]:
        return (
            slider_to_amount(self.darks_pct.get()),
            slider_to_amount(self.lights_pct.get()),
            slider_to_amount(self.brightness_pct.get()),
        )

    def _lights_rgb_amounts(self) -> tuple[float, float, float]:
        return (
            slider_to_amount(self.lights_reds_pct.get()),
            slider_to_amount(self.lights_greens_pct.get()),
            slider_to_amount(self.lights_blues_pct.get()),
        )

    def _balance_amounts(self) -> tuple[float, float, float]:
        return (
            slider_to_amount(self.balance_cyan_pct.get()),
            slider_to_amount(self.balance_magenta_pct.get()),
            slider_to_amount(self.balance_yellow_pct.get()),
        )

    def _contrast_exposure_amounts(self) -> tuple[float, float]:
        return (
            slider_to_amount(self.contrast_pct.get()),
            slider_to_amount(self.exposure_pct.get()),
        )

    def _tone_apply_kwargs(self) -> dict:
        """Named amounts for apply_tone_rgb / is_neutral_tone."""
        d, li, b = self._tone_amounts()
        lr, lg, lb = self._lights_rgb_amounts()
        c, e = self._contrast_exposure_amounts()
        bc, bm, by = self._balance_amounts()
        return {
            "darks": d,
            "lights": li,
            "brightness": b,
            "lights_reds": lr,
            "lights_greens": lg,
            "lights_blues": lb,
            "contrast": c,
            "exposure": e,
            "temperature": slider_to_amount(self.temperature_pct.get()),
            "tint": slider_to_amount(self.tint_pct.get()),
            "saturation": slider_to_amount(self.saturation_pct.get()),
            "balance_cyan": bc,
            "balance_magenta": bm,
            "balance_yellow": by,
        }

    def _lights_rgb_var(self, which: str) -> tk.DoubleVar:
        key = which.lower()
        if key == "reds":
            return self.lights_reds_pct
        if key == "greens":
            return self.lights_greens_pct
        if key == "blues":
            return self.lights_blues_pct
        raise KeyError(which)

    def _balance_var(self, which: str) -> tk.DoubleVar:
        key = which.lower()
        if key == "cyan":
            return self.balance_cyan_pct
        if key == "magenta":
            return self.balance_magenta_pct
        if key == "yellow":
            return self.balance_yellow_pct
        raise KeyError(which)

    def _sync_tone_labels(self) -> None:
        """Spinboxes already show the numeric value."""
        return

    def _sync_tone_to_map(self) -> None:
        if self.range_map is None:
            return
        kwargs = self._tone_apply_kwargs()
        self.range_map.tone_darks = kwargs["darks"]
        self.range_map.tone_lights = kwargs["lights"]
        self.range_map.tone_brightness = kwargs["brightness"]
        self.range_map.tone_contrast = kwargs["contrast"]
        self.range_map.tone_exposure = kwargs["exposure"]
        self.range_map.tone_lights_reds = kwargs["lights_reds"]
        self.range_map.tone_lights_greens = kwargs["lights_greens"]
        self.range_map.tone_lights_blues = kwargs["lights_blues"]
        self.range_map.tone_temperature = kwargs["temperature"]
        self.range_map.tone_tint = kwargs["tint"]
        self.range_map.tone_saturation = kwargs["saturation"]
        self.range_map.tone_balance_cyan = kwargs["balance_cyan"]
        self.range_map.tone_balance_magenta = kwargs["balance_magenta"]
        self.range_map.tone_balance_yellow = kwargs["balance_yellow"]
        # Legacy keys stay in lockstep so old wpedit / job-pack readers still see CMY.
        self.range_map.tone_lights_cyan = kwargs["balance_cyan"]
        self.range_map.tone_lights_magenta = kwargs["balance_magenta"]
        self.range_map.tone_lights_yellow = kwargs["balance_yellow"]

    def _sync_normalize_from_tone_sliders(self) -> None:
        """Darks/Lights are a separate grade from spatial Normalize lighting."""
        return

    def _on_tone_slider(self, _value: str) -> None:
        """Darks / lights / brightness / WB / CMY — same debounce as the wheel; 0 is identity."""
        if self._mute_ui:
            return
        self._sync_normalize_from_tone_sliders()
        self._sync_tone_labels()
        self._sync_tone_to_map()
        self._sync_slider_resets()
        self._schedule_preview()

    def _make_slider_reset(
        self, parent: tk.Misc, command, *, tip: str = "Reset"
    ) -> tk.Label:
        """Flat FA rotate-left icon; packed later when the slider leaves its default."""
        bg = ttk.Style().lookup("TFrame", "background") or "#f0f0f0"
        btn = tk.Label(
            parent,
            image=self._reset_photo,
            bd=0,
            highlightthickness=0,
            cursor="hand2",
            bg=bg,
            takefocus=0,
        )
        btn.image = self._reset_photo  # type: ignore[attr-defined]
        btn.bind("<Button-1>", lambda _e: command())
        bind_tooltip(btn, tip)
        return btn

    def _set_reset_visible(
        self,
        button: tk.Misc,
        current: float,
        default: float,
        eps: float = _RESET_EPS,
    ) -> None:
        """Pack the rotate-left icon to the right of the label only when off default."""
        show = abs(float(current) - float(default)) >= eps
        mapped = str(button.winfo_manager()) != ""
        if show and not mapped:
            button.pack(side="left", padx=(8, 0))
        elif not show and mapped:
            button.pack_forget()

    def _sync_slider_resets(self) -> None:
        """Show or hide each slider Reset without touching unrelated controls."""
        try:
            tex = float(self.texture_pct.get())
        except (tk.TclError, ValueError):
            tex = TEXTURE_DEFAULT_STRENGTH * 100.0
        self._set_reset_visible(self.texture_reset, tex, TEXTURE_DEFAULT_STRENGTH * 100.0)
        self._set_reset_visible(self.darks_reset, float(self.darks_pct.get()), TONE_NEUTRAL)
        self._set_reset_visible(self.lights_reset, float(self.lights_pct.get()), TONE_NEUTRAL)
        self._set_reset_visible(self.brightness_reset, float(self.brightness_pct.get()), TONE_NEUTRAL)
        if hasattr(self, "contrast_reset"):
            self._set_reset_visible(
                self.contrast_reset, float(self.contrast_pct.get()), TONE_NEUTRAL
            )
            self._set_reset_visible(
                self.exposure_reset, float(self.exposure_pct.get()), TONE_NEUTRAL
            )
        if hasattr(self, "lights_reds_reset"):
            self._set_reset_visible(
                self.lights_reds_reset, float(self.lights_reds_pct.get()), TONE_NEUTRAL
            )
            self._set_reset_visible(
                self.lights_greens_reset, float(self.lights_greens_pct.get()), TONE_NEUTRAL
            )
            self._set_reset_visible(
                self.lights_blues_reset, float(self.lights_blues_pct.get()), TONE_NEUTRAL
            )
        if hasattr(self, "temperature_reset"):
            self._set_reset_visible(
                self.temperature_reset, float(self.temperature_pct.get()), TONE_NEUTRAL
            )
            self._set_reset_visible(
                self.tint_reset, float(self.tint_pct.get()), TONE_NEUTRAL
            )
            self._set_reset_visible(
                self.saturation_reset, float(self.saturation_pct.get()), TONE_NEUTRAL
            )
        if hasattr(self, "balance_cyan_reset"):
            self._set_reset_visible(
                self.balance_cyan_reset, float(self.balance_cyan_pct.get()), TONE_NEUTRAL
            )
            self._set_reset_visible(
                self.balance_magenta_reset, float(self.balance_magenta_pct.get()), TONE_NEUTRAL
            )
            self._set_reset_visible(
                self.balance_yellow_reset, float(self.balance_yellow_pct.get()), TONE_NEUTRAL
            )
        if self.range_map is None:
            self._set_reset_visible(self.cover_reset, 0.0, 0.0)
        else:
            self._set_reset_visible(self.cover_reset, 0.0 if self._colors_are_generic() else 1.0, 0.0)
        try:
            mock = float(self.mockup_repeats.get())
        except (tk.TclError, ValueError):
            mock = DEFAULT_MOCKUP_REPEATS
        self._set_reset_visible(self.mockup_reset, mock, DEFAULT_MOCKUP_REPEATS)
        if hasattr(self, "crop_zoom_reset"):
            try:
                cx, cy, cz = self._crop_xy_zoom()
            except (tk.TclError, ValueError):
                cx = cy = float(CROP_XY_DEFAULT)
                cz = float(ZOOM_DEFAULT)
            self._set_reset_visible(self.crop_x_reset, cx, float(CROP_XY_DEFAULT))
            self._set_reset_visible(self.crop_y_reset, cy, float(CROP_XY_DEFAULT))
            self._set_reset_visible(
                self.crop_zoom_reset, cz, float(ZOOM_DEFAULT), eps=_ZOOM_RESET_EPS
            )
        if hasattr(self, "preview_zoom_reset"):
            try:
                pz = float(self.preview_zoom.get())
            except (tk.TclError, ValueError, TypeError):
                pz = VIEW_ZOOM_PCT_DEFAULT
            self._set_reset_visible(
                self.preview_zoom_reset, pz, VIEW_ZOOM_PCT_DEFAULT, eps=0.5
            )
        if hasattr(self, "tess_h_reset"):
            h_side, v_side, tess_built, tess_mode = self._tess_params()
            self._set_reset_visible(
                self.tess_mode_reset,
                0.0 if tess_mode == MODE_DEFAULT else 1.0,
                0.0,
            )
            self._set_reset_visible(
                self.tess_h_reset, 0.0 if h_side == SIDE_OFF else 1.0, 0.0
            )
            self._set_reset_visible(
                self.tess_v_reset, 0.0 if v_side == SIDE_OFF else 1.0, 0.0
            )
            if hasattr(self, "tess_normalize_reset"):
                self._set_reset_visible(
                    self.tess_normalize_reset,
                    1.0 if self._normalize_lighting_dirty() else 0.0,
                    0.0,
                )
            if hasattr(self, "tess_build_reset"):
                self._set_reset_visible(
                    self.tess_build_reset, 1.0 if tess_built else 0.0, 0.0
                )
            if hasattr(self, "tess_tiles_reset"):
                self._set_reset_visible(
                    self.tess_tiles_reset, float(self._tess_tiles_value()), float(TILES_DEFAULT)
                )
                self._set_reset_visible(
                    self.tess_lloyd_reset, float(self._tess_lloyd_value()), float(LLOYD_DEFAULT)
                )
        if hasattr(self, "label_size_reset"):
            self._set_reset_visible(
                self.label_size_reset,
                float(clamp_label_size(self.label_size.get())),
                float(LABEL_SIZE_DEFAULT),
            )

    def _reset_texture(self) -> None:
        """Texture slider default = full grain (TEXTURE_DEFAULT_STRENGTH)."""
        default = round(TEXTURE_DEFAULT_STRENGTH * 100.0)
        if abs(float(self.texture_pct.get()) - default) < _RESET_EPS:
            return
        before = self._capture_edit()
        self.texture_pct.set(float(default))
        self._on_texture_slider("")
        self._push_undo_state(before)

    def _reset_tone_var(self, var: tk.DoubleVar) -> None:
        if abs(float(var.get()) - TONE_NEUTRAL) < _RESET_EPS:
            return
        before = self._capture_edit()
        var.set(float(TONE_NEUTRAL))
        self._on_tone_slider("")
        self._push_undo_state(before)

    def _reset_darks(self) -> None:
        self._reset_tone_var(self.darks_pct)

    def _reset_lights(self) -> None:
        self._reset_tone_var(self.lights_pct)

    def _reset_brightness(self) -> None:
        self._reset_tone_var(self.brightness_pct)

    def _reset_contrast(self) -> None:
        self._reset_tone_var(self.contrast_pct)

    def _reset_exposure(self) -> None:
        self._reset_tone_var(self.exposure_pct)

    def _reset_lights_rgb(self, which: str) -> None:
        self._reset_tone_var(self._lights_rgb_var(which))

    def _reset_temperature(self) -> None:
        self._reset_tone_var(self.temperature_pct)

    def _reset_tint(self) -> None:
        self._reset_tone_var(self.tint_pct)

    def _reset_saturation(self) -> None:
        self._reset_tone_var(self.saturation_pct)

    def _reset_balance(self, which: str) -> None:
        self._reset_tone_var(self._balance_var(which))

    def _commit_tone_spin(self, var: tk.DoubleVar, spin: ttk.Spinbox | None = None) -> None:
        """Typed or stepped Color & lighting value: clamp −100…+100, one undo tick."""
        if self._mute_ui or self._tone_updating:
            return
        raw = spin.get() if spin is not None else var.get()
        try:
            val = float(str(raw).strip())
        except (TypeError, ValueError, tk.TclError):
            prev = self._tone_updating
            self._tone_updating = True
            try:
                var.set(float(round(float(var.get()))))
            except (tk.TclError, ValueError, TypeError):
                var.set(float(TONE_NEUTRAL))
            finally:
                self._tone_updating = prev
            return
        val = max(TONE_SLIDER_MIN, min(TONE_SLIDER_MAX, round(val, 1)))
        current = float(var.get())
        if abs(val - current) < _RESET_EPS:
            if spin is not None and str(spin.get()).strip() != str(int(val)):
                prev = self._tone_updating
                self._tone_updating = True
                try:
                    var.set(float(val))
                finally:
                    self._tone_updating = prev
            self._mark_slider_end()
            return
        if self._slider_before is None:
            before = self._capture_edit()
        else:
            before = None
        prev = self._mute_ui
        self._mute_ui = True
        try:
            var.set(float(val))
        finally:
            self._mute_ui = prev
        self._on_tone_slider("")
        if before is not None:
            self._push_undo_state(before)
        else:
            self._mark_slider_end()

    def _reset_mockup(self) -> None:
        self.mockup_repeats.set(DEFAULT_MOCKUP_REPEATS)
        self._on_mockup_scale("")
