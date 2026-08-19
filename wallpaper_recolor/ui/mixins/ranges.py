# -*- coding: utf-8 -*-
"""
wallpaper_recolor.ui.mixins.ranges
------------------------------
Color ranges, presets, coverage %, hide-eye knockout.

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


class AppRangesMixin:
    """Color ranges, presets, coverage %, hide-eye knockout."""

    # ---------------------------------------------------------------------------
    # Presets + Range by (k-means vs luma / Lab histogram splits)
    # ---------------------------------------------------------------------------
    def _reload_preset_combo(self) -> None:
        """Refresh the Presets list; Generic is always first (no named job)."""
        presets = list_presets()
        names = [GENERIC_LABEL, *[preset.name for preset in presets]]
        current = self.preset_choice.get()
        self.preset_combo.configure(values=names)
        if current in names:
            self.preset_choice.set(current)
        elif self.preset_id:
            match = get_preset(self.preset_id)
            self.preset_choice.set(match.name if match is not None else GENERIC_LABEL)
        else:
            self.preset_choice.set(GENERIC_LABEL)
        self._sync_preset_buttons()

    def _sync_preset_buttons(self) -> None:
        preset = get_preset(self.preset_choice.get())
        can_delete = preset is not None
        self.delete_preset_btn.configure(state="normal" if can_delete else "disabled")

    def _clear_preset_selection(self) -> None:
        self.preset_id = None
        self.preset_choice.set(GENERIC_LABEL)
        self._sync_preset_buttons()

    def apply_selected_preset(self) -> None:
        """Apply Generic (k-means identity) or a named palette from the combobox."""
        name = self.preset_choice.get()
        self._sync_preset_buttons()
        if is_generic_label(name):
            self.preset_id = None
            self.range_by.set(RANGE_BY_COLOR_LABEL)
            self._set_assign_mode(ASSIGN_KMEANS)
            self._sync_range_by_controls()
            if self.work_image is None:
                self.status.set("Generic — open a wallpaper to cluster from the image.")
                return
            auto_k = self._apply_auto_k()
            self.rebuild_ranges()
            if auto_k is not None:
                self.status.set(
                    f"Generic — {auto_k} ranges from silhouette / inertia."
                )
            else:
                self.status.set("Generic — N most matching colors from the image.")
            return
        preset = get_preset(name)
        if preset is None:
            self._clear_preset_selection()
            return
        self.preset_id = preset.id
        self.range_count.set(preset.range_count)
        # V6-N (and color presets): Range by Color closeness — nearest of palette hexes
        self._set_range_by_from_method(preset.split_method)
        if self.work_image is None:
            self.status.set(f"{preset.name} ready — open a wallpaper to build color ranges.")
            return
        self.rebuild_ranges()
        if self.preset_id == preset.id:
            self.status.set(f"{preset.name} — Save as… matches the Result preview (Texture slider + eye).")

    def save_preset(self) -> None:
        """Snapshot the current ranges into presets.json under a name the user types."""
        if self.range_map is None:
            messagebox.showinfo("Nothing to save", "Open an image and set colors first.", parent=self.root)
            return
        name = simpledialog.askstring("Save preset", "Preset name:", parent=self.root)
        if name is None:
            return
        name = name.strip()
        if not name:
            return
        if is_generic_label(name):
            messagebox.showerror(
                "Can't use that name",
                "Generic is the no-preset default — pick another name.",
                parent=self.root,
            )
            return
        existing = get_preset(name)
        if existing is not None:
            if not messagebox.askyesno(
                "Overwrite preset",
                f"A preset named {existing.name} already exists. Overwrite it?",
                parent=self.root,
            ):
                return
        try:
            snapshot = snapshot_preset(
                name,
                self.range_map,
                preset_id=existing.id if existing is not None else None,
            )
            saved = save_user_preset(snapshot)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not save preset", str(exc), parent=self.root)
            return
        self.preset_id = saved.id
        self._reload_preset_combo()
        self.preset_choice.set(saved.name)
        self._sync_preset_buttons()
        self.status.set(f"Saved preset “{saved.name}” to presets.json")

    def delete_selected_preset(self) -> None:
        """Remove the selected preset from presets.json (including V6-N)."""
        preset = get_preset(self.preset_choice.get())
        if preset is None:
            return
        if not messagebox.askyesno("Delete preset", f"Delete “{preset.name}”?", parent=self.root):
            return
        if not delete_user_preset(preset.id):
            messagebox.showerror("Could not delete preset", "The preset was not found on disk.", parent=self.root)
            return
        was_active = self.preset_id == preset.id
        self.preset_id = None
        self.preset_choice.set(GENERIC_LABEL)
        self._reload_preset_combo()
        self.status.set(f"Deleted preset “{preset.name}”")
        if was_active and self.work_image is not None:
            self.rebuild_ranges()

    def pick_icc(self) -> None:
        """Printer ICC for a separate soft-proof in the job pack (RGB master stays)."""
        path = filedialog.askopenfilename(title="Printer ICC profile", filetypes=ICC_FILETYPES)
        if not path:
            return
        self.icc_path = Path(path)
        self.status.set(f"ICC proof profile: {self.icc_path.name}  (applied on Export job pack)")

    def _sync_range_by_controls(self) -> None:
        """Assign when Range by is color; luma Min % + Start, or Lab Start, before Reset."""
        closeness = self.range_by.get() == RANGE_BY_COLOR_LABEL
        for w in (
            self.bin_min_caption,
            self.bin_min_spin,
            self.bin_start_caption,
            self.bin_start_spin,
            self.luma_kind_caption,
            self.luma_kind_combo,
        ):
            w.pack_forget()
        if closeness:
            self.assign_caption.pack(side="left", before=self.reset_btn)
            self.assign_combo.pack(side="left", before=self.reset_btn, padx=(4, 12))
            return
        self.assign_caption.pack_forget()
        self.assign_combo.pack_forget()
        if self.range_by.get() == RANGE_BY_LUMA_LABEL:
            self.luma_kind_caption.configure(text="Luma split:")
        else:
            self.luma_kind_caption.configure(text="Split:")
        self.luma_kind_caption.pack(side="left", before=self.reset_btn)
        self.luma_kind_combo.pack(side="left", before=self.reset_btn, padx=(4, 12))
        self._configure_bin_start_spin()
        if self.range_by.get() == RANGE_BY_LUMA_LABEL:
            self.bin_min_caption.pack(side="left", before=self.reset_btn)
            self.bin_min_spin.pack(side="left", before=self.reset_btn, padx=(4, 8))
        self.bin_start_caption.pack(side="left", before=self.reset_btn)
        self.bin_start_spin.pack(side="left", before=self.reset_btn, padx=(4, 12))

    def _configure_bin_start_spin(self) -> None:
        method = self._split_method()
        lo = resolved_bin_start(method, None)
        if is_lab_channel_split(method):
            ch = split_axis_channel(method)
            hi = 100.0 if ch == 0 else 127.0
            lo_s = 0.0 if ch == 0 else -128.0
            inc = 1.0
        else:
            lo_s, hi, inc = 0.0, 255.0, 1.0
        try:
            self.bin_start_spin.configure(from_=lo_s, to=hi, increment=inc)
        except tk.TclError:
            pass
        del lo

    def _ui_bin_start(self) -> float:
        method = self._split_method()
        try:
            raw = float(self.bin_start.get())
        except (tk.TclError, ValueError):
            raw = resolved_bin_start(method, None)
        return resolved_bin_start(method, raw)

    def _ui_min_coverage(self) -> float:
        try:
            pct = float(self.bin_min_pct.get())
        except (tk.TclError, ValueError):
            pct = MIN_COVERAGE * 100.0
        return clamp_min_coverage(pct / 100.0)

    def _sync_bin_spins_from_map(self) -> None:
        rm = self.range_map
        prev = self._bin_limits_mute
        self._bin_limits_mute = True
        try:
            if rm is None:
                self.bin_min_pct.set(round(MIN_COVERAGE * 100.0))
                self.bin_start.set(resolved_bin_start(self._split_method(), None))
                return
            self.bin_min_pct.set(round(clamp_min_coverage(rm.min_coverage) * 100.0))
            self.bin_start.set(resolved_bin_start(rm.split_method, rm.bin_start))
            self._configure_bin_start_spin()
        finally:
            self._bin_limits_mute = prev

    def _on_bin_limits(self, _event=None) -> None:
        """Min % / Start: recut histogram bins; keep match-from / change-to colors."""
        if self._mute_ui or self._bin_limits_mute or self._history_lock:
            return
        start = self._ui_bin_start()
        floor = self._ui_min_coverage()
        prev = self._bin_limits_mute
        self._bin_limits_mute = True
        try:
            self.bin_start.set(start)
            self.bin_min_pct.set(round(floor * 100.0))
        finally:
            self._bin_limits_mute = prev
        if self.range_map is None or self.work_image is None:
            return
        before = self._capture_edit()
        apply_bin_limits(self.range_map, min_coverage=floor, bin_start=start)
        self._push_undo_state(before)
        self._sync_range_widgets(update_bar=True)
        self._schedule_preview()

    def _on_range_by(self) -> None:
        """Range by: dropdown — rebuild clusters or histogram bins on the current image."""
        method = self._split_method()
        self._bin_limits_mute = True
        try:
            self.bin_start.set(resolved_bin_start(method, None))
        finally:
            self._bin_limits_mute = False
        self._sync_range_by_controls()
        self.rebuild_ranges()

    def _on_assign_mode(self) -> None:
        """Cluster from image vs snap to palette hexes — full rebuild (not a range shift)."""
        self.rebuild_ranges()

    def _assign_mode(self) -> str:
        """kmeans (default) or palette — ignored when Range by is luma."""
        return ASSIGN_BY_LABEL.get(self.assign_label.get(), ASSIGN_KMEANS)

    def _set_assign_mode(self, mode: str) -> None:
        self.assign_label.set(ASSIGN_LABEL_FOR.get(mode, ASSIGN_KMEANS_LABEL))

    def _set_range_by_from_method(self, method: str) -> None:
        """Sync Range by: (and luma sub-option) from a ColorRangeMap split_method."""
        self.range_by.set(range_by_label_for(method))
        if not is_color_split(method):
            self.luma_split_label.set(split_label_for(method))
        self._sync_range_by_controls()

    def _split_method(self) -> str:
        """Map Range by: (+ equal / even-pixel sub-option) onto build_range_map."""
        by = self.range_by.get()
        if by == RANGE_BY_COLOR_LABEL:
            return SPLIT_COLOR_CLOSENESS
        pair = _RANGE_BY_BIN_METHODS.get(by)
        if pair is None:
            return SPLIT_COLOR_CLOSENESS
        equal_m, pixel_m = pair
        if self.luma_split_label.get() == SPLIT_EQUAL_PIXELS_LABEL:
            return pixel_m
        return equal_m

    def _requested_count(self) -> int:
        try:
            n = int(self.range_count.get())
        except (tk.TclError, ValueError):
            n = DEFAULT_RANGES
        return max(MIN_RANGES, min(MAX_RANGES, n))

    def _should_auto_k(self) -> bool:
        """True for Generic cluster-from-image (not a named preset or luma/snap)."""
        if self.work_image is None:
            return False
        if self.preset_id:
            return False
        if not is_color_split(self._split_method()):
            return False
        if is_palette_assign(self._assign_mode()):
            return False
        return True

    def _apply_auto_k(self) -> int | None:
        """Set Ranges from silhouette/inertia. None when auto-k does not apply."""
        if not self._should_auto_k():
            return None
        k = choose_kmeans_k(self.work_image)
        k = max(MIN_RANGES, min(MAX_RANGES, int(k)))
        self.range_count.set(k)
        return k

    def _prepare_range_count_for_import(self) -> int | None:
        """Preset N, else auto-k for Generic k-means, else the current spinbox."""
        preset = get_preset(self.preset_id) if self.preset_id else None
        if preset is not None:
            self.range_count.set(int(preset.range_count))
            return None
        return self._apply_auto_k()

    def _on_range_count(self) -> None:
        """Spinbox: insert/drop ranges without wiping existing match-from / change-to."""
        if self._mute_ui or self._history_lock or getattr(self, "_opening", False):
            return
        n = self._requested_count()
        self.range_count.set(n)
        if self.work_image is None:
            return
        if self.range_map is None:
            self.rebuild_ranges()
            return
        current = len(self.range_map.ranges)
        if n == current:
            return
        self._push_undo_state(self._capture_edit())
        if n > current:
            for _ in range(n - current):
                insert_color_range(self.range_map)
            self.selected_index = len(self.range_map.ranges) - 1
            self.selected_half = HALF_REPLACE
            self.status.set(
                "Added a color range — pick a Pantone change-to. Other ranges kept their colors."
            )
        else:
            for _ in range(current - n):
                drop_color_range(self.range_map, len(self.range_map.ranges) - 1)
            self.selected_index = min(self.selected_index, n - 1)
            self.status.set("Removed a color range — remaining match-from / change-to kept.")
        preset = get_preset(self.preset_id) if self.preset_id else None
        if preset is not None and preset.range_count != n:
            self._clear_preset_selection()
        self._sync_texture_to_map()
        self._rebuild_chips()
        self._load_selected_onto_wheel()
        self._sync_range_widgets(update_bar=True)
        self._sync_range_layers()
        self._refresh_layers_panel()
        self._refresh_now()

    def rebuild_ranges(self) -> None:
        """Recompute clusters or luma bins. Colors reset unless a preset applies."""
        if self.work_image is None:
            return
        if self.range_map is not None and not self._history_lock and not self._opening:
            self._push_undo_state(self._capture_edit())
        n = self._requested_count()
        self.range_count.set(n)
        preset = get_preset(self.preset_id) if self.preset_id else None
        if preset is not None and n != preset.range_count:
            preset = None
            self._clear_preset_selection()
        method = self._split_method()  # Range by Color closeness or luma brightness
        palette_rgb = None
        snap_centers = is_palette_assign(self._assign_mode())
        # Named color preset + Snap: Lab targets are match-from (V6-N greens, or saved match)
        if (
            snap_centers
            and preset is not None
            and preset.palette_as_centers
            and is_color_split(method)
        ):
            palette_rgb = list(preset.match_palette_rgb)
        self.range_map = build_range_map(
            self.work_image,
            n,
            method,
            palette_rgb=palette_rgb,
            bin_start=None if is_color_split(method) else self._ui_bin_start(),
            min_coverage=self._ui_min_coverage(),
        )
        self._sync_texture_to_map()
        self._sync_tone_to_map()
        if preset is not None:
            if not apply_preset_palette(self.range_map, preset, snap_centers=snap_centers):
                self._clear_preset_selection()
                self.status.set(f"Need {preset.range_count} ranges for {preset.name}.")
            elif preset.weights is not None and snap_centers:
                apply_weights(self.range_map, preset.weights)
        self.selected_index = 0
        self.selected_half = HALF_REPLACE
        if preset is None:
            self._store_generic_colors()
        self._rebuild_chips()
        self._load_selected_onto_wheel()
        self._sync_range_widgets(update_bar=True)
        self._ensure_base_layer()
        self._refresh_layers_panel()
        self._sync_bin_spins_from_map()
        self._refresh_now()

    # ---------------------------------------------------------------------------
    # Reset match-from / change-to to image colors (does not reset tone)
    # ---------------------------------------------------------------------------
    def reset_colors(self) -> None:
        """Snap match-from and change-to to the N most matching image colors."""
        if self.range_map is None:
            return
        self._push_undo_state(self._capture_edit())
        self._clear_preset_selection()
        reset_to_image_colors(self.range_map)
        self._store_generic_colors()
        self._load_selected_onto_wheel()
        self._sync_range_widgets(update_bar=True)
        self._refresh_now()

    def _store_generic_colors(self) -> None:
        """Remember identity k-means swatches so the coverage reset icon can hide."""
        if self.range_map is None:
            self._generic_match = None
            self._generic_replace = None
            return
        self._generic_match = tuple(band.match_rgb for band in self.range_map.ranges)
        self._generic_replace = tuple(band.replacement_rgb for band in self.range_map.ranges)

    def _colors_are_generic(self) -> bool:
        if self.range_map is None or self._generic_match is None or self._generic_replace is None:
            return True
        if len(self._generic_match) != len(self.range_map.ranges):
            return False
        for band, match, repl in zip(self.range_map.ranges, self._generic_match, self._generic_replace):
            if band.match_rgb != match or band.replacement_rgb != repl:
                return False
        return True

    def _has_range_selection(self) -> bool:
        """True when a range half is the wheel / eyedrop / cluster target."""
        if self.range_map is None:
            return False
        idx = int(self.selected_index)
        return 0 <= idx < len(self.range_map.ranges)

    def _set_scratch_rgb(self, rgb: tuple[int, int, int]) -> None:
        """Keep the wheel in sync when no range is selected."""
        self._scratch_rgb = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        wheel = getattr(self, "wheel", None)
        if wheel is None:
            return
        mute = self._mute_ui
        self._mute_ui = True
        try:
            wheel.set_rgb(self._scratch_rgb, notify=False)
        finally:
            self._mute_ui = mute

    def deselect_ranges(self) -> None:
        """Clear range selection; wheel / eyedrop / cluster edit scratch color only."""
        if int(self.selected_index) < 0:
            return
        wheel = getattr(self, "wheel", None)
        if wheel is not None:
            try:
                self._scratch_rgb = tuple(int(c) for c in wheel.current_rgb())
            except (TypeError, ValueError, AttributeError):
                pass
        self.selected_index = -1
        self._sync_range_widgets(update_bar=True)
        self._sync_eyedrop_cursor()

    def _on_escape_deselect(self, _event=None) -> str | None:
        if self._focus_is_entry():
            return None
        self.deselect_ranges()
        return "break"

    def _on_layers_bg_click(self, event) -> None:
        """Empty Layers background deselects ranges (row clicks bind on children)."""
        widget = getattr(event, "widget", None)
        if widget is self.layers_list or widget is self.layers_panel.body:
            self.deselect_ranges()

    def select_range(self, index: int, half: str | None = None, *, toggle: bool = False) -> None:
        """Make ``index`` the range the wheel and eyedrop edit; ``half`` is match or replace.

        ``index < 0`` deselects. ``toggle`` (UI clicks) deselects when the same
        range+half is already active.
        """
        if self.range_map is None:
            return
        n = len(self.range_map.ranges)
        idx = int(index)
        if idx < 0 or n == 0:
            self.deselect_ranges()
            return
        idx = max(0, min(idx, n - 1))
        half_ok = half in (HALF_MATCH, HALF_REPLACE)
        if toggle and idx == int(self.selected_index):
            if not half_ok or half == self.selected_half:
                self.deselect_ranges()
                return
        self.selected_index = idx
        if half_ok:
            self.selected_half = half
        self._load_selected_onto_wheel()
        self._sync_range_widgets(update_bar=True)
        self._sync_eyedrop_cursor()

    def set_range_color(self, index: int, rgb: tuple[int, int, int]) -> None:
        """Store a wheel / eyedrop color on the active swatch and refresh."""
        if self.range_map is None:
            return
        if self.selected_half == HALF_MATCH:
            self.set_match_color(index, rgb)
            return
        record = self._should_record_swatch_undo()
        before = self._capture_edit() if record else None
        self.range_map.set_replacement(index, rgb)
        self._sync_range_widgets(update_bar=True, update_slider=False)
        self._schedule_preview()
        if record:
            self._push_undo_state(before)

    def set_match_color(self, index: int, rgb: tuple[int, int, int]) -> None:
        """Set match-from, rebuild Lab assignment (pixels nearest the N from-colors)."""
        if self.range_map is None:
            return
        record = self._should_record_swatch_undo()
        before = self._capture_edit() if record else None
        self.range_map.set_match(index, rgb)
        if is_color_split(self.range_map.split_method):
            apply_weights(self.range_map, self.range_map.weights())
        self._sync_range_widgets(update_bar=True, update_slider=False)
        self._schedule_preview()
        if record:
            self._push_undo_state(before)

    def _should_record_swatch_undo(self) -> bool:
        """True for a discrete swatch change; False during wheel drag (commit on release)."""
        return (
            self._wheel_before is None
            and not self._history_lock
            and not self._mute_ui
        )

    def set_range_visible(self, index: int, visible: bool) -> None:
        """Range 👁: hidden pixels stay original; save/export honor this."""
        if self.range_map is None:
            return
        n = len(self.range_map.ranges)
        index = max(0, min(int(index), n - 1))
        band = self.range_map.ranges[index]
        if bool(band.visible) == bool(visible):
            return
        self._push_undo_state(self._capture_edit())
        band.visible = bool(visible)
        self._sync_range_layers()
        self._sync_range_widgets(update_bar=True, update_slider=False)
        self._refresh_layers_panel()
        self._schedule_preview()

    def _on_bar_toggle_visible(self, index: int) -> None:
        if self.range_map is None:
            return
        band = self.range_map.ranges[index]
        self.set_range_visible(index, not band.visible)

    def apply_typed_percent(self, index: int, pct: float) -> None:
        """Typed coverage: this range gets pct; one neighbor gives/takes (MIN_COVERAGE).

        Color closeness: Lab-nearest cluster. Luma / L* / a* / b*: bar-adjacent.
        """
        if self.range_map is None:
            return
        self._push_undo_state(self._capture_edit())
        set_range_weight(self.range_map, index, float(pct) / 100.0)
        self.select_range(index)
        self._sync_range_widgets(update_bar=True)
        self._schedule_preview()

    def _on_wheel_color(self, rgb: tuple[int, int, int]) -> None:
        """Wheel drag / hex / shade chip — apply to the active match-from or change-to swatch."""
        if self._mute_ui:
            return
        if primary_is_label(self.layer_stack):
            if self._wheel_before is None:
                self._wheel_before = self._capture_edit()
            ly = self._selected_label_layer()
            if ly is not None:
                ly.color = tuple(int(c) for c in rgb)
                self._sync_label_fields_from_layer(ly)
                self._schedule_preview()
            return
        if self.range_map is None:
            return
        if not self._has_range_selection():
            self._set_scratch_rgb(rgb)
            return
        if self._wheel_before is None:
            self._wheel_before = self._capture_edit()
        self.set_range_color(self.selected_index, rgb)

    def _on_wheel_commit(self, rgb: tuple[int, int, int]) -> None:
        """One undo tick after wheel release / hex / shade — not every drag pixel."""
        self.wheel.record_color(rgb)
        if not self._has_range_selection() and not primary_is_label(self.layer_stack):
            self._wheel_before = None
            return
        if self._wheel_before is not None:
            self._push_undo_state(self._wheel_before)
            self._wheel_before = None

    def _on_coverage_weights(
        self,
        weights: list[float],
        pair: tuple[int, int] | None = None,
    ) -> None:
        """Coverage-bar divider drag: rebin histogram or reweight Lab clusters.

        Divider pair is the two ranges on either side of the handle (bar neighbors).
        Color closeness freezes those two labels so a far cluster cannot grow.
        """
        if self._mute_ui or self.range_map is None:
            return
        freeze = pair if is_color_split(self.range_map.split_method) else None
        apply_weights(self.range_map, weights, freeze_pair=freeze)
        # Bar already redrew itself — don't set_state or the handle jumps
        self._sync_range_widgets(update_bar=False)
        self._schedule_preview()

    def _load_selected_onto_wheel(self) -> None:
        """Push the active swatch RGB onto the wheel without re-emitting."""
        if self.range_map is None:
            return
        if not self._has_range_selection():
            rgb = tuple(int(c) for c in self._scratch_rgb)
            self._mute_ui = True
            try:
                self.wheel.set_rgb(rgb, notify=False)
            finally:
                self._mute_ui = False
            return
        band = self.range_map.ranges[self.selected_index]
        rgb = band.match_rgb if self.selected_half == HALF_MATCH else band.replacement_rgb
        self._mute_ui = True
        try:
            self.wheel.set_rgb(rgb, notify=False)
        finally:
            self._mute_ui = False

    def _sync_range_widgets(self, *, update_bar: bool, update_slider: bool = True) -> None:
        """Chips, caption, and optionally the coverage bar — not the image."""
        if self.range_map is None:
            return
        n = len(self.range_map.ranges)
        if int(self.selected_index) >= n:
            self.selected_index = n - 1 if n else -1
        closeness = is_color_split(self.range_map.split_method)
        selected = int(self.selected_index) if self._has_range_selection() else -1
        if update_bar:
            self.coverage.set_state(
                self.range_map.weights(),
                self.range_map.match_colors(),
                self.range_map.replacement_colors(),
                selected,
                visibilities=[band.visible for band in self.range_map.ranges],
                selected_half=self.selected_half,
                luma_mode=not closeness,
                luma_keys=[
                    bin_display_key(
                        band.luma_low, band.luma_high, self.range_map.split_method
                    )
                    for band in self.range_map.ranges
                ],
                min_coverage=getattr(self.range_map, "min_coverage", MIN_COVERAGE),
            )
        for chip in self.range_chips:
            chip.refresh(self.range_map, selected, self.selected_half)
        if not self._has_range_selection():
            self.edit_caption.set(
                "No range selected — wheel, eyedrop, and cluster edit a scratch color only."
            )
            self._sync_slider_resets()
            self._sync_eyedrop_cursor()
            self._sync_layer_range_rows()
            self._schedule_cluster_view()
            return
        band = self.range_map.ranges[self.selected_index]
        label = band.name or f"range {band.index + 1}"
        half_txt = "match-from" if self.selected_half == HALF_MATCH else "change-to"
        if closeness:
            self.edit_caption.set(
                f"Editing {label} {half_txt}  —  "
                f"Lab cluster  —  "
                f"{band.share * 100:.0f}% of pixels  —  "
                f"size {band.weight * 100:.0f}%"
            )
        else:
            axis = split_axis_caption(self.range_map.split_method)
            self.edit_caption.set(
                f"Editing {label} {half_txt}  —  "
                f"{axis} {band.luma_low:.0f}–{band.luma_high:.0f}  —  "
                f"{band.weight * 100:.0f}% of the image"
            )
        self._sync_slider_resets()
        self._sync_eyedrop_cursor()
        self._sync_layer_range_rows()
        self._schedule_cluster_view()

    def _rebuild_chips(self) -> None:
        for chip in self.range_chips:
            chip.frame.destroy()
        self.range_chips.clear()
