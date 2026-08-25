# -*- coding: utf-8 -*-
"""
wallpaper_recolor.ui.mixins.session
------------------------------
File open/export/job pack, edit-state, undo history.

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
    canonicalize_split_method,
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


class AppSessionMixin:
    """File open/export/job pack, edit-state, undo history."""

    # ---------------------------------------------------------------------------
    # Open / .wpedit session
    # ---------------------------------------------------------------------------
    def open_image(self) -> None:
        """Ask for a TIF/PNG/JPEG and rebuild ranges from it."""
        if self._busy:
            return
        path = filedialog.askopenfilename(title="Open wallpaper image", filetypes=OPEN_FILETYPES)
        if not path:
            return
        self._open_image_from_path(Path(path), reset_edits=True)

    def _open_image_from_path(self, path: Path, *, reset_edits: bool = True) -> bool:
        """Load pixels from ``path``. ``reset_edits`` matches File → Open (crop/tess/labels)."""
        try:
            image = load_image(path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not open image", str(exc), parent=self.root)
            return False

        self.source_path = Path(path)
        self.source_image = image
        self.work_image = _fit(image, WORK_MAX_EDGE)
        self.status.set(f"{self.source_path.name}  —  {image.size[0]} × {image.size[1]}")
        self._sync_orig_title()
        if not reset_edits:
            return True
        self.crop_x.set(float(CROP_XY_DEFAULT))
        self.crop_y.set(float(CROP_XY_DEFAULT))
        self.crop_zoom.set(float(ZOOM_DEFAULT))
        self._sync_crop_bounds(clamp=True)
        self._sync_crop_entry_text()
        self.tess_mode.set(MODE_DEFAULT)
        self._sync_tess_mode_combo()
        self.tess_h.set(SIDE_OFF)
        self.tess_v.set(SIDE_OFF)
        self.tess_built.set(False)
        self.tess_tiles.set(float(TILES_DEFAULT))
        self.tess_lloyd.set(float(LLOYD_DEFAULT))
        self.tess_normalize.set(False)
        self._lighting_auto_darks = 0.0
        self._lighting_auto_lights = 0.0
        self._sync_tess_mosaic_text()
        self._tess_committed = (SIDE_OFF, SIDE_OFF, False, MODE_DEFAULT)
        self._sync_tess_mosaic_controls()
        self._reset_labels_state()
        self.layer_stack.clear()
        self._ensure_base_layer()
        self._refresh_layers_panel()
        self._opening = True
        try:
            auto_k = self._prepare_range_count_for_import()
            self.rebuild_ranges()
        finally:
            self._opening = False
        self._clear_history()
        if self.scale_lock.get():
            self._fill_locked_other(from_width=True)
        self._refresh_scale_labels()
        if auto_k is not None:
            self.status.set(
                f"{self.source_path.name}  —  {auto_k} ranges "
                f"(silhouette / inertia, cap {AUTO_K_MAX})"
            )
        return True

    def _sync_orig_title(self) -> None:
        """Original pane header: ``Original (basename)`` when a file is open."""
        name = self.source_path.name if self.source_path is not None else ""
        if name:
            self.orig_title.set(f"Original ({name})")
        else:
            self.orig_title.set("Original")

    def save_edit_state(self, *, path: Path | None = None) -> bool:
        """Serialize the current session to JSON (.wpedit or *_edit.json).

        Returns False if the user cancels the file dialog or the write fails.
        """
        if self._busy:
            return False
        dest = path
        if dest is None:
            stem = self.source_path.stem if self.source_path is not None else "wallpaper"
            chosen = filedialog.asksaveasfilename(
                title="Save Wallpaper Edit state",
                filetypes=EDIT_STATE_FILETYPES,
                defaultextension=".wpedit",
                initialfile=f"{stem}_edit.wpedit",
                parent=self.root,
            )
            if not chosen:
                return False
            dest = Path(chosen)
        dest = Path(dest)
        try:
            self._write_edit_state(dest)
        except (OSError, TypeError, ValueError) as exc:
            messagebox.showerror("Could not save Edit state", str(exc), parent=self.root)
            return False
        self._edit_state_path = dest
        self._mark_session_clean()
        self.status.set(f"Saved Edit state {dest.name}")
        return True

    def load_edit_state(self) -> None:
        """Restore a .wpedit / *_edit.json session (reopen the image if the path exists)."""
        if self._busy:
            return
        path = filedialog.askopenfilename(
            title="Open Edit state",
            filetypes=EDIT_STATE_FILETYPES,
        )
        if not path:
            return
        try:
            self._read_edit_state(Path(path))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            messagebox.showerror("Could not open Edit state", str(exc), parent=self.root)
            return
        self._edit_state_path = Path(path)

    def _session_state_snapshot(self) -> object:
        """Canonical session payload used to detect unsaved edits."""
        return json.loads(json.dumps(self._capture_session_state()))

    def _mark_session_clean(self) -> None:
        """Treat the current session as matching the last successful save/load."""
        self._saved_session_state = self._session_state_snapshot()

    def _edit_state_is_dirty(self) -> bool:
        """True when closing would lose work that is not in the last saved session."""
        if self._saved_session_state is None:
            return self.source_image is not None or self.range_map is not None
        return self._session_state_snapshot() != self._saved_session_state

    def _write_edit_state(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._capture_session_state(), indent=2) + "\n",
            encoding="utf-8",
        )

    def _read_edit_state(self, path: Path) -> None:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Edit state file is not a JSON object.")
        self._apply_session_state(raw)
        self.status.set(f"Loaded Edit state {Path(path).name}")

    def _capture_session_state(self) -> dict:
        """Full UI session: image path, ranges, tone, crop, tess, labels, scale, zoom, history."""
        snap = self._capture_edit()
        ranges: list[dict] = []
        if snap is not None:
            for i, name in enumerate(snap.names):
                ranges.append(
                    {
                        "name": name,
                        "match_rgb": list(snap.match_rgbs[i]) if i < len(snap.match_rgbs) else [0, 0, 0],
                        "replacement_rgb": (
                            list(snap.replacements[i]) if i < len(snap.replacements) else [0, 0, 0]
                        ),
                        "weight": float(snap.weights[i]) if i < len(snap.weights) else 0.0,
                        "visible": bool(snap.visibilities[i]) if i < len(snap.visibilities) else True,
                    }
                )
        detect_roi = None
        if self._detect_roi is not None:
            detect_roi = list(self._detect_roi)
        state = {
            "format": EDIT_STATE_FORMAT,
            "version": EDIT_STATE_VERSION,
            "image_path": str(self.source_path) if self.source_path is not None else "",
            "preset_id": self.preset_id,
            "split_method": canonicalize_split_method(
                self._split_method() if snap is None else snap.split_method
            ),
            "assignment_mode": self._assign_mode() if snap is None else getattr(snap, "assignment_mode", ASSIGN_KMEANS),
            "range_count": int(self._requested_count() if snap is None else snap.range_count),
            "bin_start": None if snap is None else getattr(snap, "bin_start", None),
            "min_coverage": (
                self._ui_min_coverage()
                if snap is None
                else clamp_min_coverage(getattr(snap, "min_coverage", MIN_COVERAGE))
            ),
            "selected_index": int(self.selected_index if snap is None else snap.selected_index),
            "selected_half": str(self.selected_half if snap is None else snap.selected_half),
            "ranges": ranges,
            "texture": {
                "strength": float(self._texture_strength() if snap is None else snap.texture_strength),
                "enabled": bool(self.texture_enabled.get() if snap is None else snap.texture_enabled),
            },
            "tone": {
                "darks": float(self.darks_pct.get() / 100.0 if snap is None else snap.tone_darks),
                "lights": float(self.lights_pct.get() / 100.0 if snap is None else snap.tone_lights),
                "brightness": float(
                    self.brightness_pct.get() / 100.0 if snap is None else snap.tone_brightness
                ),
                "contrast": float(self.contrast_pct.get() / 100.0 if snap is None else snap.tone_contrast),
                "exposure": float(self.exposure_pct.get() / 100.0 if snap is None else snap.tone_exposure),
                "lights_reds": float(
                    self.lights_reds_pct.get() / 100.0 if snap is None else snap.tone_lights_reds
                ),
                "lights_greens": float(
                    self.lights_greens_pct.get() / 100.0 if snap is None else snap.tone_lights_greens
                ),
                "lights_blues": float(
                    self.lights_blues_pct.get() / 100.0 if snap is None else snap.tone_lights_blues
                ),
                "lights_cyan": float(
                    self.balance_cyan_pct.get() / 100.0
                    if snap is None
                    else getattr(
                        snap,
                        "tone_balance_cyan",
                        getattr(snap, "tone_lights_cyan", 0.0),
                    )
                ),
                "lights_magenta": float(
                    self.balance_magenta_pct.get() / 100.0
                    if snap is None
                    else getattr(
                        snap,
                        "tone_balance_magenta",
                        getattr(snap, "tone_lights_magenta", 0.0),
                    )
                ),
                "lights_yellow": float(
                    self.balance_yellow_pct.get() / 100.0
                    if snap is None
                    else getattr(
                        snap,
                        "tone_balance_yellow",
                        getattr(snap, "tone_lights_yellow", 0.0),
                    )
                ),
                "darks_cyan": float(
                    0.0 if snap is None else getattr(snap, "tone_darks_cyan", 0.0)
                ),
                "darks_magenta": float(
                    0.0 if snap is None else getattr(snap, "tone_darks_magenta", 0.0)
                ),
                "darks_yellow": float(
                    0.0 if snap is None else getattr(snap, "tone_darks_yellow", 0.0)
                ),
                "temperature": float(
                    self.temperature_pct.get() / 100.0
                    if snap is None
                    else getattr(snap, "tone_temperature", 0.0)
                ),
                "tint": float(
                    self.tint_pct.get() / 100.0
                    if snap is None
                    else getattr(snap, "tone_tint", 0.0)
                ),
                "saturation": float(
                    self.saturation_pct.get() / 100.0
                    if snap is None
                    else getattr(snap, "tone_saturation", 0.0)
                ),
                "balance_cyan": float(
                    self.balance_cyan_pct.get() / 100.0
                    if snap is None
                    else getattr(
                        snap,
                        "tone_balance_cyan",
                        getattr(snap, "tone_lights_cyan", 0.0),
                    )
                ),
                "balance_magenta": float(
                    self.balance_magenta_pct.get() / 100.0
                    if snap is None
                    else getattr(
                        snap,
                        "tone_balance_magenta",
                        getattr(snap, "tone_lights_magenta", 0.0),
                    )
                ),
                "balance_yellow": float(
                    self.balance_yellow_pct.get() / 100.0
                    if snap is None
                    else getattr(
                        snap,
                        "tone_balance_yellow",
                        getattr(snap, "tone_lights_yellow", 0.0),
                    )
                ),
            },
            "crop": {
                "x": int(round(self.crop_x.get())) if snap is None else int(snap.crop_x),
                "y": int(round(self.crop_y.get())) if snap is None else int(snap.crop_y),
                "zoom": float(self.crop_zoom.get() if snap is None else snap.crop_zoom),
            },
            "tessellate": {
                "h": str(self.tess_h.get() if snap is None else snap.tess_h),
                "v": str(self.tess_v.get() if snap is None else snap.tess_v),
                "built": bool(self.tess_built.get() if snap is None else snap.tess_built),
                "mode": str(self.tess_mode.get() if snap is None else snap.tess_mode),
                "tiles": int(self._tess_tiles_value() if snap is None else snap.tess_tiles),
                "lloyd": int(self._tess_lloyd_value() if snap is None else snap.tess_lloyd),
                "normalize": bool(self._tess_normalize_on() if snap is None else snap.tess_normalize),
                "lighting_auto_darks": float(
                    self._lighting_auto_darks if snap is None else snap.lighting_auto_darks
                ),
                "lighting_auto_lights": float(
                    self._lighting_auto_lights if snap is None else snap.lighting_auto_lights
                ),
            },
            "labels": {
                "text": str(self.label_text.get() or ""),
                "size": int(clamp_label_size(self.label_size.get())),
                "color": str(self.label_color.get() or LABEL_COLOR_DEFAULT),
                "font": str(self.label_font.get() or LABEL_FONT_DEFAULT),
                "x": int(self._label_spec().x),
                "y": int(self._label_spec().y),
                "inpaint_boxes": [list(box) for box in self._inpaint_tuple()],
                "inpaint_layer_id": str(self._inpaint_layer_id or ""),
                "detect_boxes": [list(box) for box in self._detect_boxes],
                "detect_roi": detect_roi,
            },
            "layers": self.layer_stack.to_records(),
            "selected_layer_ids": list(self.layer_stack.selected_ids),
            "scale": {
                "unit": str(self.scale_unit.get()),
                "width": str(self.scale_width.get()),
                "height": str(self.scale_height.get()),
                "lock": bool(self.scale_lock.get()),
                "dpi_choice": str(self.scale_dpi_choice.get()),
                "dpi_custom": str(self.scale_dpi_custom.get()),
                "resample": str(self.scale_resample.get()),
            },
            "mockup": {
                "cover": str(self.mockup_cover.get()),
                "repeats": float(self.mockup_repeats.get()),
            },
            "preview_zoom": float(self._composite_zoom_pct),
            "layout_profile": self._layout_profile_name,
            "color_history": [list(rgb) for rgb in self.wheel.history_colors()],
        }
        return state

    def _apply_session_state(self, data: dict) -> None:
        """Reopen the image when the path exists, then restore ranges / tone / crop / …"""
        image_path = str(data.get("image_path") or "").strip()
        path = Path(image_path) if image_path else None
        if path is not None and path.is_file():
            if not self._open_image_from_path(path, reset_edits=False):
                return
        elif path is not None and image_path:
            if self.work_image is None:
                messagebox.showerror(
                    "Could not open Edit state",
                    f"Image not found:\n{path}",
                    parent=self.root,
                )
                return
            messagebox.showwarning(
                "Image missing",
                f"Image not found:\n{path}\nApplying the rest of the edit to the current image.",
                parent=self.root,
            )
        self._sync_orig_title()

        range_rows = data.get("ranges") if isinstance(data.get("ranges"), list) else []
        names = tuple(str(row.get("name") or "") for row in range_rows if isinstance(row, dict))
        match_rgbs = tuple(
            _json_rgb(row.get("match_rgb")) for row in range_rows if isinstance(row, dict)
        )
        replacements = tuple(
            _json_rgb(row.get("replacement_rgb")) for row in range_rows if isinstance(row, dict)
        )
        weights = tuple(
            float(row.get("weight") or 0.0) for row in range_rows if isinstance(row, dict)
        )
        visibilities = tuple(
            bool(row.get("visible", True)) for row in range_rows if isinstance(row, dict)
        )
        texture = data.get("texture") if isinstance(data.get("texture"), dict) else {}
        tone = data.get("tone") if isinstance(data.get("tone"), dict) else {}
        crop = data.get("crop") if isinstance(data.get("crop"), dict) else {}
        tess = data.get("tessellate") if isinstance(data.get("tessellate"), dict) else {}
        labels = data.get("labels") if isinstance(data.get("labels"), dict) else {}
        scale = data.get("scale") if isinstance(data.get("scale"), dict) else {}
        mockup = data.get("mockup") if isinstance(data.get("mockup"), dict) else {}

        n_ranges = int(data.get("range_count") or len(range_rows) or DEFAULT_RANGES)
        n_ranges = max(MIN_RANGES, min(MAX_RANGES, n_ranges))
        split_method = canonicalize_split_method(
            data.get("split_method") or SPLIT_COLOR_CLOSENESS
        )
        assignment_mode = str(data.get("assignment_mode") or ASSIGN_KMEANS)
        if assignment_mode not in (ASSIGN_KMEANS, ASSIGN_PALETTE):
            assignment_mode = ASSIGN_KMEANS
        inpaint = tuple(
            box
            for box in (_json_box(item) for item in (labels.get("inpaint_boxes") or []))
            if box is not None
        )
        inpaint_layer_id = str(labels.get("inpaint_layer_id") or "")
        detect_boxes = [
            box
            for box in (_json_box(item) for item in (labels.get("detect_boxes") or []))
            if box is not None
        ]
        detect_roi = _json_box(labels.get("detect_roi"))

        if self.work_image is not None:
            snap = EditSnapshot(
                replacements=replacements or ((0, 0, 0),) * n_ranges,
                match_rgbs=match_rgbs or ((0, 0, 0),) * n_ranges,
                weights=weights or tuple(1.0 / max(1, n_ranges) for _ in range(n_ranges)),
                visibilities=visibilities or (True,) * n_ranges,
                names=names or tuple(f"Range {i + 1}" for i in range(n_ranges)),
                texture_strength=float(texture.get("strength", TEXTURE_DEFAULT_STRENGTH)),
                texture_enabled=bool(texture.get("enabled", True)),
                tone_darks=float(tone.get("darks", 0.0)),
                tone_lights=float(tone.get("lights", 0.0)),
                tone_brightness=float(tone.get("brightness", 0.0)),
                tone_contrast=float(tone.get("contrast", 0.0)),
                tone_exposure=float(tone.get("exposure", 0.0)),
                tone_lights_reds=float(tone.get("lights_reds", 0.0)),
                tone_lights_greens=float(tone.get("lights_greens", 0.0)),
                tone_lights_blues=float(tone.get("lights_blues", 0.0)),
                tone_lights_cyan=float(
                    tone.get("balance_cyan", tone.get("lights_cyan", 0.0))
                ),
                tone_lights_magenta=float(
                    tone.get("balance_magenta", tone.get("lights_magenta", 0.0))
                ),
                tone_lights_yellow=float(
                    tone.get("balance_yellow", tone.get("lights_yellow", 0.0))
                ),
                tone_darks_cyan=float(tone.get("darks_cyan", 0.0)),
                tone_darks_magenta=float(tone.get("darks_magenta", 0.0)),
                tone_darks_yellow=float(tone.get("darks_yellow", 0.0)),
                tone_temperature=float(tone.get("temperature", 0.0)),
                tone_tint=float(tone.get("tint", 0.0)),
                tone_saturation=float(tone.get("saturation", 0.0)),
                tone_balance_cyan=float(
                    tone.get("balance_cyan", tone.get("lights_cyan", 0.0))
                ),
                tone_balance_magenta=float(
                    tone.get("balance_magenta", tone.get("lights_magenta", 0.0))
                ),
                tone_balance_yellow=float(
                    tone.get("balance_yellow", tone.get("lights_yellow", 0.0))
                ),
                split_method=split_method,
                range_count=n_ranges,
                selected_index=int(data.get("selected_index") or 0),
                selected_half=str(data.get("selected_half") or HALF_REPLACE),
                preset_id=data.get("preset_id"),
                assignment_mode=assignment_mode,
                crop_x=int(crop.get("x", CROP_XY_DEFAULT)),
                crop_y=int(crop.get("y", CROP_XY_DEFAULT)),
                crop_zoom=float(crop.get("zoom", ZOOM_DEFAULT)),
                tess_h=str(tess.get("h", SIDE_OFF)),
                tess_v=str(tess.get("v", SIDE_OFF)),
                tess_built=bool(tess.get("built", False)),
                tess_mode=str(tess.get("mode", MODE_DEFAULT)),
                tess_tiles=int(tess.get("tiles", TILES_DEFAULT)),
                tess_lloyd=int(tess.get("lloyd", LLOYD_DEFAULT)),
                tess_normalize=bool(tess.get("normalize", False)),
                lighting_auto_darks=float(tess.get("lighting_auto_darks", 0.0)),
                lighting_auto_lights=float(tess.get("lighting_auto_lights", 0.0)),
                inpaint_boxes=inpaint,
                inpaint_layer_id=inpaint_layer_id,
                label_text=str(labels.get("text") or ""),
                label_size=int(labels.get("size") or LABEL_SIZE_DEFAULT),
                label_color=str(labels.get("color") or LABEL_COLOR_DEFAULT),
                label_x=int(labels.get("x") or 0),
                label_y=int(labels.get("y") or 0),
                label_font=str(labels.get("font") or LABEL_FONT_DEFAULT),
                layers=tuple(
                    rec
                    for rec in (data.get("layers") or [])
                    if isinstance(rec, dict)
                ),
                selected_layer_ids=tuple(
                    str(i) for i in (data.get("selected_layer_ids") or [])
                ),
                bin_start=(
                    None
                    if data.get("bin_start") is None
                    else float(data.get("bin_start"))
                ),
                min_coverage=clamp_min_coverage(data.get("min_coverage", MIN_COVERAGE)),
            )
            self._restore_edit(snap)
            self._detect_boxes = detect_boxes
            self._detect_roi = detect_roi
            self._selected_detect = set()
            self._sync_label_modes()

        self._scale_updating = True
        try:
            if scale.get("unit"):
                self.scale_unit.set(str(scale["unit"]))
            self.scale_width.set(str(scale.get("width", "")))
            self.scale_height.set(str(scale.get("height", "")))
            if "lock" in scale:
                self.scale_lock.set(bool(scale["lock"]))
            if scale.get("dpi_choice"):
                self.scale_dpi_choice.set(str(scale["dpi_choice"]))
            self.scale_dpi_custom.set(str(scale.get("dpi_custom", "")))
            if scale.get("resample"):
                self.scale_resample.set(str(scale["resample"]))
        finally:
            self._scale_updating = False
        self._sync_dpi_custom_row()
        self._refresh_scale_labels()

        cover = str(mockup.get("cover") or DEFAULT_MOCKUP_COVER)
        if cover in MOCKUP_COVER_LABELS:
            self.mockup_cover.set(cover)
        try:
            self.mockup_repeats.set(float(mockup.get("repeats", DEFAULT_MOCKUP_REPEATS)))
        except (TypeError, ValueError, tk.TclError):
            pass
        self._on_mockup_cover()
        self._on_mockup_scale("")

        try:
            self._set_preview_zoom_pct(float(data.get("preview_zoom", VIEW_ZOOM_PCT_DEFAULT)))
        except (TypeError, ValueError, tk.TclError):
            pass

        profile_name = data.get("layout_profile")
        if isinstance(profile_name, str) and profile_name.strip():
            name = profile_name.strip()
            profiles = self._load_layout_profiles()
            if name in profiles:
                self._apply_layout_profile(name)
            else:
                self._layout_profile_name = name
        else:
            self._layout_profile_name = None

        self.wheel.set_history(data.get("color_history"))
        self._clear_history()
        self._refresh_now()
        self._mark_session_clean()

    def _save_uses_grain(self) -> bool:
        """True when Result is Color/Luminosity grain (texture eye on and slider > 0)."""
        return bool(self.texture_enabled.get()) and self._texture_strength() > 0.0

    # ---------------------------------------------------------------------------
    # Save / export (background thread; preview stays live)
    # ---------------------------------------------------------------------------
    def save_image_as(self) -> None:
        """Write the full-resolution Result composite (Ctrl+S) — same mix as the preview."""
        if self._busy:
            return
        self._save_composite(grain=self._save_uses_grain())

    def _flatten_document(
        self,
        source: Image.Image,
        range_snap: ColorRangeMap,
        *,
        grain: bool,
        holes: tuple,
        crop_x: float,
        crop_y: float,
        crop_zoom: float,
        tess_h: str,
        tess_v: str,
        tess_built: bool,
        tess_mode: str,
        tess_tiles: int,
        tess_lloyd: int,
        tess_normalize: bool = False,
        stack: LayerStack | None = None,
        selected_ids: tuple[str, ...] | None = None,
        inpaint_layer_id: str | None = None,
    ) -> Image.Image:
        """Full-res Result: corrections only on selected visible Image layers."""
        stack = stack if stack is not None else self.layer_stack
        chosen = tuple(selected_ids if selected_ids is not None else stack.selected_ids)
        targets = correction_target_ids(stack, chosen)
        hole_id = str(inpaint_layer_id or "")
        base = stack.base_layer()
        if not hole_id and base is not None:
            hole_id = base.id
        wrap_holes = normalize_tess_mode(tess_mode) == MODE_TILE
        processed: dict[str, Image.Image] = {}
        for ly in stack.layers:
            if not ly.visible or not ly.is_image():
                continue
            if ly.is_base():
                img = source
            elif ly.path and Path(ly.path).is_file():
                try:
                    img = load_image(ly.path)
                except (OSError, ValueError):
                    img = ly.raster
            else:
                img = ly.raster
            if img is None:
                continue
            selected = ly.id in targets
            fill_holes = bool(holes) and ly.id == hole_id
            if selected:
                img = composite_for_image(img, range_snap, grain=grain)
            if fill_holes:
                img = inpaint_image(
                    img,
                    holes,
                    src_size=source.size,
                    wrap=wrap_holes,
                    quads=self._inpaint_quads or None,
                    style=self._wallpaper_style_key(),
                )
            if selected:
                img = apply_crop_lighting_tessellate(
                    img,
                    crop_x,
                    crop_y,
                    crop_zoom,
                    tess_h,
                    tess_v,
                    tess_built,
                    mode=tess_mode,
                    tiles=tess_tiles,
                    lloyd=tess_lloyd,
                    normalize_lighting=tess_normalize,
                )
            elif ly.is_base():
                img = apply_crop(img, crop_x, crop_y, crop_zoom)
            processed[ly.id] = img
        if not processed:
            img = composite_for_image(source, range_snap, grain=grain)
            if holes:
                img = inpaint_image(
                    img,
                    holes,
                    wrap=wrap_holes,
                    quads=self._inpaint_quads or None,
                    style=self._wallpaper_style_key(),
                )
            return apply_crop_lighting_tessellate(
                img,
                crop_x,
                crop_y,
                crop_zoom,
                tess_h,
                tess_v,
                tess_built,
                mode=tess_mode,
                tiles=tess_tiles,
                lloyd=tess_lloyd,
                normalize_lighting=tess_normalize,
            )
        sample = next(iter(processed.values()))
        base = stack.base_layer()
        if base is not None and base.id in processed:
            sample = processed[base.id]
        rgba = composite_stack(
            stack,
            sample.size,
            source.size,
            processed=processed,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_zoom=crop_zoom,
        )
        return flatten_rgb_or_keep_alpha(rgba)

    def _save_composite(self, *, grain: bool) -> None:
        if self.source_image is None or self.range_map is None:
            messagebox.showinfo("Nothing to save", "Open an image first.", parent=self.root)
            return

        suffix = "_grain" if grain else "_exact"
        kind = "grain" if grain else "exact"
        initial = f"{kind}_master.png"
        if self.source_path is not None:
            initial = f"{self.source_path.stem}{suffix}{self.source_path.suffix}"

        path = filedialog.asksaveasfilename(
            title="Save grain (texture composite)" if grain else "Save exact master (flat palette)",
            filetypes=SAVE_FILETYPES,
            defaultextension=".png",
            initialfile=initial,
        )
        if not path:
            return

        # Snapshot assignment so wheel edits during save cannot race the worker
        self._sync_texture_to_map()
        self._sync_tone_to_map()
        source = self.source_image
        snap = snapshot_assignment(self.range_map)
        dest = path
        label = "grain" if grain else "exact master"
        # Scale panel only — not view zoom or the on-screen PhotoImage size
        out_size, resample, dpi = self._output_scale_args()
        crop_x, crop_y, crop_zoom = self._crop_xy_zoom()
        tess_h, tess_v, tess_built, tess_mode = self._tess_params()
        tess_tiles = self._tess_tiles_value()
        tess_lloyd = self._tess_lloyd_value()
        holes = self._inpaint_tuple()
        hole_layer = str(self._inpaint_layer_id or "")
        label_spec = self._label_spec()
        stack = self.layer_stack
        selected_ids = tuple(stack.selected_ids)

        def work():
            result = self._flatten_document(
                source,
                snap,
                grain=grain,
                holes=holes,
                crop_x=crop_x,
                crop_y=crop_y,
                crop_zoom=crop_zoom,
                tess_h=tess_h,
                tess_v=tess_v,
                tess_built=tess_built,
                tess_mode=tess_mode,
                tess_tiles=tess_tiles,
                tess_lloyd=tess_lloyd,
                tess_normalize=self._tess_normalize_on(),
                stack=stack,
                selected_ids=selected_ids,
                inpaint_layer_id=hole_layer,
            )
            result = scale_image(result, out_size, resample)
            has_label_layer = any(ly.is_label() and ly.visible and ly.to_label_spec().is_set() for ly in stack.layers)
            if not has_label_layer and label_spec.is_set():
                result = composite_label(
                    result,
                    label_spec,
                    source.size,
                    crop_x=crop_x,
                    crop_y=crop_y,
                    crop_zoom=crop_zoom,
                )
            save_image(result, dest, dpi=dpi)
            return dest

        def on_ok(saved) -> None:
            self.status.set(f"Saved {label} {Path(saved).name}")

        def on_err(exc: BaseException) -> None:
            messagebox.showerror("Could not save image", str(exc), parent=self.root)
            self.status.set("Save failed")

        self._run_background("Saving…", work, on_ok, on_err)

    def export_pack(self) -> None:
        """Full-res masks + both composites + tile/seam/mockup (+ ICC / PSD when possible)."""
        if self._busy:
            return
        if self.source_image is None or self.range_map is None:
            messagebox.showinfo("Nothing to export", "Open an image first.", parent=self.root)
            return
        parent = filedialog.askdirectory(title="Choose folder for the job pack")
        if not parent:
            return
        stem = self.source_path.stem if self.source_path is not None else "wallpaper"
        dest = Path(parent) / f"{stem}_jobpack"
        self._sync_texture_to_map()
        self._sync_tone_to_map()
        source = self.source_image
        snap = snapshot_assignment(self.range_map)
        source_path = self.source_path
        repeats = float(self.mockup_repeats.get())
        cover_frac = self._mockup_cover_frac()
        icc_path = self.icc_path
        preset_id = self.preset_id
        out_size, resample, dpi = self._output_scale_args()
        crop_x, crop_y, crop_zoom = self._crop_xy_zoom()
        tess_h, tess_v, tess_built, tess_mode = self._tess_params()
        tess_tiles = self._tess_tiles_value()
        tess_lloyd = self._tess_lloyd_value()
        tess_normalize = self._tess_normalize_on()
        holes = self._inpaint_tuple()
        label_spec = self._label_spec()

        def work():
            return export_job_pack(
                dest,
                source,
                snap,
                source_path=source_path,
                mockup_repeats=repeats,
                mockup_cover_frac=cover_frac,
                icc_path=icc_path,
                preset_id=preset_id,
                on_status=self._status_from_worker,
                output_size=out_size,
                output_resample=resample,
                output_dpi=dpi,
                crop_x=crop_x,
                crop_y=crop_y,
                crop_zoom=crop_zoom,
                tess_h=tess_h,
                tess_v=tess_v,
                tess_built=tess_built,
                tess_mode=tess_mode,
                tess_tiles=tess_tiles,
                tess_lloyd=tess_lloyd,
                tess_normalize=tess_normalize,
                inpaint_boxes=holes,
                label=label_spec if label_spec.is_set() else None,
            )

        def on_ok(written) -> None:
            self.status.set(f"Job pack: {written}")
            messagebox.showinfo("Job pack written", str(written), parent=self.root)

        def on_err(exc: BaseException) -> None:
            messagebox.showerror("Could not export job pack", str(exc), parent=self.root)
            self.status.set("Export failed")

        self._run_background("Exporting job pack…", work, on_ok, on_err)

    def export_layers_zip(self) -> None:
        """Per-range TIF+SVG plates, texture, and composite in one zip (Ctrl not bound)."""
        if self._busy:
            return
        if self.source_image is None or self.range_map is None:
            messagebox.showinfo("Nothing to export", "Open an image first.", parent=self.root)
            return
        stem = self.source_path.stem if self.source_path is not None else "wallpaper"
        path = filedialog.asksaveasfilename(
            title="Export layers zip",
            filetypes=[("ZIP", "*.zip")],
            defaultextension=".zip",
            initialfile=f"{stem}_layers.zip",
        )
        if not path:
            return
        self._sync_texture_to_map()
        self._sync_tone_to_map()
        source = self.source_image
        snap = snapshot_assignment(self.range_map)
        source_path = self.source_path
        preset_id = self.preset_id
        out_size, resample, dpi = self._output_scale_args()
        dest = Path(path)
        crop_x, crop_y, crop_zoom = self._crop_xy_zoom()
        tess_h, tess_v, tess_built, tess_mode = self._tess_params()
        tess_tiles = self._tess_tiles_value()
        tess_lloyd = self._tess_lloyd_value()
        tess_normalize = self._tess_normalize_on()
        holes = self._inpaint_tuple()
        hole_layer = str(self._inpaint_layer_id or "")
        label_spec = self._label_spec()
        stack = self.layer_stack
        selected_ids = tuple(stack.selected_ids)
        grain = self._save_uses_grain()
        label_specs = [
            ly.to_label_spec()
            for ly in stack.layers
            if ly.is_label() and ly.visible and ly.to_label_spec().is_set()
        ]
        overlays = [
            (ly.name, ly.raster)
            for ly in stack.layers
            if ly.is_image() and not ly.is_base() and ly.visible and ly.raster is not None
        ]

        def work():
            document = self._flatten_document(
                source,
                snap,
                grain=grain,
                holes=holes,
                crop_x=crop_x,
                crop_y=crop_y,
                crop_zoom=crop_zoom,
                tess_h=tess_h,
                tess_v=tess_v,
                tess_built=tess_built,
                tess_mode=tess_mode,
                tess_tiles=tess_tiles,
                tess_lloyd=tess_lloyd,
                tess_normalize=tess_normalize,
                stack=stack,
                selected_ids=selected_ids,
                inpaint_layer_id=hole_layer,
            )
            return write_layers_zip(
                dest,
                source,
                snap,
                source_path=source_path,
                preset_id=preset_id,
                on_status=self._status_from_worker,
                output_size=out_size,
                output_resample=resample,
                output_dpi=dpi,
                crop_x=crop_x,
                crop_y=crop_y,
                crop_zoom=crop_zoom,
                tess_h=tess_h,
                tess_v=tess_v,
                tess_built=tess_built,
                tess_mode=tess_mode,
                tess_tiles=tess_tiles,
                tess_lloyd=tess_lloyd,
                tess_normalize=tess_normalize,
                inpaint_boxes=holes,
                label=None if label_specs else (label_spec if label_spec.is_set() else None),
                label_specs=label_specs or None,
                overlay_layers=overlays or None,
                document=document,
            )

        def on_ok(written) -> None:
            self.status.set(f"Layers zip: {written}")
            messagebox.showinfo("Layers zip written", str(written), parent=self.root)

        def on_err(exc: BaseException) -> None:
            messagebox.showerror("Could not export layers zip", str(exc), parent=self.root)
            self.status.set("Layers zip failed")

        self._run_background("Exporting layers zip…", work, on_ok, on_err)

    def _capture_edit(self) -> EditSnapshot | None:
        """Colors, weights, eyes, texture, tone, crop, tessellate — enough to restore the preview."""
        if self.range_map is None:
            return None
        rm = self.range_map
        cx, cy, cz = self._crop_xy_zoom()
        tess_h, tess_v, tess_built, tess_mode = self._tess_params()
        tess_tiles = self._tess_tiles_value()
        tess_lloyd = self._tess_lloyd_value()
        return EditSnapshot(
            replacements=tuple(band.replacement_rgb for band in rm.ranges),
            match_rgbs=tuple(band.match_rgb for band in rm.ranges),
            weights=tuple(round(float(band.weight), 6) for band in rm.ranges),
            visibilities=tuple(bool(band.visible) for band in rm.ranges),
            names=tuple(band.name for band in rm.ranges),
            texture_strength=float(rm.texture_strength),
            texture_enabled=bool(rm.texture_enabled),
            tone_darks=float(rm.tone_darks),
            tone_lights=float(rm.tone_lights),
            tone_brightness=float(rm.tone_brightness),
            tone_contrast=float(getattr(rm, "tone_contrast", 0.0)),
            tone_exposure=float(getattr(rm, "tone_exposure", 0.0)),
            tone_lights_reds=float(rm.tone_lights_reds),
            tone_lights_greens=float(rm.tone_lights_greens),
            tone_lights_blues=float(rm.tone_lights_blues),
            tone_lights_cyan=float(getattr(rm, "tone_balance_cyan", getattr(rm, "tone_lights_cyan", 0.0))),
            tone_lights_magenta=float(getattr(rm, "tone_balance_magenta", getattr(rm, "tone_lights_magenta", 0.0))),
            tone_lights_yellow=float(getattr(rm, "tone_balance_yellow", getattr(rm, "tone_lights_yellow", 0.0))),
            tone_darks_cyan=float(getattr(rm, "tone_darks_cyan", 0.0)),
            tone_darks_magenta=float(getattr(rm, "tone_darks_magenta", 0.0)),
            tone_darks_yellow=float(getattr(rm, "tone_darks_yellow", 0.0)),
            tone_temperature=float(getattr(rm, "tone_temperature", 0.0)),
            tone_tint=float(getattr(rm, "tone_tint", 0.0)),
            tone_saturation=float(getattr(rm, "tone_saturation", 0.0)),
            tone_balance_cyan=float(getattr(rm, "tone_balance_cyan", getattr(rm, "tone_lights_cyan", 0.0))),
            tone_balance_magenta=float(getattr(rm, "tone_balance_magenta", getattr(rm, "tone_lights_magenta", 0.0))),
            tone_balance_yellow=float(getattr(rm, "tone_balance_yellow", getattr(rm, "tone_lights_yellow", 0.0))),
            split_method=canonicalize_split_method(rm.split_method),
            range_count=rm.range_count,
            selected_index=self.selected_index,
            selected_half=self.selected_half,
            preset_id=self.preset_id,
            assignment_mode=self._assign_mode(),
            crop_x=int(round(cx)),
            crop_y=int(round(cy)),
            crop_zoom=float(cz),
            tess_h=tess_h,
            tess_v=tess_v,
            tess_built=bool(tess_built),
            tess_mode=tess_mode,
            tess_tiles=int(tess_tiles),
            tess_lloyd=int(tess_lloyd),
            tess_normalize=self._tess_normalize_on(),
            lighting_auto_darks=float(self._lighting_auto_darks),
            lighting_auto_lights=float(self._lighting_auto_lights),
            inpaint_boxes=self._inpaint_tuple(),
            inpaint_quads=tuple(self._inpaint_quads),
            inpaint_layer_id=str(self._inpaint_layer_id or ""),
            label_text=str(self.label_text.get() or ""),
            label_size=clamp_label_size(self.label_size.get()),
            label_color=str(self.label_color.get() or LABEL_COLOR_DEFAULT),
            label_x=self._label_spec().x,
            label_y=self._label_spec().y,
            label_font=str(self.label_font.get() or LABEL_FONT_DEFAULT),
            layers=tuple(self.layer_stack.to_records()),
            selected_layer_ids=tuple(self.layer_stack.selected_ids),
            bin_start=getattr(rm, "bin_start", None),
            min_coverage=clamp_min_coverage(getattr(rm, "min_coverage", MIN_COVERAGE)),
            icc_path=str(self.icc_path) if self.icc_path is not None else None,
        )

    def _push_undo_state(self, before: EditSnapshot | None) -> None:
        if before is None or self._history_lock:
            return
        current = self._capture_edit()
        if current is not None and before == current:
            return
        self._undo_stack.append(before)
        if len(self._undo_stack) > HISTORY_LIMIT:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._sync_edit_history_labels()

    def _clear_history(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._wheel_before = None
        self._slider_before = None
        self._sync_edit_history_labels()

    def _mark_slider_begin(self, _event=None) -> None:
        if self._slider_before is None:
            self._slider_before = self._capture_edit()

    def _mark_slider_end(self, _event=None) -> None:
        if self._slider_before is not None:
            self._push_undo_state(self._slider_before)
            self._slider_before = None
        self._tess_committed = self._tess_params()

    def _focus_is_entry(self) -> bool:
        w = self.root.focus_get()
        return isinstance(w, (tk.Entry, ttk.Entry, ttk.Spinbox, tk.Spinbox))

    def _on_undo_key(self, _event=None) -> str | None:
        if self._focus_is_entry():
            return None
        self.undo_edit()
        return "break"

    def _on_redo_key(self, _event=None) -> str | None:
        if self._focus_is_entry():
            return None
        self.redo_edit()
        return "break"

    def undo_edit(self) -> None:
        """Ctrl+Z — restore the previous snapshot (at most 20 deep)."""
        if not self._undo_stack or self.range_map is None:
            return
        current = self._capture_edit()
        before = self._undo_stack.pop()
        if current is not None:
            self._redo_stack.append(current)
            if len(self._redo_stack) > HISTORY_LIMIT:
                self._redo_stack.pop(0)
        self._restore_edit(before)
        self._sync_edit_history_labels()

    def redo_edit(self) -> None:
        """Ctrl+Y / Ctrl+Shift+Z — walk forward through undone edits."""
        if not self._redo_stack or self.range_map is None:
            return
        current = self._capture_edit()
        nxt = self._redo_stack.pop()
        if current is not None:
            self._undo_stack.append(current)
            if len(self._undo_stack) > HISTORY_LIMIT:
                self._undo_stack.pop(0)
        self._restore_edit(nxt)
        self._sync_edit_history_labels()

    def _restore_edit(self, snap: EditSnapshot) -> None:
        """Rebuild labels from stored weights/centers, then colors, eyes, texture, tone."""
        if self.work_image is None:
            return
        self._history_lock = True
        self._mute_ui = True
        try:
            self.range_count.set(snap.range_count)
            method = canonicalize_split_method(snap.split_method)
            self._set_range_by_from_method(method)
            self._set_assign_mode(getattr(snap, "assignment_mode", ASSIGN_KMEANS))
            self.preset_id = snap.preset_id
            self.texture_pct.set(round(snap.texture_strength * 100.0, 1))
            self.texture_enabled.set(snap.texture_enabled)
            self.texture_eye.set_shown(snap.texture_enabled)
            self.darks_pct.set(snap.tone_darks * 100.0)
            self.lights_pct.set(snap.tone_lights * 100.0)
            self.brightness_pct.set(snap.tone_brightness * 100.0)
            self.contrast_pct.set(float(getattr(snap, "tone_contrast", 0.0)) * 100.0)
            self.exposure_pct.set(float(getattr(snap, "tone_exposure", 0.0)) * 100.0)
            self.lights_reds_pct.set(getattr(snap, "tone_lights_reds", 0.0) * 100.0)
            self.lights_greens_pct.set(getattr(snap, "tone_lights_greens", 0.0) * 100.0)
            self.lights_blues_pct.set(getattr(snap, "tone_lights_blues", 0.0) * 100.0)
            bal_c = float(
                getattr(
                    snap,
                    "tone_balance_cyan",
                    getattr(snap, "tone_lights_cyan", 0.0),
                )
            )
            bal_m = float(
                getattr(
                    snap,
                    "tone_balance_magenta",
                    getattr(snap, "tone_lights_magenta", 0.0),
                )
            )
            bal_y = float(
                getattr(
                    snap,
                    "tone_balance_yellow",
                    getattr(snap, "tone_lights_yellow", 0.0),
                )
            )
            self.balance_cyan_pct.set(bal_c * 100.0)
            self.balance_magenta_pct.set(bal_m * 100.0)
            self.balance_yellow_pct.set(bal_y * 100.0)
            self.temperature_pct.set(float(getattr(snap, "tone_temperature", 0.0)) * 100.0)
            self.tint_pct.set(float(getattr(snap, "tone_tint", 0.0)) * 100.0)
            self.saturation_pct.set(float(getattr(snap, "tone_saturation", 0.0)) * 100.0)
            self.texture_label.set(f"Texture: {snap.texture_strength * 100.0:.0f}%")
            self._sync_tone_labels()
            self._crop_updating = True
            try:
                self.crop_x.set(float(snap.crop_x))
                self.crop_y.set(float(snap.crop_y))
                self.crop_zoom.set(float(clamp_zoom(snap.crop_zoom)))
                self._sync_crop_bounds(clamp=True)
                self._sync_crop_entry_text()
            finally:
                self._crop_updating = False
            self._tess_updating = True
            try:
                self.tess_mode.set(normalize_tess_mode(getattr(snap, "tess_mode", MODE_DEFAULT)))
                self._sync_tess_mode_combo()
                self.tess_h.set(normalize_h_side(snap.tess_h))
                self.tess_v.set(normalize_v_side(snap.tess_v))
                built = bool(getattr(snap, "tess_built", False))
                if not built:
                    built = coerce_built(getattr(snap, "tess_strength", 0.0))
                self.tess_built.set(built)
                self.tess_tiles.set(float(clamp_tiles(getattr(snap, "tess_tiles", TILES_DEFAULT))))
                self.tess_lloyd.set(float(clamp_lloyd(getattr(snap, "tess_lloyd", LLOYD_DEFAULT))))
                self.tess_normalize.set(
                    coerce_normalize_lighting(getattr(snap, "tess_normalize", False))
                )
                self._lighting_auto_darks = float(
                    getattr(snap, "lighting_auto_darks", 0.0)
                )
                self._lighting_auto_lights = float(
                    getattr(snap, "lighting_auto_lights", 0.0)
                )
                self._sync_tess_mosaic_text()
                self._tess_committed = self._tess_params()
                self._sync_tess_mosaic_controls()
            finally:
                self._tess_updating = False
            self._inpaint_boxes = list(getattr(snap, "inpaint_boxes", ()) or ())
            self._inpaint_quads = list(getattr(snap, "inpaint_quads", ()) or ())
            self._inpaint_layer_id = str(getattr(snap, "inpaint_layer_id", "") or "")
            self._label_updating = True
            try:
                self.label_text.set(str(getattr(snap, "label_text", "") or ""))
                size = clamp_label_size(getattr(snap, "label_size", LABEL_SIZE_DEFAULT))
                self.label_size.set(float(size))
                self.label_size_text.set(str(size))
                self.label_color.set(
                    str(getattr(snap, "label_color", LABEL_COLOR_DEFAULT) or LABEL_COLOR_DEFAULT)
                )
                self.label_x.set(str(int(getattr(snap, "label_x", 0))))
                self.label_y.set(str(int(getattr(snap, "label_y", 0))))
                self.label_font.set(
                    str(getattr(snap, "label_font", LABEL_FONT_DEFAULT) or LABEL_FONT_DEFAULT)
                )
            finally:
                self._label_updating = False
            self._restore_layer_stack(
                getattr(snap, "layers", ()) or (),
                getattr(snap, "selected_layer_ids", ()) or (),
            )
            self.range_map = build_range_map(
                self.work_image,
                snap.range_count,
                method,
                bin_start=getattr(snap, "bin_start", None),
                min_coverage=clamp_min_coverage(getattr(snap, "min_coverage", MIN_COVERAGE)),
            )
            for i, band in enumerate(self.range_map.ranges):
                if i < len(snap.match_rgbs):
                    band.match_rgb = snap.match_rgbs[i]
                if i < len(snap.replacements):
                    band.replacement_rgb = snap.replacements[i]
                    band.name = snap.names[i]
                    band.visible = snap.visibilities[i]
            if is_color_split(method):
                sync_centers_from_match(self.range_map)
            if len(snap.weights) == len(self.range_map.ranges):
                apply_weights(self.range_map, list(snap.weights))
                for i, band in enumerate(self.range_map.ranges):
                    if i < len(snap.visibilities):
                        band.visible = snap.visibilities[i]
                    if i < len(snap.match_rgbs):
                        band.match_rgb = snap.match_rgbs[i]
                    if i < len(snap.replacements):
                        band.replacement_rgb = snap.replacements[i]
                        band.name = snap.names[i]
            self.range_map.texture_strength = snap.texture_strength
            self.range_map.texture_enabled = snap.texture_enabled
            self.range_map.tone_darks = snap.tone_darks
            self.range_map.tone_lights = snap.tone_lights
            self.range_map.tone_brightness = snap.tone_brightness
            self.range_map.tone_contrast = float(getattr(snap, "tone_contrast", 0.0))
            self.range_map.tone_exposure = float(getattr(snap, "tone_exposure", 0.0))
            self.range_map.tone_lights_reds = float(getattr(snap, "tone_lights_reds", 0.0))
            self.range_map.tone_lights_greens = float(getattr(snap, "tone_lights_greens", 0.0))
            self.range_map.tone_lights_blues = float(getattr(snap, "tone_lights_blues", 0.0))
            bal_c = float(
                getattr(
                    snap,
                    "tone_balance_cyan",
                    getattr(snap, "tone_lights_cyan", 0.0),
                )
            )
            bal_m = float(
                getattr(
                    snap,
                    "tone_balance_magenta",
                    getattr(snap, "tone_lights_magenta", 0.0),
                )
            )
            bal_y = float(
                getattr(
                    snap,
                    "tone_balance_yellow",
                    getattr(snap, "tone_lights_yellow", 0.0),
                )
            )
            self.range_map.tone_balance_cyan = bal_c
            self.range_map.tone_balance_magenta = bal_m
            self.range_map.tone_balance_yellow = bal_y
            self.range_map.tone_lights_cyan = bal_c
            self.range_map.tone_lights_magenta = bal_m
            self.range_map.tone_lights_yellow = bal_y
            self.range_map.tone_darks_cyan = float(getattr(snap, "tone_darks_cyan", 0.0))
            self.range_map.tone_darks_magenta = float(getattr(snap, "tone_darks_magenta", 0.0))
            self.range_map.tone_darks_yellow = float(getattr(snap, "tone_darks_yellow", 0.0))
            self.range_map.tone_temperature = float(getattr(snap, "tone_temperature", 0.0))
            self.range_map.tone_tint = float(getattr(snap, "tone_tint", 0.0))
            self.range_map.tone_saturation = float(getattr(snap, "tone_saturation", 0.0))
            n_ranges = len(self.range_map.ranges)
            snap_sel = int(snap.selected_index)
            if snap_sel < 0:
                self.selected_index = -1
            else:
                self.selected_index = max(0, min(snap_sel, n_ranges - 1))
            self.selected_half = (
                snap.selected_half if snap.selected_half in (HALF_MATCH, HALF_REPLACE) else HALF_REPLACE
            )
            self._rebuild_chips()
            self._load_selected_onto_wheel()
            self._sync_range_widgets(update_bar=True)
            match = get_preset(snap.preset_id) if snap.preset_id else None
            self.preset_choice.set(match.name if match is not None else GENERIC_LABEL)
            self._sync_preset_buttons()
            raw_icc = getattr(snap, "icc_path", None)
            self.icc_path = Path(raw_icc) if raw_icc else None
            self._sync_icc_button()
        finally:
            self._mute_ui = False
            self._history_lock = False
        self._sync_range_layers()
        self._refresh_layers_panel()
        self._sync_bin_spins_from_map()
        self._refresh_now()
