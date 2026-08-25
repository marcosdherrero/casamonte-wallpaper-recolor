# -*- coding: utf-8 -*-
"""
wallpaper_recolor.ui.mixins.chrome
------------------------------
Window chrome: File/Edit/View/Tools/Help, View Move vs Grab Move, close-save
prompt (``.wpedit``), and the footer busy bar for full-res save/export.

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
    source_xy_to_display,
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


class AppChromeMixin:
    """Window chrome, Tools (View Move / Grab Move), close-save, busy."""

    # ---------------------------------------------------------------------------
    # Native menubar (File / Edit / View / Tools / Help)
    # ---------------------------------------------------------------------------
    def _build_menubar(self) -> None:
        """Native ``root`` menu — works even when a dock pane is hidden."""
        self.menubar = tk.Menu(self.root, tearoff=0)
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.edit_menu = tk.Menu(self.menubar, tearoff=0)
        self.view_menu = tk.Menu(self.menubar, tearoff=0)
        self.tools_menu = tk.Menu(self.menubar, tearoff=0)
        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self.text_menu = tk.Menu(self.edit_menu, tearoff=0)
        self.layout_profiles_menu = tk.Menu(self.view_menu, tearoff=0)
        self.layout_delete_menu = tk.Menu(self.layout_profiles_menu, tearoff=0)
        self._build_file_menu()
        self.help_menu.add_command(label="About Wallpaper Recolor", command=self._show_about)
        self.menubar.add_cascade(label="File", menu=self.file_menu)
        self.menubar.add_cascade(label="Edit", menu=self.edit_menu)
        self.menubar.add_cascade(label="View", menu=self.view_menu)
        self.menubar.add_cascade(label="Tools", menu=self.tools_menu)
        self.menubar.add_cascade(label="Help", menu=self.help_menu)
        self.root.config(menu=self.menubar)
        self.edit_menu.configure(postcommand=self._rebuild_edit_menu)
        self.view_menu.configure(postcommand=self._rebuild_view_menu)
        self.tools_menu.configure(postcommand=self._rebuild_tools_menu)
        self._rebuild_tools_menu()

    # ---------------------------------------------------------------------------
    # File menu
    # ---------------------------------------------------------------------------
    def _build_file_menu(self) -> None:
        menu = self.file_menu
        menu.delete(0, "end")
        menu.add_command(label="Open image…", command=self.open_image, accelerator="Ctrl+O")
        menu.add_command(label="Save as…", command=self.save_image_as, accelerator="Ctrl+S")
        menu.add_command(label="Export job pack…", command=self.export_pack, accelerator="Ctrl+E")
        menu.add_command(label="Export layers zip…", command=self.export_layers_zip)
        menu.add_separator()
        menu.add_command(label="Save Wallpaper Edit state…", command=self.save_edit_state)
        menu.add_command(label="Open Edit state…", command=self.load_edit_state)
        menu.add_separator()
        menu.add_command(label="Exit", command=self._on_app_close)

    def _rebuild_edit_menu(self) -> None:
        """One-shot panel actions plus Undo/Redo with remaining-step counts."""
        menu = getattr(self, "edit_menu", None)
        if menu is None:
            return
        undo_n = len(getattr(self, "_undo_stack", []) or [])
        redo_n = len(getattr(self, "_redo_stack", []) or [])
        menu.delete(0, "end")
        menu.add_command(
            label=_history_menu_label("Undo", undo_n),
            command=self.undo_edit,
            accelerator="Ctrl+Z" if not undo_n else "",
            state="normal" if undo_n else "disabled",
        )
        menu.add_command(
            label=_history_menu_label("Redo", redo_n),
            command=self.redo_edit,
            accelerator="Ctrl+Y" if not redo_n else "",
            state="normal" if redo_n else "disabled",
        )
        menu.add_separator()
        menu.add_command(label="Reset colors", command=self.reset_colors)
        menu.add_command(label="Normalize lighting", command=self._on_tess_normalize)
        menu.add_command(label="Tessellate Build", command=self._on_tess_build)
        menu.add_separator()
        text_menu = getattr(self, "text_menu", None)
        if text_menu is None:
            text_menu = tk.Menu(menu, tearoff=0)
            self.text_menu = text_menu
        text_menu.delete(0, "end")
        text_menu.add_command(label="Detect", command=self._on_label_detect)
        text_menu.add_command(label="Remove", command=self._on_label_remove)
        text_menu.add_command(label="Clear", command=self._on_label_clear)
        text_menu.add_command(label="Mark", command=self._on_label_mark_toggle)
        text_menu.add_command(label="Place", command=self._on_label_place_toggle)
        text_menu.add_command(label="Change-to", command=self._label_use_change_to)
        menu.add_cascade(label="Text", menu=text_menu)

    def _sync_edit_history_labels(self) -> None:
        self._rebuild_edit_menu()
        self._refresh_history_list()

    def _refresh_history_list(self) -> None:
        """Newest timeline position at top: Redo N … Current … Undo 1 … Undo N."""
        box = getattr(self, "history_list", None)
        if box is None:
            return
        rows: list[tuple[str, str, int]] = []
        redo = list(getattr(self, "_redo_stack", []) or [])
        undo = list(getattr(self, "_undo_stack", []) or [])
        for i, _snap in enumerate(reversed(redo)):
            n = len(redo) - i
            rows.append((f"Redo {n}", "redo", n))
        rows.append(("Current", "current", 0))
        for i, _snap in enumerate(reversed(undo)):
            rows.append((f"Undo {i + 1}", "undo", i + 1))
        self._history_rows = [(kind, n) for _label, kind, n in rows]
        self._history_list_mute = True
        try:
            box.delete(0, "end")
            current_i = 0
            for i, (label, kind, _n) in enumerate(rows):
                box.insert("end", label)
                if kind == "current":
                    current_i = i
            if rows:
                box.selection_clear(0, "end")
                box.selection_set(current_i)
                box.see(current_i)
        except tk.TclError:
            pass
        finally:
            self._history_list_mute = False

    def _on_history_list_select(self, _event=None) -> None:
        if getattr(self, "_history_list_mute", False):
            return
        box = getattr(self, "history_list", None)
        if box is None:
            return
        sel = box.curselection()
        if not sel:
            return
        index = int(sel[0])
        rows = getattr(self, "_history_rows", [])
        if index < 0 or index >= len(rows):
            return
        kind, n = rows[index]
        if kind == "current" or n <= 0:
            return
        if kind == "undo":
            for _ in range(int(n)):
                self.undo_edit()
        elif kind == "redo":
            for _ in range(int(n)):
                self.redo_edit()

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About Wallpaper Recolor",
            "Wallpaper Recolor remaps wallpaper color ranges — match-from / "
            "change-to clusters — so a pattern can be recolored for a job.",
            parent=self.root,
        )

    # ---------------------------------------------------------------------------
    # Pointer tools: View Move (camera) vs Grab Move (layer offset / pan)
    # ---------------------------------------------------------------------------
    def _set_pointer_tool(self, tool: str) -> None:
        key = str(tool or TOOL_VIEW_MOVE)
        if key not in POINTER_LABEL_FOR:
            key = POINTER_TOOL_BY_LABEL.get(key, TOOL_VIEW_MOVE)
        self.pointer_tool.set(key)
        label = POINTER_LABEL_FOR[key]
        var = getattr(self, "pointer_tool_label", None)
        if var is not None:
            try:
                if str(var.get()) != label:
                    var.set(label)
            except (tk.TclError, TypeError, ValueError):
                var.set(label)
        self._sync_label_modes()
        for host in self._preview_zoom_hosts():
            try:
                host._sync_host_cursor()
            except tk.TclError:
                pass

    def _view_move_on(self) -> bool:
        return self._pointer_tool_key() == TOOL_VIEW_MOVE

    def _grab_move_on(self) -> bool:
        return self._pointer_tool_key() == TOOL_GRAB_MOVE

    def _pointer_tool_key(self) -> str:
        var = getattr(self, "pointer_tool", None)
        if var is None:
            return TOOL_VIEW_MOVE
        try:
            key = str(var.get() or TOOL_VIEW_MOVE)
        except (tk.TclError, TypeError, ValueError):
            return TOOL_VIEW_MOVE
        if key in POINTER_LABEL_FOR:
            return key
        return POINTER_TOOL_BY_LABEL.get(key, TOOL_VIEW_MOVE)

    def _grab_tool_on(self) -> bool:
        return self._grab_move_on()

    def _grab_target_image(self):
        """Layer Grab Move offsets: selected Image/Label, or parent of a Color range row."""
        stack = getattr(self, "layer_stack", None)
        if stack is None:
            return None
        ly = stack.primary()
        if ly is not None and ly.is_label():
            return ly
        img = stack.image_for_selection(ly)
        return img if img is not None else stack.base_layer()

    def _grab_moves_layer(self, host=None) -> bool:
        """Grab Move repositions a non-base Image or Label. Base pans the camera."""
        if not self._grab_move_on():
            return False
        if self._label_mark_mode or self._label_place_mode:
            return False
        target = self._grab_target_image()
        if target is None:
            return False
        if target.is_label():
            return True
        return bool(target.is_image() and not target.is_base())

    def _nudge_grab_move(self, dx: int, dy: int, *, host=None) -> None:
        """Drag on Composite: overlay / label x/y. Base pan is handled by the host."""
        target = self._grab_target_image()
        if target is None or target.is_base():
            return
        self._nudge_layer_offset(target, dx, dy, host=host)

    def _nudge_selected_layer(self, dx: int, dy: int, *, host=None) -> None:
        target = self._grab_target_image()
        if target is None or target.is_base():
            return
        self._nudge_layer_offset(target, dx, dy, host=host)

    def _nudge_layer_offset(self, ly, dx: int, dy: int, *, host=None) -> None:
        if ly is None:
            return
        src = self._orig_pil if self._orig_pil is not None else self.work_image
        if src is None:
            return
        if self._layer_drag_before is None:
            self._layer_drag_before = self._capture_edit()
        photo_w = 1
        if host is not None and getattr(host, "_photo", None) is not None:
            try:
                photo_w = max(1, int(host._photo.width()))
            except (tk.TclError, TypeError, ValueError):
                photo_w = max(1, src.size[0])
        else:
            photo_w = max(1, src.size[0])
        iw = max(1, src.size[0])
        cx, cy, cz = self._crop_xy_zoom()
        _dx, _dy, sc = source_xy_to_display(
            0, 0, src.size, self._crop_src_size(), cx, cy, cz
        )
        view_sc = photo_w / float(iw)
        step = max(sc * view_sc, 1e-6)
        ly.x = int(round(ly.x + float(dx) / step))
        ly.y = int(round(ly.y + float(dy) / step))
        if ly.is_label():
            self._sync_label_fields_from_layer(ly)
        self._refresh_previews()

    def _finish_layer_drag(self) -> None:
        before = self._layer_drag_before
        self._layer_drag_before = None
        self._push_undo_state(before)

    def _rebuild_tools_menu(self) -> None:
        """File / Edit / View / Tools / Help — View Move and Grab Move stay in sync."""
        menu = getattr(self, "tools_menu", None)
        if menu is None:
            return
        menu.delete(0, "end")
        current = TOOL_VIEW_MOVE
        try:
            current = str(self.pointer_tool.get() or TOOL_VIEW_MOVE)
        except (tk.TclError, TypeError, ValueError):
            pass
        if current not in POINTER_LABEL_FOR:
            current = TOOL_VIEW_MOVE
        menu.add_radiobutton(
            label=TOOL_VIEW_MOVE_LABEL,
            variable=self.pointer_tool,
            value=TOOL_VIEW_MOVE,
            command=lambda: self._set_pointer_tool(TOOL_VIEW_MOVE),
        )
        menu.add_radiobutton(
            label=TOOL_GRAB_MOVE_LABEL,
            variable=self.pointer_tool,
            value=TOOL_GRAB_MOVE,
            command=lambda: self._set_pointer_tool(TOOL_GRAB_MOVE),
        )
        del current

    def _on_tools_combo(self) -> None:
        label = ""
        try:
            label = str(self.pointer_tool_label.get() or "")
        except (tk.TclError, TypeError, ValueError):
            label = ""
        self._set_pointer_tool(POINTER_TOOL_BY_LABEL.get(label, TOOL_VIEW_MOVE))

    # ---------------------------------------------------------------------------
    # Close: Yes/No/Cancel save .wpedit when dirty (Cancel or failed save stays open)
    # ---------------------------------------------------------------------------
    def _on_app_close(self) -> None:
        """Ask to save Edit state when dirty, then dock floaters and destroy."""
        if getattr(self, "_closing", False):
            return
        if self._busy:
            messagebox.showinfo(
                "Busy",
                "Wait for the current job to finish, or cancel it, before closing.",
                parent=self.root,
            )
            return
        if self._edit_state_is_dirty():
            answer = messagebox.askyesnocancel(
                "Save Edit state?",
                "Save the current Wallpaper Edit state before closing?",
                parent=self.root,
            )
            if answer is None:
                return
            if answer:
                if not self.save_edit_state(path=self._edit_state_path):
                    return
        self._destroy_app_window()

    def _destroy_app_window(self) -> None:
        """Dock floaters first so wm-managed frames tear down with the root."""
        self._closing = True
        self._cancel_preview_job()
        try:
            if getattr(self, "busy_progress", None) is not None:
                self.busy_progress.stop()
                try:
                    self.busy_progress.configure(value=0)
                except tk.TclError:
                    pass
                self.busy_progress.place_forget()
        except tk.TclError:
            pass
        for panel in self._all_panels:
            if panel.is_floating:
                try:
                    panel.dock()
                except tk.TclError:
                    pass
        self.root.destroy()

    def _bind_readonly_combo(self, combo: ttk.Combobox, on_select) -> None:
        """Readonly ttk: drop Windows highlight after pick; FocusOut clears leftover selection.

        ``<<ComboboxSelected>>`` fires after the list closes, so focusing root is safe then.
        Opening the dropdown must not steal focus — FocusOut only clears selection.
        """
        def _on_selected(_event=None) -> None:
            try:
                on_select()
            finally:
                self._defocus_readonly_combo(combo)

        combo.bind("<<ComboboxSelected>>", _on_selected)
        combo.bind("<FocusOut>", lambda _e, c=combo: self._clear_combo_selection(c))

    def _clear_combo_selection(self, combo: ttk.Combobox) -> None:
        try:
            combo.selection_clear()
        except tk.TclError:
            pass

    def _combobox_popdown_mapped(self) -> bool:
        """True while a ttk Combobox list Toplevel is showing."""
        try:
            for name in self.root.tk.call("winfo", "children", "."):
                try:
                    cls = str(self.root.tk.call("winfo", "class", name))
                    mapped = int(self.root.tk.call("winfo", "ismapped", name))
                except (tk.TclError, TypeError, ValueError):
                    continue
                if mapped and (cls == "ComboboxPopdown" or "popdown" in cls.lower()):
                    return True
        except tk.TclError:
            pass
        return False

    def _defocus_readonly_combo(self, combo: ttk.Combobox) -> None:
        """Clear the blue selection and move focus off the combo (Windows readonly ttk)."""
        self._clear_combo_selection(combo)
        if self._combobox_popdown_mapped():
            return

        def _after_pick() -> None:
            # Vista/Win ttk often re-selects the text after ComboboxSelected returns
            self._clear_combo_selection(combo)
            if self._combobox_popdown_mapped():
                return
            try:
                self.root.focus_set()
            except tk.TclError:
                pass

        try:
            combo.after_idle(_after_pick)
        except tk.TclError:
            pass

    def _schedule_raise_chrome(self) -> None:
        """Lift toolbar/status after layout, not on every wheel tick.

        ``lift()`` on each scroll steals the next MouseWheel on Windows (the
        header HWND becomes topmost). Layout/dock is the right time to raise
        chrome — do it now, not via after_idle (that callback dies with the
        widget and prints a Tcl error when tests call root.destroy()).
        """
        if getattr(self, "_closing", False):
            return
        self._raise_window_chrome()

    def _raise_window_chrome(self) -> None:
        """Keep the toolbar and status bar above docked panes.

        DockablePanel shells are children of each scroller, so they clip. Lift
        toolbar/status so they stay above any leftover chrome.
        """
        toolbar = getattr(self, "toolbar", None)
        footer = getattr(self, "footer", None)
        busy = getattr(self, "busy_bar", None)
        status = getattr(self, "status_bar", None)
        try:
            if toolbar is not None:
                toolbar.lift()
            if footer is not None:
                footer.lift()
            if busy is not None:
                busy.lift()
            if status is not None:
                status.lift()
        except tk.TclError:
            pass

    def _set_busy(self, busy: bool, status: str | None = None) -> None:
        """Disable file actions while a worker remaps/saves. Tk widgets only."""
        self._busy = busy
        state = "disabled" if busy else "normal"
        self.open_btn.configure(state=state)
        file_menu = getattr(self, "file_menu", None)
        if file_menu is not None:
            for label in (
                "Open image…",
                "Save as…",
                "Export job pack…",
                "Export layers zip…",
            ):
                try:
                    file_menu.entryconfigure(label, state=state)
                except tk.TclError:
                    pass
        if status is not None:
            self.status.set(status)
            self.busy_caption.set(status)
        self._set_busy_indicator(busy)
        self._sync_busy_cancel()

    def _on_busy_cancel(self) -> None:
        ev = getattr(self, "_job_cancel", None)
        if ev is not None:
            ev.set()
        self.status.set("Cancelling…")
        self.busy_caption.set("Cancelling…")

    def _sync_busy_cancel(self) -> None:
        btn = getattr(self, "busy_cancel", None)
        bar = getattr(self, "busy_bar", None)
        if btn is None or bar is None:
            return
        show = bool(self._busy) and bool(getattr(self, "_job_cancellable", False))
        try:
            if show:
                btn.pack(side="right", padx=(4, 0), pady=2)
            else:
                btn.pack_forget()
        except tk.TclError:
            pass

    def _set_busy_indicator(self, busy: bool) -> None:
        """Start/stop the compact footer bar. The 200px slot stays packed.

        ``start()`` / ``stop()`` are Tk-thread only. The encode stays on the
        background thread — this frame must not grab the event loop. Idle
        unmaps the Progressbar (blank slot, no green stub) so the footer
        height never jumps. Status copy stays in the bottom caption, not a
        header banner.
        """
        try:
            if busy:
                try:
                    self.busy_progress.configure(state="normal")
                except tk.TclError:
                    pass
                if str(self.busy_progress.winfo_manager()) != "place":
                    self.busy_progress.place(x=0, y=0, relwidth=1, relheight=1)
                self.busy_progress.start(16)
                self.root.config(cursor="watch")
                self._raise_window_chrome()
            else:
                self.busy_progress.stop()
                try:
                    self.busy_progress.configure(value=0)
                except tk.TclError:
                    pass
                self.busy_progress.place_forget()
                self.root.config(cursor="")
                try:
                    self.busy_progress.configure(state="disabled")
                except tk.TclError:
                    pass
        except tk.TclError:
            pass

    def _apply_progress_status(self, msg: str) -> None:
        """Status bar (Tk thread). Export ``on_status`` uses this."""
        self.status.set(msg)
        if self._busy:
            self.busy_caption.set(msg)

    def _status_from_worker(self, msg: str) -> None:
        """Marshal export progress onto the Tk thread."""
        try:
            self.root.after(0, lambda m=msg: self._apply_progress_status(m))
        except tk.TclError:
            pass

    def _run_background(
        self,
        status_msg: str,
        work,
        on_ok,
        on_err,
        *,
        cancellable: bool = False,
        cancel_status: str = "Cancelled.",
    ) -> None:
        """Run ``work`` on a daemon thread; finish on the Tk thread via after().

        Never touch Tk widgets from the worker (Windows "Not Responding" was
        remap + LZW TIFF on the mainloop thread). Poll ``after`` from this
        thread so tests without mainloop can pump ``update()``.
        """
        if self._busy:
            return
        self._job_cancel = threading.Event()
        self._job_cancellable = bool(cancellable)
        self._job_cancel_status = str(cancel_status or "Cancelled.")
        self._set_busy(True, status_msg)
        box: dict = {"result": None, "err": None, "done": False, "cancelled": False}

        def worker() -> None:
            try:
                box["result"] = work()
            except InterruptedError:
                box["cancelled"] = True
            except Exception as exc:  # noqa: BLE001 — surface any save/export failure
                box["err"] = exc
            finally:
                box["done"] = True

        def poll() -> None:
            if not box["done"]:
                try:
                    self.root.after(16, poll)
                except tk.TclError:
                    try:
                        self._set_busy(False)
                    except tk.TclError:
                        pass
                return
            try:
                if box.get("cancelled"):
                    self.status.set(self._job_cancel_status)
                    self.busy_caption.set(self._job_cancel_status)
                elif box["err"] is not None:
                    on_err(box["err"])
                else:
                    on_ok(box["result"])
            finally:
                self._job_cancellable = False
                self._set_busy(False)

        threading.Thread(target=worker, daemon=True).start()
        try:
            self.root.after(0, poll)
        except tk.TclError:
            self._job_cancellable = False
            self._set_busy(False)
