# -*- coding: utf-8 -*-
"""
wallpaper_recolor.ui.widgets
------------------------------
Scale bindings (wheel pages the column, does not nudge ttk.Scale), ToneKnob,
EyeToggle, RangeChip.

ToneKnob: relative drag (up/right +, down/left −); gain grows with distance
from the knob so near is fine and far is faster.

Class references (code + name only):
- CAP3321C Data Wrangling
- CAP4631C Machine Learning
- CAP4633C Machine Learning 2
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import math
import tkinter as tk
from tkinter import ttk

from PIL import ImageTk

from wallpaper_recolor.color.color_ranges import ColorRangeMap
from wallpaper_recolor.color.tone import TONE_SLIDER_MAX, TONE_SLIDER_MIN
from wallpaper_recolor.ui.tooltip import bind_tooltip
from wallpaper_recolor.transform.crop import clamp_zoom
from wallpaper_recolor.ui.constants import (
    TONE_KNOB_BASE_RATE,
    TONE_KNOB_GROWTH,
    TONE_KNOB_MIN_DEG,
    TONE_KNOB_PX,
    TONE_KNOB_REF_PX,
    TONE_KNOB_STEP,
    TONE_KNOB_SWEEP_DEG,
)

if TYPE_CHECKING:
    from wallpaper_recolor.ui.app import WallpaperRecolorApp

def _bind_tree(widget: tk.Misc, sequence: str, handler) -> None:
    """Bind ``sequence`` on ``widget`` and every child (chips are nested labels)."""
    widget.bind(sequence, handler)
    for child in widget.winfo_children():
        _bind_tree(child, sequence, handler)


def _bind_wheel_tree(widget: tk.Misc, handler) -> None:
    """Bind MouseWheel on ``widget`` and every child — page-scroll only.

    ttk.Scale on Windows swallows the wheel to nudge the value. The widget bind
    always returns ``break`` so the Scale default never runs; the handler
    scrolls the hovered column instead.
    """

    def _page_scroll(event, h=handler):
        result = h(event)
        return "break" if result is None else result

    if not getattr(widget, "_wp_col_wheel", False):
        widget.bind("<MouseWheel>", _page_scroll, add="+")
        widget.bind("<Button-4>", _page_scroll, add="+")
        widget.bind("<Button-5>", _page_scroll, add="+")
        widget._wp_col_wheel = True  # type: ignore[attr-defined]
    for child in widget.winfo_children():
        _bind_wheel_tree(child, handler)


# Coalesce auto-repeat KeyRelease into one undo tick (Windows repeats both)
_SCALE_KEY_UNDO_MS = 80


def _scale_lo_hi(scale: tk.Misc, from_=None, to_=None) -> tuple[float, float]:
    """Live Scale range (widget ``from``/``to``). Crop X/Y max follows zoom."""
    lo = 0.0 if from_ is None else float(from_)
    hi = 1.0 if to_ is None else float(to_)
    try:
        lo = float(scale.cget("from"))
    except (tk.TclError, TypeError, ValueError):
        pass
    try:
        hi = float(scale.cget("to"))
    except (tk.TclError, TypeError, ValueError):
        pass
    return lo, hi


def _quantize_scale_value(value: float, step: float, lo: float, hi: float) -> float:
    """Clamp ``value`` onto ``step`` ticks between ``lo`` and ``hi``."""
    if lo > hi:
        lo, hi = hi, lo
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = lo
    step = abs(float(step)) if step else 0.0
    if step > 0:
        n = round((value - lo) / step)
        value = lo + n * step
    return max(lo, min(hi, value))


def _invoke_scale_command(scale: tk.Misc, value: float) -> None:
    """Fire the same ``command`` ttk.Scale uses while dragging."""
    try:
        cmd = scale.cget("command")
    except tk.TclError:
        return
    if not cmd:
        return
    try:
        scale.tk.call(cmd, str(value))
    except tk.TclError:
        pass


def _bind_smooth_scale(scale, var, step=1, from_=None, to_=None, *, on_begin=None, on_end=None) -> None:
    """Stop Windows trough-jump; drag from the current value; arrows ±step.

    ttk.Scale Button-1 on the trough *jumps* to that fraction (often 0 or
    100 when the thumb is missed). Widget binds return ``break`` so the
    class Jump never runs. Press starts a drag from the live value; motion
    tracks pointer delta (quantized to ``step``). MouseWheel is not bound
    here — ``_bind_wheel_tree`` still page-scrolls and returns ``break``.

    Focus on click so Left/Right/Up/Down nudge after a mouse grab. One
    undo tick on mouse release, and one on arrow KeyRelease (debounced
    so auto-repeat is not a flood).
    """

    step = float(step) if step else 1.0
    state = {"dragging": False, "x": 0.0, "v": 0.0, "key_job": None}

    def _hook(cb, event=None) -> None:
        if cb is None:
            return
        try:
            cb(event)
        except TypeError:
            cb()

    def _apply(value: float) -> None:
        lo, hi = _scale_lo_hi(scale, from_, to_)
        new = _quantize_scale_value(value, step, lo, hi)
        try:
            old = float(var.get())
        except (tk.TclError, TypeError, ValueError):
            old = None
        eps = max(1e-9, abs(step) * 1e-6)
        if old is not None and abs(old - new) <= eps:
            return
        try:
            var.set(float(new))
        except (tk.TclError, TypeError, ValueError):
            return
        _invoke_scale_command(scale, new)

    def _on_press(event) -> str:
        try:
            scale.focus_set()
        except tk.TclError:
            pass
        state["dragging"] = True
        state["x"] = float(getattr(event, "x", 0) or 0)
        try:
            state["v"] = float(var.get())
        except (tk.TclError, TypeError, ValueError):
            lo, _hi = _scale_lo_hi(scale, from_, to_)
            state["v"] = lo
        _hook(on_begin, event)
        return "break"

    def _on_motion(event) -> str:
        if not state["dragging"]:
            return "break"
        lo, hi = _scale_lo_hi(scale, from_, to_)
        try:
            width = max(1, int(scale.winfo_width()))
        except (tk.TclError, TypeError, ValueError):
            width = 1
        dx = float(getattr(event, "x", 0) or 0) - state["x"]
        _apply(state["v"] + (dx / width) * (hi - lo))
        return "break"

    def _on_release(event) -> str:
        state["dragging"] = False
        _hook(on_end, event)
        return "break"

    def _nudge(sign: int, event=None) -> str:
        _hook(on_begin, event)
        try:
            current = float(var.get())
        except (tk.TclError, TypeError, ValueError):
            current = 0.0
        _apply(current + sign * step)
        _schedule_key_end()
        return "break"

    def _schedule_key_end() -> None:
        job = state["key_job"]
        if job is not None:
            try:
                scale.after_cancel(job)
            except (tk.TclError, ValueError):
                pass

        def _finish() -> None:
            state["key_job"] = None
            _hook(on_end)

        try:
            state["key_job"] = scale.after(_SCALE_KEY_UNDO_MS, _finish)
        except tk.TclError:
            _hook(on_end)

    def _on_key_release(_event) -> str:
        _schedule_key_end()
        return "break"

    try:
        scale.configure(takefocus=True)
    except tk.TclError:
        pass
    scale.bind("<ButtonPress-1>", _on_press)
    scale.bind("<B1-Motion>", _on_motion)
    scale.bind("<ButtonRelease-1>", _on_release)
    scale.bind("<Left>", lambda e: _nudge(-1, e))
    scale.bind("<Down>", lambda e: _nudge(-1, e))
    scale.bind("<Right>", lambda e: _nudge(1, e))
    scale.bind("<Up>", lambda e: _nudge(1, e))
    scale.bind("<KeyRelease-Left>", _on_key_release)
    scale.bind("<KeyRelease-Down>", _on_key_release)
    scale.bind("<KeyRelease-Right>", _on_key_release)
    scale.bind("<KeyRelease-Up>", _on_key_release)
    # Withdrawn tests: KeyPress goes to focus (often None); this is the Left/Right handler
    scale._smooth_nudge = _nudge  # type: ignore[attr-defined]

def tone_knob_gain(distance_px: float) -> float:
    """Units per motion pixel; grows exponentially with distance from the knob."""
    dist = max(0.0, float(distance_px))
    return TONE_KNOB_BASE_RATE * (TONE_KNOB_GROWTH ** (dist / TONE_KNOB_REF_PX))


class ToneKnob(tk.Canvas):
    """Relative dial for a −100…+100 Color & lighting DoubleVar.

    Hold button 1 and drag up/right to increase, down/left to decrease.
    Gain is an exponential of Euclidean distance from the knob center
    (near = fine, far = faster). Spinbox arrows stay ±1. Needle is a readout.
    """

    def __init__(
        self,
        parent: tk.Misc,
        var: tk.DoubleVar,
        *,
        on_change,
        on_begin=None,
        on_end=None,
        from_: float = TONE_SLIDER_MIN,
        to_: float = TONE_SLIDER_MAX,
        size: int = TONE_KNOB_PX,
        tooltip: str = "",
    ) -> None:
        bg = ttk.Style().lookup("TFrame", "background") or "#f0f0f0"
        super().__init__(
            parent,
            width=size,
            height=size,
            highlightthickness=0,
            bd=0,
            bg=bg,
            cursor="hand2",
            takefocus=0,
        )
        self.var = var
        self.on_change = on_change
        self.on_begin = on_begin
        self.on_end = on_end
        self.from_ = float(from_)
        self.to_ = float(to_)
        self._size = int(size)
        self._dragging = False
        self._mute = False
        self._frac = 0.0
        self._last_root_x = 0.0
        self._last_root_y = 0.0
        self._all_motion_id = ""
        self._all_release_id = ""
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self._trace = var.trace_add("write", self._on_var)
        self.bind("<Destroy>", self._on_destroy)
        if tooltip:
            bind_tooltip(self, tooltip)
        self.redraw()

    def _on_destroy(self, _event=None) -> None:
        if self._dragging:
            self._teardown_drag()
            self._dragging = False
        if self._trace:
            try:
                self.var.trace_remove("write", self._trace)
            except tk.TclError:
                pass
            self._trace = ""

    def _on_var(self, *_args) -> None:
        self.redraw()

    def current_value(self) -> float:
        try:
            return float(self.var.get())
        except (tk.TclError, ValueError, TypeError):
            return 0.0

    def set_value(self, value: float, *, notify: bool = True) -> None:
        """Write the DoubleVar (clamped, rounded) and optionally fire on_change."""
        lo, hi = self.from_, self.to_
        q = float(TONE_KNOB_STEP)
        val = round(float(value) / q) * q
        val = max(lo, min(hi, val))
        self._mute = True
        try:
            self.var.set(float(val))
        finally:
            self._mute = False
        self.redraw()
        if notify and self.on_change is not None:
            self.on_change()

    def apply_drag_delta(
        self,
        dy_px: float,
        distance_px: float,
        *,
        dx_px: float = 0.0,
        notify: bool = True,
    ) -> float:
        """Apply one relative step. Tk: +dy down, +dx right. Up/right increase."""
        motion = -float(dy_px) + float(dx_px)
        self._frac += motion * tone_knob_gain(distance_px)
        q = float(TONE_KNOB_STEP)
        n = int(math.trunc(self._frac / q))
        if n == 0:
            return 0.0
        applied = n * q
        self._frac -= applied
        self.set_value(self.current_value() + applied, notify=notify)
        return float(applied)

    def _t_from_value(self, value: float) -> float:
        span = self.to_ - self.from_
        if abs(span) < 1e-9:
            return 0.5
        return max(0.0, min(1.0, (float(value) - self.from_) / span))

    def _angle_deg_from_t(self, t: float) -> float:
        return TONE_KNOB_MIN_DEG - max(0.0, min(1.0, t)) * TONE_KNOB_SWEEP_DEG

    def _center_root(self) -> tuple[float, float]:
        try:
            cx = float(self.winfo_rootx()) + self._size * 0.5
            cy = float(self.winfo_rooty()) + self._size * 0.5
        except tk.TclError:
            cx = self._last_root_x
            cy = self._last_root_y
        return cx, cy

    def _distance_from_knob(self, x_root: float, y_root: float) -> float:
        cx, cy = self._center_root()
        return math.hypot(float(x_root) - cx, float(y_root) - cy)

    def redraw(self) -> None:
        self.delete("all")
        s = self._size
        pad = 1.5
        self.create_oval(
            pad,
            pad,
            s - pad,
            s - pad,
            outline="#666666" if self._dragging else "#888888",
            fill="#ececec" if self._dragging else "#e8e8e8",
            width=1,
        )
        cx = cy = s * 0.5
        if self._dragging:
            self.create_line(cx, pad, cx, s - pad, fill="#c8c8c8", width=1)
            self.create_line(pad, cy, s - pad, cy, fill="#c8c8c8", width=1)
        t = self._t_from_value(self.current_value())
        ang = math.radians(self._angle_deg_from_t(t))
        r = s * 0.5 - 3.5
        nx = cx + r * math.cos(ang)
        ny = cy - r * math.sin(ang)
        self.create_line(cx, cy, nx, ny, fill="#333333", width=2, capstyle="round")
        self.create_oval(cx - 1.5, cy - 1.5, cx + 1.5, cy + 1.5, fill="#333333", outline="")

    def _on_press(self, event) -> None:
        self._dragging = True
        self._frac = 0.0
        self._last_root_x = float(event.x_root)
        self._last_root_y = float(event.y_root)
        self._bind_drag_globals()
        try:
            self.configure(cursor="sb_v_double_arrow")
        except tk.TclError:
            pass
        self.redraw()
        if self.on_begin is not None:
            self.on_begin(event)

    def _on_drag(self, event) -> None:
        self._handle_motion(event)

    def _on_global_motion(self, event) -> None:
        self._handle_motion(event)

    def _handle_motion(self, event) -> None:
        if not self._dragging:
            return
        x_root = float(event.x_root)
        y_root = float(event.y_root)
        dy = y_root - self._last_root_y
        dx = x_root - self._last_root_x
        if dy == 0.0 and dx == 0.0:
            return
        self._last_root_x = x_root
        self._last_root_y = y_root
        self.apply_drag_delta(
            dy, self._distance_from_knob(x_root, y_root), dx_px=dx
        )

    def _on_release(self, event) -> None:
        if not self._dragging:
            return
        self._dragging = False
        self._teardown_drag()
        try:
            self.configure(cursor="hand2")
        except tk.TclError:
            pass
        self.redraw()
        if self.on_end is not None:
            self.on_end(event)

    def _bind_drag_globals(self) -> None:
        self._all_motion_id = self.bind_all("<B1-Motion>", self._on_global_motion, add="+")
        self._all_release_id = self.bind_all("<ButtonRelease-1>", self._on_release, add="+")
        try:
            self.grab_set()
        except tk.TclError:
            pass

    def _teardown_drag(self) -> None:
        try:
            self.grab_release()
        except tk.TclError:
            pass
        self._all_motion_id = self._unbind_all_func("<B1-Motion>", self._all_motion_id)
        self._all_release_id = self._unbind_all_func("<ButtonRelease-1>", self._all_release_id)

    def _unbind_all_func(self, sequence: str, funcid: str) -> str:
        if not funcid:
            return ""
        try:
            raw = self.tk.call("bind", "all", sequence)
            kept = [ln for ln in str(raw).split("\n") if funcid not in ln]
            self.tk.call("bind", "all", sequence, "\n".join(kept))
        except tk.TclError:
            pass
        return ""


class EyeToggle:
    """Clickable FA eye: solid = shown, slash = hidden."""

    def __init__(
        self,
        parent: tk.Misc,
        photos: tuple[ImageTk.PhotoImage, ImageTk.PhotoImage],
        command,
        *,
        bg: str = "#f0f0f0",
        tooltip: str = "",
    ) -> None:
        self.photos = photos
        self.command = command
        self.shown = True
        self.label = tk.Label(
            parent,
            image=photos[0],
            bd=0,
            highlightthickness=0,
            cursor="hand2",
            bg=bg,
            takefocus=0,
        )
        self.label.image = photos[0]  # type: ignore[attr-defined]
        self.label.bind("<Button-1>", self._click)
        self.tooltip = tooltip
        if tooltip:
            bind_tooltip(self.label, tooltip)

    def pack(self, **kw) -> None:
        self.label.pack(**kw)

    def configure_bg(self, bg: str) -> None:
        self.label.configure(bg=bg)

    def set_shown(self, shown: bool) -> None:
        self.shown = bool(shown)
        photo = self.photos[0] if self.shown else self.photos[1]
        self.label.configure(image=photo)
        self.label.image = photo  # type: ignore[attr-defined]

    def _click(self, _event=None) -> None:
        self.set_shown(not self.shown)
        self.command(self.shown)


class RangeChip:
    """Compact range picker: eye, name, click-to-edit %. From/to live on the two bars."""

    def __init__(self, parent: ttk.Frame, app: "WallpaperRecolorApp", index: int) -> None:
        self.app = app
        self.index = index
        self._pct_entry: tk.Entry | None = None
        # tk.Frame so highlightthickness can show the selected range
        self.frame = tk.Frame(parent, padx=6, pady=4, bg="#f5f5f5", cursor="hand2")
        head = tk.Frame(self.frame, bg="#f5f5f5", cursor="hand2")
        head.pack(fill="x")
        self.eye = EyeToggle(
            head,
            app._eye_photos,
            self._on_eye,
            bg="#f5f5f5",
            tooltip="Show or hide this range. Hidden ranges knock out of Result.",
        )
        self.eye.pack(side="left")
        self.title = tk.Label(head, text=f"Range {index + 1}", bg="#f5f5f5", cursor="hand2")
        self.title.pack(side="left", padx=(2, 0))
        self.meta = tk.Label(self.frame, text="", bg="#f5f5f5", fg="#444444", cursor="xterm")
        self.meta.pack(anchor="w")
        bind_tooltip(
            self.meta,
            "Coverage weight for this range. Click to type a percent.",
        )
        _bind_tree(self.title, "<Button-1>", self._click)
        self.frame.bind("<Button-1>", self._click)
        head.bind("<Button-1>", self._click)
        self.meta.bind("<Button-1>", self._on_percent_click)

    def grid(self, row: int, column: int) -> None:
        self.frame.grid(row=row, column=column, sticky="nsew", padx=3, pady=3)

    def refresh(self, range_map: ColorRangeMap, selected: int, selected_half: str) -> None:
        """Name + coverage %; outline the selected range (from/to live on the coverage bars)."""
        _ = selected_half
        band = range_map.ranges[self.index]
        self.title.configure(text=band.name or f"Range {band.index + 1}")
        self.meta.configure(text=f"{band.weight * 100:.0f}%")
        is_sel = self.index == selected
        bg = "#e8e8e8" if is_sel else "#f5f5f5"
        self.frame.configure(
            highlightbackground="#111111" if is_sel else "#c8c8c8",
            highlightthickness=3 if is_sel else 1,
            bg=bg,
        )
        self.title.configure(bg=bg)
        self.meta.configure(bg=bg)
        self.eye.configure_bg(bg)
        self.eye.set_shown(band.visible)
        if not band.visible:
            self.meta.configure(fg="#888888")
        else:
            self.meta.configure(fg="#444444")

    def _click(self, _event=None) -> None:
        self.app.select_range(self.index, toggle=True)

    def _on_eye(self, shown: bool) -> None:
        self.app.set_range_visible(self.index, bool(shown))

    def _on_percent_click(self, _event=None) -> None:
        self.app.select_range(self.index)
        self._begin_percent_edit()

    def _begin_percent_edit(self) -> None:
        if self._pct_entry is not None:
            return
        if self.app.range_map is None:
            return
        band = self.app.range_map.ranges[self.index]
        entry = tk.Entry(self.frame, width=6, justify="center", font=("Segoe UI", 9))
        entry.insert(0, f"{band.weight * 100:.0f}")
        entry.select_range(0, tk.END)
        self.meta.pack_forget()
        entry.pack(anchor="w")
        self._pct_entry = entry
        entry.bind("<Return>", lambda _e: self._finish_percent_edit(commit=True))
        entry.bind("<FocusOut>", lambda _e: self._finish_percent_edit(commit=True))
        entry.bind("<Escape>", lambda _e: self._finish_percent_edit(commit=False))
        entry.focus_set()

    def _finish_percent_edit(self, *, commit: bool) -> None:
        entry = self._pct_entry
        raw = entry.get() if entry is not None else ""
        if entry is not None:
            try:
                entry.destroy()
            except tk.TclError:
                pass
        self._pct_entry = None
        self.meta.pack(anchor="w")
        if not commit:
            return
        try:
            pct = float(raw.strip().rstrip("%"))
        except ValueError:
            return
        self.app.apply_typed_percent(self.index, pct)
