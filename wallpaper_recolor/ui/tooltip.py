# -*- coding: utf-8 -*-
"""
wallpaper_recolor.ui.tooltip
----------------------------
Delayed hover balloon for selectable controls. Tkinter ``Toplevel`` only —
no extra packages. Does not steal focus (Windows ``WS_EX_NOACTIVATE``).

Hover a widget: after ``DELAY_MS`` a small yellow label appears under the
pointer. Leave or press hides it. ``text`` may be a string or a callable
``(event) -> str`` so a canvas can describe the region under the cursor.

Class references (code + name only):
- CAP3321C Data Wrangling
"""

from __future__ import annotations

from collections.abc import Callable
import sys
import tkinter as tk
from types import SimpleNamespace

DELAY_MS = 400
_TIP_BG = "#ffffe0"
_TIP_FG = "#222222"
_TIP_FONT = ("Segoe UI", 8)
_ATTR = "_hover_tip"
_GWL_EXSTYLE = -20
_WS_EX_NOACTIVATE = 0x08000000
_WS_EX_TOOLWINDOW = 0x00000080

TextSpec = str | Callable[..., str | None]


def hover_tip_of(widget: tk.Misc) -> "HoverTip | None":
    """Bound ``HoverTip``, or ``None`` if this widget has no tooltip."""
    tip = getattr(widget, _ATTR, None)
    return tip if isinstance(tip, HoverTip) else None


def tooltip_text(widget: tk.Misc, event=None) -> str:
    """Resolved tooltip copy for tests and diagnostics."""
    tip = hover_tip_of(widget)
    if tip is None:
        return ""
    return tip.resolve(event)


class HoverTip:
    """One delayed balloon tied to a widget. Only one balloon is shown at a time."""

    _active: "HoverTip | None" = None

    def __init__(
        self,
        widget: tk.Misc,
        text: TextSpec,
        *,
        delay_ms: int = DELAY_MS,
        wraplength: int = 280,
    ) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = max(0, int(delay_ms))
        self.wraplength = int(wraplength)
        self._job: str | None = None
        self._win: tk.Toplevel | None = None
        self._label: tk.Label | None = None
        self._shown = ""
        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<Motion>", self._on_motion, add="+")
        widget.bind("<ButtonPress>", self._on_press, add="+")
        widget.bind("<Destroy>", self._on_destroy, add="+")

    def resolve(self, event=None) -> str:
        """Evaluate ``text`` (string or callable) to the current caption."""
        spec = self.text
        if callable(spec):
            try:
                try:
                    value = spec(event)
                except TypeError:
                    value = spec()
            except Exception:
                return ""
            return "" if value is None else str(value).strip()
        return "" if spec is None else str(spec).strip()

    def hide(self) -> None:
        self._cancel_job()
        win = self._win
        self._win = None
        self._label = None
        self._shown = ""
        if HoverTip._active is self:
            HoverTip._active = None
        if win is not None:
            try:
                win.destroy()
            except tk.TclError:
                pass

    def _on_enter(self, event=None) -> None:
        self._schedule(event)

    def _on_motion(self, event=None) -> None:
        if not callable(self.text):
            return
        new = self.resolve(event)
        if new == self._shown:
            return
        if self._win is not None:
            if new:
                self._set_caption(new)
                self._place(event)
            else:
                self.hide()
            return
        self._schedule(event)

    def _on_leave(self, event=None) -> None:
        if self._pointer_inside_widget():
            return
        self.hide()

    def _on_press(self, _event=None) -> None:
        self.hide()

    def _on_destroy(self, event=None) -> None:
        if event is not None and getattr(event, "widget", None) is not self.widget:
            return
        self.hide()

    def _schedule(self, event=None) -> None:
        self._cancel_job()
        snap = _snapshot(event)
        if not self.resolve(snap):
            return
        try:
            self._job = self.widget.after(self.delay_ms, lambda: self._show(snap))
        except tk.TclError:
            self._job = None

    def _cancel_job(self) -> None:
        job = self._job
        self._job = None
        if job is not None:
            try:
                self.widget.after_cancel(job)
            except tk.TclError:
                pass

    def _show(self, event=None) -> None:
        self._job = None
        caption = self.resolve(event)
        if not caption:
            return
        try:
            if not self.widget.winfo_exists():
                return
        except tk.TclError:
            return
        other = HoverTip._active
        if other is not None and other is not self:
            other.hide()
        if self._win is None:
            self._build_window()
        if self._win is None:
            return
        self._set_caption(caption)
        self._place(event)
        HoverTip._active = self

    def _build_window(self) -> None:
        try:
            win = tk.Toplevel(self.widget)
        except tk.TclError:
            return
        win.withdraw()
        win.wm_overrideredirect(True)
        try:
            win.wm_attributes("-topmost", True)
        except tk.TclError:
            pass
        if sys.platform == "win32":
            try:
                win.wm_attributes("-toolwindow", True)
            except tk.TclError:
                pass
        label = tk.Label(
            win,
            text="",
            bg=_TIP_BG,
            fg=_TIP_FG,
            relief="solid",
            bd=1,
            font=_TIP_FONT,
            justify="left",
            wraplength=self.wraplength,
            padx=6,
            pady=3,
            takefocus=0,
        )
        label.pack()
        self._win = win
        self._label = label

    def _set_caption(self, caption: str) -> None:
        self._shown = caption
        if self._label is not None:
            try:
                self._label.configure(text=caption, wraplength=self.wraplength)
            except tk.TclError:
                pass

    def _place(self, event=None) -> None:
        win = self._win
        if win is None:
            return
        try:
            win.update_idletasks()
            tw = int(win.winfo_reqwidth())
            th = int(win.winfo_reqheight())
            sw = int(self.widget.winfo_screenwidth())
            sh = int(self.widget.winfo_screenheight())
            if event is not None and getattr(event, "x_root", None) is not None:
                x = int(event.x_root) + 12
                y = int(event.y_root) + 16
            else:
                x = int(self.widget.winfo_rootx()) + 8
                y = int(self.widget.winfo_rooty()) + int(self.widget.winfo_height()) + 6
            x = max(0, min(x, sw - tw - 4))
            if y + th > sh - 4:
                y = max(0, int(self.widget.winfo_rooty()) - th - 6)
            win.wm_geometry(f"+{x}+{y}")
            _no_activate(win)
            win.deiconify()
            try:
                win.lift()
            except tk.TclError:
                pass
            _no_activate(win)
        except tk.TclError:
            self.hide()

    def _pointer_inside_widget(self) -> bool:
        try:
            px = int(self.widget.winfo_pointerx())
            py = int(self.widget.winfo_pointery())
            x = int(self.widget.winfo_rootx())
            y = int(self.widget.winfo_rooty())
            w = int(self.widget.winfo_width())
            h = int(self.widget.winfo_height())
        except (tk.TclError, TypeError, ValueError):
            return False
        return x <= px < x + w and y <= py < y + h


def bind_tooltip(
    widget: tk.Misc,
    text: TextSpec,
    *,
    delay_ms: int = DELAY_MS,
    wraplength: int = 280,
) -> HoverTip:
    """Attach (or replace) a hover tooltip on ``widget``. Idempotent."""
    existing = hover_tip_of(widget)
    if existing is not None:
        existing.text = text
        existing.delay_ms = max(0, int(delay_ms))
        existing.wraplength = int(wraplength)
        return existing
    tip = HoverTip(widget, text, delay_ms=delay_ms, wraplength=wraplength)
    setattr(widget, _ATTR, tip)
    return tip


def _snapshot(event) -> SimpleNamespace | None:
    if event is None:
        return None
    return SimpleNamespace(
        x=getattr(event, "x", 0),
        y=getattr(event, "y", 0),
        x_root=getattr(event, "x_root", None),
        y_root=getattr(event, "y_root", None),
        widget=getattr(event, "widget", None),
    )


def _no_activate(win: tk.Toplevel) -> None:
    """Windows: keep the balloon from taking keyboard focus."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        hwnd = int(win.winfo_id())
        user32 = ctypes.windll.user32
        getter = getattr(user32, "GetWindowLongPtrW", None) or user32.GetWindowLongW
        setter = getattr(user32, "SetWindowLongPtrW", None) or user32.SetWindowLongW
        style = getter(hwnd, _GWL_EXSTYLE)
        setter(hwnd, _GWL_EXSTYLE, int(style) | _WS_EX_NOACTIVATE | _WS_EX_TOOLWINDOW)
    except Exception:
        pass
