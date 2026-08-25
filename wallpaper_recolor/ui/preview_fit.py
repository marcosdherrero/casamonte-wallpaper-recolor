# -*- coding: utf-8 -*-
"""
wallpaper_recolor.ui.preview_fit
------------------------------
Fit/contain math and PreviewZoomHost (Original/Result camera).

100% contain-scales the whole image into the pane (no crop scrollbars).
Zoom > 100% multiplies that box. Composite Original and Result share one
dest size and pan. Clusters Lab zoom is a separate camera.

Class references (code + name only):
- CAP3321C Data Wrangling
- CAP4631C Machine Learning
- CAP4633C Machine Learning 2
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from wallpaper_recolor.transform.crop import clamp_zoom
from wallpaper_recolor.ui.constants import (
    PREVIEW_MAX_EDGE,
    PREVIEW_PANE_BG,
    TOOL_GRAB_MOVE,
    TOOL_VIEW_MOVE,
    VIEW_ZOOM_PCT_DEFAULT,
    VIEW_ZOOM_PCT_MAX,
    VIEW_ZOOM_PCT_MIN,
    VIEW_ZOOM_PCT_STEP,
    _MIN_PANE_FOR_FIT,
    _PREVIEW_PAN_DRAG_PX,
)

if TYPE_CHECKING:
    from wallpaper_recolor.ui.app import WallpaperRecolorApp

def _format_zoom_text(zoom: float) -> str:
    """Compact zoom for the entry: 2 or 1.5, not 2.00."""
    z = clamp_zoom(zoom)
    if abs(z - round(z)) < 1e-6:
        return str(int(round(z)))
    text = f"{z:.2f}".rstrip("0").rstrip(".")
    return text or "1"


def clamp_view_zoom_pct(pct: float) -> float:
    """Keep preview view-zoom in 100%–800% (100% = fit-to-pane)."""
    try:
        value = float(pct)
    except (TypeError, ValueError):
        return VIEW_ZOOM_PCT_DEFAULT
    if value != value:  # NaN
        return VIEW_ZOOM_PCT_DEFAULT
    return min(VIEW_ZOOM_PCT_MAX, max(VIEW_ZOOM_PCT_MIN, value))


def view_zoom_factor(pct: float) -> float:
    """100% → 1.0 (fit), 200% → 2.0 (2× fitted size)."""
    return clamp_view_zoom_pct(pct) / 100.0


def pane_usable_for_fit(pane_w: int, pane_h: int) -> bool:
    """True when a pane has a real layout size (not an unmapped 1×1 widget)."""
    return int(pane_w) >= _MIN_PANE_FOR_FIT and int(pane_h) >= _MIN_PANE_FOR_FIT


def shared_pane_size(
    panes: tuple[tuple[int, int], ...]
) -> tuple[int, int] | None:
    """One destination pane for Original and Result: min usable width × height.

    A longer title or a scrollbar on only one side must not give that side
    its own fit box. ``None`` when no pane is mapped yet.
    """
    usable = [
        (max(1, int(pw)), max(1, int(ph)))
        for pw, ph in panes
        if pane_usable_for_fit(pw, ph)
    ]
    if not usable:
        return None
    return (min(w for w, _h in usable), min(h for _w, h in usable))


def pane_fit_factor(src_w: int, src_h: int, pane_w: int, pane_h: int) -> float:
    """Scale that places the whole source image in the pane (no overflow)."""
    sw, sh = max(1, int(src_w)), max(1, int(src_h))
    pw, ph = max(1, int(pane_w)), max(1, int(pane_h))
    return min(pw / sw, ph / sh)


def contain_size(
    src_w: int,
    src_h: int,
    pane_w: int,
    pane_h: int,
    *,
    allow_upscale: bool = False,
) -> tuple[int, int]:
    """Pixel size of ``src`` contained in the pane (not cover). Both sides fit.

    100% view zoom uses this box so large TIFs shrink to the preview; values
    are floored so a rounding pixel cannot create overflow scrollbars.
    """
    sw, sh = max(1, int(src_w)), max(1, int(src_h))
    pw, ph = max(1, int(pane_w)), max(1, int(pane_h))
    factor = min(pw / sw, ph / sh)
    if not allow_upscale:
        factor = min(1.0, factor)
    width = max(1, min(pw, int(sw * factor)))
    height = max(1, min(ph, int(sh * factor)))
    return width, height


def letterbox_xy(
    img_w: int, img_h: int, pane_w: int, pane_h: int
) -> tuple[int, int]:
    """Top-left of a centered image inside ``pane_w`` × ``pane_h`` (or 0,0 if overflow)."""
    iw, ih = max(0, int(img_w)), max(0, int(img_h))
    pw, ph = max(1, int(pane_w)), max(1, int(pane_h))
    x = (pw - iw) // 2 if iw <= pw else 0
    y = (ph - ih) // 2 if ih <= ph else 0
    return x, y


def shared_fit_factor(
    src_w: int, src_h: int, panes: tuple[tuple[int, int], ...]
) -> float | None:
    """Fit-factor for the shared destination pane so both images stay one scale.

    Returns ``None`` when no pane is usable. Capped at 1.0 so 100% never
    upscales past the source.
    """
    pane = shared_pane_size(panes)
    if pane is None:
        return None
    return min(1.0, pane_fit_factor(src_w, src_h, *pane))


def fit_max_edge(
    src_w: int,
    src_h: int,
    panes: tuple[tuple[int, int], ...],
    *,
    fallback: int = PREVIEW_MAX_EDGE,
) -> int:
    """Long-edge cap for 100% view zoom: fit-to-pane, or ``fallback`` if unknown.

    Uses the **smaller** of the pane fit-factors so Original and Result stay
    the same scale. Never larger than the source long edge. Floor so the
    fitted bitmap cannot overflow a pane by a rounding pixel.
    """
    sw, sh = max(1, int(src_w)), max(1, int(src_h))
    long_edge = max(sw, sh)
    pane = shared_pane_size(panes)
    if pane is None:
        cap = max(1, int(fallback))
        return max(1, min(cap, long_edge))
    fitted_w, fitted_h = contain_size(sw, sh, *pane)
    return max(1, min(long_edge, max(fitted_w, fitted_h)))


def _preview_base_size(
    src_w: int, src_h: int, max_edge: int = PREVIEW_MAX_EDGE
) -> tuple[int, int]:
    """100% on-screen size (long edge at most ``max_edge`` — the fit box)."""
    w, h = max(1, int(src_w)), max(1, int(src_h))
    long_edge = max(w, h)
    if long_edge <= max_edge:
        return (w, h)
    scale = max_edge / long_edge
    return (max(1, int(w * scale)), max(1, int(h * scale)))


def _view_zoom_size(
    src_w: int,
    src_h: int,
    zoom: float,
    max_edge: int = PREVIEW_MAX_EDGE,
) -> tuple[int, int]:
    """On-screen size at this view zoom: fitted 100% box times zoom."""
    bw, bh = _preview_base_size(src_w, src_h, max_edge)
    z = float(zoom)
    if z <= 1.0 + 1e-6:
        return (bw, bh)
    return (max(1, int(round(bw * z))), max(1, int(round(bh * z))))


def _scale_view_zoom(
    image: Image.Image,
    zoom: float,
    max_edge: int = PREVIEW_MAX_EDGE,
    *,
    size: tuple[int, int] | None = None,
) -> Image.Image:
    """NEAREST-scale from a high-res source to the on-screen view-zoom size.

    100% fits ``max_edge`` (pane-fit long edge); 200–800% multiplies that
    box. Always resample the source with nearest-neighbor — never
    BILINEAR/BICUBIC, and never an already-downscaled preview bitmap.
    Pass ``size`` to force Original and Result onto the same pixel box.
    """
    target = size if size is not None else _view_zoom_size(*image.size, zoom, max_edge)
    if target == image.size:
        return image
    return image.resize(target, Image.Resampling.NEAREST)


HIT_PREVIEW = "preview"
HIT_SLIDER = "slider"
HIT_SIDEBAR = "sidebar"
HIT_NONE = "none"
_WP_TOOLTIP_ATTR = "_wp_tooltip"


def _widget_contains_root(widget: tk.Misc, x_root: int, y_root: int) -> bool:
    """True if screen point ``(x_root, y_root)`` lies on ``widget``.

    Unmapped 1×1 widgets must not match ``(0, 0)``. ``winfo_containing`` is
    not used here — grab / focus / tooltip balloons poison it on Windows.
    """
    try:
        viewable = getattr(widget, "winfo_viewable", None)
        if callable(viewable) and not viewable():
            return False
        x0 = int(widget.winfo_rootx())
        y0 = int(widget.winfo_rooty())
        w = int(widget.winfo_width())
        h = int(widget.winfo_height())
    except (tk.TclError, TypeError, ValueError, AttributeError):
        return False
    if w <= 0 or h <= 0:
        return False
    return x0 <= int(x_root) < x0 + w and y0 <= int(y_root) < y0 + h


def _is_pointer_overlay(widget: tk.Misc | None) -> bool:
    """True for tooltip balloons (they must not steal hit-tests)."""
    current: tk.Misc | None = widget
    seen: set[str] = set()
    while current is not None:
        if getattr(current, _WP_TOOLTIP_ATTR, False):
            return True
        try:
            key = str(current)
        except (tk.TclError, TypeError):
            break
        if key in seen:
            break
        seen.add(key)
        current = getattr(current, "master", None)
    return False


def _is_independent_toplevel(widget: tk.Misc) -> bool:
    try:
        return widget.winfo_toplevel() is widget
    except (tk.TclError, AttributeError):
        return False


def widget_at_root(root: tk.Misc, x_root: int, y_root: int) -> tk.Misc | None:
    """Deepest mapped widget at a screen point, ignoring tooltip balloons.

    Walks geometry (not ``winfo_containing``) so a leftover grab or the
    last focused control cannot keep the pointer pinned after a click.
    """
    x, y = int(x_root), int(y_root)
    best: tk.Misc | None = None

    def walk(widget: tk.Misc) -> None:
        nonlocal best
        if getattr(widget, _WP_TOOLTIP_ATTR, False):
            return
        try:
            if not widget.winfo_ismapped():
                return
        except (tk.TclError, AttributeError):
            return
        contains = _widget_contains_root(widget, x, y)
        try:
            children = list(widget.winfo_children())
        except (tk.TclError, AttributeError):
            children = []
        if contains:
            best = widget
            for child in reversed(children):
                walk(child)
            return
        for child in children:
            if _is_independent_toplevel(child):
                walk(child)

    walk(root)
    return best


def classify_pointer_hit(
    x: int,
    y: int,
    *,
    preview: Sequence[tk.Misc] = (),
    sliders: Sequence[tk.Misc] = (),
    sidebars: Sequence[tk.Misc] = (),
) -> str:
    """Point → preview (wheel zoom) vs slider vs sidebar (column scroll)."""
    xr, yr = int(x), int(y)
    for widget in preview:
        if _widget_contains_root(widget, xr, yr):
            return HIT_PREVIEW
    for widget in sliders:
        if _widget_contains_root(widget, xr, yr):
            return HIT_SLIDER
    for widget in sidebars:
        if _widget_contains_root(widget, xr, yr):
            return HIT_SIDEBAR
    return HIT_NONE


def _wheel_zoom_pct_delta(event) -> float:
    """Wheel-up → zoom in. Windows ``delta`` ±120 (num often ``'??'``); Linux Button-4/5."""
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
    if delta > 0 or num == 4:
        return VIEW_ZOOM_PCT_STEP
    if delta < 0 or num == 5:
        return -VIEW_ZOOM_PCT_STEP
    return 0.0


class PreviewZoomHost(tk.Frame):
    """Clipped viewport for a preview PhotoImage: view-zoom + pan, not crop.

    The image label is ``place``d inside a clipping frame. At 100% (fit) the
    whole photo is centered (letterbox) with no scrollbars. Past 100%
    click-drag pans and overflow scrollbars show.
    """

    def __init__(
        self,
        parent: tk.Misc,
        app: "WallpaperRecolorApp",
        *,
        bg: str = PREVIEW_PANE_BG,
        on_click=None,
        on_tap=None,
        on_rect=None,
        share_pan: bool = False,
    ) -> None:
        super().__init__(parent, bg=bg, highlightthickness=0)
        self.app = app
        self.on_click = on_click
        self.on_tap = on_tap
        self.on_rect = on_rect
        self.share_pan = bool(share_pan)
        self.rect_mode = False
        self._rect_start: tuple[int, int] | None = None
        self._bg = bg
        self._photo: ImageTk.PhotoImage | None = None
        self._pan_x = 0
        self._pan_y = 0
        self._press: tuple[int, int] | None = None
        self.panning = False
        self.moving_layer = False
        self.viewport = tk.Frame(self, bg=bg, highlightthickness=0)
        self.hsb = ttk.Scrollbar(self, orient="horizontal", command=self._xview)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self._yview)
        self._sb_x = False
        self._sb_y = False
        self._rect_guides: tuple[tk.Frame, tk.Frame, tk.Frame, tk.Frame] | None = None
        self.viewport.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        # tk.Label (not ttk): theme styles ignore background, which made
        # letterbox look black on one pane and chrome-gray on the other.
        self.image_label = tk.Label(
            self.viewport,
            bg=bg,
            bd=0,
            highlightthickness=0,
            padx=0,
            pady=0,
        )
        self.viewport.bind("<Configure>", lambda _e: self._layout())
        self._bind_pan(self.viewport)
        self._bind_pan(self.image_label)

    def _bind_pan(self, widget: tk.Misc) -> None:
        widget.bind("<ButtonPress-1>", self._on_press, add="+")
        widget.bind("<B1-Motion>", self._on_drag, add="+")
        widget.bind("<ButtonRelease-1>", self._on_release, add="+")

    def set_photo(self, photo: ImageTk.PhotoImage | None, *, keep_pan: bool = True) -> None:
        self._photo = photo
        if photo is None:
            try:
                self.image_label.place_forget()
                self.image_label.configure(image="")
            except tk.TclError:
                pass
            self._hide_drag_rect()
            self._sync_scrollbars(False, False)
            return
        self.image_label.configure(image=photo)
        if not keep_pan:
            self._pan_x = 0
            self._pan_y = 0
        self._layout()

    def set_pan(self, x: int, y: int, *, propagate: bool = True) -> None:
        """Scroll origin in photo pixels (top-left of the clipped view)."""
        self._pan_x = int(x)
        self._pan_y = int(y)
        self._layout(propagate=propagate)

    def _img_size(self) -> tuple[int, int]:
        photo = self._photo
        if photo is None:
            return 0, 0
        try:
            return int(photo.width()), int(photo.height())
        except tk.TclError:
            return 0, 0

    def _vp_size(self) -> tuple[int, int]:
        try:
            return max(1, int(self.viewport.winfo_width())), max(1, int(self.viewport.winfo_height()))
        except tk.TclError:
            return 1, 1

    def _max_pan(self) -> tuple[int, int]:
        iw, ih = self._img_size()
        vw, vh = self._vp_size()
        return max(0, iw - vw), max(0, ih - vh)

    def can_pan(self) -> bool:
        mx, my = self._max_pan()
        return mx > 0 or my > 0

    def _layout(self, *, propagate: bool = True) -> None:
        iw, ih = self._img_size()
        vw, vh = self._vp_size()
        max_x, max_y = self._max_pan()
        self._pan_x = min(max(0, int(self._pan_x)), max_x)
        self._pan_y = min(max(0, int(self._pan_y)), max_y)
        if self._photo is None or iw <= 0 or ih <= 0:
            self._sync_scrollbars(False, False)
            return
        fit = True
        try:
            fit = float(self.app._preview_zoom_factor()) <= 1.0 + 1e-6
        except (TypeError, ValueError, AttributeError):
            fit = True
        if fit:
            self._pan_x = 0
            self._pan_y = 0
        lx, ly = letterbox_xy(iw, ih, vw, vh)
        overflow_x = (not fit) and iw > vw
        overflow_y = (not fit) and ih > vh
        x = -self._pan_x if overflow_x else lx
        y = -self._pan_y if overflow_y else ly
        try:
            self.image_label.place(x=x, y=y, width=iw, height=ih)
        except tk.TclError:
            return
        self._sync_scrollbars(overflow_x, overflow_y)
        self._update_scrollbar_values()
        if propagate and self.share_pan:
            self.app._mirror_composite_pan(self)
        self._sync_host_cursor()

    def _sync_host_cursor(self) -> None:
        if self.rect_mode:
            cursor = "crosshair"
        elif self.app._grab_move_on():
            cursor = "hand2"
        elif self.on_click is not None and self.app._eyedrop_enabled():
            return
        elif self.app._view_move_on():
            cursor = "fleur"
        elif self.on_click is not None:
            return
        else:
            cursor = "fleur" if self.can_pan() else ""
        try:
            self.image_label.configure(cursor=cursor)
            self.viewport.configure(cursor=cursor)
        except tk.TclError:
            pass

    def _sync_scrollbars(self, show_x: bool, show_y: bool) -> None:
        if show_x and not self._sb_x:
            self.hsb.grid(row=1, column=0, sticky="ew")
            self._sb_x = True
        elif not show_x and self._sb_x:
            self.hsb.grid_forget()
            self._sb_x = False
        if show_y and not self._sb_y:
            self.vsb.grid(row=0, column=1, sticky="ns")
            self._sb_y = True
        elif not show_y and self._sb_y:
            self.vsb.grid_forget()
            self._sb_y = False

    def _update_scrollbar_values(self) -> None:
        iw, ih = self._img_size()
        vw, vh = self._vp_size()
        if iw <= 0:
            self.hsb.set(0.0, 1.0)
        else:
            self.hsb.set(self._pan_x / iw, min(1.0, (self._pan_x + vw) / iw))
        if ih <= 0:
            self.vsb.set(0.0, 1.0)
        else:
            self.vsb.set(self._pan_y / ih, min(1.0, (self._pan_y + vh) / ih))

    def _xview(self, *args) -> None:
        iw, _ih = self._img_size()
        if not args or iw <= 0:
            return
        if args[0] == "moveto":
            self._pan_x = int(round(float(args[1]) * iw))
        elif args[0] == "scroll":
            vw, _vh = self._vp_size()
            n = int(float(args[1]))
            what = str(args[2]) if len(args) > 2 else "units"
            step = max(1, vw) if what == "pages" else 40
            self._pan_x += n * step
        self._layout()

    def _yview(self, *args) -> None:
        _iw, ih = self._img_size()
        if not args or ih <= 0:
            return
        if args[0] == "moveto":
            self._pan_y = int(round(float(args[1]) * ih))
        elif args[0] == "scroll":
            _vw, vh = self._vp_size()
            n = int(float(args[1]))
            what = str(args[2]) if len(args) > 2 else "units"
            step = max(1, vh) if what == "pages" else 40
            self._pan_y += n * step
        self._layout()

    def reset_pan(self) -> None:
        self._pan_x = 0
        self._pan_y = 0
        self._layout()

    def zoom_anchor(self, old_z: float, new_z: float, x_root: int, y_root: int) -> None:
        """Keep the image point under the pointer after a view-zoom change."""
        if old_z <= 0 or abs(new_z - old_z) < 1e-6:
            self._layout()
            return
        try:
            vx = int(x_root) - int(self.viewport.winfo_rootx())
            vy = int(y_root) - int(self.viewport.winfo_rooty())
            lx = int(x_root) - int(self.image_label.winfo_rootx())
            ly = int(y_root) - int(self.image_label.winfo_rooty())
        except tk.TclError:
            self._layout()
            return
        ratio = new_z / old_z
        new_lx = lx * ratio
        new_ly = ly * ratio
        self._pan_x = int(round(new_lx - vx))
        self._pan_y = int(round(new_ly - vy))
        self._layout()

    def _event_label_xy(self, event) -> tuple[int, int]:
        x_root = getattr(event, "x_root", None)
        y_root = getattr(event, "y_root", None)
        if x_root is not None and y_root is not None:
            try:
                return (
                    int(x_root) - int(self.image_label.winfo_rootx()),
                    int(y_root) - int(self.image_label.winfo_rooty()),
                )
            except (tk.TclError, TypeError, ValueError):
                pass
        return int(getattr(event, "x", 0)), int(getattr(event, "y", 0))

    def _want_rect(self) -> bool:
        if self.on_rect is None:
            return False
        if self.rect_mode:
            return True
        # Select-area mode owns the drag. Grab Move must still reposition
        # overlays at 100% Fit, where can_pan() is False.
        if self.app._grab_move_on():
            return False
        return not self.can_pan()

    def _ensure_rect_guides(self) -> tuple[tk.Frame, tk.Frame, tk.Frame, tk.Frame]:
        if self._rect_guides is None:
            color = "#4a90d9"
            self._rect_guides = tuple(
                tk.Frame(self.viewport, bg=color, highlightthickness=0, bd=0)
                for _ in range(4)
            )
        return self._rect_guides

    def _hide_drag_rect(self) -> None:
        if self._rect_guides is None:
            return
        for bar in self._rect_guides:
            try:
                bar.place_forget()
            except tk.TclError:
                pass

    def _show_drag_rect(self, x0: int, y0: int, x1: int, y1: int) -> None:
        try:
            ox = int(self.image_label.winfo_x())
            oy = int(self.image_label.winfo_y())
        except tk.TclError:
            ox = oy = 0
        a, b = ox + min(x0, x1), oy + min(y0, y1)
        c, d = ox + max(x0, x1), oy + max(y0, y1)
        if c - a < 2 or d - b < 2:
            self._hide_drag_rect()
            return
        n, s, e, w = self._ensure_rect_guides()
        try:
            n.place(x=a, y=b, width=max(1, c - a), height=1)
            s.place(x=a, y=d - 1, width=max(1, c - a), height=1)
            w.place(x=a, y=b, width=1, height=max(1, d - b))
            e.place(x=c - 1, y=b, width=1, height=max(1, d - b))
            for bar in (n, s, e, w):
                tk.Misc.lift(bar)
        except tk.TclError:
            pass

    def _on_press(self, event) -> None:
        self._press = (int(event.x_root), int(event.y_root))
        self.panning = False
        self.moving_layer = False
        want_rect = self._want_rect() and not self.app._grab_moves_layer(host=self)
        self._rect_start = self._event_label_xy(event) if want_rect else None
        if self._rect_start is None:
            self._hide_drag_rect()

    def _on_drag(self, event) -> None:
        if self._rect_start is not None and not self.app._grab_moves_layer(host=self):
            x1, y1 = self._event_label_xy(event)
            self._show_drag_rect(self._rect_start[0], self._rect_start[1], x1, y1)
            return
        if self._press is None:
            return
        x, y = int(event.x_root), int(event.y_root)
        dx = x - self._press[0]
        dy = y - self._press[1]
        if (
            not self.panning
            and not self.moving_layer
            and dx * dx + dy * dy < _PREVIEW_PAN_DRAG_PX * _PREVIEW_PAN_DRAG_PX
        ):
            return
        if self.app._grab_moves_layer(host=self):
            self.moving_layer = True
            self._press = (x, y)
            self.app._nudge_grab_move(dx, dy, host=self)
            return
        pan = self.app._view_move_on() or self.app._grab_move_on()
        if not pan or not self.can_pan():
            return
        self.panning = True
        self._pan_x -= dx
        self._pan_y -= dy
        self._press = (x, y)
        self._layout()

    def _on_release(self, event) -> None:
        panned = self.panning or self.moving_layer
        start = self._rect_start
        self.panning = False
        self.moving_layer = False
        self._press = None
        self._rect_start = None
        self._hide_drag_rect()
        if panned:
            if panned and getattr(self.app, "_layer_drag_before", None) is not None:
                self.app._finish_layer_drag()
            return
        if start is not None and self.on_rect is not None:
            x1, y1 = self._event_label_xy(event)
            if abs(x1 - start[0]) >= 4 or abs(y1 - start[1]) >= 4:
                self.on_rect(start[0], start[1], x1, y1)
                return
        if self.on_click is not None:
            self.on_click(event)
            return
        if self.on_tap is not None:
            self.on_tap(event)
