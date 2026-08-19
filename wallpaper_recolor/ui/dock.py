# -*- coding: utf-8 -*-
"""
wallpaper_recolor.ui.dock
------------------------------
Paned right column, clip canvas, ScrollColumn, DockablePanel.

Windows does not clip native HWNDs created via Canvas ``create_window``.
Each column uses ``_ClipCanvas`` (a Frame that clips) and docked panels are
true children packed on ``column.inner``. Do not ``pack(in_=)`` from root.

Class references (code + name only):
- CAP3321C Data Wrangling
- CAP4631C Machine Learning
- CAP4633C Machine Learning 2
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import tkinter as tk
from tkinter import ttk

from wallpaper_recolor.ui.constants import (
    _COL_IDLE_BORDER,
    _COLUMN_SCROLL_TAG,
    _INSERT_MARKER_BG,
    _PANEL_BAR_BG,
    _PANEL_BAR_FG,
    _PANEL_PACK_PADY,
    _SNAP_HIGHLIGHT,
    _UNDOCK_DRAG_PX,
)
from wallpaper_recolor.ui.widgets import _bind_wheel_tree

if TYPE_CHECKING:
    from wallpaper_recolor.ui.app import WallpaperRecolorApp

class _SashSplit(tk.PanedWindow):
    """Classic paned split so the sash is a visible gutter, with ttk-like ``sashpos``."""

    def sashpos(self, index: int, newpos: int | None = None) -> int:
        """Pixel position of sash ``index``; set it when ``newpos`` is given."""
        try:
            vertical = str(self.cget("orient")) == "vertical"
            if newpos is None:
                x, y = self.sash_coord(int(index))
                return int(y if vertical else x)
            pos = int(newpos)
            if vertical:
                self.sash_place(int(index), 0, pos)
            else:
                self.sash_place(int(index), pos, 0)
            return pos
        except (tk.TclError, TypeError, ValueError):
            return 0

    def pane(self, child, **kw):
        """ttk.Panedwindow.pane compatibility (minsize / weight → stretch)."""
        opts = dict(kw)
        if "weight" in opts:
            weight = opts.pop("weight")
            try:
                stretch = "always" if int(weight) else "never"
            except (TypeError, ValueError):
                stretch = "always"
            opts["stretch"] = stretch
        if opts:
            self.paneconfigure(child, **opts)
        return self.paneconfigure(child)


class _ClipCanvas(tk.Frame):
    """Pane viewport whose HWND clips native children.

    Tk ``Canvas.create_window`` items are **not** clipped on Windows, so Color &
    lighting painted over Layers. This frame is sized by the paned pane; overflow
    stays inside this half. ``yview_scroll`` / ``itemcget`` match the old canvas
    so wheel tests can patch the same names.
    """

    def __init__(self, column: "ScrollColumn", **kw) -> None:
        super().__init__(column, **kw)
        self._column = column

    def yview_scroll(self, n, what) -> None:
        self._column._yview_scroll(int(n), str(what))

    def yview_moveto(self, fraction) -> None:
        self._column._yview_moveto(float(fraction))

    def yview(self, *args):
        if not args:
            return self._column._yview_fraction()
        op = str(args[0])
        if op == "moveto":
            self._column._yview_moveto(float(args[1]))
        elif op == "scroll":
            self._column._yview_scroll(int(float(args[1])), str(args[2]))

    def itemcget(self, _item, option):
        col = self._column
        if str(option) == "height":
            try:
                return str(max(1, int(col.inner.winfo_reqheight())))
            except tk.TclError:
                return "1"
        if str(option) == "width":
            try:
                return str(max(1, int(col.inner.winfo_reqwidth())))
            except tk.TclError:
                return "1"
        return ""

    def itemconfigure(self, _item, **_kw) -> str:
        return ""

    def configure(self, cnf=None, **kw):
        merged = {}
        if cnf:
            merged.update(cnf if isinstance(cnf, dict) else {})
        merged.update(kw)
        merged.pop("yscrollcommand", None)
        merged.pop("scrollregion", None)
        if merged:
            super().configure(**merged)

    config = configure


class ScrollColumn(tk.Frame):
    """One body column: stacked dockable panels, mouse-wheel scroll if they overflow.

    Docked panes are **true children** of ``inner`` and ``pack()`` there (not
    ``pack(in_=)`` from root). ``inner`` is a child of this pane's clip frame so
    Windows clips Exposure / Color & lighting at the sash instead of painting
    over Layers. ``inner`` is placed only as the scroll surface (not docked shells).
    """

    def __init__(self, parent: tk.Misc, app: "WallpaperRecolorApp", name: str) -> None:
        # Tiny natural size so a nested Panedwindow can split 50/50 instead of
        # sizing panes to the stacked panels (which overlapped the other half).
        super().__init__(
            parent,
            highlightthickness=2,
            highlightbackground=_COL_IDLE_BORDER,
            width=1,
            height=1,
        )
        self.pack_propagate(False)
        self.app = app
        self.column_name = name
        self.panels: list[DockablePanel] = []
        self._scroll_y = 0
        style_bg = ttk.Style().lookup("TFrame", "background") or "#f0f0f0"
        self.canvas = _ClipCanvas(self, highlightthickness=0, bg=style_bg, width=1, height=1)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg=style_bg)
        self._win = "inner"
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.pack_propagate(False)
        self.inner.place(x=0, y=0, relwidth=1)
        self._sb_shown = False
        self._in_layout = False
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel)
        self.canvas.bind("<Button-5>", self._on_mousewheel)
        self._insert_marker = tk.Frame(self.inner, height=5, bg=_INSERT_MARKER_BG)
        self._ensure_scroll_class()
        self._tag_column_widgets()

    def _ensure_scroll_class(self) -> None:
        """Bind MouseWheel once on the custom tag (Windows delta, this column only)."""
        if getattr(self.app, "_column_scroll_bound", False):
            return
        self.app.root.bind_class(_COLUMN_SCROLL_TAG, "<MouseWheel>", self.app._on_column_mousewheel)
        self.app.root.bind_class(_COLUMN_SCROLL_TAG, "<Button-4>", self.app._on_column_mousewheel)
        self.app.root.bind_class(_COLUMN_SCROLL_TAG, "<Button-5>", self.app._on_column_mousewheel)
        self.app._column_scroll_bound = True

    def _tag_column_widgets(self) -> None:
        """Tag this column, docked shells, and chrome (true children of the shell)."""
        self._tag_tree(self)
        for panel in self._docked_panels():
            self._tag_tree(panel)
            bar = getattr(panel, "bar", None)
            body = getattr(panel, "body", None)
            if bar is not None:
                self._tag_tree(bar)
            if body is not None:
                self._tag_tree(body)
        self._bind_column_wheel_widgets()

    def _bind_column_wheel_widgets(self) -> None:
        """Direct MouseWheel on canvas + every docked control (Windows ttk.Scale)."""
        handler = self.app._on_column_mousewheel
        _bind_wheel_tree(self.canvas, handler)
        _bind_wheel_tree(self.inner, handler)
        for panel in self._docked_panels():
            _bind_wheel_tree(panel, handler)
            bar = getattr(panel, "bar", None)
            body = getattr(panel, "body", None)
            if bar is not None:
                _bind_wheel_tree(bar, handler)
            if body is not None:
                _bind_wheel_tree(body, handler)

    def _tag_tree(self, widget: tk.Misc) -> None:
        tags = list(widget.bindtags())
        if _COLUMN_SCROLL_TAG not in tags:
            tags.insert(1, _COLUMN_SCROLL_TAG)
            widget.bindtags(tuple(tags))
        for child in widget.winfo_children():
            self._tag_tree(child)

    def _untag_tree(self, widget: tk.Misc) -> None:
        tags = [t for t in widget.bindtags() if t != _COLUMN_SCROLL_TAG]
        widget.bindtags(tuple(tags))
        for child in widget.winfo_children():
            self._untag_tree(child)

    def _on_inner_configure(self, _event=None) -> None:
        if self._in_layout:
            return
        self._apply_scroll_geometry()
        self._tag_column_widgets()

    def _on_canvas_configure(self, event) -> None:
        if self._in_layout:
            return
        self._sync_layout(canvas_w=event.width, canvas_h=event.height)

    def _yview_fraction(self) -> tuple[float, float]:
        try:
            view_h = max(1, int(self.canvas.winfo_height()))
            content_h = max(1, int(self.inner.winfo_reqheight()))
        except tk.TclError:
            return (0.0, 1.0)
        if content_h <= view_h:
            return (0.0, 1.0)
        first = self._scroll_y / content_h
        last = (self._scroll_y + view_h) / content_h
        return (first, last)

    def _yview_moveto(self, fraction: float) -> None:
        try:
            view_h = max(1, int(self.canvas.winfo_height()))
            content_h = max(1, int(self.inner.winfo_reqheight()))
        except tk.TclError:
            return
        max_y = max(0, content_h - view_h)
        self._scroll_y = int(round(max(0.0, min(1.0, float(fraction))) * max_y))
        self._apply_scroll_geometry()

    def _yview_scroll(self, n: int, what: str) -> None:
        try:
            view_h = max(1, int(self.canvas.winfo_height()))
            content_h = max(1, int(self.inner.winfo_reqheight()))
        except tk.TclError:
            return
        max_y = max(0, content_h - view_h)
        step = view_h if str(what) == "pages" else 40
        self._scroll_y = max(0, min(max_y, int(self._scroll_y) + int(n) * step))
        self._apply_scroll_geometry()

    def _apply_scroll_geometry(self) -> None:
        """Keep inner full content height; clip HWND is the pane — sliders stay in the stack."""
        try:
            view_h = max(1, int(self.canvas.winfo_height()))
            content_h = max(1, int(self.inner.winfo_reqheight()))
        except tk.TclError:
            return
        max_y = max(0, content_h - view_h)
        self._scroll_y = max(0, min(max_y, int(self._scroll_y)))
        was = self._in_layout
        self._in_layout = True
        try:
            # Scroll surface only — docked shells pack() in inner, they are not placed.
            self.inner.place(x=0, y=-self._scroll_y, relwidth=1)
            if content_h <= view_h:
                first, last = 0.0, 1.0
            else:
                first = self._scroll_y / content_h
                last = (self._scroll_y + view_h) / content_h
            self._on_scrollset(str(first), str(last))
        finally:
            self._in_layout = was

    def _raise_docked_stack(self) -> None:
        """Lift packed panes inside this column (true children; does not cover the sash)."""
        self.app._raise_dock_stacks()

    def _on_scrollset(self, first: str, last: str) -> None:
        self.vsb.set(first, last)
        overflow = float(first) > 0.0 or float(last) < 1.0
        if overflow and not self._sb_shown:
            self.canvas.pack_forget()
            self.vsb.pack(side="right", fill="y")
            self.canvas.pack(side="left", fill="both", expand=True)
            self._sb_shown = True
        elif not overflow and self._sb_shown:
            self.vsb.pack_forget()
            self._sb_shown = False

    def _on_mousewheel(self, event) -> str | None:
        """Windows delta is ±120; Button-4/5 are Linux/X11. Always break so ttk.Scale does not eat the wheel."""
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
        # Windows: delta != 0 (num is often '??'); Linux: num 4 up / 5 down
        if delta:
            steps = -1 * (delta // 120)
            if steps == 0:
                steps = -1 if delta > 0 else 1
        elif num == 4:
            steps = -1
        elif num == 5:
            steps = 1
        else:
            steps = 0
        if steps != 0:
            self.canvas.yview_scroll(steps, "units")
            self._after_scroll_paint()
        return "break"

    def _after_scroll_paint(self) -> None:
        """Fully invalidate HWND panes after a wheel page so Windows does not smear."""
        try:
            self.update_idletasks()
            self._apply_scroll_geometry()
            self._raise_docked_stack()
            self.canvas.update_idletasks()
            self.app._raise_window_chrome()
            if self.column_name == "left":
                for name in ("orig_zoom_host", "tex_zoom_host"):
                    host = getattr(self.app, name, None)
                    if host is not None:
                        try:
                            host.lift()
                            host.viewport.configure(bg=host._bg)
                            host._layout(propagate=False)
                        except tk.TclError:
                            pass
                self.app._sync_composite_letterbox()
        except tk.TclError:
            pass

    def contains_root(self, x_root: int, y_root: int) -> bool:
        """True if the pointer is over this column (ignores overlapping floaters)."""
        try:
            x, y = int(self.winfo_rootx()), int(self.winfo_rooty())
            w, h = max(1, int(self.winfo_width())), max(1, int(self.winfo_height()))
        except tk.TclError:
            return False
        return x <= x_root < x + w and y <= y_root < y + h

    def insert_index_at(self, y_root: int, moving: "DockablePanel | None" = None) -> int:
        """Drop slot among docked panels (above/below by vertical midpoint)."""
        docked = [p for p in self.panels if not p.is_floating and p is not moving]
        if not docked:
            return 0
        for i, panel in enumerate(docked):
            try:
                mid = int(panel.winfo_rooty()) + max(1, int(panel.winfo_height())) // 2
            except tk.TclError:
                continue
            if y_root < mid:
                return i
        return len(docked)

    def set_drop_highlight(self, on: bool) -> None:
        color = _SNAP_HIGHLIGHT if on else _COL_IDLE_BORDER
        self.configure(highlightbackground=color, highlightcolor=color, highlightthickness=3 if on else 2)

    def show_insert_marker(self, index: int, moving: "DockablePanel | None") -> None:
        self._insert_marker.pack_forget()
        docked = [p for p in self.panels if not p.is_floating and p is not moving]
        if index < len(docked):
            self._insert_marker.pack(fill="x", padx=4, pady=1, before=docked[index])
        else:
            self._insert_marker.pack(fill="x", padx=4, pady=1)

    def hide_insert_marker(self) -> None:
        self._insert_marker.pack_forget()

    def attach(self, panel: "DockablePanel", index: int | None = None) -> None:
        """Pack ``panel`` into this column (initial layout or after a snap drop)."""
        if panel in self.panels:
            self.panels.remove(panel)
        if index is None:
            self.panels.append(panel)
        else:
            self.panels.insert(max(0, min(index, len(self.panels))), panel)
        panel.column = self
        panel._last_column = self
        panel.hidden = False
        self._adopt(panel)
        self._repack()

    def _adopt(self, panel: "DockablePanel") -> None:
        """Make ``panel`` a true child of this.inner so pack() stacks and the canvas clips."""
        if panel.is_floating:
            return
        try:
            if panel.master is self.inner:
                return
        except (tk.TclError, AttributeError):
            pass
        panel.remount_into(self.inner)

    def detach(self, panel: "DockablePanel") -> None:
        """Remove ``panel`` from this column without destroying it."""
        if panel in self.panels:
            self.panels.remove(panel)
        panel.pack_forget()
        self._repack()

    def _docked_panels(self) -> list["DockablePanel"]:
        return [p for p in self.panels if not p.is_floating and not p.hidden]

    def _sync_layout(self, canvas_w: int | None = None, canvas_h: int | None = None) -> None:
        """Size the inner window to the stacked panes so sliders are never clipped.

        Preview (flex) gets leftover height in this column. Texture / Coverage /
        Tone / Scale / Crop / Tessellate / Color wheel keep their requested
        height. If that stack is taller than the column, the canvas window grows
        with the content and the mouse wheel scrolls to it — the wheel must not
        expand and swallow the column.
        """
        if self._in_layout:
            return
        self._in_layout = True
        try:
            try:
                if canvas_w is None:
                    canvas_w = int(self.canvas.winfo_width())
                if canvas_h is None:
                    canvas_h = int(self.canvas.winfo_height())
            except tk.TclError:
                return
            canvas_w = max(1, int(canvas_w))
            canvas_h = max(1, int(canvas_h))
            docked = self._docked_panels()
            flex = [p for p in docked if p.flex and getattr(p, "expanded", True)]
            fixed = [p for p in docked if not (p.flex and getattr(p, "expanded", True))]
            self.inner.update_idletasks()
            # pady under every packed pane (flex included) so leftover is the body
            pack_gap = _PANEL_PACK_PADY * len(docked)
            fixed_h = sum(max(1, int(p.winfo_reqheight())) for p in fixed) + pack_gap
            leftover = canvas_h - fixed_h
            for panel in flex:
                panel.pack_propagate(False)
                minsz = max(80, int(panel.pane_minsize))
                if canvas_h < 50:
                    height = minsz
                elif leftover >= minsz:
                    height = leftover
                else:
                    height = minsz  # stack is taller than the column — scroll
                panel.configure(height=height)
            self.inner.update_idletasks()
            # Never pin inner to the viewport height — that hid sliders
            # under the color wheel (and left a hole if the wheel was popped out).
            self._apply_scroll_geometry()
            self._tag_column_widgets()
            self._raise_docked_stack()
            self.app._schedule_raise_chrome()
        finally:
            self._in_layout = False

    def _repack(self) -> None:
        self.hide_insert_marker()
        docked = self._docked_panels()
        for panel in docked:
            panel.pack_forget()
        for panel in docked:
            try:
                panel.update_idletasks()
            except tk.TclError:
                pass
        for panel in docked:
            # True child of inner: pack() (not pack(in_=) from root) so reqheight
            # stacks titles and this pane's clip HWND hides overflow at the sash.
            if panel.flex and getattr(panel, "expanded", True):
                panel.pack(fill="both", expand=False, padx=4, pady=(0, _PANEL_PACK_PADY))
            else:
                panel.pack(fill="x", expand=False, padx=4, pady=(0, _PANEL_PACK_PADY))
        self.inner.update_idletasks()
        self._sync_layout()


class DockablePanel(tk.Frame):
    """Titled pane that snap-docks into a column. Preview may Pop out to a Toplevel.

    ``wm manage`` / ``wm forget`` turn the Preview frame into a tool window and back
    without rebuilding widgets — PhotoImage labels stay the same.
    Title-bar drag snaps panes between left, right-top, and right-bottom.
    """

    def __init__(
        self,
        column: ScrollColumn,
        app: "WallpaperRecolorApp",
        title: str,
        *,
        pop_label: str = "Pop out",
        dock_label: str = "Dock",
        pane_weight: int = 1,
        pane_minsize: int = 80,
        float_size: str = "480x240",
        allow_pop_out: bool = False,
        flex: bool = False,
    ) -> None:
        # True child of the column interior so pack() stacks and the pane HWND clips.
        super().__init__(column.inner, bd=1, relief="groove")
        self.app = app
        self.column = column
        self.panel_title = title
        self.pop_label = pop_label
        self.dock_label = dock_label
        self.pane_weight = pane_weight
        self.pane_minsize = pane_minsize
        self.float_size = float_size
        self.allow_pop_out = allow_pop_out
        self.flex = flex
        self.hidden = False
        self.is_floating = False
        self._home_column = column
        self._last_column = column
        self._last_index: int | None = None
        self._float_geom: str | None = None
        self._press_root: tuple[int, int] | None = None
        self._win_off: tuple[int, int] | None = None
        self._rearranging = False
        self._btn_text = tk.StringVar(value=pop_label)
        self._pop_btn: ttk.Button | None = None
        self.expanded = True
        self._build_chrome()

    def _build_chrome(self) -> None:
        """Title bar + body as true children of this shell (canvas clips the tree)."""
        bar = tk.Frame(self, bg=_PANEL_BAR_BG, cursor="fleur")
        self.bar = bar
        bar.pack(fill="x")
        self._twisty = tk.Label(
            bar,
            text="▼" if self.expanded else "▶",
            bg=_PANEL_BAR_BG,
            fg=_PANEL_BAR_FG,
            font=("Segoe UI", 8),
            cursor="hand2",
        )
        self._twisty.pack(side="left", padx=(6, 0), pady=3)
        self._twisty.bind("<ButtonPress-1>", self._on_twisty)
        title_lbl = tk.Label(
            bar,
            text=self.panel_title,
            bg=_PANEL_BAR_BG,
            fg=_PANEL_BAR_FG,
            font=("Segoe UI", 9, "bold"),
            cursor="fleur",
        )
        title_lbl.pack(side="left", padx=8, pady=3)
        if self.allow_pop_out:
            self._pop_btn = ttk.Button(bar, textvariable=self._btn_text, width=16, command=self.toggle)
            self._pop_btn.pack(side="right", padx=4, pady=2)
        self._bind_drag(bar)
        self._bind_drag(title_lbl)
        self.body = ttk.Frame(self, padding=4)
        if self.expanded:
            self.body.pack(fill="both", expand=True)

    def remount_into(self, inner: tk.Misc) -> None:
        """Recreate this shell as a true child of ``inner`` (Tk 8.6 cannot reparent)."""
        if self.is_floating:
            return
        try:
            if self.master is inner:
                return
        except tk.TclError:
            pass
        self.pack_forget()
        bar_was, body_was = self.bar, self.body
        try:
            exists_bar = bool(bar_was.winfo_exists())
            exists_body = bool(body_was.winfo_exists())
        except tk.TclError:
            exists_bar = exists_body = False
        try:
            self.tk.call("destroy", self._w)
        except tk.TclError:
            pass
        tk.Frame.__init__(self, inner, bd=1, relief="groove")
        try:
            still_bar = exists_bar and bool(bar_was.winfo_exists())
            still_body = exists_body and bool(body_was.winfo_exists())
        except tk.TclError:
            still_bar = still_body = False
        if still_bar and still_body:
            self.bar = bar_was
            self.body = body_was
            self.bar.pack(fill="x")
            if self.expanded:
                self.body.pack(fill="both", expand=True)
        else:
            self._build_chrome()

    def _on_twisty(self, _event=None) -> str:
        self.set_expanded(not self.expanded)
        return "break"

    def set_expanded(self, on: bool) -> None:
        """Show or hide ``body``; the title bar always stays."""
        self.expanded = bool(on)
        try:
            self._twisty.configure(text="▼" if self.expanded else "▶")
        except tk.TclError:
            pass
        if self.expanded:
            if str(self.body.winfo_manager()) != "pack":
                self.body.pack(fill="both", expand=True)
        else:
            self.body.pack_forget()
        col = getattr(self, "column", None)
        if col is not None:
            col._repack()

    def _bind_drag(self, widget: tk.Misc) -> None:
        """Title bar: drag to snap between columns; OS chrome still resizes when Preview is popped out."""
        widget.bind("<ButtonPress-1>", self._on_bar_press)
        widget.bind("<B1-Motion>", self._on_bar_drag)
        widget.bind("<ButtonRelease-1>", self._on_bar_release)

    def toggle(self) -> None:
        if self.is_floating:
            self.dock()
        else:
            self.pop_out()

    def pop_out(self) -> None:
        """Detach Preview into a toolwindow Toplevel (same widgets, live preview intact)."""
        if self.is_floating or not self.allow_pop_out:
            return
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        geom = self._float_geom
        if not geom:
            geom = f"{w}x{h}" if w >= 40 and h >= 40 else self.float_size
        self.pack_forget()
        self.column._untag_tree(self)
        self.is_floating = True
        self._btn_text.set(self.dock_label)
        self.column._repack()
        self.app.root.wm_manage(self)
        tk.Wm.wm_title(self, f"{self.panel_title} — Wallpaper Recolor")
        tk.Wm.wm_protocol(self, "WM_DELETE_WINDOW", self.dock)
        tk.Wm.wm_minsize(self, 280, 80)
        tk.Wm.wm_geometry(self, geom)
        try:
            tk.Wm.wm_attributes(self, "-toolwindow", True)
        except tk.TclError:
            pass

    def dock(self) -> None:
        """Put this pane back in its column (close on the Toplevel does this)."""
        if not self.is_floating:
            return
        try:
            self._float_geom = str(tk.Wm.wm_geometry(self))
        except tk.TclError:
            pass
        self.app.root.wm_forget(self)
        self.is_floating = False
        self._btn_text.set(self.pop_label)
        self.column._repack()

    def _unfloat_for_snap(self) -> None:
        """Leave toolwindow mode so the pane can pack into a column."""
        if not self.is_floating:
            return
        try:
            self._float_geom = str(tk.Wm.wm_geometry(self))
        except tk.TclError:
            pass
        self.app.root.wm_forget(self)
        self.is_floating = False
        self._btn_text.set(self.pop_label)

    def _on_bar_press(self, event) -> None:
        self._press_root = (event.x_root, event.y_root)
        self._win_off = None
        self._rearranging = False
        if self.is_floating:
            rx = int(self.winfo_rootx())
            ry = int(self.winfo_rooty())
            self._win_off = (event.x_root - rx, event.y_root - ry)

    def _on_bar_drag(self, event) -> None:
        if self._press_root is None:
            return
        if not self.is_floating:
            dx = event.x_root - self._press_root[0]
            dy = event.y_root - self._press_root[1]
            if dx * dx + dy * dy < _UNDOCK_DRAG_PX * _UNDOCK_DRAG_PX:
                return
            self._rearranging = True
            self.app._update_snap_target(event.x_root, event.y_root, self)
            return
        ox, oy = self._win_off if self._win_off is not None else (48, 14)
        tk.Wm.wm_geometry(self, f"+{event.x_root - ox}+{event.y_root - oy}")
        self.app._update_snap_target(event.x_root, event.y_root, self)

    def _on_bar_release(self, event) -> None:
        target = self.app._hit_column(event.x_root, event.y_root)
        if self.is_floating:
            if target is not None:
                col, idx = target
                self._unfloat_for_snap()
                self.app._place_panel(self, col, idx)
            self.app._clear_snap()
            self._press_root = None
            self._win_off = None
            self._rearranging = False
            return
        if self._rearranging:
            if target is not None:
                col, idx = target
                self.app._place_panel(self, col, idx)
            elif self.allow_pop_out:
                # Preview only: drop off both columns pops it into its own window
                self.pop_out()
                tk.Wm.wm_geometry(self, f"+{event.x_root - 48}+{event.y_root - 14}")
            # Texture / Coverage / Tone / Scale / Crop / Tessellate / Labels / Color wheel stay in their column
        elif not self.is_floating:
            self.set_expanded(not self.expanded)
        self.app._clear_snap()
        self._press_root = None
        self._win_off = None
        self._rearranging = False

