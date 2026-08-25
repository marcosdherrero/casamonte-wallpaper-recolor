# -*- coding: utf-8 -*-
"""
wallpaper_recolor.ui.coverage_bar
------------------------------
Coverage weights sit above a header and one diagonal match/change row.

The header row splits into N columns (eyes, clickable coverage %, dividers).
Click a column to select that range; drag a divider to steal from the ranges
on either side of that handle; click a percent to type a weight (luma / L* /
a* / b*: bar-adjacent steal; color closeness: Lab-nearest cluster). An eye
hides that range (transparent holes in Result).

One color row below: each segment is two colors split by the full-bar
diagonal from bottom-left to top-right:
- Left / above: match-from (Range by Color) or luma-key gray + luma %
  (Range by Luma) — not coverage weight.
- Right / below (includes the bottom-right corner): change-to replacement.
Click left vs right of the diagonal to set ``selected_half``. Yellow outline
on the selected range; the FA dropper is drawn in the selected triangle.

Range chips live as Layers sublayers, not a second strip under Coverage.

Class references (code + name only):
- CAP3321C Data Wrangling
"""

from __future__ import annotations  # Callable hints

from collections.abc import Callable, Sequence

import tkinter as tk
from tkinter import ttk

from wallpaper_recolor.color.color_ranges import MIN_COVERAGE, steal_from_adjacent
from wallpaper_recolor.ui.color_wheel import rgb_to_hex
from wallpaper_recolor.ui.tooltip import bind_tooltip

MIN_WEIGHT = MIN_COVERAGE  # keep every range at least 3% so a color does not vanish
HEAD_H = 28  # eye + coverage % above the diagonal color row
SEG_H = 32  # weighted row of corner-to-corner two-color bars
HANDLE_SLOP = 7  # px — easier to grab a divider
EYE_W = 18  # click target at the left of a column
DROP_PAD = 6  # keep the dropper inside the selected triangle
DROP_MARGIN = 16  # dropper inset from the right of a triangle
HALF_MATCH = "match"  # left / above the BL→TR diagonal
HALF_REPLACE = "replace"  # right / below — paint color
HEAD_FILL = "#e8e8e8"
# Hover copy (not packed as labels). Keep out of __init__ so layout tests stay stable.
TIP_EYE = "Show or hide this range. Hidden ranges knock out of Result."
TIP_PCT = (
    "Coverage weight. Drag a divider to steal from the bar neighbor; "
    "click the percent to type a value (color closeness: Lab-nearest)."
)
TIP_DIVIDER = "Drag to steal coverage from the range on the other side of this handle."
TIP_MATCH = "Match from image — colors this range selects."
TIP_LUMA = "Luma key — brightness this range matches."
TIP_REPLACE = "Change-to — replacement color for this range."
TIP_DROP = "Eyedropper: sample Original into the selected match or change-to half."
TIP_HEADER = "Click to select this range."
HEAD_FILL_SEL = "#d4d4d4"
SEL_OUTLINE = "#ffcc33"


def _canvas_width(canvas: tk.Canvas) -> int:
    """Allocated width, or the requested width before the widget is mapped."""
    return max(int(canvas.winfo_width()), int(canvas.winfo_reqwidth()), 40)


def _half_at_diagonal(x0: float, x1: float, height: float, px: float, py: float) -> str:
    """Match is left/above the bottom-left → top-right diagonal; replace is right/below."""
    width = x1 - x0
    if width <= 1e-6:
        return HALF_REPLACE
    # Line from (x0, height) to (x1, 0). Canvas y grows downward, so above = smaller py.
    line_y = height * (1.0 - (px - x0) / width)
    return HALF_MATCH if py <= line_y else HALF_REPLACE


def _dropper_xy(x0: float, x1: float, height: float, half: str) -> tuple[float, float]:
    """Centroid of the match (UL) or replace (BR) triangle, clamped inside the segment."""
    width = x1 - x0
    if half == HALF_MATCH:
        cx = x0 + width / 3.0
        cy = height / 3.0
    else:
        cx = x0 + 2.0 * width / 3.0
        cy = 2.0 * height / 3.0
    pad = min(DROP_PAD, max(2.0, width / 4.0), max(2.0, height / 4.0))
    cx = max(x0 + pad, min(x1 - pad, cx))
    cy = max(pad, min(height - pad, cy))
    return cx, cy


class CoverageBar(ttk.Frame):
    """Eye/% header + diagonal weight row + two full-width stacked swatches."""

    def __init__(
        self,
        parent,
        on_weights: Callable[..., None] | None = None,
        on_select: Callable[[int, str], None] | None = None,
        on_toggle_visible: Callable[[int], None] | None = None,
        on_percent_commit: Callable[[int, float], None] | None = None,
        on_edit_begin: Callable[[], None] | None = None,
        on_edit_end: Callable[[], None] | None = None,
        on_eyedrop: Callable[[], None] | None = None,
        eyedrop_photo=None,
        eye_on_photo=None,
        eye_off_photo=None,
    ) -> None:
        super().__init__(parent)
        self.on_weights = on_weights
        self.on_select = on_select
        self.on_toggle_visible = on_toggle_visible
        self.on_percent_commit = on_percent_commit
        self.on_edit_begin = on_edit_begin
        self.on_edit_end = on_edit_end
        self.on_eyedrop = on_eyedrop
        self.eyedrop_photo = eyedrop_photo  # FA dropper — selected triangle + stacked swatch
        self.min_coverage = MIN_COVERAGE
        self.eye_on_photo = eye_on_photo  # FA solid eye — range visible
        self.eye_off_photo = eye_off_photo  # FA slash eye — range hidden
        self._eye_photos: list = []  # keep Canvas image refs
        self.weights: list[float] = [0.5, 0.5]
        self.match_colors: list[tuple[int, int, int]] = [(180, 180, 180), (80, 80, 80)]
        self.replace_colors: list[tuple[int, int, int]] = [(180, 180, 180), (80, 80, 80)]
        self.visibilities: list[bool] = [True, True]
        self.luma_keys: list[float] = [0.25, 0.75]  # 0–1 Rec. 709 key; not coverage
        self.luma_mode = False
        self.selected = 0
        self.selected_half = HALF_REPLACE
        self._drag_div: int | None = None  # divider index between ranges i and i+1
        self._eye_hits: list[tuple[int, float, float]] = []
        self._pct_hits: list[tuple[int, float, float, float, float]] = []
        self._seg_hits: list[tuple[int, float, float]] = []
        self._pct_entry: tk.Entry | None = None
        self._pct_index: int | None = None

        # Header: eyes + coverage weights. Not a color bar.
        self.bar = tk.Canvas(
            self, height=HEAD_H, width=320, highlightthickness=0, cursor="sb_h_double_arrow"
        )
        self.bar.pack(fill="x", pady=(0, 4))
        self.bar.bind("<Button-1>", self._press)
        self.bar.bind("<B1-Motion>", self._move)
        self.bar.bind("<ButtonRelease-1>", self._release)
        self.bar.bind("<Configure>", lambda _e: self.redraw())

        self.segments = tk.Canvas(self, height=SEG_H, width=320, highlightthickness=0, cursor="hand2")
        self.segments.pack(fill="x", pady=(0, 0))
        self.segments.bind("<Button-1>", self._press_seg)
        self.segments.bind("<Configure>", lambda _e: self._paint_segments())

        # Dropper lives inside the selected triangle; never a sibling or extra strip.
        self.eyedrop_btn: tk.Label | None = None
        bind_tooltip(self.bar, self._header_tip)
        bind_tooltip(self.segments, self._segment_tip)

    # ---------------------------------------------------------------------------
    # State
    # ---------------------------------------------------------------------------
    def set_state(
        self,
        weights: Sequence[float],
        match_colors: Sequence[tuple[int, int, int]],
        replace_colors: Sequence[tuple[int, int, int]] | None = None,
        selected: int = 0,
        visibilities: Sequence[bool] | None = None,
        selected_half: str = HALF_REPLACE,
        luma_mode: bool = False,
        luma_keys: Sequence[float] | None = None,
        min_coverage: float | None = None,
    ) -> None:
        """Refresh header + diagonal color row from the range map. Does not notify."""
        self.weights = [float(w) for w in weights]
        self.match_colors = [tuple(c) for c in match_colors]  # type: ignore[misc]
        if replace_colors is None:
            self.replace_colors = list(self.match_colors)
        else:
            self.replace_colors = [tuple(c) for c in replace_colors]  # type: ignore[misc]
        if visibilities is None:
            self.visibilities = [True] * len(self.weights)
        else:
            self.visibilities = [bool(v) for v in visibilities]
            if len(self.visibilities) < len(self.weights):
                self.visibilities.extend([True] * (len(self.weights) - len(self.visibilities)))
        self.luma_mode = bool(luma_mode)
        if min_coverage is not None:
            try:
                self.min_coverage = max(0.0, min(0.40, float(min_coverage)))
            except (TypeError, ValueError):
                self.min_coverage = MIN_COVERAGE
        if luma_keys is None:
            self.luma_keys = [0.5] * len(self.weights)
        else:
            self.luma_keys = [max(0.0, min(1.0, float(k))) for k in luma_keys]
            if len(self.luma_keys) < len(self.weights):
                self.luma_keys.extend([0.5] * (len(self.weights) - len(self.luma_keys)))
        if self.weights:
            if int(selected) < 0:
                self.selected = -1
            else:
                self.selected = max(0, min(int(selected), len(self.weights) - 1))
        self.selected_half = selected_half if selected_half in (HALF_MATCH, HALF_REPLACE) else HALF_REPLACE
        self.redraw()

    def redraw(self) -> None:
        self._cancel_percent_edit()
        self._paint_bar()
        self._paint_segments()

    def _header_tip(self, event=None) -> str:
        """Eye / percent / divider under the header pointer."""
        if event is None:
            return TIP_PCT
        x = float(getattr(event, "x", 0))
        y = float(getattr(event, "y", 0))
        for _i, x0, x1 in self._eye_hits:
            if x0 <= x <= x1:
                return TIP_EYE
        for _i, x0, y0, x1, y1 in self._pct_hits:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return TIP_PCT
        width = _canvas_width(self.bar)
        for _i, dx in enumerate(self._divider_xs(width)):
            if abs(x - dx) <= HANDLE_SLOP:
                return TIP_DIVIDER
        return TIP_HEADER

    def _match_tip_text(self) -> str:
        return TIP_LUMA if self.luma_mode else TIP_MATCH

    def _segment_tip(self, event=None) -> str:
        """Match triangle, change-to triangle, or in-bar eyedropper."""
        if event is None:
            return self._match_tip_text()
        if self._dropper_hit(self.segments, event):
            return TIP_DROP
        if not self.weights:
            return self._match_tip_text()
        hit: tuple[int, float, float] | None = None
        for i, x0, x1 in self._seg_hits:
            if x0 <= event.x <= x1:
                hit = (i, x0, x1)
                break
        if hit is None and self._seg_hits:
            i, x0, x1 = self._seg_hits[-1]
            if event.x >= x0:
                hit = (i, x0, x1)
        if hit is None:
            return self._match_tip_text()
        _i, x0, x1 = hit
        half = _half_at_diagonal(x0, x1, SEG_H, float(event.x), float(event.y))
        if half == HALF_MATCH:
            return self._match_tip_text()
        return TIP_REPLACE

    def _match_swatch_tip(self, event=None) -> str:
        return self._match_tip_text()

    def _replace_swatch_tip(self, event=None) -> str:
        return TIP_REPLACE

    # ---------------------------------------------------------------------------
    # Header — eye + coverage % (columns sized by weight)
    # ---------------------------------------------------------------------------
    def _paint_bar(self) -> None:
        """One row of eye + coverage %; not a from/to color bar."""
        canvas = self.bar
        canvas.delete("all")
        width = _canvas_width(canvas)
        if not self.weights:
            return
        self._eye_hits = []
        self._pct_hits = []
        self._eye_photos = []
        x = 0.0
        n = len(self.weights)
        height = HEAD_H
        for i, w in enumerate(self.weights):
            x1 = width if i == n - 1 else x + w * width
            is_sel = i == self.selected
            fill = HEAD_FILL_SEL if is_sel else HEAD_FILL
            outline = "#111111" if is_sel else "#c8c8c8"
            shown = self.visibilities[i] if i < len(self.visibilities) else True
            canvas.create_rectangle(
                x, 0, x1, height, fill=fill, outline=outline, width=2 if is_sel else 1, tags=f"seg{i}"
            )
            if not shown:
                canvas.create_rectangle(x, 0, x1, height, fill="#888888", stipple="gray50", outline="")
            fg = "#888888" if not shown else "#111111"
            if x1 - x > 28:
                icon = self.eye_on_photo if shown else self.eye_off_photo
                if icon is not None:
                    canvas.create_image(x + 10, height / 2, image=icon)
                    self._eye_photos.append(icon)
                else:
                    canvas.create_text(
                        x + 10,
                        height / 2,
                        text="👁" if shown else "–",
                        fill=fg,
                        font=("Segoe UI", 9),
                    )
                self._eye_hits.append((i, x, x + EYE_W))
            if x1 - x > 44:
                pct = f"{w * 100:.0f}%"
                cx = (x + x1) / 2 + (4 if x1 - x > 56 else 0)
                canvas.create_text(cx, height / 2, text=pct, fill=fg, font=("Segoe UI", 9, "bold"), tags="coverpct")
                self._pct_hits.append((i, cx - 18, 0.0, cx + 18, float(height)))
            if i < n - 1:
                canvas.create_line(x1, 2, x1, height - 2, fill="#111111", width=3)
                canvas.create_rectangle(x1 - 3, height / 2 - 8, x1 + 3, height / 2 + 8, fill="#FFFFFF", outline="#111111")
            x = x1

    # ---------------------------------------------------------------------------
    # Percentage segments — match/luma (UL) and change-to (BR), corner-to-corner
    # ---------------------------------------------------------------------------
    def _paint_segments(self) -> None:
        """One weighted row: each range is match (left/above) / change-to (right/below)."""
        canvas = self.segments
        canvas.delete("all")
        self._seg_hits = []
        if not self.weights:
            return
        width = _canvas_width(canvas)
        height = SEG_H
        n = len(self.weights)
        x = 0.0
        for i, w in enumerate(self.weights):
            x1 = width if i == n - 1 else x + w * width
            self._seg_hits.append((i, x, x1))
            match = self.match_colors[i] if i < len(self.match_colors) else (128, 128, 128)
            repl = self.replace_colors[i] if i < len(self.replace_colors) else match
            if self.luma_mode:
                key = self.luma_keys[i] if i < len(self.luma_keys) else 0.5
                v = int(round(key * 255.0))
                match_rgb = (v, v, v)
                match_label = f"{key * 100:.0f}%"
            else:
                match_rgb = match
                match_label = None
            # UL triangle: top-left, top-right, bottom-left — hypotenuse is BL→TR.
            canvas.create_polygon(
                x,
                0,
                x1,
                0,
                x,
                height,
                fill=rgb_to_hex(match_rgb),
                outline="",
                tags=(f"seg{i}m", "match"),
            )
            # BR triangle: top-right, bottom-right, bottom-left — includes bottom-right.
            canvas.create_polygon(
                x1,
                0,
                x1,
                height,
                x,
                height,
                fill=rgb_to_hex(repl),
                outline="",
                tags=(f"seg{i}r", "replace"),
            )
            # Visible BL→TR seam so match-from and change-to stay distinct.
            canvas.create_line(
                x,
                height,
                x1,
                0,
                fill="#111111",
                width=2,
                tags=(f"seg{i}diag", "diag"),
            )
            shown = self.visibilities[i] if i < len(self.visibilities) else True
            if not shown:
                canvas.create_rectangle(x, 0, x1, height, fill="#888888", stipple="gray50", outline="")
            if match_label and (x1 - x) > 36:
                luma = 0.2126 * match_rgb[0] + 0.7152 * match_rgb[1] + 0.0722 * match_rgb[2]
                fg = "#FFFFFF" if luma < 140 else "#111111"
                lx = x + max(10.0, (x1 - x) * 0.22)
                ly = height * 0.28
                canvas.create_text(
                    lx, ly, text=match_label, fill=fg, font=("Segoe UI", 9, "bold"), tags="lumakey"
                )
            is_sel = i == self.selected
            if is_sel:
                canvas.create_rectangle(
                    x + 1,
                    1,
                    x1 - 1,
                    height - 1,
                    fill="",
                    outline=SEL_OUTLINE,
                    width=3,
                    tags=(f"seg{i}", "seloutline"),
                )
                if self.eyedrop_photo is not None:
                    dx, dy = _dropper_xy(x, x1, height, self.selected_half)
                    canvas.create_image(dx, dy, image=self.eyedrop_photo, tags="dropper")
            else:
                canvas.create_rectangle(
                    x, 0, x1, height, fill="", outline="#111111", width=1, tags=f"seg{i}"
                )
            x = x1

    # ---------------------------------------------------------------------------
    # Pointer — header columns, diagonal halves, eyedropper
    # ---------------------------------------------------------------------------
    def _divider_xs(self, width: int) -> list[float]:
        """Pixel x of each interior divider."""
        xs = []
        acc = 0.0
        for w in self.weights[:-1]:
            acc += w
            xs.append(acc * width)
        return xs

    def _press(self, event) -> None:
        width = _canvas_width(self.bar)
        for i, dx in enumerate(self._divider_xs(width)):
            if abs(event.x - dx) <= HANDLE_SLOP:
                self._drag_div = i
                self.bar.configure(cursor="sb_h_double_arrow")
                if self.on_edit_begin:
                    self.on_edit_begin()
                return
        for i, x0, x1 in self._eye_hits:
            if x0 <= event.x <= x1:
                self._select_index(i)
                if self.on_toggle_visible:
                    self.on_toggle_visible(i)
                return
        for i, x0, y0, x1, y1 in self._pct_hits:
            if x0 <= event.x <= x1 and y0 <= event.y <= y1:
                self._select_index(i)
                self._begin_percent_edit(i)
                return
        self._drag_div = None
        self._select_at(event.x, width)

    def _press_seg(self, event) -> None:
        """Click left/above vs right/below the BL→TR diagonal to set selected_half."""
        if not self.weights:
            return
        if self._dropper_hit(self.segments, event):
            if self.on_eyedrop:
                self.on_eyedrop()
            return
        hit: tuple[int, float, float] | None = None
        for i, x0, x1 in self._seg_hits:
            if x0 <= event.x <= x1:
                hit = (i, x0, x1)
                break
        if hit is None:
            # Empty coverage background — deselect so the wheel is scratch-only.
            self.selected = -1
            self.redraw()
            if self.on_select:
                self.on_select(-1, self.selected_half)
            return
        i, x0, x1 = hit
        half = _half_at_diagonal(x0, x1, SEG_H, float(event.x), float(event.y))
        self.selected = i
        self.selected_half = half
        self.redraw()
        if self.on_select:
            self.on_select(i, half)

    def _dropper_hit(self, canvas: tk.Canvas, event) -> bool:
        """True when the click is on the in-bar FA dropper."""
        if not canvas.find_withtag("dropper"):
            return False
        bbox = canvas.bbox("dropper")
        if bbox is None:
            return False
        x0, y0, x1, y1 = bbox
        return (x0 - 2) <= event.x <= (x1 + 2) and (y0 - 2) <= event.y <= (y1 + 2)

    def _select_index(self, index: int) -> None:
        """Select a range from the header; keep the current match/replace half."""
        self.selected = index
        self.redraw()
        if self.on_select:
            self.on_select(index, self.selected_half)

    def _select_at(self, x: int, width: int) -> None:
        """Click a header column to select that range; half stays until a bar click."""
        acc = 0.0
        for i, w in enumerate(self.weights):
            acc += w
            if x <= acc * width or i == len(self.weights) - 1:
                self._select_index(i)
                return

    def _begin_percent_edit(self, index: int) -> None:
        """Replace the % label with an Entry; Enter / focus-out commits."""
        self._cancel_percent_edit()
        width = _canvas_width(self.bar)
        x = 0.0
        cx = width / 2.0
        for i, w in enumerate(self.weights):
            x1 = width if i == len(self.weights) - 1 else x + w * width
            if i == index:
                cx = (x + x1) / 2.0
                break
            x = x1
        entry = tk.Entry(self.bar, width=5, justify="center", font=("Segoe UI", 9))
        entry.insert(0, f"{self.weights[index] * 100:.0f}")
        entry.select_range(0, tk.END)
        self.bar.create_window(cx, HEAD_H / 2, window=entry, tags="pctedit")
        self._pct_entry = entry
        self._pct_index = index
        entry.bind("<Return>", lambda _e: self._finish_percent_edit(commit=True))
        entry.bind("<FocusOut>", lambda _e: self._finish_percent_edit(commit=True))
        entry.bind("<Escape>", lambda _e: self._finish_percent_edit(commit=False))
        entry.focus_set()

    def _cancel_percent_edit(self) -> None:
        if self._pct_entry is not None:
            try:
                self._pct_entry.destroy()
            except tk.TclError:
                pass
        self.bar.delete("pctedit")
        self._pct_entry = None
        self._pct_index = None

    def _finish_percent_edit(self, *, commit: bool) -> None:
        entry = self._pct_entry
        index = self._pct_index
        raw = ""
        if entry is not None:
            raw = entry.get()
        self._cancel_percent_edit()
        if not commit or index is None:
            self.redraw()
            return
        try:
            pct = float(raw.strip().rstrip("%"))
        except ValueError:
            self.redraw()
            return
        if self.on_percent_commit:
            self.on_percent_commit(index, pct)
        else:
            self.redraw()

    def _move(self, event) -> None:
        if self._drag_div is None:
            return
        width = _canvas_width(self.bar)
        i = self._drag_div  # divider sits between range i and i+1
        left_before = sum(self.weights[:i])
        t = event.x / width
        new_i = t - left_before
        self.weights = steal_from_adjacent(
            self.weights, i, new_i, floor=getattr(self, "min_coverage", MIN_COVERAGE)
        )
        self.redraw()
        if self.on_weights:
            self.on_weights(self.weights, (i, i + 1))

    def _release(self, _event) -> None:
        if self._drag_div is not None and self.on_edit_end:
            self.on_edit_end()
        self._drag_div = None
