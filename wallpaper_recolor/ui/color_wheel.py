# -*- coding: utf-8 -*-
"""
wallpaper_recolor.ui.color_wheel
-----------------------------
HSV/HSL color wheel modeled on https://htmlcolorcodes.com/color-wheel/

Inner disk: hue = angle, saturation = distance from center.
Outer ring: lightness for the current hue/saturation.
White handles (circles) drag to change the color. Linked Hex / Pantone / RGBAO
fields sit under the history strip; typing in one updates the other two.
Under the wheel: a 20-slot recent-color strip (two rows of 10), then Tailwind
(50–900) plus Shades / Tints / Tones mix bars.

Class references (code + name only):
- CAP3321C Data Wrangling
"""

from __future__ import annotations  # tuple[int, int, int] without quoting

import math
import re
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk

from wallpaper_recolor.color.color_math import hls_array_to_rgb, hsl_to_rgb, rgb_to_hsl
from wallpaper_recolor.color.pantone import (
    filter_pantone_codes,
    lookup_pantone_rgb,
    pantone_code_for_rgb,
)
from wallpaper_recolor.ui.tooltip import bind_tooltip

# Panel fill behind the wheel (htmlcolorcodes-style light page)
WHEEL_BG = (245, 245, 245)
DISK_RADIUS = 112  # inner HS disk
RING_INNER = 118  # gap between disk and lightness ring
RING_OUTER = 148
CANVAS_SIZE = 320  # square canvas in pixels
HANDLE_R = 8  # white circle handles, same idea as htmlcolorcodes
BAR_H = 22  # mix-bar canvas height (room for the black-dot thumb)
THUMB_R = 6
MIX_BLACK = (0, 0, 0)
MIX_WHITE = (255, 255, 255)
MIX_GRAY = (128, 128, 128)  # medium gray, htmlcolorcodes tones left end
# Recent colors: two rows × 10, most-recent first; empty cells are chrome only
COLOR_HISTORY_COLS = 10
COLOR_HISTORY_ROWS = 2
COLOR_HISTORY_MAX = COLOR_HISTORY_COLS * COLOR_HISTORY_ROWS
COLOR_HISTORY_CELL_H = 16
COLOR_HISTORY_GAP = 2
COLOR_HISTORY_EMPTY = "#e8e8e8"
COLOR_HISTORY_EMPTY_OUTLINE = "#d0d0d0"
# Tailwind 50–900 (no 950): typical HSL lightness ramp, pale → dark
TAILWIND_STOPS = (50, 100, 200, 300, 400, 500, 600, 700, 800, 900)
TAILWIND_LIGHTNESS = (0.97, 0.94, 0.86, 0.77, 0.66, 0.58, 0.48, 0.39, 0.30, 0.22)
UNKNOWN_PANTONE = "Unknown Pantone code"
OPAQUE_ALPHA = 255  # recode pipeline is RGB; A is display-only, 255 = opaque
PANTONE_SUGGEST_LIMIT = 16
_COMPLETE_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")
_MIX_TIPS = {
    "tailwind": "Tailwind 50–900 steps of this hue and saturation.",
    "shades": "Mix toward black.",
    "tints": "Mix toward white.",
    "tones": "Mix toward medium gray.",
}


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    """CSS #RRGGBB."""
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def hex_to_rgb(text: str) -> tuple[int, int, int] | None:
    """Parse #RGB or #RRGGBB. None if the string is not a color."""
    raw = text.strip().lstrip("#")
    if len(raw) == 3 and all(c in "0123456789abcdefABCDEF" for c in raw):
        return int(raw[0] * 2, 16), int(raw[1] * 2, 16), int(raw[2] * 2, 16)
    if len(raw) == 6 and all(c in "0123456789abcdefABCDEF" for c in raw):
        return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)
    return None


def rgb_text_to_rgb(text: str) -> tuple[int, int, int] | None:
    """Parse ``r, g, b`` (commas or spaces). Each channel must be 0–255."""
    parts = [p for p in re.split(r"[,\s]+", text.strip()) if p]
    if len(parts) != 3:
        return None
    try:
        rgb = (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None
    if any(c < 0 or c > 255 for c in rgb):
        return None
    return rgb


def rgb_to_text(rgb: tuple[int, int, int]) -> str:
    """``r, g, b`` (kept for tests / call sites that still want three channels)."""
    return f"{rgb[0]}, {rgb[1]}, {rgb[2]}"


def rgbao_to_text(rgb: tuple[int, int, int], alpha: int = OPAQUE_ALPHA) -> str:
    """``r, g, b, a`` with A 0–255 (255 = opaque)."""
    a = 0 if alpha < 0 else 255 if alpha > 255 else int(alpha)
    return f"{rgb[0]}, {rgb[1]}, {rgb[2]}, {a}"


def rgbao_text_to_rgba(text: str) -> tuple[int, int, int, int] | None:
    """Parse ``r, g, b`` or ``r, g, b, a``. A defaults to 255 when omitted."""
    parts = [p for p in re.split(r"[,\s]+", text.strip()) if p]
    if len(parts) not in (3, 4):
        return None
    try:
        vals = tuple(int(p) for p in parts)
    except ValueError:
        return None
    if any(c < 0 or c > 255 for c in vals):
        return None
    if len(vals) == 3:
        return vals[0], vals[1], vals[2], OPAQUE_ALPHA
    return vals[0], vals[1], vals[2], vals[3]


def mix_rgb(
    start: tuple[int, int, int], end: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    """Linear sRGB mix: ``t=0`` → start, ``t=1`` → end."""
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return (
        int(round(start[0] + (end[0] - start[0]) * t)),
        int(round(start[1] + (end[1] - start[1]) * t)),
        int(round(start[2] + (end[2] - start[2]) * t)),
    )


def shade_rgb(base: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    """Shades: left ``t=0`` is the base, right ``t=1`` is black."""
    return mix_rgb(base, MIX_BLACK, t)


def tint_rgb(base: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    """Tints: left ``t=0`` is white, right ``t=1`` is the base."""
    return mix_rgb(MIX_WHITE, base, t)


def tone_rgb(base: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    """Tones: left ``t=0`` is medium gray, right ``t=1`` is the base."""
    return mix_rgb(MIX_GRAY, base, t)


def tailwind_palette(h: float, s: float) -> list[tuple[int, int, int]]:
    """Ten Tailwind-like steps (50–900) of hue ``h`` at saturation ``s``."""
    return [hsl_to_rgb(h, s, light) for light in TAILWIND_LIGHTNESS]


def _clamp_rgb_channel(value: object) -> int:
    try:
        channel = int(value)
    except (TypeError, ValueError):
        return 0
    return 0 if channel < 0 else 255 if channel > 255 else channel


def coerce_rgb(value: object) -> tuple[int, int, int] | None:
    """``(r, g, b)`` 0–255, or ``None`` if ``value`` is not a triple."""
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    return (
        _clamp_rgb_channel(value[0]),
        _clamp_rgb_channel(value[1]),
        _clamp_rgb_channel(value[2]),
    )


def parse_color_history(raw: object, *, limit: int = COLOR_HISTORY_MAX) -> list[tuple[int, int, int]]:
    """RGB triples from JSON / tests; invalid rows skipped; capped at ``limit``."""
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[tuple[int, int, int]] = []
    for item in raw:
        rgb = coerce_rgb(item)
        if rgb is None:
            continue
        out.append(rgb)
        if len(out) >= limit:
            break
    return out


def push_color_history(
    history: list[tuple[int, int, int]],
    rgb: tuple[int, int, int],
    *,
    limit: int = COLOR_HISTORY_MAX,
) -> list[tuple[int, int, int]]:
    """Most-recent first. Skip if already front; move an older duplicate up; cap ``limit``."""
    coerced = coerce_rgb(rgb)
    if coerced is None:
        return list(history)
    if history and history[0] == coerced:
        return list(history)
    rest = [c for c in history if c != coerced]
    return [coerced] + rest[: max(0, limit - 1)]


def render_wheel(h: float, s: float, l: float, size: int = CANVAS_SIZE) -> np.ndarray:
    """RGB uint8 image of the HS disk + lightness ring at the current HSL."""
    cx = cy = (size - 1) / 2.0
    yy, xx = np.indices((size, size), dtype=np.float32)
    dx = xx - cx
    dy = cy - yy  # +y up so hue 0 (red) sits on the +x axis like a math wheel
    dist = np.sqrt(dx * dx + dy * dy)
    angle = np.arctan2(dy, dx)  # −π … π
    hue = (angle / (2.0 * math.pi)) % 1.0  # 0 at +x (red), CCW

    bg = np.empty((size, size, 3), dtype=np.uint8)
    bg[:, :] = WHEEL_BG

    # Inner disk: saturation grows from the center; lightness is the current L
    disk = dist <= DISK_RADIUS
    sat = np.clip(dist / DISK_RADIUS, 0.0, 1.0)
    if np.any(disk):
        # Vectorized HLS → RGB for the disk pixels only
        h_pix = hue[disk]
        s_pix = sat[disk]
        l_pix = np.full(h_pix.shape, l, dtype=np.float32)
        rgb = hls_array_to_rgb(h_pix, l_pix, s_pix)
        bg[disk] = rgb

    # Outer ring: same H/S, lightness walks the full 0–1 circle
    ring = (dist >= RING_INNER) & (dist <= RING_OUTER)
    if np.any(ring):
        light = (hue[ring])  # 0 at red/+x, full turn = black→white
        h_ring = np.full(light.shape, h, dtype=np.float32)
        s_ring = np.full(light.shape, s, dtype=np.float32)
        rgb = hls_array_to_rgb(h_ring, light, s_ring)
        bg[ring] = rgb

    return bg


class ColorWheel(ttk.Frame):
    """Interactive htmlcolorcodes-style wheel with linked Hex / Pantone / RGBAO, history, mix bars."""

    def __init__(
        self,
        parent,
        on_color: Callable[[tuple[int, int, int]], None] | None = None,
        on_color_commit: Callable[[tuple[int, int, int]], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.on_color = on_color
        self.on_color_commit = on_color_commit  # ButtonRelease / hex / bar — not every drag tick
        self._h, self._s, self._l = 0.0, 1.0, 0.5
        self._drag: str | None = None  # "disk" | "ring"
        self._photo: ImageTk.PhotoImage | None = None
        self._mute_ui = False  # skip value commit while we write the field
        self._bar_drag: str | None = None  # "tailwind" | "shades" | "tints" | "tones"
        self._bar_base: tuple[int, int, int] | None = None  # frozen while a thumb is dragged
        self._mix_t = {"shades": 0.0, "tints": 1.0, "tones": 1.0}  # thumbs at pure base
        self._tailwind_index = 5  # 500
        self._mix_photos: dict[str, ImageTk.PhotoImage] = {}
        self._mix_bars: dict[str, tk.Canvas] = {}
        self._history: list[tuple[int, int, int]] = []
        self._alpha = OPAQUE_ALPHA  # UI-only; recode uses RGB
        self._pantone_popup: tk.Toplevel | None = None
        self._pantone_list: tk.Listbox | None = None
        self._pantone_focus_job: str | None = None

        self.canvas = tk.Canvas(
            self,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="#%02x%02x%02x" % WHEEL_BG,
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        bind_tooltip(self.canvas, self._wheel_canvas_tip)

        self._build_history()

        meta = ttk.Frame(self)
        meta.pack(fill="x", pady=(8, 0))

        self.swatch = tk.Label(meta, width=6, height=2, relief="groove", bd=1)
        self.swatch.grid(row=0, column=0, rowspan=3, padx=(0, 8), sticky="n")

        fields = ttk.Frame(meta)
        fields.grid(row=0, column=1, columnspan=2, sticky="ew")
        fields.columnconfigure(1, weight=1)
        self.hex_var = tk.StringVar(value="#FF0000")
        self.pantone_var = tk.StringVar(value="")
        self.rgbao_var = tk.StringVar(value="255, 0, 0, 255")
        self.input_status = tk.StringVar(value="")

        self.hex_label, self.hex_entry = self._add_color_field(
            fields, 0, "Hex", self.hex_var, 12
        )
        self.pantone_label, self.pantone_entry = self._add_color_field(
            fields, 1, "Pantone", self.pantone_var, 18
        )
        self.rgbao_label, self.rgbao_entry = self._add_color_field(
            fields, 2, "RGBAO", self.rgbao_var, 18
        )
        bind_tooltip(self.hex_entry, "Type #RRGGBB. Linked with Pantone and RGBAO.")
        bind_tooltip(
            self.pantone_entry,
            "Start typing a code for suggestions. Linked with Hex and RGBAO.",
        )
        bind_tooltip(
            self.rgbao_entry,
            "r, g, b, a (0–255). A is display-only; recode uses RGB.",
        )
        bind_tooltip(self.swatch, "Current color for the selected match or change-to half.")

        self.hex_entry.bind("<Return>", self._hex_committed)
        self.hex_entry.bind("<FocusOut>", self._hex_committed)
        self.hex_entry.bind("<KeyRelease>", self._hex_keyrelease)
        self.pantone_entry.bind("<Return>", self._pantone_return)
        self.pantone_entry.bind("<FocusOut>", self._on_pantone_focus_out)
        self.pantone_entry.bind("<KeyRelease>", self._pantone_keyrelease)
        self.pantone_entry.bind("<Down>", self._pantone_arrow_down)
        self.pantone_entry.bind("<Escape>", self._on_pantone_escape)
        self.rgbao_entry.bind("<Return>", self._rgbao_committed)
        self.rgbao_entry.bind("<FocusOut>", self._rgbao_committed)
        self.bind("<Destroy>", self._on_destroy, add="+")

        self.hsl_label = ttk.Label(meta, text="")
        self.hsl_label.grid(row=1, column=1, columnspan=2, sticky="w", pady=(2, 0))
        self.status_label = ttk.Label(meta, textvariable=self.input_status, foreground="#a33")
        self.status_label.grid(row=2, column=1, columnspan=2, sticky="w")

        self._build_mix_bars()

        self._redraw_wheel()
        self._sync_readouts()

    # ---------------------------------------------------------------------------
    # Public
    # ---------------------------------------------------------------------------
    def set_rgb(self, rgb: tuple[int, int, int], notify: bool = False) -> None:
        """Move the handles to ``rgb`` without a feedback loop unless ``notify``."""
        self._h, self._s, self._l = rgb_to_hsl(rgb)
        self._redraw_wheel()
        self._sync_readouts()
        if notify:
            self._emit()

    def current_rgb(self) -> tuple[int, int, int]:
        return hsl_to_rgb(self._h, self._s, self._l)

    def current_rgba(self) -> tuple[int, int, int, int]:
        r, g, b = self.current_rgb()
        return r, g, b, self._alpha

    def history_colors(self) -> list[tuple[int, int, int]]:
        """Most-recent first; empty slots are not included."""
        return list(self._history)

    def set_history(self, colors: object) -> None:
        """Replace the strip from saved state (no wheel / undo notify)."""
        self._history = parse_color_history(colors)
        self._paint_history()

    def record_color(self, rgb: tuple[int, int, int]) -> None:
        """Push a committed RGB onto the strip (dedupe / cap). Live drag must not call this."""
        self._history = push_color_history(self._history, rgb)
        self._paint_history()

    def pick_history(self, index: int) -> None:
        """Apply a filled swatch like a wheel click; move it to most-recent."""
        if index < 0 or index >= len(self._history):
            return
        rgb = self._history[index]
        self.set_rgb(rgb, notify=True)
        self._commit()

    # ---------------------------------------------------------------------------
    # Draw / hit-test
    # ---------------------------------------------------------------------------
    def _redraw_wheel(self) -> None:
        arr = render_wheel(self._h, self._s, self._l)
        self._photo = ImageTk.PhotoImage(Image.fromarray(arr, mode="RGB"))
        self.canvas.delete("wheel")
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw", tags="wheel")
        self._draw_handles()

    def _center(self) -> float:
        """Same origin as ``render_wheel``: (size-1)/2, not size/2."""
        return (CANVAS_SIZE - 1) / 2.0

    def _draw_handles(self) -> None:
        """White circles on the disk (H/S) and ring (lightness)."""
        self.canvas.delete("handle")
        cx = cy = self._center()
        # Disk handle: sat along radius, hue as angle (0 = +x)
        ang = self._h * 2.0 * math.pi
        dx = math.cos(ang) * self._s * DISK_RADIUS
        dy = -math.sin(ang) * self._s * DISK_RADIUS
        self._oval(cx + dx, cy + dy, HANDLE_R, hsl_to_rgb(self._h, self._s, self._l))
        # Ring handle: lightness is the same angle mapping as hue on the ring
        ring_r = (RING_INNER + RING_OUTER) / 2.0
        lang = self._l * 2.0 * math.pi
        rx = math.cos(lang) * ring_r
        ry = -math.sin(lang) * ring_r
        self._oval(cx + rx, cy + ry, HANDLE_R, hsl_to_rgb(self._h, self._s, self._l))

    def _oval(self, x: float, y: float, r: int, fill: tuple[int, int, int]) -> None:
        self.canvas.create_oval(
            x - r,
            y - r,
            x + r,
            y + r,
            outline="#111111",
            width=2,
            fill=rgb_to_hex(fill),
            tags="handle",
        )

    def _hit(self, event) -> str | None:
        cx = cy = self._center()
        dx, dy = event.x - cx, cy - event.y
        dist = math.hypot(dx, dy)
        if dist <= DISK_RADIUS + 4:
            return "disk"
        if RING_INNER - 4 <= dist <= RING_OUTER + 4:
            return "ring"
        return None

    def _wheel_canvas_tip(self, event=None) -> str:
        zone = self._hit(event) if event is not None else None
        if zone == "disk":
            return "Hue (angle) and saturation (distance from center)."
        if zone == "ring":
            return "Lightness for the current hue and saturation."
        return "Inner disk: hue and saturation. Outer ring: lightness."

    def _press(self, event) -> None:
        zone = self._hit(event)
        if zone is None:
            return
        self._drag = zone
        self._apply_pointer(event)

    def _drag_move(self, event) -> None:
        if self._drag:
            self._apply_pointer(event)

    def _release(self, _event) -> None:
        was_dragging = self._drag is not None
        self._drag = None
        if was_dragging:
            self._commit()

    def _apply_pointer(self, event) -> None:
        cx = cy = self._center()
        dx, dy = event.x - cx, cy - event.y
        dist = math.hypot(dx, dy)
        ang = math.atan2(dy, dx)  # same convention as render_wheel
        hue = (ang / (2.0 * math.pi)) % 1.0
        if self._drag == "disk":
            self._h = hue
            self._s = min(dist / DISK_RADIUS, 1.0)
        elif self._drag == "ring":
            self._l = hue  # ring uses the same 0–1 angle as lightness
        self._redraw_wheel()
        self._sync_readouts()
        self._emit()

    def _sync_readouts(self) -> None:
        rgb = self.current_rgb()
        hex_color = rgb_to_hex(rgb)
        pantone = pantone_code_for_rgb(rgb, closest=True) or ""
        rgbao = rgbao_to_text(rgb, self._alpha)
        self._mute_ui = True
        if self.hex_var.get().upper() != hex_color:
            self.hex_var.set(hex_color)
        if self.pantone_var.get() != pantone:
            self.pantone_var.set(pantone)
        if self.rgbao_var.get() != rgbao:
            self.rgbao_var.set(rgbao)
        self._mute_ui = False
        self.input_status.set("")
        self.swatch.configure(bg=hex_color)
        self.hsl_label.configure(
            text=(
                f"H {int(self._h * 360):3d}°   "
                f"S {int(self._s * 100):3d}%   "
                f"L {int(self._l * 100):3d}%   "
                f"RGB {rgb[0]}, {rgb[1]}, {rgb[2]}"
            )
        )
        if self._bar_drag is None:
            self._reset_mix_thumbs()
            self._paint_mix_bars()

    # ---------------------------------------------------------------------------
    # Color history strip
    # ---------------------------------------------------------------------------
    def _build_history(self) -> None:
        """Two rows × 10 recent swatches packed directly under the wheel canvas."""
        host = ttk.Frame(self)
        host.pack(fill="x", pady=(8, 0))
        self.history_title = tk.StringVar(value="History")
        ttk.Label(host, textvariable=self.history_title).pack(anchor="w", pady=(0, 1))
        height = (
            COLOR_HISTORY_ROWS * COLOR_HISTORY_CELL_H
            + (COLOR_HISTORY_ROWS - 1) * COLOR_HISTORY_GAP
        )
        self.history_canvas = tk.Canvas(
            host,
            width=CANVAS_SIZE,
            height=height,
            bg="#%02x%02x%02x" % WHEEL_BG,
            highlightthickness=1,
            highlightbackground="#c8c8c8",
            cursor="arrow",
        )
        self.history_canvas.pack(fill="x")
        self.history_canvas.bind("<Button-1>", self._history_press)
        self.history_canvas.bind("<Motion>", self._history_motion)
        self.history_canvas.bind("<Leave>", self._history_leave)
        self.history_canvas.bind("<Configure>", lambda _e: self._paint_history())
        bind_tooltip(self.history_canvas, "Recent colors. Click a swatch to apply it.")
        self._paint_history()

    def _history_cell_geom(self) -> tuple[float, float]:
        width = max(int(self.history_canvas.winfo_width()), 2)
        height = max(int(self.history_canvas.winfo_height()), 2)
        gap = COLOR_HISTORY_GAP
        cell_w = (width - gap * (COLOR_HISTORY_COLS - 1)) / COLOR_HISTORY_COLS
        cell_h = (height - gap * (COLOR_HISTORY_ROWS - 1)) / COLOR_HISTORY_ROWS
        return cell_w, cell_h

    def _history_index_at(self, x: float, y: float) -> int | None:
        cell_w, cell_h = self._history_cell_geom()
        gap = COLOR_HISTORY_GAP
        if cell_w <= 0 or cell_h <= 0:
            return None
        stride_x = cell_w + gap
        stride_y = cell_h + gap
        col = int(x // stride_x)
        row = int(y // stride_y)
        if col < 0 or col >= COLOR_HISTORY_COLS or row < 0 or row >= COLOR_HISTORY_ROWS:
            return None
        local_x = x - col * stride_x
        local_y = y - row * stride_y
        if local_x > cell_w or local_y > cell_h:
            return None
        return row * COLOR_HISTORY_COLS + col

    def _paint_history(self) -> None:
        canvas = getattr(self, "history_canvas", None)
        if canvas is None:
            return
        canvas.delete("all")
        cell_w, cell_h = self._history_cell_geom()
        gap = COLOR_HISTORY_GAP
        for i in range(COLOR_HISTORY_MAX):
            row, col = divmod(i, COLOR_HISTORY_COLS)
            x0 = col * (cell_w + gap)
            y0 = row * (cell_h + gap)
            filled = i < len(self._history)
            canvas.create_rectangle(
                x0,
                y0,
                x0 + cell_w,
                y0 + cell_h,
                fill=rgb_to_hex(self._history[i]) if filled else COLOR_HISTORY_EMPTY,
                outline="#333333" if filled else COLOR_HISTORY_EMPTY_OUTLINE,
                tags=("swatch" if filled else "empty", f"h{i}"),
            )

    def _history_press(self, event) -> None:
        idx = self._history_index_at(event.x, event.y)
        if idx is None:
            return
        self.pick_history(idx)

    def _history_motion(self, event) -> None:
        idx = self._history_index_at(event.x, event.y)
        if idx is not None and idx < len(self._history):
            self.history_canvas.configure(cursor="hand2")
            self.history_title.set(f"History  {rgb_to_hex(self._history[idx])}")
        else:
            self.history_canvas.configure(cursor="arrow")
            self.history_title.set("History")

    def _history_leave(self, _event=None) -> None:
        self.history_canvas.configure(cursor="arrow")
        self.history_title.set("History")

    def _build_mix_bars(self) -> None:
        """Tailwind + Shades / Tints / Tones stacked under the history strip."""
        host = ttk.Frame(self)
        host.pack(fill="x", pady=(10, 0))
        for kind, title in (
            ("tailwind", "Tailwind"),
            ("shades", "Shades"),
            ("tints", "Tints"),
            ("tones", "Tones"),
        ):
            ttk.Label(host, text=title).pack(anchor="w", pady=(6, 1))
            bar = tk.Canvas(
                host,
                height=BAR_H,
                highlightthickness=1,
                highlightbackground="#c8c8c8",
                cursor="hand2",
                bg="#e8e8e8",
            )
            bar.pack(fill="x")
            bar.bind("<Button-1>", lambda e, k=kind: self._bar_press(k, e))
            bar.bind("<B1-Motion>", lambda e, k=kind: self._bar_move(k, e))
            bar.bind("<ButtonRelease-1>", lambda e, k=kind: self._bar_release(k, e))
            bar.bind("<Configure>", lambda e, k=kind: self._paint_mix_bar(k))
            bind_tooltip(bar, _MIX_TIPS[kind])
            self._mix_bars[kind] = bar

    def _mix_base(self) -> tuple[int, int, int]:
        return self._bar_base if self._bar_base is not None else self.current_rgb()

    def _reset_mix_thumbs(self) -> None:
        """Park continuous thumbs on the pure-base end; snap Tailwind to nearest L."""
        self._mix_t["shades"] = 0.0
        self._mix_t["tints"] = 1.0
        self._mix_t["tones"] = 1.0
        self._tailwind_index = min(
            range(len(TAILWIND_LIGHTNESS)),
            key=lambda i: abs(TAILWIND_LIGHTNESS[i] - self._l),
        )

    def _paint_mix_bars(self) -> None:
        for kind in self._mix_bars:
            self._paint_mix_bar(kind)

    def _paint_mix_bar(self, kind: str) -> None:
        bar = self._mix_bars.get(kind)
        if bar is None:
            return
        width = max(int(bar.winfo_width()), 2)
        height = BAR_H
        base = self._mix_base()
        bar.delete("all")
        if kind == "tailwind":
            self._paint_tailwind_cells(bar, width, height)
            n = len(TAILWIND_STOPS)
            cell = width / n
            tx = cell * (self._tailwind_index + 0.5)
        else:
            if kind == "shades":
                left, right = base, MIX_BLACK
                t = self._mix_t["shades"]
            elif kind == "tints":
                left, right = MIX_WHITE, base
                t = self._mix_t["tints"]
            else:
                left, right = MIX_GRAY, base
                t = self._mix_t["tones"]
            photo = self._gradient_photo(left, right, width, height)
            self._mix_photos[kind] = photo
            bar.create_image(0, 0, image=photo, anchor="nw")
            tx = THUMB_R + t * max(width - 2 * THUMB_R, 1)
        cy = height / 2.0
        bar.create_oval(
            tx - THUMB_R,
            cy - THUMB_R,
            tx + THUMB_R,
            cy + THUMB_R,
            fill="#111111",
            outline="#111111",
            tags="thumb",
        )

    def _paint_tailwind_cells(self, bar: tk.Canvas, width: int, height: int) -> None:
        h, s, _l = rgb_to_hsl(self._mix_base())
        palette = tailwind_palette(h, s)
        n = len(palette)
        gap = 1
        cell = (width - gap * (n - 1)) / n
        for i, rgb in enumerate(palette):
            x0 = i * (cell + gap)
            bar.create_rectangle(
                x0,
                0,
                x0 + cell,
                height,
                fill=rgb_to_hex(rgb),
                outline="",
            )

    def _gradient_photo(
        self,
        left: tuple[int, int, int],
        right: tuple[int, int, int],
        width: int,
        height: int,
    ) -> ImageTk.PhotoImage:
        t = np.linspace(0.0, 1.0, max(width, 1), dtype=np.float32)[None, :, None]
        start = np.array(left, dtype=np.float32)
        end = np.array(right, dtype=np.float32)
        row = start * (1.0 - t) + end * t
        arr = np.repeat(np.clip(np.rint(row), 0, 255).astype(np.uint8), height, axis=0)
        return ImageTk.PhotoImage(Image.fromarray(arr, mode="RGB"))

    def _bar_t(self, kind: str, x: float) -> float:
        width = max(int(self._mix_bars[kind].winfo_width()), 1)
        inner = max(width - 2 * THUMB_R, 1)
        t = (x - THUMB_R) / inner
        return 0.0 if t < 0.0 else 1.0 if t > 1.0 else t

    def _bar_color(self, kind: str, t: float) -> tuple[int, int, int]:
        base = self._mix_base()
        if kind == "shades":
            return shade_rgb(base, t)
        if kind == "tints":
            return tint_rgb(base, t)
        if kind == "tones":
            return tone_rgb(base, t)
        idx = int(round(t * (len(TAILWIND_STOPS) - 1)))
        idx = 0 if idx < 0 else min(idx, len(TAILWIND_STOPS) - 1)
        return tailwind_palette(self._h, self._s)[idx]

    def _apply_from_bar(self, rgb: tuple[int, int, int]) -> None:
        """Push a mix-bar color onto the wheel / hex / HSL without a hex echo."""
        self._mute_ui = True
        self._h, self._s, self._l = rgb_to_hsl(rgb)
        self._redraw_wheel()
        self._sync_readouts()
        self._mute_ui = False
        self._emit()

    def apply_mix(self, kind: str, t: float, *, commit: bool = False) -> None:
        """Set the active color from a mix-bar position (tests + drag)."""
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        if self._bar_base is None:
            self._bar_base = self.current_rgb()
        if kind == "tailwind":
            self.apply_tailwind_index(int(round(t * (len(TAILWIND_STOPS) - 1))), commit=commit)
            return
        self._mix_t[kind] = t
        rgb = self._bar_color(kind, t)
        self._apply_from_bar(rgb)
        self._paint_mix_bar(kind)
        if commit:
            self._finish_bar()

    def apply_tailwind_index(self, index: int, *, commit: bool = False) -> None:
        """Snap to a Tailwind 50–900 cell and optionally commit (one undo tick)."""
        n = len(TAILWIND_STOPS)
        self._tailwind_index = max(0, min(int(index), n - 1))
        if self._bar_base is None:
            self._bar_base = self.current_rgb()
        h, s, _l = rgb_to_hsl(self._bar_base)
        rgb = tailwind_palette(h, s)[self._tailwind_index]
        self._apply_from_bar(rgb)
        self._paint_mix_bar("tailwind")
        if commit:
            self._finish_bar()

    def _tailwind_index_at(self, x: float) -> int:
        width = max(int(self._mix_bars["tailwind"].winfo_width()), 1)
        idx = int(x * len(TAILWIND_STOPS) / width)
        return max(0, min(len(TAILWIND_STOPS) - 1, idx))

    def _bar_press(self, kind: str, event) -> None:
        self._bar_drag = kind
        self._bar_base = self.current_rgb()
        if kind == "tailwind":
            self.apply_tailwind_index(self._tailwind_index_at(event.x))
        else:
            self.apply_mix(kind, self._bar_t(kind, event.x))

    def _bar_move(self, kind: str, event) -> None:
        if self._bar_drag != kind:
            return
        if kind == "tailwind":
            self.apply_tailwind_index(self._tailwind_index_at(event.x))
        else:
            self.apply_mix(kind, self._bar_t(kind, event.x))

    def _bar_release(self, kind: str, _event) -> None:
        if self._bar_drag != kind:
            return
        self._finish_bar()

    def _finish_bar(self) -> None:
        """One undo tick, then rebuild bars from the new base color."""
        self._bar_drag = None
        self._bar_base = None
        self._commit()
        self._reset_mix_thumbs()
        self._paint_mix_bars()

    def _add_color_field(
        self, parent, row: int, name: str, var: tk.StringVar, width: int
    ) -> tuple[ttk.Label, ttk.Entry]:
        label = ttk.Label(parent, text=name)
        label.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=1)
        entry = ttk.Entry(parent, textvariable=var, width=width)
        entry.grid(row=row, column=1, sticky="ew", pady=1)
        return label, entry

    def _on_destroy(self, event) -> None:
        if event.widget is not self:
            return
        self._cancel_pantone_focus_job()
        self._hide_pantone_popup()

    def _apply_typed_rgb(
        self,
        rgb: tuple[int, int, int],
        *,
        commit: bool,
        alpha: int = OPAQUE_ALPHA,
    ) -> None:
        self.input_status.set("")
        self._alpha = 0 if alpha < 0 else 255 if alpha > 255 else int(alpha)
        self.set_rgb(rgb, notify=True)
        if commit:
            self._commit()

    def _hex_keyrelease(self, _event=None) -> None:
        """Apply a finished ``#RRGGBB`` while typing; leave partial strings alone."""
        if self._mute_ui:
            return
        text = self.hex_var.get().strip()
        if not _COMPLETE_HEX.match(text):
            return
        parsed = hex_to_rgb(text)
        if parsed is None:
            return
        self._apply_typed_rgb(parsed, commit=False)

    def _hex_committed(self, _event=None) -> None:
        if self._mute_ui:
            return
        parsed = hex_to_rgb(self.hex_var.get())
        if parsed is None:
            self._mute_ui = True
            self.hex_var.set(rgb_to_hex(self.current_rgb()))
            self._mute_ui = False
            return
        self._apply_typed_rgb(parsed, commit=True)

    def _value_committed(self, _event=None) -> None:
        """Back-compat alias: Enter / FocusOut on the Hex field."""
        self._hex_committed(_event)

    def _rgbao_committed(self, _event=None) -> None:
        if self._mute_ui:
            return
        parsed = rgbao_text_to_rgba(self.rgbao_var.get())
        if parsed is None:
            self._mute_ui = True
            self.rgbao_var.set(rgbao_to_text(self.current_rgb(), self._alpha))
            self._mute_ui = False
            return
        r, g, b, a = parsed
        self._apply_typed_rgb((r, g, b), commit=True, alpha=a)

    def _pantone_committed(self, _event=None) -> None:
        if self._mute_ui:
            return
        text = self.pantone_var.get().strip()
        if not text:
            self._mute_ui = True
            self.pantone_var.set(pantone_code_for_rgb(self.current_rgb(), closest=True) or "")
            self._mute_ui = False
            self.input_status.set("")
            return
        parsed = lookup_pantone_rgb(text)
        if parsed is None:
            self.input_status.set(UNKNOWN_PANTONE)
            return
        self._apply_typed_rgb(parsed, commit=True)

    def _pantone_return(self, _event=None):
        if self._pantone_list is not None and self._pantone_list.size():
            sel = self._pantone_list.curselection()
            idx = int(sel[0]) if sel else 0
            self._apply_pantone_suggestion(self._pantone_list.get(idx))
            return "break"
        self._hide_pantone_popup()
        self._pantone_committed()

    def _on_pantone_escape(self, _event=None):
        self._hide_pantone_popup()
        return "break"

    def _on_pantone_focus_out(self, _event=None) -> None:
        self._cancel_pantone_focus_job()
        self._pantone_focus_job = self.after(120, self._pantone_focus_out_later)

    def _cancel_pantone_focus_job(self) -> None:
        job = self._pantone_focus_job
        if job is not None:
            try:
                self.after_cancel(job)
            except tk.TclError:
                pass
            self._pantone_focus_job = None

    def _pantone_focus_out_later(self) -> None:
        self._pantone_focus_job = None
        try:
            focused = self.focus_get()
        except tk.TclError:
            focused = None
        popup = self._pantone_popup
        if popup is not None and focused is not None:
            try:
                if str(focused) == str(popup) or str(focused).startswith(str(popup)):
                    return
            except tk.TclError:
                pass
        self._hide_pantone_popup()
        self._pantone_committed()

    def _pantone_keyrelease(self, event=None) -> None:
        if self._mute_ui:
            return
        if event is not None and event.keysym in (
            "Return",
            "Escape",
            "Up",
            "Down",
            "Tab",
        ):
            return
        self._update_pantone_popup(filter_pantone_codes(self.pantone_var.get()))

    def _pantone_arrow_down(self, _event=None):
        if self._pantone_popup is None:
            self._update_pantone_popup(filter_pantone_codes(self.pantone_var.get()))
        if self._pantone_list is not None and self._pantone_list.size():
            self._pantone_list.selection_clear(0, tk.END)
            self._pantone_list.selection_set(0)
            self._pantone_list.activate(0)
            self._pantone_list.focus_set()
        return "break"

    def _update_pantone_popup(self, matches: list[str]) -> None:
        matches = matches[:PANTONE_SUGGEST_LIMIT]
        if not matches:
            self._hide_pantone_popup()
            return
        if self._pantone_popup is None:
            self._create_pantone_popup()
        listbox = self._pantone_list
        assert listbox is not None
        listbox.delete(0, tk.END)
        for item in matches:
            listbox.insert(tk.END, item)
        listbox.selection_clear(0, tk.END)
        listbox.selection_set(0)
        listbox.activate(0)
        listbox.configure(height=min(len(matches), 8))
        self._place_pantone_popup()

    def _create_pantone_popup(self) -> None:
        popup = tk.Toplevel(self)
        popup.overrideredirect(True)
        popup.withdraw()
        listbox = tk.Listbox(
            popup,
            exportselection=False,
            activestyle="dotbox",
            height=8,
            width=28,
        )
        listbox.pack(fill="both", expand=True)
        listbox.bind("<ButtonRelease-1>", self._on_pantone_list_click)
        listbox.bind("<Return>", self._pantone_return)
        listbox.bind("<Escape>", self._on_pantone_escape)
        listbox.bind("<Up>", self._on_pantone_list_up)
        self._pantone_popup = popup
        self._pantone_list = listbox

    def _place_pantone_popup(self) -> None:
        popup = self._pantone_popup
        if popup is None:
            return
        self.update_idletasks()
        x = int(self.pantone_entry.winfo_rootx())
        y = int(self.pantone_entry.winfo_rooty() + self.pantone_entry.winfo_height())
        width = max(int(self.pantone_entry.winfo_width()), 180)
        popup.geometry(f"{width}x{popup.winfo_reqheight()}+{x}+{y}")
        popup.deiconify()
        popup.lift()

    def _hide_pantone_popup(self) -> None:
        popup = self._pantone_popup
        if popup is None:
            return
        try:
            popup.destroy()
        except tk.TclError:
            pass
        self._pantone_popup = None
        self._pantone_list = None

    def _on_pantone_list_click(self, event) -> None:
        listbox = self._pantone_list
        if listbox is None:
            return
        idx = listbox.nearest(event.y)
        if idx < 0:
            return
        self._apply_pantone_suggestion(listbox.get(idx))

    def _on_pantone_list_up(self, _event=None):
        listbox = self._pantone_list
        if listbox is None:
            return
        sel = listbox.curselection()
        if sel and int(sel[0]) == 0:
            self.pantone_entry.focus_set()
            return "break"
        return None

    def _apply_pantone_suggestion(self, shown: str) -> None:
        self._cancel_pantone_focus_job()
        self._hide_pantone_popup()
        self._mute_ui = True
        self.pantone_var.set(shown)
        self._mute_ui = False
        self._pantone_committed()

    def _emit(self, _event=None) -> None:
        if self.on_color:
            self.on_color(self.current_rgb())

    def _commit(self) -> None:
        """Record recent color + one undo tick after drag, value Enter, or mix-bar release."""
        rgb = self.current_rgb()
        self.record_color(rgb)
        if self.on_color_commit:
            self.on_color_commit(rgb)
