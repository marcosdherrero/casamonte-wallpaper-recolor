# -*- coding: utf-8 -*-
"""
wallpaper_recolor.ui.mixins.swatch
----------------------------------
Alternate room photo with a physical Pantone chip: sample the chip, confirm
the code, set Color & lighting Temperature / Tint / Exposure on the wallpaper.

Class references (code + name only):
- CAP3321C Data Wrangling
"""

from __future__ import annotations

from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import numpy as np
from PIL import Image, ImageTk

from wallpaper_recolor.color.pantone import filter_pantone_codes, lookup_pantone_rgb
from wallpaper_recolor.color.swatch_match import (
    SWATCH_SAMPLE_RADIUS,
    correction_for_pantone,
    guess_swatch_pantone,
    mean_rgb_at,
    mean_rgb_region,
)
from wallpaper_recolor.io.image_io import OPEN_FILETYPES, load_image
from wallpaper_recolor.ui.color_wheel import UNKNOWN_PANTONE, rgb_to_hex
from wallpaper_recolor.ui.coverage_bar import HALF_REPLACE
from wallpaper_recolor.ui.snapshot import _fit
from wallpaper_recolor.ui.tooltip import bind_tooltip

_SWATCH_PREVIEW_MAX = 560
_SWATCH_DRAG_MIN = 6


class AppSwatchMixin:
    """Swatch photo window, sample/marquee, Match swatch on Color & lighting."""

    def _ensure_swatch_state(self) -> None:
        if getattr(self, "swatch_pantone_var", None) is not None:
            return
        self.swatch_pantone_var = tk.StringVar(value="")
        self._swatch_image: Image.Image | None = None
        self._swatch_path: Path | None = None
        self._swatch_rgb: np.ndarray | None = None
        self._swatch_sampled_rgb: tuple[int, int, int] | None = None
        self._swatch_win: tk.Toplevel | None = None
        self._swatch_photo: ImageTk.PhotoImage | None = None
        self._swatch_place = (0, 0, 1, 1)
        self._swatch_drag: tuple[int, int] | None = None
        self._swatch_rect = None

    def _build_swatch_controls(self) -> None:
        """File menu item plus Color & lighting buttons (after layout exists)."""
        self._ensure_swatch_state()
        menu = getattr(self, "file_menu", None)
        if menu is not None:
            try:
                labels = [
                    str(menu.entrycget(i, "label"))
                    for i in range(int(menu.index("end") or -1) + 1)
                    if str(menu.type(i)) == "command"
                ]
            except tk.TclError:
                labels = []
            if "Open swatch photo…" not in labels:
                menu.insert_command(
                    1,
                    label="Open swatch photo…",
                    command=self.open_swatch_photo,
                )
        wp = getattr(self, "white_patch_btn", None)
        host = wp.master.master if wp is not None else getattr(self, "tone_panel", None)
        if host is None:
            return
        if hasattr(host, "body"):
            host = host.body
        row = ttk.Frame(host)
        row.pack(fill="x", pady=(2, 2))
        self.swatch_photo_btn = ttk.Button(
            row, text="Swatch photo…", command=self._on_swatch_photo_button
        )
        self.swatch_photo_btn.pack(side="left")
        bind_tooltip(
            self.swatch_photo_btn,
            "Open a room photo with a Pantone chip. Click the chip, confirm the code, "
            "then Match swatch.",
        )
        self.match_swatch_btn = ttk.Button(
            row, text="Match swatch", command=self._on_match_swatch
        )
        self.match_swatch_btn.pack(side="left", padx=(6, 0))
        bind_tooltip(
            self.match_swatch_btn,
            "Sets Temperature / Tint / Exposure so the sampled chip matches the "
            "official Pantone. You can still nudge the numbers.",
        )

    def open_swatch_photo(self) -> None:
        """File → Open swatch photo… — room photo with a physical Pantone chip."""
        if self._busy:
            return
        self._ensure_swatch_state()
        path = filedialog.askopenfilename(
            title="Open swatch photo",
            filetypes=OPEN_FILETYPES,
        )
        if not path:
            return
        self._load_swatch_image(Path(path))

    def _on_swatch_photo_button(self) -> None:
        """Color & lighting: reopen the last photo, or ask for one."""
        self._ensure_swatch_state()
        if self._swatch_image is not None:
            self._show_swatch_window()
            return
        self.open_swatch_photo()

    def _load_swatch_image(self, path: Path) -> bool:
        self._ensure_swatch_state()
        try:
            image = load_image(path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not open image", str(exc), parent=self.root)
            return False
        rgb = image.convert("RGB")
        self._swatch_path = Path(path)
        self._swatch_image = rgb
        self._swatch_rgb = np.asarray(rgb)
        self._swatch_sampled_rgb = None
        self._show_swatch_window()
        self._sync_swatch_dialog()
        self.status.set(f"Swatch photo {path.name} — click the chip.")
        return True

    def _show_swatch_window(self) -> None:
        self._ensure_swatch_state()
        win = self._swatch_win
        if win is not None:
            try:
                if win.winfo_exists():
                    self._refresh_swatch_preview()
                    win.deiconify()
                    win.lift()
                    return
            except tk.TclError:
                self._swatch_win = None
        win = tk.Toplevel(self.root)
        win.title("Swatch photo")
        win.transient(self.root)
        self._swatch_win = win
        win.protocol("WM_DELETE_WINDOW", self._on_swatch_window_close)

        hint = ttk.Label(
            win,
            text="Click or drag on the chip. Confirm the Pantone code, then Match swatch.",
        )
        hint.pack(anchor="w", padx=8, pady=(8, 4))

        self._swatch_canvas = tk.Canvas(win, highlightthickness=0, cursor="crosshair")
        self._swatch_canvas.pack(padx=8, pady=4)
        self._swatch_canvas.bind("<ButtonPress-1>", self._on_swatch_press)
        self._swatch_canvas.bind("<B1-Motion>", self._on_swatch_drag)
        self._swatch_canvas.bind("<ButtonRelease-1>", self._on_swatch_release)

        meta = ttk.Frame(win)
        meta.pack(fill="x", padx=8, pady=(0, 4))
        ttk.Label(meta, text="Sampled").pack(side="left")
        self._swatch_chip = tk.Label(meta, width=4, height=1, relief="groove", bd=1)
        self._swatch_chip.pack(side="left", padx=(6, 8))
        self._swatch_rgb_label = ttk.Label(meta, text="—")
        self._swatch_rgb_label.pack(side="left")

        pantone_row = ttk.Frame(win)
        pantone_row.pack(fill="x", padx=8, pady=(0, 4))
        ttk.Label(pantone_row, text="Pantone").pack(side="left")
        self.swatch_pantone_combo = ttk.Combobox(
            pantone_row,
            textvariable=self.swatch_pantone_var,
            width=18,
        )
        self.swatch_pantone_combo.pack(side="left", padx=(6, 0), fill="x", expand=True)
        self.swatch_pantone_combo.bind("<KeyRelease>", self._on_swatch_pantone_key, add="+")
        bind_tooltip(
            self.swatch_pantone_combo,
            "Code of the physical chip. Typing suggests catalog names. "
            "Sample fills the nearest.",
        )
        guess_btn = ttk.Button(pantone_row, text="Guess", command=self._on_swatch_guess)
        guess_btn.pack(side="left", padx=(6, 0))
        bind_tooltip(guess_btn, "Nearest table code to the sampled chip.")

        btns = ttk.Frame(win)
        btns.pack(fill="x", padx=8, pady=(4, 8))
        open_btn = ttk.Button(btns, text="Open…", command=self.open_swatch_photo)
        open_btn.pack(side="left")
        bind_tooltip(open_btn, "Replace this photo with another room / chip shot.")
        use_btn = ttk.Button(
            btns, text="Use as change-to", command=self._on_swatch_use_change_to
        )
        use_btn.pack(side="right")
        bind_tooltip(
            use_btn,
            "Copy the official Pantone RGB onto the selected range’s change-to.",
        )
        match_btn = ttk.Button(btns, text="Match swatch", command=self._on_match_swatch)
        match_btn.pack(side="right", padx=(0, 6))
        bind_tooltip(
            match_btn,
            "Grade the wallpaper so this chip would match the official Pantone.",
        )
        self._swatch_status = ttk.Label(win, text="")
        self._swatch_status.pack(anchor="w", padx=8, pady=(0, 8))
        self._refresh_swatch_preview()
        self._sync_swatch_dialog()

    def _on_swatch_window_close(self) -> None:
        win = self._swatch_win
        self._swatch_win = None
        if win is not None:
            try:
                win.destroy()
            except tk.TclError:
                pass

    def _refresh_swatch_preview(self) -> None:
        canvas = getattr(self, "_swatch_canvas", None)
        image = self._swatch_image
        if canvas is None or image is None:
            return
        preview = _fit(image, _SWATCH_PREVIEW_MAX)
        self._swatch_photo = ImageTk.PhotoImage(preview, master=self.root)
        w, h = preview.size
        canvas.configure(width=w, height=h)
        canvas.delete("all")
        canvas.create_image(0, 0, image=self._swatch_photo, anchor="nw")
        self._swatch_place = (0, 0, w, h)
        self._swatch_rect = None

    def _on_swatch_pantone_key(self, _event=None) -> None:
        combo = getattr(self, "swatch_pantone_combo", None)
        if combo is None:
            return
        matches = filter_pantone_codes(self.swatch_pantone_var.get(), limit=16)
        combo.configure(values=matches)

    def _on_swatch_guess(self) -> None:
        sampled = self._swatch_sampled_rgb
        if sampled is None:
            self._set_swatch_status("Click the chip first.")
            return
        code = guess_swatch_pantone(sampled)
        self.swatch_pantone_var.set(code)
        self._set_swatch_status(f"Nearest: {code}" if code else UNKNOWN_PANTONE)

    def _canvas_to_swatch_xy(self, cx: int, cy: int) -> tuple[int, int] | None:
        rgb = self._swatch_rgb
        image = self._swatch_image
        if rgb is None or image is None:
            return None
        ox, oy, dw, dh = self._swatch_place
        if dw < 1 or dh < 1:
            return None
        px = cx - ox
        py = cy - oy
        if px < 0 or py < 0 or px >= dw or py >= dh:
            return None
        src_w, src_h = image.size
        x = min(src_w - 1, max(0, int(px * src_w / dw)))
        y = min(src_h - 1, max(0, int(py * src_h / dh)))
        return x, y

    def _on_swatch_press(self, event) -> None:
        self._swatch_drag = (int(event.x), int(event.y))
        canvas = getattr(self, "_swatch_canvas", None)
        if canvas is not None and self._swatch_rect is not None:
            canvas.delete(self._swatch_rect)
            self._swatch_rect = None

    def _on_swatch_drag(self, event) -> None:
        start = self._swatch_drag
        canvas = getattr(self, "_swatch_canvas", None)
        if start is None or canvas is None:
            return
        x0, y0 = start
        x1, y1 = int(event.x), int(event.y)
        if self._swatch_rect is None:
            self._swatch_rect = canvas.create_rectangle(
                x0, y0, x1, y1, outline="#f5f5f5", width=1
            )
        else:
            canvas.coords(self._swatch_rect, x0, y0, x1, y1)

    def _on_swatch_release(self, event) -> None:
        start = self._swatch_drag
        self._swatch_drag = None
        rgb = self._swatch_rgb
        if start is None or rgb is None:
            return
        x1, y1 = int(event.x), int(event.y)
        dx = abs(x1 - start[0])
        dy = abs(y1 - start[1])
        if dx >= _SWATCH_DRAG_MIN or dy >= _SWATCH_DRAG_MIN:
            a = self._canvas_to_swatch_xy(*start)
            b = self._canvas_to_swatch_xy(x1, y1)
            if a is None or b is None:
                return
            sampled = mean_rgb_region(rgb, a[0], a[1], b[0], b[1])
        else:
            mapped = self._canvas_to_swatch_xy(x1, y1)
            if mapped is None:
                return
            sampled = mean_rgb_at(rgb, mapped[0], mapped[1], SWATCH_SAMPLE_RADIUS)
        self._apply_swatch_sample(sampled)

    def _apply_swatch_sample(self, rgb: tuple[int, int, int]) -> None:
        self._swatch_sampled_rgb = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        if not self.swatch_pantone_var.get().strip():
            self.swatch_pantone_var.set(guess_swatch_pantone(self._swatch_sampled_rgb))
        self._sync_swatch_dialog()
        self._set_swatch_status("Chip sampled. Confirm the Pantone code, then Match swatch.")

    def _sync_swatch_dialog(self) -> None:
        sampled = self._swatch_sampled_rgb
        chip = getattr(self, "_swatch_chip", None)
        label = getattr(self, "_swatch_rgb_label", None)
        if sampled is None:
            if chip is not None:
                chip.configure(bg="#e8e8e8")
            if label is not None:
                label.configure(text="—")
            return
        if chip is not None:
            chip.configure(bg=rgb_to_hex(sampled))
        if label is not None:
            label.configure(text=f"{sampled[0]}, {sampled[1]}, {sampled[2]}")

    def _set_swatch_status(self, text: str) -> None:
        status = getattr(self, "_swatch_status", None)
        if status is not None:
            try:
                if status.winfo_exists():
                    status.configure(text=text)
            except tk.TclError:
                pass

    def _resolved_swatch_pantone(self) -> str:
        self._ensure_swatch_state()
        code = self.swatch_pantone_var.get().strip()
        if code:
            return code
        sampled = self._swatch_sampled_rgb
        if sampled is None:
            return ""
        guessed = guess_swatch_pantone(sampled)
        if guessed:
            self.swatch_pantone_var.set(guessed)
        return guessed

    def _on_match_swatch(self) -> None:
        """Sampled chip vs official Pantone → Temperature / Tint / Exposure."""
        if self._mute_ui:
            return
        self._ensure_swatch_state()
        sampled = self._swatch_sampled_rgb
        if sampled is None:
            self._on_swatch_photo_button()
            return
        code = self._resolved_swatch_pantone()
        amounts = correction_for_pantone(sampled, code) if code else None
        if amounts is None:
            self._set_swatch_status(UNKNOWN_PANTONE)
            self.status.set(UNKNOWN_PANTONE)
            return
        temp, tint, exposure = amounts
        before = self._capture_edit()
        prev = self._mute_ui
        self._mute_ui = True
        try:
            self.temperature_pct.set(float(round(float(temp) * 100.0)))
            self.tint_pct.set(float(round(float(tint) * 100.0)))
            self.exposure_pct.set(float(round(float(exposure) * 100.0)))
        finally:
            self._mute_ui = prev
        self._on_tone_slider("")
        self._push_undo_state(before)
        shown = code.strip() or "Pantone"
        msg = f"Color & lighting matched to {shown}."
        self.status.set(msg)
        self._set_swatch_status(msg)

    def _on_swatch_use_change_to(self) -> None:
        """Official Pantone RGB → selected range’s change-to."""
        self._ensure_swatch_state()
        code = self._resolved_swatch_pantone()
        target = lookup_pantone_rgb(code) if code else None
        if target is None:
            self._set_swatch_status(UNKNOWN_PANTONE)
            return
        if self.range_map is None:
            self._set_swatch_status("Open a wallpaper first.")
            return
        self.selected_half = HALF_REPLACE
        self._apply_eyedrop_rgb(target)
        self._set_swatch_status(f"Change-to set to {code}.")
        self.status.set(f"Change-to set to {code}.")
