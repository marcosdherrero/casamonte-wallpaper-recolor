# -*- coding: utf-8 -*-
"""
wallpaper_recolor.ui.icons
------------------------------
Font Awesome SVG raster, eyedrop loupe, slider reset glyphs.

Class references (code + name only):
- CAP3321C Data Wrangling
- CAP4631C Machine Learning
- CAP4633C Machine Learning 2
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import base64
import math
import re
import tkinter as tk
import xml.etree.ElementTree as ET

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageTk

from wallpaper_recolor.ui.constants import (
    _EYE_ICON_FG,
    _EYE_ICON_PX,
    _EYE_OFF_PNG_NAME,
    _EYE_OFF_SVG_NAME,
    _EYE_ON_PNG_NAME,
    _EYE_ON_SVG_NAME,
    _EYEDROP_ICON_FG,
    _EYEDROP_ICON_HALO,
    _EYEDROP_ICON_PX,
    _EYEDROP_PNG_NAME,
    _EYEDROP_SVG_NAME,
    _ICONS_DIR,
    _LOUPE_GAP,
    _LOUPE_PX,
    _LOUPE_SRC_PX,
    _LOUPE_ZOOM,
    _RESET_ICON_FG,
    _RESET_ICON_PX,
    _RESET_PNG_NAME,
    _RESET_SVG_NAME,
    _ZOOM_ICON_PX,
    _ZOOM_IN_PNG_NAME,
    _ZOOM_IN_SVG_NAME,
    _ZOOM_OUT_PNG_NAME,
    _ZOOM_OUT_SVG_NAME,
)

def _flatten_cubic(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    steps: int = 16,
) -> list[tuple[float, float]]:
    """Line segments approximating a cubic Bezier (FA SVG path)."""
    pts: list[tuple[float, float]] = []
    for i in range(1, steps + 1):
        t = i / steps
        u = 1.0 - t
        x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
        y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
        pts.append((x, y))
    return pts


def _svg_path_contours(d: str) -> list[list[tuple[float, float]]]:
    """Absolute/relative M/L/C/Z path → closed polylines (Font Awesome glyphs)."""
    tok = re.findall(r"[MmLlCcZz]|[-+]?(?:\d+\.?\d*|\.\d+)", d)
    i = 0
    cmd: str | None = None
    cx = cy = 0.0
    start = (0.0, 0.0)
    pts: list[tuple[float, float]] = []
    contours: list[list[tuple[float, float]]] = []

    def nums(n: int) -> list[float]:
        nonlocal i
        out = [float(tok[i + k]) for k in range(n)]
        i += n
        return out

    def flush() -> None:
        nonlocal pts
        if len(pts) >= 3:
            if pts[-1] != start:
                pts.append(start)
            contours.append(pts)
        pts = []

    while i < len(tok):
        t = tok[i]
        if t.isalpha():
            cmd = t
            i += 1
            if cmd in "Zz":
                flush()
                continue
        if cmd is None:
            break
        if cmd in "Mm":
            if pts:
                flush()
            x, y = nums(2)
            if cmd == "m":
                x += cx
                y += cy
            cx, cy = x, y
            start = (cx, cy)
            pts.append((cx, cy))
            cmd = "L" if cmd == "M" else "l"
        elif cmd in "Ll":
            x, y = nums(2)
            if cmd == "l":
                x += cx
                y += cy
            cx, cy = x, y
            pts.append((cx, cy))
        elif cmd in "Cc":
            v = nums(6)
            if cmd == "c":
                v[0] += cx
                v[1] += cy
                v[2] += cx
                v[3] += cy
                v[4] += cx
                v[5] += cy
            p0 = (cx, cy)
            p3 = (v[4], v[5])
            pts.extend(_flatten_cubic(p0, (v[0], v[1]), (v[2], v[3]), p3))
            cx, cy = p3
        else:
            break
    flush()
    return contours


def _rasterize_fa_svg(
    svg_path: Path,
    size: int,
    fill: tuple[int, int, int, int],
    *,
    halo: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    """Pillow-fill FA path(s) with even-odd (no cairosvg). Supersample so small icons stay sharp.

    FA compound glyphs (eye pupil, dropper interior, plus/minus cutouts) are
    successive closed contours. XOR each contour onto the mask so holes stay
    transparent. Halo dilates that same mask, so it follows the glyph outline.
    """
    tree = ET.parse(svg_path)
    root_el = tree.getroot()
    d = ""
    for el in root_el.iter():
        if el.tag.endswith("path") and el.get("d"):
            d = str(el.get("d"))
            break
    vb = [float(x) for x in (root_el.get("viewBox") or "0 0 640 640").split()]
    vw, vh = vb[2], vb[3]
    ss = 4
    out_s = size * ss
    pad = ss
    sx = (out_s - 2 * pad) / vw
    sy = (out_s - 2 * pad) / vh
    mask = Image.new("L", (out_s, out_s), 0)
    for pts in _svg_path_contours(d):
        xy = [(pad + x * sx, pad + y * sy) for x, y in pts]
        if len(xy) < 3:
            continue
        layer = Image.new("L", (out_s, out_s), 0)
        ImageDraw.Draw(layer).polygon(xy, fill=255)
        mask = ImageChops.difference(mask, layer)
    img = Image.new("RGBA", (out_s, out_s), (0, 0, 0, 0))
    if halo is not None and halo[3] > 0:
        # 1px-class outline (MaxFilter 3 at 4×); a fatter dilate closes dropper / pupil holes
        halo_m = mask.filter(ImageFilter.MaxFilter(3))
        halo_layer = Image.new("RGBA", (out_s, out_s), halo)
        img = Image.composite(halo_layer, img, halo_m)
    fill_layer = Image.new("RGBA", (out_s, out_s), fill)
    img = Image.composite(fill_layer, img, mask)
    return img.resize((size, size), Image.Resampling.LANCZOS)


def _rasterize_rotate_left_svg(svg_path: Path, size: int = _RESET_ICON_PX) -> Image.Image:
    """Pillow-fill the FA path (no cairosvg). Supersample so 18px stays sharp."""
    return _rasterize_fa_svg(svg_path, size, _RESET_ICON_FG)


def _fa_icon_image(
    png_name: str,
    svg_name: str,
    size: int,
    fill: tuple[int, int, int, int],
    *,
    halo: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    """icons/ PNG if present, else rasterize the SVG and cache the PNG there."""
    _ICONS_DIR.mkdir(parents=True, exist_ok=True)
    png_path = _ICONS_DIR / png_name
    svg_path = _ICONS_DIR / svg_name
    if png_path.is_file():
        img = Image.open(png_path).convert("RGBA")
        if img.size != (size, size):
            img = img.resize((size, size), Image.Resampling.LANCZOS)
        if not _icon_has_opaque_plate(img):
            return img
        try:
            png_path.unlink()
        except OSError:
            pass
    img = _rasterize_fa_svg(svg_path, size, fill, halo=halo)
    try:
        img.save(png_path)
    except OSError:
        pass
    return img


def _icon_has_opaque_plate(img: Image.Image) -> bool:
    """True when corners are opaque — a filled rectangle around the glyph, not a halo."""
    rgba = img.convert("RGBA")
    w, h = rgba.size
    if w < 2 or h < 2:
        return False
    corners = ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))
    return any(int(rgba.getpixel(p)[3]) > 32 for p in corners)


def _tk_photo_png(img: Image.Image, master: tk.Misc) -> tk.PhotoImage | ImageTk.PhotoImage:
    """Tk PhotoImage from PNG bytes so transparent pixels are not composited on black."""
    rgba = img.convert("RGBA")
    buf = BytesIO()
    rgba.save(buf, format="PNG")
    try:
        return tk.PhotoImage(master=master, data=base64.b64encode(buf.getvalue()).decode("ascii"))
    except tk.TclError:
        return ImageTk.PhotoImage(rgba, master=master)


def _paste_rgba(dst: Image.Image, src: Image.Image, dest: tuple[int, int]) -> None:
    """Alpha-paste ``src`` onto ``dst`` at ``dest``, clipping to bounds."""
    dx, dy = dest
    sw, sh = src.size
    dw, dh = dst.size
    sx0 = max(0, -dx)
    sy0 = max(0, -dy)
    dx0 = max(0, dx)
    dy0 = max(0, dy)
    sx1 = min(sw, sx0 + (dw - dx0))
    sy1 = min(sh, sy0 + (dh - dy0))
    if sx1 <= sx0 or sy1 <= sy0:
        return
    piece = src.crop((sx0, sy0, sx1, sy1))
    dst.paste(piece, (dx0, dy0), piece)


def _make_eyedrop_loupe_image(src: Image.Image, px: int, py: int) -> Image.Image:
    """Circular 10× crop of ``src`` centered on pipette-tip pixel ``(px, py)``.

    ``(px, py)`` is the same display pixel ``_sample_original_rgb`` uses. The
    crosshair box is that pixel (11×11 neighborhood, 10× nearest). Out-of-bounds
    is host charcoal.
    """
    src_rgb = src.convert("RGB")
    sw, sh = src_rgb.size
    n = _LOUPE_SRC_PX
    z = _LOUPE_ZOOM
    out_s = n * z
    half = n // 2
    tile = Image.new("RGB", (n, n), (42, 42, 42))
    left, top = int(px) - half, int(py) - half
    src_l = max(0, left)
    src_t = max(0, top)
    src_r = min(sw, left + n)
    src_b = min(sh, top + n)
    if src_r > src_l and src_b > src_t:
        tile.paste(src_rgb.crop((src_l, src_t, src_r, src_b)), (src_l - left, src_t - top))
    zoomed = tile.resize((out_s, out_s), Image.Resampling.NEAREST)
    rgba = zoomed.convert("RGBA")
    draw = ImageDraw.Draw(rgba)
    c0 = half * z
    c1 = c0 + z - 1
    mid = (c0 + c1) // 2
    white, black = (255, 255, 255, 255), (20, 20, 20, 255)

    def _hair(a: tuple[int, int], b: tuple[int, int]) -> None:
        draw.line([a, b], fill=black, width=3)
        draw.line([a, b], fill=white, width=1)

    _hair((mid, 2), (mid, max(2, c0 - 2)))
    _hair((mid, min(out_s - 3, c1 + 2)), (mid, out_s - 3))
    _hair((2, mid), (max(2, c0 - 2), mid))
    _hair((min(out_s - 3, c1 + 2), mid), (out_s - 3, mid))
    draw.rectangle((c0 - 1, c0 - 1, c1 + 1, c1 + 1), outline=black)
    draw.rectangle((c0, c0, c1, c1), outline=white)
    mask = Image.new("L", (out_s, out_s), 0)
    ImageDraw.Draw(mask).ellipse((1, 1, out_s - 2, out_s - 2), fill=255)
    rgba.putalpha(mask)
    draw.ellipse((1, 1, out_s - 2, out_s - 2), outline=black)
    draw.ellipse((2, 2, out_s - 3, out_s - 3), outline=white)
    rgba.putalpha(mask)  # keep square corners fully transparent after rim strokes
    return rgba


def _glyph_hotspot(img: Image.Image) -> tuple[int, int]:
    """Lowest opaque pixel — FA eye-dropper pipette tip."""
    alpha = img.split()[-1]
    w, h = img.size
    px = alpha.load()
    for y in range(h - 1, -1, -1):
        for x in range(w):
            if px[x, y] > 32:
                return (x, y)
    return (0, h - 1)


def _reset_icon_photo(master: tk.Misc) -> ImageTk.PhotoImage:
    """18px rotate-left PhotoImage from icons/, or rasterized SVG."""
    img = _fa_icon_image(_RESET_PNG_NAME, _RESET_SVG_NAME, _RESET_ICON_PX, _RESET_ICON_FG)
    return ImageTk.PhotoImage(img, master=master)


def _eyedrop_icon_image() -> Image.Image:
    """22px FA eye-dropper RGBA (cached PNG or rasterized SVG), glyph only."""
    return _fa_icon_image(
        _EYEDROP_PNG_NAME,
        _EYEDROP_SVG_NAME,
        _EYEDROP_ICON_PX,
        _EYEDROP_ICON_FG,
        halo=_EYEDROP_ICON_HALO,
    )


def _eyedrop_icon_photo(master: tk.Misc) -> tuple[tk.PhotoImage, tuple[int, int]]:
    """22px eye-dropper PhotoImage plus pipette-tip hotspot (icons/ PNG or SVG)."""
    img = _eyedrop_icon_image()
    return _tk_photo_png(img, master), _glyph_hotspot(img)


def _eye_icon_photos(master: tk.Misc) -> tuple[ImageTk.PhotoImage, ImageTk.PhotoImage]:
    """Solid eye (shown) and slash-eye (hidden) for texture / range toggles."""
    on = _fa_icon_image(_EYE_ON_PNG_NAME, _EYE_ON_SVG_NAME, _EYE_ICON_PX, _EYE_ICON_FG)
    off = _fa_icon_image(_EYE_OFF_PNG_NAME, _EYE_OFF_SVG_NAME, _EYE_ICON_PX, _EYE_ICON_FG)
    return ImageTk.PhotoImage(on, master=master), ImageTk.PhotoImage(off, master=master)


def _zoom_icon_photos(master: tk.Misc) -> tuple[ImageTk.PhotoImage, ImageTk.PhotoImage]:
    """Magnifying-glass minus / plus for Crop zoom and Preview view-zoom."""
    out_img = _fa_icon_image(_ZOOM_OUT_PNG_NAME, _ZOOM_OUT_SVG_NAME, _ZOOM_ICON_PX, _EYE_ICON_FG)
    in_img = _fa_icon_image(_ZOOM_IN_PNG_NAME, _ZOOM_IN_SVG_NAME, _ZOOM_ICON_PX, _EYE_ICON_FG)
    return ImageTk.PhotoImage(out_img, master=master), ImageTk.PhotoImage(in_img, master=master)
