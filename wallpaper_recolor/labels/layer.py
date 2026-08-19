# -*- coding: utf-8 -*-
"""
wallpaper_recolor.labels.layer
------------------------------
Editable label layer: SVG ``<text>`` (Illustrator) and RGBA raster (Photoshop).

Positions and font size are source pixels, mapped through Crop onto the
export / preview size.

Class references (code + name only):
- CAP3321C Data Wrangling
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape
import os
import sys

from PIL import Image, ImageDraw, ImageFont

from wallpaper_recolor.labels.boxes import source_xy_to_display

LABEL_SIZE_MIN = 8
LABEL_SIZE_MAX = 256
LABEL_SIZE_DEFAULT = 48
LABEL_COLOR_DEFAULT = "#222222"
LABEL_FONT_DEFAULT = "Segoe UI"

_FONT_CANDIDATES = (
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
)

_FALLBACK_FAMILIES = (
    "Arial",
    "Calibri",
    "Cambria",
    "Comic Sans MS",
    "Courier New",
    "Georgia",
    "Impact",
    "Segoe UI",
    "Tahoma",
    "Times New Roman",
    "Trebuchet MS",
    "Verdana",
)

_FAMILY_FILES = {
    "arial": "arial.ttf",
    "arial black": "ariblk.ttf",
    "calibri": "calibri.ttf",
    "cambria": "cambria.ttc",
    "comic sans ms": "comic.ttf",
    "courier new": "cour.ttf",
    "georgia": "georgia.ttf",
    "impact": "impact.ttf",
    "segoe ui": "segoeui.ttf",
    "tahoma": "tahoma.ttf",
    "times new roman": "times.ttf",
    "trebuchet ms": "trebuc.ttf",
    "verdana": "verdana.ttf",
    "consolas": "consola.ttf",
    "palatino linotype": "pala.ttf",
    "microsoft sans serif": "micross.ttf",
    "garamond": "gara.ttf",
}


@dataclass(frozen=True)
class LabelSpec:
    """User label in source-pixel space (top-left of the text)."""

    text: str = ""
    size: int = LABEL_SIZE_DEFAULT
    color: tuple[int, int, int] = (34, 34, 34)
    x: int = 0
    y: int = 0
    font: str = LABEL_FONT_DEFAULT

    def is_set(self) -> bool:
        return bool((self.text or "").strip())


def clamp_label_size(value: object) -> int:
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        n = LABEL_SIZE_DEFAULT
    return max(LABEL_SIZE_MIN, min(LABEL_SIZE_MAX, n))


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{int(rgb[0]):02X}{int(rgb[1]):02X}{int(rgb[2]):02X}"


def hex_to_rgb(text: str) -> tuple[int, int, int] | None:
    raw = (text or "").strip().lstrip("#")
    if len(raw) == 3 and all(c in "0123456789abcdefABCDEF" for c in raw):
        return int(raw[0] * 2, 16), int(raw[1] * 2, 16), int(raw[2] * 2, 16)
    if len(raw) == 6 and all(c in "0123456789abcdefABCDEF" for c in raw):
        return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)
    return None


def parse_label_color(text: str | None) -> tuple[int, int, int]:
    rgb = hex_to_rgb(text or "")
    if rgb is None:
        rgb = hex_to_rgb(LABEL_COLOR_DEFAULT)
    return rgb if rgb is not None else (34, 34, 34)


def windows_fonts_dir() -> Path:
    windir = os.environ.get("WINDIR") or os.environ.get("SystemRoot") or "C:/Windows"
    return Path(windir) / "Fonts"


def list_font_families(root=None) -> list[str]:
    """Installed family names (Tk) plus a Windows fallback list."""
    names: list[str] = []
    try:
        import tkinter.font as tkfont

        host = root
        if host is None:
            import tkinter as tk

            host = tk._default_root
        if host is not None:
            names = [str(n) for n in tkfont.families(host) if str(n).strip()]
    except (ImportError, TypeError, RuntimeError, OSError):
        names = []
    if not names:
        names = list(_FALLBACK_FAMILIES)
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        key = name.casefold()
        if key in seen or name.startswith("@"):
            continue
        seen.add(key)
        out.append(name)
    out.sort(key=str.casefold)
    return out


def resolve_font_path(family: str) -> str | None:
    """Map a family name to a .ttf/.otf/.ttc file (Windows Fonts + registry)."""
    want = (family or "").strip()
    if not want:
        return None
    fonts_dir = windows_fonts_dir()
    key = want.casefold()
    mapped = _FAMILY_FILES.get(key)
    if mapped:
        path = fonts_dir / mapped
        if path.is_file():
            return str(path)
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts",
            ) as hive:
                i = 0
                while True:
                    try:
                        name, value, _typ = winreg.EnumValue(hive, i)
                    except OSError:
                        break
                    i += 1
                    stem = str(name).split("(")[0].strip().casefold()
                    if stem == key or stem.startswith(key + " "):
                        raw = str(value).strip()
                        path = Path(raw) if Path(raw).is_file() else fonts_dir / raw
                        if path.is_file():
                            return str(path)
        except (OSError, ImportError, ValueError):
            pass
    if fonts_dir.is_dir():
        slug = "".join(ch for ch in key if ch.isalnum())
        for path in fonts_dir.iterdir():
            if path.suffix.lower() not in {".ttf", ".otf", ".ttc"}:
                continue
            if "".join(ch for ch in path.stem.casefold() if ch.isalnum()) == slug:
                return str(path)
    return None


def load_label_font(size: int, family: str | None = None) -> ImageFont.ImageFont:
    """PIL truetype for preview/export — not the Tk display font."""
    px = clamp_label_size(size)
    want = (family or "").strip()
    if want:
        path = resolve_font_path(want)
        if path:
            try:
                return ImageFont.truetype(path, px)
            except (OSError, ValueError):
                pass
        try:
            return ImageFont.truetype(want, px)
        except (OSError, ValueError):
            pass
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, px)
        except (OSError, ValueError):
            continue
    return ImageFont.load_default()


def render_label_rgba(
    size: tuple[int, int],
    spec: LabelSpec,
    source_size: tuple[int, int],
    *,
    crop_x: float = 0.0,
    crop_y: float = 0.0,
    crop_zoom: float = 1.0,
) -> Image.Image:
    """Transparent plate the same size as the export / preview image."""
    w, h = max(1, int(size[0])), max(1, int(size[1]))
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if spec is None or not spec.is_set():
        return layer
    dx, dy, scale = source_xy_to_display(
        spec.x, spec.y, (w, h), source_size, crop_x, crop_y, crop_zoom
    )
    font_px = max(LABEL_SIZE_MIN, int(round(float(spec.size) * scale)))
    font = load_label_font(font_px, getattr(spec, "font", None))
    draw = ImageDraw.Draw(layer)
    fill = (*tuple(int(c) for c in spec.color), 255)
    try:
        draw.text((dx, dy), spec.text, font=font, fill=fill, anchor="lt")
    except TypeError:
        draw.text((dx, dy), spec.text, font=font, fill=fill)
    return layer


def composite_label(
    image: Image.Image,
    spec: LabelSpec,
    source_size: tuple[int, int],
    *,
    crop_x: float = 0.0,
    crop_y: float = 0.0,
    crop_zoom: float = 1.0,
) -> Image.Image:
    """Paste the label on top of ``image`` (RGB or RGBA)."""
    if spec is None or not spec.is_set():
        return image
    plate = render_label_rgba(
        image.size,
        spec,
        source_size,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_zoom=crop_zoom,
    )
    base = image.convert("RGBA")
    out = Image.alpha_composite(base, plate)
    return out.convert(image.mode) if image.mode != "RGBA" else out


def _svg_size_attrs(width_px: int, height_px: int, dpi: float | None) -> str:
    w, h = int(width_px), int(height_px)
    if dpi is not None and float(dpi) > 0.0:
        w_mm = w / float(dpi) * 25.4
        h_mm = h / float(dpi) * 25.4
        return (
            f'width="{w_mm:.4f}mm" height="{h_mm:.4f}mm" '
            f'viewBox="0 0 {w} {h}"'
        )
    return f'width="{w}px" height="{h}px" viewBox="0 0 {w} {h}"'


def label_svg_document(
    width_px: int,
    height_px: int,
    spec: LabelSpec,
    source_size: tuple[int, int],
    *,
    crop_x: float = 0.0,
    crop_y: float = 0.0,
    crop_zoom: float = 1.0,
    dpi: float | None = None,
) -> str:
    """SVG with real ``<text>`` so Illustrator can edit the letters."""
    w, h = int(width_px), int(height_px)
    size = _svg_size_attrs(w, h, dpi)
    dx, dy, scale = source_xy_to_display(
        spec.x, spec.y, (w, h), source_size, crop_x, crop_y, crop_zoom
    )
    font_px = max(LABEL_SIZE_MIN, float(spec.size) * scale)
    fill = rgb_to_hex(spec.color)
    text = escape(spec.text)
    family = escape(str(getattr(spec, "font", "") or LABEL_FONT_DEFAULT))
    # SVG y is baseline; approximate cap-height from font-size
    baseline = dy + font_px * 0.8
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" {size}>\n'
        f"  <title>Label</title>\n"
        f'  <text x="{dx:.2f}" y="{baseline:.2f}" '
        f'font-family="{family}, Arial, sans-serif" '
        f'font-size="{font_px:.2f}" fill="{fill}">{text}</text>\n'
        "</svg>\n"
    )


def write_label_files(
    dest_dir: Path,
    size: tuple[int, int],
    spec: LabelSpec,
    source_size: tuple[int, int],
    *,
    crop_x: float = 0.0,
    crop_y: float = 0.0,
    crop_zoom: float = 1.0,
    dpi: float | None = None,
    stem: str = "label",
) -> tuple[str, str, str]:
    """Write ``stem``.tif / .png / .svg. Returns the three filenames."""
    from wallpaper_recolor.io.image_io import save_image

    dest_dir = Path(dest_dir)
    plate = render_label_rgba(
        size,
        spec,
        source_size,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_zoom=crop_zoom,
    )
    save_image(plate, dest_dir / f"{stem}.tif", dpi=dpi)
    save_image(plate, dest_dir / f"{stem}.png", dpi=dpi)
    (dest_dir / f"{stem}.svg").write_text(
        label_svg_document(
            size[0],
            size[1],
            spec,
            source_size,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_zoom=crop_zoom,
            dpi=dpi,
        ),
        encoding="utf-8",
    )
    return f"{stem}.tif", f"{stem}.png", f"{stem}.svg"
