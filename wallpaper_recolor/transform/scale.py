# -*- coding: utf-8 -*-
"""
wallpaper_recolor.transform.scale
-----------------------
Output size for Save as… / Export job pack: pixels, inches, or centimetres,
plus a DPI tag and a Pillow resample filter.

Applied **after** remap + tone on the full-res composite (background save
thread), not on the live preview. Empty / 0 width and height means keep the
source pixel size (no resample). DPI still writes into TIFF/JPEG/PNG so the
print size is defined even when the pixel count does not change.

Inches / cm → pixels: ``px = inches * dpi``, ``px = cm / 2.54 * dpi``.
Default DPI is 300 (print wallpaper). Dropdown presets: 72, 96, 150, 300, 600,
plus Custom….

No tkinter — same rule as tone / color_math / layers.

Class references (code + name only):
- CAP3321C Data Wrangling
"""

from __future__ import annotations

from PIL import Image

# Combobox labels (UI + tests). Physical units need DPI to become pixels.
UNIT_PIXELS = "Pixels"
UNIT_INCHES = "Inches"
UNIT_CM = "Centimetres"
UNITS = (UNIT_PIXELS, UNIT_INCHES, UNIT_CM)

CM_PER_INCH = 2.54
# Print wallpaper default; 150 is the other common press value
DPI_DEFAULT = 300
DPI_PRESETS = (72, 96, 150, 300, 600)
DPI_CUSTOM_LABEL = "Custom…"
DPI_CHOICES = tuple(str(v) for v in DPI_PRESETS) + (DPI_CUSTOM_LABEL,)

# Pillow filters — nearest keeps a pixel look; Lanczos is the photo default
RESAMPLE_NEAREST = "Nearest neighbor (hard edges / keep pixel look)"
RESAMPLE_BILINEAR = "Bilinear"
RESAMPLE_BICUBIC = "Bicubic"
RESAMPLE_LANCZOS = "Lanczos (high quality downsample)"
RESAMPLE_LABELS = (
    RESAMPLE_NEAREST,
    RESAMPLE_BILINEAR,
    RESAMPLE_BICUBIC,
    RESAMPLE_LANCZOS,
)
RESAMPLE_FILTERS = {
    RESAMPLE_NEAREST: Image.Resampling.NEAREST,
    RESAMPLE_BILINEAR: Image.Resampling.BILINEAR,
    RESAMPLE_BICUBIC: Image.Resampling.BICUBIC,
    RESAMPLE_LANCZOS: Image.Resampling.LANCZOS,
}
DEFAULT_RESAMPLE = RESAMPLE_LANCZOS


def is_physical_unit(unit: str) -> bool:
    """True when width/height are inches or centimetres (DPI changes pixel count)."""
    return unit in (UNIT_INCHES, UNIT_CM)


def parse_dim(text: str | float | None) -> float | None:
    """Positive finite number, or None for empty / 0 / junk (treat as unset)."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        value = float(text)
        return value if value > 0.0 else None
    raw = str(text).strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if value != value or value <= 0.0:  # NaN or non-positive
        return None
    return value


def parse_dpi_choice(choice: str, custom_text: str = "", default: float = DPI_DEFAULT) -> float:
    """Dropdown label → DPI. Custom… uses the extra entry; invalid falls back."""
    label = (choice or "").strip()
    if label == DPI_CUSTOM_LABEL:
        parsed = parse_dim(custom_text)
        return float(parsed) if parsed is not None else float(default)
    parsed = parse_dim(label)
    return float(parsed) if parsed is not None else float(default)


def dpi_choice_for(dpi: float) -> str:
    """Preset label if ``dpi`` matches 72/96/150/300/600, else Custom…."""
    rounded = int(round(float(dpi)))
    if abs(float(dpi) - rounded) < 1e-6 and rounded in DPI_PRESETS:
        return str(rounded)
    return DPI_CUSTOM_LABEL


def to_pixels(value: float, unit: str, dpi: float) -> int:
    """One dimension in ``unit`` → integer pixels (minimum 1)."""
    amount = float(value)
    res = max(1.0, float(dpi))
    if unit == UNIT_INCHES:
        px = amount * res
    elif unit == UNIT_CM:
        px = amount / CM_PER_INCH * res
    else:
        px = amount
    return max(1, int(round(px)))


def from_pixels(px: int, unit: str, dpi: float) -> float:
    """Integer pixels → ``unit`` (inverse of to_pixels)."""
    res = max(1.0, float(dpi))
    count = float(max(1, int(px)))
    if unit == UNIT_INCHES:
        return count / res
    if unit == UNIT_CM:
        return count / res * CM_PER_INCH
    return count


def format_dim(value: float, unit: str) -> str:
    """Compact field text: whole pixels, trimmed physical decimals."""
    if unit == UNIT_PIXELS:
        return str(max(1, int(round(float(value)))))
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text or "0"


def resampling_filter(label: str) -> Image.Resampling:
    """Map a dropdown label onto Image.Resampling.*; unknown → Lanczos."""
    return RESAMPLE_FILTERS.get(label, Image.Resampling.LANCZOS)


def resolve_output_size(
    src_w: int,
    src_h: int,
    width: float | None,
    height: float | None,
    unit: str,
    dpi: float,
    aspect_lock: bool,
) -> tuple[int, int] | None:
    """Pixel (w, h) for save, or None when both dimensions are empty/0.

    Aspect lock fills the missing side from the source ratio. Both unset
    means “original size” (no resample). DPI only affects inches/cm.
    """
    src_w = max(1, int(src_w))
    src_h = max(1, int(src_h))
    w_set = width is not None and float(width) > 0.0
    h_set = height is not None and float(height) > 0.0
    if not w_set and not h_set:
        return None

    aspect = src_w / float(src_h)
    if w_set and h_set:
        return to_pixels(float(width), unit, dpi), to_pixels(float(height), unit, dpi)
    if w_set:
        pw = to_pixels(float(width), unit, dpi)
        if aspect_lock:
            return pw, max(1, int(round(pw / aspect)))
        return pw, src_h
    ph = to_pixels(float(height), unit, dpi)
    if aspect_lock:
        return max(1, int(round(ph * aspect))), ph
    return src_w, ph


def scale_image(
    image: Image.Image,
    size: tuple[int, int] | None,
    resample_label: str = DEFAULT_RESAMPLE,
) -> Image.Image:
    """Resize after remap/tone. None or the current size is a no-op."""
    if size is None:
        return image
    width, height = max(1, int(size[0])), max(1, int(size[1]))
    if (width, height) == image.size:
        return image
    return image.resize((width, height), resampling_filter(resample_label))
