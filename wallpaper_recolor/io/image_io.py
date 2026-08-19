# -*- coding: utf-8 -*-
"""
wallpaper_recolor.io.image_io
--------------------------
Load and save wallpaper files as RGB(A) so range mapping can run in numpy.

Class references (code + name only):
- CAP3321C Data Wrangling
"""

from __future__ import annotations  # Path hints without runtime cost

from pathlib import Path
import warnings

from PIL import Image

# Print wallpaper files (200MP+) are normal, not a DOS vector — raise Pillow's
# default ~179MP cap but keep a bound so a truly insane file still fails.
Image.MAX_IMAGE_PIXELS = 1_000_000_000

# Suffixes the Open / Save dialogs accept (Pillow TIFF covers .tif and .tiff)
SUPPORTED_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# File-dialog filter string used by tkinter.filedialog
OPEN_FILETYPES = [
    ("Images", "*.tif *.tiff *.png *.jpg *.jpeg"),
    ("TIFF", "*.tif *.tiff"),
    ("PNG", "*.png"),
    ("JPEG", "*.jpg *.jpeg"),
    ("All files", "*.*"),
]

SAVE_FILETYPES = [
    ("PNG", "*.png"),
    ("TIFF", "*.tif *.tiff"),
    ("JPEG", "*.jpg *.jpeg"),
]


def _header_size(path: Path) -> tuple[int, int]:
    """Read width/height from the file header without decoding pixels."""
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None  # header only; cap restored in finally
    try:
        with Image.open(path) as probe:
            return probe.size
    except OSError:
        return (0, 0)
    finally:
        Image.MAX_IMAGE_PIXELS = previous


def load_image(path: str | Path) -> Image.Image:
    """Open a TIF/PNG/JPEG and return an 8-bit RGB or RGBA Pillow image.

    CMYK TIFF (print files) is converted to RGB. Palette and 16-bit modes
    are flattened to 8-bit so the range mapper can use a single uint8 array.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix or '(none)'}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(path)
            image.load()  # decode now so the file handle can close
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        width, height = _header_size(path)
        raise ValueError(
            f"Image is too large to open ({width}x{height} pixels)."
        ) from None

    # Keep alpha when the source already has it (PNG / some TIFF)
    wants_alpha = "A" in image.getbands()

    if image.mode == "CMYK":
        # Print TIFF → sRGB working copy; remap is RGB, not ink channels
        image = image.convert("RGB")
    elif image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA" if wants_alpha else "RGB")

    return image


# LZW on ~207MP print files is often the encode bottleneck (minutes, and if
# it ran on the Tk thread Windows showed "Not Responding"). Default is Adobe
# Deflate (ZIP) at level 1 — lossless, much faster. Uncompressed ("raw"/None)
# is fastest to write if disk budget allows. LZW remains available if a RIP
# requires it.
TIFF_COMPRESSION_FAST = "tiff_adobe_deflate"
TIFF_COMPRESS_LEVEL = 1  # 1 = fast / larger; 9 = slow / smaller
TIFF_COMPRESSION_RAW = "raw"
TIFF_COMPRESSION_LZW = "tiff_lzw"
# PNG optimize + zlib-6 is a multi-pass stall (even at 1200² when the layers
# zip writes many plates). Same trade as TIFF: fast / larger, lossless.
PNG_COMPRESS_LEVEL_FAST = 1


def save_image(
    image: Image.Image,
    path: str | Path,
    *,
    tiff_compression: str | None = TIFF_COMPRESSION_FAST,
    dpi: float | None = None,
) -> None:
    """Write RGB(A) to TIF, PNG, or JPEG. JPEG drops alpha (no alpha in JPEG).

    ``dpi`` tags print size on TIFF/JPEG/PNG (Pillow ``dpi=`` / ``info``).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported save type: {suffix or '(none)'}")

    to_save = image
    save_kwargs: dict = {}

    if suffix in {".jpg", ".jpeg"}:
        # JPEG has no alpha — composite on white so pale inks stay visible
        if to_save.mode == "RGBA":
            background = Image.new("RGB", to_save.size, (255, 255, 255))
            background.paste(to_save, mask=to_save.split()[-1])
            to_save = background
        elif to_save.mode != "RGB":
            to_save = to_save.convert("RGB")
        save_kwargs["quality"] = 95  # wallpaper-quality stills, not web 75
        save_kwargs["subsampling"] = 0  # 4:4:4 — keep edges on pattern work
    elif suffix in {".tif", ".tiff"}:
        # LZW was too slow on ~207MP print files; ZIP level 1 (or raw) is the fast path
        comp = TIFF_COMPRESSION_FAST if tiff_compression is None else tiff_compression
        if comp in {None, "raw", "none"}:
            # Uncompressed — fastest encode; ~3 bytes/px RGB
            save_kwargs["compression"] = "raw"
        else:
            save_kwargs["compression"] = comp
            if comp in {"tiff_adobe_deflate", "tiff_deflate"}:
                save_kwargs["compress_level"] = TIFF_COMPRESS_LEVEL
    elif suffix == ".png":
        # Not optimize=True — that re-filters every scanline. Fast zlib.
        save_kwargs["compress_level"] = PNG_COMPRESS_LEVEL_FAST

    # Print size: pixels / dpi. Used in pixel mode too (no resample, still tagged).
    if dpi is not None and float(dpi) > 0.0:
        dpi_pair = (float(dpi), float(dpi))
        save_kwargs["dpi"] = dpi_pair
        to_save.info["dpi"] = dpi_pair

    try:
        to_save.save(path, **save_kwargs)
    except (TypeError, ValueError, OSError):
        # Older Pillow may reject compress_level or dpi — retry without those
        if "compress_level" in save_kwargs:
            save_kwargs.pop("compress_level", None)
            try:
                to_save.save(path, **save_kwargs)
                return
            except (TypeError, ValueError, OSError):
                pass
        if "dpi" in save_kwargs:
            save_kwargs.pop("dpi", None)
            to_save.save(path, **save_kwargs)
        else:
            raise
