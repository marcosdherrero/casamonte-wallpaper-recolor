# -*- coding: utf-8 -*-
"""
wallpaper_recolor.io.proof
-----------------------
Soft-proof an RGB master through a printer ICC (Pillow ImageCms).

Hex / on-screen RGB is not a print guarantee. The RGB master stays
untouched; this file is a separate proof image. No ICC → skip.

Class references (code + name only):
- CAP3321C Data Wrangling
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

try:
    from PIL import ImageCms

    _HAS_CMS = True
except ImportError:  # wheels without LCMS
    ImageCms = None  # type: ignore[misc, assignment]
    _HAS_CMS = False


def cms_available() -> bool:
    return _HAS_CMS


def soft_proof(image: Image.Image, icc_path: str | Path) -> Image.Image:
    """Simulate ``icc_path`` on an sRGB display. Raises ValueError on failure."""
    if not _HAS_CMS:
        raise ValueError("Pillow was built without ImageCms (LittleCMS).")
    path = Path(icc_path)
    if not path.is_file():
        raise ValueError(f"ICC profile not found: {path}")

    rgb = image.convert("RGB")
    try:
        srgb = ImageCms.createProfile("sRGB")
        printer = ImageCms.getOpenProfile(str(path))
        flags = getattr(ImageCms.Flags, "SOFTPROOFING", 16384)
        transform = ImageCms.buildProofTransformFromOpenProfiles(
            srgb,
            srgb,
            printer,
            "RGB",
            "RGB",
            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            proofRenderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            flags=flags,
        )
        return ImageCms.applyTransform(rgb, transform)
    except Exception as exc:  # ImageCms raises PyCMSError / OSError
        raise ValueError(f"Could not soft-proof with {path.name}: {exc}") from exc
