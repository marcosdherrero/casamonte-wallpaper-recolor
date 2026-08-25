# -*- coding: utf-8 -*-
"""
wallpaper_recolor.io.proof
-----------------------
Soft-proof an RGB master through a printer ICC (Pillow ImageCms),
list ICC files, and convert a preview into a selected output space.

Hex / on-screen RGB is not a print guarantee. The RGB master stays
untouched; job-pack ``icc_proof.png`` is a separate proof image.

Class references (code + name only):
- CAP3321C Data Wrangling
"""

from __future__ import annotations

from pathlib import Path
from io import BytesIO
import os
import re

from PIL import Image

from wallpaper_recolor.io.onyx import (
    EXTRACTED_DIR_NAME,
    collect_onyx_icc_paths,
    read_onyx_sidecar,
)

try:
    from PIL import ImageCms

    _HAS_CMS = True
except ImportError:  # wheels without LCMS
    ImageCms = None  # type: ignore[misc, assignment]
    _HAS_CMS = False

# Default printer/working-space folder (overridable via env or ``icc_profiles_dir``).
DEFAULT_ICC_PROFILES_DIR = Path(r"C:\Users\marco\OneDrive\Documents\Work\Color Profiles")
ICC_PROFILE_SUFFIXES = {".icc", ".icm"}
_ICC_MAGIC = b"acsp"  # ICC header bytes 36–39
_SKIP_SUFFIXES = {
    ".pdf",
    ".zip",
    ".url",
    ".msi",
    ".tif",
    ".tiff",
    ".jpg",
    ".jpeg",
    ".png",
    ".ai",
    ".tmp",
    ".json",
    ".txt",
    ".oml",
}
_MAX_SNIFF_BYTES = 20_000_000
SRGB_PROFILE_KEY = "sRGB"

# Shop-specific / common stems. Keys are ``_norm_profile_key`` results.
KNOWN_ICC_TOOLTIPS: dict[str, str] = {
    "srgb": (
        "Keep sRGB as the working space (IEC 61966-2-1). "
        "Screen RGB; no printer conversion."
    ),
    "srgb iec61966 2.1": (
        "Keep sRGB as the working space (IEC 61966-2-1). "
        "Screen RGB; no printer conversion."
    ),
    "adobe rgb (1998)": (
        "Convert from sRGB into Adobe RGB (1998). Wider-gamut RGB so saturated "
        "print colors stay in gamut that sRGB would clip."
    ),
    "adobergb1998": (
        "Convert from sRGB into Adobe RGB (1998). Wider-gamut RGB so saturated "
        "print colors stay in gamut that sRGB would clip."
    ),
    "adobe rgb": (
        "Convert from sRGB into Adobe RGB (1998). Wider-gamut RGB so saturated "
        "print colors stay in gamut that sRGB would clip."
    ),
    "generic canvas 460 mus2506": (
        "Canon Colorado Generic Canvas 460 media. Converts the preview from sRGB "
        "into that printer/media output space."
    ),
    "generic canvas 460": (
        "Canon Colorado Generic Canvas 460 media. Converts the preview from sRGB "
        "into that printer/media output space."
    ),
    "display p3": (
        "Convert from sRGB into Display P3 (DCI-P3 primaries, sRGB transfer). "
        "Wider than sRGB; typical of wide-gamut monitors."
    ),
    "prophoto rgb": (
        "Convert from sRGB into ProPhoto RGB, a very wide photographic working "
        "space. Out-of-sRGB colors stay representable."
    ),
}

_DISPLAY_XFORM_CACHE: dict[str, tuple] = {}


def cms_available() -> bool:
    return _HAS_CMS


def icc_profiles_dir(override: str | Path | None = None) -> Path:
    """Folder of ``.icc`` / ``.icm`` files. ``override``, then env, then default."""
    if override is not None:
        return Path(override)
    env = str(os.environ.get("WALLPAPER_ICC_PROFILES_DIR") or "").strip()
    if env:
        return Path(env)
    return DEFAULT_ICC_PROFILES_DIR


def _norm_profile_key(name: str) -> str:
    return re.sub(r"[\s_\-]+", " ", str(name or "")).strip().lower()


def humanize_profile_stem(stem: str) -> str:
    """Filename stem → menu label (underscores to spaces)."""
    return str(stem or "").replace("_", " ").strip() or "ICC profile"


def looks_like_icc(path: Path) -> bool:
    """True for ``.icc`` / ``.icm`` or a file whose header is an ICC profile."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in ICC_PROFILE_SUFFIXES:
        return True
    if suffix in _SKIP_SUFFIXES or not suffix:
        if suffix in _SKIP_SUFFIXES:
            return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size < 128 or size > _MAX_SNIFF_BYTES:
        return False
    try:
        with path.open("rb") as handle:
            header = handle.read(40)
    except OSError:
        return False
    return header[36:40] == _ICC_MAGIC


def list_icc_profiles(folder: str | Path | None = None) -> list[Path]:
    """Sorted ICC/ICM (and ONYX-extracted) profiles under ``folder``. Missing → []."""
    root = icc_profiles_dir(folder)
    if not root.is_dir():
        return []
    found: list[Path] = []
    try:
        for path in root.rglob("*"):
            if EXTRACTED_DIR_NAME in path.parts:
                continue
            if not path.is_file():
                continue
            if looks_like_icc(path):
                found.append(path)
    except OSError:
        return []
    try:
        found.extend(collect_onyx_icc_paths(root))
    except OSError:
        pass
    uniq: list[Path] = []
    seen: set[str] = set()
    for path in found:
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(path)
    uniq.sort(key=lambda item: profile_menu_label(item).lower())
    return uniq


def _open_profile(path: Path):
    """Load an ICC from bytes so Windows does not keep the file locked."""
    return ImageCms.getOpenProfile(BytesIO(path.read_bytes()))


def read_profile_description(path: str | Path) -> str:
    """ICC ``desc`` tag via ImageCms, or empty."""
    if not _HAS_CMS:
        return ""
    try:
        profile = _open_profile(Path(path))
        text = ImageCms.getProfileDescription(profile)
        return str(text or "").strip()
    except Exception:
        try:
            profile = _open_profile(Path(path))
            inner = getattr(profile, "profile", None)
            return str(getattr(inner, "profile_description", "") or "").strip()
        except Exception:
            return ""


def read_profile_copyright(path: str | Path) -> str:
    """ICC copyright tag via ImageCms, or empty."""
    if not _HAS_CMS:
        return ""
    try:
        profile = _open_profile(Path(path))
        text = ImageCms.getProfileCopyright(profile)
        return str(text or "").strip()
    except Exception:
        try:
            profile = _open_profile(Path(path))
            inner = getattr(profile, "profile", None)
            return str(getattr(inner, "copyright", "") or "").strip()
        except Exception:
            return ""


def srgb_profile_tooltip() -> str:
    return KNOWN_ICC_TOOLTIPS["srgb"]


def profile_menu_label(path: Path) -> str:
    file_path = Path(path)
    sidecar = read_onyx_sidecar(file_path)
    if sidecar:
        label = str(sidecar.get("label") or "").strip()
        if label:
            return label
    return humanize_profile_stem(file_path.stem)


def profile_tooltip(path: str | Path, *, known: dict[str, str] | None = None) -> str:
    """Hover copy: ONYX sidecar, known-name map, ICC description, or filename."""
    file_path = Path(path)
    sidecar = read_onyx_sidecar(file_path)
    if sidecar:
        tip = str(sidecar.get("tooltip") or "").strip()
        if tip:
            return tip
        media = str(sidecar.get("media") or file_path.stem).strip()
        mode = str(sidecar.get("mode") or "").strip()
        device = str(sidecar.get("device") or "Canon Colorado M-series").strip()
        extra = f", print mode {mode}" if mode else ""
        return (
            f"{device} + {media}{extra}. "
            "Converts the sRGB preview into that printer/media output space."
        )
    table = KNOWN_ICC_TOOLTIPS if known is None else known
    human = humanize_profile_stem(file_path.stem)
    mapped = ""
    for key in (
        _norm_profile_key(file_path.stem),
        _norm_profile_key(human),
        _norm_profile_key(file_path.name),
    ):
        if key in table:
            mapped = table[key]
            break
    desc = read_profile_description(file_path)
    copy = read_profile_copyright(file_path)
    parts: list[str] = []
    if mapped:
        parts.append(mapped)
        if desc and desc.lower() not in mapped.lower() and "built-in" not in desc.lower():
            parts.append(desc)
    elif desc:
        parts.append(f"Converts the preview from sRGB into {desc}.")
    else:
        parts.append(f"Converts the preview from sRGB into {human}.")
    if copy and "no copyright" not in copy.lower():
        parts.append(copy)
    return " ".join(parts).strip()


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
        printer = _open_profile(path)
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


def apply_icc(image: Image.Image, icc_path: str | Path | None) -> Image.Image:
    """Convert ``image`` from sRGB into ``icc_path`` for on-screen preview.

    ``None`` is identity (working space stays sRGB). RGB output profiles are
    converted dest→sRGB for display; other spaces use ``soft_proof``.
    Raises ValueError if the profile cannot be applied.
    """
    if icc_path is None:
        return image
    if not _HAS_CMS:
        raise ValueError("Pillow was built without ImageCms (LittleCMS).")
    path = Path(icc_path)
    if not path.is_file():
        raise ValueError(f"ICC profile not found: {path}")

    alpha = image.split()[-1] if "A" in image.getbands() else None
    rgb = image.convert("RGB")
    try:
        cache_key = str(path.resolve())
    except OSError:
        cache_key = str(path)
    cached = _DISPLAY_XFORM_CACHE.get(cache_key)
    try:
        if cached is None:
            srgb = ImageCms.createProfile("sRGB")
            dest = _open_profile(path)
            space = ""
            try:
                inner = getattr(dest, "profile", None)
                space = str(getattr(inner, "xcolor_space", "") or "")
            except Exception:
                space = ""
            if space.strip().upper().startswith("RGB"):
                to_dest = ImageCms.buildTransformFromOpenProfiles(
                    srgb,
                    dest,
                    "RGB",
                    "RGB",
                    renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                )
                to_display = ImageCms.buildTransformFromOpenProfiles(
                    dest,
                    srgb,
                    "RGB",
                    "RGB",
                    renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                )
                cached = ("rgb", to_dest, to_display)
            else:
                cached = ("proof", None, None)
            _DISPLAY_XFORM_CACHE[cache_key] = cached
        kind, to_dest, to_display = cached
        if kind == "rgb" and to_dest is not None and to_display is not None:
            converted = ImageCms.applyTransform(rgb, to_dest)
            out = ImageCms.applyTransform(converted, to_display)
        else:
            out = soft_proof(rgb, path)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not apply {path.name}: {exc}") from exc

    if alpha is not None:
        if out.mode != "RGBA":
            out = out.convert("RGBA")
        out.putalpha(alpha)
    return out
