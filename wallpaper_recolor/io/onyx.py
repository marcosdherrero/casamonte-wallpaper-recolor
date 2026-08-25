# -*- coding: utf-8 -*-
"""
wallpaper_recolor.io.onyx
-------------------------
Canon Colorado M-series ONYX media libraries (``.oml`` / zip packs).

Each pack embeds CMYK printer ICC profiles (one per print mode). We extract
those to ``_extracted_icc/`` beside the pack so ImageCms can apply them.

Class references (code + name only):
- CAP3321C Data Wrangling
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
import struct
import zipfile

EXTRACTED_DIR_NAME = "_extracted_icc"
CANON_ONYX_DOWNLOADS = (
    "https://downloads.canon.com/csaassets/LFSColorProfiles/"
    "Colorado M-series/Onyx/"
)
CANON_PROFILES_PAGE = (
    "https://www.usa.canon.com/business/products/toner-supplies/"
    "large-format-color-profiles"
)

_PRINT_MODE = re.compile(rb'<node name="PrintMode" value="([^"]+)"')
_RES_ALIAS = re.compile(rb'<node name="ResAlias" value="([^"]+)"')
_MEDIA_TYPE = re.compile(rb'MediaTypeName \{ "([^"]+)" \}')
_MEDIA_NAME = re.compile(rb'MediaName \{ "([^"]+)" \}')
_DEVICE = re.compile(rb'<node name="DllDevice" value="([^"]+)"')
_ICC_MAGIC = b"acsp"
_MAX_ICC = 20_000_000


@dataclass(frozen=True)
class OnyxIcc:
    """One print-mode ICC pulled from an ONYX media library."""

    media: str
    mode: str
    device: str
    blob: bytes

    @property
    def label(self) -> str:
        media = self.media or "ONYX media"
        if self.mode:
            return f"{media} — {self.mode}"
        return media

    @property
    def tooltip(self) -> str:
        device = self.device or "Canon Colorado M-series"
        media = self.media or "this ONYX media"
        mode = f", print mode {self.mode}" if self.mode else ""
        return (
            f"{device} + {media}{mode}. "
            "Converts the sRGB preview into that printer/media output space."
        )


def _unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", "replace").strip()


def _safe_name(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", str(name or "")).strip(" .")
    return cleaned or "profile"


def iter_icc_blobs(raw: bytes) -> list[bytes]:
    """ICC payloads whose size header matches the ``acsp`` tag."""
    blobs: list[bytes] = []
    idx = 0
    while True:
        magic = raw.find(_ICC_MAGIC, idx)
        if magic < 0:
            break
        start = magic - 36
        idx = magic + 4
        if start < 0 or start + 4 > len(raw):
            continue
        size = struct.unpack(">I", raw[start : start + 4])[0]
        if size < 128 or size > _MAX_ICC or start + size > len(raw):
            continue
        blobs.append(raw[start : start + size])
    return blobs


def parse_oml(raw: bytes, *, fallback_media: str = "") -> list[OnyxIcc]:
    """Map unique ONYX print modes onto embedded ICC blobs (index order)."""
    device = ""
    match = _DEVICE.search(raw)
    if match:
        device = _decode(match.group(1))
    media = fallback_media
    match = _MEDIA_TYPE.search(raw)
    if match:
        media = _decode(match.group(1)) or media
    if not media:
        match = _MEDIA_NAME.search(raw)
        if match:
            media = re.sub(r"\s*\[[^\]]*\]\s*$", "", _decode(match.group(1))).strip()
    modes = _unique_in_order(_decode(m) for m in _PRINT_MODE.findall(raw))
    if not modes:
        modes = _unique_in_order(_decode(m) for m in _RES_ALIAS.findall(raw))
    blobs = iter_icc_blobs(raw)
    count = min(len(modes) or len(blobs), len(blobs))
    records: list[OnyxIcc] = []
    for i in range(count):
        mode = modes[i] if i < len(modes) else f"Profile {i + 1}"
        records.append(
            OnyxIcc(media=media or fallback_media, mode=mode, device=device, blob=blobs[i])
        )
    return records


def _oml_bytes_from_path(path: Path) -> bytes | None:
    suffix = path.suffix.lower()
    try:
        if suffix == ".oml":
            return path.read_bytes()
        if suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                names = [
                    name
                    for name in archive.namelist()
                    if name.lower().endswith(".oml") and not name.endswith("/")
                ]
                if not names:
                    return None
                return archive.read(names[0])
    except (OSError, zipfile.BadZipFile, KeyError):
        return None
    return None


def _stamp_path(dest_dir: Path) -> Path:
    return dest_dir / ".source_stamp"


def _source_stamp(path: Path) -> str:
    stat = path.stat()
    return f"{path.name}|{stat.st_mtime_ns}|{stat.st_size}"


def _write_sidecar(icc_path: Path, record: OnyxIcc, source_name: str) -> None:
    payload = {
        "label": record.label,
        "media": record.media,
        "mode": record.mode,
        "device": record.device,
        "source": source_name,
        "tooltip": record.tooltip,
    }
    icc_path.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_onyx_sidecar(path: Path) -> dict | None:
    """Metadata written next to an extracted ONYX ICC, or None."""
    meta = Path(path).with_suffix(".json")
    if not meta.is_file():
        return None
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def extract_oml_pack(source: Path, cache_root: Path) -> list[Path]:
    """Write ICC files for one ``.oml`` / zip pack into ``cache_root``."""
    source = Path(source)
    raw = _oml_bytes_from_path(source)
    if not raw:
        return []
    fallback = source.stem
    records = parse_oml(raw, fallback_media=fallback)
    if not records:
        return []
    media = records[0].media or fallback
    dest_dir = Path(cache_root) / _safe_name(media)
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = _source_stamp(source)
    stamp_file = _stamp_path(dest_dir)
    paths: list[Path] = []
    if stamp_file.is_file() and stamp_file.read_text(encoding="utf-8") == stamp:
        for record in records:
            icc_path = dest_dir / f"{_safe_name(record.mode)}.icc"
            if icc_path.is_file():
                paths.append(icc_path)
        if len(paths) == len(records):
            return paths
    for record in records:
        icc_path = dest_dir / f"{_safe_name(record.mode)}.icc"
        icc_path.write_bytes(record.blob)
        _write_sidecar(icc_path, record, source.name)
        paths.append(icc_path)
    stamp_file.write_text(stamp, encoding="utf-8")
    return paths


def collect_onyx_icc_paths(folder: str | Path) -> list[Path]:
    """Extract ICC from ONYX packs under ``folder`` and return those files."""
    root = Path(folder)
    if not root.is_dir():
        return []
    cache_root = root / EXTRACTED_DIR_NAME
    found: list[Path] = []
    try:
        children = list(root.iterdir())
    except OSError:
        return []
    for path in children:
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".oml", ".zip"}:
            continue
        try:
            found.extend(extract_oml_pack(path, cache_root))
        except OSError:
            continue
    found.sort(key=lambda item: item.as_posix().lower())
    return found
