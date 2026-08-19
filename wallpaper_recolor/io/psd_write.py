# -*- coding: utf-8 -*-
"""
wallpaper_recolor.io.psd_write
---------------------------
Minimal 8-bit RGB PSD with raster layers (color + alpha as transparency).

Not a full Solid Color + layer-mask PSD: Photoshop can still edit pixels
and Layer > Layer Mask > From Transparency. Large print TIFFs skip PSD
(4-byte layer-section length); the PNG mask pack is the production master.

Class references (code + name only):
- CAP3321C Data Wrangling
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np

# ~4K; 207MP × several RGBA layers overflows the PSD uint32 section length
MAX_PSD_PIXELS = 16_000_000


@dataclass
class PsdLayer:
    name: str
    rgb: np.ndarray  # H×W×3 uint8
    alpha: np.ndarray  # H×W uint8
    opacity: int = 255  # 0–255
    blend: bytes = b"norm"  # 4-char Photoshop key (over = Overlay)
    visible: bool = True


def _u16(n: int) -> bytes:
    return struct.pack(">H", n & 0xFFFF)


def _u32(n: int) -> bytes:
    return struct.pack(">I", n & 0xFFFFFFFF)


def _i16(n: int) -> bytes:
    return struct.pack(">h", n)


def _i32(n: int) -> bytes:
    return struct.pack(">i", n)


def _pascal4(name: str) -> bytes:
    raw = name.encode("ascii", "replace")[:255]
    payload = bytes([len(raw)]) + raw
    pad = (4 - (len(payload) % 4)) % 4
    return payload + b"\x00" * pad


def _raw_channel(plane: np.ndarray) -> bytes:
    data = np.ascontiguousarray(plane, dtype=np.uint8).tobytes()
    return _u16(0) + data  # compression 0 = raw


def _layer_block(layer: PsdLayer, width: int, height: int) -> tuple[bytes, bytes]:
    """(layer record, concatenated channel payloads)."""
    r = layer.rgb[:, :, 0]
    g = layer.rgb[:, :, 1]
    b = layer.rgb[:, :, 2]
    a = layer.alpha
    payloads = [_raw_channel(r), _raw_channel(g), _raw_channel(b), _raw_channel(a)]
    ch_ids = (0, 1, 2, -1)
    rec = _i32(0) + _i32(0) + _i32(height) + _i32(width)
    rec += _u16(4)
    for cid, blob in zip(ch_ids, payloads):
        rec += _i16(cid) + _u32(len(blob))
    rec += b"8BIM" + layer.blend[:4].ljust(4, b" ")
    rec += bytes([int(layer.opacity) & 0xFF, 0])
    flags = 0 if layer.visible else 2  # bit 1 = hidden
    rec += bytes([flags, 0])
    extra = _u32(0) + _u32(0) + _pascal4(layer.name)  # no vector mask, no ranges
    rec += _u32(len(extra)) + extra
    return rec, b"".join(payloads)


def write_psd(
    path: str | Path,
    layers: list[PsdLayer],
    composite_rgb: np.ndarray,
) -> None:
    """Write a version-1 RGB PSD. ``composite_rgb`` is the flattened preview."""
    height, width = composite_rgb.shape[:2]
    if width * height > MAX_PSD_PIXELS:
        raise ValueError("Image too large for a simple PSD layer section")
    if not layers:
        raise ValueError("PSD needs at least one layer")

    records: list[bytes] = []
    channels: list[bytes] = []
    for layer in layers:
        rec, blob = _layer_block(layer, width, height)
        records.append(rec)
        channels.append(blob)

    layer_body = _i16(len(layers)) + b"".join(records) + b"".join(channels)
    if len(layer_body) % 2:
        layer_body += b"\x00"
    layer_info = _u32(len(layer_body)) + layer_body
    section = layer_info + _u32(0)  # no global layer mask
    layer_and_mask = _u32(len(section)) + section

    header = (
        b"8BPS"
        + _u16(1)
        + b"\x00" * 6
        + _u16(3)
        + _u32(height)
        + _u32(width)
        + _u16(8)
        + _u16(3)
    )
    merged = b"".join(_raw_channel(composite_rgb[:, :, c]) for c in range(3))
    Path(path).write_bytes(header + _u32(0) + _u32(0) + layer_and_mask + merged)
