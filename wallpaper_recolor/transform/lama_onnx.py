# -*- coding: utf-8 -*-
"""
wallpaper_recolor.transform.lama_onnx
-------------------------------------
Optional LaMa inpaint via onnxruntime.

Weights: Carve/LaMa-ONNX ``lama_fp32.onnx`` (Apache-2.0), documented at
https://huggingface.co/Carve/LaMa-ONNX

Cache path: ``wallpaper_recolor/models/lama.onnx`` (never download on Remove).
Call ``download_lama_onnx()`` once to fetch. Tests must not call it.

Crops are **not** downscaled. Height/width are padded to a multiple of 8
(and, if the ONNX is a fixed 512 canvas, 1:1-padded onto that canvas when
the crop fits). A crop larger than a fixed canvas returns None so OpenCV
Navier–Stokes can run at source resolution.

Class references (code + name only):
- CAP3321C Data Wrangling
- CAP4631C Machine Learning
"""

from __future__ import annotations

from pathlib import Path
import urllib.request

import numpy as np

LAMA_ONNX_URL = (
    "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
)
LAMA_ONNX_PATH = Path(__file__).resolve().parents[1] / "models" / "lama.onnx"
_ALIGN = 8
_session = None
_session_path: str = ""


def lama_onnx_path() -> Path:
    return LAMA_ONNX_PATH


def lama_onnx_available() -> bool:
    path = LAMA_ONNX_PATH
    try:
        return path.is_file() and path.stat().st_size > 1_000_000
    except OSError:
        return False


def download_lama_onnx(*, force: bool = False, timeout: float = 120.0) -> Path:
    """Fetch ``lama_fp32.onnx`` into the app cache. Not used by Remove."""
    dest = LAMA_ONNX_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1_000_000 and not force:
        return dest
    tmp = dest.with_suffix(".onnx.part")
    req = urllib.request.Request(
        LAMA_ONNX_URL,
        headers={"User-Agent": "wallpaper-recolor/lama-onnx"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)
    return dest


def pad_to_multiple(value: int, multiple: int = _ALIGN) -> int:
    m = max(1, int(multiple))
    n = max(1, int(value))
    return n if n % m == 0 else n + (m - n % m)


def _ort_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError:
        return []
    available = list(ort.get_available_providers())
    out: list[str] = []
    if "CUDAExecutionProvider" in available:
        out.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        out.append("CPUExecutionProvider")
    return out or available


def _session_or_none():
    """Cached InferenceSession, or None if onnxruntime / weights are missing."""
    global _session, _session_path
    if not lama_onnx_available():
        return None
    path = str(LAMA_ONNX_PATH)
    if _session is not None and _session_path == path:
        return _session
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    providers = _ort_providers()
    if not providers:
        return None
    sess = ort.InferenceSession(path, providers=providers)
    _session = sess
    _session_path = path
    return sess


def _fixed_spatial(session) -> tuple[int, int] | None:
    """(H, W) when the first image input is a static spatial size."""
    try:
        dims = list(session.get_inputs()[0].shape)
    except (IndexError, AttributeError, TypeError):
        return None
    if len(dims) == 4:
        h, w = dims[2], dims[3]
    elif len(dims) == 3:
        h, w = dims[1], dims[2]
    else:
        return None
    try:
        hi, wi = int(h), int(w)
    except (TypeError, ValueError):
        return None
    if hi > 0 and wi > 0:
        return hi, wi
    return None


def prepare_lama_canvas(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    canvas_hw: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    """1:1 pad ``rgb`` / ``mask`` (no interpolation). None if the crop cannot fit."""
    src = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
    hole = np.asarray(mask, dtype=bool)
    h, w = hole.shape
    if src.shape[0] != h or src.shape[1] != w:
        return None
    ph, pw = pad_to_multiple(h), pad_to_multiple(w)
    if canvas_hw is not None:
        ch, cw = int(canvas_hw[0]), int(canvas_hw[1])
        if h > ch or w > cw:
            return None
        ph, pw = ch, cw
    canvas = np.zeros((ph, pw, 3), dtype=np.uint8)
    mcanvas = np.zeros((ph, pw), dtype=np.uint8)
    canvas[:h, :w] = src
    mcanvas[:h, :w] = np.where(hole, 255, 0).astype(np.uint8)
    return canvas, mcanvas, h, w


def lama_inpaint_crop(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Fill ``mask`` True pixels of a source-resolution crop. None if unusable."""
    session = _session_or_none()
    if session is None:
        return None
    prepared = prepare_lama_canvas(rgb, mask, canvas_hw=_fixed_spatial(session))
    if prepared is None:
        return None
    canvas, mcanvas, h, w = prepared
    image = canvas.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))[None]
    hole = (mcanvas.astype(np.float32) / 255.0)[None, None]
    names = [inp.name for inp in session.get_inputs()]
    feeds: dict[str, np.ndarray] = {}
    if len(names) >= 2:
        feeds[names[0]] = image
        feeds[names[1]] = hole
    elif len(names) == 1:
        feeds[names[0]] = np.concatenate([image, hole], axis=1)
    else:
        return None
    try:
        out = session.run(None, feeds)[0]
    except Exception:
        return None
    arr = np.asarray(out)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
    return arr[:h, :w]
