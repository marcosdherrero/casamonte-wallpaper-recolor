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


def _taper_1d(length: int, left: int, right: int) -> np.ndarray:
    """Linear fade at the ends so overlapping LaMa windows blend."""
    n = max(1, int(length))
    w = np.ones(n, dtype=np.float64)
    left = max(0, min(int(left), n // 2))
    right = max(0, min(int(right), n - left))
    if left:
        w[:left] = np.linspace(0.05, 1.0, left, endpoint=False)
    if right:
        w[n - right :] = np.linspace(1.0, 0.05, right, endpoint=False)
    return w


def lama_inpaint_windows(
    rgb: np.ndarray,
    mask: np.ndarray,
    windows: list[tuple[int, int, int, int]],
    *,
    overlap: int = 64,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Fill ``mask`` by running LaMa on overlapping ``(y0, x0, y1, x1)`` crops.

    Returns ``(rgb, filled_mask)`` or ``(None, None)`` when no window ran.
    Only mask pixels with weight are written; seam context stays from ``rgb``.
    """
    hole = np.asarray(mask, dtype=bool)
    if not np.any(hole) or not windows:
        return None, None
    src = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
    h, w = hole.shape
    if src.shape[0] != h or src.shape[1] != w:
        return None, None
    acc = np.zeros((h, w, 3), dtype=np.float64)
    wgt = np.zeros((h, w), dtype=np.float64)
    ran = False
    ov = max(0, int(overlap))
    for y0, x0, y1, x1 in windows:
        y0, x0, y1, x1 = int(y0), int(x0), int(y1), int(x1)
        if y1 <= y0 or x1 <= x0:
            continue
        y0 = max(0, min(y0, h - 1))
        x0 = max(0, min(x0, w - 1))
        y1 = max(y0 + 1, min(y1, h))
        x1 = max(x0 + 1, min(x1, w))
        sub = hole[y0:y1, x0:x1]
        if not np.any(sub):
            continue
        filled = lama_inpaint_crop(src[y0:y1, x0:x1], sub)
        if filled is None:
            continue
        ran = True
        wy = _taper_1d(y1 - y0, ov if y0 > 0 else 0, ov if y1 < h else 0)
        wx = _taper_1d(x1 - x0, ov if x0 > 0 else 0, ov if x1 < w else 0)
        ww = wy[:, None] * wx[None, :]
        acc[y0:y1, x0:x1] += filled.astype(np.float64) * ww[..., None]
        wgt[y0:y1, x0:x1] += ww
    if not ran:
        return None, None
    out = np.array(src, copy=True)
    ok = hole & (wgt > 1e-6)
    blended = acc / np.maximum(wgt[..., None], 1e-6)
    out[ok] = np.clip(np.rint(blended[ok]), 0, 255).astype(np.uint8)
    return out, ok
