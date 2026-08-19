# -*- coding: utf-8 -*-
"""
wallpaper_recolor.labels.detect
-------------------------------
Find text / label regions **painted into the wallpaper raster**.

Prefers optional EasyOCR (``pip install -r requirements-ocr.txt``) and
returns quadrilaterals for fillPoly masks. Falls back to Tesseract, then
a numpy connected-component pass so the base Pillow+numpy install still
Detects. Never runs on a 12k canvas here — callers pass a work preview
or Select-area ROI; boxes map back to source pixels in the UI.

Empty results are fine — the UI falls back to drag-rect.

Class references (code + name only):
- CAP3321C Data Wrangling
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

from wallpaper_recolor.color.color_ranges import luma_channel
from wallpaper_recolor.labels.boxes import Box, aabb_quad, normalize_box

Quad = tuple[tuple[int, int], ...]
_EASYOCR_CONF = 0.15
_easyocr_reader = None


@dataclass(frozen=True)
class TextRegion:
    """Axis-aligned box plus the OCR quadrilateral (4 points) in image pixels."""

    box: Box
    quad: Quad


def _quad_to_box(quad: Sequence[tuple[float, float]], width: int, height: int) -> Box | None:
    xs = [float(p[0]) for p in quad]
    ys = [float(p[1]) for p in quad]
    if not xs or not ys:
        return None
    return normalize_box(min(xs), min(ys), max(xs), max(ys), width, height)

_TESS_CONF_MIN = 20.0
_TESS_CONF_MIN_ROI = 5.0
_CC_MIN_AREA = 24
_CC_MAX_FRAC = 0.18
_CC_MIN_H = 4
_CC_MIN_W = 4
_DETECT_MAX_EDGE = 900
_UPSCALE = 3
_TESS_PSMS = (7, 8, 6, 11)


def _raise_if_cancelled(cancel) -> None:
    """Abort Detect when the footer Cancel event is set (cooperative)."""
    if cancel is not None and cancel.is_set():
        raise InterruptedError("cancelled")


def tesseract_available() -> bool:
    """True when pytesseract imports and the Tesseract binary runs."""
    try:
        import pytesseract
    except ImportError:
        return False
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return False
    return True


def easyocr_available() -> bool:
    try:
        import easyocr  # noqa: F401
    except ImportError:
        return False
    return True


def tesseract_status_text() -> str:
    """Panel status: hint only when OCR extras are missing."""
    if easyocr_available():
        return ""
    extra = "pip install -r requirements-ocr.txt for EasyOCR + LaMa"
    if tesseract_available():
        return extra
    return f"Install Tesseract, or {extra}"


def _cuda_for_easyocr() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is not None:
        return _easyocr_reader
    import easyocr

    _easyocr_reader = easyocr.Reader(["en"], gpu=_cuda_for_easyocr())
    return _easyocr_reader


def _parse_tesseract_data(data: dict, width: int, height: int, conf_min: float) -> list[Box]:
    texts = data.get("text") or []
    n = len(texts)
    w = max(1, int(width))
    h = max(1, int(height))
    boxes: list[Box] = []
    for i in range(n):
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError, KeyError):
            conf = -1.0
        if conf < conf_min:
            continue
        try:
            bw = int(data["width"][i])
            bh = int(data["height"][i])
            x = int(data["left"][i])
            y = int(data["top"][i])
        except (TypeError, ValueError, KeyError):
            continue
        if bw < 2 or bh < 3:
            continue
        box = normalize_box(x - 1, y - 1, x + bw + 1, y + bh + 1, w, h)
        if box is not None:
            boxes.append(box)
    return boxes


def _ocr_passes(image: Image.Image, *, extra_psm: bool) -> list[Image.Image]:
    rgb = image.convert("RGB")
    frames = [rgb]
    inverted = Image.eval(rgb, lambda p: 255 - p)
    frames.append(inverted)
    if extra_psm:
        frames.append(rgb.convert("L").point(lambda p: 255 if p > 140 else 0).convert("RGB"))
        frames.append(inverted.convert("L").point(lambda p: 255 if p > 140 else 0).convert("RGB"))
    return frames


def _try_tesseract_boxes(
    image: Image.Image,
    *,
    extra_psm: bool = False,
    conf_min: float | None = None,
    cancel=None,
) -> list[Box] | None:
    """Boxes from OCR, or None if Tesseract is not usable."""
    _raise_if_cancelled(cancel)
    try:
        import pytesseract
    except ImportError:
        return None
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return None
    floor = float(_TESS_CONF_MIN_ROI if extra_psm else _TESS_CONF_MIN)
    if conf_min is not None:
        floor = float(conf_min)
    psms = _TESS_PSMS if extra_psm else (11,)
    w, h = image.size
    found: list[Box] = []
    seen: set[Box] = set()
    try:
        for frame in _ocr_passes(image, extra_psm=extra_psm):
            _raise_if_cancelled(cancel)
            for psm in psms:
                _raise_if_cancelled(cancel)
                data = pytesseract.image_to_data(
                    frame,
                    output_type=pytesseract.Output.DICT,
                    config=f"--psm {int(psm)}",
                )
                for box in _parse_tesseract_data(data, w, h, floor):
                    if box not in seen:
                        seen.add(box)
                        found.append(box)
    except InterruptedError:
        raise
    except Exception:
        return None if not found else found
    return found


def _box_mean(gray: np.ndarray, radius: int) -> np.ndarray:
    """Local mean via integral image (odd window 2r+1)."""
    r = max(1, int(radius))
    s = 2 * r + 1
    g = gray.astype(np.float32)
    padded = np.pad(g, r, mode="edge")
    integ = np.pad(padded.cumsum(0).cumsum(1), ((1, 0), (1, 0)), mode="constant")
    h, w = gray.shape
    aa = integ[0:h, 0:w]
    ab = integ[0:h, s : s + w]
    ba = integ[s : s + h, 0:w]
    bb = integ[s : s + h, s : s + w]
    return (bb - ab - ba + aa) / float(s * s)


def _connected_boxes(
    mask: np.ndarray,
    min_area: int,
    max_area: int,
    *,
    min_h: int = _CC_MIN_H,
    min_w: int = _CC_MIN_W,
    cancel=None,
) -> list[Box]:
    """4-connected components on True pixels → bounding boxes (union-find)."""
    _raise_if_cancelled(cancel)
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return []
    n = int(ys.size)
    idx = np.full((h, w), -1, dtype=np.int32)
    idx[ys, xs] = np.arange(n, dtype=np.int32)
    parent = np.arange(n, dtype=np.int32)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        if i & 8191 == 0:
            _raise_if_cancelled(cancel)
        y = int(ys[i])
        x = int(xs[i])
        if x > 0:
            left = int(idx[y, x - 1])
            if left >= 0:
                union(i, left)
        if y > 0:
            up = int(idx[y - 1, x])
            if up >= 0:
                union(i, up)

    roots = np.empty(n, dtype=np.int32)
    for i in range(n):
        roots[i] = find(i)
    boxes: list[Box] = []
    for root in np.unique(roots):
        sel = roots == root
        area = int(sel.sum())
        if area < min_area or area > max_area:
            continue
        xs_c = xs[sel]
        ys_c = ys[sel]
        x0 = int(xs_c.min())
        y0 = int(ys_c.min())
        x1 = int(xs_c.max()) + 1
        y1 = int(ys_c.max()) + 1
        bw = x1 - x0
        bh = y1 - y0
        if bh < min_h or bw < min_w:
            continue
        aspect = bw / float(max(bh, 1))
        if aspect < 0.12 or aspect > 28.0:
            continue
        if bh > int(0.45 * h) and bw > int(0.45 * w):
            continue
        box = normalize_box(x0, y0, x1, y1, w, h)
        if box is not None:
            boxes.append(box)
    return boxes


def _merge_line_boxes(boxes: Sequence[Box], *, gap: int = 10) -> list[Box]:
    """Join horizontally adjacent boxes on the same text line."""
    if not boxes:
        return []
    items = sorted(boxes, key=lambda b: (b[1], b[0]))
    used = [False] * len(items)
    merged: list[Box] = []
    for i, box in enumerate(items):
        if used[i]:
            continue
        x0, y0, x1, y1 = box
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j, other in enumerate(items):
                if used[j]:
                    continue
                ox0, oy0, ox1, oy1 = other
                h = max(1, min(y1, oy1) - max(y0, oy0))
                if h < 0.45 * min(y1 - y0, oy1 - oy0):
                    continue
                if ox0 > x1 + gap or x0 > ox1 + gap:
                    continue
                x0, y0, x1, y1 = min(x0, ox0), min(y0, oy0), max(x1, ox1), max(y1, oy1)
                used[j] = True
                changed = True
        merged.append((x0, y0, x1, y1))
    return merged


def _contrast_boxes(image: Image.Image, *, small: bool = False, cancel=None) -> list[Box]:
    """MSER-lite: high-contrast blobs, plus bright-on-dark for tiny white labels."""
    _raise_if_cancelled(cancel)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = luma_channel(rgb)
    h, w = gray.shape
    area = h * w
    min_h = 3 if small else _CC_MIN_H
    min_w = 3 if small else _CC_MIN_W
    min_area = 12 if small else max(_CC_MIN_AREA, area // 8000)
    max_area = max(min_area + 1, int(area * _CC_MAX_FRAC))
    radius = max(4, min(h, w) // 50)
    blur = _box_mean(gray, radius)
    boxes: list[Box] = []
    for delta in (12.0, 16.0, 28.0):
        _raise_if_cancelled(cancel)
        dark = (gray + delta < blur) & (gray < 210)
        light = (gray > blur + delta) & (gray > 40)
        boxes.extend(
            _connected_boxes(
                dark, min_area, max_area, min_h=min_h, min_w=min_w, cancel=cancel
            )
        )
        boxes.extend(
            _connected_boxes(
                light, min_area, max_area, min_h=min_h, min_w=min_w, cancel=cancel
            )
        )
    med = float(np.median(gray))
    thr = max(150.0, med + 36.0)
    bright = gray >= thr
    boxes.extend(
        _connected_boxes(
            bright, min_area, max_area, min_h=min_h, min_w=min_w, cancel=cancel
        )
    )
    return _merge_line_boxes(boxes, gap=8 if small else 10)


def _upscale(image: Image.Image, factor: int) -> tuple[Image.Image, float]:
    f = max(1, int(factor))
    if f <= 1:
        return image, 1.0
    w, h = image.size
    return (
        image.resize((max(1, w * f), max(1, h * f)), Image.Resampling.NEAREST),
        float(f),
    )


def _map_boxes(boxes: Sequence[Box], scale: float, dest_w: int, dest_h: int) -> list[Box]:
    if not boxes:
        return []
    mapped_boxes: list[Box] = []
    inv = 1.0 if abs(scale - 1.0) < 1e-6 else 1.0 / scale
    for x0, y0, x1, y1 in boxes:
        box = normalize_box(
            float(x0) * inv,
            float(y0) * inv,
            max(float(x0) * inv + 2.0, float(x1) * inv),
            max(float(y0) * inv + 2.0, float(y1) * inv),
            dest_w,
            dest_h,
        )
        if box is not None:
            mapped_boxes.append(box)
    return mapped_boxes


def _offset_boxes(boxes: Sequence[Box], dx: int, dy: int, dest_w: int, dest_h: int) -> list[Box]:
    out: list[Box] = []
    for x0, y0, x1, y1 in boxes:
        box = normalize_box(x0 + dx, y0 + dy, x1 + dx, y1 + dy, dest_w, dest_h)
        if box is not None:
            out.append(box)
    return out


def _detect_in_frame(
    image: Image.Image, *, extra_psm: bool, small: bool, cancel=None
) -> list[Box]:
    """OCR (optional) plus connected components on this frame."""
    _raise_if_cancelled(cancel)
    iw, ih = image.size
    work, scale = _upscale(image, _UPSCALE if small or extra_psm else 1)
    boxes: list[Box] = []
    tess = _try_tesseract_boxes(work, extra_psm=extra_psm, cancel=cancel)
    if tess:
        boxes.extend(tess)
    boxes.extend(
        _contrast_boxes(
            work, small=True if extra_psm or small else small, cancel=cancel
        )
    )
    return _map_boxes(_merge_line_boxes(boxes), scale, iw, ih)


def _top_left_crop(image: Image.Image) -> tuple[Image.Image, int, int]:
    w, h = image.size
    pw = max(48, min(w, max(w // 5, 96)))
    ph = max(24, min(h, max(h // 8, 48)))
    return image.crop((0, 0, pw, ph)), 0, 0


def _map_quads(
    quads: Sequence[Quad], scale: float, dest_w: int, dest_h: int
) -> list[Quad]:
    if not quads:
        return []
    inv = 1.0 if abs(scale - 1.0) < 1e-6 else 1.0 / scale
    w = max(1, int(dest_w))
    h = max(1, int(dest_h))
    out: list[Quad] = []
    for quad in quads:
        pts = tuple(
            (
                int(max(0, min(w, round(float(x) * inv)))),
                int(max(0, min(h, round(float(y) * inv)))),
            )
            for x, y in quad
        )
        if len(pts) >= 3:
            out.append(pts)
    return out


def _offset_quads(quads: Sequence[Quad], dx: int, dy: int, dest_w: int, dest_h: int) -> list[Quad]:
    w = max(1, int(dest_w))
    h = max(1, int(dest_h))
    out: list[Quad] = []
    for quad in quads:
        pts = tuple(
            (
                int(max(0, min(w, round(float(x) + dx)))),
                int(max(0, min(h, round(float(y) + dy)))),
            )
            for x, y in quad
        )
        if len(pts) >= 3:
            out.append(pts)
    return out


def _regions_from_boxes(boxes: Sequence[Box]) -> list[TextRegion]:
    return [TextRegion(box=box, quad=aabb_quad(box)) for box in boxes]


def _regions_from_quads(quads: Sequence[Quad], width: int, height: int) -> list[TextRegion]:
    out: list[TextRegion] = []
    for quad in quads:
        box = _quad_to_box(quad, width, height)
        if box is None:
            continue
        out.append(TextRegion(box=box, quad=tuple((int(x), int(y)) for x, y in quad)))
    return out


def _try_easyocr_regions(image: Image.Image, *, cancel=None) -> list[TextRegion] | None:
    """EasyOCR quads in this frame, or None when the extra is not installed."""
    if not easyocr_available():
        return None
    _raise_if_cancelled(cancel)
    try:
        reader = _get_easyocr_reader()
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        results = reader.readtext(rgb, detail=1, paragraph=False)
    except Exception:
        return None
    _raise_if_cancelled(cancel)
    iw, ih = image.size
    regions: list[TextRegion] = []
    for item in results or ():
        _raise_if_cancelled(cancel)
        if not item:
            continue
        bbox = item[0]
        conf = 1.0
        if len(item) >= 3:
            try:
                conf = float(item[2])
            except (TypeError, ValueError):
                conf = 1.0
        if conf < _EASYOCR_CONF:
            continue
        try:
            quad = tuple((int(round(float(p[0]))), int(round(float(p[1])))) for p in bbox)
        except (TypeError, ValueError, IndexError):
            continue
        if len(quad) < 3:
            continue
        box = _quad_to_box(quad, iw, ih)
        if box is None:
            continue
        regions.append(TextRegion(box=box, quad=quad))
    return regions


def detect_text_regions(
    image: Image.Image,
    roi: Box | None = None,
    cancel=None,
) -> list[TextRegion]:
    """OCR regions (box + quad) in this raster. EasyOCR first; Tesseract/CC fallback."""
    _raise_if_cancelled(cancel)
    if image is None:
        return []
    iw, ih = image.size
    if roi is not None:
        rx0, ry0, rx1, ry1 = roi
        clipped = normalize_box(rx0, ry0, rx1, ry1, iw, ih)
        if clipped is None:
            return []
        x0, y0, x1, y1 = clipped
        crop = image.crop((x0, y0, x1, y1))
        easy = _try_easyocr_regions(crop, cancel=cancel)
        if easy:
            quads = _offset_quads([r.quad for r in easy], x0, y0, iw, ih)
            return _regions_from_quads(quads, iw, ih)
        found = _detect_in_frame(crop, extra_psm=True, small=True, cancel=cancel)
        return _regions_from_boxes(_offset_boxes(found, x0, y0, iw, ih))

    long_edge = max(iw, ih)
    work = image
    scale = 1.0
    if long_edge > _DETECT_MAX_EDGE:
        _raise_if_cancelled(cancel)
        scale = _DETECT_MAX_EDGE / float(long_edge)
        size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
        work = image.resize(size, Image.Resampling.BILINEAR)
    easy = _try_easyocr_regions(work, cancel=cancel)
    if easy:
        quads = _map_quads([r.quad for r in easy], scale, iw, ih)
        return _regions_from_quads(quads, iw, ih)
    boxes = _detect_in_frame(work, extra_psm=False, small=False, cancel=cancel)
    boxes = _map_boxes(boxes, scale, iw, ih)
    _raise_if_cancelled(cancel)
    probe, dx, dy = _top_left_crop(image)
    corner = _detect_in_frame(probe, extra_psm=True, small=True, cancel=cancel)
    boxes.extend(_offset_boxes(corner, dx, dy, iw, ih))
    return _regions_from_boxes(_merge_line_boxes(boxes))


def detect_text_boxes(
    image: Image.Image,
    roi: Box | None = None,
    cancel=None,
) -> list[Box]:
    """Text-like boxes in this **image raster's** pixel space (work preview).

    Searches baked-in glyphs (EasyOCR, Tesseract, or contrast blobs). It does
    not read Label stack overlays. ``roi`` is an inclusive-exclusive box in
    the same space. Without a ROI, a downscaled work image is used so 12k
    wallpapers are not OCR'd at full resolution. ``cancel`` is a
    threading.Event checked between OCR passes so footer Cancel can abort.
    """
    return [region.box for region in detect_text_regions(image, roi=roi, cancel=cancel)]


def refine_text_boxes(
    image: Image.Image,
    boxes: Sequence[Box],
    *,
    cancel=None,
) -> list[Box]:
    """Prefer Tesseract word/char boxes inside each region; else keep the region."""
    _raise_if_cancelled(cancel)
    if image is None or not boxes:
        return list(boxes or [])
    iw, ih = image.size
    out: list[Box] = []
    for box in boxes:
        _raise_if_cancelled(cancel)
        clipped = normalize_box(box[0], box[1], box[2], box[3], iw, ih)
        if clipped is None:
            continue
        x0, y0, x1, y1 = clipped
        crop = image.crop((x0, y0, x1, y1))
        found = _try_tesseract_boxes(
            crop, extra_psm=True, conf_min=_TESS_CONF_MIN_ROI, cancel=cancel
        )
        if found:
            out.extend(_offset_boxes(found, x0, y0, iw, ih))
        else:
            out.append(clipped)
    return out or list(boxes)
