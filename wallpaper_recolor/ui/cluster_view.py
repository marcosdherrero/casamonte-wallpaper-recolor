# -*- coding: utf-8 -*-
"""
wallpaper_recolor.ui.cluster_view
---------------------------------
3D CIE Lab scatter of k-means ranges (notebook-style pixels + centers).

The MDC k-means color-compression notebook plots pixels in an RGB cube
(Plotly Scatter3d, colored by source RGB, centroids as larger markers).
This app clusters in Lab, so the same idea is drawn on L* / a* / b*.

matplotlib is optional (requirements-plot.txt). Without it the widget
falls back to a 2D a*–b* Tk canvas. Subsampling keeps 12k wallpapers
interactive — never scatter the full raster on the Tk thread.

Class references (code + name only):
- CAP4631C Machine Learning
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import ttk

from wallpaper_recolor.color.color_math import lab_tuple_to_rgb, rgb_to_lab_array, rgb_tuple_to_lab
from wallpaper_recolor.color.color_ranges import ColorRangeMap

CLUSTER_MAX_POINTS = 6000  # subsample; never scatter a 12k wallpaper on the Tk thread
CLUSTER_SEED = 7  # stable subsample so the cloud does not jump between refreshes
MODE_SOURCE = "source"
MODE_REPLACE = "replace"
MODE_LABELS = ("Source RGB", "Change-to")
MODE_KEYS = (MODE_SOURCE, MODE_REPLACE)
CLUSTER_HINT = (
    "Drag to orbit COM · Wheel zooms in on the cloud · Shift-drag / right-drag to pan · "
    "Double-click a point to sample its color · Middle-drag moves the selected color in Lab · "
    "Cube: face/edge/corner · Center: iso + fit on mass · "
    "XYZ cycles Front/Right/Back/Left/Top/Bottom"
)
CLUSTER_ZOOM_PCT_MIN = 100.0
CLUSTER_ZOOM_PCT_MAX = 4000.0
_PICK_DRAG_PX = 5
_PICK_MARKER_S = 92
_PICK_RING_S = 170
_ORBIT_GAIN = 0.4
_PAN_GAIN = 0.35
_MOVE_GAIN = 0.35
DEFAULT_LOOK = (0.0, 0.0, 50.0)
DEFAULT_ELEV = 20.0
DEFAULT_AZIM = -60.0
XYZ_NORMAL_ELEV = 0.0
XYZ_NORMAL_AZIM = 0.0
XYZ_CYCLE = ("front", "right", "back", "left", "top", "bottom")
ISO_ELEV = 35.26438968
_CUBE_PX = 86
_CUBE_HIT_CORNER = 9
_CUBE_HIT_EDGE = 6
CLOUD_ALPHA = 0.12
CLOUD_STD_K = 2.0
CLOUD_MIN_R = 2.5
CLOUD_WIRE_U = 14
CLOUD_WIRE_V = 8

# Face toward the camera → matplotlib elev / azim.
# Plot axes: X=a*, Y=b*, Z=L* (L* is Z-up).
FACE_VIEWS: dict[str, tuple[float, float]] = {
    "front": (0.0, 0.0),  # +a* face toward you (look along −a*)
    "back": (0.0, 180.0),  # −a*
    "right": (0.0, 90.0),  # +b*
    "left": (0.0, -90.0),  # −b*
    "top": (90.0, -90.0),  # +L*
    "bottom": (-90.0, -90.0),  # −L*
}
FACE_LABELS = {
    "front": "+a*",
    "back": "−a*",
    "right": "+b*",
    "left": "−b*",
    "top": "+L*",
    "bottom": "−L*",
}
# Outward normals in (a*, b*, L*) = (x, y, z)
FACE_NORMALS = {
    "front": (1.0, 0.0, 0.0),
    "back": (-1.0, 0.0, 0.0),
    "right": (0.0, 1.0, 0.0),
    "left": (0.0, -1.0, 0.0),
    "top": (0.0, 0.0, 1.0),
    "bottom": (0.0, 0.0, -1.0),
}
FACE_COLORS = {
    "front": "#c9d8ea",
    "back": "#9aa8b8",
    "right": "#d4e6c3",
    "left": "#a8b896",
    "top": "#ead7b8",
    "bottom": "#b8a888",
}


def clamp_cluster_zoom_pct(pct: float) -> float:
    """Clusters Lab camera: 100%–4000% (tighter box than Composite’s 800%)."""
    try:
        value = float(pct)
    except (TypeError, ValueError):
        return CLUSTER_ZOOM_PCT_MIN
    if value != value:
        return CLUSTER_ZOOM_PCT_MIN
    return min(CLUSTER_ZOOM_PCT_MAX, max(CLUSTER_ZOOM_PCT_MIN, value))


def matplotlib_available() -> bool:
    try:
        import matplotlib  # noqa: F401

        return True
    except ImportError:
        return False


def camera_xyz(elev: float, azim: float) -> np.ndarray:
    """Unit vector from origin toward the matplotlib camera (elev/azim degrees)."""
    er = np.radians(float(elev))
    ar = np.radians(float(azim))
    return np.array(
        (np.cos(er) * np.cos(ar), np.cos(er) * np.sin(ar), np.sin(er)),
        dtype=np.float64,
    )


def camera_basis(elev: float, azim: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Screen-right, screen-up, and camera-from-origin unit vectors in (a*, b*, L*)."""
    cam = camera_xyz(elev, azim)
    world_up = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    if abs(float(np.dot(cam, world_up))) > 0.98:
        world_up = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    right = np.cross(world_up, cam)
    rn = float(np.linalg.norm(right))
    right = right / rn if rn > 1e-9 else np.array((1.0, 0.0, 0.0), dtype=np.float64)
    up = np.cross(cam, right)
    return right, up, cam


def lab_delta_from_view_pixels(
    dx: float,
    dy: float,
    elev: float,
    azim: float,
    zoom_pct: float,
    *,
    gain: float = _MOVE_GAIN,
) -> tuple[float, float, float]:
    """Screen-pixel drag → (dL*, da*, db*) in the camera right/up plane."""
    z = max(1.0, float(zoom_pct) / 100.0)
    scale = float(gain) / z
    right, up, _cam = camera_basis(elev, azim)
    # Object-follow (not grab-the-world pan): pointer right → +screen-x;
    # Tk y grows downward so pointer down → −screen-up.
    delta_abl = (float(dx) * right - float(dy) * up) * scale
    return (float(delta_abl[2]), float(delta_abl[0]), float(delta_abl[1]))


def projected_lab_screen_xy(
    lab: np.ndarray | tuple[float, float, float],
    elev: float,
    azim: float,
) -> tuple[float, float]:
    """Orthographic screen-right / screen-up of Lab in the camera plane (a*, b*, L*)."""
    right, up, _cam = camera_basis(elev, azim)
    arr = np.asarray(lab, dtype=np.float64).reshape(3)
    abl = np.array((float(arr[1]), float(arr[2]), float(arr[0])), dtype=np.float64)
    return float(np.dot(abl, right)), float(np.dot(abl, up))


def clamp_lab_tuple(lab: np.ndarray | tuple[float, float, float]) -> tuple[float, float, float]:
    """Keep L* in 0–100 and a*/b* in the usual sRGB-safe span."""
    arr = np.asarray(lab, dtype=np.float64).reshape(3)
    return (
        float(min(100.0, max(0.0, arr[0]))),
        float(min(127.0, max(-128.0, arr[1]))),
        float(min(127.0, max(-128.0, arr[2]))),
    )


def view_for_face(name: str) -> tuple[float, float]:
    """elev, azim so that cube face ``name`` looks toward the camera."""
    key = str(name or "").strip().lower()
    if key not in FACE_VIEWS:
        raise KeyError(f"Unknown cube face: {name}")
    return FACE_VIEWS[key]


def view_for_corner(sign_a: int, sign_b: int, sign_l: int) -> tuple[float, float]:
    """Isometric elev/azim for the octant (sign of a*, b*, L*)."""
    sa = 1 if int(sign_a) >= 0 else -1
    sb = 1 if int(sign_b) >= 0 else -1
    sl = 1 if int(sign_l) >= 0 else -1
    elev = ISO_ELEV if sl > 0 else -ISO_ELEV
    azim = float(np.degrees(np.arctan2(sb, sa)))
    return elev, azim


def view_for_edge(axis_a: int, axis_b: int, axis_l: int) -> tuple[float, float]:
    """Two-axis view: exactly two of ±a*/±b*/±L* are nonzero (±1, 0)."""
    vec = np.array((float(axis_a), float(axis_b), float(axis_l)), dtype=np.float64)
    n = float(np.linalg.norm(vec))
    if n < 1e-9:
        return XYZ_NORMAL_ELEV, XYZ_NORMAL_AZIM
    vec = vec / n
    elev = float(np.degrees(np.arcsin(np.clip(vec[2], -1.0, 1.0))))
    azim = float(np.degrees(np.arctan2(vec[1], vec[0])))
    return elev, azim


def _norm_azim(azim: float) -> float:
    a = float(azim) % 360.0
    if a > 180.0:
        a -= 360.0
    return a


def _point_in_poly(px: float, py: float, pts: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > py) != (yj > py) and px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def _dist_point_seg(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    vx, vy = bx - ax, by - ay
    t = ((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy + 1e-12)
    t = max(0.0, min(1.0, t))
    dx, dy = ax + t * vx - px, ay + t * vy - py
    return float(np.hypot(dx, dy))


def _empty_sample() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    empty = np.zeros((0, 3), dtype=np.float32)
    return empty, np.zeros((0, 3), dtype=np.uint8), np.zeros((0,), dtype=np.int32), np.zeros(
        (0, 2), dtype=np.int32
    )


def subsample_labeled_pixels(
    rgb: np.ndarray,
    labels: np.ndarray,
    *,
    max_points: int = CLUSTER_MAX_POINTS,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pick up to ``max_points`` labeled pixels.

    Returns Lab, source RGB uint8, labels, and ``(y, x)`` work-image coords
    so a click maps to a real sampled pixel — not a cluster center.
    """
    arr = np.asarray(rgb)
    lbl = np.asarray(labels)
    if arr.ndim != 3 or arr.shape[-1] < 3 or lbl.ndim != 2:
        return _empty_sample()
    h, w = lbl.shape
    if arr.shape[0] != h or arr.shape[1] != w:
        return _empty_sample()
    flat_rgb = np.asarray(arr[..., :3], dtype=np.uint8).reshape(-1, 3)
    flat_lbl = lbl.reshape(-1)
    idx = np.flatnonzero(flat_lbl >= 0)
    if idx.size == 0:
        return _empty_sample()
    cap = max(1, int(max_points))
    if idx.size > cap:
        gen = rng if rng is not None else np.random.default_rng(CLUSTER_SEED)
        idx = np.sort(gen.choice(idx, size=cap, replace=False))
    sample_rgb = flat_rgb[idx]
    sample_lbl = flat_lbl[idx].astype(np.int32)
    sample_lab = rgb_to_lab_array(sample_rgb.reshape(-1, 1, 3)).reshape(-1, 3)
    ys = (idx // w).astype(np.int32)
    xs = (idx % w).astype(np.int32)
    coords = np.stack([ys, xs], axis=1)
    return sample_lab, sample_rgb, sample_lbl, coords


def source_pixel_at(
    data: dict, index: int
) -> tuple[tuple[int, int, int], int, int] | None:
    """Exact source RGB and ``(y, x)`` for a scatter index."""
    if data is None:
        return None
    src = np.asarray(data.get("source_rgb"))
    coords = np.asarray(data.get("coords"))
    i = int(index)
    if src.ndim != 2 or i < 0 or i >= src.shape[0]:
        return None
    rgb = (int(src[i, 0]), int(src[i, 1]), int(src[i, 2]))
    if coords.ndim == 2 and i < coords.shape[0]:
        return rgb, int(coords[i, 0]), int(coords[i, 1])
    return rgb, 0, 0


def cluster_scatter_data(
    range_map: ColorRangeMap,
    work: Image.Image | None,
    *,
    mode: str = MODE_SOURCE,
    max_points: int = CLUSTER_MAX_POINTS,
) -> dict | None:
    """Build the scatter payload from the current range map + work RGB."""
    if range_map is None or range_map.labels is None:
        return None
    if work is not None:
        rgb = np.asarray(work.convert("RGB"), dtype=np.uint8)
    elif range_map.rgb is not None:
        rgb = np.asarray(range_map.rgb, dtype=np.uint8)
    else:
        return None
    labels = np.asarray(range_map.labels)
    if rgb.shape[:2] != labels.shape[:2]:
        rgb = np.asarray(
            Image.fromarray(rgb, mode="RGB").resize(
                (labels.shape[1], labels.shape[0]), Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        )
    lab, src_rgb, lbl, coords = subsample_labeled_pixels(rgb, labels, max_points=max_points)
    n = len(range_map.ranges)
    matches = [tuple(int(c) for c in band.match_rgb) for band in range_map.ranges]
    replaces = [tuple(int(c) for c in band.replacement_rgb) for band in range_map.ranges]
    if range_map.centers is not None and int(range_map.centers.shape[0]) == n:
        centers_lab = np.asarray(range_map.centers, dtype=np.float32)
    else:
        centers_lab = np.stack([rgb_tuple_to_lab(rgb_t) for rgb_t in matches]).astype(
            np.float32
        ) if matches else np.zeros((0, 3), dtype=np.float32)
    use_replace = str(mode) == MODE_REPLACE
    if lab.shape[0] == 0:
        point_rgb = src_rgb
    elif use_replace:
        palette = np.array(replaces if replaces else [(128, 128, 128)], dtype=np.uint8)
        safe = np.clip(lbl, 0, len(palette) - 1)
        point_rgb = palette[safe]
    else:
        point_rgb = src_rgb
    return {
        "lab": lab,
        "point_rgb": point_rgb,
        "source_rgb": src_rgb,
        "coords": coords,
        "labels": lbl,
        "centers_lab": centers_lab,
        "match_rgb": matches,
        "replace_rgb": replaces,
        "mode": MODE_REPLACE if use_replace else MODE_SOURCE,
        "extents": cluster_range_extents(lab, lbl, matches, centers_lab),
    }


def _rgb_norm(colors: Sequence[tuple[int, int, int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(colors, dtype=np.float64)
    if arr.size == 0:
        return arr.reshape(0, 3)
    return np.clip(arr.reshape(-1, 3) / 255.0, 0.0, 1.0)


def cluster_look_target(data: dict | None) -> tuple[float, float, float]:
    """Look-at in (a*, b*, L*): mean of sampled Lab, else mean of k-means centers."""
    if not data:
        return DEFAULT_LOOK
    lab = np.asarray(data.get("lab", []))
    if lab.ndim == 2 and lab.shape[0] and lab.shape[1] >= 3:
        m = lab.mean(axis=0)
        return (float(m[1]), float(m[2]), float(m[0]))
    centers = np.asarray(data.get("centers_lab", []))
    if centers.ndim == 2 and centers.shape[0] and centers.shape[1] >= 3:
        m = centers.mean(axis=0)
        return (float(m[1]), float(m[2]), float(m[0]))
    return DEFAULT_LOOK


def cluster_range_extents(
    lab: np.ndarray,
    labels: np.ndarray,
    match_rgb: Sequence | None = None,
    centers_lab: np.ndarray | None = None,
) -> list[dict]:
    """Axis-aligned Lab ellipsoid per non-empty cluster (diameter of that range).

    Semi-axes are ``max(2σ, 95th |dev|, CLOUD_MIN_R)`` on L*, a*, b* so the
    shell covers ~95% of the subsample. Also stores RMS and max Lab radius.
    """
    pts = np.asarray(lab, dtype=np.float64)
    lbl = np.asarray(labels)
    if pts.ndim != 2 or pts.shape[0] == 0 or lbl.shape[0] != pts.shape[0]:
        return []
    n_lbl = int(lbl.max()) + 1 if lbl.size else 0
    n_rgb = len(match_rgb) if match_rgb is not None else 0
    n = max(n_lbl, n_rgb)
    centers = None if centers_lab is None else np.asarray(centers_lab, dtype=np.float64)
    out: list[dict] = []
    for i in range(n):
        members = pts[lbl == i]
        if members.shape[0] == 0:
            continue
        if centers is not None and i < centers.shape[0]:
            mean = centers[i]
        else:
            mean = members.mean(axis=0)
        devs = members - mean
        std = np.std(devs, axis=0)
        if members.shape[0] >= 4:
            p95 = np.percentile(np.abs(devs), 95.0, axis=0)
        else:
            p95 = np.max(np.abs(devs), axis=0)
        radii = np.maximum(np.maximum(CLOUD_STD_K * std, p95), CLOUD_MIN_R)
        d2 = np.sum(devs * devs, axis=1)
        rms = float(np.sqrt(np.mean(d2))) if d2.size else 0.0
        r_max = float(np.sqrt(np.max(d2))) if d2.size else 0.0
        rgb = (180, 180, 180)
        if match_rgb is not None and i < len(match_rgb):
            rgb = tuple(int(c) for c in match_rgb[i][:3])
        out.append(
            {
                "index": i,
                "center": (float(mean[0]), float(mean[1]), float(mean[2])),
                "radii": (float(radii[0]), float(radii[1]), float(radii[2])),
                "rms": rms,
                "r_max": r_max,
                "rgb": rgb,
                "n": int(members.shape[0]),
            }
        )
    return out


def _ellipsoid_mesh(
    center_lab: tuple[float, float, float],
    radii_lab: tuple[float, float, float],
    *,
    nu: int = CLOUD_WIRE_U,
    nv: int = CLOUD_WIRE_V,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wireframe mesh in plot axes: X=a*, Y=b*, Z=L*."""
    cl, ca, cb = (float(center_lab[0]), float(center_lab[1]), float(center_lab[2]))
    rl, ra, rb = (float(radii_lab[0]), float(radii_lab[1]), float(radii_lab[2]))
    u = np.linspace(0.0, 2.0 * np.pi, int(nu))
    v = np.linspace(0.0, np.pi, int(nv))
    xs = ca + ra * np.outer(np.cos(u), np.sin(v))
    ys = cb + rb * np.outer(np.sin(u), np.sin(v))
    zs = cl + rl * np.outer(np.ones_like(u), np.cos(v))
    return xs, ys, zs


def _extents_of(data: dict | None) -> list[dict]:
    if not data:
        return []
    cached = data.get("extents")
    if cached:
        return list(cached)
    return cluster_range_extents(
        data.get("lab", np.zeros((0, 3))),
        data.get("labels", np.zeros((0,), dtype=np.int32)),
        data.get("match_rgb"),
        data.get("centers_lab"),
    )


class LabViewCube(tk.Canvas):
    """Fusion-style view cube: click a face / edge / corner to set elev/azim."""

    def __init__(self, parent: tk.Misc, on_view, *, size: int = _CUBE_PX) -> None:
        super().__init__(
            parent,
            width=size,
            height=size,
            highlightthickness=1,
            highlightbackground="#888888",
            bg="#f4f4f4",
            cursor="hand2",
        )
        self._size = int(size)
        self.on_view = on_view
        self._elev = DEFAULT_ELEV
        self._azim = DEFAULT_AZIM
        self._hits: list[tuple[str, object]] = []
        self.bind("<Button-1>", self._on_click)
        self.bind("<Configure>", lambda _e: self.redraw(self._elev, self._azim))

    def redraw(self, elev: float, azim: float) -> None:
        self._elev = float(elev)
        self._azim = float(azim)
        self.delete("all")
        s = max(40, int(self.winfo_width() or self._size), int(self.winfo_height() or self._size))
        cx = cy = s * 0.5
        scale = s * 0.28
        cam = camera_xyz(self._elev, self._azim)
        world_up = np.array((0.0, 0.0, 1.0))
        if abs(float(np.dot(cam, world_up))) > 0.98:
            world_up = np.array((0.0, 1.0, 0.0))
        right = np.cross(world_up, cam)
        rn = float(np.linalg.norm(right))
        if rn < 1e-9:
            right = np.array((1.0, 0.0, 0.0))
        else:
            right = right / rn
        up = np.cross(cam, right)

        def proj(p: tuple[float, float, float]) -> tuple[float, float, float]:
            vec = np.asarray(p, dtype=np.float64)
            x = cx + float(np.dot(vec, right)) * scale
            y = cy - float(np.dot(vec, up)) * scale
            depth = float(np.dot(vec, cam))
            return x, y, depth

        verts = {
            (sa, sb, sl): proj((sa, sb, sl))
            for sa in (-1.0, 1.0)
            for sb in (-1.0, 1.0)
            for sl in (-1.0, 1.0)
        }
        faces = {
            "front": ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)),
            "back": ((-1, 1, -1), (-1, -1, -1), (-1, -1, 1), (-1, 1, 1)),
            "right": ((-1, 1, -1), (1, 1, -1), (1, 1, 1), (-1, 1, 1)),
            "left": ((1, -1, -1), (-1, -1, -1), (-1, -1, 1), (1, -1, 1)),
            "top": ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)),
            "bottom": ((-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)),
        }
        drawn: list[tuple[float, str, list[tuple[float, float]]]] = []
        for name, corners in faces.items():
            nrm = np.asarray(FACE_NORMALS[name], dtype=np.float64)
            if float(np.dot(nrm, cam)) <= 0.04:
                continue
            pts = [verts[c][:2] for c in corners]
            depth = float(np.mean([verts[c][2] for c in corners]))
            drawn.append((depth, name, pts))
        drawn.sort(key=lambda row: row[0])
        self._hits = []
        for _depth, name, pts in drawn:
            flat = [c for xy in pts for c in xy]
            self.create_polygon(
                *flat,
                fill=FACE_COLORS[name],
                outline="#333333",
                width=1,
                tags=f"face:{name}",
            )
            mx = sum(p[0] for p in pts) / 4.0
            my = sum(p[1] for p in pts) / 4.0
            self.create_text(mx, my, text=FACE_LABELS[name], fill="#222222", font=("Segoe UI", 7))
            self._hits.append(("face", name))
        vis_verts = [key for key, val in verts.items() if val[2] > -0.15]
        for key in vis_verts:
            x, y, _d = verts[key]
            self.create_oval(x - 2.5, y - 2.5, x + 2.5, y + 2.5, fill="#222222", outline="")
            self._hits.append(("corner", (int(key[0]), int(key[1]), int(key[2]))))
        edges = (
            ((1, -1, -1), (1, 1, -1)),
            ((1, 1, -1), (1, 1, 1)),
            ((1, 1, 1), (1, -1, 1)),
            ((1, -1, 1), (1, -1, -1)),
            ((-1, -1, -1), (-1, 1, -1)),
            ((-1, 1, -1), (-1, 1, 1)),
            ((-1, 1, 1), (-1, -1, 1)),
            ((-1, -1, 1), (-1, -1, -1)),
            ((1, -1, -1), (-1, -1, -1)),
            ((1, 1, -1), (-1, 1, -1)),
            ((1, 1, 1), (-1, 1, 1)),
            ((1, -1, 1), (-1, -1, 1)),
        )
        for a, b in edges:
            if verts[a][2] <= -0.2 and verts[b][2] <= -0.2:
                continue
            self._hits.append(("edge", (a, b)))

    def hit_at(self, x: float, y: float) -> tuple[str, object] | None:
        """Nearest corner, then edge, then face under ``(x, y)``."""
        s = max(40, int(self.winfo_width() or self._size))
        cx = cy = s * 0.5
        scale = s * 0.28
        cam = camera_xyz(self._elev, self._azim)
        world_up = np.array((0.0, 0.0, 1.0))
        if abs(float(np.dot(cam, world_up))) > 0.98:
            world_up = np.array((0.0, 1.0, 0.0))
        right = np.cross(world_up, cam)
        rn = float(np.linalg.norm(right))
        right = right / rn if rn > 1e-9 else np.array((1.0, 0.0, 0.0))
        up = np.cross(cam, right)

        def proj(p: tuple[float, float, float]) -> tuple[float, float, float]:
            vec = np.asarray(p, dtype=np.float64)
            return (
                cx + float(np.dot(vec, right)) * scale,
                cy - float(np.dot(vec, up)) * scale,
                float(np.dot(vec, cam)),
            )

        best_corner = None
        best_cd = _CUBE_HIT_CORNER
        for sa in (-1.0, 1.0):
            for sb in (-1.0, 1.0):
                for sl in (-1.0, 1.0):
                    px, py, depth = proj((sa, sb, sl))
                    if depth < -0.15:
                        continue
                    d = float(np.hypot(px - x, py - y))
                    if d < best_cd:
                        best_cd = d
                        best_corner = ("corner", (int(sa), int(sb), int(sl)))
        if best_corner is not None:
            return best_corner
        best_edge = None
        best_ed = _CUBE_HIT_EDGE
        corners = (
            ((1, -1, -1), (1, 1, -1)),
            ((1, 1, -1), (1, 1, 1)),
            ((1, 1, 1), (1, -1, 1)),
            ((1, -1, 1), (1, -1, -1)),
            ((-1, -1, -1), (-1, 1, -1)),
            ((-1, 1, -1), (-1, 1, 1)),
            ((-1, 1, 1), (-1, -1, 1)),
            ((-1, -1, 1), (-1, -1, -1)),
            ((1, -1, -1), (-1, -1, -1)),
            ((1, 1, -1), (-1, 1, -1)),
            ((1, 1, 1), (-1, 1, 1)),
            ((1, -1, 1), (-1, -1, 1)),
        )
        for a, b in corners:
            pa, pb = proj(a), proj(b)
            if pa[2] < -0.2 and pb[2] < -0.2:
                continue
            d = _dist_point_seg(x, y, pa[0], pa[1], pb[0], pb[1])
            if d < best_ed:
                best_ed = d
                mid = (
                    int(np.sign(a[0] + b[0]) or 0),
                    int(np.sign(a[1] + b[1]) or 0),
                    int(np.sign(a[2] + b[2]) or 0),
                )
                best_edge = ("edge", mid)
        if best_edge is not None:
            return best_edge
        faces = {
            "front": ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)),
            "back": ((-1, 1, -1), (-1, -1, -1), (-1, -1, 1), (-1, 1, 1)),
            "right": ((-1, 1, -1), (1, 1, -1), (1, 1, 1), (-1, 1, 1)),
            "left": ((1, -1, -1), (-1, -1, -1), (-1, -1, 1), (1, -1, 1)),
            "top": ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)),
            "bottom": ((-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)),
        }
        best_face = None
        best_fd = -1.0
        for name, corn in faces.items():
            nrm = np.asarray(FACE_NORMALS[name], dtype=np.float64)
            facing = float(np.dot(nrm, cam))
            if facing <= 0.04:
                continue
            pts = [proj(c)[:2] for c in corn]
            if _point_in_poly(x, y, pts) and facing > best_fd:
                best_fd = facing
                best_face = ("face", name)
        return best_face

    def _on_click(self, event) -> None:
        hit = self.hit_at(float(event.x), float(event.y))
        if hit is None or self.on_view is None:
            return
        kind, payload = hit
        if kind == "face":
            self.on_view("face", payload)
        elif kind == "corner":
            self.on_view("corner", payload)
        else:
            self.on_view("edge", payload)


class ClusterPlot(ttk.Frame):
    """Notebook tab body: matplotlib 3D Lab, or 2D a*–b* canvas fallback."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, padding=4)
        self._mode = tk.StringVar(value=MODE_LABELS[0])
        self._status = tk.StringVar(value="")
        self.on_pick = None  # (rgb, y, x) -> None
        self.on_zoom = None  # (pct: float) -> None
        self.on_selected_rgb = None  # () -> rgb | None
        self.on_move_start = None  # () -> None
        self.on_move = None  # (rgb) -> None
        self.on_move_end = None  # (rgb) -> None
        self._fig = None
        self._ax = None
        self._canvas = None
        self._mpl_widget = None
        self._ab_canvas: tk.Canvas | None = None
        self._last_key: tuple | None = None
        self._cached_data = None
        self._zoom_pct = 100.0
        self._elev = DEFAULT_ELEV
        self._azim = DEFAULT_AZIM
        self._look = DEFAULT_LOOK
        self.view_cube: LabViewCube | None = None
        self.center_btn: ttk.Button | None = None
        self.xyz_btn: ttk.Button | None = None
        self.home_btn = None
        self.view_overlay: ttk.Frame | None = None
        self._xyz_i = 0
        self._press: tuple[int, int] | None = None
        self._orbit_last: tuple[int, int] | None = None
        self._did_drag = False
        self._picked_index: int | None = None
        self._pick_artists: list = []
        self._move_lab: tuple[float, float, float] | None = None
        self._mmb_last: tuple[int, int] | None = None

        head = ttk.Frame(self)
        head.pack(fill="x")
        ttk.Label(head, text="Point color").pack(side="left")
        self.mode_combo = ttk.Combobox(
            head,
            textvariable=self._mode,
            values=list(MODE_LABELS),
            state="readonly",
            width=14,
        )
        self.mode_combo.pack(side="left", padx=(8, 0))
        ttk.Label(head, textvariable=self._status, foreground="#555555").pack(
            side="left", padx=(12, 0)
        )
        ttk.Label(self, text=CLUSTER_HINT, foreground="#555555").pack(anchor="w", pady=(2, 0))

        self.host = ttk.Frame(self)
        self.host.pack(fill="both", expand=True, pady=(4, 0))
        self.host.rowconfigure(0, weight=1)
        self.host.columnconfigure(0, weight=1)
        self._build_backend()
        self._build_view_overlay()
        self.bind("<Destroy>", self._on_destroy, add="+")

    def mode_key(self) -> str:
        label = str(self._mode.get() or MODE_LABELS[0])
        if label == MODE_LABELS[1]:
            return MODE_REPLACE
        return MODE_SOURCE

    def zoom_pct(self) -> float:
        return float(self._zoom_pct)

    def set_zoom_pct(self, pct: float, *, notify: bool = False) -> None:
        self._zoom_pct = clamp_cluster_zoom_pct(pct)
        self._apply_camera()
        if self._ab_canvas is not None:
            self._redraw_ab(None)
        if notify and self.on_zoom is not None:
            self.on_zoom(self._zoom_pct)

    def pick_index(self, index: int) -> tuple[tuple[int, int, int], int, int] | None:
        """Apply a known scatter index (tests / click). Returns source RGB, y, x."""
        hit = source_pixel_at(self._cached_data or {}, index)
        if hit is None:
            return None
        self._picked_index = int(index)
        self._draw_pick_highlight()
        if self.on_pick is not None:
            self.on_pick(*hit)
        return hit

    def _on_destroy(self, event=None) -> None:
        if event is not None and event.widget is not self:
            return
        fig = self._fig
        self._fig = None
        self._ax = None
        self._mpl_widget = None
        if fig is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(fig)
            except Exception:
                pass

    def bind_mode(self, command) -> None:
        self.mode_combo.bind("<<ComboboxSelected>>", lambda _e: command(), add="+")

    def contains_root(self, x_root: int, y_root: int) -> bool:
        try:
            x0 = int(self.winfo_rootx())
            y0 = int(self.winfo_rooty())
            w = max(1, int(self.winfo_width()))
            h = max(1, int(self.winfo_height()))
        except tk.TclError:
            return False
        return x0 <= int(x_root) < x0 + w and y0 <= int(y_root) < y0 + h

    def _clear_pick_highlight(self) -> None:
        self._picked_index = None
        if self._mmb_last is None:
            self._move_lab = None
        self._remove_pick_artists()
        if self._ab_canvas is not None:
            try:
                self._ab_canvas.delete("pick")
            except tk.TclError:
                pass
        if self._mpl_widget is not None:
            self._mpl_widget.draw_idle()

    def _remove_pick_artists(self) -> None:
        for art in self._pick_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._pick_artists = []

    def _draw_pick_highlight(self) -> None:
        """White-ringed marker on the picked scatter point or the MMB-dragged Lab."""
        self._remove_pick_artists()
        if self._ab_canvas is not None:
            try:
                self._ab_canvas.delete("pick")
            except tk.TclError:
                pass
        data = self._cached_data
        move = self._move_lab
        idx = self._picked_index
        l_s = a_s = b_s = None
        fill = (1.0, 1.0, 1.0)
        if move is not None:
            l_s, a_s, b_s = float(move[0]), float(move[1]), float(move[2])
            rgb = lab_tuple_to_rgb(move)
            fill = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
        elif data is not None and idx is not None:
            lab = np.asarray(data.get("lab"))
            if lab.ndim != 2 or idx < 0 or idx >= lab.shape[0]:
                self._picked_index = None
                return
            l_s, a_s, b_s = float(lab[idx, 0]), float(lab[idx, 1]), float(lab[idx, 2])
            src = np.asarray(data.get("source_rgb", data.get("point_rgb")))
            if src.ndim == 2 and idx < src.shape[0]:
                fill = (
                    float(src[idx, 0]) / 255.0,
                    float(src[idx, 1]) / 255.0,
                    float(src[idx, 2]) / 255.0,
                )
        if l_s is None or a_s is None or b_s is None:
            return
        if self._ax is not None:
            ring = self._ax.scatter(
                [a_s],
                [b_s],
                [l_s],
                s=_PICK_RING_S,
                marker="o",
                facecolors="none",
                edgecolors="#111111",
                linewidths=2.8,
                depthshade=False,
            )
            core = self._ax.scatter(
                [a_s],
                [b_s],
                [l_s],
                s=_PICK_MARKER_S,
                marker="o",
                c=[fill],
                edgecolors="#ffffff",
                linewidths=1.8,
                depthshade=False,
            )
            self._pick_artists = [ring, core]
            if self._mpl_widget is not None:
                self._mpl_widget.draw_idle()
        if self._ab_canvas is not None:
            try:
                w = max(40, int(self._ab_canvas.winfo_width()))
                h = max(40, int(self._ab_canvas.winfo_height()))
            except tk.TclError:
                return
            px, py = self._ab_map(a_s, b_s, w, h)
            self._ab_canvas.create_oval(
                px - 8, py - 8, px + 8, py + 8, outline="#111111", width=3, tags="pick"
            )
            self._ab_canvas.create_oval(
                px - 6, py - 6, px + 6, py + 6, outline="#ffffff", width=2, tags="pick"
            )
            hex_c = (
                f"#{int(round(fill[0]*255)):02x}"
                f"{int(round(fill[1]*255)):02x}"
                f"{int(round(fill[2]*255)):02x}"
            )
            self._ab_canvas.create_oval(
                px - 4, py - 4, px + 4, py + 4, fill=hex_c, outline="", tags="pick"
            )

    def _build_backend(self) -> None:
        for child in self.host.winfo_children():
            child.destroy()
        self._fig = None
        self._ax = None
        self._canvas = None
        self._mpl_widget = None
        self._ab_canvas = None
        if matplotlib_available():
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure

            self._fig = Figure(figsize=(5.2, 4.2), dpi=100)
            self._ax = self._fig.add_subplot(111, projection="3d")
            self._fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04)
            self._mpl_widget = FigureCanvasTkAgg(self._fig, master=self.host)
            self._canvas = self._mpl_widget.get_tk_widget()
            self._canvas.grid(row=0, column=0, sticky="nsew")
            self._status.set("CIE Lab · trackball orbit")
            self._bind_orbit_widget(self._canvas)
            try:
                self._ax.disable_mouse_rotation()
            except AttributeError:
                pass
        else:
            self._ab_canvas = tk.Canvas(self.host, bg="#1e1e1e", highlightthickness=0)
            self._ab_canvas.grid(row=0, column=0, sticky="nsew")
            self._ab_canvas.bind("<Configure>", lambda _e: self._redraw_ab(None))
            self._bind_orbit_widget(self._ab_canvas)
            self._status.set("CIE Lab a*–b* · pip install -r requirements-plot.txt for 3D")
        self._cached_data = None

    def _build_view_overlay(self) -> None:
        overlay = ttk.Frame(self.host)
        self.view_overlay = overlay
        self.view_cube = LabViewCube(overlay, self._on_cube_view)
        self.view_cube.pack()
        btns = ttk.Frame(overlay)
        btns.pack(fill="x", pady=(2, 0))
        self.center_btn = ttk.Button(btns, text="Center", width=7, command=self.center_view)
        self.center_btn.pack(side="left")
        self.xyz_btn = ttk.Button(btns, text="XYZ", width=4, command=self.cycle_xyz_view)
        self.xyz_btn.pack(side="left", padx=(2, 0))
        overlay.place(relx=1.0, rely=0.0, x=-4, y=4, anchor="ne")
        overlay.lift()
        self.view_cube.redraw(self._elev, self._azim)

    def _fitted_zoom_pct(self) -> float:
        """Zoom so the Lab half-box frames the cloud around COM."""
        data = self._cached_data
        look = cluster_look_target(data)
        if not data:
            return 100.0
        lab = np.asarray(data.get("lab", []))
        da = db = dl = 0.0
        if lab.ndim == 2 and lab.shape[0] and lab.shape[1] >= 3:
            da = float(np.max(np.abs(lab[:, 1] - look[0])))
            db = float(np.max(np.abs(lab[:, 2] - look[1])))
            dl = float(np.max(np.abs(lab[:, 0] - look[2])))
        for ext in _extents_of(data):
            cl, ca, cb = ext["center"]
            rl, ra, rb = ext["radii"]
            da = max(da, abs(float(ca) - look[0]) + float(ra))
            db = max(db, abs(float(cb) - look[1]) + float(rb))
            dl = max(dl, abs(float(cl) - look[2]) + float(rl))
        pad = 1.15
        za = 80.0 / max(da * pad, 8.0)
        zb = 80.0 / max(db * pad, 8.0)
        zl = 50.0 / max(dl * pad, 5.0)
        return max(100.0, min(800.0, min(za, zb, zl) * 100.0))

    def center_view(self) -> None:
        """Isometric orbit, fitted distance, look-at = COM (clears pan)."""
        self._look = cluster_look_target(self._cached_data)
        self._elev = DEFAULT_ELEV
        self._azim = DEFAULT_AZIM
        self._zoom_pct = self._fitted_zoom_pct()
        self._apply_camera()
        if self._ab_canvas is not None:
            self._redraw_ab(None)
        if self.on_zoom is not None:
            self.on_zoom(self._zoom_pct)

    def orbit_by(self, dx: float, dy: float) -> None:
        """Gyro orbit: elev/azim around COM. Pan offset stays in the camera frame."""
        old_elev, old_azim = self._elev, self._azim
        self._elev = max(-90.0, min(90.0, self._elev + float(dy) * _ORBIT_GAIN))
        self._azim = (self._azim - float(dx) * _ORBIT_GAIN) % 360.0
        self._reseat_look_around_com(old_elev, old_azim)
        self._apply_camera()

    def pan_by(self, dx: float, dy: float) -> None:
        """Translate look-at in the screen-right / screen-up plane (not an extra orbit)."""
        z = max(1.0, float(self._zoom_pct) / 100.0)
        scale = _PAN_GAIN / z
        right, up, _cam = camera_basis(self._elev, self._azim)
        delta = (-float(dx) * right + float(dy) * up) * scale
        self._look = (
            float(self._look[0] + delta[0]),
            float(self._look[1] + delta[1]),
            float(self._look[2] + delta[2]),
        )
        self._apply_camera()
        if self._ab_canvas is not None:
            self._redraw_ab(None)

    def _reseat_look_around_com(self, old_elev: float, old_azim: float) -> None:
        """Keep the COM-relative pan in camera space so rotate pivots on COM, not look-at."""
        com = np.asarray(cluster_look_target(self._cached_data), dtype=np.float64)
        offset = np.asarray(self._look, dtype=np.float64) - com
        r0, u0, c0 = camera_basis(old_elev, old_azim)
        r1, u1, c1 = camera_basis(self._elev, self._azim)
        oc = np.array(
            (
                float(np.dot(offset, r0)),
                float(np.dot(offset, u0)),
                float(np.dot(offset, c0)),
            ),
            dtype=np.float64,
        )
        moved = oc[0] * r1 + oc[1] * u1 + oc[2] * c1
        self._look = (
            float(com[0] + moved[0]),
            float(com[1] + moved[1]),
            float(com[2] + moved[2]),
        )

    def cycle_xyz_view(self) -> str:
        """Advance Front → Right → Back → Left → Top → Bottom (cube face views)."""
        name = XYZ_CYCLE[self._xyz_i % len(XYZ_CYCLE)]
        self.look_face(name)
        return name

    def look_face(self, name: str) -> None:
        """Aim the camera so cube face ``name`` looks toward you."""
        key = str(name or "").strip().lower()
        self._elev, azim = view_for_face(key)
        self._azim = _norm_azim(azim)
        if key in XYZ_CYCLE:
            self._xyz_i = (XYZ_CYCLE.index(key) + 1) % len(XYZ_CYCLE)
        self._apply_camera()

    def _on_cube_view(self, kind: str, payload) -> None:
        if kind == "face":
            self.look_face(str(payload))
            return
        if kind == "corner":
            sa, sb, sl = payload
            self._elev, azim = view_for_corner(int(sa), int(sb), int(sl))
            self._azim = _norm_azim(azim)
            self._apply_camera()
            return
        sa, sb, sl = payload
        self._elev, azim = view_for_edge(int(sa), int(sb), int(sl))
        self._azim = _norm_azim(azim)
        self._apply_camera()

    def _bind_orbit_widget(self, widget: tk.Misc) -> None:
        widget.bind("<ButtonPress-1>", self._on_press, add="+")
        widget.bind("<B1-Motion>", self._on_drag, add="+")
        widget.bind("<ButtonRelease-1>", self._on_release, add="+")
        widget.bind("<Double-Button-1>", self._on_double, add="+")
        widget.bind("<ButtonPress-3>", self._on_press, add="+")
        widget.bind("<B3-Motion>", self._on_drag, add="+")
        widget.bind("<ButtonPress-2>", self._on_mmb_press, add="+")
        widget.bind("<B2-Motion>", self._on_mmb_drag, add="+")
        widget.bind("<ButtonRelease-2>", self._on_mmb_release, add="+")

    def clear(self) -> None:
        self._cached_data = None
        self._last_key = None
        self._clear_pick_highlight()
        if self._ax is not None:
            self._ax.clear()
            self._ax.set_xlabel("a*")
            self._ax.set_ylabel("b*")
            self._ax.set_zlabel("L*")
            if self._mpl_widget is not None:
                self._mpl_widget.draw_idle()
        if self._ab_canvas is not None:
            self._ab_canvas.delete("all")

    def set_data(self, data: dict | None, *, force: bool = False) -> None:
        """Draw ``cluster_scatter_data`` output. Camera (orbit/zoom) is kept."""
        if data is None:
            self.clear()
            return
        key = (
            data["lab"].shape,
            data["mode"],
            tuple(data["match_rgb"]),
            tuple(data["replace_rgb"]),
            int(data["labels"].sum()) if data["labels"].size else 0,
            int(data["point_rgb"].sum()) if data["point_rgb"].size else 0,
        )
        first = self._cached_data is None
        if key == self._last_key and not force:
            self._cached_data = data
            return
        self._last_key = key
        self._cached_data = data
        if first:
            self._look = cluster_look_target(data)
        if self._ax is not None:
            self._draw_mpl(data)
        else:
            self._redraw_ab(data)

    def _draw_mpl(self, data: dict) -> None:
        ax = self._ax
        assert ax is not None and self._mpl_widget is not None
        self._remember_view()
        ax.clear()
        self._draw_cluster_clouds(ax, _extents_of(data))
        lab = data["lab"]
        if lab.shape[0]:
            colors = _rgb_norm(data["point_rgb"])
            ax.scatter(
                lab[:, 1],
                lab[:, 2],
                lab[:, 0],
                c=colors,
                s=8,
                depthshade=False,
                linewidths=0,
                alpha=0.85,
                picker=True,
                pickradius=6,
            )
        centers = data["centers_lab"]
        if centers.shape[0]:
            fill = _rgb_norm(data["match_rgb"])
            edge = _rgb_norm(data["replace_rgb"])
            ax.scatter(
                centers[:, 1],
                centers[:, 2],
                centers[:, 0],
                c=fill,
                s=90,
                marker="D",
                edgecolors=edge if edge.size else "k",
                linewidths=1.4,
                depthshade=False,
            )
        ax.set_xlabel("a*")
        ax.set_ylabel("b*")
        ax.set_zlabel("L*")
        self._apply_camera()
        self._draw_pick_highlight()
        self._mpl_widget.draw_idle()

    def _draw_cluster_clouds(self, ax, extents: list[dict]) -> None:
        """Faint axis-aligned Lab wireframe ellipsoids (behind the pixels)."""
        for ext in extents:
            if int(ext.get("n", 0)) <= 0:
                continue
            xs, ys, zs = _ellipsoid_mesh(ext["center"], ext["radii"])
            r, g, b = ext["rgb"]
            color = (r / 255.0, g / 255.0, b / 255.0)
            ax.plot_wireframe(
                xs,
                ys,
                zs,
                color=color,
                alpha=CLOUD_ALPHA,
                linewidth=0.45,
                rcount=CLOUD_WIRE_V,
                ccount=CLOUD_WIRE_U,
            )

    def _remember_view(self) -> None:
        ax = self._ax
        if ax is None:
            return
        try:
            self._elev = float(ax.elev)
            self._azim = float(ax.azim)
        except (AttributeError, TypeError, ValueError):
            pass

    def _apply_camera(self) -> None:
        ax = self._ax
        if ax is not None:
            z = max(1.0, float(self._zoom_pct) / 100.0)
            la, lb, ll = self._look
            ha, hb, hl = 80.0 / z, 80.0 / z, 50.0 / z
            ax.view_init(elev=self._elev, azim=self._azim)
            ax.set_xlim(la - ha, la + ha)
            ax.set_ylim(lb - hb, lb + hb)
            ax.set_zlim(ll - hl, ll + hl)
            if self._mpl_widget is not None:
                self._mpl_widget.draw_idle()
        if self.view_cube is not None:
            self.view_cube.redraw(self._elev, self._azim)

    def _shift_down(self, event) -> bool:
        try:
            return bool(int(getattr(event, "state", 0)) & 0x0001)
        except (TypeError, ValueError):
            return False

    def _on_press(self, event) -> None:
        self._press = (int(event.x_root), int(event.y_root))
        self._orbit_last = self._press
        self._did_drag = False

    def _on_drag(self, event) -> None:
        if self._mmb_last is not None:
            return
        if self._orbit_last is None:
            return
        x, y = int(event.x_root), int(event.y_root)
        dx = x - self._orbit_last[0]
        dy = y - self._orbit_last[1]
        if dx == 0 and dy == 0:
            return
        self._orbit_last = (x, y)
        if self._press is not None:
            if abs(x - self._press[0]) + abs(y - self._press[1]) >= _PICK_DRAG_PX:
                self._did_drag = True
        right = str(getattr(event, "num", "") or "") == "3" or bool(
            int(getattr(event, "state", 0) or 0) & 0x0400
        )
        if self._shift_down(event) or right:
            self.pan_by(dx, dy)
            return
        self.orbit_by(dx, dy)

    def _on_release(self, event) -> None:
        self._press = None
        self._orbit_last = None

    def _on_double(self, event) -> str | None:
        """Double-click (no drag) samples the nearest projected scatter point."""
        if self._did_drag:
            return "break"
        idx = self.nearest_index_at_root(int(event.x_root), int(event.y_root))
        if idx is not None:
            self.pick_index(idx)
        return "break"

    def _selected_start_lab(self) -> tuple[float, float, float] | None:
        data = self._cached_data
        idx = self._picked_index
        if data is not None and idx is not None:
            lab = np.asarray(data.get("lab"))
            if lab.ndim == 2 and 0 <= idx < lab.shape[0]:
                return clamp_lab_tuple(lab[idx])
        rgb = None
        if self.on_selected_rgb is not None:
            try:
                rgb = self.on_selected_rgb()
            except Exception:
                rgb = None
        if rgb is None:
            return None
        return clamp_lab_tuple(rgb_tuple_to_lab((int(rgb[0]), int(rgb[1]), int(rgb[2]))))

    def move_selected_by_pixels(self, dx: float, dy: float) -> tuple[int, int, int] | None:
        """Translate the selected color in the camera plane. Tests / MMB-drag."""
        if self._move_lab is None:
            start = self._selected_start_lab()
            if start is None:
                return None
            self._move_lab = start
        d_l, d_a, d_b = lab_delta_from_view_pixels(
            dx, dy, self._elev, self._azim, self._zoom_pct
        )
        self._move_lab = clamp_lab_tuple(
            (
                self._move_lab[0] + d_l,
                self._move_lab[1] + d_a,
                self._move_lab[2] + d_b,
            )
        )
        rgb = lab_tuple_to_rgb(self._move_lab)
        self._draw_pick_highlight()
        if self.on_move is not None:
            self.on_move(rgb)
        return rgb

    def _on_mmb_press(self, event) -> str | None:
        start = self._selected_start_lab()
        if start is None:
            self._status.set("Select a range half, then middle-drag to move it in Lab")
            return "break"
        self._move_lab = start
        self._mmb_last = (int(event.x_root), int(event.y_root))
        self._did_drag = False
        if self.on_move_start is not None:
            self.on_move_start()
        self._draw_pick_highlight()
        return "break"

    def _on_mmb_drag(self, event) -> str | None:
        if self._mmb_last is None:
            return "break"
        x, y = int(event.x_root), int(event.y_root)
        dx = x - self._mmb_last[0]
        dy = y - self._mmb_last[1]
        if dx == 0 and dy == 0:
            return "break"
        self._mmb_last = (x, y)
        self._did_drag = True
        self.move_selected_by_pixels(dx, dy)
        return "break"

    def _on_mmb_release(self, event) -> str | None:
        del event
        rgb = None
        if self._move_lab is not None:
            rgb = lab_tuple_to_rgb(self._move_lab)
        self._mmb_last = None
        if rgb is not None and self.on_move_end is not None:
            self.on_move_end(rgb)
        return "break"

    def projected_scatter_xy(self) -> np.ndarray | None:
        """Canvas-pixel XY of cached Lab points in the current projection."""
        data = self._cached_data
        if data is None or data["lab"].shape[0] == 0:
            return None
        lab = data["lab"]
        if self._ax is not None and self._canvas is not None:
            try:
                from mpl_toolkits.mplot3d import proj3d

                ax = self._ax
                x2, y2, _z2 = proj3d.proj_transform(
                    lab[:, 1], lab[:, 2], lab[:, 0], ax.get_proj()
                )
                disp = ax.transData.transform(np.column_stack([x2, y2]))
                pts = np.asarray(disp, dtype=np.float64)
                if pts.size == 0:
                    return None
                fig = self._fig
                if fig is not None:
                    h_in = float(fig.get_figheight()) * float(fig.dpi)
                    pts = pts.copy()
                    pts[:, 1] = h_in - pts[:, 1]
                return pts
            except (Exception, tk.TclError):
                pass
        if self._ab_canvas is not None:
            try:
                w = max(40, int(self._ab_canvas.winfo_width()))
                h = max(40, int(self._ab_canvas.winfo_height()))
            except tk.TclError:
                w, h = 200, 200
            pts = np.array(
                [self._ab_map(lab[i, 1], lab[i, 2], w, h) for i in range(lab.shape[0])],
                dtype=np.float64,
            )
            return pts
        pts = np.column_stack([lab[:, 1], -lab[:, 2]]).astype(np.float64)
        return pts

    def nearest_index_at_root(self, x_root: int, y_root: int) -> int | None:
        pts = self.projected_scatter_xy()
        if pts is None or pts.shape[0] == 0:
            return None
        host = self._canvas if self._canvas is not None else self._ab_canvas
        if host is None:
            return int(((pts[:, 0]) ** 2 + (pts[:, 1]) ** 2).argmin())
        try:
            cx = int(x_root) - int(host.winfo_rootx())
            cy = int(y_root) - int(host.winfo_rooty())
        except tk.TclError:
            return None
        d2 = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
        return int(d2.argmin())

    def _ab_map(self, a: float, b: float, w: int, h: int) -> tuple[float, float]:
        pad = 18
        z = max(1.0, self._zoom_pct / 100.0)
        la, lb, _ll = self._look
        ha, hb = 80.0 / z, 80.0 / z
        x = pad + (float(a) - (la - ha)) / max(1e-6, 2 * ha) * (w - 2 * pad)
        y = pad + ((lb + hb) - float(b)) / max(1e-6, 2 * hb) * (h - 2 * pad)
        return x, y

    def _redraw_ab(self, data: dict | None) -> None:
        canvas = self._ab_canvas
        if canvas is None:
            return
        payload = data if data is not None else self._cached_data
        canvas.delete("all")
        try:
            w = max(40, int(canvas.winfo_width()))
            h = max(40, int(canvas.winfo_height()))
        except tk.TclError:
            return
        pad = 18
        canvas.create_rectangle(0, 0, w, h, fill="#1e1e1e", outline="")
        if payload is None or payload["lab"].shape[0] == 0:
            canvas.create_text(
                w // 2, h // 2, text="Open an image to plot Lab clusters.", fill="#888888"
            )
            return
        lab = payload["lab"]
        rgb = np.asarray(payload["point_rgb"], dtype=np.uint8)
        for ext in _extents_of(payload):
            cl, ca, cb = ext["center"]
            rl, ra, rb = ext["radii"]
            x0, y0 = self._ab_map(ca - ra, cb + rb, w, h)
            x1, y1 = self._ab_map(ca + ra, cb - rb, w, h)
            r, g, b = ext["rgb"]
            mix = CLOUD_ALPHA + 0.08
            fr = int(30 + mix * (r - 30))
            fg = int(30 + mix * (g - 30))
            fb = int(30 + mix * (b - 30))
            canvas.create_oval(
                min(x0, x1),
                min(y0, y1),
                max(x0, x1),
                max(y0, y1),
                outline=f"#{fr:02x}{fg:02x}{fb:02x}",
                width=1,
            )
        for i in range(lab.shape[0]):
            px, py = self._ab_map(lab[i, 1], lab[i, 2], w, h)
            color = f"#{int(rgb[i, 0]):02x}{int(rgb[i, 1]):02x}{int(rgb[i, 2]):02x}"
            canvas.create_rectangle(px, py, px + 2, py + 2, fill=color, outline="")
        for i, center in enumerate(payload["centers_lab"]):
            px, py = self._ab_map(center[1], center[2], w, h)
            match = payload["match_rgb"][i] if i < len(payload["match_rgb"]) else (200, 200, 200)
            repl = payload["replace_rgb"][i] if i < len(payload["replace_rgb"]) else (20, 20, 20)
            fill = f"#{match[0]:02x}{match[1]:02x}{match[2]:02x}"
            edge = f"#{repl[0]:02x}{repl[1]:02x}{repl[2]:02x}"
            canvas.create_rectangle(px - 5, py - 5, px + 5, py + 5, fill=fill, outline=edge, width=2)
        canvas.create_text(pad, h - 6, text="a*", fill="#aaaaaa", anchor="sw")
        canvas.create_text(w - 6, pad, text="b*", fill="#aaaaaa", anchor="ne")
        self._draw_pick_highlight()
