# -*- coding: utf-8 -*-
"""
wallpaper_recolor.preview.preview_tools
-------------------------------
Repeat inspection (3×3 tile, Photoshop-style Offset 50%/50%) and a simple
room mockup with adjustable wallpaper scale and wall-cover height.
Preview-sized — not 9× print.

Class references (code + name only):
- CAP3321C Data Wrangling
"""

from __future__ import annotations  # tuple[int, int, int, int]

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Back-wall paint when wallpaper does not cover the full height (beige-grey).
BACK_WALL_RGB = (196, 190, 182)

# Vertical fraction of the back wall covered from the floor up.
MOCKUP_COVER_FULL = 1.0
MOCKUP_COVER_HALF = 0.5
MOCKUP_COVER_THIRD = 1.0 / 3.0
MOCKUP_COVER_QUARTER = 0.25
MOCKUP_COVER_FRACS = {
    "full": MOCKUP_COVER_FULL,
    "half": MOCKUP_COVER_HALF,
    "third": MOCKUP_COVER_THIRD,
    "quarter": MOCKUP_COVER_QUARTER,
}
MOCKUP_COVER_LABELS = {
    "full": "Full",
    "half": "½",
    "third": "⅓",
    "quarter": "¼",
}


def cover_frac_from_key(key: str, default: float = MOCKUP_COVER_FULL) -> float:
    """Map a UI cover key (full/half/third/quarter) to a 0–1 fraction."""
    return float(MOCKUP_COVER_FRACS.get(str(key), default))


def fit_max_edge(image: Image.Image, max_edge: int) -> Image.Image:
    """Downscale so the long side is at most ``max_edge``."""
    w, h = image.size
    long_edge = max(w, h)
    if long_edge <= max_edge:
        return image.copy()
    scale = max_edge / long_edge
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return image.resize(size, Image.Resampling.BILINEAR)


def tile_repeat(image: Image.Image, grid: int = 3, cell_max_edge: int = 420) -> Image.Image:
    """``grid`` × ``grid`` tiled preview of a repeating wallpaper."""
    cell = fit_max_edge(image.convert("RGB"), cell_max_edge)
    w, h = cell.size
    canvas = Image.new("RGB", (w * grid, h * grid))
    for y in range(grid):
        for x in range(grid):
            canvas.paste(cell, (x * w, y * h))
    return canvas


def offset_seam(image: Image.Image, cell_max_edge: int = 720) -> Image.Image:
    """Photoshop Offset 50% / 50% — wrap seams land in the middle for inspection."""
    src = fit_max_edge(image.convert("RGB"), cell_max_edge)
    arr = np.asarray(src)
    dy, dx = arr.shape[0] // 2, arr.shape[1] // 2
    rolled = np.roll(np.roll(arr, dy, axis=0), dx, axis=1)
    return Image.fromarray(rolled, mode="RGB")


def make_room_plate(width: int = 1200, height: int = 780) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Gray interior plate. Returns (room RGB, back-wall x0,y0,x1,y1)."""
    room = Image.new("RGB", (width, height), (210, 204, 196))
    draw = ImageDraw.Draw(room)

    x0, y0 = int(width * 0.20), int(height * 0.11)
    x1, y1 = int(width * 0.86), int(height * 0.60)

    # Floor — trapezoid into the camera
    draw.polygon(
        [(x0, y1), (x1, y1), (width, height), (0, height)],
        fill=(118, 112, 106),
    )
    # Faint floor boards (perspective: closer together toward the wall)
    for i in range(1, 10):
        t = i / 10.0
        lx = int(x0 * (1.0 - t))
        rx = int(x1 + (width - x1) * t)
        yy = int(y1 + (height - y1) * (t ** 1.35))
        draw.line([(lx, yy), (rx, yy)], fill=(102, 96, 90), width=1)

    # Left / right walls
    draw.polygon(
        [(0, int(height * 0.05)), (x0, y0), (x0, y1), (0, height)],
        fill=(166, 160, 152),
    )
    draw.polygon(
        [(width, int(height * 0.07)), (x1, y0), (x1, y1), (width, height)],
        fill=(174, 168, 160),
    )

    # Ceiling plane hint
    draw.polygon(
        [(0, int(height * 0.05)), (width, int(height * 0.07)), (x1, y0), (x0, y0)],
        fill=(226, 222, 214),
    )

    # Back wall — beige-grey paint; wallpaper may cover only a floor-up fraction
    draw.rectangle([x0, y0, x1, y1], fill=BACK_WALL_RGB)
    return room, (x0, y0, x1, y1)


def _tiled_wallpaper(tile: Image.Image, wall_w: int, wall_h: int, repeats_x: float) -> Image.Image:
    """Fill a wall rectangle with ``repeats_x`` horizontal repeats of ``tile``."""
    repeats_x = max(1.0, float(repeats_x))
    tw = max(8, int(round(wall_w / repeats_x)))
    scale = tw / max(1, tile.width)
    th = max(8, int(round(tile.height * scale)))
    cell = tile.convert("RGB").resize((tw, th), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (wall_w, wall_h))
    y = 0
    while y < wall_h:
        x = 0
        while x < wall_w:
            canvas.paste(cell, (x, y))
            x += tw
        y += th
    return canvas


def room_mockup(
    wallpaper: Image.Image,
    repeats_x: float = 4.0,
    width: int = 1200,
    height: int = 780,
    cover_frac: float = MOCKUP_COVER_FULL,
) -> Image.Image:
    """Wallpaper on the back wall; ``cover_frac`` is the floor-up height (1 = full)."""
    room, (x0, y0, x1, y1) = make_room_plate(width, height)
    wall_w, wall_h = max(1, x1 - x0), max(1, y1 - y0)
    paper = _tiled_wallpaper(wallpaper, wall_w, wall_h, repeats_x)
    try:
        frac = float(cover_frac)
    except (TypeError, ValueError):
        frac = MOCKUP_COVER_FULL
    frac = min(1.0, max(0.0, frac))
    cover_h = int(round(wall_h * frac))
    if cover_h >= wall_h:
        room.paste(paper, (x0, y0))
    elif cover_h > 0:
        paper = paper.crop((0, wall_h - cover_h, wall_w, wall_h))
        room.paste(paper, (x0, y1 - cover_h))

    # Baseboard + a short floor shadow so the paper reads as sitting on a wall
    draw = ImageDraw.Draw(room)
    bb = max(6, wall_h // 70)
    draw.rectangle([x0, y1 - bb, x1, y1], fill=(62, 58, 54))
    shadow = Image.new("L", (wall_w, max(12, wall_h // 14)), 0)
    sdraw = ImageDraw.Draw(shadow)
    for i in range(shadow.height):
        a = int(90 * (1.0 - i / max(1, shadow.height - 1)))
        sdraw.line([(0, i), (wall_w, i)], fill=a)
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=3))
    dark = Image.new("RGB", shadow.size, (20, 16, 12))
    room.paste(dark, (x0, y1), shadow)

    # Thin inner edge so the wall rectangle reads in the plate
    draw.rectangle([x0, y0, x1, y1], outline=(48, 44, 40), width=1)
    return room.convert("RGB")
