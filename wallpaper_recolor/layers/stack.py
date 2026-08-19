# -*- coding: utf-8 -*-
"""
wallpaper_recolor.layers.stack
------------------------------
Ordered document layers (images + labels). Index 0 is the front (paints last).

Hidden layers are skipped by composite, Build, and color/lighting. Recolor,
tone, tessellate Build, and Normalize lighting apply only to **selected**
visible Image layers. A selected Label is edited by color/font tools instead.

Class references (code + name only):
- CAP3321C Data Wrangling
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence
import itertools
import json

import numpy as np
from PIL import Image

from wallpaper_recolor.labels.boxes import source_xy_to_display
from wallpaper_recolor.labels.layer import (
    LABEL_COLOR_DEFAULT,
    LABEL_SIZE_DEFAULT,
    LabelSpec,
    clamp_label_size,
    composite_label,
    parse_label_color,
    rgb_to_hex,
)

LAYER_IMAGE = "image"
LAYER_LABEL = "label"
LAYER_GROUP = "group"
LAYER_RANGE = "range"
ROLE_BASE = "base"
ROLE_RANGE_GROUP = "color_ranges"

_SCALE_MIN = 0.05
_SCALE_MAX = 8.0


def clamp_layer_scale(value: object) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        n = 1.0
    if n != n:  # NaN
        n = 1.0
    return max(_SCALE_MIN, min(_SCALE_MAX, n))


def default_font_family() -> str:
    return "Segoe UI"


@dataclass
class StackLayer:
    """One stack row. ``raster`` is in-memory only (not written to .wpedit)."""

    id: str
    name: str
    kind: str
    visible: bool = True
    x: int = 0
    y: int = 0
    scale: float = 1.0
    path: str = ""
    role: str = ""
    raster: Image.Image | None = field(default=None, repr=False, compare=False)
    text: str = ""
    size: int = LABEL_SIZE_DEFAULT
    color: tuple[int, int, int] = (34, 34, 34)
    font: str = ""
    parent_id: str = ""
    range_index: int = -1
    expanded: bool = True  # Color ranges group twisty (default open)

    def is_image(self) -> bool:
        return self.kind == LAYER_IMAGE

    def is_label(self) -> bool:
        return self.kind == LAYER_LABEL

    def is_group(self) -> bool:
        return self.kind == LAYER_GROUP

    def is_range(self) -> bool:
        return self.kind == LAYER_RANGE

    def is_range_group(self) -> bool:
        return self.kind == LAYER_GROUP and self.role == ROLE_RANGE_GROUP

    def is_base(self) -> bool:
        return self.kind == LAYER_IMAGE and self.role == ROLE_BASE

    def to_label_spec(self) -> LabelSpec:
        return LabelSpec(
            text=str(self.text or ""),
            size=clamp_label_size(self.size),
            color=tuple(int(c) for c in self.color),
            x=int(self.x),
            y=int(self.y),
            font=str(self.font or default_font_family()),
        )

    def apply_label_spec(self, spec: LabelSpec) -> None:
        self.text = str(spec.text or "")
        self.size = clamp_label_size(spec.size)
        self.color = tuple(int(c) for c in spec.color)
        self.x = int(spec.x)
        self.y = int(spec.y)
        self.font = str(getattr(spec, "font", "") or self.font or default_font_family())

    def to_record(self) -> dict:
        """JSON-safe row for .wpedit / undo (no pixel buffers)."""
        rec = {
            "id": self.id,
            "name": self.name,
            "type": self.kind,
            "visible": bool(self.visible),
            "x": int(self.x),
            "y": int(self.y),
            "scale": float(clamp_layer_scale(self.scale)),
            "path": str(self.path or ""),
            "role": str(self.role or ""),
            "parent_id": str(self.parent_id or ""),
            "range_index": int(self.range_index),
            "expanded": bool(self.expanded),
        }
        if self.is_label():
            rec["text"] = str(self.text or "")
            rec["font"] = str(self.font or default_font_family())
            rec["size"] = int(clamp_label_size(self.size))
            rec["color"] = rgb_to_hex(self.color)
        return rec

    @classmethod
    def from_record(cls, data: dict, *, raster: Image.Image | None = None) -> "StackLayer":
        kind = str(data.get("type") or data.get("kind") or LAYER_IMAGE)
        if kind not in (LAYER_IMAGE, LAYER_LABEL, LAYER_GROUP, LAYER_RANGE):
            kind = LAYER_IMAGE
        color = data.get("color")
        if isinstance(color, (list, tuple)) and len(color) >= 3:
            rgb = (int(color[0]), int(color[1]), int(color[2]))
        else:
            rgb = parse_label_color(str(color or LABEL_COLOR_DEFAULT))
        return cls(
            id=str(data.get("id") or ""),
            name=str(data.get("name") or ("Label" if kind == LAYER_LABEL else "Image")),
            kind=kind,
            visible=bool(data.get("visible", True)),
            x=int(data.get("x") or 0),
            y=int(data.get("y") or 0),
            scale=clamp_layer_scale(data.get("scale", 1.0)),
            path=str(data.get("path") or data.get("image_path") or ""),
            role=str(data.get("role") or ""),
            raster=raster,
            text=str(data.get("text") or ""),
            size=clamp_label_size(data.get("size", LABEL_SIZE_DEFAULT)),
            color=rgb,
            font=str(data.get("font") or default_font_family()),
            parent_id=str(data.get("parent_id") or ""),
            range_index=int(data.get("range_index") if data.get("range_index") is not None else -1),
            expanded=bool(data.get("expanded", True)),
        )


class LayerStack:
    """Front-to-back list (``layers[0]`` is on top) plus a selection of ids."""

    def __init__(self) -> None:
        self.layers: list[StackLayer] = []
        self.selected_ids: list[str] = []
        self._ids = itertools.count(1)

    def _new_id(self) -> str:
        used = {ly.id for ly in self.layers}
        while True:
            candidate = f"L{next(self._ids):04d}"
            if candidate not in used:
                return candidate

    def clear(self) -> None:
        self.layers.clear()
        self.selected_ids.clear()

    def get(self, layer_id: str) -> StackLayer | None:
        for ly in self.layers:
            if ly.id == layer_id:
                return ly
        return None

    def base_layer(self) -> StackLayer | None:
        for ly in self.layers:
            if ly.is_base():
                return ly
        for ly in self.layers:
            if ly.is_image():
                return ly
        return None

    def primary(self) -> StackLayer | None:
        """Last-clicked selected layer, or the frontmost selected row."""
        if self.selected_ids:
            found = self.get(self.selected_ids[-1])
            if found is not None:
                return found
        chosen = set(self.selected_ids)
        for ly in self.layers:
            if ly.id in chosen:
                return ly
        return self.layers[0] if self.layers else None

    def selected(self) -> list[StackLayer]:
        chosen = set(self.selected_ids)
        return [ly for ly in self.layers if ly.id in chosen]

    def selected_image_layers(self, *, visible_only: bool = True) -> list[StackLayer]:
        out = []
        for ly in self.selected():
            if not ly.is_image():
                continue
            if visible_only and not ly.visible:
                continue
            out.append(ly)
        return out

    def image_for_selection(self, ly: StackLayer | None) -> StackLayer | None:
        """Owning Image for a row: itself, or parent of Color ranges / Range N."""
        if ly is None:
            return None
        if ly.is_image():
            return ly
        cur = ly
        seen: set[str] = set()
        while cur is not None and cur.id not in seen:
            seen.add(cur.id)
            parent = self.get(cur.parent_id) if cur.parent_id else None
            if parent is None:
                break
            if parent.is_image():
                return parent
            cur = parent
        return None

    def visible_bottom_to_top(self) -> list[StackLayer]:
        return [ly for ly in reversed(self.layers) if ly.visible]

    def select(self, layer_id: str, *, additive: bool = False) -> None:
        if self.get(layer_id) is None:
            return
        if additive:
            if layer_id not in self.selected_ids:
                self.selected_ids.append(layer_id)
            return
        self.selected_ids = [layer_id]

    def add_image(
        self,
        *,
        name: str,
        path: str = "",
        raster: Image.Image | None = None,
        role: str = "",
        visible: bool = True,
        x: int = 0,
        y: int = 0,
        scale: float = 1.0,
        layer_id: str | None = None,
        select: bool = True,
    ) -> StackLayer:
        ly = StackLayer(
            id=layer_id or self._new_id(),
            name=name or "Image",
            kind=LAYER_IMAGE,
            visible=visible,
            x=int(x),
            y=int(y),
            scale=clamp_layer_scale(scale),
            path=str(path or ""),
            role=str(role or ""),
            raster=raster,
        )
        self.layers.insert(0, ly)
        if select:
            self.select(ly.id)
        return ly

    def add_label(
        self,
        *,
        name: str = "Label",
        text: str = "",
        size: int = LABEL_SIZE_DEFAULT,
        color: tuple[int, int, int] = (34, 34, 34),
        font: str = "",
        x: int = 0,
        y: int = 0,
        visible: bool = True,
        layer_id: str | None = None,
        select: bool = True,
    ) -> StackLayer:
        ly = StackLayer(
            id=layer_id or self._new_id(),
            name=name or "Label",
            kind=LAYER_LABEL,
            visible=visible,
            x=int(x),
            y=int(y),
            text=str(text or ""),
            size=clamp_label_size(size),
            color=tuple(int(c) for c in color),
            font=str(font or default_font_family()),
        )
        self.layers.insert(0, ly)
        if select:
            self.select(ly.id)
        return ly

    def remove(self, layer_id: str) -> StackLayer | None:
        ly = self.get(layer_id)
        if ly is None or ly.is_base() or ly.is_range() or ly.is_range_group():
            return None
        self.layers = [row for row in self.layers if row.id != layer_id]
        self.selected_ids = [i for i in self.selected_ids if i != layer_id]
        if not self.selected_ids and self.layers:
            self.selected_ids = [self.layers[0].id]
        return ly

    def children_of(self, parent_id: str) -> list[StackLayer]:
        pid = str(parent_id or "")
        return [ly for ly in self.layers if str(ly.parent_id or "") == pid]

    def range_group_for(self, image_id: str) -> StackLayer | None:
        for ly in self.layers:
            if ly.is_range_group() and ly.parent_id == image_id:
                return ly
        return None

    def sync_range_children(
        self,
        image_id: str,
        n: int,
        visibilities: Sequence[bool] | None = None,
        names: Sequence[str] | None = None,
    ) -> StackLayer | None:
        """Keep a Color ranges group under ``image_id`` with ``n`` range children."""
        parent = self.get(image_id)
        if parent is None or not parent.is_image():
            return None
        n = max(0, int(n))
        group = self.range_group_for(image_id)
        if group is None:
            group = StackLayer(
                id=self._new_id(),
                name="Color ranges",
                kind=LAYER_GROUP,
                parent_id=image_id,
                role=ROLE_RANGE_GROUP,
                visible=True,
            )
            idx = next(i for i, ly in enumerate(self.layers) if ly.id == image_id)
            self.layers.insert(idx + 1, group)
        kids = [ly for ly in self.layers if ly.is_range() and ly.parent_id == group.id]
        by_idx = {int(ly.range_index): ly for ly in kids if int(ly.range_index) >= 0}
        keep_ids = {group.id}
        new_kids: list[StackLayer] = []
        for i in range(n):
            vis = True
            if visibilities is not None and i < len(visibilities):
                vis = bool(visibilities[i])
            name = f"Range {i + 1}"
            if names is not None and i < len(names) and str(names[i] or "").strip():
                name = str(names[i])
            old = by_idx.get(i)
            if old is None:
                old = StackLayer(
                    id=self._new_id(),
                    name=name,
                    kind=LAYER_RANGE,
                    parent_id=group.id,
                    range_index=i,
                    visible=vis,
                )
            else:
                old.name = name
                old.visible = vis
                old.parent_id = group.id
                old.range_index = i
            new_kids.append(old)
            keep_ids.add(old.id)
        others = [
            ly
            for ly in self.layers
            if not (ly.is_range() and ly.parent_id == group.id)
        ]
        insert_at = next(i for i, ly in enumerate(others) if ly.id == group.id) + 1
        self.layers = others[:insert_at] + new_kids + others[insert_at:]
        return group

    def walk_visible_tree(self, parent_id: str = "") -> list[tuple[StackLayer, int]]:
        """Pre-order (layer, depth) for the Layers panel, starting at ``parent_id``.

        Collapsed groups (``expanded`` is False) still appear; their children do not.
        """
        rows: list[tuple[StackLayer, int]] = []

        def walk(pid: str, depth: int) -> None:
            for ly in self.children_of(pid):
                rows.append((ly, depth))
                if ly.is_group() and not bool(ly.expanded):
                    continue
                walk(ly.id, depth + 1)

        walk(str(parent_id or ""), 0)
        return rows

    def set_expanded(self, layer_id: str, expanded: bool) -> bool:
        ly = self.get(layer_id)
        if ly is None or not ly.is_group():
            return False
        ly.expanded = bool(expanded)
        return True

    def move_up(self, layer_id: str) -> bool:
        """Toward the front (lower index)."""
        idx = next((i for i, ly in enumerate(self.layers) if ly.id == layer_id), -1)
        if idx <= 0:
            return False
        self.layers[idx - 1], self.layers[idx] = self.layers[idx], self.layers[idx - 1]
        return True

    def move_down(self, layer_id: str) -> bool:
        """Toward the back (higher index)."""
        idx = next((i for i, ly in enumerate(self.layers) if ly.id == layer_id), -1)
        if idx < 0 or idx >= len(self.layers) - 1:
            return False
        self.layers[idx + 1], self.layers[idx] = self.layers[idx], self.layers[idx + 1]
        return True

    def set_visible(self, layer_id: str, visible: bool) -> None:
        ly = self.get(layer_id)
        if ly is not None:
            ly.visible = bool(visible)

    def to_records(self) -> list[dict]:
        return [ly.to_record() for ly in self.layers]

    def replace_from_records(
        self,
        records: Sequence[dict],
        selected_ids: Sequence[str] | None = None,
        *,
        rasters: dict[str, Image.Image] | None = None,
    ) -> None:
        keep = rasters or {}
        new_layers: list[StackLayer] = []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            rid = str(rec.get("id") or "")
            ly = StackLayer.from_record(rec, raster=keep.get(rid))
            if not ly.id:
                ly.id = self._new_id()
            new_layers.append(ly)
        self.layers = new_layers
        ids = {ly.id for ly in self.layers}
        if selected_ids:
            self.selected_ids = [str(i) for i in selected_ids if str(i) in ids]
        else:
            self.selected_ids = [self.layers[0].id] if self.layers else []
        if not self.selected_ids and self.layers:
            self.selected_ids = [self.layers[0].id]


def correction_target_ids(
    stack: LayerStack,
    selected_ids: Sequence[str] | None = None,
) -> set[str]:
    """Visible Image layers that should receive recolor / tone / lighting.

    Range N and Color ranges rows resolve to their parent Image. A Label (or
    empty) selection falls back to the base wallpaper so Result still remaps.
    An explicitly selected overlay Image still excludes the unselected base.
    """
    chosen = [str(i) for i in (selected_ids if selected_ids is not None else stack.selected_ids)]
    ids: set[str] = set()
    for sid in chosen:
        image = stack.image_for_selection(stack.get(sid))
        if image is not None and image.visible and image.is_image():
            ids.add(image.id)
    if not ids:
        base = stack.base_layer()
        if base is not None and base.visible:
            ids.add(base.id)
    return ids


def inpaint_target_layer(stack: LayerStack) -> StackLayer | None:
    """Image under text: selected visible Image, else the base wallpaper."""
    for ly in stack.selected():
        if ly.visible and ly.is_image():
            return ly
    return stack.base_layer()


def primary_is_label(stack: LayerStack) -> bool:
    ly = stack.primary()
    return ly is not None and ly.is_label()


def _paste_image(
    canvas: Image.Image,
    image: Image.Image,
    x: int,
    y: int,
    scale: float = 1.0,
) -> Image.Image:
    plate = canvas if canvas.mode == "RGBA" else canvas.convert("RGBA")
    img = image.convert("RGBA")
    sc = clamp_layer_scale(scale)
    if abs(sc - 1.0) > 1e-4:
        nw = max(1, int(round(img.size[0] * sc)))
        nh = max(1, int(round(img.size[1] * sc)))
        img = img.resize((nw, nh), Image.Resampling.BILINEAR)
    layer = Image.new("RGBA", plate.size, (0, 0, 0, 0))
    layer.paste(img, (int(x), int(y)), img)
    return Image.alpha_composite(plate, layer)


def composite_stack(
    stack: LayerStack,
    canvas_size: tuple[int, int],
    source_size: tuple[int, int],
    *,
    processed: dict[str, Image.Image] | None = None,
    crop_x: float = 0.0,
    crop_y: float = 0.0,
    crop_zoom: float = 1.0,
    map_xy: bool = True,
) -> Image.Image:
    """Paint visible layers bottom → top onto a transparent canvas.

    ``processed[id]`` replaces that image layer's raster (recolor / tessellate).
    ``map_xy`` maps source x/y through Crop onto ``canvas_size``.
    """
    w, h = max(1, int(canvas_size[0])), max(1, int(canvas_size[1]))
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    cooked = processed or {}
    for ly in stack.visible_bottom_to_top():
        if ly.is_group() or ly.is_range():
            continue
        if ly.is_image():
            img = cooked.get(ly.id, ly.raster)
            if img is None:
                continue
            if not map_xy:
                canvas = _paste_image(canvas, img, ly.x, ly.y, ly.scale)
                continue
            dx, dy, sc = source_xy_to_display(
                ly.x, ly.y, (w, h), source_size, crop_x, crop_y, crop_zoom
            )
            if ly.is_base() and img.size == (w, h):
                dx0, dy0, _sc0 = source_xy_to_display(
                    0, 0, (w, h), source_size, crop_x, crop_y, crop_zoom
                )
                canvas = _paste_image(
                    canvas,
                    img,
                    int(round(dx - dx0)),
                    int(round(dy - dy0)),
                    ly.scale,
                )
            else:
                canvas = _paste_image(
                    canvas, img, int(round(dx)), int(round(dy)), ly.scale * sc
                )
            continue
        spec = ly.to_label_spec()
        if not spec.is_set():
            continue
        canvas = composite_label(
            canvas,
            spec,
            source_size,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_zoom=crop_zoom,
        )
    return canvas


CHECKER_LIGHT = (255, 255, 255)
CHECKER_MID = (204, 204, 204)  # slightly darker gray
CHECKER_TILE_PX = 8  # modest squares on the preview (8–16 px)


def checkerboard_rgb(
    width: int,
    height: int,
    *,
    tile: int = CHECKER_TILE_PX,
    light: tuple[int, int, int] = CHECKER_LIGHT,
    dark: tuple[int, int, int] = CHECKER_MID,
) -> Image.Image:
    """Photoshop-style empty-plate checker (preview only; never saved)."""
    w, h = max(1, int(width)), max(1, int(height))
    cell = max(1, int(tile))
    yy, xx = np.ogrid[:h, :w]
    chk = ((xx // cell) + (yy // cell)) & 1
    rgb = np.empty((h, w, 3), dtype=np.uint8)
    rgb[:] = light
    rgb[chk == 1] = dark
    return Image.fromarray(rgb, mode="RGB")


def composite_over_checker(
    image: Image.Image,
    *,
    tile: int = CHECKER_TILE_PX,
) -> Image.Image:
    """Blit RGBA over a checker. Opaque RGB is unchanged (no plate baked in)."""
    if image.mode == "RGB":
        return image
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    if alpha.getextrema()[0] >= 255:
        return rgba.convert("RGB")
    bg = checkerboard_rgb(rgba.size[0], rgba.size[1], tile=tile)
    bg.paste(rgba, mask=alpha)
    return bg


def flatten_rgb(image: Image.Image, *, background: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Composite onto an opaque RGB plate (save flatten when filling holes)."""
    if image.mode == "RGB":
        return image
    rgba = image.convert("RGBA")
    bg = Image.new("RGB", rgba.size, background)
    bg.paste(rgba, mask=rgba.split()[-1])
    return bg


def flatten_rgb_or_keep_alpha(image: Image.Image) -> Image.Image:
    """Keep RGBA when any pixel is transparent; otherwise opaque RGB (no checker)."""
    if image.mode == "RGB":
        return image
    rgba = image.convert("RGBA")
    if rgba.getchannel("A").getextrema()[0] >= 255:
        return flatten_rgb(rgba)
    return rgba


def records_json(records: Sequence[dict]) -> str:
    return json.dumps(list(records), indent=2)


def load_overlay_image(path: str | Path) -> Image.Image:
    from wallpaper_recolor.io.image_io import load_image

    return load_image(path)
