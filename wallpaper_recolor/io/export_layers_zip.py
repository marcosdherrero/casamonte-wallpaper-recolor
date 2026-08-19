# -*- coding: utf-8 -*-
"""
wallpaper_recolor.io.export_layers_zip
-----------------------------------
Write a .zip of per-range color plates (TIF + SVG) plus texture and a
flattened composite, at the same Scale/DPI size as Save as….

Color TIF: that range only. RGB is the on-screen pixels in the range
(solid fill when the texture eye is off, Color/Luminosity grain when it
is on); alpha is the range mask. Photoshop Normal, bottom → top, rebuilds
the composite (hidden ranges are omitted; composite has transparent holes).

SVG is a wrapper (not an Image Trace): viewBox in px, optional mm size
from DPI, ``<image href="sibling.png">`` so Chrome can show the stack.
TIFF plates sit beside the PNGs for Photoshop.

Class references (code + name only):
- CAP3321C Data Wrangling
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from xml.sax.saxutils import escape
import json
import tempfile
import zipfile

import numpy as np
from PIL import Image

from wallpaper_recolor.color.color_ranges import ColorRangeMap, is_color_split
from wallpaper_recolor.labels.layer import LabelSpec, write_label_files
from wallpaper_recolor.transform.crop import apply_crop_array, is_identity_crop
from wallpaper_recolor.transform.inpaint import inpaint_image
from wallpaper_recolor.transform.tessellate import (
    apply_crop_lighting_tessellate,
    coerce_built,
    coerce_normalize_lighting,
    is_identity_tessellate,
    tessellate_array,
)
from wallpaper_recolor.io.image_io import save_image
from wallpaper_recolor.color.layers import (
    effective_texture_strength,
    labeled_composite_for_image,
)
from wallpaper_recolor.transform.scale import DEFAULT_RESAMPLE, scale_image

# 1 in = 25.4 mm — SVG print size when DPI is tagged
_MM_PER_INCH = 25.4


def _hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def _slug(name: str, index: int) -> str:
    raw = "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")
    while "__" in raw:
        raw = raw.replace("__", "_")
    return raw or f"band_{index:02d}"


def _color_stem(band, index: int) -> str:
    """01_name, keeping the range index so a hidden eye leaves a gap, not a rename."""
    return f"{index + 1:02d}_{_slug(band.name, index)}"


def _note(msg: str, on_status: Callable[[str], None] | None) -> None:
    if on_status:
        on_status(msg)


def _scale_labels(labels: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor so labels stay a partition after Scale."""
    h, w = int(labels.shape[0]), int(labels.shape[1])
    if (w, h) == size:
        return labels.astype(np.int32, copy=False)
    im = Image.fromarray(labels.astype(np.int32), mode="I")
    out = im.resize(size, Image.Resampling.NEAREST)
    return np.asarray(out, dtype=np.int32)


def _rgba_plate(rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    """RGB from the master; alpha = range mask (optionally × source alpha)."""
    rgba = np.empty((*mask.shape, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = mask
    return Image.fromarray(rgba, mode="RGBA")


def _svg_size_attrs(width_px: int, height_px: int, dpi: float | None) -> str:
    """width/height in px, plus mm when DPI is known (Scale panel)."""
    w, h = int(width_px), int(height_px)
    if dpi is not None and float(dpi) > 0.0:
        w_mm = w / float(dpi) * _MM_PER_INCH
        h_mm = h / float(dpi) * _MM_PER_INCH
        return (
            f'width="{w_mm:.4f}mm" height="{h_mm:.4f}mm" '
            f'viewBox="0 0 {w} {h}"'
        )
    return f'width="{w}px" height="{h}px" viewBox="0 0 {w} {h}"'


def _svg_image_tag(href: str, width_px: int, height_px: int, blend: str | None = None) -> str:
    safe = escape(href, {'"': "&quot;"})
    blend_attr = ""
    if blend:
        blend_attr = f' style="mix-blend-mode: {escape(blend)}"'
    return (
        f'  <image href="{safe}" xlink:href="{safe}" x="0" y="0" '
        f'width="{int(width_px)}" height="{int(height_px)}" '
        f'preserveAspectRatio="none"{blend_attr}/>'
    )


def _svg_document(
    width_px: int,
    height_px: int,
    images: list[tuple[str, str | None]],
    *,
    dpi: float | None = None,
    title: str | None = None,
) -> str:
    """Layered SVG wrapper: sibling raster hrefs, no tracing."""
    size = _svg_size_attrs(width_px, height_px, dpi)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" {size}>',
    ]
    if title:
        lines.append(f"  <title>{escape(title)}</title>")
    for href, blend in images:
        lines.append(_svg_image_tag(href, width_px, height_px, blend))
    lines.append("</svg>")
    lines.append("")
    return "\n".join(lines)


def _write_raster_pair(image: Image.Image, dest_dir: Path, stem: str, dpi: float | None) -> None:
    """TIF (Photoshop) + PNG (SVG href). Fast encode — no PNG optimize."""
    save_image(image, dest_dir / f"{stem}.tif", dpi=dpi)
    save_image(image, dest_dir / f"{stem}.png", dpi=dpi)


def _zip_add_file(zf: zipfile.ZipFile, path: Path) -> None:
    """Store rasters as-is; deflate JSON/SVG/README only.

    PNG and Adobe-Deflate TIF are already zlib-wrapped. ZIP_DEFLATED on those
    is a long CPU pass that barely shrinks the archive.
    """
    suffix = path.suffix.lower()
    compress = (
        zipfile.ZIP_STORED if suffix in {".tif", ".tiff", ".png"} else zipfile.ZIP_DEFLATED
    )
    zf.write(path, path.name, compress_type=compress)


def export_layers_zip(
    dest_zip: Path,
    source_image: Image.Image,
    range_map: ColorRangeMap,
    *,
    source_path: Path | None = None,
    preset_id: str | None = None,
    on_status: Callable[[str], None] | None = None,
    output_size: tuple[int, int] | None = None,
    output_resample: str | None = None,
    output_dpi: float | None = None,
    crop_x: float = 0.0,
    crop_y: float = 0.0,
    crop_zoom: float = 1.0,
    tess_h: str = "off",
    tess_v: str = "off",
    tess_built: bool = False,
    tess_strength: float = 0.0,
    tess_mode: str = "tile",
    tess_tiles: int = 9000,
    tess_lloyd: int = 2,
    tess_normalize: bool = False,
    inpaint_boxes: Sequence[tuple[int, int, int, int]] | None = None,
    label: LabelSpec | None = None,
    label_specs: Sequence[LabelSpec] | None = None,
    overlay_layers: Sequence[tuple[str, Image.Image]] | None = None,
    document: Image.Image | None = None,
) -> Path:
    """Write ``dest_zip`` and return it. Full-res remap, then crop, lighting, tessellate, Scale, zip."""
    dest_zip = Path(dest_zip)
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    resample = output_resample or DEFAULT_RESAMPLE
    dpi = float(output_dpi) if output_dpi is not None and output_dpi > 0 else None
    grain = bool(range_map.texture_enabled)
    tess_on = bool(tess_built) or coerce_built(tess_strength)
    tess_flat = coerce_normalize_lighting(tess_normalize)

    _note("Labeling color ranges…", on_status)
    # One master only (exact XOR grain). The unused twin remap + tessellate
    # used to run on the full frame and was discarded.
    master_src, rgb, alpha, labels = labeled_composite_for_image(
        source_image, range_map, grain=grain
    )
    holes = tuple(inpaint_boxes) if inpaint_boxes else ()
    original_base = source_image
    if holes:
        _note("Inpainting labels…", on_status)
        original_base = inpaint_image(source_image, holes)
        master_src = inpaint_image(master_src, holes)
    master_src = apply_crop_lighting_tessellate(
        master_src,
        crop_x,
        crop_y,
        crop_zoom,
        tess_h,
        tess_v,
        tess_on,
        mode=tess_mode,
        tiles=tess_tiles,
        lloyd=tess_lloyd,
        normalize_lighting=tess_flat,
    )
    original_src = apply_crop_lighting_tessellate(
        original_base,
        crop_x,
        crop_y,
        crop_zoom,
        tess_h,
        tess_v,
        tess_on,
        mode=tess_mode,
        tiles=tess_tiles,
        lloyd=tess_lloyd,
        normalize_lighting=tess_flat,
    )
    sw, sh = source_image.size
    if not is_identity_crop(sw, sh, crop_x, crop_y, crop_zoom):
        labels = apply_crop_array(labels, crop_x, crop_y, crop_zoom, fill=-1)
        if alpha is not None:
            alpha = apply_crop_array(alpha, crop_x, crop_y, crop_zoom, fill=0)
    if not is_identity_tessellate(tess_h, tess_v, tess_on, mode=tess_mode):
        labels = tessellate_array(
            labels,
            tess_h,
            tess_v,
            tess_on,
            nearest=True,
            mode=tess_mode,
            tiles=tess_tiles,
            lloyd=tess_lloyd,
        )
        if alpha is not None:
            alpha = tessellate_array(
                alpha, tess_h, tess_v, tess_on, mode=tess_mode, tiles=tess_tiles, lloyd=tess_lloyd
            )
    master_img = scale_image(master_src, output_size, resample)
    original_img = scale_image(original_src, output_size, resample)
    width, height = master_img.size
    labels_s = _scale_labels(labels, (width, height))

    master_rgb = np.asarray(master_img.convert("RGB"), dtype=np.uint8)
    src_a: np.ndarray | None
    if master_img.mode == "RGBA":
        src_a = np.asarray(master_img.split()[-1], dtype=np.uint8)
    elif alpha is not None:
        a_img = scale_image(Image.fromarray(alpha, mode="L"), (width, height), resample)
        src_a = np.asarray(a_img, dtype=np.uint8)
    else:
        src_a = None

    def _mask_for(index: int) -> np.ndarray:
        mask = (labels_s == int(index)).astype(np.uint8) * 255
        if src_a is not None:
            mask = (mask.astype(np.uint16) * src_a.astype(np.uint16) // 255).astype(
                np.uint8
            )
        return mask

    with tempfile.TemporaryDirectory(prefix="wp_layers_") as tmp:
        dest_dir = Path(tmp)
        stack_pngs: list[tuple[str, str | None]] = []
        layer_records: list[dict] = []

        _note("Writing original…", on_status)
        _write_raster_pair(original_img, dest_dir, "00_original", dpi)
        stack_pngs.append(("00_original.png", None))
        layer_records.append(
            {
                "id": "00",
                "role": "original",
                "name": "Original",
                "parent": "",
                "path": "Image",
                "tif": "00_original.tif",
                "png": "00_original.png",
                "svg": None,
                "visible": True,
                "blend": "Normal",
            }
        )

        _note("Writing color plates…", on_status)
        for i, band in enumerate(range_map.ranges):
            if not band.visible:
                continue
            stem = _color_stem(band, i)
            plate = _rgba_plate(master_rgb, _mask_for(i))
            _write_raster_pair(plate, dest_dir, stem, dpi)
            hex_color = _hex(band.replacement_rgb)
            title = f"{band.name or f'Range {i + 1}'} {hex_color}"
            (dest_dir / f"{stem}.svg").write_text(
                _svg_document(
                    width,
                    height,
                    [(f"{stem}.png", None)],
                    dpi=dpi,
                    title=title,
                ),
                encoding="utf-8",
            )
            stack_pngs.append((f"{stem}.png", None))
            layer_records.append(
                {
                    "id": f"{i + 1:02d}",
                    "role": "color",
                    "name": band.name or f"Range {i + 1}",
                    "parent": "Image/Color ranges",
                    "path": f"Image/Color ranges/{band.name or f'Range {i + 1}'}",
                    "hex": hex_color,
                    "rgb": list(band.replacement_rgb),
                    "tif": f"{stem}.tif",
                    "png": f"{stem}.png",
                    "svg": f"{stem}.svg",
                    "visible": True,
                    "blend": "Normal",
                    "index": i,
                }
            )
            del plate

        texture_record: dict | None = None
        if grain:
            _note("Writing texture plate…", on_status)
            tex_rgb = scale_image(
                Image.fromarray(rgb, mode="RGB"), (width, height), resample
            )
            if src_a is not None:
                tex_rgba = tex_rgb.convert("RGBA")
                tex_rgba.putalpha(Image.fromarray(src_a, mode="L"))
                tex_out = tex_rgba
            else:
                tex_out = tex_rgb.convert("RGBA")
                opaque = np.full((height, width), 255, dtype=np.uint8)
                tex_out.putalpha(Image.fromarray(opaque, mode="L"))
            _write_raster_pair(tex_out, dest_dir, "texture", dpi)
            (dest_dir / "texture.svg").write_text(
                _svg_document(
                    width,
                    height,
                    [("texture.png", "luminosity")],
                    dpi=dpi,
                    title="Texture (Luminosity)",
                ),
                encoding="utf-8",
            )
            texture_record = {
                "id": "texture",
                "role": "texture",
                "name": "Texture / original luminosity",
                "tif": "texture.tif",
                "png": "texture.png",
                "svg": "texture.svg",
                "visible": True,
                "blend": "Luminosity",
                "note": (
                    "Original RGB (scaled). Color plates already contain grain "
                    "when the texture eye is on — hide this in a Normal stack. "
                    "Use Luminosity on top of solid hex fills to rebuild grain "
                    "from palette.json."
                ),
            }
            layer_records.append(texture_record)
            del tex_out

        label_record: dict | None = None
        specs: list[LabelSpec] = []
        if label_specs:
            specs = [spec for spec in label_specs if spec is not None and spec.is_set()]
        elif label is not None and label.is_set():
            specs = [label]
        if specs:
            _note("Writing label layer…", on_status)
            for i, spec in enumerate(specs):
                stem = "label" if i == 0 else f"label_{i + 1:02d}"
                _write_label_tif, label_png, label_svg = write_label_files(
                    dest_dir,
                    (width, height),
                    spec,
                    source_image.size,
                    crop_x=crop_x,
                    crop_y=crop_y,
                    crop_zoom=crop_zoom,
                    dpi=dpi,
                    stem=stem,
                )
                del _write_label_tif
                stack_pngs.append((label_png, None))
                rec = {
                    "id": "label" if i == 0 else stem,
                    "role": "label",
                    "name": spec.text.strip() or "Label",
                    "tif": f"{stem}.tif",
                    "png": label_png,
                    "svg": label_svg,
                    "visible": True,
                    "blend": "Normal",
                    "text": spec.text,
                    "font": str(getattr(spec, "font", "") or ""),
                    "font_size": int(spec.size),
                    "hex": _hex(spec.color),
                    "xy": [int(spec.x), int(spec.y)],
                    "note": (
                        "Editable text layer. SVG is live SVG text (Illustrator). "
                        "TIF / PNG are RGBA for Photoshop. Stack on top of composite."
                    ),
                }
                layer_records.append(rec)
                if i == 0:
                    label_record = rec

        if overlay_layers:
            _note("Writing overlay image layers…", on_status)
            for i, (name, overlay) in enumerate(overlay_layers):
                stem = f"overlay_{i + 1:02d}"
                plate = overlay.convert("RGBA")
                if plate.size != (width, height):
                    plate = plate.resize((width, height), Image.Resampling.BILINEAR)
                _write_raster_pair(plate, dest_dir, stem, dpi)
                stack_pngs.append((f"{stem}.png", None))
                layer_records.append(
                    {
                        "id": stem,
                        "role": "overlay",
                        "name": name or f"Overlay {i + 1}",
                        "tif": f"{stem}.tif",
                        "png": f"{stem}.png",
                        "svg": None,
                        "visible": True,
                        "blend": "Normal",
                    }
                )

        _note("Writing composite…", on_status)
        composite_out = document if document is not None else master_img
        if composite_out.size != (width, height):
            composite_out = composite_out.resize((width, height), Image.Resampling.BILINEAR)
        _write_raster_pair(composite_out, dest_dir, "composite", dpi)
        (dest_dir / "composite.svg").write_text(
            _svg_document(
                width,
                height,
                stack_pngs,
                dpi=dpi,
                title="Composite (Normal stack)",
            ),
            encoding="utf-8",
        )

        blend_note = (
            "Photoshop, bottom → top, Normal: 00_original, then visible color "
            "TIFs (alpha = range mask). Texture eye off: color RGB is the solid "
            "hex (after tone). Texture eye on: color RGB is grain (original L* + "
            "hex a*b*, after tone) so Normal stacking matches composite; "
            "texture.tif is the original luminosity plate (Luminosity blend if "
            "you rebuild from solid hexes instead). Hidden range eyes knock "
            "those pixels out of composite.tif (alpha 0), not original ink. "
            "SVG hrefs are sibling PNGs (Chrome); TIF is the "
            "Photoshop plate. Not an Image Trace."
        )
        palette = {
            "preset": preset_id,
            "split_method": range_map.split_method,
            "texture_strength": float(range_map.texture_strength),
            "texture_enabled": grain,
            "effective_texture_strength": float(effective_texture_strength(range_map)),
            "tone_darks": float(range_map.tone_darks),
            "tone_lights": float(range_map.tone_lights),
            "tone_brightness": float(range_map.tone_brightness),
            "tone_contrast": float(getattr(range_map, "tone_contrast", 0.0)),
            "tone_exposure": float(getattr(range_map, "tone_exposure", 0.0)),
            "tone_lights_reds": float(range_map.tone_lights_reds),
            "tone_lights_greens": float(range_map.tone_lights_greens),
            "tone_lights_blues": float(range_map.tone_lights_blues),
            "tone_temperature": float(getattr(range_map, "tone_temperature", 0.0)),
            "tone_tint": float(getattr(range_map, "tone_tint", 0.0)),
            "tone_saturation": float(getattr(range_map, "tone_saturation", 0.0)),
            "tone_balance_cyan": float(getattr(range_map, "tone_balance_cyan", 0.0)),
            "tone_balance_magenta": float(getattr(range_map, "tone_balance_magenta", 0.0)),
            "tone_balance_yellow": float(getattr(range_map, "tone_balance_yellow", 0.0)),
            "tone_lights_cyan": float(getattr(range_map, "tone_lights_cyan", 0.0)),
            "tone_lights_magenta": float(getattr(range_map, "tone_lights_magenta", 0.0)),
            "tone_lights_yellow": float(getattr(range_map, "tone_lights_yellow", 0.0)),
            "tone_darks_cyan": float(getattr(range_map, "tone_darks_cyan", 0.0)),
            "tone_darks_magenta": float(getattr(range_map, "tone_darks_magenta", 0.0)),
            "tone_darks_yellow": float(getattr(range_map, "tone_darks_yellow", 0.0)),
            "output_px": [width, height],
            "output_resample": resample if output_size is not None else None,
            "output_dpi": dpi,
            "source": source_path.name if source_path is not None else None,
            "note": (
                "Color closeness (default / V6-N): nearest cluster or palette hex "
                "in CIE Lab."
                if is_color_split(range_map.split_method)
                else "Luma split: non-overlapping Rec. 709 bands."
            ),
            "blend_note": blend_note,
            "svg_href": "relative PNG companions (Chrome / Inkscape / Illustrator)",
            "tif_plate": "same pixels as the PNG; use in Photoshop",
            "layers": layer_records,
            "texture": texture_record,
            "label": label_record,
            "composite": {
                "tif": "composite.tif",
                "png": "composite.png",
                "svg": "composite.svg",
            },
        }
        (dest_dir / "palette.json").write_text(
            json.dumps(palette, indent=2), encoding="utf-8"
        )

        readme_lines = [
            "Wallpaper layers zip",
            "",
            "Stack (bottom → top):",
            "  00_original.tif     source, scaled to Save as… size",
            "  Color plates nest in the Layers panel as Image → Color ranges →",
            "  Range N (mask + fill). This zip is a flat named stack in that",
            "  order under the image; Photoshop grouping is not written.",
            "  palette.json layers[].path records Image/Color ranges/Range N.",
            "  texture.tif         Texture eye on: original RGB. Hide in a",
            "                      Normal stack (grain is already in the color",
            "                      plates). Luminosity on solid hex fills to",
            "                      rebuild grain from palette.json.",
            "  composite.tif/png   flattened result (tone + scale; holes stay",
            "                      transparent — checker is preview only).",
            "  label.tif/png/svg   optional editable text layer (when set).",
            "                      SVG is live <text> for Illustrator;",
            "                      raster is RGBA for Photoshop. Stack on top.",
            "",
            "SVG: wrappers with <image href=\"sibling.png\"> (not a trace).",
            "composite.svg stacks 00 + color PNGs in order (Normal) = composite.",
            "",
            blend_note,
            "",
            "Palette hexes: palette.json",
        ]
        (dest_dir / "README.txt").write_text(
            "\n".join(readme_lines) + "\n", encoding="utf-8"
        )

        _note("Zipping…", on_status)
        with zipfile.ZipFile(dest_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(dest_dir.iterdir()):
                if path.is_file():
                    _zip_add_file(zf, path)

    _note(f"Layers zip written: {dest_zip}", on_status)
    return dest_zip
