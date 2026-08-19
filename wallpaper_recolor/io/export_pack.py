# -*- coding: utf-8 -*-
"""
wallpaper_recolor.io.export_pack
-----------------------------
Write a job folder for one range-separation (V6-N or custom):

  00_original.*              copy of the source file (untouched)
  01/02/03_*_mask.png        8-bit grayscale masks (non-overlapping)
  04_texture_detail.png      Rec. 709 luma of the original (reference, not Overlay)
  composite.*                grain / texture presentation (starring output;
                             scaled to the Scale panel size when set)
  exact_composite.*          flat palette master (production extra; same scale)
  tile_3x3.png               repeat preview (grain)
  seam_offset.png            Offset 50%/50% seam inspection (grain)
  room_mockup.png            simple interior, wallpaper at mockup_repeats
                             (cover_frac = floor-up wall height)
  palette.json / palette.txt hex names, cuts, notes
  icc_proof.png              soft-proof if an ICC was selected (else omitted)
  layers.psd                 small files only; else reconstruct from PNG+JSON

Photoshop: Solid Color fill per hex, paste the matching mask. No Adobe APIs.
Hex is screen RGB — keep the RGB master and the ICC proof as separate files.

Class references (code + name only):
- CAP3321C Data Wrangling
- CAP4631C Machine Learning
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
import json
import shutil

import numpy as np
from PIL import Image

from wallpaper_recolor.color.color_ranges import ColorRangeMap, is_color_split
from wallpaper_recolor.labels.layer import LabelSpec, write_label_files
from wallpaper_recolor.transform.tessellate import (
    apply_crop_lighting_tessellate,
    coerce_built,
    coerce_normalize_lighting,
)
from wallpaper_recolor.transform.inpaint import inpaint_image
from wallpaper_recolor.io.image_io import save_image
from wallpaper_recolor.color.layers import (
    composites_for_image,
    effective_texture_strength,
    masks_for_image,
    texture_detail_gray,
)
from wallpaper_recolor.color.presets import V6N_BANDS, V6N_ID
from wallpaper_recolor.preview.preview_tools import offset_seam, room_mockup, tile_repeat
from wallpaper_recolor.transform.scale import DEFAULT_RESAMPLE, scale_image
from wallpaper_recolor.io.proof import soft_proof
from wallpaper_recolor.io.psd_write import MAX_PSD_PIXELS, PsdLayer, write_psd


def _hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def _slug(name: str, index: int) -> str:
    raw = "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")
    while "__" in raw:
        raw = raw.replace("__", "_")
    return raw or f"band_{index:02d}"


def _mask_stem(band, index: int) -> str:
    """V6-N uses 01 light / 02 mid / 03 dark filenames; others use index order."""
    for name, _rgb, slug in V6N_BANDS:
        if band.name == name:
            return f"{slug}_mask"
    return f"{index:02d}_{_slug(band.name, index)}_mask"


def _note(msg: str, on_status: Callable[[str], None] | None) -> None:
    if on_status:
        on_status(msg)


def export_job_pack(
    dest_dir: Path,
    source_image: Image.Image,
    range_map: ColorRangeMap,
    *,
    source_path: Path | None = None,
    mockup_repeats: float = 4.0,
    mockup_cover_frac: float = 1.0,
    icc_path: Path | None = None,
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
) -> Path:
    """Build ``dest_dir`` and return it. Full-res masks; scaled master composites."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    resample = output_resample or DEFAULT_RESAMPLE
    dpi = float(output_dpi) if output_dpi is not None and output_dpi > 0 else None
    tess_on = bool(tess_built) or coerce_built(tess_strength)
    tess_flat = coerce_normalize_lighting(tess_normalize)

    _note("Labeling color ranges…", on_status)
    exact_img, tex_img, rgb, _alpha, labels = composites_for_image(source_image, range_map)
    holes = tuple(inpaint_boxes) if inpaint_boxes else ()
    if holes:
        exact_img = inpaint_image(exact_img, holes)
        tex_img = inpaint_image(tex_img, holes)
    masks = masks_for_image(labels, range_map)
    gray = texture_detail_gray(rgb)
    exact_img = apply_crop_lighting_tessellate(
        exact_img,
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
    tex_img = apply_crop_lighting_tessellate(
        tex_img,
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
    # PSD / masks stay source-sized; masters are scaled after remap+tone+crop+tessellate
    master_exact = scale_image(exact_img, output_size, resample)
    master_tex = scale_image(tex_img, output_size, resample)

    # --- original (untouched) ---
    if source_path is not None and source_path.is_file():
        shutil.copy2(source_path, dest_dir / f"00_original{source_path.suffix.lower()}")
    else:
        save_image(source_image, dest_dir / "00_original.png")

    _note("Writing masks and composites…", on_status)
    pixels = source_image.size[0] * source_image.size[1]
    png_kw: dict = {}
    if pixels <= 8_000_000:
        png_kw["optimize"] = True
    for i, (band, mask) in enumerate(zip(range_map.ranges, masks)):
        Image.fromarray(mask, mode="L").save(
            dest_dir / f"{_mask_stem(band, i)}.png",
            **png_kw,
        )
    Image.fromarray(gray, mode="L").save(dest_dir / "04_texture_detail.png", **png_kw)

    out_pixels = master_tex.size[0] * master_tex.size[1]
    ext = "tif" if out_pixels > 8_000_000 else "png"
    # Grain/texture is the starring file; exact stays as a production extra
    comp_path_tex = dest_dir / f"composite.{ext}"
    comp_path_exact = dest_dir / f"exact_composite.{ext}"
    save_image(master_tex, comp_path_tex, dpi=dpi)
    save_image(master_exact, comp_path_exact, dpi=dpi)

    _note("Tile, seam, mockup…", on_status)
    # Previews downscale internally — using the scaled master shows the repeat at output aspect
    tile_repeat(master_tex).save(dest_dir / "tile_3x3.png", optimize=True)
    offset_seam(master_tex).save(dest_dir / "seam_offset.png", optimize=True)
    room_mockup(
        master_tex,
        repeats_x=mockup_repeats,
        cover_frac=mockup_cover_frac,
    ).save(
        dest_dir / "room_mockup.png",
        optimize=True,
    )

    icc_note = "No ICC profile selected — RGB master is screen RGB, not a print proof."
    icc_written: str | None = None
    if icc_path is not None:
        _note("ICC soft-proof…", on_status)
        try:
            from wallpaper_recolor.preview.preview_tools import fit_max_edge

            proof = soft_proof(fit_max_edge(master_exact, 2048), icc_path)
            proof.save(dest_dir / "icc_proof.png", optimize=True)
            icc_written = Path(icc_path).name
            icc_note = f"Soft-proof of exact master through {icc_written}. RGB master is unchanged."
        except ValueError as exc:
            icc_note = str(exc)

    psd_note = (
        "PNG masks + palette hex: create Solid Color fills in Photoshop and paste masks. "
        "This writer stores raster layers with transparency (Layer Mask > From Transparency)."
    )
    if pixels <= MAX_PSD_PIXELS:
        _note("Writing layers.psd…", on_status)
        try:
            _write_pack_psd(
                dest_dir / "layers.psd",
                rgb,
                masks,
                range_map,
                np.asarray(tex_img.convert("RGB"), dtype=np.uint8),
            )
            psd_note = (
                "layers.psd: named raster layers in document order under the image "
                "(Color ranges / Range N). This writer has no PSD group section; "
                "the Layers panel nest (Image → Color ranges → Range N) is the "
                "source of truth. Hide 00 Original; hide 04 for the exact master."
            )
        except (OSError, ValueError) as exc:
            psd_note = f"PSD skipped ({exc}). Reconstruct from PNG masks + palette hex."
    else:
        psd_note = (
            f"PSD omitted ({pixels:,} px exceeds {MAX_PSD_PIXELS:,}; uint32 layer section). "
            "Use the PNG masks + palette hex as the layered master."
        )

    palette = {
        "preset": preset_id,
        "split_method": range_map.split_method,
        "texture_strength": float(range_map.texture_strength),
        "texture_enabled": bool(range_map.texture_enabled),
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
        "output_px": list(master_tex.size),
        "output_resample": resample if output_size is not None else None,
        "output_dpi": dpi,
        "crop_x": int(round(float(crop_x))),
        "crop_y": int(round(float(crop_y))),
        "crop_zoom": float(crop_zoom),
        "tess_h": str(tess_h),
        "tess_v": str(tess_v),
        "tess_built": bool(tess_on),
        "tess_strength": 1.0 if tess_on else 0.0,
        "tess_mode": str(tess_mode),
        "tess_tiles": int(tess_tiles),
        "tess_lloyd": int(tess_lloyd),
        "tess_normalize": bool(tess_flat),
        "note": (
            "Color closeness (default / V6-N): each pixel maps to the nearest "
            "cluster or palette hex in CIE Lab. Equal lightness / even pixel "
            "split are Rec. 709 luma bands. composite is Color/Luminosity "
            "grain (original L*, replacement a*b*). exact_composite is the "
            "flat palette."
            if is_color_split(range_map.split_method)
            else "Luma split: non-overlapping Rec. 709 bands (percentile thirds "
            "when split is equal_pixels). composite is Color/Luminosity grain "
            "(original L*, replacement a*b*). exact_composite is the flat palette."
        ),
        "icc": icc_written,
        "icc_note": icc_note,
        "psd_note": psd_note,
        "mockup_repeats_across_wall": mockup_repeats,
        "mockup_wall_cover_frac": float(mockup_cover_frac),
        "edges_luma": range_map.edges.tolist() if range_map.edges is not None else None,
        "centers_lab": range_map.centers.tolist() if range_map.centers is not None else None,
        "layers": [
            {
                "id": f"{i:02d}",
                "name": band.name or f"Range {i + 1}",
                "parent": "Image/Color ranges",
                "path": f"Image/Color ranges/{band.name or f'Range {i + 1}'}",
                "hex": _hex(band.replacement_rgb),
                "rgb": list(band.replacement_rgb),
                "luma_low": band.luma_low,
                "luma_high": band.luma_high,
                "weight": band.weight,
                "visible": bool(band.visible),
                "mask": f"{_mask_stem(band, i)}.png",
                "index_darkest_is_zero": i,
            }
            for i, band in enumerate(range_map.ranges)
        ],
    }
    (dest_dir / "palette.json").write_text(json.dumps(palette, indent=2), encoding="utf-8")

    lines = [
        palette["note"],
        f"Preset: {preset_id or '(custom)'}",
        f"Split: {range_map.split_method}",
        icc_note,
        psd_note,
        "",
        "Palette (screen RGB):",
    ]
    for layer in palette["layers"]:
        lines.append(
            f"  {layer['name']:28s}  {layer['hex']}  mask {layer['mask']}"
        )
    if label is not None and label.is_set():
        write_label_files(
            dest_dir,
            master_tex.size,
            label,
            source_image.size,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_zoom=crop_zoom,
            dpi=dpi,
        )
        lines.extend(
            [
                "",
                "Label layer: label.svg (editable SVG text in Illustrator),",
                "label.tif / label.png (RGBA for Photoshop). Stack on top of",
                "the composite. The layers zip uses the same files.",
            ]
        )
        (dest_dir / "README.txt").write_text(
            "\n".join(
                [
                    "Job pack",
                    "",
                    "Label layer: label.svg is live SVG text (Illustrator);",
                    "label.tif / label.png are RGBA for Photoshop. Place on",
                    "top of composite after removing wallpaper text.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    (dest_dir / "palette.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _note(f"Job pack written: {dest_dir}", on_status)
    return dest_dir


def _write_pack_psd(
    path: Path,
    rgb: np.ndarray,
    masks: list[np.ndarray],
    range_map: ColorRangeMap,
    composite: np.ndarray,
) -> None:
    h, w = rgb.shape[:2]
    opaque = np.full((h, w), 255, dtype=np.uint8)
    layers = [
        PsdLayer("00 Original", rgb, opaque, visible=False),
    ]
    # Bottom → top: range fills, then original in Luminosity (Color blend analog)
    for i, (band, mask) in enumerate(zip(range_map.ranges, masks)):
        fill = np.empty_like(rgb)
        fill[:] = np.array(band.replacement_rgb, dtype=np.uint8)
        title = f"Color ranges / {band.name or f'Range {i + 1}'}"
        layers.append(PsdLayer(title[:251], fill, mask, visible=bool(band.visible)))
    layers.append(
        PsdLayer(
            "04 Original Luminosity",
            rgb,
            opaque,
            opacity=int(round(effective_texture_strength(range_map) * 255)),
            blend=b"lumi",
            visible=True,
        )
    )
    write_psd(path, layers, composite)
