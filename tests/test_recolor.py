# -*- coding: utf-8 -*-
"""
tests.test_recolor
------------------
Regression checks for Wallpaper Recolor: Lab clusters, crop/frame, layout,
preview Fit (contain), View Move / Grab Move, and job-pack I/O.

Tk tests withdraw the root so they do not flash a maximized window. Layout
tests deiconify only when they must read sash / title-bar geometry.

Class references (code + name only):
- CAP3321C Data Wrangling
- CAP4631C Machine Learning
"""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wallpaper_recolor.color.color_math import lab_to_rgb_array, rgb_to_lab_array, rgb_tuple_to_lab
from wallpaper_recolor.color.color_ranges import (
    ASSIGN_KMEANS,
    ASSIGN_KMEANS_LABEL,
    ASSIGN_PALETTE,
    ASSIGN_PALETTE_LABEL,
    RANGE_BY_COLOR_LABEL,
    RANGE_BY_LAB_A_LABEL,
    RANGE_BY_LAB_B_LABEL,
    RANGE_BY_LAB_L_LABEL,
    RANGE_BY_LUMA_LABEL,
    SPLIT_COLOR_CLOSENESS,
    SPLIT_EQUAL_LIGHTNESS,
    SPLIT_EQUAL_LIGHTNESS_LABEL,
    SPLIT_EQUAL_PIXELS,
    SPLIT_EQUAL_PIXELS_LABEL,
    SPLIT_LAB_A_EQUAL,
    SPLIT_LAB_A_PIXELS,
    SPLIT_LAB_L_EQUAL,
    SPLIT_LAB_L_PIXELS,
    build_range_map,
    drop_color_range,
    insert_color_range,
    is_color_split,
    luma_channel,
    snapshot_assignment,
    set_range_weight,
    choose_kmeans_k,
    AUTO_K_MAX,
)
from wallpaper_recolor.transform.crop import apply_crop, crop_box, crop_array, is_identity_crop
from wallpaper_recolor.io.image_io import (
    PNG_COMPRESS_LEVEL_FAST,
    TIFF_COMPRESSION_FAST,
    save_image,
)
from wallpaper_recolor.color.layers import (
    TEXTURE_DEFAULT_STRENGTH,
    exact_rgb,
    presentation_rgb,
)
from wallpaper_recolor.color.presets import V6N_DARK, V6N_LIGHT, V6N_MID, v6n_preset


def _widget_under(widget, ancestor) -> bool:
    """True when ``widget`` is ``ancestor`` or packed/gridded under it."""
    w = widget
    seen: set[int] = set()
    while w is not None and id(w) not in seen:
        if w is ancestor:
            return True
        seen.add(id(w))
        nxt = None
        try:
            info = w.pack_info()
            geom = info.get("in")
            if geom is not None:
                nxt = w.nametowidget(str(geom)) if isinstance(geom, str) else geom
        except Exception:
            nxt = None
        if nxt is None:
            nxt = getattr(w, "master", None)
        w = nxt
    return False


def _drain_busy(app, root, timeout: float = 8.0) -> None:
    """Pump Tk until a ``_run_background`` job (Build / save) finishes."""
    import time

    deadline = time.monotonic() + timeout
    while getattr(app, "_busy", False) and time.monotonic() < deadline:
        root.update()
        time.sleep(0.01)
    root.update()
    if getattr(app, "_busy", False):
        raise AssertionError("background job did not finish")


def _lab_dist2(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.dot(d, d))


class TestCrop(unittest.TestCase):
    """Position & Zoom frame: center-origin X/Y, zoom about center, empty = alpha."""
    def test_identity_crop_box(self) -> None:
        self.assertEqual(crop_box(100, 80, 0, 0, 1), (0, 0, 100, 80))
        self.assertTrue(is_identity_crop(100, 80, 0, 0, 1))

    def test_zoom_2_half_window(self) -> None:
        # (0,0) zoom 2: frame shows the source center half, scaled about the center
        self.assertEqual(crop_box(100, 80, 0, 0, 2), (25, 20, 75, 60))
        self.assertFalse(is_identity_crop(100, 80, 0, 0, 2))

    def test_origin_not_clamped_inside(self) -> None:
        left, top, right, bottom = crop_box(100, 80, 999, 0, 1)
        self.assertLess(left, 0)
        self.assertLess(right, 100)

    def test_apply_crop_and_array(self) -> None:
        im = Image.new("RGB", (100, 50), (10, 20, 30))
        im.putpixel((0, 0), (255, 0, 0))
        im.putpixel((50, 25), (0, 255, 0))
        out = apply_crop(im, 0, 0, 2.0)
        self.assertEqual(out.size, (100, 50))
        self.assertNotEqual(out.convert("RGB").getpixel((0, 0))[:3], (255, 0, 0))
        arr = np.arange(100 * 50, dtype=np.uint16).reshape(50, 100)
        sliced = crop_array(arr, 0, 0, 2.0)
        self.assertEqual(sliced.shape, (25, 50))
        np.testing.assert_array_equal(sliced, arr[:25, :50])
        work = im.resize((50, 25), Image.Resampling.NEAREST)
        mapped = apply_crop(work, 0, 0, 1.0, src_size=(100, 50))
        self.assertEqual(mapped.size, (50, 25))

    def test_zoom_1_offset_leaves_transparent_margin(self) -> None:
        from wallpaper_recolor.transform.crop import apply_crop_array

        im = Image.new("RGB", (40, 20), (10, 80, 30))
        im.putpixel((0, 0), (255, 0, 0))
        centered = apply_crop(im, 0, 0, 1.0)
        self.assertIs(centered, im)
        shifted = apply_crop(im, 8, 0, 1.0)
        self.assertEqual(shifted.size, (40, 20))
        self.assertEqual(shifted.mode, "RGBA")
        self.assertEqual(shifted.getpixel((0, 10))[3], 0)
        self.assertGreater(shifted.getpixel((8, 0))[3], 0)
        self.assertEqual(shifted.getpixel((8, 0))[:3], (255, 0, 0))
        labels = np.arange(40 * 20, dtype=np.int32).reshape(20, 40)
        placed = apply_crop_array(labels, 8, 0, 1.0, fill=-1)
        self.assertEqual(placed.shape, (20, 40))
        self.assertEqual(int(placed[0, 0]), -1)


class TestTessellate(unittest.TestCase):
    """Wrap / mosaic Build — independent of preview view-zoom."""
    def test_identity_is_noop(self) -> None:
        from wallpaper_recolor.transform.tessellate import apply_tessellate, is_identity_tessellate

        im = Image.fromarray(np.arange(24, dtype=np.uint8).reshape(4, 6), mode="L")
        self.assertTrue(is_identity_tessellate("off", "off", True))
        self.assertTrue(is_identity_tessellate("left", "top", False))
        self.assertTrue(is_identity_tessellate("left", "top", 0.0))
        self.assertFalse(is_identity_tessellate("left", "top", True))
        self.assertTrue(is_identity_tessellate("off", "off", False, normalize_lighting=True))
        self.assertTrue(is_identity_tessellate("off", "off", True, normalize_lighting=True))
        out = apply_tessellate(im, "left", "top", False)
        self.assertIs(out, im)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(im))
        skipped = apply_tessellate(im, "off", "off", True)
        self.assertIs(skipped, im)
        off_until_build = apply_tessellate(im, "off", "off", False)
        self.assertIs(off_until_build, im)

    def test_already_tiled_is_bit_identical(self) -> None:
        """np.tile / wrap-identical edges: Build is a no-op (MSE cap ~0)."""
        from wallpaper_recolor.transform.tessellate import (
            apply_tessellate,
            edges_already_match,
            plan_tessellate_crop,
        )

        rng = np.random.default_rng(11)
        motif = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        motif[:, -1] = motif[:, 0]
        motif[-1] = motif[0]
        tiled = np.tile(motif, (4, 5, 1))
        self.assertTrue(edges_already_match(tiled, "left", "top"))
        self.assertEqual(plan_tessellate_crop(tiled, "left", "top"), (0, 0, 1.0))
        im = Image.fromarray(tiled, mode="RGB")
        out = apply_tessellate(im, "left", "top", True)
        self.assertIs(out, im)
        np.testing.assert_array_equal(np.asarray(out), tiled)
        mse = float(np.mean((np.asarray(out, dtype=np.float32) - tiled.astype(np.float32)) ** 2))
        self.assertLessEqual(mse, 0.25)
        self.assertTrue(edges_already_match(tiled, "right", "bottom"))
        im2 = Image.fromarray(tiled, mode="RGB")
        out2 = apply_tessellate(im2, "right", "bottom", True)
        self.assertIs(out2, im2)

    def test_tessellate_does_not_flatten_lighting(self) -> None:
        from wallpaper_recolor.transform.tessellate import apply_tessellate

        h, w = 48, 48
        yy, xx = np.mgrid[0:h, 0:w]
        luma = np.clip(80.0 + 140.0 * (xx / max(w - 1, 1)), 0, 255)
        rgb = np.stack((luma, luma, luma), axis=-1).astype(np.uint8)
        im = Image.fromarray(rgb, mode="RGB")
        out = apply_tessellate(im, "off", "off", True)
        self.assertIs(out, im)

    def test_output_size_equals_input(self) -> None:
        from wallpaper_recolor.transform.tessellate import apply_tessellate

        im = Image.fromarray(
            np.random.default_rng(0).integers(0, 256, (32, 48, 3), dtype=np.uint8),
            mode="RGB",
        )
        out = apply_tessellate(im, "left", "top", True)
        self.assertEqual(out.size, im.size)
        self.assertEqual(out.mode, im.mode)

    def test_hilbert_xy_to_d_order1(self) -> None:
        from wallpaper_recolor.transform.tessellate import hilbert_xy_to_d

        x = np.array([[0, 1], [0, 1]], dtype=np.int64)
        y = np.array([[0, 0], [1, 1]], dtype=np.int64)
        d = hilbert_xy_to_d(x, y, 1)
        np.testing.assert_array_equal(d, np.array([[0, 3], [1, 2]]))

    def test_crinkly_front_is_not_a_linear_wipe(self) -> None:
        from wallpaper_recolor.transform.tessellate import _crinkly_front_alpha

        alpha = _crinkly_front_alpha(16, 16, 0, 16, from_low=True, axis=1)
        np.testing.assert_allclose(alpha[:, 0], 1.0, atol=1e-5)
        np.testing.assert_allclose(alpha[:, -1], 0.0, atol=1e-5)
        self.assertGreater(float(np.std(alpha[:, 6])), 1e-4)

    def test_built_wrap_corners_and_edges(self) -> None:
        from wallpaper_recolor.transform.tessellate import (
            MODE_TESSELLATE,
            apply_tessellate,
            tessellate_array,
        )

        rng = np.random.default_rng(1)
        rgb = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
        im = Image.fromarray(rgb, mode="RGB")
        out = apply_tessellate(im, "right", "bottom", True, mode=MODE_TESSELLATE)
        self.assertEqual(out.size, (50, 40))
        arr = np.asarray(out)
        np.testing.assert_allclose(
            arr[:, 0].astype(np.float32), arr[:, -1].astype(np.float32), atol=2.0
        )
        np.testing.assert_allclose(
            arr[0].astype(np.float32), arr[-1].astype(np.float32), atol=2.0
        )
        labels = np.arange(40 * 50, dtype=np.int32).reshape(40, 50)
        blended = tessellate_array(
            labels, "left", "off", True, nearest=True, mode=MODE_TESSELLATE
        )
        self.assertEqual(blended.shape, labels.shape)
        self.assertEqual(blended.dtype, labels.dtype)

    def test_left_matches_opposite_edge_when_not_self_similar(self) -> None:
        from wallpaper_recolor.transform.tessellate import MODE_TESSELLATE, apply_tessellate

        rgb = np.zeros((16, 24, 3), dtype=np.uint8)
        rgb[:, :12] = (200, 10, 10)
        rgb[:, 12:] = (10, 10, 200)
        im = Image.fromarray(rgb, mode="RGB")
        out = np.asarray(apply_tessellate(im, "left", "off", True, mode=MODE_TESSELLATE))
        np.testing.assert_allclose(
            out[:, 0].astype(np.float32), out[:, -1].astype(np.float32), atol=12.0
        )

    def test_mesh_mode_still_warps_labels(self) -> None:
        from wallpaper_recolor.transform.tessellate import MODE_MESH, apply_tessellate, tessellate_array

        rng = np.random.default_rng(2)
        rgb = rng.integers(0, 256, (24, 30, 3), dtype=np.uint8)
        im = Image.fromarray(rgb, mode="RGB")
        out = apply_tessellate(im, "left", "off", True, mode=MODE_MESH)
        self.assertEqual(out.size, im.size)
        arr = np.asarray(out)
        np.testing.assert_allclose(
            arr[:, 0].astype(np.float32), arr[:, -1].astype(np.float32), atol=2.0
        )
        labels = np.arange(24 * 30, dtype=np.int32).reshape(24, 30)
        warped = tessellate_array(labels, "left", "off", True, nearest=True, mode=MODE_MESH)
        self.assertEqual(warped.shape, labels.shape)
        self.assertEqual(warped.dtype, labels.dtype)
        self.assertFalse(np.array_equal(warped, labels))

    def test_voronoi_keeps_size_on_landscape_and_portrait(self) -> None:
        from wallpaper_recolor.transform.tessellate import (
            MODE_VORONOI,
            apply_tessellate,
            is_identity_tessellate,
        )

        self.assertTrue(is_identity_tessellate("off", "off", False, mode=MODE_VORONOI))
        self.assertFalse(is_identity_tessellate("off", "off", True, mode=MODE_VORONOI))
        rng = np.random.default_rng(3)
        for shape in ((32, 48, 3), (48, 32, 3), (40, 40, 3)):
            rgb = rng.integers(0, 256, shape, dtype=np.uint8)
            im = Image.fromarray(rgb, mode="RGB")
            out = apply_tessellate(
                im, "off", "off", True, mode=MODE_VORONOI, tiles=32, lloyd=0
            )
            self.assertEqual(out.size, im.size)
            self.assertEqual(out.mode, im.mode)

    def test_voronoi_build_wraps_when_sides_set(self) -> None:
        """Detail mosaic Build still Hilbert-wraps so opposite edges pin."""
        from wallpaper_recolor.transform.tessellate import MODE_VORONOI, apply_tessellate

        rgb = np.zeros((24, 32, 3), dtype=np.uint8)
        rgb[:, :16] = (200, 10, 10)
        rgb[:, 16:] = (10, 10, 200)
        im = Image.fromarray(rgb, mode="RGB")
        out = np.asarray(
            apply_tessellate(im, "left", "top", True, mode=MODE_VORONOI, tiles=32, lloyd=0)
        )
        self.assertEqual(out.shape, rgb.shape)
        np.testing.assert_allclose(
            out[:, 0].astype(np.float32), out[:, -1].astype(np.float32), atol=12.0
        )
        np.testing.assert_allclose(
            out[0].astype(np.float32), out[-1].astype(np.float32), atol=12.0
        )
        self.assertFalse(np.array_equal(out, rgb))

    def test_normalize_lighting_keeps_size_and_flattens_pillow(self) -> None:
        from wallpaper_recolor.transform.tessellate import (
            apply_normalize_lighting,
            apply_tessellate,
        )

        h, w = 96, 96
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
        rng = np.random.default_rng(4)
        grain = rng.normal(0.0, 5.0, (h, w))
        luma = np.clip(175.0 * (1.0 - 0.5 * np.clip(r, 0.0, 1.25)) + grain, 40.0, 255.0)
        rgb = np.stack((luma, luma * 0.93, luma * 0.80), axis=-1).astype(np.uint8)
        im = Image.fromarray(rgb, mode="RGB")
        unchanged = apply_tessellate(im, "off", "off", True)
        self.assertIs(unchanged, im)
        out = apply_normalize_lighting(im)
        self.assertEqual(out.size, im.size)
        self.assertEqual(out.mode, im.mode)
        src = np.asarray(im, dtype=np.float32)
        dst = np.asarray(out, dtype=np.float32)
        src_y = 0.2126 * src[..., 0] + 0.7152 * src[..., 1] + 0.0722 * src[..., 2]
        dst_y = 0.2126 * dst[..., 0] + 0.7152 * dst[..., 1] + 0.0722 * dst[..., 2]
        bw = 8
        src_center = float(np.mean(src_y[h // 2 - bw : h // 2 + bw, w // 2 - bw : w // 2 + bw]))
        dst_center = float(np.mean(dst_y[h // 2 - bw : h // 2 + bw, w // 2 - bw : w // 2 + bw]))
        src_corners = float(
            np.mean(
                (
                    np.mean(src_y[:bw, :bw]),
                    np.mean(src_y[:bw, -bw:]),
                    np.mean(src_y[-bw:, :bw]),
                    np.mean(src_y[-bw:, -bw:]),
                )
            )
        )
        dst_corners = float(
            np.mean(
                (
                    np.mean(dst_y[:bw, :bw]),
                    np.mean(dst_y[:bw, -bw:]),
                    np.mean(dst_y[-bw:, :bw]),
                    np.mean(dst_y[-bw:, -bw:]),
                )
            )
        )
        self.assertLess(abs(dst_center - dst_corners), abs(src_center - src_corners))
        src_lr = abs(float(np.mean(src_y[:, :bw])) - float(np.mean(src_y[:, -bw:])))
        dst_lr = abs(float(np.mean(dst_y[:, :bw])) - float(np.mean(dst_y[:, -bw:])))
        # Symmetric pillow: left≈right already; keep the gap from growing.
        self.assertLessEqual(dst_lr, src_lr + 2.0)

    def test_normalize_lighting_reduces_side_gradient(self) -> None:
        from wallpaper_recolor.transform.tessellate import apply_normalize_lighting

        h, w = 80, 96
        xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
        illum = 0.45 + 0.55 * xx[None, :]
        grain = np.random.default_rng(5).normal(0.0, 4.0, (h, w))
        luma = np.clip(160.0 * illum + grain, 20.0, 255.0)
        rgb = np.stack((luma, luma * 0.9, luma * 0.75), axis=-1).astype(np.uint8)
        im = Image.fromarray(rgb, mode="RGB")
        out = np.asarray(apply_normalize_lighting(im))
        self.assertEqual(out.shape, rgb.shape)
        src_y = rgb[:, :, 1].astype(np.float32)
        dst_y = out[:, :, 1].astype(np.float32)
        bw = 6
        src_gap = abs(float(np.mean(src_y[:, :bw])) - float(np.mean(src_y[:, -bw:])))
        dst_gap = abs(float(np.mean(dst_y[:, :bw])) - float(np.mean(dst_y[:, -bw:])))
        self.assertLess(dst_gap, 0.25 * src_gap)
        wrap_jump = abs(float(np.mean(dst_y[:, 0])) - float(np.mean(dst_y[:, -1])))
        src_jump = abs(float(np.mean(src_y[:, 0])) - float(np.mean(src_y[:, -1])))
        self.assertLess(wrap_jump, 0.25 * src_jump)

    def test_normalize_lighting_flattens_vertical_gradient(self) -> None:
        from wallpaper_recolor.transform.tessellate import (
            apply_normalize_lighting,
            apply_tessellate,
        )

        h, w = 96, 80
        yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        illum = 0.40 + 0.60 * yy  # dark top, bright bottom
        grain = np.random.default_rng(6).normal(0.0, 4.0, (h, w))
        luma = np.clip(165.0 * illum + grain, 20.0, 255.0)
        rgb = np.stack((luma, luma * 0.92, luma * 0.78), axis=-1).astype(np.uint8)
        im = Image.fromarray(rgb, mode="RGB")
        unchanged = apply_tessellate(im, "off", "off", True)
        self.assertIs(unchanged, im)
        out = apply_normalize_lighting(im)
        self.assertEqual(out.size, im.size)
        self.assertEqual(out.mode, im.mode)
        src = np.asarray(im, dtype=np.float32)
        dst = np.asarray(out, dtype=np.float32)
        src_y = 0.2126 * src[..., 0] + 0.7152 * src[..., 1] + 0.0722 * src[..., 2]
        dst_y = 0.2126 * dst[..., 0] + 0.7152 * dst[..., 1] + 0.0722 * dst[..., 2]
        bw = 6
        src_gap = abs(float(np.mean(src_y[:bw])) - float(np.mean(src_y[-bw:])))
        dst_gap = abs(float(np.mean(dst_y[:bw])) - float(np.mean(dst_y[-bw:])))
        self.assertLess(dst_gap, 0.25 * src_gap)
        wrap_jump = abs(float(np.mean(dst_y[0])) - float(np.mean(dst_y[-1])))
        src_jump = abs(float(np.mean(src_y[0])) - float(np.mean(src_y[-1])))
        self.assertLess(wrap_jump, 0.25 * src_jump)
        src_grain = float(np.std(src_y - src_y.mean(axis=1, keepdims=True)))
        dst_grain = float(np.std(dst_y - dst_y.mean(axis=1, keepdims=True)))
        self.assertGreater(dst_grain, 0.35 * src_grain)

    def test_normalize_then_wrap_pins_both_axes(self) -> None:
        from wallpaper_recolor.transform.tessellate import (
            apply_normalize_lighting,
            apply_tessellate,
        )

        h, w = 64, 72
        yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
        illum = 0.35 + 0.40 * yy + 0.25 * xx
        grain = np.random.default_rng(7).normal(0.0, 3.5, (h, w))
        luma = np.clip(170.0 * illum + grain, 25.0, 255.0)
        rgb = np.stack((luma, luma * 0.9, luma * 0.8), axis=-1).astype(np.uint8)
        im = Image.fromarray(rgb, mode="RGB")
        out = np.asarray(apply_tessellate(apply_normalize_lighting(im), "left", "top", True))
        self.assertEqual(out.shape, rgb.shape)
        np.testing.assert_allclose(
            out[:, 0].astype(np.float32), out[:, -1].astype(np.float32), atol=2.0
        )
        np.testing.assert_allclose(
            out[0].astype(np.float32), out[-1].astype(np.float32), atol=2.0
        )
        dst_y = out[:, :, 1].astype(np.float32)
        src_y = rgb[:, :, 1].astype(np.float32)
        bw = 5
        dst_v = abs(float(np.mean(dst_y[:bw])) - float(np.mean(dst_y[-bw:])))
        src_v = abs(float(np.mean(src_y[:bw])) - float(np.mean(src_y[-bw:])))
        dst_h = abs(float(np.mean(dst_y[:, :bw])) - float(np.mean(dst_y[:, -bw:])))
        src_h = abs(float(np.mean(src_y[:, :bw])) - float(np.mean(src_y[:, -bw:])))
        self.assertLess(dst_v, 0.35 * src_v)
        self.assertLess(dst_h, 0.35 * src_h)

    def test_normalize_independent_of_tessellate(self) -> None:
        from wallpaper_recolor.color.tone import apply_tone_rgb
        from wallpaper_recolor.transform.tessellate import (
            apply_crop_lighting_tessellate,
            apply_normalize_lighting,
            apply_tessellate,
            estimate_normalize_tone,
        )

        h, w = 64, 80
        xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
        luma = np.clip(50.0 + 160.0 * xx[None, :], 0, 255)
        rgb = np.stack((luma, luma * 0.9, luma * 0.8), axis=-1).astype(np.uint8)
        im = Image.fromarray(rgb, mode="RGB")
        flat = apply_normalize_lighting(im)
        via_pipeline = apply_crop_lighting_tessellate(
            im, 0, 0, 1.0, "off", "off", False, normalize_lighting=True
        )
        np.testing.assert_array_equal(np.asarray(via_pipeline), rgb)
        wrapped_only = apply_tessellate(im, "off", "off", True)
        self.assertIs(wrapped_only, im)
        self.assertFalse(np.array_equal(np.asarray(flat), rgb))
        darks, lights = estimate_normalize_tone(im)
        # Flatten lifts shadows → negative Darks (left = pull darks back).
        self.assertLess(darks, -0.02)
        self.assertLess(lights, -0.02)
        graded = apply_tone_rgb(rgb, darks, lights, 0.0)
        src_y = rgb[:, :, 1].astype(np.float32)
        dst_y = graded[:, :, 1].astype(np.float32)
        bw = 6
        src_gap = abs(float(np.mean(src_y[:, :bw])) - float(np.mean(src_y[:, -bw:])))
        dst_gap = abs(float(np.mean(dst_y[:, :bw])) - float(np.mean(dst_y[:, -bw:])))
        self.assertLess(dst_gap, src_gap)

    def test_preview_has_no_dashed_tessellate_overlay(self) -> None:
        """Composite Original/Result must not paint a dashed Hilbert/tile guide."""
        import wallpaper_recolor.transform as transform_pkg
        from wallpaper_recolor.transform import tessellate as tess_mod
        from wallpaper_recolor.ui import app as ui_mod

        self.assertFalse(hasattr(tess_mod, "draw_tessellate_guides"))
        self.assertFalse(hasattr(transform_pkg, "draw_tessellate_guides"))
        tess_src = inspect.getsource(tess_mod)
        self.assertNotIn("_draw_dashed_polyline", tess_src)
        self.assertNotIn("_OVERLAY_DASH", tess_src)
        self.assertNotIn("draw_tessellate_guides", tess_src)
        pils_src = inspect.getsource(ui_mod.WallpaperRecolorApp._preview_pils)
        self.assertNotIn("draw_tessellate_guides", pils_src)
        self.assertIn("apply_tessellate", pils_src)

    def test_tile_mode_is_default_and_periodic_is_identity(self) -> None:
        from wallpaper_recolor.transform.tessellate import (
            MODE_DEFAULT,
            MODE_LABELS,
            MODE_TILE,
            apply_tessellate,
            estimate_axis_period,
            image_already_periodic,
            normalize_tess_mode,
            plan_tessellate_crop,
            tess_mode_label,
        )

        self.assertEqual(MODE_DEFAULT, MODE_TILE)
        self.assertEqual(normalize_tess_mode(None), MODE_TILE)
        self.assertEqual(normalize_tess_mode("Tessellation"), "tessellate")
        self.assertEqual(tess_mode_label(MODE_TILE), "Tile (Repeating Design)")
        self.assertEqual(
            list(MODE_LABELS),
            [
                "Tile (Repeating Design)",
                "Tessellation",
                "Mesh",
                "Detail mosaic",
            ],
        )
        h, w, period = 64, 80, 16
        yy, xx = np.mgrid[0:h, 0:w]
        cell = ((xx % period) < 6) & ((yy % period) < 6)
        luma = np.where(cell, 220.0, 40.0)
        rgb = np.stack((luma, luma * 0.92, luma * 0.8), axis=-1).astype(np.uint8)
        self.assertTrue(image_already_periodic(rgb, "left", "top"))
        self.assertEqual(plan_tessellate_crop(rgb, "left", "top"), (0, 0, 1.0))
        px = estimate_axis_period(rgb, 1)
        py = estimate_axis_period(rgb, 0)
        self.assertLessEqual(abs(px - period), 1)
        self.assertLessEqual(abs(py - period), 1)
        im = Image.fromarray(rgb, mode="RGB")
        out = apply_tessellate(im, "left", "top", True, mode=MODE_TILE)
        self.assertIs(out, im)
        np.testing.assert_array_equal(np.asarray(out), rgb)

    def test_tile_mode_wraps_off_period_repeat(self) -> None:
        """Cropped-off-period grid: Tile Build fills leftover so wrap MAE is small."""
        from wallpaper_recolor.transform.tessellate import MODE_TILE, apply_tessellate

        period, radius = 16, 3
        full_h, full_w = 64, 64
        yy, xx = np.mgrid[0:full_h, 0:full_w]
        rgb = np.full((full_h, full_w, 3), 36, dtype=np.uint8)
        for y0 in range(period // 2, full_h, period):
            for x0 in range(period // 2, full_w, period):
                mask = (yy - y0) ** 2 + (xx - x0) ** 2 <= radius ** 2
                rgb[mask] = (230, 220, 210)
        cropped = rgb[3:61, 5:60]
        im = Image.fromarray(cropped, mode="RGB")
        out = np.asarray(apply_tessellate(im, "left", "top", True, mode=MODE_TILE))
        self.assertEqual(out.shape, cropped.shape)
        mae_h = float(
            np.mean(np.abs(out[:, 0].astype(np.float32) - out[:, -1].astype(np.float32)))
        )
        mae_v = float(
            np.mean(np.abs(out[0].astype(np.float32) - out[-1].astype(np.float32)))
        )
        self.assertLess(mae_h, 4.0)
        self.assertLess(mae_v, 4.0)
        montage = np.concatenate([out, out], axis=1)
        seam = float(
            np.mean(
                np.abs(
                    montage[:, out.shape[1] - 1].astype(np.float32)
                    - montage[:, out.shape[1]].astype(np.float32)
                )
            )
        )
        self.assertLess(seam, 4.0)
        self.assertFalse(np.array_equal(out, cropped))


class TestColorCloseness(unittest.TestCase):
    """Lab k-means vs snap-to-palette; match-from / change-to stay on insert."""
    def test_import_ui(self) -> None:
        from wallpaper_recolor.color.presets import range_by_label_for
        from wallpaper_recolor.ui import run
        import wallpaper_recolor.ui as ui

        self.assertTrue(callable(run))
        self.assertTrue(hasattr(ui, "WallpaperRecolorApp"))
        self.assertTrue(hasattr(ui, "write_layers_zip"))
        self.assertEqual(
            list(ui.RANGE_BY_LABELS),
            [
                RANGE_BY_COLOR_LABEL,
                RANGE_BY_LUMA_LABEL,
                RANGE_BY_LAB_L_LABEL,
                RANGE_BY_LAB_A_LABEL,
                RANGE_BY_LAB_B_LABEL,
            ],
        )
        self.assertEqual(list(ui.ASSIGN_LABELS), [ASSIGN_KMEANS_LABEL, ASSIGN_PALETTE_LABEL])
        self.assertEqual(ui.LUMA_SPLIT_LABELS[ui.SPLIT_EQUAL_PIXELS_LABEL], SPLIT_EQUAL_PIXELS)
        self.assertEqual(range_by_label_for(SPLIT_COLOR_CLOSENESS), RANGE_BY_COLOR_LABEL)
        self.assertEqual(range_by_label_for(SPLIT_EQUAL_PIXELS), RANGE_BY_LUMA_LABEL)
        self.assertEqual(range_by_label_for(SPLIT_EQUAL_LIGHTNESS), RANGE_BY_LUMA_LABEL)
        self.assertEqual(range_by_label_for(SPLIT_LAB_L_EQUAL), RANGE_BY_LAB_L_LABEL)
        self.assertEqual(range_by_label_for(SPLIT_LAB_A_EQUAL), RANGE_BY_LAB_A_LABEL)

    def test_range_by_dropdown_rebuilds_split_method(self) -> None:
        """Range by: Color closeness → Lab clusters; luma / L* / a* / b* → bins."""
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod

        root = tk.Tk()
        root.withdraw()
        methods: list[str] = []
        real = ui_mod.build_range_map

        def spy(image, n, method, palette_rgb=None, **kwargs):
            methods.append(method)
            return real(image, n, method, palette_rgb=palette_rgb, **kwargs)

        try:
            app = ui_mod.WallpaperRecolorApp(root)
            self.assertEqual(list(app.range_by_combo.cget("values")), list(ui_mod.RANGE_BY_LABELS))
            self.assertEqual(app.range_by.get(), RANGE_BY_COLOR_LABEL)
            self.assertEqual(app.assign_label.get(), ASSIGN_KMEANS_LABEL)
            self.assertEqual(app._assign_mode(), ASSIGN_KMEANS)
            self.assertEqual(app._split_method(), SPLIT_COLOR_CLOSENESS)

            im = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB")
            app.work_image = im
            with patch.object(ui_mod, "build_range_map", spy):
                app._on_range_by()  # Color closeness
                app.range_by.set(RANGE_BY_LUMA_LABEL)
                app._on_range_by()  # luma default = even pixel split
                app.luma_split_label.set(SPLIT_EQUAL_LIGHTNESS_LABEL)
                app.rebuild_ranges()
                app.range_by.set(RANGE_BY_LAB_A_LABEL)
                app.luma_split_label.set(SPLIT_EQUAL_PIXELS_LABEL)
                app._on_range_by()

            self.assertIn(SPLIT_COLOR_CLOSENESS, methods)
            self.assertIn(SPLIT_EQUAL_PIXELS, methods)
            self.assertIn(SPLIT_EQUAL_LIGHTNESS, methods)
            self.assertIn(SPLIT_LAB_A_PIXELS, methods)
            self.assertEqual(app._split_method(), SPLIT_LAB_A_PIXELS)
            self.assertTrue(app.coverage.luma_mode)
            app.coverage.update_idletasks()
            self.assertTrue(app.coverage.segments.find_withtag("lumakey"))
            self.assertEqual(app.cover_hint.get().strip(), "")
            self.assertIn("luma key", app.coverage._match_swatch_tip().lower())

            from wallpaper_recolor.color.presets import get_preset, list_presets

            named = "V6-N" if get_preset("V6-N") is not None else next(
                (p.name for p in list_presets() if is_color_split(p.split_method)),
                None,
            )
            if named is None:
                self.skipTest("no color-closeness named preset")
            app.preset_choice.set(named)
            app.apply_selected_preset()
            self.assertEqual(app.range_by.get(), RANGE_BY_COLOR_LABEL)
            self.assertEqual(app._split_method(), SPLIT_COLOR_CLOSENESS)
            self.assertFalse(app.coverage.luma_mode)
            self.assertFalse(app.coverage.segments.find_withtag("lumakey"))
            self.assertEqual(app.cover_hint.get().strip(), "")
            self.assertIn("match from", app.coverage._match_swatch_tip().lower())
        finally:
            root.destroy()

    def test_preset_combo_clears_highlight_after_pick(self) -> None:
        """Readonly Presets combo drops selection/focus after <<ComboboxSelected>>."""
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            combo = app.preset_combo
            combo.focus_set()
            try:
                combo.selection_range(0, "end")
            except tk.TclError:
                self.skipTest("selection_range unavailable on this Tk")
            root.update_idletasks()
            try:
                before = bool(combo.selection_present())
            except tk.TclError:
                self.skipTest("selection_present unavailable on this Tk")
            if not before:
                self.skipTest("readonly combo selection is not observable under withdraw()")
            combo.event_generate("<<ComboboxSelected>>")
            root.update_idletasks()
            self.assertFalse(bool(combo.selection_present()))
            self.assertIsNot(root.focus_get(), combo)
        finally:
            root.destroy()

    def test_kmeans_three_clusters_red_patch_not_luma(self) -> None:
        """Pure red goes to the red-ish Lab center, not 'because it is dark'."""
        rng = np.random.default_rng(1)
        img = rng.integers(70, 190, (48, 64, 3), dtype=np.uint8)
        img[8:24, 8:28] = (230, 20, 20)  # red
        img[8:24, 36:56] = (20, 210, 40)  # green
        img[28:44, 20:44] = (30, 50, 220)  # blue
        im = Image.fromarray(img, mode="RGB")
        range_map = build_range_map(im, 3, SPLIT_COLOR_CLOSENESS)
        self.assertEqual(range_map.range_count, 3)
        self.assertEqual(range_map.split_method, SPLIT_COLOR_CLOSENESS)
        self.assertIsNotNone(range_map.centers)
        self.assertIsNotNone(range_map.labels)
        assert range_map.centers is not None
        assert range_map.labels is not None

        red_labels = range_map.labels[8:24, 8:28]
        counts = np.bincount(red_labels.ravel(), minlength=3)
        rid = int(counts.argmax())
        self.assertGreater(counts[rid] / red_labels.size, 0.85)

        red_lab = rgb_tuple_to_lab((230, 20, 20))
        dark_lab = rgb_tuple_to_lab((25, 25, 25))
        center = range_map.centers[rid]
        self.assertLess(_lab_dist2(center, red_lab), _lab_dist2(center, dark_lab))
        # Nearest of the three centers to pure red — chromatic, not luma
        d_red = [_lab_dist2(range_map.centers[i], red_lab) for i in range(3)]
        self.assertEqual(rid, int(np.argmin(d_red)))
        # That center is the reddest (highest Lab a*)
        self.assertEqual(rid, int(np.argmax(range_map.centers[:, 1])))

    def test_luma_key_is_mid_bin_not_coverage_weight(self) -> None:
        """Luma key % is the Rec. 709 bin midpoint, not the coverage weight."""
        from wallpaper_recolor.color.color_ranges import ColorRange

        band = ColorRange(
            index=0,
            luma_low=0.0,
            luma_high=64.0,
            mean_rgb=(20, 20, 20),
            match_rgb=(20, 20, 20),
            replacement_rgb=(20, 20, 20),
            pixel_count=50,
            total_pixels=100,
            weight=0.5,
        )
        self.assertAlmostEqual(band.luma_key, 32.0 / 255.0, places=5)
        self.assertGreater(abs(band.luma_key - band.weight), 0.05)

    def test_v6n_nearest_palette_hex(self) -> None:
        """V6-N: pixels map to nearest of the three greens in Lab."""
        arr = np.zeros((24, 24, 3), dtype=np.uint8)
        arr[:, :] = V6N_LIGHT
        arr[:8, :8] = V6N_DARK
        arr[:8, 16:] = V6N_MID
        im = Image.fromarray(arr, mode="RGB")
        preset = v6n_preset()
        self.assertEqual(preset.split_method, SPLIT_COLOR_CLOSENESS)
        self.assertTrue(preset.palette_as_centers)
        range_map = build_range_map(
            im,
            3,
            SPLIT_COLOR_CLOSENESS,
            palette_rgb=list(preset.palette_rgb),
        )
        self.assertEqual(range_map.range_count, 3)
        assert range_map.labels is not None
        self.assertGreater((range_map.labels[:8, :8] == 0).mean(), 0.9)
        self.assertGreater((range_map.labels[:8, 16:] == 1).mean(), 0.9)
        self.assertGreater((range_map.labels[12:, 12:] == 2).mean(), 0.9)


class TestRangeShift(unittest.TestCase):
    """Insert / drop a range without resetting existing swatches."""
    def test_insert_keeps_existing_match_and_replace(self) -> None:
        """Adding a range must not reset Pantone change-to on the others."""
        rng = np.random.default_rng(2)
        img = rng.integers(40, 220, (40, 48, 3), dtype=np.uint8)
        img[:12, :16] = (30, 40, 50)
        img[:12, 24:] = (200, 30, 40)
        img[20:, :20] = (40, 180, 70)
        im = Image.fromarray(img, mode="RGB")
        range_map = build_range_map(im, 2, SPLIT_COLOR_CLOSENESS)
        kept_match = [band.match_rgb for band in range_map.ranges]
        range_map.set_replacement(0, (12, 34, 56))
        range_map.set_replacement(1, (200, 10, 10))
        range_map.ranges[0].name = "Ink A"
        range_map.texture_strength = 0.85
        range_map.texture_enabled = True
        range_map.tone_lights_cyan = 0.4
        range_map.tone_darks_yellow = -0.25
        range_map.tone_balance_cyan = 0.15
        range_map.tone_temperature = -0.2
        new_i = insert_color_range(range_map)
        self.assertEqual(new_i, 2)
        self.assertEqual(len(range_map.ranges), 3)
        self.assertEqual(range_map.range_count, 3)
        self.assertEqual(range_map.ranges[0].match_rgb, kept_match[0])
        self.assertEqual(range_map.ranges[1].match_rgb, kept_match[1])
        self.assertEqual(range_map.ranges[0].replacement_rgb, (12, 34, 56))
        self.assertEqual(range_map.ranges[1].replacement_rgb, (200, 10, 10))
        self.assertEqual(range_map.ranges[0].name, "Ink A")
        self.assertAlmostEqual(range_map.texture_strength, 0.85)
        self.assertTrue(range_map.texture_enabled)
        self.assertAlmostEqual(range_map.tone_lights_cyan, 0.4)
        self.assertAlmostEqual(range_map.tone_darks_yellow, -0.25)
        self.assertAlmostEqual(range_map.tone_balance_cyan, 0.15)
        self.assertAlmostEqual(range_map.tone_temperature, -0.2)
        self.assertIsNotNone(range_map.labels)
        self.assertGreater(int((range_map.labels == 2).sum()), 0)

    def test_drop_keeps_surviving_swatches(self) -> None:
        im = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8), mode="RGB")
        im.putpixel((0, 0), (220, 20, 20))
        im.putpixel((8, 8), (20, 220, 20))
        range_map = build_range_map(im, 3, SPLIT_COLOR_CLOSENESS)
        range_map.set_replacement(0, (1, 2, 3))
        range_map.set_replacement(1, (4, 5, 6))
        range_map.set_replacement(2, (7, 8, 9))
        keep_match = range_map.ranges[0].match_rgb
        drop_color_range(range_map, 2)
        self.assertEqual(len(range_map.ranges), 2)
        self.assertEqual(range_map.ranges[0].match_rgb, keep_match)
        self.assertEqual(range_map.ranges[0].replacement_rgb, (1, 2, 3))
        self.assertEqual(range_map.ranges[1].replacement_rgb, (4, 5, 6))

    def test_insert_texture_grain_still_applies(self) -> None:
        grain = np.linspace(40, 220, 32, dtype=np.uint8)
        rgb = np.stack([grain, grain, grain], axis=-1)
        rgb = np.broadcast_to(rgb[None, :, :], (24, 32, 3)).copy()
        rgb[4:10, 4:12] = (20, 40, 80)
        im = Image.fromarray(rgb, mode="RGB")
        range_map = build_range_map(im, 2, SPLIT_COLOR_CLOSENESS)
        range_map.set_replacement(0, (40, 160, 80))
        range_map.set_replacement(1, (40, 160, 80))
        insert_color_range(range_map)
        range_map.set_replacement(2, (40, 160, 80))
        assert range_map.rgb is not None and range_map.labels is not None
        exact = exact_rgb(range_map.rgb, range_map.labels, range_map.ranges)
        woven = presentation_rgb(
            range_map.rgb, range_map.labels, range_map.ranges, strength=1.0
        )
        src_l = luma_channel(range_map.rgb)
        out_l = luma_channel(woven)
        exact_l = luma_channel(exact)
        self.assertGreater(float(src_l.std()), 10.0)
        self.assertLess(float(exact_l.std()), 8.0)
        self.assertGreater(float(out_l.std()), 0.5 * float(src_l.std()))

    def test_ui_range_spin_does_not_rebuild_colors(self) -> None:
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod
        from wallpaper_recolor.color.pantone import lookup_pantone_rgb

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            self.assertEqual(app._assign_mode(), ASSIGN_KMEANS)
            rng = np.random.default_rng(3)
            im = Image.fromarray(rng.integers(30, 220, (24, 24, 3), dtype=np.uint8), mode="RGB")
            app.work_image = im
            app.source_image = im
            app.range_count.set(2)
            app.rebuild_ranges()
            assert app.range_map is not None
            app.range_map.set_replacement(0, (9, 9, 9))
            app.range_map.set_replacement(1, (11, 22, 33))
            kept = [band.replacement_rgb for band in app.range_map.ranges]
            app.balance_cyan_pct.set(40.0)
            app.balance_yellow_pct.set(-25.0)
            app._sync_tone_to_map()
            app.range_count.set(3)
            app._on_range_count()
            self.assertEqual(len(app.range_map.ranges), 3)
            self.assertEqual(app.range_map.ranges[0].replacement_rgb, kept[0])
            self.assertEqual(app.range_map.ranges[1].replacement_rgb, kept[1])
            self.assertEqual(app.selected_index, 2)
            self.assertAlmostEqual(app.range_map.texture_strength, TEXTURE_DEFAULT_STRENGTH)
            self.assertTrue(app.range_map.texture_enabled)
            self.assertAlmostEqual(app.balance_cyan_pct.get(), 40.0)
            self.assertAlmostEqual(app.balance_yellow_pct.get(), -25.0)
            self.assertAlmostEqual(app.range_map.tone_balance_cyan, 0.4)
            self.assertAlmostEqual(app.range_map.tone_balance_yellow, -0.25)
            self.assertAlmostEqual(app.range_map.tone_lights_cyan, 0.4)
            pantone_red = lookup_pantone_rgb("186 C")
            self.assertIsNotNone(pantone_red)
            app.selected_half = "replace"
            app.wheel.set_rgb(pantone_red, notify=True)
            self.assertEqual(app.range_map.ranges[2].replacement_rgb, pantone_red)
            app.range_count.set(2)
            app._on_range_count()
            self.assertEqual(len(app.range_map.ranges), 2)
            self.assertEqual(app.range_map.ranges[0].replacement_rgb, kept[0])
            self.assertEqual(app.range_map.ranges[1].replacement_rgb, kept[1])
        finally:
            root.destroy()

    def test_cluster_preset_keeps_match_snap_uses_hexes(self) -> None:
        """Cluster from image paints change-to only; Snap uses palette as Lab centers."""
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod
        from wallpaper_recolor.color.presets import WHITE

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            arr = np.zeros((12, 12, 3), dtype=np.uint8)
            arr[:, :] = (20, 20, 20)
            arr[:4, :4] = (200, 30, 30)
            im = Image.fromarray(arr, mode="RGB")
            app.work_image = im
            app.source_image = im
            app.range_count.set(2)
            app._set_assign_mode(ASSIGN_KMEANS)
            app.preset_choice.set("White and Black")
            app.apply_selected_preset()
            assert app.range_map is not None
            self.assertEqual(app.range_map.ranges[0].replacement_rgb, WHITE)
            self.assertNotEqual(app.range_map.ranges[0].match_rgb, WHITE)
            kmeans_match = app.range_map.ranges[0].match_rgb

            app._set_assign_mode(ASSIGN_PALETTE)
            app.preset_choice.set("White and Black")
            app.apply_selected_preset()
            self.assertEqual(app.range_map.ranges[0].match_rgb, WHITE)
            self.assertEqual(app.range_map.ranges[0].replacement_rgb, WHITE)
            self.assertNotEqual(kmeans_match, WHITE)
        finally:
            root.destroy()


class TestAutoK(unittest.TestCase):
    """Silhouette / inertia k pick — cap so a busy weave does not explode range count."""
    def _rgb_patches(self) -> Image.Image:
        arr = np.zeros((36, 54, 3), dtype=np.uint8)
        arr[:, :18] = (220, 20, 20)
        arr[:, 18:36] = (20, 200, 30)
        arr[:, 36:] = (30, 40, 220)
        return Image.fromarray(arr, mode="RGB")

    def _analogous_teal_three_tone(self, light_frac: float = 0.23) -> Image.Image:
        """Dark / mid / near-white teal with grain — same hue family, split L*."""
        rng = np.random.default_rng(2)
        h, w = 80, 120
        dark = np.array((50, 85, 90), dtype=np.float64)
        mid = np.array((95, 125, 128), dtype=np.float64)
        light = np.array((190, 210, 205), dtype=np.float64)
        yy, xx = np.mgrid[0:h, 0:w]
        field = 0.55 + 0.45 * np.sin(xx * 0.21) + 0.2 * np.sin(yy * 0.17)
        t = np.clip(field, 0.0, 1.0)
        light_cut = float(np.quantile(t, 1.0 - light_frac))
        dark_cut = float(np.quantile(t, 0.40))
        arr = np.empty((h, w, 3), dtype=np.float64)
        arr[:] = mid
        arr[t < dark_cut] = dark
        arr[t >= light_cut] = light
        arr = np.clip(arr + rng.normal(0.0, 7.0, arr.shape), 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def _black_white(self, noise: float = 10.0) -> Image.Image:
        rng = np.random.default_rng(3)
        arr = np.zeros((48, 64, 3), dtype=np.uint8)
        arr[:, :32] = 18
        arr[:, 32:] = 232
        if noise:
            arr = np.clip(
                arr.astype(np.float64) + rng.normal(0.0, noise, arr.shape),
                0,
                255,
            ).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def test_choose_kmeans_k_three_color_patches(self) -> None:
        self.assertEqual(AUTO_K_MAX, 8)
        k = choose_kmeans_k(self._rgb_patches())
        self.assertEqual(k, 3)

    def test_choose_kmeans_k_analogous_teal_three_tone(self) -> None:
        self.assertEqual(choose_kmeans_k(self._analogous_teal_three_tone()), 3)

    def test_choose_kmeans_k_two_color_black_white(self) -> None:
        self.assertEqual(choose_kmeans_k(self._black_white()), 2)
        clean = np.zeros((40, 40, 3), dtype=np.uint8)
        clean[:, :20] = 10
        clean[:, 20:] = 240
        self.assertEqual(choose_kmeans_k(Image.fromarray(clean, mode="RGB")), 2)

    def test_choose_kmeans_k_v6n_scan_if_present(self) -> None:
        path = ROOT / "Wallpapers" / "V6-N" / "V6-N.tif"
        if not path.is_file():
            self.skipTest("Wallpapers/V6-N/V6-N.tif is not in this checkout")
        from wallpaper_recolor.io.image_io import load_image
        from wallpaper_recolor.ui.app import WORK_MAX_EDGE, _fit

        image = load_image(path)
        work = _fit(image, WORK_MAX_EDGE)
        del image
        self.assertEqual(choose_kmeans_k(work), 3)

    def test_open_image_sets_auto_k_and_insert_keeps_swatches(self) -> None:
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "rgb_patches.png"
                self._rgb_patches().save(path)
                self.assertTrue(app._open_image_from_path(path, reset_edits=True))
                k = int(app.range_count.get())
                self.assertEqual(k, 3)
                self.assertEqual(len(app.range_map.ranges), 3)
                self.assertIn("silhouette", app.status.get().lower())
                app.range_map.set_replacement(0, (9, 9, 9))
                app.range_map.set_replacement(1, (11, 22, 33))
                kept = [band.replacement_rgb for band in app.range_map.ranges[:2]]
                app.range_count.set(k + 1)
                app._on_range_count()
                self.assertEqual(len(app.range_map.ranges), k + 1)
                self.assertEqual(app.range_map.ranges[0].replacement_rgb, kept[0])
                self.assertEqual(app.range_map.ranges[1].replacement_rgb, kept[1])
        finally:
            root.destroy()

    def test_named_preset_forces_range_count(self) -> None:
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "rgb_patches.png"
                self._rgb_patches().save(path)
                self.assertTrue(app._open_image_from_path(path, reset_edits=True))
                self.assertEqual(int(app.range_count.get()), 3)
                app.preset_choice.set("White and Black")
                app.apply_selected_preset()
                self.assertEqual(int(app.range_count.get()), 2)
                self.assertEqual(len(app.range_map.ranges), 2)
        finally:
            root.destroy()


class TestSavePath(unittest.TestCase):
    """Background save / export; busy bar; grain vs flat fill."""
    def test_tiff_default_is_fast_not_lzw(self) -> None:
        self.assertEqual(TIFF_COMPRESSION_FAST, "tiff_adobe_deflate")
        src = inspect.getsource(save_image)
        self.assertIn("tiff_adobe_deflate", src)
        self.assertIn("LZW", src)  # documented as too slow
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.tif"
            im = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8), mode="RGB")
            save_image(im, path)
            self.assertTrue(path.is_file())
            with Image.open(path) as probe:
                self.assertEqual(probe.size, (4, 4))

    def test_png_default_is_fast_not_optimize(self) -> None:
        """Layers zip writes many PNGs; optimize=True was the encode stall."""
        self.assertEqual(PNG_COMPRESS_LEVEL_FAST, 1)
        src = inspect.getsource(save_image)
        self.assertIn("PNG_COMPRESS_LEVEL_FAST", src)
        self.assertNotIn('["optimize"] = True', src)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.png"
            im = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8), mode="RGB")
            save_image(im, path)
            self.assertTrue(path.is_file())
            with Image.open(path) as probe:
                self.assertEqual(probe.size, (4, 4))

    def test_save_starts_background_thread(self) -> None:
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod

        root = tk.Tk()
        root.withdraw()
        started: list = []
        apply_called = {"n": 0}
        save_called = {"n": 0}

        class FakeThread:
            def __init__(self, target=None, daemon=None, **_kwargs) -> None:
                self.target = target
                started.append(self)

            def start(self) -> None:
                pass  # do not run the worker on this (main) thread

        def _fake_apply(*_a, **_k):
            apply_called["n"] += 1
            return Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB")

        def _fake_save(*_a, **_k):
            save_called["n"] += 1

        try:
            app = ui_mod.WallpaperRecolorApp(root)
            self.assertEqual(app.range_by.get(), RANGE_BY_COLOR_LABEL)
            im = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB")
            app.source_image = im
            app.range_map = snapshot_assignment(build_range_map(im, 2, SPLIT_COLOR_CLOSENESS))
            with patch.object(ui_mod.filedialog, "asksaveasfilename", return_value=str(ROOT / "_out.tif")):
                with patch.object(ui_mod.threading, "Thread", FakeThread):
                    with patch.object(ui_mod, "composite_for_image", _fake_apply):
                        with patch.object(ui_mod, "save_image", _fake_save):
                            app.save_image_as()
            self.assertEqual(len(started), 1)
            self.assertTrue(callable(started[0].target))
            self.assertEqual(apply_called["n"], 0)
            self.assertEqual(save_called["n"], 0)
            run_src = inspect.getsource(ui_mod.run)
            self.assertIn("mainloop", run_src)
            self.assertNotIn("save_image", run_src)
            self.assertNotIn("composite_for_image", run_src)
            self.assertFalse(hasattr(app, "save_mode_combo"))
            self.assertFalse(hasattr(ui_mod, "SAVE_MODE_TEXTURE"))
            self.assertFalse(hasattr(ui_mod, "SAVE_MODES"))
        finally:
            root.destroy()

    def test_save_as_follows_texture_eye_and_slider(self) -> None:
        """Save as… matches Result: eye off or 0% → exact; otherwise grain at slider mix."""
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod

        root = tk.Tk()
        root.withdraw()
        grains: list[bool] = []
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            self.assertFalse(hasattr(app, "save_mode_combo"))
            self.assertFalse(hasattr(app, "save_mode"))
            self.assertFalse(hasattr(ui_mod, "SAVE_MODE_TEXTURE"))
            self.assertFalse(hasattr(ui_mod, "SAVE_MODES"))
            layout = inspect.getsource(ui_mod.WallpaperRecolorApp._build_layout)
            self.assertNotIn("save_mode_combo", layout)
            self.assertNotIn("SAVE_MODES", layout)
            save_src = inspect.getsource(ui_mod.WallpaperRecolorApp.save_image_as)
            self.assertIn("_save_uses_grain", save_src)
            grain_src = inspect.getsource(ui_mod.WallpaperRecolorApp._save_uses_grain)
            self.assertIn("texture_enabled", grain_src)
            self.assertIn("_texture_strength", grain_src)
            self.assertNotIn("save_mode", grain_src)
            tool_src = inspect.getsource(ui_mod.WallpaperRecolorApp._master_work)
            self.assertNotIn("save_mode", tool_src)
            refresh_src = inspect.getsource(ui_mod.WallpaperRecolorApp._refresh_tool_tab)
            self.assertNotIn("save_mode", refresh_src)

            self.assertTrue(app.texture_enabled.get())
            self.assertAlmostEqual(app._texture_strength(), TEXTURE_DEFAULT_STRENGTH)
            self.assertTrue(app._save_uses_grain())

            orig_save = app._save_composite

            def _capture(*, grain: bool) -> None:
                grains.append(grain)

            app._save_composite = _capture  # type: ignore[method-assign]
            app.save_image_as()
            self.assertEqual(grains, [True])

            app.texture_enabled.set(False)
            self.assertFalse(app._save_uses_grain())
            app.save_image_as()
            self.assertEqual(grains[-1], False)

            app.texture_enabled.set(True)
            app.texture_pct.set(0.0)
            self.assertEqual(app._texture_strength(), 0.0)
            self.assertFalse(app._save_uses_grain())
            app.save_image_as()
            self.assertEqual(grains[-1], False)

            app.texture_pct.set(40.0)
            self.assertTrue(app._save_uses_grain())
            app.save_image_as()
            self.assertEqual(grains[-1], True)
            app._save_composite = orig_save  # type: ignore[method-assign]
        finally:
            root.destroy()

    def test_busy_progress_indicator(self) -> None:
        """Save/Export busy state runs a compact bar in the footer; slot never reflows."""
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod

        root = tk.Tk()
        root.withdraw()
        started: list[int] = []
        stopped: list[int] = []
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            self.assertTrue(hasattr(app, "footer"))
            self.assertTrue(hasattr(app, "busy_bar"))
            self.assertTrue(hasattr(app, "busy_progress"))
            self.assertIs(app.open_btn.master, app.toolbar)
            self.assertIs(app.tools_combo.master, app.tool_strip)
            self.assertIs(app.tool_strip.master, app.toolbar)
            self.assertNotIn("Save as…", [str(w.cget("text")) for w in app.toolbar.pack_slaves() if w.winfo_class() in ("TButton", "Button")])
            self.assertIs(app.busy_bar.master, app.footer)
            self.assertIs(app.status_bar.master, app.footer)
            self.assertIs(app.footer.master, root)
            self.assertIsNot(app.busy_bar.master, app.toolbar)
            self.assertEqual(str(app.busy_progress.cget("mode")), "indeterminate")
            bar_len = int(str(app.busy_progress.cget("length")))
            self.assertGreaterEqual(bar_len, 180)
            self.assertLessEqual(bar_len, 240)
            self.assertEqual(str(app.busy_bar.winfo_manager()), "pack")
            idle_info = app.busy_bar.pack_info()
            self.assertNotEqual(str(idle_info.get("fill", "none")), "x")
            # Slot stays reserved; the green fill is not mapped while idle.
            self.assertFalse(app.busy_bar.pack_propagate())
            self.assertEqual(str(app.busy_progress.winfo_manager()), "")
            self.assertEqual(str(app.busy_cancel.winfo_manager()), "")
            self.assertEqual(str(app.open_btn.cget("state")), "normal")

            orig_start = app.busy_progress.start
            orig_stop = app.busy_progress.stop

            def _start(*args, **kwargs):
                started.append(1)
                return orig_start(*args, **kwargs)

            def _stop(*args, **kwargs):
                stopped.append(1)
                return orig_stop(*args, **kwargs)

            app.busy_progress.start = _start  # type: ignore[method-assign]
            app.busy_progress.stop = _stop  # type: ignore[method-assign]

            root.update_idletasks()
            idle_footer_h = app.footer.winfo_reqheight()
            idle_toolbar_h = app.toolbar.winfo_reqheight()

            app._set_busy(True, "Saving…")
            root.update_idletasks()
            self.assertTrue(app._busy)
            self.assertEqual(str(app.busy_bar.winfo_manager()), "pack")
            info = app.busy_bar.pack_info()
            self.assertIn(str(info.get("side", "")), ("left", "right"))
            self.assertNotEqual(str(info.get("fill", "none")), "x")
            self.assertNotIn(app.busy_bar, list(app.toolbar.pack_slaves()))
            self.assertIn(app.open_btn, list(app.toolbar.pack_slaves()))
            self.assertEqual(str(app.busy_progress.winfo_manager()), "place")
            self.assertEqual(str(app.busy_cancel.winfo_manager()), "")
            self.assertEqual(app.busy_caption.get(), "Saving…")
            self.assertEqual(app.status.get(), "Saving…")
            self.assertEqual(str(app.open_btn.cget("state")), "disabled")
            self.assertEqual(str(app.file_menu.entrycget("Save as…", "state")), "disabled")
            self.assertEqual(str(root.cget("cursor")), "watch")
            self.assertGreaterEqual(sum(started), 1)
            self.assertEqual(app.footer.winfo_reqheight(), idle_footer_h)
            self.assertEqual(app.toolbar.winfo_reqheight(), idle_toolbar_h)

            app._apply_progress_status("Writing TIFF…")
            self.assertEqual(app.busy_caption.get(), "Writing TIFF…")
            self.assertEqual(app.status.get(), "Writing TIFF…")

            app._set_busy(False)
            root.update_idletasks()
            self.assertFalse(app._busy)
            self.assertEqual(str(app.busy_bar.winfo_manager()), "pack")
            self.assertEqual(str(app.busy_progress.winfo_manager()), "")
            self.assertEqual(float(str(app.busy_progress.cget("value"))), 0.0)
            self.assertEqual(str(app.open_btn.cget("state")), "normal")
            self.assertEqual(str(app.file_menu.entrycget("Save as…", "state")), "normal")
            self.assertEqual(str(root.cget("cursor")), "")
            self.assertGreaterEqual(sum(stopped), 1)
            self.assertEqual(app.footer.winfo_reqheight(), idle_footer_h)
            self.assertEqual(app.toolbar.winfo_reqheight(), idle_toolbar_h)

            app._job_cancellable = True
            app._set_busy(True, "Detecting…")
            root.update_idletasks()
            self.assertEqual(str(app.busy_cancel.winfo_manager()), "pack")
            cancel_info = app.busy_cancel.pack_info()
            bar_info = app.busy_bar.pack_info()
            self.assertEqual(str(cancel_info.get("side", "")), "right")
            self.assertEqual(str(bar_info.get("side", "")), "right")
            slaves = [w for w in app.footer.pack_slaves() if w in (app.busy_cancel, app.busy_bar)]
            self.assertEqual(slaves, [app.busy_bar, app.busy_cancel])
            app._set_busy(False)
            root.update_idletasks()
            self.assertEqual(str(app.busy_cancel.winfo_manager()), "")
            busy_src = inspect.getsource(ui_mod.WallpaperRecolorApp._set_busy)
            self.assertIn("_set_busy_indicator", busy_src)
            ind_src = inspect.getsource(ui_mod.WallpaperRecolorApp._set_busy_indicator)
            self.assertIn("start(", ind_src)
            self.assertIn("stop(", ind_src)
            self.assertIn("place_forget", ind_src)
            self.assertNotIn("pack_forget", ind_src)
            self.assertNotIn("before=self.save_btn", ind_src)
            self.assertIn("watch", ind_src)
            self.assertNotIn("after=self.toolbar", ind_src)
            self.assertNotIn('fill="x"', ind_src)
            layout_src = inspect.getsource(ui_mod.WallpaperRecolorApp._build_layout)
            self.assertIn("self.footer", layout_src)
            self.assertNotIn("save_cluster", layout_src)
            run_src = inspect.getsource(ui_mod.WallpaperRecolorApp._run_background)
            self.assertIn("Thread", run_src)
            self.assertIn("daemon", run_src)
            self.assertNotIn("save_image(", run_src)
            self.assertIn("finally:", run_src)
            self.assertIn("_set_busy(False)", run_src)
        finally:
            root.destroy()

    def test_tess_build_toggles_busy_bar(self) -> None:
        """Tessellate Build maps the footer bar, then unmaps on success or error."""
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod

        root = tk.Tk()
        root.withdraw()
        started: list[int] = []
        stopped: list[int] = []
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            im = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB")
            app.work_image = im
            app.source_image = im
            app.rebuild_ranges()
            app._clear_history()

            orig_start = app.busy_progress.start
            orig_stop = app.busy_progress.stop

            def _start(*args, **kwargs):
                started.append(1)
                return orig_start(*args, **kwargs)

            def _stop(*args, **kwargs):
                stopped.append(1)
                return orig_stop(*args, **kwargs)

            app.busy_progress.start = _start  # type: ignore[method-assign]
            app.busy_progress.stop = _stop  # type: ignore[method-assign]

            busy_calls: list[tuple[bool, str | None]] = []
            orig_busy = app._set_busy

            def _busy(busy: bool, status: str | None = None) -> None:
                busy_calls.append((busy, status))
                orig_busy(busy, status)

            app._set_busy = _busy  # type: ignore[method-assign]

            self.assertEqual(str(app.busy_progress.winfo_manager()), "")
            app.tess_h.set("left")
            app._tess_committed = ("off", "off", False, "tessellate")
            app._on_tess_build()
            self.assertTrue(app._busy)
            self.assertEqual(str(app.busy_progress.winfo_manager()), "place")
            self.assertEqual(app.status.get(), "Building…")
            self.assertGreaterEqual(sum(started), 1)
            _drain_busy(app, root)
            self.assertFalse(app._busy)
            self.assertEqual(str(app.busy_progress.winfo_manager()), "")
            self.assertEqual(float(str(app.busy_progress.cget("value"))), 0.0)
            self.assertGreaterEqual(sum(stopped), 1)
            self.assertTrue(bool(app.tess_built.get()))
            self.assertEqual(busy_calls[0], (True, "Building…"))
            self.assertEqual(busy_calls[-1], (False, None))

            src = inspect.getsource(ui_mod.WallpaperRecolorApp._on_tess_build)
            self.assertIn("_run_background", src)
            self.assertIn("Building…", src)
            self.assertIn("_preview_pils", src)
            self.assertIn("tess_build_btn", inspect.getsource(ui_mod.WallpaperRecolorApp._build_tessellate_panel))
            menu_src = inspect.getsource(ui_mod.WallpaperRecolorApp._rebuild_edit_menu)
            self.assertIn("Tessellate Build", menu_src)
            self.assertIn("_on_tess_build", menu_src)

            app.tess_built.set(False)
            app._tess_committed = ("off", "off", False, "tessellate")
            with patch.object(ui_mod.messagebox, "showerror"):
                with patch.object(app, "_preview_pils", side_effect=RuntimeError("boom")):
                    app._on_tess_build()
                    _drain_busy(app, root)
            self.assertFalse(app._busy)
            self.assertEqual(str(app.busy_progress.winfo_manager()), "")
            self.assertEqual(app.status.get(), "Build failed")
        finally:
            root.destroy()


class TestTextureStrength(unittest.TestCase):
    """Color/Luminosity grain mix; Fit contain; View Move vs Grab Move."""
    def test_overlay_gray_film_path_is_gone(self) -> None:
        import wallpaper_recolor.color.layers as layers

        self.assertFalse(hasattr(layers, "overlay_luma_on_rgb"))
        src = inspect.getsource(layers)
        self.assertNotIn("2.0 * base * blend", src)
        self.assertNotIn("Photoshop Overlay", src)

    def test_grain_keeps_luma_std_and_shifts_mean(self) -> None:
        """Color/Luminosity: weave stays; mean RGB moves toward the picked hex."""
        h, w = 48, 64
        yy, xx = np.indices((h, w))
        grain = (100 + 50 * np.sin(xx * 0.4) + 20 * np.sin(yy * 0.7)).clip(0, 255)
        grain = grain.astype(np.uint8)
        rgb = np.stack(
            [
                grain,
                (grain.astype(np.int16) * 55 // 100).clip(0, 255).astype(np.uint8),
                (grain.astype(np.int16) * 35 // 100).clip(0, 255).astype(np.uint8),
            ],
            axis=-1,
        )
        im = Image.fromarray(rgb, mode="RGB")
        range_map = build_range_map(im, 1, SPLIT_COLOR_CLOSENESS)
        range_map.set_replacement(0, (40, 160, 80))
        assert range_map.rgb is not None and range_map.labels is not None

        exact = exact_rgb(range_map.rgb, range_map.labels, range_map.ranges)
        none = presentation_rgb(
            range_map.rgb, range_map.labels, range_map.ranges, strength=0.0
        )
        woven = presentation_rgb(
            range_map.rgb, range_map.labels, range_map.ranges, strength=1.0
        )
        np.testing.assert_array_equal(none, exact)

        src_l = luma_channel(range_map.rgb)
        out_l = luma_channel(woven)
        exact_l = luma_channel(exact)
        src_std = float(src_l.std())
        self.assertGreater(src_std, 15.0)
        self.assertLess(float(exact_l.std()), 1.0)
        self.assertGreater(float(out_l.std()), 0.7 * src_std)

        src_mean = range_map.rgb.astype(np.float64).mean(axis=(0, 1))
        out_mean = woven.astype(np.float64).mean(axis=(0, 1))
        self.assertGreater(out_mean[1] - out_mean[0], src_mean[1] - src_mean[0])

    def test_lab_roundtrip_neutral(self) -> None:
        rgb = np.array([[[40, 40, 40], [200, 200, 200]]], dtype=np.uint8)
        back = lab_to_rgb_array(rgb_to_lab_array(rgb))
        np.testing.assert_allclose(back.astype(np.int16), rgb.astype(np.int16), atol=2)

    def test_snapshot_keeps_texture_strength(self) -> None:
        im = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB")
        range_map = build_range_map(im, 2, SPLIT_COLOR_CLOSENESS)
        range_map.texture_strength = 0.8
        snap = snapshot_assignment(range_map)
        self.assertAlmostEqual(snap.texture_strength, 0.8)

    def test_ui_texture_slider_wired(self) -> None:
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod
        from wallpaper_recolor.ui import run

        self.assertTrue(callable(run))
        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            self.assertIsNotNone(app.texture_scale)
            self.assertAlmostEqual(app._texture_strength(), TEXTURE_DEFAULT_STRENGTH, places=2)
            app.texture_pct.set(0.0)
            app._on_texture_slider("0")
            self.assertEqual(app._texture_strength(), 0.0)
            self.assertFalse(app._save_uses_grain())
            app.texture_pct.set(100.0)
            app._on_texture_slider("100")
            self.assertEqual(app._texture_strength(), 1.0)
            self.assertTrue(app._save_uses_grain())
            self.assertLess(float(app.texture_scale.cget("from")), float(app.texture_scale.cget("to")))
            app.texture_enabled.set(False)
            self.assertFalse(app._save_uses_grain())
        finally:
            root.destroy()

    def test_ui_preview_pop_out_keeps_live_preview(self) -> None:
        """Only Preview pops out; other panes stay in columns. Same widgets after dock."""
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod
        from wallpaper_recolor.ui import run

        self.assertTrue(callable(run))
        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            orig = app.orig_label
            wheel_cb = app.wheel.on_color
            self.assertTrue(app.preview_panel.allow_pop_out)
            self.assertTrue(app.preview_panel.flex)
            self.assertIsNotNone(app.preview_panel._pop_btn)
            self.assertFalse(app.preview_panel.is_floating)
            app.preview_panel.pop_out()
            self.assertTrue(app.preview_panel.is_floating)
            self.assertIs(app.orig_label, orig)
            self.assertIs(app.wheel.on_color, wheel_cb)
            app._schedule_preview()
            app._refresh_previews()
            app.preview_panel.dock()
            self.assertFalse(app.preview_panel.is_floating)
            self.assertIs(app.orig_label, orig)
            for panel in (
                app.texture_panel,
                app.coverage_panel,
                app.tone_panel,
                app.scale_panel,
                app.crop_panel,
                app.tess_panel,
                app.layers_panel,
                app.labels_panel,
                app.wheel_panel,
            ):
                self.assertFalse(panel.allow_pop_out)
                self.assertFalse(panel.flex)
                self.assertIsNone(panel._pop_btn)
                panel.pop_out()  # no-op — sliders / wheel never undock
                self.assertFalse(panel.is_floating)
            self.assertIsNotNone(app.texture_scale)
            self.assertIs(app.wheel.on_color, wheel_cb)
            self.assertEqual(app.wheel.on_color.__func__, ui_mod.WallpaperRecolorApp._on_wheel_color)
            self.assertFalse(hasattr(app, "exact_label"))
            self.assertEqual(app.left_column.panels, [app.preview_panel, app.coverage_panel])
            self.assertEqual(
                app.right_column.panels,
                [
                    app.wheel_panel,
                    app.texture_panel,
                    app.tone_panel,
                    app.scale_panel,
                    app.crop_panel,
                ],
            )
            self.assertFalse(app.tess_panel.hidden)
            self.assertIs(app.tess_panel.column, app.right_bottom_column)
            self.assertIs(app.history_panel.column, app.right_bottom_column)
            self.assertIs(app.wheel_panel.column, app.right_column)
            self.assertIs(app.preview_panel.column, app.left_column)
            self.assertIs(app.coverage_panel.column, app.left_column)
            self.assertIs(app.texture_panel.column, app.right_column)
            self.assertIs(app.tone_panel.column, app.right_column)
            self.assertIs(app.scale_panel.column, app.right_column)
            self.assertIs(app.crop_panel.column, app.right_column)
            self.assertEqual(len(app.body_paned.panes()), 2)
            self.assertTrue(callable(app.body_paned.sashpos))

            # Snap coverage under the wheel: they pack in the scrollable inner, not clipped
            app._place_panel(app.texture_panel, app.right_column, 1)
            app._place_panel(app.coverage_panel, app.right_column, 2)
            app._place_panel(app.tone_panel, app.right_column, 3)
            docked = app.right_column._docked_panels()
            self.assertEqual(
                docked,
                [
                    app.wheel_panel,
                    app.texture_panel,
                    app.coverage_panel,
                    app.tone_panel,
                    app.scale_panel,
                    app.crop_panel,
                ],
            )
            for panel in docked:
                self.assertFalse(panel.is_floating)
                info = panel.pack_info()
                self.assertEqual(str(info["in"]), str(app.right_column.inner))
                self.assertEqual(str(info.get("expand", "0")), "0")
            root.update_idletasks()
            app.right_column._sync_layout()
            inner_h = int(app.right_column.inner.winfo_reqheight())
            win_h = int(float(app.right_column.canvas.itemcget(app.right_column._win, "height")))
            self.assertGreaterEqual(win_h, inner_h - 2)
            self.assertGreater(inner_h, int(app.wheel_panel.winfo_reqheight()))
            self.assertIs(app.right_column.inner.master, app.right_column.canvas)
        finally:
            root.destroy()

    def test_preview_view_zoom_independent_of_crop(self) -> None:
        """View zoom defaults to 100%; 200% does not change crop; wheel over image zooms."""
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod
        from wallpaper_recolor.ui import run

        self.assertTrue(callable(run))
        handler_src = inspect.getsource(ui_mod.WallpaperRecolorApp._on_column_mousewheel)
        self.assertIn("_on_preview_ctrl_wheel", handler_src)
        wheel_src = inspect.getsource(ui_mod.WallpaperRecolorApp._on_preview_ctrl_wheel)
        self.assertIn("_pointer_over_preview_image", wheel_src)
        self.assertIn("_wheel_zoom_pct_delta", wheel_src)
        self.assertNotIn("_event_ctrl_down", wheel_src)

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            self.assertTrue(callable(app._on_preview_ctrl_wheel))
            self.assertAlmostEqual(app.preview_zoom.get(), 100.0)
            self.assertEqual(app.preview_zoom_caption.get(), "100%")
            self.assertAlmostEqual(app.crop_zoom.get(), 1.0)
            self.assertAlmostEqual(app.crop_x.get(), 0.0)
            self.assertAlmostEqual(app.crop_y.get(), 0.0)
            self.assertTrue(hasattr(app, "preview_zoom_scale"))
            self.assertTrue(hasattr(app, "preview_zoom_fit"))
            self.assertTrue(hasattr(app, "orig_zoom_host"))
            self.assertIs(app.orig_label, app.orig_zoom_host.image_label)
            self.assertIs(app.tex_label, app.tex_zoom_host.image_label)

            app.crop_zoom.set(1.0)
            app.crop_x.set(12.0)
            app.crop_y.set(8.0)
            app.preview_zoom.set(200.0)
            app._on_preview_zoom_slider("200")
            self.assertAlmostEqual(app.preview_zoom.get(), 200.0)
            self.assertEqual(app.preview_zoom_caption.get(), "200%")
            self.assertAlmostEqual(app.crop_zoom.get(), 1.0)
            self.assertAlmostEqual(app.crop_x.get(), 12.0)
            self.assertAlmostEqual(app.crop_y.get(), 8.0)

            app._orig_pil = Image.new("RGB", (40, 20), (10, 20, 30))
            app._orig_photo = None
            mapped = app._orig_click_to_display(20, 10)
            self.assertEqual(mapped, (10, 5))

            class _WheelUpOrig:
                delta = 120
                state = 0
                widget = app.orig_label
                x_root = 0
                y_root = 0
                num = "??"

            self.assertEqual(app._on_preview_ctrl_wheel(_WheelUpOrig()), "break")
            self.assertAlmostEqual(app.preview_zoom.get(), 225.0)
            self.assertAlmostEqual(app.crop_zoom.get(), 1.0)
            self.assertAlmostEqual(app.crop_x.get(), 12.0)

            class _CtrlWheel:
                delta = 120
                state = 0x4
                widget = app.orig_label
                x_root = 0
                y_root = 0
                num = 0

            self.assertEqual(app._on_preview_ctrl_wheel(_CtrlWheel()), "break")
            self.assertAlmostEqual(app.preview_zoom.get(), 250.0)

            app._reset_preview_zoom()
            self.assertAlmostEqual(app.preview_zoom.get(), 100.0)
            self.assertEqual(app.preview_zoom_caption.get(), "100%")
            self.assertAlmostEqual(app.crop_zoom.get(), 1.0)
            self.assertAlmostEqual(app.crop_x.get(), 12.0)
            self.assertAlmostEqual(app.crop_y.get(), 8.0)
            app._orig_photo = None
            self.assertEqual(app._orig_click_to_display(20, 10), (20, 10))

            scale_src = inspect.getsource(ui_mod._scale_view_zoom)
            self.assertIn("Image.Resampling.NEAREST", scale_src)
            self.assertNotIn("Image.Resampling.BILINEAR", scale_src)
            self.assertNotIn("Image.Resampling.BICUBIC", scale_src)
            refresh_src = inspect.getsource(ui_mod.WallpaperRecolorApp._refresh_previews)
            self.assertNotIn("_fit(", refresh_src)

            work = Image.new("RGB", (800, 400), (10, 20, 30))
            work.putpixel((0, 0), (255, 0, 0))
            app.work_image = work
            app.source_image = work
            # Treat hosts as unmapped so 100% uses PREVIEW_MAX_EDGE (job layout can map a pane).
            app._fit_pane_size = lambda _host: (1, 1)  # type: ignore[method-assign]
            app.rebuild_ranges()
            self.assertEqual(app._orig_pil.size, (800, 400))
            # Unmapped panes fall back to PREVIEW_MAX_EDGE
            self.assertEqual(app._orig_photo.width(), 560)
            self.assertEqual(app._orig_photo.height(), 280)
            app.preview_zoom.set(200.0)
            app._on_preview_zoom_slider("200")
            self.assertAlmostEqual(app.crop_zoom.get(), 1.0)
            self.assertEqual(app._orig_photo.width(), 1120)
            self.assertEqual(app._orig_photo.height(), 560)
            app._orig_photo = None
            self.assertEqual(app._orig_click_to_display(560, 280), (400, 200))

            orig_pane = (400, 300)
            tex_pane = (360, 280)

            def _stub_panes(host, _orig=orig_pane, _tex=tex_pane):
                if host is app.orig_zoom_host:
                    return _orig
                if host is app.tex_zoom_host:
                    return _tex
                return (640, 480)

            app._fit_pane_size = _stub_panes  # type: ignore[method-assign]
            fit_edge = ui_mod.fit_max_edge(800, 400, (orig_pane, tex_pane))
            self.assertEqual(fit_edge, 360)
            fit_w, fit_h = ui_mod._preview_base_size(800, 400, fit_edge)
            self.assertEqual((fit_w, fit_h), (360, 180))
            self.assertLessEqual(fit_w, min(orig_pane[0], tex_pane[0]))
            self.assertLessEqual(fit_h, min(orig_pane[1], tex_pane[1]))
            app._reset_preview_zoom()
            self.assertAlmostEqual(app.preview_zoom.get(), 100.0)
            self.assertEqual(app.preview_zoom_caption.get(), "100%")
            self.assertEqual(app._orig_photo.width(), fit_w)
            self.assertEqual(app._orig_photo.height(), fit_h)
            self.assertEqual(app._tex_photo.width(), fit_w)
            self.assertEqual(app._tex_photo.height(), fit_h)
            app.preview_zoom.set(200.0)
            app._on_preview_zoom_slider("200")
            self.assertEqual(app._orig_photo.width(), fit_w * 2)
            self.assertEqual(app._orig_photo.height(), fit_h * 2)
            self.assertEqual(app._tex_photo.width(), app._orig_photo.width())
            self.assertEqual(app._tex_photo.height(), app._orig_photo.height())
            app.preview_zoom_fit.event_generate("<Button-1>")
            # Label bind may not fire under withdraw; call Fit directly
            app._reset_preview_zoom()
            self.assertAlmostEqual(app.preview_zoom.get(), 100.0)
            self.assertEqual((app._orig_photo.width(), app._orig_photo.height()), (fit_w, fit_h))

            empty_args = app._output_scale_args()
            app.preview_zoom.set(400.0)
            app._on_preview_zoom_slider("400")
            self.assertEqual(app._output_scale_args(), empty_args)
            self.assertIsNone(app._output_scale_args()[0])
            app.scale_width.set("400")
            app.scale_height.set("400")
            size_400, _filt, _dpi = app._output_scale_args()
            self.assertEqual(size_400, (400, 400))
            app.preview_zoom.set(400.0)
            app._on_preview_zoom_slider("400")
            self.assertEqual(app._output_scale_args()[0], (400, 400))
            self.assertNotEqual(
                (app._orig_photo.width(), app._orig_photo.height()), (400, 400)
            )
        finally:
            root.destroy()

    def test_preview_wheel_zooms_over_image_same_orig_result_size(self) -> None:
        """Wheel-up over Original zooms; Original and Result PhotoImages stay equal."""
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod

        delta_src = inspect.getsource(ui_mod._wheel_zoom_pct_delta)
        self.assertIn("TypeError", delta_src)
        self.assertIn("ValueError", delta_src)

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            work = Image.new("RGB", (800, 400), (10, 20, 30))
            app.work_image = work
            app.source_image = work
            app.rebuild_ranges()
            self.assertAlmostEqual(app.preview_zoom.get(), 100.0)
            self.assertIsNotNone(app._orig_photo)
            self.assertIsNotNone(app._tex_photo)
            size_100 = (app._orig_photo.width(), app._orig_photo.height())
            self.assertEqual(
                (app._tex_photo.width(), app._tex_photo.height()), size_100
            )

            class _WheelUpOrig:
                delta = 120
                state = 0
                widget = app.orig_label
                x_root = 0
                y_root = 0
                num = "??"

            left_hits: list[int] = []
            app.left_column._on_mousewheel = (  # type: ignore[method-assign]
                lambda _e: left_hits.append(1) or "break"
            )
            self.assertEqual(app._on_column_mousewheel(_WheelUpOrig()), "break")
            self.assertAlmostEqual(app.preview_zoom.get(), 125.0)
            self.assertEqual(left_hits, [])

            class _WheelDownOrig:
                delta = -120
                state = 0
                widget = app.orig_label
                x_root = 0
                y_root = 0
                num = "??"

            self.assertEqual(app._on_column_mousewheel(_WheelDownOrig()), "break")
            self.assertAlmostEqual(app.preview_zoom.get(), 100.0)

            orig_pane = (400, 300)
            tex_pane = (360, 280)

            def _stub_panes(host, _orig=orig_pane, _tex=tex_pane):
                if host is app.orig_zoom_host:
                    return _orig
                if host is app.tex_zoom_host:
                    return _tex
                return (640, 480)

            app._fit_pane_size = _stub_panes  # type: ignore[method-assign]
            app._reset_preview_zoom()
            fit_w, fit_h = ui_mod._preview_base_size(
                800, 400, ui_mod.fit_max_edge(800, 400, (orig_pane, tex_pane))
            )
            self.assertEqual((app._orig_photo.width(), app._orig_photo.height()), (fit_w, fit_h))
            self.assertEqual((app._tex_photo.width(), app._tex_photo.height()), (fit_w, fit_h))
            app.preview_zoom.set(200.0)
            app._on_preview_zoom_slider("200")
            self.assertEqual(app._orig_photo.width(), fit_w * 2)
            self.assertEqual(app._orig_photo.height(), fit_h * 2)
            self.assertEqual(app._tex_photo.width(), app._orig_photo.width())
            self.assertEqual(app._tex_photo.height(), app._orig_photo.height())

            class _WheelOffImage:
                delta = 120
                state = 0
                widget = app.texture_scale
                x_root = -10_000
                y_root = -10_000
                num = "??"

            zoom_before = float(app.preview_zoom.get())
            app._pointer_over_preview_image = lambda _e: False  # type: ignore[method-assign]
            app.left_column.contains_root = lambda _x, _y: False  # type: ignore[method-assign]
            app.right_column.contains_root = lambda _x, _y: True  # type: ignore[method-assign]
            right_hits: list[int] = []
            app.right_column._on_mousewheel = (  # type: ignore[method-assign]
                lambda _e: right_hits.append(1) or "break"
            )
            self.assertEqual(app._on_column_mousewheel(_WheelOffImage()), "break")
            self.assertAlmostEqual(app.preview_zoom.get(), zoom_before)
            self.assertEqual(right_hits, [1])
        finally:
            root.destroy()

    def test_scale_view_zoom_nearest_from_high_res(self) -> None:
        """View zoom NEAREST-scales the work bitmap, not a BILINEAR 560px preview."""
        import wallpaper_recolor.ui.app as ui_mod

        arr = np.zeros((400, 800, 3), dtype=np.uint8)
        arr[::2, ::2] = (255, 0, 0)
        arr[1::2, 1::2] = (255, 0, 0)
        arr[::2, 1::2] = (0, 0, 255)
        arr[1::2, ::2] = (0, 0, 255)
        im = Image.fromarray(arr, mode="RGB")
        out = ui_mod._scale_view_zoom(im, 2.0)
        self.assertEqual(out.size, (1120, 560))
        uniq = {tuple(int(c) for c in px) for px in np.unique(np.asarray(out).reshape(-1, 3), axis=0)}
        self.assertTrue(uniq.issubset({(255, 0, 0), (0, 0, 255)}), uniq)

        muddy = im.resize((560, 280), Image.Resampling.BILINEAR)
        muddy = muddy.resize((1120, 560), Image.Resampling.NEAREST)
        muddy_uniq = {
            tuple(int(c) for c in px)
            for px in np.unique(np.asarray(muddy).reshape(-1, 3), axis=0)
        }
        self.assertGreater(len(muddy_uniq), 2)

    def test_shared_fit_factor_uses_smaller_pane(self) -> None:
        """Original and Result share the min pane-fit; 100% cannot overflow either."""
        import wallpaper_recolor.ui.app as ui_mod

        orig_pane = (400, 300)
        tex_pane = (360, 280)
        f_orig = ui_mod.pane_fit_factor(800, 400, *orig_pane)
        f_tex = ui_mod.pane_fit_factor(800, 400, *tex_pane)
        self.assertAlmostEqual(f_orig, 0.5)
        self.assertAlmostEqual(f_tex, 0.45)
        shared = ui_mod.shared_fit_factor(800, 400, (orig_pane, tex_pane))
        self.assertAlmostEqual(shared, min(f_orig, f_tex))
        self.assertEqual(ui_mod.contain_size(800, 400, 360, 280), (360, 180))
        self.assertEqual(ui_mod.contain_size(1600, 800, 240, 180), (240, 120))
        edge = ui_mod.fit_max_edge(800, 400, (orig_pane, tex_pane))
        self.assertEqual(edge, 360)
        bw, bh = ui_mod._preview_base_size(800, 400, edge)
        self.assertEqual((bw, bh), (360, 180))
        self.assertLessEqual(bw, orig_pane[0])
        self.assertLessEqual(bh, orig_pane[1])
        self.assertLessEqual(bw, tex_pane[0])
        self.assertLessEqual(bh, tex_pane[1])
        self.assertEqual(ui_mod._view_zoom_size(800, 400, 2.0, edge), (720, 360))
        self.assertEqual(ui_mod._view_zoom_size(800, 400, 8.0, edge), (2880, 1440))
        self.assertEqual(ui_mod.fit_max_edge(800, 400, ((1, 1), (1, 1))), 560)
        tool_edge = ui_mod.fit_max_edge(1260, 1260, ((500, 400),), fallback=1260)
        self.assertEqual(tool_edge, 400)

    def test_shared_pane_and_letterbox_match_for_square_in_rect(self) -> None:
        """Square source in a wide pane: one dest box, centered, same on both sides."""
        import wallpaper_recolor.ui.app as ui_mod

        self.assertEqual(ui_mod.shared_pane_size(((500, 400), (370, 400))), (370, 400))
        self.assertIsNone(ui_mod.shared_pane_size(((1, 1), (8, 8))))
        self.assertEqual(ui_mod.letterbox_xy(300, 300, 400, 300), (50, 0))
        self.assertEqual(ui_mod.letterbox_xy(300, 300, 370, 400), (35, 50))
        edge = ui_mod.fit_max_edge(1600, 1600, ((500, 400), (370, 400)))
        self.assertEqual(edge, 370)
        bw, bh = ui_mod._preview_base_size(1600, 1600, edge)
        self.assertEqual((bw, bh), (370, 370))
        self.assertEqual(ui_mod.letterbox_xy(bw, bh, 370, 400), (0, 15))

    def test_square_preview_photos_match_in_rectangular_pane(self) -> None:
        """Square work image: Original and Result PhotoImages share W×H in a wide pane."""
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            work = Image.new("RGB", (1600, 1600), (40, 50, 60))
            app.work_image = work
            app.source_image = work
            pane = (400, 300)

            def _stub_panes(host, _pane=pane):
                if host in (app.orig_zoom_host, app.tex_zoom_host):
                    return _pane
                return (640, 480)

            app._fit_pane_size = _stub_panes  # type: ignore[method-assign]
            app.rebuild_ranges()
            edge = ui_mod.fit_max_edge(1600, 1600, (pane, pane))
            fit_w, fit_h = ui_mod._preview_base_size(1600, 1600, edge)
            self.assertEqual((fit_w, fit_h), (300, 300))
            self.assertEqual(app._orig_photo.width(), app._tex_photo.width())
            self.assertEqual(app._orig_photo.height(), app._tex_photo.height())
            self.assertEqual(
                (app._orig_photo.width(), app._orig_photo.height()), (fit_w, fit_h)
            )
            self.assertEqual(
                (app._tex_photo.width(), app._tex_photo.height()), (fit_w, fit_h)
            )
            comp = app.orig_host.master
            c0 = comp.grid_columnconfigure(0)
            c1 = comp.grid_columnconfigure(1)
            self.assertEqual(c0.get("uniform"), "preview")
            self.assertEqual(c1.get("uniform"), "preview")
            self.assertEqual(int(c0.get("weight") or 0), 1)
            self.assertEqual(int(c1.get("weight") or 0), 1)
        finally:
            root.destroy()

    def test_fit_contains_full_image_and_pointer_tools(self) -> None:
        """Fit 100% contain-scales into the dest pane; Grab Move changes crop X/Y only."""
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            self.assertEqual(list(app.tools_combo.cget("values")), ["View Move", "Grab Move"])
            self.assertEqual(app.pointer_tool.get(), ui_mod.TOOL_VIEW_MOVE)
            app._rebuild_tools_menu()
            self.assertEqual(_menu_labels(app.tools_menu), ["View Move", "Grab Move"])

            work = Image.new("RGB", (1600, 800), (10, 20, 30))
            app.work_image = work
            app.source_image = work
            pane = (240, 180)

            def _stub_panes(host, _pane=pane):
                if host in (app.orig_zoom_host, app.tex_zoom_host):
                    return _pane
                return (640, 480)

            app._fit_pane_size = _stub_panes  # type: ignore[method-assign]
            app.rebuild_ranges()
            app._reset_preview_zoom()
            fit_w, fit_h = ui_mod.contain_size(1600, 800, *pane)
            self.assertEqual((fit_w, fit_h), (240, 120))
            self.assertEqual(
                (app._orig_photo.width(), app._orig_photo.height()), (fit_w, fit_h)
            )
            self.assertEqual(
                (app._tex_photo.width(), app._tex_photo.height()), (fit_w, fit_h)
            )
            self.assertLessEqual(app._orig_photo.width(), pane[0])
            self.assertLessEqual(app._orig_photo.height(), pane[1])
            self.assertFalse(app.orig_zoom_host._sb_x)
            self.assertFalse(app.orig_zoom_host._sb_y)
            self.assertFalse(app.tex_zoom_host._sb_x)
            self.assertFalse(app.tex_zoom_host._sb_y)

            zoom_before = app.preview_zoom.get()
            app._set_pointer_tool(ui_mod.TOOL_GRAB_MOVE)
            self.assertEqual(app.pointer_tool.get(), ui_mod.TOOL_GRAB_MOVE)
            self.assertEqual(app.pointer_tool_label.get(), "Grab Move")
            self.assertEqual(app.crop_x.get(), 0.0)
            app._nudge_grab_move(40, 10, host=app.orig_zoom_host)
            self.assertNotEqual(int(round(app.crop_x.get())), 0)
            self.assertAlmostEqual(app.preview_zoom.get(), zoom_before)
        finally:
            root.destroy()

    def test_composite_orig_result_same_crop_size_and_pan(self) -> None:
        """Original and Result share crop window, bitmap size, pan, and view zoom."""
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            work = Image.new("RGB", (80, 40), (10, 80, 30))
            for x in range(80):
                work.putpixel((x, 0), (255, 255, 255))
            for y in range(40):
                work.putpixel((0, y), (255, 255, 255))
            app.work_image = work
            app.source_image = work
            app.rebuild_ranges()
            self.assertIsNotNone(app._orig_pil)
            self.assertIsNotNone(app._tex_pil)
            self.assertEqual(app._orig_pil.size, app._tex_pil.size)
            self.assertEqual(app._orig_pil.size, (80, 40))
            orig_rgb = np.asarray(app._orig_pil.convert("RGB"))
            tex_rgb = np.asarray(app._tex_pil.convert("RGB"))
            self.assertEqual(tex_rgb.shape[:2], orig_rgb.shape[:2])
            self.assertTrue(np.all(orig_rgb[0, :, 0] > 200), "top white border on Original")
            self.assertTrue(np.all(orig_rgb[:, 0, 0] > 200), "left white border on Original")
            self.assertEqual(app._orig_photo.width(), app._tex_photo.width())
            self.assertEqual(app._orig_photo.height(), app._tex_photo.height())

            app.orig_zoom_host.set_pan(18, 7)
            self.assertEqual(app.tex_zoom_host._pan_x, app.orig_zoom_host._pan_x)
            self.assertEqual(app.tex_zoom_host._pan_y, app.orig_zoom_host._pan_y)
            app.preview_zoom.set(200.0)
            app._on_preview_zoom_slider("200")
            self.assertEqual(app._orig_photo.width(), app._tex_photo.width())
            self.assertEqual(app._orig_photo.height(), app._tex_photo.height())
            self.assertEqual(app.orig_zoom_host._pan_x, app.tex_zoom_host._pan_x)
            self.assertEqual(app.orig_zoom_host._pan_y, app.tex_zoom_host._pan_y)
        finally:
            root.destroy()


class TestToneAndEyes(unittest.TestCase):
    """Color & lighting identity at 0; range eyes knockout to checker."""
    def test_neutral_tone_is_identity(self) -> None:
        from wallpaper_recolor.color.tone import apply_tone_rgb

        rng = np.random.default_rng(2)
        rgb = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
        out = apply_tone_rgb(rgb, 0.0, 0.0, 0.0)
        np.testing.assert_array_equal(out, rgb)
        out_rgb = apply_tone_rgb(rgb, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        np.testing.assert_array_equal(out_rgb, rgb)

    def test_positive_darks_strengthens_shadows(self) -> None:
        from wallpaper_recolor.color.tone import apply_tone_rgb

        rgb = np.full((8, 8, 3), 80, dtype=np.uint8)
        ident = apply_tone_rgb(rgb, 0.0, 0.0, 0.0)
        plus = apply_tone_rgb(rgb, 0.5, 0.0, 0.0)
        minus = apply_tone_rgb(rgb, -0.5, 0.0, 0.0)
        np.testing.assert_array_equal(ident, rgb)
        self.assertLess(float(plus.mean()), float(rgb.mean()))
        self.assertGreater(float(minus.mean()), float(rgb.mean()))
        self.assertLess(float(plus.mean()), float(minus.mean()))

    def test_positive_lights_lifts_highlights(self) -> None:
        from wallpaper_recolor.color.tone import apply_tone_rgb

        rgb = np.full((8, 8, 3), 200, dtype=np.uint8)
        ident = apply_tone_rgb(rgb, 0.0, 0.0, 0.0)
        plus = apply_tone_rgb(rgb, 0.0, 0.5, 0.0)
        minus = apply_tone_rgb(rgb, 0.0, -0.5, 0.0)
        np.testing.assert_array_equal(ident, rgb)
        self.assertGreater(float(plus.mean()), float(rgb.mean()))
        self.assertLess(float(minus.mean()), float(rgb.mean()))
        self.assertGreater(float(plus.mean()), float(minus.mean()))

    def test_positive_brightness_raises_luma(self) -> None:
        from wallpaper_recolor.color.tone import apply_tone_rgb

        rgb = np.full((8, 8, 3), 120, dtype=np.uint8)
        ident = apply_tone_rgb(rgb, 0.0, 0.0, 0.0)
        plus = apply_tone_rgb(rgb, 0.0, 0.0, 0.4)
        minus = apply_tone_rgb(rgb, 0.0, 0.0, -0.4)
        np.testing.assert_array_equal(ident, rgb)
        self.assertGreater(float(plus.mean()), float(rgb.mean()))
        self.assertLess(float(minus.mean()), float(rgb.mean()))
        self.assertGreater(float(plus.mean()), float(minus.mean()))

    def test_neutral_contrast_exposure_is_identity(self) -> None:
        from wallpaper_recolor.color.tone import apply_tone_rgb

        rng = np.random.default_rng(11)
        rgb = rng.integers(0, 256, (20, 24, 3), dtype=np.uint8)
        out = apply_tone_rgb(rgb, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        np.testing.assert_array_equal(out, rgb)

    def test_positive_contrast_increases_luma_std(self) -> None:
        from wallpaper_recolor.color.color_ranges import LUMA_B, LUMA_G, LUMA_R
        from wallpaper_recolor.color.tone import apply_tone_rgb

        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        rgb[:8] = 80
        rgb[8:] = 180
        luma = LUMA_R * rgb[..., 0] + LUMA_G * rgb[..., 1] + LUMA_B * rgb[..., 2]
        plus = apply_tone_rgb(rgb, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0)
        luma_plus = LUMA_R * plus[..., 0] + LUMA_G * plus[..., 1] + LUMA_B * plus[..., 2]
        self.assertGreater(float(luma_plus.std()), float(luma.std()))

    def test_positive_exposure_raises_mean_luma(self) -> None:
        from wallpaper_recolor.color.color_ranges import LUMA_B, LUMA_G, LUMA_R
        from wallpaper_recolor.color.tone import apply_tone_rgb

        rgb = np.full((8, 8, 3), 120, dtype=np.uint8)
        luma = LUMA_R * rgb[..., 0] + LUMA_G * rgb[..., 1] + LUMA_B * rgb[..., 2]
        plus = apply_tone_rgb(rgb, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4)
        minus = apply_tone_rgb(rgb, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4)
        luma_plus = LUMA_R * plus[..., 0] + LUMA_G * plus[..., 1] + LUMA_B * plus[..., 2]
        luma_minus = LUMA_R * minus[..., 0] + LUMA_G * minus[..., 1] + LUMA_B * minus[..., 2]
        self.assertGreater(float(luma_plus.mean()), float(luma.mean()))
        self.assertLess(float(luma_minus.mean()), float(luma.mean()))

    def test_lights_rgb_zero_is_identity_and_size(self) -> None:
        from wallpaper_recolor.color.tone import apply_tone_rgb

        rng = np.random.default_rng(3)
        rgb = rng.integers(0, 256, (18, 22, 3), dtype=np.uint8)
        graded = apply_tone_rgb(rgb, 0.2, -0.15, 0.0)
        same = apply_tone_rgb(rgb, 0.2, -0.15, 0.0, 0.0, 0.0, 0.0)
        np.testing.assert_array_equal(same, graded)
        self.assertEqual(same.shape, rgb.shape)

    def test_lights_reds_raises_r_more_in_highlights(self) -> None:
        from wallpaper_recolor.color.tone import apply_tone_rgb

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        rgb[:4] = 40
        rgb[4:] = 220
        out = apply_tone_rgb(rgb, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        pulled = apply_tone_rgb(rgb, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0)
        self.assertEqual(out.shape, rgb.shape)
        dark_d_r = int(out[1, 1, 0]) - 40
        light_d_r = int(out[6, 1, 0]) - 220
        self.assertGreater(light_d_r, dark_d_r)
        self.assertGreater(light_d_r, 8)
        self.assertEqual(int(out[1, 1, 0]), 40)
        self.assertLess(int(pulled[6, 1, 0]), 220)
        self.assertGreater(int(out[6, 1, 0]), int(pulled[6, 1, 0]))

    def test_cmy_zero_is_identity(self) -> None:
        from wallpaper_recolor.color.tone import apply_tone_rgb

        rng = np.random.default_rng(13)
        rgb = rng.integers(0, 256, (20, 24, 3), dtype=np.uint8)
        out = apply_tone_rgb(
            rgb, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )
        np.testing.assert_array_equal(out, rgb)
        kw = apply_tone_rgb(rgb, lights_cyan=0.0, darks_magenta=0.0)
        np.testing.assert_array_equal(kw, rgb)

    def test_lights_cyan_pulls_r_more_in_highlights(self) -> None:
        from wallpaper_recolor.color.tone import apply_tone_rgb

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        rgb[:4] = 40
        rgb[4:] = 220
        out = apply_tone_rgb(rgb, lights_cyan=1.0)
        mag = apply_tone_rgb(rgb, lights_magenta=1.0)
        yel = apply_tone_rgb(rgb, lights_yellow=1.0)
        reds = apply_tone_rgb(rgb, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        self.assertEqual(out.shape, rgb.shape)
        self.assertEqual(int(out[1, 1, 0]), 40)
        self.assertLess(int(out[6, 1, 0]), 220)
        dark_d_r = int(out[1, 1, 0]) - 40
        light_d_r = int(out[6, 1, 0]) - 220
        self.assertLess(light_d_r, dark_d_r)
        self.assertLess(light_d_r, -8)
        self.assertGreater(int(reds[6, 1, 0]), 220)
        self.assertLess(int(out[6, 1, 0]), int(reds[6, 1, 0]))
        self.assertLess(int(mag[6, 1, 1]), 220)
        self.assertEqual(int(mag[1, 1, 1]), 40)
        self.assertLess(int(yel[6, 1, 2]), 220)
        self.assertEqual(int(yel[1, 1, 2]), 40)

    def test_darks_cyan_pulls_r_more_in_shadows(self) -> None:
        from wallpaper_recolor.color.tone import apply_tone_rgb

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        rgb[:4] = 40
        rgb[4:] = 220
        out = apply_tone_rgb(rgb, darks_cyan=1.0)
        self.assertEqual(int(out[6, 1, 0]), 220)
        self.assertLess(int(out[1, 1, 0]), 40)
        dark_d_r = int(out[1, 1, 0]) - 40
        light_d_r = int(out[6, 1, 0]) - 220
        self.assertLess(dark_d_r, light_d_r)
        self.assertLess(dark_d_r, -4)

    def test_cmy_keeps_texture_grain(self) -> None:
        from wallpaper_recolor.color.layers import composites_from_map

        grain = np.linspace(40, 220, 32, dtype=np.uint8)
        rgb = np.stack([grain, grain * 0.9, grain * 0.8], axis=-1).astype(np.uint8)
        rgb = np.broadcast_to(rgb[None, :, :], (24, 32, 3)).copy()
        rgb[4:10, 4:12] = (20, 40, 80)
        im = Image.fromarray(rgb, mode="RGB")
        range_map = build_range_map(im, 2, SPLIT_COLOR_CLOSENESS)
        range_map.set_replacement(0, (40, 160, 80))
        range_map.set_replacement(1, (200, 30, 30))
        range_map.texture_strength = 1.0
        range_map.texture_enabled = True
        range_map.tone_lights_cyan = 0.6
        range_map.tone_darks_magenta = 0.5
        range_map.tone_balance_cyan = 0.4
        exact, tex = composites_from_map(range_map)
        exact_a = np.asarray(exact.convert("RGB"))
        tex_a = np.asarray(tex.convert("RGB"))
        self.assertFalse(np.array_equal(tex_a, exact_a))

    def test_gray_world_and_white_patch_gains(self) -> None:
        from wallpaper_recolor.color.tone import gray_world_gains, white_patch_gains

        red_cast = np.zeros((10, 10, 3), dtype=np.uint8)
        red_cast[:] = (180, 80, 80)
        gw = gray_world_gains(red_cast)
        self.assertLess(float(gw[0]), 1.0)
        self.assertGreater(float(gw[1]), 1.0)
        wp = white_patch_gains(red_cast, percentile=100)
        self.assertAlmostEqual(float(wp[0]), 1.0, places=5)
        self.assertGreater(float(wp[1]), 1.0)
        uneven = np.zeros((8, 8, 3), dtype=np.uint8)
        uneven[:] = (100, 150, 200)
        wp2 = white_patch_gains(uneven, percentile=100)
        self.assertGreater(float(wp2[0]), float(wp2[2]))

    def test_temperature_warms_red_and_identity_at_zero(self) -> None:
        from wallpaper_recolor.color.tone import apply_tone_rgb

        rgb = np.full((8, 8, 3), 120, dtype=np.uint8)
        ident = apply_tone_rgb(rgb, temperature=0.0, tint=0.0)
        np.testing.assert_array_equal(ident, rgb)
        warm = apply_tone_rgb(rgb, temperature=1.0)
        cool = apply_tone_rgb(rgb, temperature=-1.0)
        self.assertGreater(int(warm[0, 0, 0]), 120)
        self.assertLess(int(warm[0, 0, 2]), 120)
        self.assertLess(int(cool[0, 0, 0]), 120)
        self.assertGreater(int(cool[0, 0, 2]), 120)

    def test_balance_cyan_is_global_unlike_lights_cyan(self) -> None:
        from wallpaper_recolor.color.tone import apply_tone_from_map, apply_tone_rgb

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        rgb[:4] = 40
        rgb[4:] = 220
        regional = apply_tone_rgb(rgb, lights_cyan=1.0)
        global_c = apply_tone_rgb(rgb, balance_cyan=1.0)
        self.assertLess(int(global_c[1, 1, 0]), 40)
        self.assertEqual(int(regional[1, 1, 0]), 40)
        self.assertLess(int(global_c[6, 1, 0]), 220)
        im = Image.fromarray(rgb, mode="RGB")
        range_map = build_range_map(im, 2, SPLIT_COLOR_CLOSENESS)
        range_map.tone_lights_cyan = 1.0
        fallback = apply_tone_from_map(rgb, range_map)
        np.testing.assert_array_equal(fallback, regional)
        range_map.tone_balance_cyan = 1.0
        mapped = apply_tone_from_map(rgb, range_map)
        np.testing.assert_array_equal(mapped, global_c)

    def test_saturation_scales_chroma_not_luma(self) -> None:
        from wallpaper_recolor.color.color_ranges import LUMA_B, LUMA_G, LUMA_R
        from wallpaper_recolor.color.tone import apply_tone_rgb

        rgb = np.zeros((6, 6, 3), dtype=np.uint8)
        rgb[:] = (180, 40, 40)
        ident = apply_tone_rgb(rgb, saturation=0.0)
        np.testing.assert_array_equal(ident, rgb)
        gray = apply_tone_rgb(rgb, saturation=-1.0)
        punch = apply_tone_rgb(rgb, saturation=1.0)
        luma = LUMA_R * rgb[..., 0] + LUMA_G * rgb[..., 1] + LUMA_B * rgb[..., 2]
        luma_gray = LUMA_R * gray[..., 0] + LUMA_G * gray[..., 1] + LUMA_B * gray[..., 2]
        self.assertLess(abs(int(gray[0, 0, 0]) - int(gray[0, 0, 1])), 8)
        self.assertGreater(int(punch[0, 0, 0]) - int(punch[0, 0, 1]), int(rgb[0, 0, 0]) - int(rgb[0, 0, 1]))
        self.assertAlmostEqual(float(luma_gray.mean()), float(luma.mean()), delta=3.0)

    def test_gray_world_button_sets_temperature(self) -> None:
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            red = np.full((16, 16, 3), (200, 40, 40), dtype=np.uint8)
            im = Image.fromarray(red, mode="RGB")
            app.work_image = im
            app.source_image = im
            app.rebuild_ranges()
            app._on_gray_world()
            self.assertLess(app.temperature_pct.get(), -5.0)
            self.assertEqual(str(app.temperature_reset.winfo_manager()), "pack")
            self.assertAlmostEqual(app.range_map.tone_temperature, app.temperature_pct.get() / 100.0, places=4)
        finally:
            root.destroy()

    def test_snapshot_keeps_tone_and_visibility(self) -> None:
        im = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB")
        range_map = build_range_map(im, 2, SPLIT_COLOR_CLOSENESS)
        range_map.tone_darks = -0.2
        range_map.tone_lights = 0.1
        range_map.tone_brightness = 0.05
        range_map.tone_contrast = 0.3
        range_map.tone_exposure = -0.15
        range_map.tone_lights_reds = 0.25
        range_map.tone_lights_greens = -0.1
        range_map.tone_lights_blues = 0.05
        range_map.tone_lights_cyan = 0.35
        range_map.tone_lights_magenta = -0.2
        range_map.tone_lights_yellow = 0.15
        range_map.tone_darks_cyan = -0.3
        range_map.tone_darks_magenta = 0.4
        range_map.tone_darks_yellow = -0.05
        range_map.tone_temperature = 0.22
        range_map.tone_tint = -0.18
        range_map.tone_saturation = 0.4
        range_map.tone_balance_cyan = 0.12
        range_map.tone_balance_magenta = -0.08
        range_map.tone_balance_yellow = 0.05
        range_map.texture_enabled = False
        range_map.ranges[1].visible = False
        snap = snapshot_assignment(range_map)
        self.assertAlmostEqual(snap.tone_darks, -0.2)
        self.assertAlmostEqual(snap.tone_lights, 0.1)
        self.assertAlmostEqual(snap.tone_contrast, 0.3)
        self.assertAlmostEqual(snap.tone_exposure, -0.15)
        self.assertAlmostEqual(snap.tone_lights_reds, 0.25)
        self.assertAlmostEqual(snap.tone_lights_greens, -0.1)
        self.assertAlmostEqual(snap.tone_lights_blues, 0.05)
        self.assertAlmostEqual(snap.tone_lights_cyan, 0.35)
        self.assertAlmostEqual(snap.tone_lights_magenta, -0.2)
        self.assertAlmostEqual(snap.tone_lights_yellow, 0.15)
        self.assertAlmostEqual(snap.tone_darks_cyan, -0.3)
        self.assertAlmostEqual(snap.tone_darks_magenta, 0.4)
        self.assertAlmostEqual(snap.tone_darks_yellow, -0.05)
        self.assertAlmostEqual(snap.tone_temperature, 0.22)
        self.assertAlmostEqual(snap.tone_tint, -0.18)
        self.assertAlmostEqual(snap.tone_saturation, 0.4)
        self.assertAlmostEqual(snap.tone_balance_cyan, 0.12)
        self.assertAlmostEqual(snap.tone_balance_magenta, -0.08)
        self.assertAlmostEqual(snap.tone_balance_yellow, 0.05)
        self.assertFalse(snap.texture_enabled)
        self.assertFalse(snap.ranges[1].visible)

    def test_estimate_tone_amounts_recovers_darks_lights(self) -> None:
        from wallpaper_recolor.color.tone import apply_tone_rgb, estimate_tone_amounts

        rng = np.random.default_rng(9)
        rgb = rng.integers(20, 230, (40, 48, 3), dtype=np.uint8)
        dst = apply_tone_rgb(rgb, 0.35, -0.25, 0.0)
        darks, lights = estimate_tone_amounts(rgb, dst)
        self.assertGreater(darks, 0.15)
        self.assertLess(lights, -0.10)

    def test_normalize_lighting_skips_original_preview(self) -> None:
        """Normalize lighting grades Result only; Original stays the source crop."""
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod
        from wallpaper_recolor.color.layers import live_composite_from_map

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            yy, xx = np.mgrid[0:32, 0:48]
            luma = np.clip(40.0 + 180.0 * (xx / 47.0), 0, 255)
            rgb = np.stack((luma, luma * 0.9, luma * 0.8), axis=-1).astype(np.uint8)
            im = Image.fromarray(rgb, mode="RGB")
            app.work_image = im
            app.source_image = im
            app.rebuild_ranges()
            self.assertIsNotNone(app._orig_pil)
            self.assertIsNotNone(app._work_live)
            orig_before = np.asarray(app._orig_pil.convert("RGB")).copy()
            live_before = np.asarray(app._work_live.convert("RGB")).copy()
            title_before = app.orig_title.get()
            app._on_tess_normalize()
            self.assertTrue(bool(app.tess_normalize.get()))
            self.assertGreater(abs(app.darks_pct.get()) + abs(app.lights_pct.get()), 2.0)
            orig_after = np.asarray(app._orig_pil.convert("RGB"))
            live_after = np.asarray(app._work_live.convert("RGB"))
            np.testing.assert_array_equal(orig_after, orig_before)
            self.assertFalse(np.array_equal(live_after, live_before))
            self.assertFalse(np.array_equal(orig_after, live_after))
            self.assertEqual(app.orig_title.get(), title_before)
            assert app.range_map is not None
            saved = np.asarray(live_composite_from_map(app.range_map).convert("RGB"))
            self.assertFalse(np.array_equal(saved, rgb))
        finally:
            root.destroy()

    def test_texture_eye_off_matches_exact(self) -> None:
        from wallpaper_recolor.color.layers import exact_rgb, live_composite_from_map, presentation_rgb

        im = Image.fromarray(np.full((12, 12, 3), 80, dtype=np.uint8), mode="RGB")
        range_map = build_range_map(im, 2, SPLIT_COLOR_CLOSENESS)
        range_map.set_replacement(0, (40, 160, 80))
        range_map.set_replacement(1, (200, 30, 30))
        range_map.texture_strength = 1.0
        range_map.texture_enabled = False
        assert range_map.rgb is not None and range_map.labels is not None
        live = np.asarray(live_composite_from_map(range_map).convert("RGB"))
        exact = exact_rgb(range_map.rgb, range_map.labels, range_map.ranges)
        np.testing.assert_array_equal(live, exact)
        woven = presentation_rgb(
            range_map.rgb, range_map.labels, range_map.ranges, strength=1.0
        )
        self.assertFalse(np.array_equal(woven, exact))

    def test_hidden_range_knocks_out_pixels(self) -> None:
        from wallpaper_recolor.color.layers import live_composite_from_map

        arr = np.zeros((16, 16, 3), dtype=np.uint8)
        arr[:8] = (10, 20, 30)
        arr[8:] = (200, 210, 220)
        im = Image.fromarray(arr, mode="RGB")
        range_map = build_range_map(im, 2, SPLIT_COLOR_CLOSENESS)
        range_map.set_replacement(0, (255, 0, 0))
        range_map.set_replacement(1, (0, 255, 0))
        range_map.texture_enabled = False
        range_map.ranges[0].visible = False
        live_im = live_composite_from_map(range_map)
        self.assertEqual(live_im.mode, "RGBA")
        live = np.asarray(live_im)
        assert range_map.labels is not None and range_map.rgb is not None
        hidden = range_map.labels == 0
        np.testing.assert_array_equal(live[hidden, 3], 0)
        rgb_hidden = live[hidden, :3]
        self.assertFalse(np.array_equal(rgb_hidden, range_map.rgb[hidden]))
        self.assertFalse(np.all(rgb_hidden == np.array([255, 0, 0], dtype=np.uint8)))
        shown = range_map.labels == 1
        np.testing.assert_array_equal(live[shown, 3], 255)
        self.assertTrue(np.all(live[shown, :3] == np.array([0, 255, 0], dtype=np.uint8)))

    def test_preview_blit_over_checker_not_source_rgb(self) -> None:
        """Transparent holes composite onto a modest checker, not source RGB or black."""
        import wallpaper_recolor.ui.app as ui_mod
        from wallpaper_recolor.layers.stack import (
            CHECKER_LIGHT,
            CHECKER_MID,
            CHECKER_TILE_PX,
            composite_over_checker,
            flatten_rgb_or_keep_alpha,
        )

        self.assertGreaterEqual(CHECKER_TILE_PX, 8)
        self.assertLessEqual(CHECKER_TILE_PX, 16)
        rgba = Image.new("RGBA", (32, 32), (255, 0, 0, 0))
        ink = Image.new("RGBA", (16, 16), (10, 20, 30, 255))
        rgba.paste(ink, (16, 16))
        out = composite_over_checker(rgba)
        self.assertEqual(out.mode, "RGB")
        hole_px = out.getpixel((0, 0))
        self.assertIn(hole_px, (CHECKER_LIGHT, CHECKER_MID))
        self.assertNotEqual(hole_px, (255, 0, 0))
        self.assertNotEqual(hole_px, (10, 20, 30))
        self.assertNotEqual(hole_px, (0, 0, 0))
        self.assertEqual(out.getpixel((24, 24)), (10, 20, 30))
        self.assertNotEqual(out.getpixel((0, 0)), out.getpixel((CHECKER_TILE_PX, 0)))
        self.assertEqual(out.getpixel((0, 0)), out.getpixel((CHECKER_TILE_PX, CHECKER_TILE_PX)))
        opaque = Image.new("RGB", (8, 8), (1, 2, 3))
        self.assertIs(composite_over_checker(opaque), opaque)
        kept = flatten_rgb_or_keep_alpha(rgba)
        self.assertEqual(kept.mode, "RGBA")
        self.assertEqual(kept.getpixel((0, 0))[3], 0)
        self.assertNotEqual(kept.getpixel((0, 0))[:3], hole_px)
        zoom_src = inspect.getsource(ui_mod.WallpaperRecolorApp._apply_preview_zoom)
        self.assertIn("composite_over_checker", zoom_src)
        preview_src = inspect.getsource(ui_mod.WallpaperRecolorApp._preview_pils)
        self.assertNotIn("flatten_rgb", preview_src)

    def test_typed_percent_steals_from_right_neighbor_only(self) -> None:
        from wallpaper_recolor.color.color_ranges import MIN_COVERAGE, set_range_weight, steal_from_adjacent

        arr = np.zeros((12, 36, 3), dtype=np.uint8)
        arr[:, :12] = 20
        arr[:, 12:24] = 120
        arr[:, 24:] = 220
        im = Image.fromarray(arr, mode="RGB")
        range_map = build_range_map(im, 3, SPLIT_EQUAL_PIXELS)
        before = range_map.weights()
        self.assertAlmostEqual(before[2], 1.0 / 3.0, places=5)
        set_range_weight(range_map, 0, 0.40)
        weights = range_map.weights()
        self.assertAlmostEqual(weights[0], 0.40, places=5)
        self.assertAlmostEqual(weights[2], before[2], places=5)
        self.assertAlmostEqual(sum(weights), 1.0, places=5)
        for w in weights:
            self.assertGreaterEqual(w, MIN_COVERAGE - 1e-9)

        moved = steal_from_adjacent([0.30, 0.40, 0.30], 0, 0.35)
        self.assertAlmostEqual(moved[0], 0.35, places=5)
        self.assertAlmostEqual(moved[2], 0.30, places=5)
        self.assertAlmostEqual(sum(moved), 1.0, places=5)
        last = steal_from_adjacent([0.30, 0.40, 0.30], 2, 0.45)
        self.assertAlmostEqual(last[2], 0.45, places=5)
        self.assertAlmostEqual(last[0], 0.30, places=5)

    def test_color_closeness_weight_steals_lab_nearest_not_far(self) -> None:
        """Changing cluster A % must not grow a far cluster C when B is Lab-nearer."""
        from wallpaper_recolor.color.color_ranges import (
            MIN_COVERAGE,
            lab_nearest_other,
            set_range_weight,
        )

        arr = np.zeros((16, 48, 3), dtype=np.uint8)
        arr[:, :16] = (220, 20, 20)
        arr[:, 16:32] = (20, 180, 20)
        arr[:, 32:] = (20, 20, 210)
        im = Image.fromarray(arr, mode="RGB")
        range_map = build_range_map(im, 3, SPLIT_COLOR_CLOSENESS)
        self.assertEqual(len(range_map.ranges), 3)
        assert range_map.labels is not None and range_map.centers is not None

        def _nearest_mean(target: tuple[int, int, int]) -> int:
            best_i = 0
            best_d = 1e18
            for i, band in enumerate(range_map.ranges):
                d = sum((a - b) ** 2 for a, b in zip(band.mean_rgb, target))
                if d < best_d:
                    best_d = d
                    best_i = i
            return best_i

        red_i = _nearest_mean((220, 20, 20))
        green_i = _nearest_mean((20, 180, 20))
        blue_i = _nearest_mean((20, 20, 210))
        self.assertEqual(len({red_i, green_i, blue_i}), 3)
        neighbor = lab_nearest_other(range_map.centers, red_i)
        self.assertIn(neighbor, (green_i, blue_i))
        far_i = blue_i if neighbor == green_i else green_i
        before_far = int(np.count_nonzero(range_map.labels == far_i))
        before_near = int(np.count_nonzero(range_map.labels == neighbor))
        before_w = range_map.weights()
        set_range_weight(range_map, red_i, before_w[red_i] + 0.15)
        after_w = range_map.weights()
        self.assertAlmostEqual(after_w[far_i], before_w[far_i], places=5)
        self.assertGreater(after_w[red_i], before_w[red_i])
        self.assertLess(after_w[neighbor], before_w[neighbor])
        self.assertEqual(int(np.count_nonzero(range_map.labels == far_i)), before_far)
        self.assertLessEqual(int(np.count_nonzero(range_map.labels == neighbor)), before_near)
        for w in after_w:
            self.assertGreaterEqual(w, MIN_COVERAGE - 1e-9)

    def test_lab_a_split_separates_red_and_green(self) -> None:
        arr = np.zeros((8, 16, 3), dtype=np.uint8)
        arr[:, :8] = (220, 30, 30)
        arr[:, 8:] = (30, 180, 30)
        im = Image.fromarray(arr, mode="RGB")
        range_map = build_range_map(im, 2, SPLIT_LAB_A_EQUAL)
        self.assertEqual(range_map.split_method, SPLIT_LAB_A_EQUAL)
        assert range_map.labels is not None
        self.assertNotEqual(int(range_map.labels[0, 0]), int(range_map.labels[0, 12]))
        kmeans = build_range_map(im, 2, SPLIT_COLOR_CLOSENESS)
        self.assertEqual(kmeans.split_method, SPLIT_COLOR_CLOSENESS)
        self.assertIsNotNone(kmeans.centers)

    def test_lab_l_split_differs_from_rec709_luma(self) -> None:
        arr = np.zeros((8, 16, 3), dtype=np.uint8)
        arr[:, :8] = (255, 0, 0)
        arr[:, 8:] = (0, 80, 0)
        im = Image.fromarray(arr, mode="RGB")
        luma_map = build_range_map(im, 2, SPLIT_EQUAL_LIGHTNESS)
        lab_map = build_range_map(im, 2, SPLIT_LAB_L_EQUAL)
        assert luma_map.labels is not None and lab_map.labels is not None
        luma_pair = (int(luma_map.labels[0, 0]), int(luma_map.labels[0, 12]))
        lab_pair = (int(lab_map.labels[0, 0]), int(lab_map.labels[0, 12]))
        self.assertNotEqual(luma_pair[0], luma_pair[1])
        self.assertNotEqual(lab_pair[0], lab_pair[1])
        self.assertNotEqual(luma_pair, lab_pair)

    def test_luma_start_excludes_darker_pixels(self) -> None:
        arr = np.zeros((4, 10, 3), dtype=np.uint8)
        arr[:, :5] = 20
        arr[:, 5:] = 200
        im = Image.fromarray(arr, mode="RGB")
        range_map = build_range_map(im, 2, SPLIT_EQUAL_LIGHTNESS, bin_start=80.0)
        assert range_map.labels is not None
        self.assertEqual(int(range_map.labels[0, 0]), -1)
        self.assertGreaterEqual(int(range_map.labels[0, 8]), 0)
        self.assertGreaterEqual(float(range_map.edges[0]), 80.0 - 1e-3)

    def test_lab_a_start_excludes_lower_a(self) -> None:
        arr = np.zeros((4, 12, 3), dtype=np.uint8)
        arr[:, :6] = (30, 180, 30)
        arr[:, 6:] = (220, 30, 30)
        im = Image.fromarray(arr, mode="RGB")
        range_map = build_range_map(im, 2, SPLIT_LAB_A_EQUAL, bin_start=10.0)
        assert range_map.labels is not None
        self.assertEqual(int(range_map.labels[0, 1]), -1)
        self.assertGreaterEqual(int(range_map.labels[0, 10]), 0)

    def test_min_coverage_floor_prevents_zero_range(self) -> None:
        arr = np.linspace(0, 255, 32, dtype=np.uint8)
        rgb = np.stack([np.tile(arr, (16, 1))] * 3, axis=-1)
        im = Image.fromarray(rgb, mode="RGB")
        range_map = build_range_map(im, 3, SPLIT_EQUAL_PIXELS, min_coverage=0.03)
        set_range_weight(range_map, 0, 0.0)
        for w in range_map.weights():
            self.assertGreaterEqual(w, 0.03 - 1e-9)

    def test_tone_knob_vertical_drag_exponential(self) -> None:
        import tkinter as tk
        from types import SimpleNamespace

        import wallpaper_recolor.ui.app as ui_mod

        root = tk.Tk()
        root.withdraw()
        try:
            var = tk.DoubleVar(value=0.0)
            begins: list[int] = []
            ends: list[int] = []
            knob = ui_mod.ToneKnob(
                root,
                var,
                on_change=None,
                on_begin=lambda _e: begins.append(1),
                on_end=lambda _e: ends.append(1),
            )
            g_near = ui_mod.tone_knob_gain(15.0)
            g_far = ui_mod.tone_knob_gain(110.0)
            self.assertGreater(g_far, g_near)
            ratio = ui_mod.tone_knob_gain(150.0) / ui_mod.tone_knob_gain(50.0)
            expected = ui_mod.TONE_KNOB_GROWTH ** (
                (150.0 - 50.0) / ui_mod.TONE_KNOB_REF_PX
            )
            self.assertAlmostEqual(ratio, expected, places=6)

            knob.apply_drag_delta(-20.0, 15.0)
            self.assertGreater(var.get(), 0.0)
            up_near = var.get()
            var.set(0.0)
            knob._frac = 0.0
            knob.apply_drag_delta(20.0, 15.0)
            self.assertLess(var.get(), 0.0)
            var.set(0.0)
            knob._frac = 0.0
            knob.apply_drag_delta(-20.0, 110.0)
            self.assertGreater(var.get(), up_near)

            var.set(0.0)
            knob._frac = 0.0
            knob.apply_drag_delta(0.0, 40.0, dx_px=20.0)
            self.assertGreater(var.get(), 0.0)
            var.set(0.0)
            knob._frac = 0.0
            knob.apply_drag_delta(0.0, 40.0, dx_px=-20.0)
            self.assertLess(var.get(), 0.0)

            var.set(0.0)
            knob._frac = 0.0
            press = SimpleNamespace(x=11, y=11, x_root=200, y_root=200)
            knob._on_press(press)
            self.assertEqual(begins, [1])
            self.assertAlmostEqual(var.get(), 0.0)
            knob._handle_motion(SimpleNamespace(x_root=200, y_root=180))
            self.assertGreater(var.get(), 0.0)
            knob._on_release(SimpleNamespace(x_root=200, y_root=180))
            self.assertEqual(ends, [1])
            self.assertFalse(knob._dragging)
            press_src = inspect.getsource(ui_mod.ToneKnob._on_press)
            self.assertNotIn("_t_from_xy", press_src)
            self.assertNotIn("_apply_pointer", press_src)
        finally:
            root.destroy()


class TestLayoutHistory(unittest.TestCase):
    """Dock columns, right sash, undo stack, Composite tab, close prompt."""
    def test_ui_import_layout_eyes_and_undo(self) -> None:
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod
        from wallpaper_recolor.ui import HISTORY_LIMIT, run

        self.assertTrue(callable(run))
        self.assertEqual(HISTORY_LIMIT, 20)
        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            self.assertEqual(str(app.body_paned.cget("orient")), "horizontal")
            self.assertEqual(len(app.body_paned.panes()), 2)
            self.assertEqual(app.left_column.panels, [app.preview_panel, app.coverage_panel])
            self.assertEqual(
                app.right_column.panels,
                [
                    app.wheel_panel,
                    app.texture_panel,
                    app.tone_panel,
                    app.scale_panel,
                    app.crop_panel,
                ],
            )
            self.assertIs(app.wheel_panel.column, app.right_column)
            self.assertIn(app.preview_panel, app.left_column.panels)
            self.assertIn(app.coverage_panel, app.left_column.panels)
            self.assertIn(app.tone_panel, app.right_column.panels)
            self.assertIn(app.scale_panel, app.right_column.panels)
            self.assertIn(app.crop_panel, app.right_column.panels)
            self.assertTrue(app.tess_panel.expanded)
            self.assertTrue(app.layers_panel.expanded)
            self.assertTrue(app.labels_panel.expanded)
            self.assertFalse(app.tess_panel.hidden)
            self.assertFalse(app.layers_panel.hidden)
            self.assertFalse(app.labels_panel.hidden)
            self.assertIn(app.tess_panel, app.right_bottom_column.panels)
            self.assertIn(app.layers_panel, app.right_bottom_column.panels)
            self.assertIn(app.labels_panel, app.right_bottom_column.panels)
            self.assertFalse(hasattr(app, "lighting_panel"))
            self.assertTrue(_widget_under(app.tess_normalize_btn, app.tone_panel))
            self.assertTrue(_widget_under(app.tess_normalize_reset, app.tone_panel))
            self.assertEqual(app.tess_normalize_btn.winfo_class(), "TButton")
            self.assertFalse(hasattr(app, "tess_normalize_chk"))
            layout_src = inspect.getsource(ui_mod.WallpaperRecolorApp._build_layout)
            self.assertNotIn("Click a range (bar or chip)", layout_src)
            self.assertNotIn("save_mode_combo", layout_src)
            self.assertFalse(hasattr(app, "save_mode_combo"))
            self.assertFalse(hasattr(ui_mod, "SAVE_MODE_TEXTURE"))
            self.assertFalse(hasattr(ui_mod, "SAVE_MODES"))
            from wallpaper_recolor.ui.coverage_bar import CoverageBar

            bar_src = inspect.getsource(CoverageBar.__init__)
            self.assertNotIn("drag a divider", bar_src)
            self.assertNotIn("Palette preview", bar_src)
            self.assertIsNotNone(app.texture_eye)
            self.assertTrue(hasattr(app.texture_eye, "set_shown"))
            self.assertTrue(app.texture_eye.shown)
            self.assertTrue(app.texture_enabled.get())
            self.assertAlmostEqual(app.crop_x.get(), 0.0)
            self.assertAlmostEqual(app.crop_y.get(), 0.0)
            self.assertAlmostEqual(app.crop_zoom.get(), 1.0)
            self.assertEqual(str(app.crop_x_reset.winfo_manager()), "")
            self.assertEqual(str(app.crop_zoom_reset.winfo_manager()), "")
            self.assertNotEqual(str(app._zoom_minus_icon.cget("image")), "")
            self.assertNotEqual(str(app._zoom_plus_icon.cget("image")), "")
            self.assertAlmostEqual(app.darks_pct.get(), 0.0)
            self.assertAlmostEqual(app.lights_pct.get(), 0.0)
            self.assertAlmostEqual(app.brightness_pct.get(), 0.0)
            self.assertAlmostEqual(app.contrast_pct.get(), 0.0)
            self.assertAlmostEqual(app.exposure_pct.get(), 0.0)
            self.assertAlmostEqual(app.lights_reds_pct.get(), 0.0)
            self.assertAlmostEqual(app.lights_greens_pct.get(), 0.0)
            self.assertAlmostEqual(app.lights_blues_pct.get(), 0.0)
            self.assertAlmostEqual(app.temperature_pct.get(), 0.0)
            self.assertAlmostEqual(app.tint_pct.get(), 0.0)
            self.assertAlmostEqual(app.saturation_pct.get(), 0.0)
            self.assertAlmostEqual(app.balance_cyan_pct.get(), 0.0)
            self.assertAlmostEqual(app.balance_magenta_pct.get(), 0.0)
            self.assertAlmostEqual(app.balance_yellow_pct.get(), 0.0)
            self.assertEqual(app.tone_panel.panel_title, "Color & lighting")
            self.assertTrue(_widget_under(app.lights_reds_spin, app.tone_panel))
            self.assertTrue(_widget_under(app.lights_greens_spin, app.tone_panel))
            self.assertTrue(_widget_under(app.lights_blues_spin, app.tone_panel))
            self.assertTrue(_widget_under(app.balance_cyan_spin, app.tone_panel))
            self.assertTrue(_widget_under(app.temperature_spin, app.tone_panel))
            self.assertTrue(_widget_under(app.saturation_spin, app.tone_panel))
            self.assertTrue(_widget_under(app.darks_spin, app.tone_panel))
            self.assertTrue(_widget_under(app.exposure_spin, app.tone_panel))
            self.assertTrue(_widget_under(app.contrast_spin, app.tone_panel))
            self.assertTrue(_widget_under(app.gray_world_btn, app.tone_panel))
            self.assertTrue(_widget_under(app.white_patch_btn, app.tone_panel))
            self.assertEqual(app.darks_spin.winfo_class(), "TSpinbox")
            self.assertEqual(app.balance_cyan_spin.winfo_class(), "TSpinbox")
            self.assertEqual(len(app._tone_knobs), 14)
            self.assertTrue(_widget_under(app.darks_knob, app.tone_panel))
            self.assertTrue(_widget_under(app.temperature_knob, app.tone_panel))
            self.assertTrue(_widget_under(app.balance_cyan_knob, app.tone_panel))
            self.assertTrue(_widget_under(app.saturation_knob, app.tone_panel))
            self.assertEqual(app.darks_knob.winfo_class(), "Canvas")
            self.assertFalse(hasattr(app, "texture_knob"))
            self.assertFalse(hasattr(app, "darks_scale"))
            self.assertFalse(hasattr(app, "lights_cyan_scale"))
            self.assertFalse(hasattr(app, "lights_cyan_pct"))
            self.assertEqual(str(app.lights_reds_reset.winfo_manager()), "")
            self.assertEqual(str(app.balance_cyan_reset.winfo_manager()), "")
            self.assertEqual(str(app.temperature_reset.winfo_manager()), "")
            self.assertEqual(str(app.contrast_reset.winfo_manager()), "")
            self.assertEqual(str(app.exposure_reset.winfo_manager()), "")
            self.assertEqual(str(app.texture_reset.winfo_manager()), "")
            self.assertEqual(str(app.darks_reset.winfo_manager()), "")
            self.assertEqual(str(app.cover_reset.winfo_manager()), "")
            self.assertEqual(type(app.texture_reset).__name__, "Label")
            self.assertNotEqual(str(app.texture_reset.cget("image")), "")
            self.assertEqual(str(app.texture_reset.cget("text")), "")
            app.texture_pct.set(40.0)
            app._on_texture_slider("40")
            self.assertEqual(str(app.texture_reset.winfo_manager()), "pack")
            app._reset_texture()
            self.assertAlmostEqual(app.texture_pct.get(), 100.0)
            self.assertEqual(str(app.texture_reset.winfo_manager()), "")
            app.darks_pct.set(25.0)
            app._on_tone_slider("25")
            self.assertEqual(str(app.darks_reset.winfo_manager()), "pack")
            self.assertEqual(str(app.lights_reset.winfo_manager()), "")
            app._reset_darks()
            self.assertAlmostEqual(app.darks_pct.get(), 0.0)
            self.assertAlmostEqual(app.lights_pct.get(), 0.0)
            app.lights_reds_pct.set(40.0)
            app._on_tone_slider("40")
            self.assertEqual(str(app.lights_reds_reset.winfo_manager()), "pack")
            self.assertEqual(str(app.lights_greens_reset.winfo_manager()), "")
            app._reset_lights_rgb("reds")
            self.assertAlmostEqual(app.lights_reds_pct.get(), 0.0)
            self.assertEqual(str(app.lights_reds_reset.winfo_manager()), "")
            app.balance_cyan_pct.set(40.0)
            app._on_tone_slider("40")
            self.assertEqual(str(app.balance_cyan_reset.winfo_manager()), "pack")
            self.assertEqual(str(app.balance_magenta_reset.winfo_manager()), "")
            app._reset_balance("cyan")
            self.assertAlmostEqual(app.balance_cyan_pct.get(), 0.0)
            self.assertEqual(str(app.balance_cyan_reset.winfo_manager()), "")
            app.balance_yellow_spin.set("35")
            app._commit_tone_spin(app.balance_yellow_pct, app.balance_yellow_spin)
            self.assertAlmostEqual(app.balance_yellow_pct.get(), 35.0, delta=0.6)
            app._reset_balance("yellow")
            self.assertAlmostEqual(app.balance_yellow_pct.get(), 0.0)
            app.contrast_pct.set(40.0)
            app._on_tone_slider("40")
            self.assertEqual(str(app.contrast_reset.winfo_manager()), "pack")
            self.assertEqual(str(app.exposure_reset.winfo_manager()), "")
            app._reset_contrast()
            self.assertAlmostEqual(app.contrast_pct.get(), 0.0)
            self.assertEqual(str(app.contrast_reset.winfo_manager()), "")
            app.exposure_spin.set("35")
            app._commit_tone_spin(app.exposure_pct, app.exposure_spin)
            self.assertAlmostEqual(app.exposure_pct.get(), 35.0, delta=0.6)
            app._reset_exposure()
            self.assertAlmostEqual(app.exposure_pct.get(), 0.0)
            app.lights_reds_spin.set("35")
            app._commit_tone_spin(app.lights_reds_pct, app.lights_reds_spin)
            self.assertAlmostEqual(app.lights_reds_pct.get(), 35.0, delta=0.6)
            app._reset_lights_rgb("reds")
            self.assertEqual(app.scale_dpi_choice.get(), "300")
            self.assertEqual(list(app.scale_dpi_combo.cget("values")), ["72", "96", "150", "300", "600", "Custom…"])
            self.assertTrue(app.scale_lock.get())
            src = inspect.getsource(ui_mod.ScrollColumn)
            self.assertNotIn(".bind_all(", src)

            im = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB")
            app.work_image = im
            app.source_image = im
            app.rebuild_ranges()
            app._clear_history()
            self.assertIsNotNone(app.range_map)
            assert app.range_map is not None
            app.crop_zoom.set(2.0)
            app._on_crop_zoom_slider("2")
            self.assertEqual(str(app.crop_zoom_reset.winfo_manager()), "pack")
            self.assertAlmostEqual(app._capture_edit().crop_zoom, 2.0)
            app._reset_crop_zoom()
            self.assertAlmostEqual(app.crop_zoom.get(), 1.0)
            self.assertEqual(str(app.crop_zoom_reset.winfo_manager()), "")
            self.assertEqual(app.tess_h.get(), "off")
            self.assertEqual(app.tess_v.get(), "off")
            self.assertEqual(app.tess_mode.get(), "tile")
            self.assertEqual(app.tess_mode_label.get(), "Tile (Repeating Design)")
            self.assertEqual(
                list(app.tess_mode_combo.cget("values")),
                [
                    "Tile (Repeating Design)",
                    "Tessellation",
                    "Mesh",
                    "Detail mosaic",
                ],
            )
            self.assertFalse(bool(app.tess_built.get()))
            self.assertFalse(bool(app.tess_normalize.get()))
            self.assertEqual(str(app.tess_h_reset.winfo_manager()), "")
            self.assertEqual(str(app.tess_mode_reset.winfo_manager()), "")
            self.assertEqual(str(app.tess_build_reset.winfo_manager()), "")
            self.assertEqual(str(app.tess_normalize_reset.winfo_manager()), "")
            snap = app._capture_edit()
            self.assertEqual(snap.tess_h, "off")
            self.assertEqual(snap.tess_mode, "tile")
            self.assertFalse(snap.tess_built)
            self.assertFalse(snap.tess_normalize)
            app.tess_h.set("left")
            app._tess_committed = ("off", "off", False, "tile")
            app._on_tess_side()
            self.assertEqual(str(app.tess_h_reset.winfo_manager()), "pack")
            self.assertFalse(app._capture_edit().tess_built)
            app._on_tess_build()
            _drain_busy(app, root)
            self.assertTrue(bool(app.tess_built.get()))
            self.assertTrue(app._capture_edit().tess_built)
            self.assertEqual(str(app.tess_build_reset.winfo_manager()), "pack")
            app._reset_tess_h()
            app._reset_tess_build()
            self.assertEqual(app.tess_h.get(), "off")
            self.assertFalse(bool(app.tess_built.get()))
            app._on_tess_normalize()
            self.assertTrue(bool(app.tess_normalize.get()))
            self.assertTrue(app._capture_edit().tess_normalize)
            self.assertEqual(str(app.tess_normalize_reset.winfo_manager()), "pack")
            self.assertFalse(bool(app.tess_built.get()))
            app._on_tess_build()
            _drain_busy(app, root)
            self.assertEqual(app.tess_h.get(), "left")
            self.assertEqual(app.tess_v.get(), "top")
            self.assertTrue(bool(app.tess_built.get()))
            self.assertTrue(bool(app.tess_normalize.get()))
            app._reset_tess_normalize()
            self.assertFalse(bool(app.tess_normalize.get()))
            self.assertEqual(str(app.tess_normalize_reset.winfo_manager()), "")
            app._reset_tess_build()
            app._clear_history()

            yy, xx = np.mgrid[0:48, 0:64]
            luma = np.clip(40.0 + 180.0 * (xx / 63.0), 0, 255)
            pillow = np.stack((luma, luma * 0.9, luma * 0.8), axis=-1).astype(np.uint8)
            pillow_im = Image.fromarray(pillow, mode="RGB")
            app.work_image = pillow_im
            app.source_image = pillow_im
            app.rebuild_ranges()
            app._clear_history()
            app.darks_pct.set(0.0)
            app.lights_pct.set(0.0)
            app.balance_cyan_pct.set(30.0)
            app.temperature_pct.set(20.0)
            app.saturation_pct.set(-10.0)
            app._sync_tone_to_map()
            app._on_tess_normalize()
            self.assertGreater(abs(app.darks_pct.get()) + abs(app.lights_pct.get()), 2.0)
            self.assertTrue(bool(app.tess_normalize.get()))
            snap_lit = app._capture_edit()
            self.assertTrue(snap_lit.tess_normalize)
            self.assertAlmostEqual(snap_lit.tone_darks, app.darks_pct.get() / 100.0, places=4)
            self.assertAlmostEqual(snap_lit.tone_lights_reds, 0.0, places=4)
            self.assertAlmostEqual(snap_lit.tone_balance_cyan, 0.3, places=4)
            self.assertAlmostEqual(snap_lit.tone_lights_cyan, 0.3, places=4)
            self.assertAlmostEqual(snap_lit.tone_temperature, 0.2, places=4)
            self.assertAlmostEqual(snap_lit.tone_saturation, -0.1, places=4)
            self.assertAlmostEqual(snap_lit.tone_contrast, 0.0, places=4)
            self.assertAlmostEqual(snap_lit.tone_exposure, 0.0, places=4)
            self.assertAlmostEqual(app.lights_reds_pct.get(), 0.0)
            self.assertAlmostEqual(app.balance_cyan_pct.get(), 30.0)
            self.assertAlmostEqual(app.temperature_pct.get(), 20.0)
            self.assertAlmostEqual(app.saturation_pct.get(), -10.0)
            self.assertAlmostEqual(app.contrast_pct.get(), 0.0)
            self.assertAlmostEqual(app.exposure_pct.get(), 0.0)
            self.assertAlmostEqual(app.brightness_pct.get(), 0.0)
            first_d = float(app.darks_pct.get())
            first_l = float(app.lights_pct.get())
            app._on_tess_normalize()
            self.assertAlmostEqual(app.darks_pct.get(), first_d, delta=0.51)
            self.assertAlmostEqual(app.lights_pct.get(), first_l, delta=0.51)
            app.darks_pct.set(80.0)
            app._on_tess_normalize()
            self.assertAlmostEqual(app.darks_pct.get(), first_d, delta=0.51)
            self.assertAlmostEqual(app.lights_pct.get(), first_l, delta=0.51)
            app._reset_tess_normalize()
            self.assertFalse(bool(app.tess_normalize.get()))
            self.assertAlmostEqual(app.darks_pct.get(), 0.0, delta=1.0)
            self.assertAlmostEqual(app.lights_pct.get(), 0.0, delta=1.0)
            app._on_tess_normalize()
            self.assertTrue(bool(app.tess_normalize.get()))
            self.assertGreater(abs(app.darks_pct.get()) + abs(app.lights_pct.get()), 2.0)
            app.darks_pct.set(0.0)
            app.lights_pct.set(0.0)
            app._on_tone_slider("")
            self.assertFalse(bool(app.tess_normalize.get()))
            self.assertEqual(str(app.tess_normalize_reset.winfo_manager()), "")
            app.balance_cyan_pct.set(0.0)
            app.temperature_pct.set(0.0)
            app.saturation_pct.set(0.0)
            app._sync_tone_to_map()
            app._clear_history()

            split = np.zeros((48, 64, 3), dtype=np.uint8)
            split[:, :32] = (200, 10, 10)
            split[:, 32:] = (10, 10, 200)
            split_im = Image.fromarray(split, mode="RGB")
            app.work_image = split_im
            app.source_image = split_im
            app.rebuild_ranges()
            app._clear_history()
            from wallpaper_recolor.transform.tessellate import plan_tessellate_crop

            planned = plan_tessellate_crop(split, "left", "off")
            app.tess_h.set("left")
            app._tess_committed = ("off", "off", False, "tile")
            app._on_tess_side()
            before_crop = app._crop_xy_zoom()
            app._on_tess_build()
            _drain_busy(app, root)
            cx, cy, cz = app._crop_xy_zoom()
            self.assertEqual(int(round(cx)), int(planned[0]))
            self.assertEqual(int(round(cy)), int(planned[1]))
            self.assertAlmostEqual(cz, planned[2], places=4)
            self.assertTrue(bool(app.tess_built.get()))
            app.undo_edit()
            self.assertFalse(bool(app.tess_built.get()))
            ux, uy, uz = app._crop_xy_zoom()
            self.assertEqual(int(round(ux)), int(round(before_crop[0])))
            self.assertEqual(int(round(uy)), int(round(before_crop[1])))
            self.assertAlmostEqual(uz, before_crop[2], places=4)
            app._clear_history()

            app.tess_h.set("off")
            app.tess_v.set("off")
            app.tess_built.set(False)
            app.tess_mode.set("tile")
            app._sync_tess_mode_combo()
            app._tess_committed = ("off", "off", False, "tile")
            app.tess_mode.set("voronoi")
            app._on_tess_mode()
            app._on_tess_build()
            _drain_busy(app, root)
            self.assertEqual(app.tess_mode.get(), "voronoi")
            self.assertEqual(app.tess_mode_label.get(), "Detail mosaic")
            self.assertTrue(bool(app.tess_built.get()))
            self.assertEqual(app.tess_h.get(), "left")
            self.assertEqual(app.tess_v.get(), "top")
            self.assertIsNotNone(app._work_live)
            self.assertEqual(app._work_live.size, app._apply_view_transform(split_im).size)
            app._reset_tess_build()
            app._reset_tess_h()
            app._reset_tess_v()
            app._reset_tess_mode()
            self.assertEqual(app.tess_mode.get(), "tile")
            self.assertEqual(app.tess_mode_label.get(), "Tile (Repeating Design)")
            app._clear_history()

            for i in range(HISTORY_LIMIT + 1):
                app._wheel_before = app._capture_edit()
                app.set_range_color(0, (i, 10, 20))
                app._on_wheel_commit((i, 10, 20))
            self.assertEqual(len(app._undo_stack), HISTORY_LIMIT)
            first_kept = app._undo_stack[0].replacements[0]
            self.assertEqual(first_kept, (0, 10, 20))  # original pre-edit dropped; last 20 befores remain
            app.undo_edit()
            self.assertEqual(app.range_map.ranges[0].replacement_rgb, (HISTORY_LIMIT - 1, 10, 20))
            for _ in range(HISTORY_LIMIT - 1):
                app.undo_edit()
            self.assertEqual(len(app._undo_stack), 0)
            self.assertEqual(app.range_map.ranges[0].replacement_rgb, (0, 10, 20))
            app.undo_edit()  # extra undo is a no-op
            self.assertEqual(app.range_map.ranges[0].replacement_rgb, (0, 10, 20))
            app.redo_edit()
            self.assertEqual(app.range_map.ranges[0].replacement_rgb, (1, 10, 20))
            self.assertEqual(str(app.cover_reset.winfo_manager()), "pack")

            app.reset_colors()
            self.assertEqual(str(app.cover_reset.winfo_manager()), "")
            self.assertFalse(hasattr(app, "cover_scale"))
            self.assertFalse(hasattr(app, "_on_cover_slider"))
            n_ranges = len(app.range_map.ranges)
            before_match = tuple(band.match_rgb for band in app.range_map.ranges)
            app.apply_typed_percent(0, 40)
            self.assertAlmostEqual(app.range_map.ranges[0].weight, 0.40, places=4)
            self.assertAlmostEqual(app.range_map.ranges[2].weight, 1.0 / n_ranges, places=3)
            self.assertAlmostEqual(sum(app.range_map.weights()), 1.0, places=5)
            self.assertEqual(str(app.cover_reset.winfo_manager()), "")
            app.set_range_color(0, (10, 20, 30))
            self.assertEqual(app.range_map.ranges[0].replacement_rgb, (10, 20, 30))
            self.assertEqual(str(app.cover_reset.winfo_manager()), "pack")
            self.assertGreaterEqual(len(app._undo_stack), 1)
            swatch_before = app._undo_stack[-1].replacements[0]
            app.undo_edit()
            self.assertEqual(app.range_map.ranges[0].replacement_rgb, swatch_before)
            app.redo_edit()
            self.assertEqual(app.range_map.ranges[0].replacement_rgb, (10, 20, 30))
            from wallpaper_recolor.ui.coverage_bar import HALF_MATCH, HALF_REPLACE

            old_match = app.range_map.ranges[0].match_rgb
            app.selected_half = HALF_MATCH
            app.set_match_color(0, (9, 8, 7))
            self.assertEqual(app.range_map.ranges[0].match_rgb, (9, 8, 7))
            app.undo_edit()
            self.assertEqual(app.range_map.ranges[0].match_rgb, old_match)
            app.selected_half = HALF_REPLACE
            app.reset_colors()
            self.assertEqual(app.range_map.ranges[0].match_rgb, app.range_map.ranges[0].replacement_rgb)
            self.assertEqual(tuple(band.match_rgb for band in app.range_map.ranges), before_match)
            self.assertEqual(str(app.cover_reset.winfo_manager()), "")

            app.set_range_visible(0, False)
            self.assertFalse(app.range_map.ranges[0].visible)

            app._opening = True
            app.rebuild_ranges()
            app._opening = False
            app._clear_history()
            self.assertEqual(app._undo_stack, [])
            self.assertEqual(app._redo_stack, [])
        finally:
            root.destroy()

    def test_right_column_split_sash_and_reparent(self) -> None:
        """Right host is a vertical paned split; panels pack as true children of each scroller."""
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod

        self.assertAlmostEqual(ui_mod.RIGHT_SPLIT_FRACTION, 0.5)
        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            self.assertEqual(str(app.right_host.cget("orient")), "vertical")
            self.assertEqual(len(app.right_host.panes()), 2)
            self.assertIs(app.right_column, app.right_top_column)
            self.assertIsNot(app.right_top_column, app.right_bottom_column)
            self.assertIsNot(app.right_top_column.canvas, app.right_bottom_column.canvas)
            self.assertEqual(
                app.right_top_column.panels,
                [
                    app.wheel_panel,
                    app.texture_panel,
                    app.tone_panel,
                    app.scale_panel,
                    app.crop_panel,
                ],
            )
            self.assertEqual(
                app.right_bottom_column.panels,
                [
                    app.layers_panel,
                    app.labels_panel,
                    app.tess_panel,
                    app.history_panel,
                ],
            )
            self.assertIs(app.wheel_panel.master, app.right_top_column.inner)
            self.assertIs(app.layers_panel.master, app.right_bottom_column.inner)
            self.assertIsNot(app.wheel_panel.master, app.root)
            sash_src = inspect.getsource(ui_mod.WallpaperRecolorApp._set_default_sash)
            self.assertIn("RIGHT_SPLIT_FRACTION", sash_src)
            self.assertIn("_apply_right_sash_fraction", sash_src)
            hit_src = inspect.getsource(ui_mod.WallpaperRecolorApp._hit_column)
            self.assertIn("right_host", hit_src)
            self.assertIn("sashpos", hit_src)
            repack_src = inspect.getsource(ui_mod.ScrollColumn._repack)
            self.assertNotIn("in_=self.inner", repack_src)
            self.assertNotIn('in_=self.inner', inspect.getsource(ui_mod.DockablePanel.__init__))
            chrome_src = inspect.getsource(ui_mod.DockablePanel._build_chrome)
            self.assertNotIn("pack(in_=", chrome_src)
            self.assertNotIn("app.root", chrome_src)

            for panel in (*app.right_top_column.panels, *app.right_bottom_column.panels):
                self.assertIs(panel.bar.master, panel)
                self.assertIs(panel.body.master, panel)

            root.geometry("1280x800+40+40")
            root.deiconify()
            root.update()
            app._set_default_sash()
            root.update()
            top_docked = app.right_top_column._docked_panels()
            bot_docked = app.right_bottom_column._docked_panels()
            height = int(app.right_host.winfo_height())
            if height >= 80:
                pos = int(app.right_host.sashpos(0))
                sash_root = int(app.right_host.winfo_rooty()) + pos
                self.assertAlmostEqual(pos / height, 0.5, delta=0.12)
                top_pane_bottom = int(app.right_top_column.winfo_rooty()) + int(
                    app.right_top_column.winfo_height()
                )
                layers_top = int(app.layers_panel.bar.winfo_rooty())
                self.assertLessEqual(top_pane_bottom, sash_root + 8)
                self.assertGreaterEqual(layers_top, sash_root - 8)
                self.assertIs(app.right_top_column.inner.master, app.right_top_column.canvas)
                self.assertIs(app.right_bottom_column.inner.master, app.right_bottom_column.canvas)
                self.assertEqual(str(app.tone_panel.winfo_manager()), "pack")
                self.assertEqual(str(app.layers_panel.winfo_manager()), "pack")

            for panel in (*app.right_top_column.panels, *app.right_bottom_column.panels):
                panel.set_expanded(False)
            root.update()
            app._set_default_sash()
            root.update()
            top_docked = app.right_top_column._docked_panels()
            bot_docked = app.right_bottom_column._docked_panels()
            self.assertEqual(
                [p for p in app.right_top_column.inner.pack_slaves() if p in top_docked],
                top_docked,
            )
            top_ys = [int(p.bar.winfo_rooty()) for p in top_docked]
            bot_ys = [int(p.bar.winfo_rooty()) for p in bot_docked]
            self.assertEqual(len(top_ys), len(set(top_ys)), msg=f"top titles piled: {top_ys}")
            self.assertEqual(len(bot_ys), len(set(bot_ys)), msg=f"bottom titles piled: {bot_ys}")
            all_ys = top_ys + bot_ys
            self.assertEqual(
                len(all_ys),
                len(set(all_ys)),
                msg=f"titles piled across sash: top={top_ys} bot={bot_ys}",
            )
            height = int(app.right_host.winfo_height())
            if height >= 80:
                pos = int(app.right_host.sashpos(0))
                top_y = int(app.right_top_column.winfo_y())
                bot_y = int(app.right_bottom_column.winfo_y())
                self.assertLess(top_y, bot_y)
                sash_root = int(app.right_host.winfo_rooty()) + pos
                top_bottom = max(
                    int(p.bar.winfo_rooty()) + int(p.bar.winfo_height())
                    for p in top_docked
                )
                bot_top = min(int(p.bar.winfo_rooty()) for p in bot_docked)
                self.assertLessEqual(top_bottom, sash_root + 8)
                self.assertGreaterEqual(bot_top, sash_root - 8)
            root.withdraw()

            app._place_panel(app.wheel_panel, app.right_bottom_column, 0)
            self.assertIs(app.wheel_panel.column, app.right_bottom_column)
            self.assertIn(app.wheel_panel, app.right_bottom_column.panels)
            self.assertNotIn(app.wheel_panel, app.right_top_column.panels)
            self.assertIs(app.wheel_panel.master, app.right_bottom_column.inner)
            info = app.wheel_panel.pack_info()
            self.assertEqual(str(info["in"]), str(app.right_bottom_column.inner))
        finally:
            root.destroy()

    def test_preview_opens_on_composite_tab(self) -> None:
        """Preview notebook starts on Composite, not Clusters."""
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            selected = str(app.notebook.tab(app.notebook.select(), "text"))
            self.assertEqual(selected, "Composite")
            self.assertNotEqual(str(app.notebook.select()), str(app.cluster_plot))
        finally:
            root.destroy()

    def test_close_asks_save_edit_state(self) -> None:
        """WM_DELETE_WINDOW / Exit ask Yes-No-Cancel before destroy."""
        import tkinter as tk
        from unittest.mock import patch

        import wallpaper_recolor.ui.app as ui_mod

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            handler = str(root.protocol("WM_DELETE_WINDOW"))
            self.assertTrue(
                handler.endswith("_on_app_close"),
                f"WM_DELETE_WINDOW is {handler!r}",
            )
            with patch.object(app, "_destroy_app_window") as destroy:
                with patch.object(ui_mod.messagebox, "askyesnocancel", return_value=None):
                    app._on_app_close()
                destroy.assert_not_called()
                with patch.object(ui_mod.messagebox, "askyesnocancel", return_value=False):
                    app._on_app_close()
                destroy.assert_called_once()
                destroy.reset_mock()
                with patch.object(ui_mod.messagebox, "askyesnocancel", return_value=True):
                    with patch.object(app, "save_edit_state", return_value=False) as save:
                        app._on_app_close()
                        save.assert_called_once()
                destroy.assert_not_called()
                with patch.object(ui_mod.messagebox, "askyesnocancel", return_value=True):
                    with patch.object(app, "save_edit_state", return_value=True):
                        app._on_app_close()
                destroy.assert_called_once()
        finally:
            try:
                root.destroy()
            except tk.TclError:
                pass

    def test_coverage_bar_two_swatches_not_grid(self) -> None:
        """Header + one diagonal match/change row — no extra Coverage swatches."""
        from types import SimpleNamespace
        import tkinter as tk

        from wallpaper_recolor.ui.coverage_bar import (
            HALF_MATCH,
            HALF_REPLACE,
            SEG_H,
            CoverageBar,
            _half_at_diagonal,
        )
        from wallpaper_recolor.ui import run

        self.assertTrue(callable(run))
        root = tk.Tk()
        root.withdraw()
        try:
            bar = CoverageBar(root)
            bar.pack(fill="x")
            bar.update_idletasks()
            self.assertFalse(hasattr(bar, "preview"))
            self.assertFalse(hasattr(bar, "match_swatch"))
            self.assertFalse(hasattr(bar, "replace_swatch"))
            canvases = [c for c in bar.winfo_children() if isinstance(c, tk.Canvas)]
            self.assertEqual(canvases, [bar.bar, bar.segments])
            self.assertEqual(list(bar.pack_slaves()), [bar.bar, bar.segments])
            self.assertIsNone(bar.eyedrop_btn)
            self.assertEqual([c for c in bar.winfo_children() if c.winfo_class() == "Label"], [])

            bar.set_state(
                [0.5, 0.5],
                [(255, 0, 0), (0, 0, 255)],
                [(0, 255, 0), (255, 255, 0)],
                selected=0,
                selected_half=HALF_REPLACE,
            )
            bar.update_idletasks()
            self.assertEqual(len(bar.bar.find_withtag("seg0")), 1)
            self.assertEqual(len(bar.bar.find_withtag("seg1")), 1)
            self.assertEqual(bar.bar.find_withtag("seg0m"), ())
            self.assertEqual(bar.bar.find_withtag("seg0r"), ())
            head_fill = bar.bar.itemcget(bar.bar.find_withtag("seg0")[0], "fill").lower()
            self.assertNotIn(head_fill, ("#ff0000", "#0000ff"))
            self.assertTrue(bar.bar.find_withtag("coverpct"))
            cover_txt = [bar.bar.itemcget(i, "text") for i in bar.bar.find_withtag("coverpct")]
            self.assertIn("50%", cover_txt)
            self.assertTrue(bar.segments.find_withtag("seg0m"))
            self.assertTrue(bar.segments.find_withtag("seg0r"))
            self.assertTrue(bar.segments.find_withtag("seg1m"))
            self.assertTrue(bar.segments.find_withtag("seg1r"))
            self.assertFalse(bar.segments.find_withtag("lumakey"))
            self.assertEqual(
                bar.segments.itemcget(bar.segments.find_withtag("seg0m")[0], "fill"),
                "#FF0000",
            )
            self.assertEqual(
                bar.segments.itemcget(bar.segments.find_withtag("seg0r")[0], "fill"),
                "#00FF00",
            )
            m = bar.segments.coords(bar.segments.find_withtag("seg0m")[0])
            r = bar.segments.coords(bar.segments.find_withtag("seg0r")[0])
            self.assertEqual(len(m), 6)
            self.assertEqual(len(r), 6)
            self.assertAlmostEqual(m[0], m[4])
            self.assertAlmostEqual(m[1], 0.0)
            self.assertAlmostEqual(m[3], 0.0)
            self.assertAlmostEqual(m[2], r[0])
            self.assertAlmostEqual(m[2], r[2])
            self.assertAlmostEqual(m[5], r[3])
            self.assertAlmostEqual(m[5], r[5])
            self.assertGreater(m[2] - m[0], SEG_H)

            i0, x0, x1 = bar._seg_hits[0]
            self.assertEqual(i0, 0)
            bar._press_seg(SimpleNamespace(x=x0 + 3, y=3))
            self.assertEqual(bar.selected, 0)
            self.assertEqual(bar.selected_half, HALF_MATCH)
            bar._press_seg(SimpleNamespace(x=x1 - 3, y=SEG_H - 3))
            self.assertEqual(bar.selected_half, HALF_REPLACE)
            self.assertEqual(_half_at_diagonal(x0, x1, SEG_H, x0 + 3, 3), HALF_MATCH)
            self.assertEqual(_half_at_diagonal(x0, x1, SEG_H, x1 - 3, SEG_H - 3), HALF_REPLACE)

            bar.set_state(
                [0.3, 0.4, 0.3],
                [(255, 0, 0), (0, 255, 0), (0, 0, 255)],
                [(128, 0, 0), (0, 128, 0), (0, 0, 128)],
                selected=1,
                selected_half=HALF_MATCH,
            )
            bar.update_idletasks()
            self.assertEqual(len(bar.bar.find_withtag("seg2")), 1)
            self.assertEqual(bar.bar.find_withtag("seg2m"), ())
            self.assertEqual(bar.selected, 1)
            self.assertEqual(bar.selected_half, HALF_MATCH)
            self.assertEqual(
                bar.segments.itemcget(bar.segments.find_withtag("seg1m")[0], "fill"),
                "#00FF00",
            )
            self.assertEqual(
                bar.segments.itemcget(bar.segments.find_withtag("seg1r")[0], "fill"),
                "#008000",
            )
        finally:
            root.destroy()
    def test_coverage_bar_luma_top_and_dropper_inside_selected(self) -> None:
        """Diagonal row only; luma key on match triangle; dropper in selected half."""
        from types import SimpleNamespace
        import tkinter as tk

        from wallpaper_recolor.ui.coverage_bar import (
            HALF_MATCH,
            HALF_REPLACE,
            SEG_H,
            SEL_OUTLINE,
            CoverageBar,
            _half_at_diagonal,
        )

        root = tk.Tk()
        root.withdraw()
        try:
            photo = tk.PhotoImage(master=root, width=8, height=8)
            hits: list[int] = []
            bar = CoverageBar(root, on_eyedrop=lambda: hits.append(1), eyedrop_photo=photo)
            bar.pack(fill="x")
            bar.update_idletasks()
            self.assertIsNone(bar.eyedrop_btn)
            self.assertEqual([c for c in bar.winfo_children() if c.winfo_class() == "Label"], [])
            self.assertIs(bar.eyedrop_photo, photo)

            bar.set_state(
                [0.5, 0.5],
                [(255, 0, 0), (0, 0, 255)],
                [(0, 255, 0), (255, 255, 0)],
                selected=0,
                selected_half=HALF_REPLACE,
                luma_mode=True,
                luma_keys=[0.25, 0.75],
            )
            bar.update_idletasks()
            luma_txt = [bar.segments.itemcget(i, "text") for i in bar.segments.find_withtag("lumakey")]
            self.assertIn("25%", luma_txt)
            self.assertEqual(
                bar.segments.itemcget(bar.segments.find_withtag("seg0m")[0], "fill"),
                "#404040",
            )
            self.assertEqual(
                bar.segments.itemcget(bar.segments.find_withtag("seg0r")[0], "fill"),
                "#00FF00",
            )
            cover_txt = [bar.bar.itemcget(i, "text") for i in bar.bar.find_withtag("coverpct")]
            self.assertIn("50%", cover_txt)
            self.assertNotIn("25%", cover_txt)
            self.assertNotEqual(luma_txt[0], cover_txt[0] if cover_txt else "")
            self.assertTrue(bar.segments.find_withtag("dropper"))
            self.assertEqual(
                bar.segments.itemcget(bar.segments.find_withtag("seloutline")[0], "outline"),
                SEL_OUTLINE,
            )
            i0, x0, x1 = bar._seg_hits[0]
            drop = bar.segments.bbox("dropper")
            self.assertIsNotNone(drop)
            cx = (drop[0] + drop[2]) / 2
            cy = (drop[1] + drop[3]) / 2
            self.assertEqual(_half_at_diagonal(x0, x1, SEG_H, cx, cy), HALF_REPLACE)
            bar._press_seg(SimpleNamespace(x=cx, y=cy))
            self.assertEqual(hits, [1])
            bar._press_seg(SimpleNamespace(x=x0 + 3, y=3))
            self.assertEqual(hits, [1])
            self.assertEqual(bar.selected_half, HALF_MATCH)
            drop = bar.segments.bbox("dropper")
            self.assertIsNotNone(drop)
            cx = (drop[0] + drop[2]) / 2
            cy = (drop[1] + drop[3]) / 2
            i0, x0, x1 = bar._seg_hits[0]
            self.assertEqual(_half_at_diagonal(x0, x1, SEG_H, cx, cy), HALF_MATCH)
            self.assertEqual(
                bar.segments.itemcget(bar.segments.find_withtag("seloutline")[0], "outline"),
                SEL_OUTLINE,
            )
            self.assertTrue(bar.segments.find_withtag("lumakey"))
            bar._press_seg(SimpleNamespace(x=cx, y=cy))
            self.assertEqual(hits, [1, 1])

            bar.set_state(
                [0.5, 0.5],
                [(255, 0, 0), (0, 0, 255)],
                [(0, 255, 0), (255, 255, 0)],
                selected=0,
                selected_half=HALF_REPLACE,
                luma_mode=False,
            )
            self.assertFalse(bar.segments.find_withtag("lumakey"))
            self.assertEqual(
                bar.segments.itemcget(bar.segments.find_withtag("seg0m")[0], "fill"),
                "#FF0000",
            )
            self.assertTrue(bar.segments.find_withtag("dropper"))
            i0, x0, x1 = bar._seg_hits[0]
            drop = bar.segments.bbox("dropper")
            cx = (drop[0] + drop[2]) / 2
            cy = (drop[1] + drop[3]) / 2
            self.assertEqual(_half_at_diagonal(x0, x1, SEG_H, cx, cy), HALF_REPLACE)
        finally:
            root.destroy()

class TestOutputScale(unittest.TestCase):
    """Save/export W×H independent of preview zoom; column wheel; View menu."""
    def test_inches_at_300_dpi(self) -> None:
        from wallpaper_recolor.transform.scale import UNIT_INCHES, resolve_output_size

        size = resolve_output_size(1000, 1000, 10, 10, UNIT_INCHES, 300, True)
        self.assertEqual(size, (3000, 3000))

    def test_cm_at_300_dpi(self) -> None:
        from wallpaper_recolor.transform.scale import UNIT_CM, resolve_output_size

        size = resolve_output_size(100, 100, 25.4, 25.4, UNIT_CM, 300, True)
        self.assertEqual(size, (3000, 3000))

    def test_empty_or_zero_keeps_original(self) -> None:
        from wallpaper_recolor.transform.scale import UNIT_PIXELS, resolve_output_size, scale_image

        self.assertIsNone(resolve_output_size(800, 600, None, None, UNIT_PIXELS, 300, True))
        self.assertIsNone(resolve_output_size(800, 600, 0, 0, UNIT_PIXELS, 300, True))
        im = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB")
        self.assertIs(scale_image(im, None), im)
        self.assertEqual(scale_image(im, (8, 8)).size, (8, 8))

    def test_nearest_and_lanczos_callable(self) -> None:
        from wallpaper_recolor.transform.scale import (
            RESAMPLE_LANCZOS,
            RESAMPLE_NEAREST,
            resampling_filter,
            scale_image,
        )

        self.assertIs(resampling_filter(RESAMPLE_NEAREST), Image.Resampling.NEAREST)
        self.assertIs(resampling_filter(RESAMPLE_LANCZOS), Image.Resampling.LANCZOS)
        im = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8), mode="RGB")
        near = scale_image(im, (20, 20), RESAMPLE_NEAREST)
        lanc = scale_image(im, (20, 20), RESAMPLE_LANCZOS)
        self.assertEqual(near.size, (20, 20))
        self.assertEqual(lanc.size, (20, 20))

    def test_dpi_presets_and_custom(self) -> None:
        from wallpaper_recolor.transform.scale import (
            DPI_CHOICES,
            DPI_CUSTOM_LABEL,
            DPI_DEFAULT,
            parse_dpi_choice,
        )

        self.assertEqual(DPI_CHOICES, ("72", "96", "150", "300", "600", "Custom…"))
        self.assertEqual(DPI_DEFAULT, 300)
        self.assertEqual(parse_dpi_choice("150"), 150.0)
        self.assertEqual(parse_dpi_choice(DPI_CUSTOM_LABEL, "240"), 240.0)
        self.assertEqual(parse_dpi_choice(DPI_CUSTOM_LABEL, ""), 300.0)

    def test_dpi_changes_pixels_only_for_physical_units(self) -> None:
        from wallpaper_recolor.transform.scale import UNIT_INCHES, UNIT_PIXELS, resolve_output_size

        at_300 = resolve_output_size(500, 500, 10, 10, UNIT_INCHES, 300, True)
        at_150 = resolve_output_size(500, 500, 10, 10, UNIT_INCHES, 150, True)
        self.assertEqual(at_300, (3000, 3000))
        self.assertEqual(at_150, (1500, 1500))
        px_300 = resolve_output_size(500, 500, 400, 400, UNIT_PIXELS, 300, True)
        px_150 = resolve_output_size(500, 500, 400, 400, UNIT_PIXELS, 150, True)
        self.assertEqual(px_300, (400, 400))
        self.assertEqual(px_150, (400, 400))

    def test_save_image_tags_dpi(self) -> None:
        from wallpaper_recolor.io.image_io import save_image

        im = Image.fromarray(np.zeros((12, 12, 3), dtype=np.uint8), mode="RGB")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dpi.png"
            save_image(im, path, dpi=300)
            with Image.open(path) as loaded:
                dpi = loaded.info.get("dpi")
            self.assertIsNotNone(dpi)
            self.assertAlmostEqual(float(dpi[0]), 300.0, delta=1.0)

    def test_ui_scale_panel_and_import(self) -> None:
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod
        from wallpaper_recolor.transform.scale import DPI_CUSTOM_LABEL, RESAMPLE_LANCZOS, UNIT_INCHES
        from wallpaper_recolor.ui import run

        self.assertTrue(callable(run))
        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            self.assertFalse(app.scale_panel.allow_pop_out)
            self.assertIs(app.scale_panel.column, app.right_column)
            self.assertFalse(app.crop_panel.allow_pop_out)
            self.assertIs(app.crop_panel.column, app.right_column)
            self.assertFalse(app.tess_panel.allow_pop_out)
            self.assertIs(app.tess_panel.column, app.right_bottom_column)
            self.assertFalse(app.labels_panel.allow_pop_out)
            self.assertIs(app.labels_panel.column, app.right_bottom_column)
            self.assertFalse(app.layers_panel.allow_pop_out)
            self.assertIs(app.layers_panel.column, app.right_bottom_column)
            self.assertFalse(hasattr(app, "lighting_panel"))
            self.assertTrue(_widget_under(app.tess_normalize_btn, app.tone_panel))
            self.assertAlmostEqual(app.crop_zoom.get(), 1.0)
            self.assertEqual(app.scale_resample.get(), RESAMPLE_LANCZOS)
            self.assertEqual(app.scale_dpi_choice.get(), "300")
            self.assertAlmostEqual(app._scale_dpi(), 300.0)
            im = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB")
            app.source_image = im
            app.scale_unit.set(UNIT_INCHES)
            app._scale_unit_prev = UNIT_INCHES
            app.scale_width.set("10")
            app.scale_height.set("10")
            app._refresh_scale_labels()
            size, _filt, dpi = app._output_scale_args()
            self.assertEqual(size, (3000, 3000))
            self.assertEqual(dpi, 300.0)
            app.scale_dpi_choice.set("150")
            app._on_scale_dpi_choice()
            size, _filt, dpi = app._output_scale_args()
            self.assertEqual(size, (1500, 1500))
            self.assertEqual(dpi, 150.0)
            app.scale_dpi_choice.set(DPI_CUSTOM_LABEL)
            app._on_scale_dpi_choice()
            self.assertEqual(app.scale_dpi_custom.get(), "150")
            app.scale_dpi_custom.set("600")
            app._on_scale_dpi_custom()
            size, _filt, dpi = app._output_scale_args()
            self.assertEqual(size, (6000, 6000))
            self.assertEqual(dpi, 600.0)
            # Pixel mode: DPI does not change pixel count
            app.scale_unit.set("Pixels")
            app._scale_unit_prev = "Pixels"
            app.scale_width.set("400")
            app.scale_height.set("400")
            app.scale_dpi_choice.set("72")
            app._on_scale_dpi_choice()
            size, _filt, dpi = app._output_scale_args()
            self.assertEqual(size, (400, 400))
            self.assertEqual(dpi, 72.0)
            before = app._output_scale_args()
            app.preview_zoom.set(400.0)
            app._on_preview_zoom_slider("400")
            self.assertEqual(app._output_scale_args(), before)
            self.assertEqual(app._output_scale_args()[0], (400, 400))
            save_src = inspect.getsource(ui_mod.WallpaperRecolorApp._save_composite)
            self.assertNotIn("preview_zoom", save_src)
            self.assertNotIn("_scale_view_zoom", save_src)
            self.assertNotIn("_orig_photo", save_src)
            self.assertIn("_output_scale_args", save_src)
            self.assertIn("scale_image", save_src)
            args_src = inspect.getsource(ui_mod.WallpaperRecolorApp._output_scale_args)
            self.assertNotIn("preview_zoom", args_src)
            self.assertIn("resolve_output_size", args_src)
            export_src = inspect.getsource(ui_mod.WallpaperRecolorApp.export_pack)
            self.assertNotIn("preview_zoom", export_src)
            zip_src = inspect.getsource(ui_mod.WallpaperRecolorApp.export_layers_zip)
            self.assertNotIn("preview_zoom", zip_src)
        finally:
            root.destroy()


    def test_column_mousewheel_scrolls_hovered_column(self) -> None:
        """Wheel over a column scrolls that canvas; the other column stays put."""
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod
        from wallpaper_recolor.ui import run

        self.assertTrue(callable(run))
        init_src = inspect.getsource(ui_mod.WallpaperRecolorApp.__init__)
        self.assertIn("<MouseWheel>", init_src)
        self.assertIn("bind_all", init_src)
        col_src = inspect.getsource(ui_mod.ScrollColumn)
        self.assertNotIn(".bind_all(", col_src)
        handler_src = inspect.getsource(ui_mod.WallpaperRecolorApp._on_column_mousewheel)
        self.assertIn("winfo_containing", handler_src)
        xy_src = inspect.getsource(ui_mod.WallpaperRecolorApp._wheel_event_xy)
        self.assertIn("x_root", xy_src)
        self.assertIn("winfo_pointerx", xy_src)
        self.assertIn("_bind_wheel_tree", inspect.getsource(ui_mod.ScrollColumn._bind_column_wheel_widgets))

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            right = app.right_column
            left = app.left_column
            scrolled: list[tuple[int, str]] = []
            left_scrolled: list[tuple[int, str]] = []
            right.canvas.yview_scroll = lambda n, u: scrolled.append((n, u))  # type: ignore[method-assign]
            left.canvas.yview_scroll = lambda n, u: left_scrolled.append((n, u))  # type: ignore[method-assign]

            class _Ev:
                delta = -120
                num = 0
                widget = app.texture_scale
                x_root = 0
                y_root = 0

            self.assertEqual(right._on_mousewheel(_Ev()), "break")
            self.assertEqual(scrolled, [(1, "units")])
            self.assertEqual(left_scrolled, [])

            class _EvUp:
                delta = 120
                num = 0
                widget = app.texture_scale
                x_root = 0
                y_root = 0

            self.assertEqual(right._on_mousewheel(_EvUp()), "break")
            self.assertEqual(scrolled[-1], (-1, "units"))

            class _EvWinPlaceholder:
                delta = 120
                num = "??"
                widget = app.texture_scale
                x_root = 0
                y_root = 0

            self.assertEqual(right._on_mousewheel(_EvWinPlaceholder()), "break")
            self.assertEqual(scrolled[-1], (-1, "units"))

            class _EvLinuxUp:
                delta = 0
                num = 4
                widget = app.texture_scale
                x_root = 0
                y_root = 0

            self.assertEqual(right._on_mousewheel(_EvLinuxUp()), "break")
            self.assertEqual(scrolled[-1], (-1, "units"))

            class _EvLinuxDown:
                delta = 0
                num = 5
                widget = app.texture_scale
                x_root = 0
                y_root = 0

            self.assertEqual(right._on_mousewheel(_EvLinuxDown()), "break")
            self.assertEqual(scrolled[-1], (1, "units"))

            left_hits: list[int] = []
            right_hits: list[int] = []
            app._pointer_over_preview_image = lambda _e: False  # type: ignore[method-assign]
            app._pointer_over_clusters = lambda _e: False  # type: ignore[method-assign]
            app.left_column.contains_root = lambda _x, _y: False  # type: ignore[method-assign]
            app.right_column.contains_root = lambda _x, _y: True  # type: ignore[method-assign]
            app.right_bottom_column.contains_root = lambda _x, _y: False  # type: ignore[method-assign]
            app.left_column._on_mousewheel = lambda _e: left_hits.append(1) or "break"  # type: ignore[method-assign]
            app.right_column._on_mousewheel = lambda _e: right_hits.append(1) or "break"  # type: ignore[method-assign]
            app._on_column_mousewheel(_Ev())
            self.assertEqual(left_hits, [])
            self.assertEqual(right_hits, [1])
            self.assertTrue(app.texture_scale.bind("<MouseWheel>"))
            self.assertEqual(type(app.texture_reset).__name__, "Label")
            self.assertNotEqual(str(app.texture_reset.cget("image")), "")
            self.assertEqual(str(app.texture_reset.cget("text")), "")
            self.assertTrue(hasattr(app, "toolbar"))
            self.assertTrue(hasattr(app, "status_bar"))
            wheel_src = inspect.getsource(ui_mod.ScrollColumn._on_mousewheel)
            self.assertNotIn("_raise_window_chrome", wheel_src)
            self.assertIn("break", wheel_src)
            self.assertIn("ValueError", wheel_src)
            layout_src = inspect.getsource(ui_mod.ScrollColumn._sync_layout)
            self.assertIn("_schedule_raise_chrome", layout_src)
            self.assertIn("_raise_docked_stack", layout_src)
            raise_src = inspect.getsource(ui_mod.ScrollColumn._raise_docked_stack)
            self.assertIn("_raise_dock_stacks", raise_src)
            self.assertIn("lift()", inspect.getsource(ui_mod.WallpaperRecolorApp._raise_dock_stacks))
            bind_src = inspect.getsource(ui_mod._bind_wheel_tree)
            self.assertIn("break", bind_src)
            app._raise_window_chrome()
        finally:
            root.destroy()

    def test_tone_spinboxes_step_by_one(self) -> None:
        """Color & lighting uses ttk.Spinbox ±1; 0 stays identity; Texture still a Scale."""
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod
        from wallpaper_recolor.color.tone import apply_tone_rgb, is_neutral_tone

        self.assertTrue(callable(ui_mod._bind_smooth_scale))
        smooth_src = inspect.getsource(ui_mod._bind_smooth_scale)
        self.assertIn("break", smooth_src)
        self.assertNotIn('bind("<MouseWheel>"', smooth_src)
        wheel_src = inspect.getsource(ui_mod._bind_wheel_tree)
        self.assertIn("break", wheel_src)

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            self.assertEqual(app.darks_spin.winfo_class(), "TSpinbox")
            self.assertAlmostEqual(float(app.darks_spin.cget("increment")), 1.0)
            self.assertAlmostEqual(float(app.darks_spin.cget("from")), -100.0)
            self.assertAlmostEqual(float(app.darks_spin.cget("to")), 100.0)
            self.assertTrue(app.darks_spin.bind("<Return>"))
            self.assertTrue(app.darks_spin.bind("<FocusOut>"))
            self.assertTrue(app.texture_scale.bind("<MouseWheel>"))

            self.assertAlmostEqual(app.darks_pct.get(), 0.0)
            kwargs = app._tone_apply_kwargs()
            self.assertTrue(is_neutral_tone(**kwargs))
            rgb = np.arange(96, dtype=np.uint8).reshape(8, 4, 3)
            np.testing.assert_array_equal(apply_tone_rgb(rgb, **kwargs), rgb)

            app.darks_spin.set("1")
            app._commit_tone_spin(app.darks_pct, app.darks_spin)
            root.update_idletasks()
            self.assertAlmostEqual(app.darks_pct.get(), 1.0)
            app.darks_spin.set("0")
            app._commit_tone_spin(app.darks_pct, app.darks_spin)
            root.update_idletasks()
            self.assertAlmostEqual(app.darks_pct.get(), 0.0)
            kwargs = app._tone_apply_kwargs()
            self.assertTrue(is_neutral_tone(**kwargs))
            np.testing.assert_array_equal(apply_tone_rgb(rgb, **kwargs), rgb)

            app.darks_pct.set(40.0)
            app.darks_spin.update_idletasks()
            app.darks_spin.event_generate("<ButtonPress-1>", x=2, y=4)
            root.update_idletasks()
            self.assertAlmostEqual(app.darks_pct.get(), 40.0, delta=0.6)
            app.darks_spin.event_generate("<ButtonRelease-1>", x=2, y=4)

            app.darks_knob.set_value(40)
            root.update_idletasks()
            self.assertAlmostEqual(app.darks_pct.get(), 40.0)
            self.assertEqual(str(app.darks_reset.winfo_manager()), "pack")
            app._reset_darks()
            self.assertAlmostEqual(app.darks_pct.get(), 0.0)
            self.assertEqual(str(app.darks_reset.winfo_manager()), "")
            app.darks_spin.set("1")
            app._commit_tone_spin(app.darks_pct, app.darks_spin)
            self.assertAlmostEqual(app.darks_pct.get(), 1.0)
            rng = np.random.default_rng(4)
            im = Image.fromarray(rng.integers(30, 220, (16, 16, 3), dtype=np.uint8), mode="RGB")
            app.work_image = im
            app.source_image = im
            app.rebuild_ranges()
            app.darks_knob.set_value(-25)
            app._sync_tone_to_map()
            assert app.range_map is not None
            self.assertAlmostEqual(app.range_map.tone_darks, -0.25, places=4)
            app.balance_cyan_knob.set_value(50)
            app._sync_tone_to_map()
            self.assertAlmostEqual(app.range_map.tone_balance_cyan, 0.5, places=4)
        finally:
            root.destroy()


    def test_view_menu_hide_show_and_reset_layout(self) -> None:
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod
        from wallpaper_recolor.ui import run

        self.assertTrue(callable(run))
        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            self.assertFalse(hasattr(app, "view_menubutton"))
            self.assertEqual(str(app.root.cget("menu")), str(app.menubar))
            self.assertEqual(int(app.view_menu.cget("tearoff")), 0)
            app._rebuild_view_menu()
            end = int(app.view_menu.index("end") or -1)
            view_labels = []
            for i in range(end + 1):
                try:
                    view_labels.append(str(app.view_menu.entrycget(i, "label")))
                except tk.TclError:
                    continue
            self.assertIn("Color & lighting", view_labels)
            self.assertNotIn("Tone", view_labels)
            self.assertIn("Position & Zoom", view_labels)
            self.assertIn("History", view_labels)
            self.assertIn("Layers", view_labels)
            self.assertIn("Labels", view_labels)
            self.assertIn("Tessellate", view_labels)
            self.assertIn("3×3 tile", view_labels)
            self.assertIn("Seam offset", view_labels)
            self.assertIn("Room mockup", view_labels)
            self.assertNotIn("Crop", view_labels)
            self.assertEqual(str(app.crop_panel.panel_title), "Position & Zoom")
            self.assertIn("Fit", view_labels)
            self.assertIn("Zoom in", view_labels)
            self.assertIn("Zoom out", view_labels)
            self.assertIn("Reset preview", view_labels)
            self.assertIn("Layout profiles", view_labels)
            self.assertIn("Reset layout", view_labels)
            self.assertNotIn("Normalize lighting", view_labels)
            self.assertIs(app.tone_panel.column, app.right_top_column)
            self.assertIs(app.tess_panel.column, app.right_bottom_column)
            self.assertIn(app.tone_panel, app.right_column.panels)
            app.hide_panel(app.tone_panel)
            self.assertTrue(app.tone_panel.hidden)
            self.assertNotIn(app.tone_panel, app.right_column.panels)
            self.assertTrue(app.tone_panel.winfo_exists())
            app.show_panel(app.tone_panel)
            self.assertFalse(app.tone_panel.hidden)
            self.assertIn(app.tone_panel, app.right_column.panels)
            app.preview_panel.pop_out()
            self.assertTrue(app.preview_panel.is_floating)
            app.hide_panel(app.coverage_panel)
            app.reset_layout()
            self.assertFalse(app.preview_panel.is_floating)
            self.assertFalse(app.coverage_panel.hidden)
            self.assertEqual(app.left_column.panels, [app.preview_panel, app.coverage_panel])
            self.assertEqual(
                app.right_column.panels,
                [
                    app.wheel_panel,
                    app.texture_panel,
                    app.tone_panel,
                    app.scale_panel,
                    app.crop_panel,
                ],
            )
            self.assertFalse(app.tess_panel.hidden)
            self.assertFalse(app.layers_panel.hidden)
            self.assertFalse(app.labels_panel.hidden)
            self.assertEqual(
                app.right_bottom_column.panels,
                [
                    app.layers_panel,
                    app.labels_panel,
                    app.tess_panel,
                    app.history_panel,
                ],
            )
            self.assertTrue(app.tess_panel.expanded)
            self.assertEqual(str(app.tess_panel.body.winfo_manager()), "pack")
            app.tess_panel.set_expanded(False)
            self.assertFalse(app.tess_panel.expanded)
            self.assertEqual(str(app.tess_panel.body.winfo_manager()), "")
            self.assertEqual(str(app.tess_panel.bar.winfo_manager()), "pack")
            app.tess_panel.set_expanded(True)
            mapped_tabs = set(app.notebook.tabs())
            self.assertIn(str(app.tile_zoom_host), mapped_tabs)
            self.assertIn(str(app.seam_zoom_host), mapped_tabs)
            self.assertIn(str(app.mock_tab), mapped_tabs)
            # Old layout_profiles.json used the pane title "Crop"
            app._apply_layout_spec(
                {
                    "left": ["Preview", "Coverage"],
                    "right": [
                        "Color wheel",
                        "Texture",
                        "Tone",
                        "Scale",
                        "Crop",
                        "Tessellate",
                        "Layers",
                        "Labels",
                    ],
                    "hidden": [],
                    "sash_fraction": 0.72,
                }
            )
            self.assertIn(app.crop_panel, app.right_column.panels)
            self.assertEqual(str(app.crop_panel.panel_title), "Position & Zoom")
        finally:
            root.destroy()


def _menu_labels(menu) -> list[str]:
    """Visible labels on a Tk menu (skips separators)."""
    end = menu.index("end")
    if end is None:
        return []
    labels: list[str] = []
    for i in range(int(end) + 1):
        try:
            labels.append(str(menu.entrycget(i, "label")))
        except Exception:
            continue
    return labels


class TestMenubarEditState(unittest.TestCase):
    """File / Edit / View / Tools / Help; .wpedit round-trip."""
    def test_layout_profiles_path_next_to_presets(self) -> None:
        from wallpaper_recolor.color.presets import default_presets_path
        from wallpaper_recolor.ui.app import default_layout_profiles_path

        self.assertEqual(
            default_layout_profiles_path(),
            default_presets_path().with_name("layout_profiles.json"),
        )

    def test_menubar_file_edit_view_help(self) -> None:
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            self.assertEqual(str(app.root.cget("menu")), str(app.menubar))
            self.assertEqual(_menu_labels(app.menubar), ["File", "Edit", "View", "Tools", "Help"])
            file_labels = _menu_labels(app.file_menu)
            self.assertIn("Open image…", file_labels)
            self.assertIn("Save as…", file_labels)
            self.assertIn("Export job pack…", file_labels)
            self.assertIn("Export layers zip…", file_labels)
            self.assertIn("Save Wallpaper Edit state…", file_labels)
            self.assertIn("Open Edit state…", file_labels)
            self.assertIn("Exit", file_labels)
            self.assertTrue(hasattr(app, "open_btn"))
            self.assertTrue(hasattr(app, "tools_combo"))
            self.assertEqual(app.pointer_tool.get(), ui_mod.TOOL_VIEW_MOVE)
            self.assertEqual(app.pointer_tool_label.get(), "View Move")
            self.assertEqual(list(app.tools_combo.cget("values")), ["View Move", "Grab Move"])
            app._rebuild_tools_menu()
            self.assertEqual(_menu_labels(app.tools_menu), ["View Move", "Grab Move"])
            self.assertFalse(hasattr(app, "save_btn"))
            self.assertFalse(hasattr(app, "export_btn"))
            self.assertFalse(hasattr(app, "export_layers_btn"))
            app._rebuild_edit_menu()
            edit_labels = _menu_labels(app.edit_menu)
            self.assertTrue(any(label.startswith("Undo") for label in edit_labels))
            self.assertTrue(any(label.startswith("Redo") for label in edit_labels))
            self.assertIn("Reset colors", edit_labels)
            self.assertIn("Normalize lighting", edit_labels)
            self.assertIn("Tessellate Build", edit_labels)
            self.assertIn("Text", edit_labels)
            self.assertNotIn("Detect", edit_labels)
            self.assertNotIn("Remove", edit_labels)
            self.assertNotIn("Clear", edit_labels)
            self.assertNotIn("Mark", edit_labels)
            self.assertNotIn("Place", edit_labels)
            self.assertNotIn("Change-to", edit_labels)
            text_labels = _menu_labels(app.text_menu)
            self.assertEqual(
                text_labels,
                ["Detect", "Remove", "Clear", "Mark", "Place", "Change-to"],
            )
            app.hide_panel(app.labels_panel)
            app._rebuild_edit_menu()
            self.assertIn("Text", _menu_labels(app.edit_menu))
            self.assertEqual(
                _menu_labels(app.text_menu),
                ["Detect", "Remove", "Clear", "Mark", "Place", "Change-to"],
            )
            help_labels = _menu_labels(app.help_menu)
            self.assertTrue(any("About" in label for label in help_labels))
        finally:
            root.destroy()

    def test_undo_menu_label_includes_stack_length(self) -> None:
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            im = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB")
            app.work_image = im
            app.source_image = im
            app.rebuild_ranges()
            app._clear_history()
            before = app._capture_edit()
            app.crop_zoom.set(2.0)
            app._sync_crop_bounds(clamp=True)
            app._push_undo_state(before)
            self.assertEqual(len(app._undo_stack), 1)
            app._rebuild_edit_menu()
            undo_label = str(app.edit_menu.entrycget(0, "label"))
            try:
                undo_acc = str(app.edit_menu.entrycget(0, "accelerator"))
            except tk.TclError:
                undo_acc = ""
            self.assertIn("1", undo_label + " " + undo_acc)
            before2 = app._capture_edit()
            app.crop_zoom.set(3.0)
            app._sync_crop_bounds(clamp=True)
            app._push_undo_state(before2)
            self.assertEqual(len(app._undo_stack), 2)
            undo_label = str(app.edit_menu.entrycget(0, "label"))
            try:
                undo_acc = str(app.edit_menu.entrycget(0, "accelerator"))
            except tk.TclError:
                undo_acc = ""
            self.assertIn("2", undo_label + " " + undo_acc)
        finally:
            root.destroy()

    def test_edit_state_round_trip_crop_zoom(self) -> None:
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            with tempfile.TemporaryDirectory() as tmp:
                img_path = Path(tmp) / "tiny.png"
                Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB").save(img_path)
                self.assertTrue(app._open_image_from_path(img_path, reset_edits=True))
                app.crop_zoom.set(2.0)
                app._sync_crop_bounds(clamp=True)
                app.contrast_pct.set(40.0)
                app.balance_cyan_pct.set(50.0)
                app.temperature_pct.set(-30.0)
                app._sync_tone_to_map()
                before = app._capture_edit()
                state_path = Path(tmp) / "tiny_edit.wpedit"
                app._write_edit_state(state_path)
                payload = json.loads(state_path.read_text(encoding="utf-8"))
                self.assertEqual(payload.get("format"), "wpedit")
                self.assertAlmostEqual(float(payload["crop"]["zoom"]), 2.0, places=4)
                self.assertAlmostEqual(float(payload["tone"]["contrast"]), 0.4, places=4)
                self.assertAlmostEqual(float(payload["tone"]["balance_cyan"]), 0.5, places=4)
                self.assertAlmostEqual(float(payload["tone"]["lights_cyan"]), 0.5, places=4)
                self.assertAlmostEqual(float(payload["tone"]["temperature"]), -0.3, places=4)
                app.crop_zoom.set(1.0)
                app.contrast_pct.set(0.0)
                app.balance_cyan_pct.set(0.0)
                app.temperature_pct.set(0.0)
                app._sync_tone_to_map()
                app._read_edit_state(state_path)
                self.assertAlmostEqual(app.crop_zoom.get(), 2.0, places=4)
                self.assertAlmostEqual(app.contrast_pct.get(), 40.0, delta=0.6)
                self.assertAlmostEqual(app.balance_cyan_pct.get(), 50.0, delta=0.6)
                self.assertAlmostEqual(app.temperature_pct.get(), -30.0, delta=0.6)
                app.balance_cyan_pct.set(0.0)
                app._sync_tone_to_map()
                app._restore_edit(before)
                self.assertAlmostEqual(app.balance_cyan_pct.get(), 50.0, delta=0.6)
                self.assertAlmostEqual(app.temperature_pct.get(), -30.0, delta=0.6)
                assert app.range_map is not None
                self.assertAlmostEqual(app.range_map.tone_balance_cyan, 0.5, places=4)
                self.assertAlmostEqual(app.range_map.tone_lights_cyan, 0.5, places=4)
        finally:
            root.destroy()

    def test_original_pane_title_shows_image_basename(self) -> None:
        """Original header is Original until a file is open, then Original (basename)."""
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            self.assertEqual(app.orig_title.get(), "Original")
            self.assertEqual(str(app.orig_title_label.cget("text")), "Original")
            result_texts = [
                str(w.cget("text"))
                for w in app.orig_title_label.master.grid_slaves(row=0)
                if str(w.cget("text")).startswith("Result")
            ]
            self.assertEqual(result_texts, ["Result"])
            with tempfile.TemporaryDirectory() as tmp:
                first = Path(tmp) / "living_room.tif"
                Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB").save(first)
                self.assertTrue(app._open_image_from_path(first, reset_edits=True))
                self.assertEqual(app.orig_title.get(), "Original (living_room.tif)")
                self.assertEqual(str(app.orig_title_label.cget("text")), "Original (living_room.tif)")
                self.assertNotIn(str(first), app.orig_title.get())
                result_after = [
                    str(w.cget("text"))
                    for w in app.orig_title_label.master.grid_slaves(row=0)
                    if str(w.cget("text")).startswith("Result")
                ]
                self.assertEqual(result_after, ["Result"])
                self.assertNotIn("living_room.tif", result_after[0])

                second = Path(tmp) / "kitchen.png"
                Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB").save(second)
                state_path = Path(tmp) / "living_edit.wpedit"
                app._write_edit_state(state_path)
                self.assertTrue(app._open_image_from_path(second, reset_edits=True))
                self.assertEqual(app.orig_title.get(), "Original (kitchen.png)")
                app._read_edit_state(state_path)
                self.assertEqual(app.orig_title.get(), "Original (living_room.tif)")
                self.assertEqual(str(app.orig_title_label.cget("text")), "Original (living_room.tif)")
        finally:
            root.destroy()


class TestLayersZip(unittest.TestCase):
    """Per-range masks + composite zip for analog plates."""
    def _two_range_image(self) -> tuple[Image.Image, object]:
        arr = np.zeros((16, 32, 3), dtype=np.uint8)
        arr[:, :16] = (220, 20, 20)
        arr[:, 16:] = (20, 180, 40)
        im = Image.fromarray(arr, mode="RGB")
        range_map = build_range_map(im, 2, SPLIT_COLOR_CLOSENESS)
        range_map.set_replacement(0, (10, 20, 200))
        range_map.set_replacement(1, (200, 180, 20))
        range_map.texture_enabled = True
        range_map.texture_strength = TEXTURE_DEFAULT_STRENGTH
        return im, snapshot_assignment(range_map)

    def test_zip_contents_and_normal_stack_matches_composite(self) -> None:
        """2 visible ranges → 2 color TIF/SVG + texture + composite; Normal stack = composite."""
        import zipfile

        from wallpaper_recolor.io.export_layers_zip import export_layers_zip
        from wallpaper_recolor.ui import run

        self.assertTrue(callable(run))
        im, snap = self._two_range_image()
        with tempfile.TemporaryDirectory() as tmp:
            zpath = Path(tmp) / "layers.zip"
            export_layers_zip(zpath, im, snap, output_dpi=300)
            self.assertTrue(zpath.is_file())
            with zipfile.ZipFile(zpath) as zf:
                names = set(zf.namelist())
                color_tifs = sorted(
                    n for n in names if n.endswith(".tif") and n[:2].isdigit() and n != "00_original.tif"
                )
                color_svgs = sorted(
                    n for n in names if n.endswith(".svg") and n[:2].isdigit()
                )
                self.assertEqual(len(color_tifs), 2, names)
                self.assertEqual(len(color_svgs), 2, names)
                self.assertIn("texture.tif", names)
                self.assertIn("texture.svg", names)
                self.assertIn("composite.tif", names)
                self.assertIn("composite.png", names)
                self.assertIn("composite.svg", names)
                self.assertIn("00_original.tif", names)
                self.assertIn("palette.json", names)
                self.assertIn("README.txt", names)
                palette = json.loads(zf.read("palette.json"))
                self.assertEqual(len(palette["layers"]), 4)  # 00 + 2 colors + texture
                hexes = [layer["hex"] for layer in palette["layers"] if layer.get("role") == "color"]
                self.assertEqual(hexes, ["#0A14C8", "#C8B414"])
                comp_svg = zf.read("composite.svg").decode("utf-8")
                self.assertIn('href="00_original.png"', comp_svg)
                self.assertNotIn("mix-blend-mode", comp_svg)
                self.assertNotIn("<path", comp_svg)  # wrapper, not a trace
                for tif_name in color_tifs:
                    png_name = tif_name[:-4] + ".png"
                    svg_name = tif_name[:-4] + ".svg"
                    self.assertIn(png_name, names)
                    self.assertIn(svg_name, names)
                    svg = zf.read(svg_name).decode("utf-8")
                    self.assertIn(f'href="{png_name}"', svg)
                    self.assertIn("viewBox=", svg)
                    self.assertIn("mm", svg)  # 300 DPI → print size
                tex_svg = zf.read("texture.svg").decode("utf-8")
                self.assertIn('href="texture.png"', tex_svg)
                self.assertIn("mix-blend-mode: luminosity", tex_svg)

                extract = Path(tmp) / "out"
                zf.extractall(extract)

            color_imgs = [
                Image.open(extract / name).convert("RGBA")
                for name in sorted(p.name for p in extract.glob("0[1-9]_*.tif"))
            ]
            self.assertEqual(len(color_imgs), 2)
            stacked = Image.new("RGBA", color_imgs[0].size, (0, 0, 0, 0))
            for plate in color_imgs:
                stacked = Image.alpha_composite(stacked, plate)
            composite = Image.open(extract / "composite.tif").convert("RGB")
            stacked_rgb = Image.new("RGB", stacked.size, (0, 0, 0))
            stacked_rgb.paste(stacked, mask=stacked.split()[-1])
            np.testing.assert_array_equal(
                np.asarray(stacked_rgb),
                np.asarray(composite),
            )
            self.assertEqual(composite.size, im.size)

    def test_texture_eye_off_omits_texture_plate(self) -> None:
        import zipfile

        from wallpaper_recolor.io.export_layers_zip import export_layers_zip

        im, snap = self._two_range_image()
        snap.texture_enabled = False
        with tempfile.TemporaryDirectory() as tmp:
            zpath = Path(tmp) / "exact.zip"
            export_layers_zip(zpath, im, snap)
            with zipfile.ZipFile(zpath) as zf:
                names = set(zf.namelist())
            self.assertNotIn("texture.tif", names)
            self.assertIn("composite.tif", names)
            color_tifs = [
                n for n in names if n.endswith(".tif") and n[:2].isdigit() and n != "00_original.tif"
            ]
            self.assertEqual(len(color_tifs), 2)

    def test_small_export_finishes_quickly(self) -> None:
        """Keep TIF+PNG (SVG hrefs PNG); encode must stay fast (no PNG optimize)."""
        import time
        import zipfile

        from wallpaper_recolor.io.export_layers_zip import export_layers_zip

        arr = np.zeros((96, 96, 3), dtype=np.uint8)
        arr[:, :48] = (220, 20, 20)
        arr[:, 48:] = (20, 180, 40)
        im = Image.fromarray(arr, mode="RGB")
        range_map = build_range_map(im, 2, SPLIT_COLOR_CLOSENESS)
        range_map.set_replacement(0, (10, 20, 200))
        range_map.set_replacement(1, (200, 180, 20))
        range_map.texture_enabled = True
        range_map.texture_strength = TEXTURE_DEFAULT_STRENGTH
        snap = snapshot_assignment(range_map)
        with tempfile.TemporaryDirectory() as tmp:
            zpath = Path(tmp) / "layers.zip"
            started = time.perf_counter()
            export_layers_zip(zpath, im, snap, output_dpi=72)
            elapsed = time.perf_counter() - started
            self.assertLess(elapsed, 8.0, f"small layers zip took {elapsed:.2f}s")
            with zipfile.ZipFile(zpath) as zf:
                names = set(zf.namelist())
                info = {item.filename: item for item in zf.infolist()}
            self.assertIn("composite.tif", names)
            self.assertIn("composite.png", names)
            self.assertIn("00_original.png", names)
            self.assertEqual(info["composite.tif"].compress_type, zipfile.ZIP_STORED)
            self.assertEqual(info["composite.png"].compress_type, zipfile.ZIP_STORED)
            self.assertEqual(info["palette.json"].compress_type, zipfile.ZIP_DEFLATED)

    def test_ui_has_layers_zip_button(self) -> None:
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            self.assertIn("Export layers zip…", _menu_labels(app.file_menu))
            self.assertTrue(callable(app.export_layers_zip))
            self.assertFalse(hasattr(app, "export_layers_btn"))
            src = inspect.getsource(app._run_background)
            self.assertIn("daemon", src)
        finally:
            root.destroy()


class TestPresetsAndEyedrop(unittest.TestCase):
    """Named palettes force range_count; Original click samples into the swatch."""
    def test_seed_palettes_delete_v6n_and_no_reseed(self) -> None:
        from wallpaper_recolor.color.presets import (
            BLACK,
            BLUE_DARK,
            BLUE_LIGHT,
            BLUE_MID,
            GENERIC_LABEL,
            GRAY,
            GREEN_DARK,
            GREEN_LIGHT,
            GREEN_MID,
            RED_DARK,
            RED_LIGHT,
            RED_MID,
            WHITE,
            default_seed_presets,
            delete_user_preset,
            ensure_default_presets,
            get_preset,
            list_presets,
        )
        from wallpaper_recolor.ui import run

        self.assertTrue(callable(run))
        seeds = default_seed_presets()
        self.assertEqual(
            [p.name for p in seeds],
            [
                "V6-N",
                "White and Black",
                "White Gray Black",
                "Analogous Reds",
                "Analogous Greens",
                "Analogous Blues",
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "presets.json"
            ensure_default_presets(path)
            names = [p.name for p in list_presets(path)]
            self.assertEqual(names[0], "V6-N")
            wb = get_preset("White and Black", path)
            assert wb is not None
            self.assertEqual(wb.range_count, 2)
            self.assertEqual(wb.palette_rgb, (WHITE, BLACK))
            self.assertEqual(wb.match_palette_rgb, (WHITE, BLACK))
            self.assertTrue(wb.palette_as_centers)
            self.assertEqual(wb.split_method, SPLIT_COLOR_CLOSENESS)
            self.assertEqual(wb.weights, (0.5, 0.5))
            gray = get_preset("White Gray Black", path)
            assert gray is not None
            self.assertEqual(gray.palette_rgb, (WHITE, GRAY, BLACK))
            reds = get_preset("Analogous Reds", path)
            assert reds is not None
            self.assertEqual(reds.palette_rgb, (RED_DARK, RED_MID, RED_LIGHT))
            greens = get_preset("Analogous Greens", path)
            assert greens is not None
            self.assertEqual(greens.palette_rgb, (GREEN_DARK, GREEN_MID, GREEN_LIGHT))
            blues = get_preset("Analogous Blues", path)
            assert blues is not None
            self.assertEqual(blues.palette_rgb, (BLUE_DARK, BLUE_MID, BLUE_LIGHT))
            self.assertTrue(delete_user_preset("V6-N", path))
            self.assertIsNone(get_preset("V6-N", path))
            ensure_default_presets(path)
            self.assertIsNone(get_preset("V6-N", path))
            self.assertIsNotNone(get_preset("White and Black", path))
            self.assertNotEqual(GENERIC_LABEL, "V6-N")

    def test_ui_generic_eyedrop_overwrite_and_delete(self) -> None:
        import tkinter as tk

        from types import SimpleNamespace

        from PIL import ImageTk

        import wallpaper_recolor.color.presets as presets_mod
        import wallpaper_recolor.ui.app as ui_mod
        from wallpaper_recolor.ui.coverage_bar import HALF_MATCH, HALF_REPLACE
        from wallpaper_recolor.color.presets import GENERIC_LABEL, WHITE
        from wallpaper_recolor.ui import run

        self.assertTrue(callable(run))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "presets.json"
            with patch.object(presets_mod, "default_presets_path", return_value=path):
                root = tk.Tk()
                root.withdraw()
                try:
                    app = ui_mod.WallpaperRecolorApp(root)
                    values = list(app.preset_combo.cget("values"))
                    self.assertEqual(values[0], GENERIC_LABEL)
                    self.assertIn("V6-N", values)
                    self.assertIn("White and Black", values)
                    self.assertIn("Analogous Reds", values)
                    self.assertEqual(app.preset_choice.get(), GENERIC_LABEL)
                    self.assertEqual(str(app.delete_preset_btn.cget("state")), "disabled")

                    im = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB")
                    im.putpixel((0, 0), (200, 10, 10))
                    im.putpixel((1, 0), (10, 200, 10))
                    app.work_image = im
                    app.source_image = im
                    app.rebuild_ranges()
                    assert app.range_map is not None
                    self.assertEqual(len(app.range_map.ranges), 4)
                    self.assertEqual(
                        app.range_map.ranges[0].match_rgb,
                        app.range_map.ranges[0].replacement_rgb,
                    )

                    app.preset_choice.set("White and Black")
                    app.apply_selected_preset()
                    self.assertEqual(len(app.range_map.ranges), 2)
                    self.assertEqual(app.range_map.ranges[0].replacement_rgb, WHITE)
                    self.assertNotEqual(app.range_map.ranges[0].match_rgb, WHITE)

                    app.preset_choice.set(GENERIC_LABEL)
                    app.apply_selected_preset()
                    self.assertIsNone(app.preset_id)
                    self.assertEqual(app.range_by.get(), RANGE_BY_COLOR_LABEL)
                    self.assertNotEqual(app.range_map.ranges[0].match_rgb, WHITE)

                    icons = ROOT / "wallpaper_recolor" / "icons"
                    svg = icons / "eye-dropper-solid-full.svg"
                    png = icons / "eye-dropper-solid-full.png"
                    self.assertTrue(svg.is_file())
                    self.assertTrue(png.is_file())
                    self.assertTrue((icons / "eye-solid-full.svg").is_file())
                    self.assertTrue((icons / "eye-slash-solid-full.svg").is_file())
                    self.assertTrue((icons / "magnifying-glass-plus-solid-full.svg").is_file())
                    self.assertTrue((icons / "magnifying-glass-minus-solid-full.svg").is_file())
                    self.assertIsNotNone(app._eyedrop_photo)
                    self.assertIsNone(app.coverage.eyedrop_btn)
                    self.assertIs(app.coverage.eyedrop_photo, app._eyedrop_photo)
                    self.assertEqual(
                        [c for c in app.coverage.winfo_children() if c.winfo_class() == "Label"],
                        [],
                    )
                    app._sync_eyedrop_cursor()
                    self.assertIn(str(app.orig_label.cget("cursor")), ("none", "crosshair"))
                    app._on_orig_eyedrop_move(SimpleNamespace(x=4, y=4))
                    self.assertEqual(str(app._eyedrop_overlay.winfo_manager()), "place")
                    disp = Image.new("RGB", (48, 48), (80, 40, 20))
                    app._orig_pil = disp
                    app._orig_photo = ImageTk.PhotoImage(disp, master=root)
                    with patch.object(app, "_orig_click_to_display", return_value=(8, 8)):
                        app._place_eyedrop_overlay(8, 8)
                    self.assertIsNotNone(app._orig_eyedrop_photo)
                    self.assertEqual(str(app._eyedrop_overlay.winfo_manager()), "")
                    app._hide_eyedrop_overlay()
                    self.assertEqual(str(app._eyedrop_overlay.winfo_manager()), "")
                    self.assertIsNone(app._orig_eyedrop_photo)
                    app._on_orig_eyedrop_move(SimpleNamespace(x=4, y=4))

                    app.select_range(0, HALF_MATCH)
                    app._apply_eyedrop_rgb((200, 10, 10))
                    self.assertEqual(app.range_map.ranges[0].match_rgb, (200, 10, 10))
                    app.select_range(0, HALF_REPLACE)
                    app._apply_eyedrop_rgb((10, 200, 10))
                    self.assertEqual(app.range_map.ranges[0].replacement_rgb, (10, 200, 10))
                    self.assertEqual(app.range_map.ranges[0].match_rgb, (200, 10, 10))

                    with patch.object(ui_mod.simpledialog, "askstring", return_value="White and Black"):
                        with patch.object(ui_mod.messagebox, "askyesno", return_value=True) as ask:
                            app.save_preset()
                            ask.assert_called_once()
                    self.assertEqual(app.preset_choice.get(), "White and Black")
                    saved = presets_mod.get_preset("White and Black")
                    assert saved is not None
                    self.assertEqual(saved.bands[0].match_rgb, (200, 10, 10))

                    app.preset_choice.set("V6-N")
                    app.apply_selected_preset()
                    with patch.object(ui_mod.messagebox, "askyesno", return_value=True):
                        app.delete_selected_preset()
                    self.assertIsNone(presets_mod.get_preset("V6-N"))
                    self.assertEqual(app.preset_choice.get(), GENERIC_LABEL)
                    values = list(app.preset_combo.cget("values"))
                    self.assertEqual(values[0], GENERIC_LABEL)
                    self.assertNotIn("V6-N", values)
                finally:
                    root.destroy()

    def test_fa_svg_evenodd_holes_and_loupe(self) -> None:
        """FA compound glyphs punch holes; loupe is a 110px circle with transparent corners."""
        import tkinter as tk

        from PIL import ImageTk

        import wallpaper_recolor.ui.app as ui_mod

        icons = ROOT / "wallpaper_recolor" / "icons"
        eye = ui_mod._rasterize_fa_svg(icons / "eye-solid-full.svg", 64, (34, 34, 34, 255))
        self.assertLess(eye.getpixel((32, 20))[3], 40, "eye sclera / pupil hole should be transparent")
        self.assertGreater(eye.getpixel((32, 32))[3], 200)
        drop = ui_mod._rasterize_fa_svg(icons / "eye-dropper-solid-full.svg", 64, (250, 250, 250, 255))
        self.assertLess(drop.getpixel((32, 32))[3], 40, "dropper interior should be a hole, not a blob")
        drop_halo = ui_mod._rasterize_fa_svg(
            icons / "eye-dropper-solid-full.svg",
            22,
            ui_mod._EYEDROP_ICON_FG,
            halo=ui_mod._EYEDROP_ICON_HALO,
        )
        self.assertLess(drop_halo.getpixel((10, 11))[3], 80, "halo must follow the outline, not fill the tube")
        slash = ui_mod._rasterize_fa_svg(icons / "eye-slash-solid-full.svg", 16, (34, 34, 34, 255))
        self.assertLess(slash.getpixel((8, 5))[3], 80)

        src = Image.new("RGB", (40, 40), (200, 10, 10))
        src.putpixel((20, 20), (10, 200, 10))
        loupe = ui_mod._make_eyedrop_loupe_image(src, 20, 20)
        self.assertEqual(loupe.size, (ui_mod._LOUPE_PX, ui_mod._LOUPE_PX))
        self.assertEqual(ui_mod._LOUPE_PX, 110)
        self.assertEqual(ui_mod._LOUPE_ZOOM, 10)
        self.assertEqual(ui_mod._LOUPE_GAP, 16)
        lw, lh = loupe.size
        for corner in ((0, 0), (lw - 1, 0), (0, lh - 1), (lw - 1, lh - 1)):
            self.assertEqual(loupe.getpixel(corner)[3], 0, f"loupe corner {corner} must be alpha 0")
        self.assertGreater(loupe.getpixel((ui_mod._LOUPE_PX // 2, ui_mod._LOUPE_PX // 2))[3], 200)
        half = ui_mod._LOUPE_SRC_PX // 2
        z = ui_mod._LOUPE_ZOOM
        c0 = half * z
        self.assertEqual(loupe.getpixel((c0 + z // 2, c0 + z // 2))[:3], (10, 200, 10))
        paper = Image.new("RGBA", (200, 200), (200, 10, 10, 255))
        layer = Image.new("RGBA", paper.size, (0, 0, 0, 0))
        ui_mod._paste_rgba(layer, loupe, (40, 40))
        stamped = Image.alpha_composite(paper, layer)
        self.assertEqual(stamped.getpixel((40, 40))[:3], (200, 10, 10), "loupe corner must show wallpaper, not a black square")

        drop_img = ui_mod._eyedrop_icon_image()
        hx, hy = ui_mod._glyph_hotspot(drop_img)
        alpha = np.array(drop_img.split()[-1])
        ys, xs = np.where(alpha > 32)
        self.assertGreater(len(xs), 0)
        self.assertGreaterEqual(hy, int(ys.max()) - 1)
        self.assertEqual(hx, int(xs[ys == hy].min()))
        self.assertGreater(hy, drop_img.size[1] // 2)
        self.assertLess(hx, drop_img.size[0] // 2)
        w, h = drop_img.size
        for corner in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
            self.assertLess(drop_img.getpixel(corner)[3], 16, f"dropper corner {corner} must be transparent")
        png = icons / "eye-dropper-solid-full.png"
        self.assertTrue(png.is_file())
        cached = Image.open(png).convert("RGBA")
        cw, ch = cached.size
        for corner in ((0, 0), (cw - 1, 0), (0, ch - 1), (cw - 1, ch - 1)):
            self.assertLess(cached.getpixel(corner)[3], 16, f"cached PNG corner {corner} must be transparent")
        self.assertFalse(ui_mod._icon_has_opaque_plate(drop_img))
        self.assertFalse(ui_mod._icon_has_opaque_plate(cached))

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            self.assertEqual(int(str(app._eyedrop_overlay.cget("highlightthickness"))), 0)
            self.assertEqual(str(app._eyedrop_overlay.cget("bg")), str(app.orig_host.cget("bg")))
            app._orig_pil = src
            app._orig_photo = ImageTk.PhotoImage(src, master=root)
            with patch.object(ui_mod, "_make_eyedrop_loupe_image", wraps=ui_mod._make_eyedrop_loupe_image) as made:
                with patch.object(app, "_orig_click_to_display", return_value=(20, 20)):
                    app._place_eyedrop_overlay(20, 20)
                made.assert_called()
                self.assertEqual(made.call_args[0][1:], (20, 20))
            self.assertIsNotNone(app._orig_eyedrop_photo)
            app._hide_eyedrop_overlay()
            self.assertIsNone(app._orig_eyedrop_photo)
        finally:
            root.destroy()


class TestRoomMockup(unittest.TestCase):
    """Inspection tab: wallpaper on a room plate; not the Composite camera."""
    """Back-wall wallpaper cover: full / half / third / quarter from the floor."""

    _PAPER = (220, 30, 180)

    def _wall_geom(self, width: int = 1200, height: int = 780):
        from wallpaper_recolor.preview.preview_tools import make_room_plate

        _, (x0, y0, x1, y1) = make_room_plate(width, height)
        wall_w, wall_h = x1 - x0, y1 - y0
        mx = x0 + wall_w // 2
        bb = max(6, wall_h // 70)
        top = (mx, y0 + 6)
        near_floor = (mx, y1 - bb - 8)
        return (x0, y0, x1, y1), wall_h, bb, top, near_floor

    def _paper_tile(self) -> Image.Image:
        return Image.new("RGB", (48, 48), self._PAPER)

    def _is_paper(self, rgb: tuple[int, int, int]) -> bool:
        return abs(rgb[0] - self._PAPER[0]) < 12 and abs(rgb[1] - self._PAPER[1]) < 12

    def _is_wall_paint(self, rgb: tuple[int, int, int]) -> bool:
        from wallpaper_recolor.preview.preview_tools import BACK_WALL_RGB

        return all(abs(int(a) - int(b)) < 8 for a, b in zip(rgb, BACK_WALL_RGB))

    def test_full_cover_paints_whole_back_wall(self) -> None:
        from wallpaper_recolor.preview.preview_tools import MOCKUP_COVER_FULL, room_mockup

        _, wall_h, _bb, top, near_floor = self._wall_geom()
        mock = room_mockup(self._paper_tile(), repeats_x=4.0, cover_frac=MOCKUP_COVER_FULL)
        self.assertTrue(self._is_paper(mock.getpixel(top)), mock.getpixel(top))
        self.assertTrue(self._is_paper(mock.getpixel(near_floor)), mock.getpixel(near_floor))
        # Default matches full cover
        default = room_mockup(self._paper_tile(), repeats_x=4.0)
        self.assertEqual(default.getpixel(top), mock.getpixel(top))
        self.assertGreater(wall_h, 100)

    def test_partial_cover_only_lower_fraction(self) -> None:
        from wallpaper_recolor.preview.preview_tools import (
            MOCKUP_COVER_HALF,
            MOCKUP_COVER_QUARTER,
            MOCKUP_COVER_THIRD,
            room_mockup,
        )

        (x0, y0, x1, y1), wall_h, bb, top, near_floor = self._wall_geom()
        mx = x0 + (x1 - x0) // 2
        cases = (
            (MOCKUP_COVER_HALF, "half"),
            (MOCKUP_COVER_THIRD, "third"),
            (MOCKUP_COVER_QUARTER, "quarter"),
        )
        for frac, name in cases:
            with self.subTest(cover=name):
                mock = room_mockup(self._paper_tile(), repeats_x=4.0, cover_frac=frac)
                self.assertTrue(
                    self._is_wall_paint(mock.getpixel(top)),
                    f"{name} top {mock.getpixel(top)} should be wall paint",
                )
                self.assertTrue(
                    self._is_paper(mock.getpixel(near_floor)),
                    f"{name} floor {mock.getpixel(near_floor)} should be wallpaper",
                )
                cover_h = int(round(wall_h * frac))
                cut_y = y1 - cover_h
                above = (mx, cut_y - 4)
                below = (mx, min(cut_y + 4, y1 - bb - 4))
                self.assertGreater(cut_y, y0 + 8)
                self.assertTrue(
                    self._is_wall_paint(mock.getpixel(above)),
                    f"{name} above cut {mock.getpixel(above)}",
                )
                self.assertTrue(
                    self._is_paper(mock.getpixel(below)),
                    f"{name} below cut {mock.getpixel(below)}",
                )

    def test_ui_wall_cover_control(self) -> None:
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod
        from wallpaper_recolor.preview.preview_tools import MOCKUP_COVER_FRACS

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            self.assertEqual(app.mockup_cover.get(), "full")
            self.assertAlmostEqual(app._mockup_cover_frac(), 1.0)
            self.assertAlmostEqual(app.mockup_repeats.get(), ui_mod.DEFAULT_MOCKUP_REPEATS)
            self.assertLess(
                float(app.mockup_scale.cget("from")),
                float(app.mockup_scale.cget("to")),
            )
            self.assertLess(float(app.crop_zoom_scale.cget("from")), float(app.crop_zoom_scale.cget("to")))
            self.assertLess(float(app.tess_tiles_scale.cget("from")), float(app.tess_tiles_scale.cget("to")))
            self.assertLess(float(app.tess_lloyd_scale.cget("from")), float(app.tess_lloyd_scale.cget("to")))
            self.assertFalse(hasattr(app, "tess_strength_scale"))
            for key, frac in MOCKUP_COVER_FRACS.items():
                app.mockup_cover.set(key)
                self.assertAlmostEqual(app._mockup_cover_frac(), frac)
            layout = inspect.getsource(ui_mod.WallpaperRecolorApp._build_layout)
            self.assertIn("Wall cover:", layout)
            self.assertIn("(from floor)", layout)
            self.assertIn("selection_clear", inspect.getsource(ui_mod.WallpaperRecolorApp._clear_combo_selection))
            self.assertIn("focus_set", inspect.getsource(ui_mod.WallpaperRecolorApp._defocus_readonly_combo))
        finally:
            root.destroy()

    def test_export_pack_uses_cover_frac(self) -> None:
        from wallpaper_recolor.io.export_pack import export_job_pack
        from wallpaper_recolor.preview.preview_tools import make_room_plate

        arr = np.full((32, 32, 3), self._PAPER, dtype=np.uint8)
        im = Image.fromarray(arr, mode="RGB")
        range_map = build_range_map(im, 2, SPLIT_COLOR_CLOSENESS)
        range_map.texture_enabled = False
        range_map.set_replacement(0, self._PAPER)
        range_map.set_replacement(1, self._PAPER)
        range_map.tone_lights_cyan = 0.3
        range_map.tone_darks_yellow = -0.2
        snap = snapshot_assignment(range_map)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "pack"
            export_job_pack(
                dest,
                im,
                snap,
                mockup_repeats=4.0,
                mockup_cover_frac=0.5,
            )
            mock = Image.open(dest / "room_mockup.png").convert("RGB")
            _, (x0, y0, x1, y1) = make_room_plate()
            mx = x0 + (x1 - x0) // 2
            self.assertTrue(self._is_wall_paint(mock.getpixel((mx, y0 + 6))))
            bb = max(6, (y1 - y0) // 70)
            self.assertTrue(self._is_paper(mock.getpixel((mx, y1 - bb - 8))))
            palette = json.loads((dest / "palette.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(float(palette["mockup_wall_cover_frac"]), 0.5)
            self.assertAlmostEqual(float(palette["tone_lights_cyan"]), 0.3)
            self.assertAlmostEqual(float(palette["tone_darks_yellow"]), -0.2)
            self.assertAlmostEqual(float(palette["tone_balance_cyan"]), 0.0)
            self.assertAlmostEqual(float(palette["tone_temperature"]), 0.0)


class TestColorWheelMixBars(unittest.TestCase):
    """htmlcolorcodes-style Tailwind / Shades / Tints / Tones under the wheel."""

    def test_shade_tint_tone_endpoints(self) -> None:
        from wallpaper_recolor.ui.color_wheel import shade_rgb, tint_rgb, tone_rgb

        base = (220, 40, 80)
        self.assertEqual(shade_rgb(base, 0.0), base)
        self.assertEqual(shade_rgb(base, 1.0), (0, 0, 0))
        self.assertEqual(tint_rgb(base, 0.0), (255, 255, 255))
        self.assertEqual(tint_rgb(base, 1.0), base)
        self.assertEqual(tone_rgb(base, 0.0), (128, 128, 128))
        self.assertEqual(tone_rgb(base, 1.0), base)

    def test_tailwind_has_ten_pale_to_dark(self) -> None:
        from wallpaper_recolor.color.color_math import rgb_to_hsl
        from wallpaper_recolor.ui.color_wheel import TAILWIND_STOPS, tailwind_palette

        palette = tailwind_palette(0.0, 1.0)
        self.assertEqual(len(TAILWIND_STOPS), 10)
        self.assertEqual(len(palette), 10)
        lights = [rgb_to_hsl(c)[2] for c in palette]
        self.assertGreater(lights[0], lights[-1])
        self.assertEqual(sorted(lights, reverse=True), lights)

    def test_commit_shade_updates_wheel_rgb(self) -> None:
        import tkinter as tk

        from wallpaper_recolor.ui.color_wheel import ColorWheel, rgb_to_hex

        live: list[tuple[int, int, int]] = []
        committed: list[tuple[int, int, int]] = []
        root = tk.Tk()
        root.withdraw()
        try:
            wheel = ColorWheel(root, on_color=live.append, on_color_commit=committed.append)
            wheel.set_rgb((255, 0, 0))
            live.clear()
            committed.clear()
            wheel.apply_mix("shades", 1.0, commit=True)
            self.assertEqual(wheel.current_rgb(), (0, 0, 0))
            self.assertEqual(committed[-1], (0, 0, 0))
            self.assertEqual(live[-1], (0, 0, 0))
            self.assertEqual(wheel.hex_var.get().upper(), "#000000")
            self.assertIn("RGB 0, 0, 0", wheel.hsl_label.cget("text"))
            self.assertEqual(rgb_to_hex(wheel.current_rgb()), "#000000")
        finally:
            root.destroy()

    def test_commit_tint_and_tone_and_tailwind(self) -> None:
        import tkinter as tk

        from wallpaper_recolor.ui.color_wheel import ColorWheel, tailwind_palette

        committed: list[tuple[int, int, int]] = []
        root = tk.Tk()
        root.withdraw()
        try:
            wheel = ColorWheel(root, on_color_commit=committed.append)
            wheel.set_rgb((255, 0, 0))
            committed.clear()
            wheel.apply_mix("tints", 0.0, commit=True)
            self.assertEqual(wheel.current_rgb(), (255, 255, 255))
            wheel.set_rgb((255, 0, 0))
            wheel.apply_mix("tones", 0.0, commit=True)
            self.assertEqual(wheel.current_rgb(), (128, 128, 128))
            wheel.set_rgb((255, 0, 0))
            expected = tailwind_palette(0.0, 1.0)[0]
            wheel.apply_tailwind_index(0, commit=True)
            self.assertEqual(wheel.current_rgb(), expected)
            self.assertEqual(len(wheel._mix_bars), 4)
        finally:
            root.destroy()


class TestLabels(unittest.TestCase):
    def test_identity_inpaint_on_empty_mask(self) -> None:
        from wallpaper_recolor.transform.inpaint import inpaint_array, inpaint_image

        arr = np.zeros((16, 20, 3), dtype=np.uint8)
        arr[:, :] = (40, 120, 80)
        arr[4:8, 4:8] = (200, 10, 10)
        hole = np.zeros((16, 20), dtype=bool)
        out = inpaint_array(arr, hole)
        np.testing.assert_array_equal(out, arr)
        im = Image.fromarray(arr, mode="RGB")
        copied = inpaint_image(im, [])
        np.testing.assert_array_equal(np.asarray(copied), arr)

    def test_inpaint_fills_a_hole(self) -> None:
        from wallpaper_recolor.transform.inpaint import inpaint_array, inpaint_image

        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        arr[:, :] = (20, 180, 40)
        arr[10:22, 10:22] = (220, 20, 20)
        hole = np.zeros((32, 32), dtype=bool)
        hole[10:22, 10:22] = True
        out = inpaint_array(arr, hole)
        self.assertFalse(np.array_equal(out, arr))
        patch = out[12:20, 12:20]
        self.assertGreater(float(np.mean(patch[..., 1])), 80.0)
        self.assertLess(float(np.mean(patch[..., 0])), 120.0)
        im = Image.fromarray(arr, mode="RGB")
        filled = inpaint_image(im, [(10, 10, 22, 22)])
        filled_arr = np.asarray(filled)
        self.assertGreater(float(np.mean(filled_arr[12:20, 12:20, 1])), 80.0)

    def test_drag_rect_stored_in_source_coords(self) -> None:
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod
        from wallpaper_recolor.labels.boxes import display_box_to_source

        mapped = display_box_to_source(
            (10, 10, 30, 20), (100, 50), (400, 200), 0.0, 0.0, 1.0
        )
        self.assertEqual(mapped, (40, 40, 120, 80))
        # Zoom 2 about the frame center: display (200×100) is the 400×200
        # frame; (10,10)→(110,60) in source (center half), not a top-left crop.
        cropped = display_box_to_source(
            (10, 10, 30, 20), (200, 100), (400, 200), 0.0, 0.0, 2.0
        )
        self.assertIsNotNone(cropped)
        self.assertEqual(cropped, (110, 60, 130, 70))

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            src = Image.fromarray(np.zeros((200, 400, 3), dtype=np.uint8), mode="RGB")
            work = Image.fromarray(np.zeros((50, 100, 3), dtype=np.uint8), mode="RGB")
            app.source_image = src
            app.work_image = work
            app._orig_pil = work.copy()
            box = app._commit_drag_rect(10, 10, 30, 20, display_size=(100, 50))
            self.assertEqual(box, (40, 40, 120, 80))
            self.assertEqual(app._detect_roi, (40, 40, 120, 80))
        finally:
            root.destroy()

    def test_layers_zip_includes_label_files_when_set(self) -> None:
        import zipfile

        from wallpaper_recolor.io.export_layers_zip import export_layers_zip
        from wallpaper_recolor.labels.layer import LabelSpec

        arr = np.zeros((16, 32, 3), dtype=np.uint8)
        arr[:, :16] = (220, 20, 20)
        arr[:, 16:] = (20, 180, 40)
        im = Image.fromarray(arr, mode="RGB")
        range_map = build_range_map(im, 2, SPLIT_COLOR_CLOSENESS)
        range_map.set_replacement(0, (10, 20, 200))
        range_map.set_replacement(1, (200, 180, 20))
        snap = snapshot_assignment(range_map)
        spec = LabelSpec(text="Room 12", size=24, color=(12, 24, 48), x=4, y=6)
        with tempfile.TemporaryDirectory() as tmp:
            zpath = Path(tmp) / "layers.zip"
            export_layers_zip(zpath, im, snap, label=spec)
            with zipfile.ZipFile(zpath) as zf:
                names = set(zf.namelist())
                self.assertIn("label.tif", names)
                self.assertIn("label.png", names)
                self.assertIn("label.svg", names)
                svg = zf.read("label.svg").decode("utf-8")
                self.assertIn("<text", svg)
                self.assertIn("Room 12", svg)
                readme = zf.read("README.txt").decode("utf-8")
                self.assertIn("label.tif", readme)
                palette = json.loads(zf.read("palette.json"))
                self.assertIsNotNone(palette.get("label"))
                self.assertEqual(palette["label"]["text"], "Room 12")

    def test_ui_labels_panel_and_ocr_status(self) -> None:
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod
        from wallpaper_recolor.labels.detect import tesseract_status_text
        from wallpaper_recolor.ui import run

        self.assertTrue(callable(run))
        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            self.assertEqual(str(app.labels_panel.panel_title), "Labels")
            self.assertTrue(_widget_under(app.labels_detect_btn, app.labels_panel))
            self.assertTrue(_widget_under(app.labels_remove_btn, app.labels_panel))
            self.assertTrue(_widget_under(app.labels_clear_btn, app.labels_panel))
            self.assertTrue(_widget_under(app.label_text_entry, app.labels_panel))
            self.assertEqual(app.label_ocr_status.get(), tesseract_status_text())
            self.assertTrue(callable(app._on_label_detect))
            self.assertTrue(callable(app._on_label_remove))
            self.assertEqual(str(app.labels_mark_btn.cget("text")), "Select area")
            self.assertTrue(hasattr(app, "label_font_combo"))
            self.assertTrue(hasattr(app, "layers_panel"))
            self.assertEqual(str(app.layers_panel.panel_title), "Layers")
        finally:
            root.destroy()

    def test_detect_finds_v6n_on_dark_and_roi(self) -> None:
        """Connected-component fallback finds small white V6-N; ROI is source-space."""
        from PIL import ImageDraw

        from wallpaper_recolor.labels.detect import detect_text_boxes

        im = Image.new("RGB", (240, 160), (18, 42, 28))
        draw = ImageDraw.Draw(im)
        draw.text((2, 1), "V6-N", fill=(250, 250, 248))
        found = detect_text_boxes(im)
        self.assertTrue(found, "full-frame detect must find V6-N without Tesseract")
        hit = False
        for x0, y0, x1, y1 in found:
            if x0 < 40 and y0 < 30 and x1 > 8 and y1 > 6:
                hit = True
                break
        self.assertTrue(hit, found)

        roi = (0, 0, 80, 40)
        roi_found = detect_text_boxes(im, roi=roi)
        self.assertTrue(roi_found, "ROI detect must find V6-N")
        for x0, y0, x1, y1 in roi_found:
            self.assertLess(x0, 80)
            self.assertLess(y0, 40)
            self.assertGreater(x1, 0)
            self.assertGreater(y1, 0)

    def test_label_remove_inpaints_blob_and_edit_menu(self) -> None:
        """Remove inpaints the image (not just overlay); Edit → Text → Remove is wired."""
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod

        arr = np.zeros((32, 48, 3), dtype=np.uint8)
        arr[:, :] = (20, 160, 50)
        arr[8:20, 10:30] = (240, 240, 235)
        im = Image.fromarray(arr, mode="RGB")
        hole = (10, 8, 30, 20)
        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            app.source_image = im
            app.work_image = im
            app.rebuild_ranges()
            app._clear_history()
            app._refresh_now()
            live_before = np.asarray(app._work_live)
            blob_before = live_before[12:16, 16:24]
            self.assertGreater(float(np.mean(blob_before[..., 0])), 180.0)

            src = inspect.getsource(ui_mod.WallpaperRecolorApp._on_label_remove)
            self.assertIn("_run_background", src)
            self.assertIn("Removing…", src)
            detect_src = inspect.getsource(ui_mod.WallpaperRecolorApp._on_label_detect)
            self.assertIn("_run_background", detect_src)
            self.assertIn("Detecting…", detect_src)
            self.assertIn("work_image", detect_src)
            self.assertIn("detect_text_regions", detect_src)
            menu_src = inspect.getsource(ui_mod.WallpaperRecolorApp._rebuild_edit_menu)
            self.assertIn("_on_label_remove", menu_src)
            self.assertIn("_on_label_detect", menu_src)
            self.assertIn("_on_label_place_toggle", menu_src)
            panel_src = inspect.getsource(ui_mod.WallpaperRecolorApp._build_labels_panel)
            self.assertIn("_on_label_remove", panel_src)

            app._on_label_remove()
            self.assertIn("Detect", app.status.get())
            self.assertFalse(app._inpaint_boxes)

            app._detect_boxes = [hole]
            app._selected_detect = set()
            app._rebuild_edit_menu()
            busy_calls: list[tuple[bool, str | None]] = []
            orig_busy = app._set_busy

            def _busy(busy: bool, status: str | None = None) -> None:
                busy_calls.append((busy, status))
                orig_busy(busy, status)

            app._set_busy = _busy  # type: ignore[method-assign]
            app.text_menu.invoke(app.text_menu.index("Remove"))
            _drain_busy(app, root)
            self.assertTrue(any(on and msg == "Removing…" for on, msg in busy_calls))
            self.assertEqual(app._inpaint_boxes, [hole])
            self.assertFalse(app._detect_boxes)
            live_after = np.asarray(app._work_live)
            blob_after = live_after[12:16, 16:24]
            self.assertLess(float(np.mean(blob_after[..., 0])), 120.0)
            self.assertGreater(float(np.mean(blob_after[..., 1])), 80.0)
            self.assertFalse(np.array_equal(live_after, live_before))
            self.assertIn("pattern", app.status.get().lower())

            app.undo_edit()
            self.assertFalse(app._inpaint_boxes)
            blob_undo = np.asarray(app._work_live)[12:16, 16:24]
            self.assertGreater(float(np.mean(blob_undo[..., 0])), 180.0)
        finally:
            root.destroy()

    def test_detect_remove_place_printed_text_on_pattern(self) -> None:
        """Detect raster glyphs → Hilbert fill toward the motif → Place a new label."""
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod
        from wallpaper_recolor.labels.detect import detect_text_boxes

        arr = np.zeros((48, 64, 3), dtype=np.uint8)
        arr[:, 0::2] = (22, 128, 48)
        arr[:, 1::2] = (36, 152, 60)
        arr[14:30, 18:42] = (248, 248, 242)
        im = Image.fromarray(arr, mode="RGB")
        letter = (18, 14, 42, 30)
        found = detect_text_boxes(im)
        self.assertTrue(found, "contrast detect must find the letter blob on the raster")
        hit = False
        for x0, y0, x1, y1 in found:
            if x0 < letter[2] and x1 > letter[0] and y0 < letter[3] and y1 > letter[1]:
                hit = True
                break
        self.assertTrue(hit, found)

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            app.source_image = im
            app.work_image = im
            app.rebuild_ranges()
            app._clear_history()
            kept = app.layer_stack.add_label(name="Label", text="Room", x=2, y=40, select=False)
            self.assertTrue(kept.visible)
            app._refresh_now()
            live_before = np.asarray(app._work_live)
            self.assertGreater(float(np.mean(live_before[18:26, 24:36, 0])), 180.0)

            app._rebuild_edit_menu()
            app.text_menu.invoke(app.text_menu.index("Detect"))
            _drain_busy(app, root)
            self.assertTrue(app._detect_boxes)
            self.assertIn("Found", app.status.get())
            self.assertTrue(kept.visible)

            app.text_menu.invoke(app.text_menu.index("Remove"))
            _drain_busy(app, root)
            live_after = np.asarray(app._work_live)
            patch = live_after[18:26, 24:36]
            self.assertLess(float(np.mean(patch[..., 0])), 120.0)
            self.assertGreater(float(np.mean(patch[..., 1])), 70.0)
            self.assertFalse(np.array_equal(live_after[18:26, 24:36], live_before[18:26, 24:36]))
            self.assertTrue(kept.visible)
            self.assertTrue(
                any(ly.is_label() and ly.text == "Room" and ly.visible for ly in app.layer_stack.layers)
            )

            base = app.layer_stack.base_layer()
            if base is not None:
                app.layer_stack.select(base.id)
            app.label_text.set("Studio")
            app.text_menu.invoke(app.text_menu.index("Place"))
            self.assertTrue(app._label_place_mode)
            app._place_label_at_display(22, 16)
            self.assertTrue(
                any(ly.is_label() and ly.text == "Studio" for ly in app.layer_stack.layers)
            )
            self.assertTrue(app._inpaint_boxes)
        finally:
            root.destroy()

    def test_detect_cancel_raises_before_work(self) -> None:
        import threading

        from wallpaper_recolor.labels.detect import detect_text_boxes

        im = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8), mode="RGB")
        ev = threading.Event()
        ev.set()
        with self.assertRaises(InterruptedError) as ctx:
            detect_text_boxes(im, cancel=ev)
        self.assertIn("cancel", str(ctx.exception).lower())
        from wallpaper_recolor.labels import detect as detect_mod

        self.assertIn("_raise_if_cancelled", inspect.getsource(detect_mod.detect_text_regions))

    def test_wallpaper_mask_geometric_wider_than_floral(self) -> None:
        from wallpaper_recolor.transform.inpaint import (
            STYLE_FLORAL_SPEC,
            STYLE_GEOMETRIC_SPEC,
            text_mask_from_quads,
        )
        from wallpaper_recolor.transform.lama_onnx import prepare_lama_canvas, pad_to_multiple

        h, w = 40, 80
        quad = ((20, 16), (36, 16), (36, 22), (20, 22))
        geo = text_mask_from_quads(h, w, [quad], style=STYLE_GEOMETRIC_SPEC)
        floral = text_mask_from_quads(h, w, [quad], style=STYLE_FLORAL_SPEC)
        self.assertGreater(int(geo.sum()), int(floral.sum()))
        self.assertEqual(STYLE_GEOMETRIC_SPEC.close_kernel, (15, 3))
        self.assertEqual(STYLE_GEOMETRIC_SPEC.dilate_kernel, (5, 5))
        self.assertEqual(STYLE_GEOMETRIC_SPEC.pad_y_down, 6)
        self.assertEqual(STYLE_FLORAL_SPEC.close_kernel, (7, 3))
        self.assertEqual(STYLE_FLORAL_SPEC.dilate_iterations, 1)
        rgb = np.zeros((11, 13, 3), dtype=np.uint8)
        rgb[:] = (10, 20, 30)
        mask = np.zeros((11, 13), dtype=bool)
        mask[3:6, 4:8] = True
        prepared = prepare_lama_canvas(rgb, mask)
        self.assertIsNotNone(prepared)
        canvas, mcanvas, oh, ow = prepared
        self.assertEqual((oh, ow), (11, 13))
        self.assertEqual(canvas.shape[0] % 8, 0)
        self.assertEqual(canvas.shape[1] % 8, 0)
        self.assertEqual(pad_to_multiple(11), 16)
        np.testing.assert_array_equal(canvas[:11, :13], rgb)
        inpaint_src = inspect.getsource(
            __import__("wallpaper_recolor.transform.inpaint", fromlist=["inpaint_image"]).inpaint_image
        )
        self.assertNotIn("download_lama", inpaint_src)
        lama_src = inspect.getsource(
            __import__("wallpaper_recolor.transform.lama_onnx", fromlist=["lama_inpaint_crop"]).lama_inpaint_crop
        )
        self.assertNotIn("resize", lama_src.lower())

    def test_easyocr_mock_returns_quad_not_fat_rect(self) -> None:
        from unittest.mock import patch

        from wallpaper_recolor.labels.detect import TextRegion, detect_text_regions
        from wallpaper_recolor.transform.inpaint import STYLE_GEOMETRIC_SPEC, text_mask_from_quads

        im = Image.fromarray(np.zeros((32, 64, 3), dtype=np.uint8), mode="RGB")
        quad = ((8, 8), (20, 8), (18, 18), (6, 18))
        region = TextRegion(box=(6, 8, 20, 18), quad=quad)

        def fake_easy(_image, *, cancel=None):
            return [region]

        with patch("wallpaper_recolor.labels.detect._try_easyocr_regions", fake_easy):
            found = detect_text_regions(im)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].quad, quad)
        mask = text_mask_from_quads(32, 64, [quad], style=STYLE_GEOMETRIC_SPEC)
        self.assertTrue(np.any(mask[10:16, 8:18]))
        self.assertFalse(np.any(mask[0:3, 50:64]))
        self.assertLess(int((mask > 0).sum()), 32 * 64 // 2)


class TestPantoneTable(unittest.TestCase):
    def test_json_key_maps_to_stored_hex(self) -> None:
        from wallpaper_recolor.color.pantone import _JSON_PATH, lookup_pantone_hex

        table = json.loads(_JSON_PATH.read_text(encoding="utf-8"))
        self.assertGreater(len(table), 100)
        code, hex_color = next(iter(table.items()))
        self.assertEqual(lookup_pantone_hex(code), hex_color)
        self.assertEqual(lookup_pantone_hex(f"Pantone {code}"), hex_color)
        self.assertEqual(lookup_pantone_hex("186 C"), table["186 c"])
        self.assertIsNone(lookup_pantone_hex("not-a-real-pantone-code"))
        self.assertIsNone(lookup_pantone_hex("18-1664 TPX"))


class TestColorInputModes(unittest.TestCase):
    def test_three_linked_fields_no_mode_combobox(self) -> None:
        import tkinter as tk

        from wallpaper_recolor.ui.color_wheel import ColorWheel

        root = tk.Tk()
        root.withdraw()
        try:
            wheel = ColorWheel(root)
            self.assertFalse(hasattr(wheel, "mode_combo"))
            self.assertFalse(hasattr(wheel, "input_mode"))
            self.assertEqual(str(wheel.hex_label.cget("text")), "Hex")
            self.assertEqual(str(wheel.pantone_label.cget("text")), "Pantone")
            self.assertEqual(str(wheel.rgbao_label.cget("text")), "RGBAO")
            self.assertIsInstance(wheel.hex_entry, tk.Widget)
            self.assertIsInstance(wheel.pantone_entry, tk.Widget)
            self.assertIsInstance(wheel.rgbao_entry, tk.Widget)
            self.assertTrue(wheel.hex_var.get().upper().startswith("#"))
            self.assertRegex(wheel.rgbao_var.get(), r"^\d+,\s*\d+,\s*\d+,\s*\d+$")
            self.assertTrue(wheel.pantone_var.get())
        finally:
            root.destroy()

    def test_hex_updates_rgbao_and_nearest_pantone(self) -> None:
        import tkinter as tk

        from wallpaper_recolor.color.pantone import lookup_pantone_hex, pantone_code_for_rgb
        from wallpaper_recolor.ui.color_wheel import (
            ColorWheel,
            rgbao_text_to_rgba,
            rgb_to_hex,
        )

        committed: list[tuple[int, int, int]] = []
        root = tk.Tk()
        root.withdraw()
        try:
            wheel = ColorWheel(root, on_color_commit=committed.append)
            wheel.set_rgb((0, 0, 255))
            committed.clear()
            wheel.hex_var.set("#FF0000")
            wheel._hex_committed()
            self.assertEqual(wheel.current_rgb(), (255, 0, 0))
            self.assertEqual(committed[-1], (255, 0, 0))
            self.assertEqual(rgb_to_hex(wheel.current_rgb()), "#FF0000")
            self.assertEqual(rgbao_text_to_rgba(wheel.rgbao_var.get()), (255, 0, 0, 255))
            nearest = pantone_code_for_rgb((255, 0, 0), closest=True)
            self.assertIsNotNone(nearest)
            self.assertEqual(wheel.pantone_var.get(), nearest)
            self.assertIsNotNone(lookup_pantone_hex(wheel.pantone_var.get()))
        finally:
            root.destroy()

    def test_rgbao_updates_hex_and_pantone(self) -> None:
        import tkinter as tk

        from wallpaper_recolor.color.pantone import pantone_code_for_rgb
        from wallpaper_recolor.ui.color_wheel import ColorWheel, rgb_to_hex

        root = tk.Tk()
        root.withdraw()
        try:
            wheel = ColorWheel(root)
            wheel.set_rgb((255, 0, 0))
            wheel.rgbao_var.set("0, 0, 255, 255")
            wheel._rgbao_committed()
            self.assertEqual(wheel.current_rgb(), (0, 0, 255))
            self.assertEqual(wheel.hex_var.get().upper(), "#0000FF")
            self.assertEqual(rgb_to_hex(wheel.current_rgb()), "#0000FF")
            self.assertEqual(
                wheel.pantone_var.get(),
                pantone_code_for_rgb((0, 0, 255), closest=True),
            )
        finally:
            root.destroy()

    def test_known_pantone_maps_to_json_hex(self) -> None:
        import tkinter as tk

        from wallpaper_recolor.color.pantone import lookup_pantone_hex
        from wallpaper_recolor.ui.color_wheel import ColorWheel, rgb_to_hex

        expected = lookup_pantone_hex("186 C")
        self.assertIsNotNone(expected)
        root = tk.Tk()
        root.withdraw()
        try:
            wheel = ColorWheel(root)
            wheel.pantone_var.set("Pantone 186 C")
            wheel._pantone_committed()
            self.assertEqual(rgb_to_hex(wheel.current_rgb()), expected)
            self.assertEqual(wheel.hex_var.get().upper(), expected)
            self.assertEqual(wheel.input_status.get(), "")
        finally:
            root.destroy()

    def test_unknown_pantone_does_not_crash(self) -> None:
        import tkinter as tk

        from wallpaper_recolor.ui.color_wheel import UNKNOWN_PANTONE, ColorWheel, rgb_to_hex

        root = tk.Tk()
        root.withdraw()
        try:
            wheel = ColorWheel(root)
            wheel.set_rgb((10, 20, 30))
            before = wheel.current_rgb()
            before_hex = wheel.hex_var.get()
            wheel.pantone_var.set("not-a-real-pantone-code")
            wheel._pantone_committed()
            self.assertEqual(wheel.input_status.get(), UNKNOWN_PANTONE)
            self.assertEqual(wheel.current_rgb(), before)
            self.assertEqual(rgb_to_hex(wheel.current_rgb()), rgb_to_hex(before))
            self.assertEqual(wheel.hex_var.get(), before_hex)
            wheel.pantone_var.set("18-1664 TPX")
            wheel._pantone_committed()
            self.assertEqual(wheel.input_status.get(), UNKNOWN_PANTONE)
            self.assertEqual(wheel.current_rgb(), before)
        finally:
            root.destroy()

    def test_autocomplete_filter_prefix_and_substring(self) -> None:
        from wallpaper_recolor.color.pantone import filter_pantone_codes, list_pantone_codes

        codes = list_pantone_codes()
        self.assertGreater(len(codes), 2000)
        hits = filter_pantone_codes("19")
        self.assertTrue(hits)
        self.assertLessEqual(len(hits), 24)
        for shown in hits:
            self.assertIn("19", shown.casefold())
        hyphen = filter_pantone_codes("19-")
        for shown in hyphen:
            self.assertIn("19-", shown.casefold())
        black = filter_pantone_codes("Black")
        self.assertTrue(black)
        self.assertTrue(any("black" in shown.casefold() for shown in black))
        self.assertEqual(filter_pantone_codes(""), [])
        self.assertEqual(filter_pantone_codes("   "), [])

    def test_partial_hex_is_not_reverted_until_commit(self) -> None:
        import tkinter as tk

        from wallpaper_recolor.ui.color_wheel import ColorWheel

        root = tk.Tk()
        root.withdraw()
        try:
            wheel = ColorWheel(root)
            wheel.set_rgb((255, 0, 0))
            wheel.hex_var.set("#00")
            wheel._hex_keyrelease()
            self.assertEqual(wheel.current_rgb(), (255, 0, 0))
            self.assertEqual(wheel.hex_var.get(), "#00")
            wheel.hex_var.set("#0000FF")
            wheel._hex_keyrelease()
            self.assertEqual(wheel.current_rgb(), (0, 0, 255))
            self.assertEqual(wheel.hex_var.get().upper(), "#0000FF")
            self.assertEqual(wheel.rgbao_var.get(), "0, 0, 255, 255")
        finally:
            root.destroy()

    def test_wheel_and_history_refresh_all_three_fields(self) -> None:
        import tkinter as tk

        from wallpaper_recolor.color.pantone import pantone_code_for_rgb
        from wallpaper_recolor.ui.color_wheel import ColorWheel, rgb_to_hex

        root = tk.Tk()
        root.withdraw()
        try:
            wheel = ColorWheel(root)
            wheel.set_rgb((0, 255, 0))
            self.assertEqual(wheel.hex_var.get().upper(), "#00FF00")
            self.assertEqual(wheel.rgbao_var.get(), "0, 255, 0, 255")
            self.assertEqual(
                wheel.pantone_var.get(),
                pantone_code_for_rgb((0, 255, 0), closest=True),
            )
            wheel._commit()
            wheel.set_rgb((255, 0, 0))
            wheel._commit()
            self.assertEqual(wheel.history_colors()[0], (255, 0, 0))
            self.assertEqual(wheel.history_colors()[1], (0, 255, 0))
            wheel.pick_history(1)
            self.assertEqual(wheel.current_rgb(), (0, 255, 0))
            self.assertEqual(wheel.hex_var.get().upper(), rgb_to_hex((0, 255, 0)))
            self.assertEqual(wheel.rgbao_var.get(), "0, 255, 0, 255")
            self.assertEqual(
                wheel.pantone_var.get(),
                pantone_code_for_rgb((0, 255, 0), closest=True),
            )
            wheel.apply_mix("shades", 1.0, commit=True)
            self.assertEqual(wheel.current_rgb(), (0, 0, 0))
            self.assertEqual(wheel.hex_var.get().upper(), "#000000")
            self.assertEqual(wheel.rgbao_var.get(), "0, 0, 0, 255")
            self.assertTrue(wheel.pantone_var.get())
        finally:
            root.destroy()


class TestColorHistory(unittest.TestCase):
    """Recent-color strip under the wheel: commits, cap, click, .wpedit round-trip."""

    def test_push_color_history_dedupes_and_caps(self) -> None:
        from wallpaper_recolor.ui.color_wheel import (
            COLOR_HISTORY_MAX,
            parse_color_history,
            push_color_history,
        )

        self.assertEqual(COLOR_HISTORY_MAX, 20)
        self.assertEqual(push_color_history([], (10, 20, 30)), [(10, 20, 30)])
        self.assertEqual(push_color_history([(10, 20, 30)], (10, 20, 30)), [(10, 20, 30)])
        moved = push_color_history([(1, 0, 0), (2, 0, 0), (3, 0, 0)], (3, 0, 0))
        self.assertEqual(moved, [(3, 0, 0), (1, 0, 0), (2, 0, 0)])
        filled = [(i, 0, 0) for i in range(20)]
        capped = push_color_history(filled, (99, 1, 2))
        self.assertEqual(len(capped), 20)
        self.assertEqual(capped[0], (99, 1, 2))
        self.assertEqual(capped[-1], (18, 0, 0))
        self.assertNotIn((19, 0, 0), capped)
        self.assertEqual(parse_color_history([[1, 2, 3], "nope", [4, 5]]), [(1, 2, 3)])
        self.assertEqual(parse_color_history(None), [])

    def test_history_starts_empty_commits_fill_and_21st_drops_oldest(self) -> None:
        import tkinter as tk

        from wallpaper_recolor.ui.color_wheel import COLOR_HISTORY_MAX, ColorWheel

        root = tk.Tk()
        root.withdraw()
        try:
            wheel = ColorWheel(root)
            root.update_idletasks()
            slaves = list(wheel.pack_slaves())
            self.assertIs(slaves[0], wheel.canvas)
            self.assertIs(slaves[1], wheel.history_canvas.master)
            mix_host = wheel._mix_bars["tailwind"].master
            self.assertGreater(slaves.index(mix_host), slaves.index(wheel.history_canvas.master))
            self.assertEqual(wheel.history_colors(), [])
            self.assertEqual(len(wheel.history_canvas.find_withtag("swatch")), 0)
            self.assertEqual(len(wheel.history_canvas.find_withtag("empty")), COLOR_HISTORY_MAX)

            wheel.set_rgb((10, 20, 30), notify=True)
            self.assertEqual(wheel.history_colors(), [])

            committed: list[tuple[int, int, int]] = []
            for rgb in ((255, 0, 0), (0, 255, 0), (0, 0, 255)):
                wheel.set_rgb(rgb)
                wheel._commit()
                committed.append(wheel.current_rgb())
            self.assertEqual(len(wheel.history_colors()), 3)
            self.assertEqual(wheel.history_colors(), list(reversed(committed)))
            self.assertEqual(len(wheel.history_canvas.find_withtag("swatch")), 3)
            self.assertEqual(len(wheel.history_canvas.find_withtag("empty")), COLOR_HISTORY_MAX - 3)

            for i in range(21):
                wheel.record_color((i, 1, 2))
            hist = wheel.history_colors()
            self.assertEqual(len(hist), 20)
            self.assertEqual(hist[0], (20, 1, 2))
            self.assertEqual(hist[-1], (1, 1, 2))
            self.assertNotIn((0, 1, 2), hist)
            self.assertEqual(len(wheel.history_canvas.find_withtag("swatch")), 20)
            self.assertEqual(len(wheel.history_canvas.find_withtag("empty")), 0)
        finally:
            root.destroy()

    def test_history_click_sets_selected_range_and_edit_state_round_trip(self) -> None:
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod
        from wallpaper_recolor.ui.coverage_bar import HALF_MATCH, HALF_REPLACE

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            with tempfile.TemporaryDirectory() as tmp:
                img_path = Path(tmp) / "tiny.png"
                Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB").save(img_path)
                self.assertTrue(app._open_image_from_path(img_path, reset_edits=True))
                self.assertEqual(app.wheel.history_colors(), [])

                app.select_range(0, HALF_REPLACE)
                app.wheel.set_rgb((255, 0, 0), notify=True)
                app.wheel._commit()
                app.wheel.set_rgb((0, 255, 0), notify=True)
                app.wheel._commit()
                self.assertEqual(app.range_map.ranges[0].replacement_rgb, app.wheel.current_rgb())
                red = app.wheel.history_colors()[1]
                green = app.wheel.history_colors()[0]
                self.assertNotEqual(red, green)
                app.wheel.pick_history(1)
                self.assertEqual(app.range_map.ranges[0].replacement_rgb, red)
                self.assertEqual(app.wheel.history_colors()[0], red)
                app.undo_edit()
                self.assertEqual(app.range_map.ranges[0].replacement_rgb, green)

                app.select_range(0, HALF_MATCH)
                app.wheel.pick_history(0)
                self.assertEqual(app.range_map.ranges[0].match_rgb, red)

                app._apply_eyedrop_rgb((11, 22, 33))
                self.assertEqual(app.wheel.history_colors()[0], (11, 22, 33))

                state_path = Path(tmp) / "tiny_edit.wpedit"
                app._write_edit_state(state_path)
                payload = json.loads(state_path.read_text(encoding="utf-8"))
                self.assertEqual(payload["color_history"][0], [11, 22, 33])
                self.assertLessEqual(len(payload["color_history"]), 20)
                saved = [tuple(row) for row in payload["color_history"]]
                app.wheel.set_history([])
                self.assertEqual(app.wheel.history_colors(), [])
                app._read_edit_state(state_path)
                self.assertEqual(app.wheel.history_colors(), saved)
        finally:
            root.destroy()


class TestHoverTooltips(unittest.TestCase):
    """Section help lives on hover balloons, not always-visible Labels."""

    def test_helper_and_key_widget_tips(self) -> None:
        import tkinter as tk
        from types import SimpleNamespace

        import wallpaper_recolor.ui.app as ui_mod
        from wallpaper_recolor.ui.tooltip import DELAY_MS, HoverTip, bind_tooltip, tooltip_text

        self.assertEqual(DELAY_MS, 400)
        root = tk.Tk()
        root.withdraw()
        try:
            btn = tk.Label(root, text="x")
            tip = bind_tooltip(btn, "Hello", delay_ms=0)
            self.assertIsInstance(tip, HoverTip)
            self.assertEqual(tooltip_text(btn), "Hello")
            tip._show()
            self.assertIsNotNone(tip._win)
            tip.hide()
            self.assertIsNone(tip._win)

            app = ui_mod.WallpaperRecolorApp(root)
            self.assertEqual(app.cover_hint.get().strip(), "")
            build_tip = tooltip_text(app.tess_build_btn).lower()
            self.assertIn("position", build_tip)
            self.assertIn("zoom", build_tip)
            mode_tip = tooltip_text(app.tess_mode_combo).lower()
            self.assertIn("repeat", mode_tip)
            self.assertIn("hilbert", mode_tip)
            self.assertIn("warp", mode_tip)
            self.assertTrue(hasattr(app.coverage.bar, "_hover_tip"))
            app.coverage.update_idletasks()
            app.coverage.redraw()
            eye_tip = app.coverage._header_tip(SimpleNamespace(x=8, y=14)).lower()
            self.assertIn("hide", eye_tip)
            pct_tip = tooltip_text(app.coverage.bar).lower()
            self.assertIn("weight", pct_tip)
            tess_src = inspect.getsource(ui_mod.WallpaperRecolorApp._build_tessellate_panel)
            self.assertNotIn("wraplength=320", tess_src)
            crop_src = inspect.getsource(ui_mod.WallpaperRecolorApp._build_crop_panel)
            self.assertNotIn("wraplength=320", crop_src)
            layout_src = inspect.getsource(ui_mod.WallpaperRecolorApp._build_layout)
            self.assertNotIn("Top: match from image", layout_src)
            self.assertIn("(from floor)", layout_src)
            tools_tip = tooltip_text(app.tools_combo).lower()
            self.assertIn("view move", tools_tip)
            self.assertIn("grab move", tools_tip)
            add_tip = tooltip_text(app.layers_add_image_btn).lower()
            self.assertIn("overlay", add_tip)
            font_tip = tooltip_text(app.label_font_combo).lower()
            self.assertIn("font", font_tip)
        finally:
            root.destroy()


class TestLayerStack(unittest.TestCase):
    """Document layers: visibility, order, selection-scoped ops, fonts, .wpedit."""

    def test_default_one_image_layer_visible_selected(self) -> None:
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod
        from wallpaper_recolor.layers.stack import LAYER_IMAGE, ROLE_BASE

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            im = Image.fromarray(np.zeros((12, 16, 3), dtype=np.uint8), mode="RGB")
            app.source_image = im
            app.work_image = im
            app.rebuild_ranges()
            base = app.layer_stack.base_layer()
            self.assertIsNotNone(base)
            group = app.layer_stack.range_group_for(base.id)
            self.assertIsNotNone(group)
            self.assertEqual(group.name, "Color ranges")
            self.assertEqual(group.parent_id, base.id)
            n = len(app.range_map.ranges)
            kids = [ly for ly in app.layer_stack.layers if ly.is_range()]
            self.assertEqual(len(kids), n)
            self.assertTrue(all(kid.parent_id == group.id for kid in kids))
            self.assertEqual(base.kind, LAYER_IMAGE)
            self.assertEqual(base.role, ROLE_BASE)
            self.assertTrue(base.visible)
            self.assertIn(base.id, app.layer_stack.selected_ids)
            self.assertEqual(app.pointer_tool.get(), ui_mod.TOOL_VIEW_MOVE)
            self.assertEqual(app.pointer_tool_label.get(), "View Move")
            before_n = n
            app.range_count.set(before_n + 1)
            app._on_range_count()
            kids2 = [ly for ly in app.layer_stack.layers if ly.is_range()]
            self.assertEqual(len(kids2), before_n + 1)
            group2 = app.layer_stack.range_group_for(base.id)
            self.assertTrue(all(kid.parent_id == group2.id for kid in kids2))
            self.assertFalse(any(ly.is_range() and not ly.parent_id for ly in app.layer_stack.layers))
            self.assertTrue(group.expanded)
            self.assertTrue(app._layer_range_rows)
            first = app._layer_range_rows[0]
            self.assertEqual(first["spin"].winfo_class(), "TSpinbox")
            self.assertEqual(first["match"].winfo_class(), "Canvas")
            self.assertEqual(first["replace"].winfo_class(), "Canvas")
            walked = [ly for ly, _d in app.layer_stack.walk_visible_tree() if ly.is_range()]
            self.assertEqual(len(walked), len(app.range_map.ranges))
            app._on_layer_twisty(group.id, False)
            hidden = [ly for ly, _d in app.layer_stack.walk_visible_tree() if ly.is_range()]
            self.assertEqual(hidden, [])
            self.assertFalse(app.layer_stack.range_group_for(base.id).expanded)
            app._on_layer_twisty(group.id, True)
            self.assertTrue(any(ly.is_range() for ly, _d in app.layer_stack.walk_visible_tree()))
            tabs = [app.notebook.tab(tid, "text") for tid in app.notebook.tabs()]
            self.assertIn("Clusters", tabs)
            self.assertIn("Composite", tabs)
            app.apply_typed_percent(0, 25)
            self.assertAlmostEqual(app.range_map.weights()[0], 0.25, places=2)
            self.assertEqual(app._layer_range_rows[0]["pct"].get(), "25")
            app._on_layer_range_swatch(0, ui_mod.HALF_MATCH)
            self.assertEqual(app.selected_index, 0)
            self.assertEqual(app.selected_half, ui_mod.HALF_MATCH)
            from wallpaper_recolor.layers.stack import correction_target_ids

            self.assertIn(base.id, correction_target_ids(app.layer_stack))
        finally:
            root.destroy()

    def test_result_preview_shows_replacement_not_original(self) -> None:
        """Result must remapped pixels after a change-to edit, even if Range N is selected."""
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod
        from wallpaper_recolor.layers.stack import correction_target_ids

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            arr = np.full((20, 24, 3), 40, dtype=np.uint8)
            arr[:, 12:] = (200, 30, 30)
            work = Image.fromarray(arr, mode="RGB")
            app.work_image = work
            app.source_image = work
            app.rebuild_ranges()
            self.assertIsNotNone(app.range_map)
            app.range_map.texture_enabled = False
            app._on_layer_range_swatch(0, ui_mod.HALF_REPLACE)
            base = app.layer_stack.base_layer()
            self.assertIsNotNone(base)
            self.assertIn(base.id, correction_target_ids(app.layer_stack))
            app.range_map.set_replacement(0, (5, 220, 15))
            if len(app.range_map.ranges) > 1:
                app.range_map.set_replacement(1, (10, 20, 240))
            app._refresh_now()
            orig = np.asarray(app._orig_pil.convert("RGB"))
            result = np.asarray(app._tex_pil.convert("RGB"))
            live = np.asarray(app._work_live.convert("RGB"))
            self.assertEqual(orig.shape, result.shape)
            self.assertFalse(np.array_equal(orig, result))
            self.assertFalse(np.array_equal(orig, live))
            labels = np.asarray(app.range_map.labels)
            hits = np.flatnonzero(labels.reshape(-1) == 0)
            self.assertGreater(hits.size, 0)
            y0, x0 = divmod(int(hits[0]), labels.shape[1])
            if live.shape[:2] == labels.shape:
                self.assertFalse(np.array_equal(live[y0, x0], orig[y0, x0] if orig.shape[:2] == labels.shape else arr[y0, x0]))
        finally:
            root.destroy()

    def test_hide_eye_skips_composite(self) -> None:
        from wallpaper_recolor.layers.stack import LayerStack, composite_stack, flatten_rgb

        red = Image.new("RGB", (8, 8), (200, 10, 10))
        blue = Image.new("RGB", (8, 8), (10, 10, 200))
        stack = LayerStack()
        stack.add_image(name="Bottom", raster=red, role="base", select=False)
        top = stack.add_image(name="Top", raster=blue)
        out = flatten_rgb(composite_stack(stack, (8, 8), (8, 8), map_xy=False))
        self.assertEqual(out.getpixel((2, 2)), (10, 10, 200))
        stack.set_visible(top.id, False)
        hidden = flatten_rgb(composite_stack(stack, (8, 8), (8, 8), map_xy=False))
        self.assertEqual(hidden.getpixel((2, 2)), (200, 10, 10))

    def test_higher_stack_covers_lower_opaque(self) -> None:
        from wallpaper_recolor.layers.stack import LayerStack, composite_stack, flatten_rgb

        bottom = Image.new("RGB", (10, 10), (0, 255, 0))
        top = Image.new("RGB", (10, 10), (255, 0, 0))
        stack = LayerStack()
        stack.add_image(name="Green", raster=bottom, role="base", select=False)
        stack.add_image(name="Red", raster=top)
        self.assertEqual(stack.layers[0].name, "Red")
        out = flatten_rgb(composite_stack(stack, (10, 10), (10, 10), map_xy=False))
        self.assertEqual(out.getpixel((5, 5)), (255, 0, 0))
        stack.move_down(stack.layers[0].id)
        out2 = flatten_rgb(composite_stack(stack, (10, 10), (10, 10), map_xy=False))
        self.assertEqual(out2.getpixel((5, 5)), (0, 255, 0))

    def test_recolor_skips_unselected_image_layer(self) -> None:
        from wallpaper_recolor.layers.stack import (
            LayerStack,
            composite_stack,
            correction_target_ids,
            flatten_rgb,
        )

        a = Image.new("RGB", (6, 6), (30, 30, 30))
        b = Image.new("RGB", (6, 6), (80, 80, 80))
        stack = LayerStack()
        base = stack.add_image(name="A", raster=a, role="base", select=False)
        over = stack.add_image(name="B", raster=b)
        stack.select(over.id)
        targets = correction_target_ids(stack)
        self.assertEqual(targets, {over.id})
        painted = Image.new("RGB", (6, 6), (9, 9, 250))
        processed = {over.id: painted, base.id: a}
        out = flatten_rgb(composite_stack(stack, (6, 6), (6, 6), processed=processed, map_xy=False))
        self.assertEqual(out.getpixel((1, 1)), (9, 9, 250))
        stack.select(base.id)
        processed2 = {base.id: painted, over.id: b}
        out2 = flatten_rgb(
            composite_stack(stack, (6, 6), (6, 6), processed=processed2, map_xy=False)
        )
        self.assertEqual(out2.getpixel((1, 1)), (80, 80, 80))

    def test_correction_targets_resolve_range_row_to_parent_image(self) -> None:
        from wallpaper_recolor.layers.stack import LayerStack, ROLE_BASE, correction_target_ids

        stack = LayerStack()
        base = stack.add_image(name="Image", role=ROLE_BASE)
        group = stack.sync_range_children(base.id, 2)
        self.assertIsNotNone(group)
        kids = [ly for ly in stack.layers if ly.is_range()]
        self.assertTrue(kids)
        stack.select(kids[0].id)
        self.assertEqual(correction_target_ids(stack), {base.id})
        stack.select(group.id)
        self.assertEqual(correction_target_ids(stack), {base.id})
        lab = stack.add_label(name="L", text="Hi")
        stack.select(lab.id)
        self.assertEqual(correction_target_ids(stack), {base.id})

    def test_inpaint_target_prefers_selected_image(self) -> None:
        from wallpaper_recolor.layers.stack import LayerStack, ROLE_BASE, inpaint_target_layer

        stack = LayerStack()
        base = stack.add_image(name="A", role=ROLE_BASE, select=False)
        over = stack.add_image(name="B")
        stack.select(over.id)
        self.assertIs(inpaint_target_layer(stack), over)
        stack.select(base.id)
        self.assertIs(inpaint_target_layer(stack), base)
        lab = stack.add_label(name="L", text="Hi")
        stack.select(lab.id)
        self.assertIs(inpaint_target_layer(stack), base)

    def test_label_font_round_trip(self) -> None:
        from wallpaper_recolor.labels.layer import (
            LabelSpec,
            list_font_families,
            load_label_font,
            render_label_rgba,
        )

        families = list_font_families()
        self.assertTrue(families)
        spec_a = LabelSpec(text="Ab", size=24, color=(255, 0, 0), x=2, y=2, font="Arial")
        spec_b = LabelSpec(text="Ab", size=24, color=(255, 0, 0), x=2, y=2, font="Times New Roman")
        plate_a = render_label_rgba((80, 40), spec_a, (80, 40))
        plate_b = render_label_rgba((80, 40), spec_b, (80, 40))
        arr_a = np.asarray(plate_a)
        arr_b = np.asarray(plate_b)
        self.assertGreater(int(arr_a[..., 3].sum()), 0)
        self.assertGreater(int(arr_b[..., 3].sum()), 0)
        self.assertIsNotNone(load_label_font(24, "Arial"))
        self.assertIsNotNone(load_label_font(24, "Times New Roman"))

    def test_edit_state_includes_layers(self) -> None:
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod
        from wallpaper_recolor.labels.layer import LABEL_FONT_DEFAULT

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            im = Image.fromarray(np.zeros((10, 12, 3), dtype=np.uint8), mode="RGB")
            app.source_image = im
            app.work_image = im
            app.rebuild_ranges()
            app.label_text.set("Room")
            app.label_font.set("Georgia")
            app._write_label_fields_to_layer()
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "layers_edit.wpedit"
                app._write_edit_state(path)
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertIn("layers", payload)
                kinds = [row.get("type") for row in payload["layers"]]
                self.assertIn("image", kinds)
                self.assertIn("label", kinds)
                label_row = next(row for row in payload["layers"] if row.get("type") == "label")
                self.assertEqual(label_row.get("text"), "Room")
                self.assertEqual(label_row.get("font"), "Georgia")
                app.label_font.set(LABEL_FONT_DEFAULT)
                app._read_edit_state(path)
                self.assertEqual(app.label_font.get(), "Georgia")
                self.assertTrue(
                    any(ly.is_label() and ly.text == "Room" for ly in app.layer_stack.layers)
                )
        finally:
            root.destroy()

    def test_walk_visible_tree_hides_collapsed_range_children(self) -> None:
        from wallpaper_recolor.layers.stack import LayerStack, ROLE_BASE

        stack = LayerStack()
        base = stack.add_image(name="Image", role=ROLE_BASE)
        group = stack.sync_range_children(base.id, 3)
        self.assertIsNotNone(group)
        self.assertTrue(group.expanded)
        names = [ly.name for ly, _d in stack.walk_visible_tree()]
        self.assertIn("Color ranges", names)
        self.assertIn("Range 1", names)
        self.assertIn("Range 3", names)
        stack.set_expanded(group.id, False)
        names2 = [ly.name for ly, _d in stack.walk_visible_tree()]
        self.assertIn("Color ranges", names2)
        self.assertNotIn("Range 1", names2)
        rec = group.to_record()
        self.assertFalse(rec["expanded"])

    def test_cluster_subsample_caps_points_and_uses_lab(self) -> None:
        from wallpaper_recolor.ui.cluster_view import (
            cluster_scatter_data,
            matplotlib_available,
            subsample_labeled_pixels,
        )
        from wallpaper_recolor.color.color_ranges import build_range_map

        rgb = np.zeros((80, 80, 3), dtype=np.uint8)
        rgb[:, :40] = (200, 30, 30)
        rgb[:, 40:] = (30, 30, 200)
        labels = np.zeros((80, 80), dtype=np.int32)
        labels[:, 40:] = 1
        lab, sample_rgb, lbl, coords = subsample_labeled_pixels(rgb, labels, max_points=200)
        self.assertLessEqual(lab.shape[0], 200)
        self.assertEqual(lab.shape[1], 3)
        self.assertEqual(sample_rgb.shape[0], lab.shape[0])
        self.assertEqual(lbl.shape[0], lab.shape[0])
        self.assertEqual(coords.shape, (lab.shape[0], 2))
        self.assertTrue(np.all(coords[:, 0] >= 0) and np.all(coords[:, 1] >= 0))
        self.assertTrue(bool(matplotlib_available()) or not matplotlib_available())
        im = Image.fromarray(rgb, mode="RGB")
        range_map = build_range_map(im, 2)
        data = cluster_scatter_data(range_map, im, mode="source", max_points=150)
        self.assertIsNotNone(data)
        self.assertLessEqual(data["lab"].shape[0], 150)
        self.assertEqual(data["centers_lab"].shape[0], 2)
        self.assertEqual(len(data["match_rgb"]), 2)
        self.assertEqual(data["coords"].shape[0], data["lab"].shape[0])
        self.assertEqual(data["source_rgb"].shape[0], data["lab"].shape[0])
        y0, x0 = int(data["coords"][0, 0]), int(data["coords"][0, 1])
        self.assertEqual(tuple(int(c) for c in data["source_rgb"][0]), tuple(int(c) for c in rgb[y0, x0]))

    def test_cluster_zoom_does_not_change_wallpaper_zoom(self) -> None:
        """Clusters view-zoom is stored separately; Composite wallpaper size stays put."""
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            work = Image.new("RGB", (80, 40), (10, 20, 30))
            app.work_image = work
            app.source_image = work
            app.rebuild_ranges()
            app._fit_pane_size = lambda _host: (1, 1)  # type: ignore[method-assign]
            app._reset_preview_zoom()
            w100 = app._orig_photo.width()
            app.preview_zoom.set(200.0)
            app._on_preview_zoom_slider("200")
            self.assertAlmostEqual(app._composite_zoom_pct, 200.0)
            w200 = app._orig_photo.width()
            self.assertEqual(w200, w100 * 2)
            app.notebook.select(app.cluster_plot)
            app._on_tab_changed()
            app.preview_zoom.set(400.0)
            app._on_preview_zoom_slider("400")
            self.assertAlmostEqual(app._cluster_zoom_pct, 400.0)
            self.assertAlmostEqual(app._composite_zoom_pct, 200.0)
            self.assertEqual(app._orig_photo.width(), w200)
            self.assertAlmostEqual(app.cluster_plot.zoom_pct(), 400.0)
            for tid in app.notebook.tabs():
                if app.notebook.tab(tid, "text") == "Composite":
                    app.notebook.select(tid)
                    break
            app._on_tab_changed()
            self.assertAlmostEqual(app.preview_zoom.get(), 200.0)
            self.assertEqual(app._orig_photo.width(), w200)
        finally:
            root.destroy()

    def test_cluster_pick_sets_selected_half_from_source_pixel(self) -> None:
        """Pick pixel applies the exact sampled source RGB to match-from / change-to."""
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod
        from wallpaper_recolor.ui.cluster_view import source_pixel_at

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            arr = np.zeros((24, 24, 3), dtype=np.uint8)
            arr[:, :] = (40, 50, 60)
            arr[3, 5] = (11, 22, 33)
            work = Image.fromarray(arr, mode="RGB")
            app.work_image = work
            app.source_image = work
            app.rebuild_ranges()
            app.notebook.select(app.cluster_plot)
            app._on_tab_changed()
            app._flush_cluster_view()
            data = app.cluster_plot._cached_data
            self.assertIsNotNone(data)
            coords = data["coords"]
            hits = np.flatnonzero((coords[:, 0] == 3) & (coords[:, 1] == 5))
            if hits.size == 0:
                data["source_rgb"][0] = (11, 22, 33)
                data["coords"][0] = (3, 5)
                idx = 0
            else:
                idx = int(hits[0])
            hit = source_pixel_at(data, idx)
            self.assertIsNotNone(hit)
            rgb, yy, xx = hit
            app.selected_half = ui_mod.HALF_REPLACE
            app.cluster_plot.pick_index(idx)
            self.assertEqual(
                app.range_map.ranges[app.selected_index].replacement_rgb, rgb
            )
            app.selected_half = ui_mod.HALF_MATCH
            app.cluster_plot.pick_index(idx)
            self.assertEqual(app.range_map.ranges[app.selected_index].match_rgb, rgb)
            self.assertEqual((yy, xx), (int(data["coords"][idx, 0]), int(data["coords"][idx, 1])))
            plot = app.cluster_plot
            self.assertFalse(hasattr(plot, "pick_on"))
            self.assertFalse(hasattr(plot, "pick_btn"))
            plot._did_drag = False
            plot.nearest_index_at_root = lambda *_a, **_k: idx  # type: ignore[method-assign]

            class _Dbl:
                x_root = 0
                y_root = 0

            app.selected_half = ui_mod.HALF_REPLACE
            plot._on_double(_Dbl())
            self.assertEqual(
                app.range_map.ranges[app.selected_index].replacement_rgb, rgb
            )
        finally:
            root.destroy()

    def test_cluster_mmb_drag_moves_selected_rgb_not_orbit(self) -> None:
        """Middle-drag translates the selected swatch in Lab; left-drag does not."""
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod
        from wallpaper_recolor.ui.cluster_view import ClusterPlot, lab_delta_from_view_pixels

        root = tk.Tk()
        root.withdraw()
        try:
            d_l, d_a, d_b = lab_delta_from_view_pixels(40.0, 0.0, 20.0, -60.0, 100.0)
            self.assertTrue(abs(d_l) + abs(d_a) + abs(d_b) > 0.01)
            from wallpaper_recolor.ui.cluster_view import projected_lab_screen_xy

            lab0 = (50.0, 20.0, -10.0)
            x0, _y0 = projected_lab_screen_xy(lab0, 20.0, -60.0)
            x1, _y1 = projected_lab_screen_xy(
                (lab0[0] + d_l, lab0[1] + d_a, lab0[2] + d_b), 20.0, -60.0
            )
            self.assertGreater(x1, x0)

            plot = ClusterPlot(root)
            moved: list[tuple[int, int, int]] = []
            starts: list[int] = []
            ends: list[tuple[int, int, int]] = []
            plot.on_selected_rgb = lambda: (128, 64, 32)
            plot.on_move_start = lambda: starts.append(1)
            plot.on_move = lambda rgb: moved.append(rgb)
            plot.on_move_end = lambda rgb: ends.append(rgb)
            look_before = plot._look
            elev_before = plot._elev

            class _Ev:
                def __init__(self, x: int, y: int, num: int = 2) -> None:
                    self.x_root = x
                    self.y_root = y
                    self.num = num
                    self.state = 0

            plot._on_mmb_press(_Ev(10, 10))
            self.assertEqual(starts, [1])
            plot._on_mmb_drag(_Ev(50, 10))
            self.assertTrue(moved)
            self.assertNotEqual(moved[-1], (128, 64, 32))
            plot._on_mmb_release(_Ev(50, 10))
            self.assertEqual(ends[-1], moved[-1])
            self.assertEqual(plot._look, look_before)
            self.assertAlmostEqual(plot._elev, elev_before)

            count = len(moved)
            plot._on_press(_Ev(10, 10, num=1))
            plot._on_drag(_Ev(80, 40, num=1))
            plot._on_release(_Ev(80, 40, num=1))
            self.assertEqual(len(moved), count)
        finally:
            root.destroy()

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            arr = np.zeros((12, 12, 3), dtype=np.uint8)
            arr[:, :] = (40, 80, 120)
            work = Image.fromarray(arr, mode="RGB")
            app.work_image = work
            app.source_image = work
            app.rebuild_ranges()
            app.selected_half = ui_mod.HALF_REPLACE
            before = app.range_map.ranges[app.selected_index].replacement_rgb

            class _Ev2:
                x_root = 0
                y_root = 0
                num = 2
                state = 0

            class _Ev2b:
                x_root = 80
                y_root = -40
                num = 2
                state = 0

            app.cluster_plot._on_mmb_press(_Ev2())
            app.cluster_plot._on_mmb_drag(_Ev2b())
            after = app.range_map.ranges[app.selected_index].replacement_rgb
            self.assertNotEqual(after, before)
            app.cluster_plot._on_mmb_release(_Ev2b())
        finally:
            root.destroy()

    def test_deselect_ranges_wheel_does_not_write_map(self) -> None:
        """Escape / empty coverage: wheel and cluster pick keep range swatches unchanged."""
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod
        from wallpaper_recolor.ui.coverage_bar import HALF_REPLACE

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            arr = np.zeros((12, 12, 3), dtype=np.uint8)
            arr[:, :] = (40, 80, 120)
            work = Image.fromarray(arr, mode="RGB")
            app.work_image = work
            app.source_image = work
            app.rebuild_ranges()
            app.selected_half = HALF_REPLACE
            app.select_range(0, HALF_REPLACE)
            before = [tuple(b.replacement_rgb) for b in app.range_map.ranges]
            app.deselect_ranges()
            self.assertEqual(app.selected_index, -1)
            app.wheel.set_rgb((9, 8, 7), notify=True)
            app._on_wheel_color((9, 8, 7))
            after = [tuple(b.replacement_rgb) for b in app.range_map.ranges]
            self.assertEqual(after, before)
            self.assertEqual(app._scratch_rgb, (9, 8, 7))
            app.select_range(0, HALF_REPLACE)
            app._on_wheel_color((1, 2, 3))
            self.assertEqual(app.range_map.ranges[0].replacement_rgb, (1, 2, 3))
        finally:
            root.destroy()

    def test_cluster_center_uses_centroid_and_iso(self) -> None:
        import tkinter as tk

        from wallpaper_recolor.ui.cluster_view import (
            ClusterPlot,
            DEFAULT_AZIM,
            DEFAULT_ELEV,
            cluster_look_target,
        )

        root = tk.Tk()
        root.withdraw()
        try:
            plot = ClusterPlot(root)
            plot._cached_data = {
                "lab": np.array([[10.0, 20.0, 30.0], [30.0, 40.0, 50.0]], dtype=np.float64),
                "centers_lab": np.array([[99.0, 99.0, 99.0]], dtype=np.float64),
                "labels": np.array([0, 0], dtype=np.int32),
                "match_rgb": [(10, 20, 30)],
            }
            self.assertEqual(cluster_look_target(plot._cached_data), (30.0, 40.0, 20.0))
            plot._look = (12.0, -8.0, 70.0)
            plot._elev = 0.0
            plot._azim = 0.0
            plot.center_view()
            self.assertAlmostEqual(plot._look[0], 30.0)
            self.assertAlmostEqual(plot._look[1], 40.0)
            self.assertAlmostEqual(plot._look[2], 20.0)
            self.assertNotEqual(plot._look, (0.0, 0.0, 50.0))
            self.assertAlmostEqual(plot._elev, DEFAULT_ELEV)
            self.assertAlmostEqual(plot._azim, DEFAULT_AZIM)
        finally:
            root.destroy()

    def test_cluster_pan_translates_rotate_pivots_com(self) -> None:
        import tkinter as tk

        from wallpaper_recolor.ui.cluster_view import ClusterPlot, cluster_look_target

        root = tk.Tk()
        root.withdraw()
        try:
            plot = ClusterPlot(root)
            plot._cached_data = {
                "lab": np.array([[10.0, 20.0, 30.0], [30.0, 40.0, 50.0]], dtype=np.float64),
                "centers_lab": np.array([[99.0, 99.0, 99.0]], dtype=np.float64),
                "labels": np.array([0, 0], dtype=np.int32),
                "match_rgb": [(10, 20, 30)],
            }
            com = np.asarray(cluster_look_target(plot._cached_data), dtype=np.float64)
            plot._look = tuple(float(c) for c in com)
            plot._elev = 0.0
            plot._azim = 0.0
            plot.pan_by(40.0, -12.0)
            look_pan = np.asarray(plot._look, dtype=np.float64)
            self.assertGreater(float(np.linalg.norm(look_pan - com)), 0.2)
            self.assertAlmostEqual(plot._elev, 0.0)
            self.assertAlmostEqual(plot._azim, 0.0)
            dist = float(np.linalg.norm(look_pan - com))
            plot.orbit_by(50.0, 0.0)
            look_orb = np.asarray(plot._look, dtype=np.float64)
            self.assertGreater(abs(plot._azim), 1.0)
            self.assertAlmostEqual(float(np.linalg.norm(look_orb - com)), dist, places=5)
            self.assertGreater(float(np.linalg.norm(look_orb - look_pan)), 0.05)
        finally:
            root.destroy()

    def test_cluster_xyz_cycles_faces(self) -> None:
        import tkinter as tk

        from wallpaper_recolor.ui.cluster_view import ClusterPlot, FACE_VIEWS, XYZ_CYCLE

        root = tk.Tk()
        root.withdraw()
        try:
            plot = ClusterPlot(root)
            plot._cached_data = {
                "lab": np.array([[10.0, 20.0, 30.0], [30.0, 40.0, 50.0]], dtype=np.float64),
                "labels": np.array([0, 0], dtype=np.int32),
                "match_rgb": [(10, 20, 30)],
            }
            plot._look = (12.0, -8.0, 70.0)
            seen = [plot.cycle_xyz_view() for _ in range(len(XYZ_CYCLE))]
            self.assertEqual(tuple(seen), XYZ_CYCLE)
            self.assertEqual(plot.cycle_xyz_view(), "front")
            self.assertEqual(plot._look, (12.0, -8.0, 70.0))
            plot.look_face("top")
            elev, azim = FACE_VIEWS["top"]
            self.assertAlmostEqual(plot._elev, elev)
            self.assertAlmostEqual(plot._azim, azim)
            self.assertEqual(plot.cycle_xyz_view(), "bottom")
            self.assertEqual(plot._look, (12.0, -8.0, 70.0))
        finally:
            root.destroy()

    def test_cluster_face_click_sets_known_elev_azim(self) -> None:
        import tkinter as tk

        from wallpaper_recolor.ui.cluster_view import ClusterPlot, FACE_VIEWS, view_for_face

        root = tk.Tk()
        root.withdraw()
        try:
            plot = ClusterPlot(root)
            plot.look_face("front")
            self.assertAlmostEqual(plot._elev, 0.0)
            self.assertAlmostEqual(plot._azim, 0.0)
            plot.look_face("right")
            elev, azim = view_for_face("right")
            self.assertAlmostEqual(plot._elev, elev)
            self.assertAlmostEqual(plot._azim, azim)
            plot.look_face("top")
            self.assertAlmostEqual(plot._elev, FACE_VIEWS["top"][0])
            self.assertAlmostEqual(plot._azim, FACE_VIEWS["top"][1])
            plot._on_cube_view("face", "back")
            self.assertAlmostEqual(plot._elev, 0.0)
            self.assertAlmostEqual(abs(plot._azim), 180.0)
        finally:
            root.destroy()

    def test_cluster_view_cube_widgets_exist(self) -> None:
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod
        from wallpaper_recolor.ui.cluster_view import CLUSTER_HINT

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            plot = app.cluster_plot
            self.assertIsNotNone(plot.view_cube)
            self.assertIsNotNone(plot.center_btn)
            self.assertIsNotNone(plot.xyz_btn)
            self.assertIsNone(plot.home_btn)
            self.assertEqual(str(plot.center_btn.cget("text")), "Center")
            self.assertEqual(str(plot.xyz_btn.cget("text")), "XYZ")
            self.assertTrue(plot.view_cube.winfo_exists())
            self.assertIn("Center", CLUSTER_HINT)
            self.assertIn("XYZ", CLUSTER_HINT)
            self.assertIn("Cube", CLUSTER_HINT)
            self.assertIn("pan", CLUSTER_HINT)
            self.assertIn("Double-click", CLUSTER_HINT)
            self.assertNotIn("Pick pixel", CLUSTER_HINT)
            self.assertIn("Middle-drag", CLUSTER_HINT)
            self.assertNotIn("loupe", CLUSTER_HINT.lower())
            self.assertNotIn("Home", CLUSTER_HINT)
        finally:
            root.destroy()

    def test_cluster_range_extents_and_pick_highlight(self) -> None:
        import tkinter as tk

        from wallpaper_recolor.ui.cluster_view import (
            CLOUD_MIN_R,
            CLUSTER_ZOOM_PCT_MAX,
            ClusterPlot,
            cluster_range_extents,
        )

        lab = np.array(
            [
                [40.0, -10.0, 5.0],
                [42.0, -8.0, 6.0],
                [80.0, 40.0, -20.0],
            ],
            dtype=np.float64,
        )
        labels = np.array([0, 0, 1], dtype=np.int32)
        src = np.array([[11, 22, 33], [40, 50, 60], [200, 10, 10]], dtype=np.uint8)
        extents = cluster_range_extents(
            lab, labels, [(11, 22, 33), (200, 10, 10)], np.array([[41.0, -9.0, 5.5], [80.0, 40.0, -20.0]])
        )
        self.assertEqual(len(extents), 2)
        self.assertGreaterEqual(extents[0]["radii"][0], CLOUD_MIN_R)
        empty = cluster_range_extents(lab, np.array([0, 0, 0]), [(1, 2, 3), (4, 5, 6)])
        self.assertEqual([e["index"] for e in empty], [0])

        root = tk.Tk()
        root.withdraw()
        try:
            plot = ClusterPlot(root)
            plot._cached_data = {
                "lab": lab,
                "source_rgb": src,
                "point_rgb": src,
                "labels": labels,
                "match_rgb": [(11, 22, 33), (200, 10, 10)],
                "replace_rgb": [(0, 0, 0), (0, 0, 0)],
                "centers_lab": np.zeros((0, 3)),
            }
            self.assertFalse(hasattr(plot, "make_loupe"))
            self.assertFalse(hasattr(plot, "scatter_loupe_for_index"))
            plot.pick_index(1)
            self.assertEqual(plot._picked_index, 1)
            plot.pick_index(0)
            self.assertEqual(plot._picked_index, 0)
            plot._clear_pick_highlight()
            self.assertIsNone(plot._picked_index)
            plot.set_zoom_pct(2500.0)
            self.assertAlmostEqual(plot.zoom_pct(), 2500.0)
            plot.set_zoom_pct(9000.0)
            self.assertAlmostEqual(plot.zoom_pct(), CLUSTER_ZOOM_PCT_MAX)
        finally:
            root.destroy()

    def test_cluster_pick_has_no_wallpaper_loupe(self) -> None:
        import tkinter as tk

        import wallpaper_recolor.ui.app as ui_mod

        root = tk.Tk()
        root.withdraw()
        try:
            app = ui_mod.WallpaperRecolorApp(root)
            self.assertFalse(hasattr(app, "_cluster_loupe_image"))
            self.assertFalse(hasattr(app.cluster_plot, "make_loupe"))
            arr = np.zeros((16, 16, 3), dtype=np.uint8)
            arr[:, :] = (9, 9, 9)
            work = Image.fromarray(arr, mode="RGB")
            app.work_image = work
            app.source_image = work
            app.rebuild_ranges()
            app.notebook.select(app.cluster_plot)
            app._on_tab_changed()
            app._flush_cluster_view()
            plot = app.cluster_plot
            self.assertIsNotNone(plot._cached_data)
            self.assertTrue(plot._cached_data.get("extents"))
            plot.pick_index(0)
            self.assertEqual(plot._picked_index, 0)
            app._set_preview_zoom_pct(2000.0)
            self.assertAlmostEqual(app._cluster_zoom_pct, 2000.0)
            self.assertLessEqual(app._composite_zoom_pct, 800.0)
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
