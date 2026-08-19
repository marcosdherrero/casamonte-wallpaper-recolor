# -*- coding: utf-8 -*-
"""
wallpaper_recolor.transform
---------------------------
Crop, output scale, lighting flatten, tessellate (tile from pattern,
Hilbert diffuse, mesh warp, detail mosaic). Inpaint is ``transform.inpaint``
(not imported here — that would cycle labels.boxes → this package → detect).

Class references (code + name only):
- CAP3321C Data Wrangling
- CAP4631C Machine Learning
- CAP4633C Machine Learning 2
"""

from wallpaper_recolor.transform.tessellate import (
    apply_crop_lighting_tessellate,
    apply_normalize_lighting,
    apply_tessellate,
    edges_already_match,
    estimate_axis_period,
    estimate_normalize_tone,
    hilbert_xy_to_d,
    image_already_periodic,
    is_identity_tessellate,
    plan_tessellate_crop,
    tess_mode_label,
)

__all__ = (
    "apply_crop_lighting_tessellate",
    "apply_normalize_lighting",
    "apply_tessellate",
    "edges_already_match",
    "estimate_axis_period",
    "estimate_normalize_tone",
    "hilbert_xy_to_d",
    "image_already_periodic",
    "is_identity_tessellate",
    "plan_tessellate_crop",
    "tess_mode_label",
)
