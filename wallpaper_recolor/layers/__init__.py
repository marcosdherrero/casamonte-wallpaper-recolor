# -*- coding: utf-8 -*-
"""
wallpaper_recolor.layers
------------------------
Document layer stack (images + labels) for preview, Build, and export.

Hidden color ranges knock Result pixels to alpha 0; preview blits that
over an 8px checker. Save/export keeps the real alpha.

Class references (code + name only):
- CAP3321C Data Wrangling
- CAP4631C Machine Learning
"""

from wallpaper_recolor.layers.stack import (
    LAYER_GROUP,
    LAYER_IMAGE,
    LAYER_LABEL,
    LAYER_RANGE,
    ROLE_BASE,
    ROLE_RANGE_GROUP,
    LayerStack,
    StackLayer,
    clamp_layer_scale,
    composite_stack,
    composite_over_checker,
    correction_target_ids,
    default_font_family,
    flatten_rgb,
    flatten_rgb_or_keep_alpha,
    inpaint_target_layer,
    primary_is_label,
)

__all__ = (
    "LAYER_GROUP",
    "LAYER_IMAGE",
    "LAYER_LABEL",
    "LAYER_RANGE",
    "ROLE_BASE",
    "ROLE_RANGE_GROUP",
    "LayerStack",
    "StackLayer",
    "clamp_layer_scale",
    "composite_stack",
    "composite_over_checker",
    "correction_target_ids",
    "default_font_family",
    "flatten_rgb",
    "flatten_rgb_or_keep_alpha",
    "inpaint_target_layer",
    "primary_is_label",
)
