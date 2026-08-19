# -*- coding: utf-8 -*-
"""
wallpaper_recolor.labels
------------------------
Detect wallpaper text, inpaint it out, and export an editable label layer.

Class references (code + name only):
- CAP3321C Data Wrangling
- CAP4631C Machine Learning
- CAP4633C Machine Learning 2
"""

from wallpaper_recolor.labels.boxes import (
    Box,
    display_box_to_source,
    display_xy_to_source,
    source_box_to_display,
)
from wallpaper_recolor.labels.detect import (
    detect_text_boxes,
    detect_text_regions,
    easyocr_available,
    tesseract_available,
    tesseract_status_text,
)
from wallpaper_recolor.labels.layer import (
    LABEL_COLOR_DEFAULT,
    LABEL_FONT_DEFAULT,
    LABEL_SIZE_DEFAULT,
    LABEL_SIZE_MAX,
    LABEL_SIZE_MIN,
    LabelSpec,
    clamp_label_size,
    list_font_families,
    parse_label_color,
    render_label_rgba,
    write_label_files,
)
from wallpaper_recolor.labels.overlay import decorate_preview, draw_label_boxes

__all__ = (
    "Box",
    "LABEL_COLOR_DEFAULT",
    "LABEL_FONT_DEFAULT",
    "LABEL_SIZE_DEFAULT",
    "LABEL_SIZE_MAX",
    "LABEL_SIZE_MIN",
    "LabelSpec",
    "clamp_label_size",
    "decorate_preview",
    "detect_text_boxes",
    "detect_text_regions",
    "display_box_to_source",
    "display_xy_to_source",
    "draw_label_boxes",
    "easyocr_available",
    "parse_label_color",
    "list_font_families",
    "render_label_rgba",
    "source_box_to_display",
    "tesseract_available",
    "tesseract_status_text",
    "write_label_files",
)
