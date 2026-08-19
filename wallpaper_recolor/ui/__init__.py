# -*- coding: utf-8 -*-
"""
wallpaper_recolor.ui
--------------------
Tkinter app, color wheel, coverage bar, and dockable layout.

``from wallpaper_recolor.ui import run`` still launches the window.
Feature code lives in ``ui/mixins/``; widgets in ``dock``, ``preview_fit``,
``widgets``, ``coverage_bar``, ``color_wheel``, ``cluster_view``.

Class references (code + name only):
- CAP3321C Data Wrangling
- CAP4631C Machine Learning
- CAP4633C Machine Learning 2
"""

from wallpaper_recolor.ui.app import (
    ASSIGN_LABELS,
    HISTORY_LIMIT,
    LUMA_SPLIT_LABELS,
    RANGE_BY_LABELS,
    SPLIT_EQUAL_PIXELS_LABEL,
    DockablePanel,
    ScrollColumn,
    WallpaperRecolorApp,
    build_range_map,
    default_layout_profiles_path,
    filedialog,
    messagebox,
    run,
    simpledialog,
    threading,
    write_layers_zip,
)

__all__ = (
    "ASSIGN_LABELS",
    "HISTORY_LIMIT",
    "LUMA_SPLIT_LABELS",
    "RANGE_BY_LABELS",
    "SPLIT_EQUAL_PIXELS_LABEL",
    "DockablePanel",
    "ScrollColumn",
    "WallpaperRecolorApp",
    "build_range_map",
    "default_layout_profiles_path",
    "filedialog",
    "messagebox",
    "run",
    "simpledialog",
    "threading",
    "write_layers_zip",
)
