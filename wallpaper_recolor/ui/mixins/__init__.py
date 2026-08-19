# -*- coding: utf-8 -*-
"""
wallpaper_recolor.ui.mixins
------------------------------
WallpaperRecolorApp mixins — imported by ui.app, not a public API.

Class references (code + name only):
- CAP3321C Data Wrangling
- CAP4631C Machine Learning
- CAP4633C Machine Learning 2
"""

from wallpaper_recolor.ui.mixins.chrome import AppChromeMixin
from wallpaper_recolor.ui.mixins.layout import AppLayoutMixin
from wallpaper_recolor.ui.mixins.layers_labels import AppLayersLabelsMixin
from wallpaper_recolor.ui.mixins.preview import AppPreviewMixin
from wallpaper_recolor.ui.mixins.adjust import AppAdjustMixin
from wallpaper_recolor.ui.mixins.session import AppSessionMixin
from wallpaper_recolor.ui.mixins.ranges import AppRangesMixin

__all__ = (
    "AppChromeMixin",
    "AppLayoutMixin",
    "AppLayersLabelsMixin",
    "AppPreviewMixin",
    "AppAdjustMixin",
    "AppSessionMixin",
    "AppRangesMixin",
)
