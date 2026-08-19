# -*- coding: utf-8 -*-
"""
wallpaper_recolor.color
-----------------------
Lab/HSL math, range maps, grain layers, tone, Pantone tables, and named presets.

Color closeness clusters in CIE Lab (k-means or snap-to-palette). Texture
composites keep original L* (weave) with replacement a*, b*. Print CMY
balance and analog-ink jobs live here so preview and save stay in lockstep.

Class references (code + name only):
- CAP3321C Data Wrangling
- CAP4631C Machine Learning
"""
