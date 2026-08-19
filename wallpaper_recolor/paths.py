# -*- coding: utf-8 -*-
"""
wallpaper_recolor.paths
-----------------------
Install vs PyInstaller roots. Source layout is unchanged; a frozen exe
reads package data from ``sys._MEIPASS`` and writes presets next to the .exe.

Class references (code + name only):
- CAP3321C Data Wrangling
"""

from __future__ import annotations

from pathlib import Path
import sys


def is_frozen() -> bool:
    """True when running from a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False))


def package_dir() -> Path:
    """``wallpaper_recolor/`` — icons, ``pantone_hex.json``."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", ".")).resolve() / "wallpaper_recolor"
    return Path(__file__).resolve().parent


def user_data_dir() -> Path:
    """Writable folder: repo root in source, folder containing the .exe when frozen."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent
