# -*- coding: utf-8 -*-
"""
run_app
-------
Launch Wallpaper Recolor (Casamonte analog-ink remapper).

Opens a Tk window maximized on the monitor under the cursor. Color closeness
is k-means in CIE Lab; Texture keeps weave luminosity; File → Save writes the
Result composite. Optional OCR / 3D Clusters extras are extra requirement files.

Class references (code + name only):
- CAP3321C Data Wrangling
- CAP4631C Machine Learning

Usage:
    python run_app.py
    pip install -r requirements.txt
"""

from __future__ import annotations

from wallpaper_recolor.ui import run  # run(): withdraw → build UI → zoomed on launch monitor

if __name__ == "__main__":
    # Blocks in Tk mainloop until File → Exit / window close (Yes/No/Cancel .wpedit)
    run()
