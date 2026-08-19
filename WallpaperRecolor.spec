# -*- mode: python ; coding: utf-8 -*-
"""
Lean Windows onedir build for a coworker with no Python install.

Includes Pillow, numpy, Tkinter, matplotlib (Clusters 3D).
Excludes EasyOCR, OpenCV, onnxruntime, torch, LaMa weights.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve()

datas = [
    (str(ROOT / "wallpaper_recolor" / "icons"), "wallpaper_recolor/icons"),
    (
        str(ROOT / "wallpaper_recolor" / "color" / "pantone_hex.json"),
        "wallpaper_recolor/color",
    ),
]
datas += collect_data_files("matplotlib", includes=["mpl-data/**"])

hiddenimports = [
    "PIL",
    "PIL.Image",
    "PIL.ImageChops",
    "PIL.ImageDraw",
    "PIL.ImageFilter",
    "PIL.ImageTk",
    "PIL.JpegImagePlugin",
    "PIL.PngImagePlugin",
    "PIL.TiffImagePlugin",
    "numpy",
    "tkinter",
    "tkinter.filedialog",
    "tkinter.messagebox",
    "tkinter.simpledialog",
    "tkinter.ttk",
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.figure",
    "matplotlib.backends.backend_agg",
    "matplotlib.backends.backend_tkagg",
]
hiddenimports += collect_submodules("wallpaper_recolor")

# Optional OCR / inpaint stack — huge, not for the building USB zip.
# pandas/scipy/lxml often ride in from this machine's site-packages; the app does not need them.
excludes = [
    "easyocr",
    "cv2",
    "onnxruntime",
    "onnxruntime.capi",
    "torch",
    "torchvision",
    "torchaudio",
    "tensorflow",
    "skimage",
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
    "pandas",
    "scipy",
    "lxml",
    "openpyxl",
    "yaml",
    "traitlets",
    "jinja2",
]

a = Analysis(
    [str(ROOT / "run_app.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="WallpaperRecolor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="WallpaperRecolor",
)
