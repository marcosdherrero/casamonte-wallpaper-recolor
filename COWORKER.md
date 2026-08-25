# Wallpaper Recolor — coworker copy (own Python)

This zip is **source + installer**, not a frozen exe. Python is downloaded into this folder on first install. No admin, no system-wide Python.

## Install once

1. Unzip the whole `WallpaperRecolor` folder (do not run files from inside the zip).
2. Double-click **Install.bat**. First run needs **network**. It puts CPython 3.12 in `runtime\` and pip-installs Pillow, numpy, and matplotlib.
3. After that you can run **offline**.

If Windows blocks the script, use “More info → Run anyway”, or right-click Install.bat → Run as the logged-in user (not Administrator).

## Run

Double-click **Run-LocalPython.bat** (or the Desktop / Start Menu “Wallpaper Recolor” shortcut Install creates).

1. File → Open a TIF / PNG / JPEG. Samples are in `examples\`.
2. Click **Fit** so the whole wallpaper sits in Composite.
3. File → Save / Save As for the Result image. Closing asks Yes/No/Cancel to save a `.wpedit`.

## Included vs not

Works: Composite, color ranges, Clusters (3D matplotlib), crop/zoom, Texture, export.

This is a **lean building build**: EasyOCR / LaMa / torch are **not** installed. Labels **Detect** / **Remove** may be unavailable. You can still draw label boxes by hand if the UI offers it.

Do not expect the huge `Wallpapers\` scan TIFs in this zip.

## Optional frozen exe

`scripts\build_exe.bat` still builds a PyInstaller onedir zip. Prefer **Install.bat** for coworkers.
