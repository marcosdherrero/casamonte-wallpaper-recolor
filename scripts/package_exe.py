# -*- coding: utf-8 -*-
"""Copy examples + README into dist/WallpaperRecolor and zip to release/."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "dist" / "WallpaperRecolor"
EXE = DIST_DIR / "WallpaperRecolor.exe"
RELEASE_DIR = ROOT / "release"
ZIP_PATH = RELEASE_DIR / "WallpaperRecolor-windows.zip"
EXAMPLES_SRC = ROOT / "docs" / "examples"
README_SRC = ROOT / "COWORKER.md"


def _copy_tree(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dest / item.name)


def main() -> int:
    if not EXE.is_file():
        print(f"Missing {EXE} — run PyInstaller first.")
        return 1

    if README_SRC.is_file():
        shutil.copy2(README_SRC, DIST_DIR / "README.txt")

    if EXAMPLES_SRC.is_dir():
        _copy_tree(EXAMPLES_SRC, DIST_DIR / "examples")

    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in DIST_DIR.rglob("*"):
            if path.is_file():
                zf.write(path, Path("WallpaperRecolor") / path.relative_to(DIST_DIR))

    unzipped = sum(p.stat().st_size for p in DIST_DIR.rglob("*") if p.is_file())
    print(f"Folder: {DIST_DIR}")
    print(f"Zip:    {ZIP_PATH}")
    print(f"Unzipped bytes: {unzipped}")
    print(f"Zip bytes:      {ZIP_PATH.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
