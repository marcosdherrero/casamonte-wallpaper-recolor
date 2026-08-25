# -*- coding: utf-8 -*-
"""Stage source + installer into release/WallpaperRecolor-setup and zip it."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "release" / "WallpaperRecolor-setup"
ZIP_PATH = ROOT / "release" / "WallpaperRecolor-setup.zip"
EXAMPLES_SRC = ROOT / "docs" / "examples"

SKIP_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "runtime",
    ".cache",
    "build",
    "dist",
    "release",
    "Wallpapers",
    "tests",
    "docs",
}

SKIP_FILE_SUFFIXES = {".pyc", ".pyo", ".onnx", ".onnx.part", ".wpedit"}


def _copy_package(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        if any(part in SKIP_DIR_NAMES or part == ".git" for part in rel.parts):
            continue
        if path.is_dir():
            continue
        if path.suffix.lower() in SKIP_FILE_SUFFIXES:
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def main() -> int:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    for name in (
        "run_app.py",
        "requirements.txt",
        "requirements-plot.txt",
        "Install.bat",
        "Run-LocalPython.bat",
    ):
        shutil.copy2(ROOT / name, STAGE / name)

    readme = ROOT / "COWORKER.md"
    if readme.is_file():
        shutil.copy2(readme, STAGE / "README.txt")

    scripts = STAGE / "scripts"
    scripts.mkdir()
    for name in ("install_coworker.ps1", "install_coworker.bat"):
        shutil.copy2(ROOT / "scripts" / name, scripts / name)

    _copy_package(ROOT / "wallpaper_recolor", STAGE / "wallpaper_recolor")

    if EXAMPLES_SRC.is_dir():
        examples = STAGE / "examples"
        examples.mkdir()
        for item in EXAMPLES_SRC.iterdir():
            if item.is_file():
                shutil.copy2(item, examples / item.name)

    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in STAGE.rglob("*"):
            if path.is_file():
                zf.write(path, Path("WallpaperRecolor") / path.relative_to(STAGE))

    unzipped = sum(p.stat().st_size for p in STAGE.rglob("*") if p.is_file())
    print(f"Folder: {STAGE}")
    print(f"Zip:    {ZIP_PATH}")
    print(f"Unzipped bytes: {unzipped}")
    print(f"Zip bytes:      {ZIP_PATH.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
