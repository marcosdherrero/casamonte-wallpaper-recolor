@echo off
REM Lean Windows onedir zip for a coworker (no Python on their PC).
REM From repo root:  scripts\build_exe.bat
setlocal EnableExtensions
cd /d "%~dp0\.."

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
  echo Need Python 3.10+ with Tkinter on this machine to build the exe.
  exit /b 1
)

echo Installing PyInstaller and app deps ^(Pillow, numpy, matplotlib^)...
python -m pip install -q "pyinstaller>=6.0" -r requirements.txt -r requirements-plot.txt
if errorlevel 1 exit /b 1

echo Building dist\WallpaperRecolor\ ^(onedir, no EasyOCR/LaMa^)...
python -m PyInstaller --noconfirm --clean WallpaperRecolor.spec
if errorlevel 1 exit /b 1

echo Packaging examples, README, and zip...
python scripts\package_exe.py
if errorlevel 1 exit /b 1

echo Done.
exit /b 0
