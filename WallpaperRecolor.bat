@echo off
REM Launch Wallpaper Recolor using the folder-local Python from Install.bat.
cd /d "%~dp0"
set "PY=%~dp0runtime\python.exe"
set "PYW=%~dp0runtime\pythonw.exe"
if not exist "%PY%" set "PY=%LOCALAPPDATA%\WallpaperRecolor\python\python.exe"
if not exist "%PYW%" set "PYW=%LOCALAPPDATA%\WallpaperRecolor\python\pythonw.exe"
if not exist "%PY%" (
  echo Run Install.bat once first ^(needs network; installs Python in this folder^).
  pause
  exit /b 1
)
if exist "%PYW%" (
  start "" "%PYW%" "run_app.py"
) else (
  "%PY%" "run_app.py"
  if errorlevel 1 pause
)
