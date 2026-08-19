@echo off
REM One-time setup: private CPython in runtime\ (no admin) + Pillow/numpy/matplotlib.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_coworker.ps1"
if errorlevel 1 (
  echo.
  echo Install failed. Need network the first time. No admin should be required.
  pause
  exit /b 1
)
echo.
pause
