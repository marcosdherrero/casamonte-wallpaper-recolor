@echo off
REM Launch the wallpaper color-range remapper (TIF / PNG / JPEG).
cd /d "%~dp0"
python run_app.py
if errorlevel 1 pause
