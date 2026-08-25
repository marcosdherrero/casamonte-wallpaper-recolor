@echo off
REM Launch using Global Python (whatever `python` is on PATH).
cd /d "%~dp0"
python run_app.py
if errorlevel 1 pause
