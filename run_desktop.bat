@echo off
title ASL Real-Time Translator // Neubrutalism Edition
echo ========================================================
echo   Launching ASL Real-Time Translator (Desktop Mode)...
echo ========================================================

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" desktop_app.py
) else (
    python desktop_app.py
)

if errorlevel 1 (
    echo.
    echo [!] Application exited with an error.
    pause
)
