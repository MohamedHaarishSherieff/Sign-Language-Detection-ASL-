@echo off
title ASL Real-Time Translator // Web Server
echo ========================================================
echo   Launching ASL Real-Time Translator (Web Server)...
echo ========================================================

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py
) else (
    python app.py
)

if errorlevel 1 (
    echo.
    echo [!] Server stopped or exited with an error.
    pause
)
