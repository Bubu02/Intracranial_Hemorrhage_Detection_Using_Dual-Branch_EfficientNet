@echo off
REM AI Brain Hemorrhage Detection System - Startup Script
REM Uses the local conda environment

echo ========================================
echo  AI Brain Hemorrhage Detection System
echo ========================================
echo.
echo Starting Flask server...
echo.

REM Activate conda environment and run Flask app
.\.conda\python.exe app.py

pause
