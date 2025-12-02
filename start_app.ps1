# AI Brain Hemorrhage Detection System - Startup Script (PowerShell)
# Uses the local conda environment

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " AI Brain Hemorrhage Detection System" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting Flask server..." -ForegroundColor Green
Write-Host ""

# Run Flask app with local conda Python
& ".\.conda\python.exe" app.py
