@echo off
setlocal
cd /d "%~dp0"
"venv\Scripts\python.exe" -m factory.status_reporter
echo.
pause
