@echo off
setlocal
cd /d "%~dp0.."
"venv\Scripts\python.exe" -m factory.status_reporter >> "factory\status.log" 2>> "factory\status-error.log"
