@echo off
setlocal
cd /d "%~dp0.."
"venv\Scripts\python.exe" -m factory.monitor --once >> "factory\monitor.log" 2>> "factory\monitor-error.log"
