@echo off
setlocal
cd /d "%~dp0.."
"venv\Scripts\python.exe" -m factory.planner run --execute --max-cpu-workers 2 >> "factory\planner.log" 2>> "factory\planner-error.log"
