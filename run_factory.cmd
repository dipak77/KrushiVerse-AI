@echo off
setlocal
cd /d "%~dp0"
echo Starting KrushiVerseAI Autonomous Factory Planner...
"venv\Scripts\python.exe" -m factory.planner run --execute --max-cpu-workers 4
pause
