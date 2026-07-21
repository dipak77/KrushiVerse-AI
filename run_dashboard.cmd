@echo off
setlocal
cd /d "%~dp0"
echo Starting KrushiVerseAI Autonomous Factory Dashboard on http://localhost:8501 ...
"venv\Scripts\python.exe" -m streamlit run factory/dashboard.py --server.port 8501
pause
