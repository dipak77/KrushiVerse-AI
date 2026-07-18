#!/bin/bash
set -e

VENV_DIR="./venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    ./venv/bin/pip install --upgrade pip
    ./venv/bin/pip install -r requirements.txt
fi

case "$1" in
    api)
        echo "Starting FastAPI backend server on http://localhost:8000 ..."
        ./venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
        ;;
    ui)
        echo "Starting Streamlit UI dashboard on http://localhost:8501 ..."
        ./venv/bin/streamlit run ui/dashboard.py --server.port 8501
        ;;
    test)
        echo "Running pytest suite..."
        ./venv/bin/pytest -v
        ;;
    *)
        echo "Usage: ./run.sh {api|ui|test}"
        exit 1
        ;;
esac
