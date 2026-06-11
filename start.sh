#!/bin/bash
# SmartHouse AI local environment bootstrapper script
set -e

echo "============================================="
echo "   SMARTHOUSE AI PLATFORM LOCAL BOOTSTRAP   "
echo "============================================="

# 1. Setup Virtual Environment
if [ ! -d ".venv" ]; then
    echo "[MLOps] Virtual environment not found. Creating..."
    python3 -m venv .venv
fi

echo "[MLOps] Activating virtual environment..."
source .venv/bin/activate

# 2. Install Dependencies
echo "[MLOps] Installing python package requirements..."
pip install --upgrade pip
pip install -r backend/requirements.txt

# 3. Database Initialization & Data Loading
echo "[Data] Bootstrapping synthetic relational records (100,000 listings)..."
python data/generate_data.py

# 4. Train Valuation Regressors
echo "[ML] Running feature engineering and training 7 regressors..."
python ml/train.py

# 5. Generate Forecasting Predictions
echo "[ML] Running time-series market forecasting trend models..."
python ml/forecaster.py

# 6. Start FastAPI Backend Web Server
echo "[Server] Launching FastAPI Web Service and mounting dashboard on http://127.0.0.1:8000"
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
