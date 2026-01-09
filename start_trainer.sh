#!/bin/bash
set -e

# Checks
if [ "$EUID" -ne 0 ]; then 
  echo "Please run as root"
  exit 1
fi

echo "--- 2. Setting up Trainer & Data ---"

# Setup Python (Assume venv created by orch script, or create if missing)
if [ ! -d ".venv" ]; then
    python3 -m venv .venv --system-site-packages
fi
source .venv/bin/activate

echo "Installing Trainer Dependencies..."
pip install wheel setuptools datasets tqdm transformers torch accelerate sentry-sdk
if [ -f "services/trainer/requirements.txt" ]; then pip install -r services/trainer/requirements.txt; fi

# Env Vars
export STACK_SERVER_URL="http://localhost:8000"
export PYTHONPATH=/root/A_1
cd /root/A_1

# Data Download
echo "--- Checking Curriculum Data ---"
python3 scripts/download_data.py

echo "--- Starting Training (System 1) ---"
python3 services/trainer/train_drrl.py
