#!/bin/bash
set -e

echo "--- Setting up Native Environment (No Docker) ---"

# 1. Install System Dependencies
# Check if root
if [ "$EUID" -ne 0 ]; then 
  echo "Please run as root"
  exit 1
fi

apt-get update
# Install Postgres and Python dev tools
apt-get install -y python3-pip python3-venv postgresql postgresql-contrib libpq-dev

# 2. Setup Postgres
echo "Configuring Postgres..."
service postgresql start
# Create user if not exists
sudo -u postgres psql -c "DO \$\$ BEGIN IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'drrl_user') THEN CREATE ROLE drrl_user LOGIN PASSWORD 'drrl_pass'; END IF; END \$\$;"
sudo -u postgres psql -c "CREATE DATABASE drrl_db OWNER drrl_user;" || true
sudo -u postgres psql -c "ALTER USER drrl_user CREATEDB;"

# 3. Setup Python Environment
echo "Setting up Python Virtual Environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv --system-site-packages
fi
source .venv/bin/activate

echo "Installing Python Dependencies..."
pip install --upgrade pip
# Install typical build deps just in case
pip install wheel setuptools

# Install Service Requirements
if [ -f "services/trainer/requirements.txt" ]; then pip install -r services/trainer/requirements.txt; fi
if [ -f "services/orchestrator/requirements.txt" ]; then pip install -r services/orchestrator/requirements.txt; fi
if [ -f "services/stack_server/requirements.txt" ]; then pip install -r services/stack_server/requirements.txt; fi

# Install missing specific deps that might not be in reqs
pip install uvicorn psycopg2-binary asyncpg

# 4. Start Orchestrator
echo "Starting Orchestrator Service..."
export DATABASE_URL="postgresql://drrl_user:drrl_pass@localhost/drrl_db"
export STACK_SERVER_URL="http://localhost:8000"
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run in background
nohup uvicorn services.orchestrator.main:app --host 0.0.0.0 --port 8000 > orchestrator.log 2>&1 &
ORCH_PID=$!
echo "Orchestrator PID: $ORCH_PID"

# Wait for healthy
echo "Waiting for Orchestrator..."
sleep 5

# 5. Run Trainer
echo "--- Starting Training ---"

# Data Setup
echo "Checking/Downloading Datasets..."
pip install datasets
python3 scripts/download_data.py

if [ ! -f "training_data/01_english.jsonl" ]; then
    echo "WARNING: Dataset download failed or incomplete."
    echo "Script will pause for 10 seconds then attempt to run..."
    sleep 10
fi

python3 services/trainer/train_drrl.py

# Cleanup on exit
kill $ORCH_PID
