#!/bin/bash
set -e

# Checks
if [ "$EUID" -ne 0 ]; then 
  echo "Please run as root"
  exit 1
fi

echo "--- 1. Setting up Database & Backend ---"

# Install System Dependencies
echo "Installing System Packages..."
apt-get update
apt-get install -y python3-pip python3-venv postgresql postgresql-contrib libpq-dev

# Setup Postgres
echo "Configuring Postgres..."
service postgresql start
sudo -u postgres psql -c "DO \$\$ BEGIN IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'drrl_user') THEN CREATE ROLE drrl_user LOGIN PASSWORD 'drrl_pass'; END IF; END \$\$;"
sudo -u postgres psql -c "CREATE DATABASE drrl_db OWNER drrl_user;" || true
sudo -u postgres psql -c "ALTER USER drrl_user CREATEDB;"

# Setup Python
echo "Setting up Python Environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv --system-site-packages
fi
source .venv/bin/activate

echo "Installing Backend Dependencies..."
pip install wheel setuptools uvicorn psycopg2-binary asyncpg
if [ -f "services/orchestrator/requirements.txt" ]; then pip install -r services/orchestrator/requirements.txt; fi

# Env Vars
export DATABASE_URL="postgresql://drrl_user:drrl_pass@localhost/drrl_db"
export PYTHONPATH=/root/A_1
cd /root/A_1

echo "--- Starting Orchestrator (System 2) ---"
echo "Listening on port 8000..."
uvicorn services.orchestrator.main:app --host 0.0.0.0 --port 8000 --reload
