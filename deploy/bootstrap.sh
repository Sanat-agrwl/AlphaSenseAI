#!/bin/bash
# AlphaSense EC2 Bootstrap Script
# Run once on a fresh Ubuntu 22.04 t3.small instance
# Usage: bash bootstrap.sh
set -e

echo "════════════════════════════════════════════"
echo "  AlphaSense — EC2 Bootstrap"
echo "════════════════════════════════════════════"

# ── System packages ───────────────────────────────────────────────
echo "[1/7] Installing system packages..."
sudo apt-get update -q
sudo apt-get install -y -q \
  python3-pip python3-venv python3-dev \
  nginx git ffmpeg build-essential \
  htop curl wget unzip

# ── Swap (t3.small only has 2GB RAM — swap helps with FinBERT) ───
echo "[2/7] Setting up 4GB swap..."
if [ ! -f /swapfile ]; then
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi
echo "Swap: $(free -h | grep Swap)"

# ── Clone repo ────────────────────────────────────────────────────
echo "[3/7] Cloning AlphaSenseAI repo..."
cd /home/ubuntu
if [ ! -d "AlphaSenseAI" ]; then
  git clone https://github.com/Sanat-agrwl/AlphaSenseAI.git
fi
cd AlphaSenseAI

# ── Python virtualenv ─────────────────────────────────────────────
echo "[4/7] Creating Python virtualenv..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "Python: $(python --version)"

# ── .env file ─────────────────────────────────────────────────────
echo "[5/7] Setting up .env..."
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo ""
  echo "⚠️  Edit /home/ubuntu/AlphaSenseAI/.env with your API keys before running the pipeline!"
  echo "    nano /home/ubuntu/AlphaSenseAI/.env"
fi

# Create logs dir
mkdir -p logs data/nse data/bse data/news

# ── Streamlit systemd service ─────────────────────────────────────
echo "[6/7] Setting up Streamlit as a service..."
sudo tee /etc/systemd/system/alphasense.service > /dev/null <<'SERVICE'
[Unit]
Description=AlphaSense Streamlit Dashboard
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/AlphaSenseAI
ExecStart=/home/ubuntu/AlphaSenseAI/venv/bin/streamlit run alphasense/dashboard/app.py \
  --server.port 8501 \
  --server.headless true \
  --server.address 0.0.0.0
Restart=always
RestartSec=10
Environment="PYTHONPATH=/home/ubuntu/AlphaSenseAI"

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable alphasense
sudo systemctl start alphasense

# ── Nginx reverse proxy ───────────────────────────────────────────
sudo tee /etc/nginx/sites-available/alphasense > /dev/null <<'NGINX'
server {
    listen 80;
    location / {
        proxy_pass         http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_read_timeout 86400;
    }
}
NGINX

sudo ln -sf /etc/nginx/sites-available/alphasense /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

# ── Cron jobs (IST = UTC+5:30) ────────────────────────────────────
echo "[7/7] Setting up cron jobs..."
(crontab -l 2>/dev/null; cat <<'CRON'
# AlphaSense pipeline — Mon-Fri only
# Pre-market 08:30 IST = 03:00 UTC
0 3 * * 1-5 cd /home/ubuntu/AlphaSenseAI && venv/bin/python alphasense/pipeline/daily.py --pre-market >> logs/cron.log 2>&1
# Post-close 15:45 IST = 10:15 UTC
15 10 * * 1-5 cd /home/ubuntu/AlphaSenseAI && venv/bin/python alphasense/pipeline/daily.py --post-close >> logs/cron.log 2>&1
# Nightly report 23:00 IST = 17:30 UTC
30 17 * * 1-5 cd /home/ubuntu/AlphaSenseAI && venv/bin/python scripts/run_backtest.py --period test >> logs/cron.log 2>&1
CRON
) | crontab -

echo ""
echo "════════════════════════════════════════════"
echo "  Bootstrap complete!"
echo "════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Fill in secrets:  nano /home/ubuntu/AlphaSenseAI/.env"
echo "  2. Backfill prices:  source venv/bin/activate && python scripts/fetch_prices.py --backfill"
echo "  3. Build universe:   python scripts/build_universe.py --build"
echo "  4. Dashboard:        http://$(curl -s ifconfig.me)"
echo ""
echo "Service status:  sudo systemctl status alphasense"
echo "Logs:            tail -f /home/ubuntu/AlphaSenseAI/logs/cron.log"
