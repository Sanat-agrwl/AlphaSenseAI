#!/bin/bash
# Starts IB Gateway headlessly with Xvfb + IBC auto-login
# Reads IBKR_USERNAME and IBKR_PASSWORD from AlphaSenseAI/.env

set -e
ENV_FILE="/home/ubuntu/AlphaSenseAI/.env"
if [ -f "$ENV_FILE" ]; then
    export $(grep -E '^IBKR_' "$ENV_FILE" | xargs)
fi

if [ -z "$IBKR_USERNAME" ] || [ -z "$IBKR_PASSWORD" ]; then
    echo "ERROR: IBKR_USERNAME and IBKR_PASSWORD must be set in .env"
    exit 1
fi

sed -i "s/IbLoginId=.*/IbLoginId=${IBKR_USERNAME}/" /home/ubuntu/ibc/config.ini
sed -i "s/IbPassword=.*/IbPassword=${IBKR_PASSWORD}/" /home/ubuntu/ibc/config.ini

pkill -f "Xvfb :1" 2>/dev/null || true
sleep 1

Xvfb :1 -screen 0 1024x768x24 &
echo $! > /tmp/xvfb.pid
sleep 2

export DISPLAY=:1
export TWS_MAJOR_VRSN=1037
export IBC_INI=/home/ubuntu/ibc/config.ini
export TWS_PATH=/home/ubuntu/ibgateway
export IBC_PATH=/home/ubuntu/ibc
export LOG_PATH=/home/ubuntu/AlphaSenseAI/logs
export TWSUSERID=$IBKR_USERNAME
export TWSPASSWORD=$IBKR_PASSWORD
export TRADING_MODE=paper

mkdir -p $LOG_PATH
cd /home/ubuntu/ibc

echo "Starting IB Gateway (paper) for $IBKR_USERNAME ..."
exec /home/ubuntu/ibc/gatewaystart.sh \
    "$TWS_MAJOR_VRSN" \
    "$IBC_INI" \
    "$TWS_PATH" \
    "$IBC_PATH" \
    "$LOG_PATH" \
    "$TRADING_MODE" \
    >> "$LOG_PATH/ibgateway.log" 2>&1
