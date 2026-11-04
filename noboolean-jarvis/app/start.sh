#!/bin/bash
set -e
cd /data
mkdir -p home/.grok img voices npm hf
export NPM_CONFIG_PREFIX=/data/npm PATH=/data/npm/bin:$PATH HF_HOME=/data/hf
if [ ! -x /data/venv/bin/uvicorn ]; then
  python3 -m venv /data/venv
  /data/venv/bin/pip install --no-cache-dir -r /app/requirements.txt
fi
[ -x /data/npm/bin/grok ] || npm install -g @xai-official/grok
ln -sf /app/server.py /data/server.py
exec /data/venv/bin/uvicorn server:app --app-dir /data --host 0.0.0.0 --port 8765
