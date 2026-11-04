#!/bin/bash
set -e
cd /data
# secrets (ELEVENLABS_API_KEY, EL_VOICE_ID) live only in app-data, never in the repo
if [ -f /data/.env ]; then set -a; . /data/.env; set +a; fi
mkdir -p home/.grok img voices npm hf
export NPM_CONFIG_PREFIX=/data/npm PATH=/data/npm/bin:$PATH HF_HOME=/data/hf
if [ ! -x /data/venv/bin/uvicorn ]; then
  python3 -m venv /data/venv
fi
# keep deps in sync with the pinned requirements on every start (no-op when already satisfied)
/data/venv/bin/pip install -q --no-cache-dir -r /app/requirements.txt
[ -x /data/npm/bin/grok ] || npm install -g @xai-official/grok
ln -sf /app/server.py /data/server.py
exec /data/venv/bin/uvicorn server:app --app-dir /data --host 0.0.0.0 --port 8765
