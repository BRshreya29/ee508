#!/usr/bin/env bash
# Serve Charades videos over a public HTTPS tunnel for Colab to access.
# Requirements: cloudflared (installed automatically below)
#
# Usage:
#   cd ~/studies/project/ee508
#   bash scripts/serve_videos.sh
#
# Copy the tunnel URL printed (e.g. https://xxxx.trycloudflare.com) into
# the Colab notebook when prompted.

set -e

VIDEO_DIR="/media/shreya/Elements/ee508_data/charades_videos/Charades_v1_480"
PORT=8765

# ── 1. Install cloudflared if not present ───────────────────────────────────
if ! command -v cloudflared &>/dev/null; then
    echo "Installing cloudflared..."
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -O /tmp/cloudflared.deb
    sudo dpkg -i /tmp/cloudflared.deb
fi

# ── 2. Start Python HTTP file server in background ──────────────────────────
echo "Serving $VIDEO_DIR on port $PORT ..."
cd "$VIDEO_DIR"
python3 -m http.server $PORT &
SERVER_PID=$!
echo "HTTP server PID: $SERVER_PID"

# ── 3. Start cloudflared tunnel ─────────────────────────────────────────────
echo ""
echo "Starting cloudflared tunnel..."
echo "─────────────────────────────────────────────────────────"
echo "Copy the URL that appears below (https://xxxx.trycloudflare.com)"
echo "into the Colab notebook TUNNEL_URL variable."
echo "─────────────────────────────────────────────────────────"
echo ""

cloudflared tunnel --protocol http2 --url http://localhost:$PORT &
TUNNEL_PID=$!

# Wait for Ctrl+C, then clean up
trap "echo 'Shutting down...'; kill $SERVER_PID $TUNNEL_PID 2>/dev/null" EXIT INT TERM
wait $TUNNEL_PID
