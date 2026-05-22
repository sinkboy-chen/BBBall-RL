#!/usr/bin/env python3
"""
web_server.py — Flask web server for scrcpy remote control
===========================================================
Streams the Android screen via MJPEG and provides REST API for control.

Usage:
    source .venv/bin/activate
    python web_server.py

Then SSH tunnel:
    ssh -L 5000:localhost:5000 <your_id>@meow1.csie.ntu.edu.tw

Open: http://localhost:5000
"""

import io
import time
import threading
import signal
import sys
import os
import json

import cv2
import numpy as np
from flask import Flask, Response, request, jsonify

from scrcpy_client import ScrcpyClient

# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
client: ScrcpyClient = None

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>scrcpy Web Client</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Inter', sans-serif;
    background: #0f0f13;
    color: #e0e0e6;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  header {
    width: 100%;
    padding: 16px 24px;
    background: linear-gradient(135deg, #1a1a24 0%, #12121a 100%);
    border-bottom: 1px solid #2a2a3a;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  header h1 {
    font-size: 18px;
    font-weight: 600;
    background: linear-gradient(135deg, #7c6cf0, #4ecdc4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  #status {
    font-size: 12px;
    color: #888;
    margin-left: auto;
  }
  #status .dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #4ecdc4;
    margin-right: 6px;
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }
  .main {
    display: flex;
    gap: 20px;
    padding: 20px;
    max-width: 1200px;
    width: 100%;
    flex-wrap: wrap;
    justify-content: center;
  }
  .video-wrap {
    position: relative;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    border: 1px solid #2a2a3a;
    background: #000;
    flex-shrink: 0;
  }
  #stream {
    display: block;
    max-height: 80vh;
    max-width: 100%;
    cursor: crosshair;
  }
  #tap-indicator {
    position: absolute;
    width: 28px; height: 28px;
    border: 2px solid #4ecdc4;
    border-radius: 50%;
    transform: translate(-50%, -50%);
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.1s;
  }
  #tap-indicator.show {
    opacity: 1;
    animation: tap-ring 0.5s ease-out forwards;
  }
  @keyframes tap-ring {
    0% { transform: translate(-50%,-50%) scale(0.5); opacity: 1; }
    100% { transform: translate(-50%,-50%) scale(1.5); opacity: 0; }
  }
  .panel {
    background: linear-gradient(135deg, #1a1a24, #15151f);
    border: 1px solid #2a2a3a;
    border-radius: 12px;
    padding: 20px;
    min-width: 260px;
    max-width: 320px;
  }
  .panel h2 {
    font-size: 14px;
    font-weight: 600;
    color: #aaa;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 16px;
  }
  .info-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #1f1f2e;
    font-size: 13px;
  }
  .info-row .label { color: #777; }
  .info-row .value { color: #ccc; font-weight: 500; }
  .btn-row { display: flex; gap: 8px; margin: 8px 0; flex-wrap: wrap; }
  .btn {
    padding: 8px 16px;
    border: 1px solid #3a3a4a;
    background: #22222e;
    color: #ccc;
    border-radius: 8px;
    cursor: pointer;
    font-size: 13px;
    font-family: inherit;
    transition: all 0.15s;
  }
  .btn:hover { background: #2e2e3e; border-color: #4ecdc4; color: #fff; }
  .btn:active { transform: scale(0.96); }
  .coord-input {
    display: flex; gap: 8px; margin: 8px 0; align-items: center;
  }
  .coord-input input {
    width: 70px; padding: 6px 10px;
    background: #1a1a24; border: 1px solid #3a3a4a;
    border-radius: 6px; color: #e0e0e6;
    font-family: 'Inter', monospace; font-size: 13px;
  }
  .coord-input input:focus { outline: none; border-color: #7c6cf0; }
  #log {
    margin-top: 12px;
    max-height: 120px;
    overflow-y: auto;
    font-size: 11px;
    color: #666;
    font-family: monospace;
    background: #12121a;
    padding: 8px;
    border-radius: 6px;
  }
  #log div { padding: 2px 0; }
  #coords-display {
    position: absolute;
    bottom: 8px; left: 8px;
    background: rgba(0,0,0,0.7);
    color: #4ecdc4;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-family: monospace;
    pointer-events: none;
  }
</style>
</head>
<body>
<header>
  <h1>scrcpy Web Client</h1>
  <div id="status"><span class="dot"></span><span id="status-text">Connecting...</span></div>
</header>
<div class="main">
  <div class="video-wrap">
    <img id="stream" src="/video_feed" alt="Android Screen">
    <div id="tap-indicator"></div>
    <div id="coords-display">—</div>
  </div>
  <div class="panel">
    <h2>Device Info</h2>
    <div class="info-row"><span class="label">Device</span><span class="value" id="dev-name">—</span></div>
    <div class="info-row"><span class="label">Resolution</span><span class="value" id="dev-res">—</span></div>

    <h2 style="margin-top:20px">Navigation</h2>
    <div class="btn-row">
      <button class="btn" onclick="sendKey('back')">◀ Back</button>
      <button class="btn" onclick="sendKey('home')">● Home</button>
      <button class="btn" onclick="sendKey('recent')">■ Recent</button>
    </div>

    <h2 style="margin-top:20px">Manual Tap</h2>
    <div class="coord-input">
      <span style="color:#777;font-size:13px">X</span>
      <input type="number" id="tap-x" placeholder="x">
      <span style="color:#777;font-size:13px">Y</span>
      <input type="number" id="tap-y" placeholder="y">
      <button class="btn" onclick="manualTap()">Tap</button>
    </div>

    <h2 style="margin-top:20px">Manual Swipe</h2>
    <div class="coord-input">
      <input type="number" id="sw-x1" placeholder="x1" style="width:55px">
      <input type="number" id="sw-y1" placeholder="y1" style="width:55px">
      <span style="color:#555">→</span>
      <input type="number" id="sw-x2" placeholder="x2" style="width:55px">
      <input type="number" id="sw-y2" placeholder="y2" style="width:55px">
    </div>
    <button class="btn" onclick="manualSwipe()" style="margin-top:4px">Swipe</button>

    <h2 style="margin-top:20px">Log</h2>
    <div id="log"></div>
  </div>
</div>

<script>
const img = document.getElementById('stream');
const indicator = document.getElementById('tap-indicator');
const coordsDisp = document.getElementById('coords-display');
const logEl = document.getElementById('log');
let devWidth = 0, devHeight = 0;

function log(msg) {
  const d = document.createElement('div');
  d.textContent = new Date().toLocaleTimeString() + ' ' + msg;
  logEl.prepend(d);
  while (logEl.children.length > 50) logEl.lastChild.remove();
}

// Fetch device info
function fetchInfo() {
  fetch('/api/info').then(r => r.json()).then(d => {
    document.getElementById('dev-name').textContent = d.device_name || '—';
    document.getElementById('dev-res').textContent = d.width + 'x' + d.height;
    document.getElementById('status-text').textContent = d.device_name || 'Connected';
    devWidth = d.width;
    devHeight = d.height;
  }).catch(() => {});
}
fetchInfo();
setInterval(fetchInfo, 5000);

// Click-to-tap on video
img.addEventListener('click', function(e) {
  if (!devWidth || !devHeight) return;
  const rect = img.getBoundingClientRect();
  const scaleX = devWidth / rect.width;
  const scaleY = devHeight / rect.height;
  const x = Math.round((e.clientX - rect.left) * scaleX);
  const y = Math.round((e.clientY - rect.top) * scaleY);

  // Visual indicator
  indicator.style.left = (e.clientX - rect.left) + 'px';
  indicator.style.top = (e.clientY - rect.top) + 'px';
  indicator.classList.remove('show');
  void indicator.offsetWidth;
  indicator.classList.add('show');

  fetch('/api/tap', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({x, y})
  }).then(r => r.json()).then(d => log('Tap (' + x + ', ' + y + ')')).catch(e => log('Error: ' + e));
});

// Mouse move shows coordinates
img.addEventListener('mousemove', function(e) {
  if (!devWidth || !devHeight) return;
  const rect = img.getBoundingClientRect();
  const x = Math.round((e.clientX - rect.left) * devWidth / rect.width);
  const y = Math.round((e.clientY - rect.top) * devHeight / rect.height);
  coordsDisp.textContent = x + ', ' + y;
});

function sendKey(key) {
  fetch('/api/key', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({key})
  }).then(r => r.json()).then(() => log('Key: ' + key)).catch(e => log('Error: ' + e));
}

function manualTap() {
  const x = parseInt(document.getElementById('tap-x').value);
  const y = parseInt(document.getElementById('tap-y').value);
  if (isNaN(x) || isNaN(y)) return;
  fetch('/api/tap', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({x, y})
  }).then(r => r.json()).then(() => log('Tap (' + x + ', ' + y + ')')).catch(e => log('Error: ' + e));
}

function manualSwipe() {
  const x1 = parseInt(document.getElementById('sw-x1').value);
  const y1 = parseInt(document.getElementById('sw-y1').value);
  const x2 = parseInt(document.getElementById('sw-x2').value);
  const y2 = parseInt(document.getElementById('sw-y2').value);
  if ([x1,y1,x2,y2].some(isNaN)) return;
  fetch('/api/swipe', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({x1, y1, x2, y2, duration_ms: 300})
  }).then(r => r.json()).then(() => log('Swipe')).catch(e => log('Error: ' + e));
}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return HTML_PAGE


def gen_mjpeg():
    """Generator yielding MJPEG frames."""
    while True:
        frame = client.get_frame()
        if frame is not None:
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   jpeg.tobytes() + b"\r\n")
        time.sleep(1 / 30)  # ~30 fps cap


@app.route("/video_feed")
def video_feed():
    return Response(gen_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/info")
def api_info():
    return jsonify({
        "device_name": client.device_name,
        "width": client.video_width,
        "height": client.video_height,
        "running": client.is_running,
    })


@app.route("/api/tap", methods=["POST"])
def api_tap():
    data = request.get_json()
    x, y = int(data["x"]), int(data["y"])
    duration = int(data.get("duration_ms", 50))
    threading.Thread(target=client.tap, args=(x, y, duration), daemon=True).start()
    return jsonify({"ok": True, "x": x, "y": y})


@app.route("/api/swipe", methods=["POST"])
def api_swipe():
    data = request.get_json()
    x1, y1 = int(data["x1"]), int(data["y1"])
    x2, y2 = int(data["x2"]), int(data["y2"])
    duration = int(data.get("duration_ms", 300))
    steps = int(data.get("steps", 20))
    threading.Thread(target=client.swipe,
                     args=(x1, y1, x2, y2, duration, steps), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/key", methods=["POST"])
def api_key():
    data = request.get_json()
    key = data["key"]
    key_map = {"back": client.press_back, "home": client.press_home,
               "recent": client.press_app_switch}
    fn = key_map.get(key)
    if fn:
        threading.Thread(target=fn, daemon=True).start()
        return jsonify({"ok": True, "key": key})
    return jsonify({"ok": False, "error": "unknown key"}), 400


@app.route("/api/screenshot")
def api_screenshot():
    frame = client.get_frame()
    if frame is None:
        return jsonify({"error": "no frame"}), 503
    _, png = cv2.imencode('.png', frame)
    return Response(png.tobytes(), mimetype="image/png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global client
    import argparse
    parser = argparse.ArgumentParser(description="scrcpy Web Server")
    parser.add_argument("-p", "--port", type=int, default=5000, help="Flask port (default 5000)")
    parser.add_argument("--max-size", type=int, default=480, help="Max video dimension (default 480)")
    parser.add_argument("--scrcpy-port", type=int, default=27183, help="scrcpy tunnel port")
    args = parser.parse_args()

    client = ScrcpyClient(max_size=args.max_size, port=args.scrcpy_port)

    def shutdown(sig, frame):
        client.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        client.start()
        print(f"\n  Web server starting on http://0.0.0.0:{args.port}")
        print(f"  SSH tunnel: ssh -L {args.port}:localhost:{args.port} <user>@meow1.csie.ntu.edu.tw")
        print(f"  Then open:  http://localhost:{args.port}\n")
        app.run(host="0.0.0.0", port=args.port, threaded=True)
    finally:
        client.stop()


if __name__ == "__main__":
    main()
