#!/usr/bin/env python3
"""Tiny read-only API: GET /api/v1.0/get-containers -> running Docker containers (via /var/run/docker.sock)."""
import json, os, socket, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SOCK = "/var/run/docker.sock"
PORT = int(os.environ.get("CONTAINERS_API_PORT", "34002"))

def docker_get(path):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(3)
    try:
        s.connect(SOCK)
        s.sendall(f"GET {path} HTTP/1.0\r\nHost: localhost\r\n\r\n".encode())
        data = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
    finally:
        s.close()
    return json.loads(data.split(b"\r\n\r\n", 1)[1].decode())

def short_name(n):
    parts = n.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        n = "_".join(parts[:-1])
    if n.endswith("_app"):
        n = n[:-4]
    return n

def containers():
    out = []
    for c in docker_get("/containers/json?all=1"):
        name = (c.get("Names") or ["/" + c.get("Id", "")[:12]])[0].lstrip("/")
        labels = c.get("Labels") or {}
        out.append({
            "name": name,
            "app": labels.get("com.docker.compose.project") or short_name(name),
            "service": labels.get("com.docker.compose.service"),
            "image": c.get("Image", "").split("@")[0],
            "state": c.get("State"),
            "status": c.get("Status"),
            "created": c.get("Created"),
        })
    out.sort(key=lambda x: (x["state"] != "running", x["app"] or "", x["name"]))
    return out

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def do_GET(self):
        if self.path.split("?")[0].rstrip("/") != "/api/v1.0/get-containers":
            self.send_response(404); self.end_headers(); return
        try:
            body = {"status": True, "data": containers(), "time": int(time.time() * 1000)}
            code = 200
        except Exception as e:
            body, code = {"status": False, "error": str(e)}, 500
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
