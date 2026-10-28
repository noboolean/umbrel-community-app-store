from itertools import islice
import json
import socket
import time

from pm_auto.libs.utils import get_font
from pm_auto.libs.oled_page import OLEDPage

font = get_font("UbuntuSans-Regular.ttf")

def _docker_get(path):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(2)
    try:
        s.connect("/var/run/docker.sock")
        req = f"GET {path} HTTP/1.0\r\nHost: localhost\r\n\r\n".encode()
        s.sendall(req)
        data = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
    finally:
        s.close()
    if b"\r\n\r\n" not in data:
        return []
    body = data.split(b"\r\n\r\n", 1)[1]
    return json.loads(body.decode())

def _short_name(c):
    names = c.get("Names") or []
    if not names:
        return c.get("Id", "")[:8]
    n = names[0].lstrip("/")
    parts = n.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        n = "_".join(parts[:-1])
        if n.endswith("_app"):
            n = n[:-4]
    return n[:18]

class PageContainers(OLEDPage):
    def __init__(self):
        super().__init__()
        self.index = 0
        self.count = 0
        self.cycle_time_start = 0
        self.cache = []
        self.cache_ts = 0

    def main(self, oled, data, config):
        scroll_interval = config.get("scroll_interval", 3)
        now = time.time()
        if now - self.cache_ts > 5:
            try:
                raw = _docker_get("/containers/json")
                self.cache = sorted({_short_name(c) for c in raw})
            except Exception:
                self.cache = []
            self.cache_ts = now

        names = self.cache
        oled.clear()
        oled.draw_text(f"APPS {len(names)}", 0, 0, size=12, font_path=font)

        if not names:
            oled.draw_text("none / no sock", 0, 24, size=12, font_path=font)
            oled.display()
            return

        if self.count != len(names):
            self.index = 0
            self.cycle_time_start = now
            self.count = len(names)

        if now - self.cycle_time_start >= scroll_interval:
            self.cycle_time_start = now
            self.index += 3
            if self.index >= len(names):
                self.index = 0

        for i, name in enumerate(islice(names, self.index, self.index + 3)):
            oled.draw_text(name, 0, 16 + i * 16, size=12, font_path=font)
        oled.display()
