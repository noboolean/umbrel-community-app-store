# noboolean: top-3 containers by CPU% (mem as tiebreak). Docker stats are gathered in a
# background thread every SWEEP seconds so the OLED loop never blocks.
import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from pm_auto.libs.utils import get_font
from pm_auto.libs.oled_page import OLEDPage
from PIL import ImageFont

font = get_font("UbuntuSans-Regular.ttf")
SOCK = "/var/run/docker.sock"
SWEEP = 20


def _docker_get(path, timeout=5):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
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
    if b"\r\n\r\n" not in data:
        return None
    return json.loads(data.split(b"\r\n\r\n", 1)[1].decode())


def _short_name(c):
    labels = c.get("Labels") or {}
    proj, svc = labels.get("com.docker.compose.project"), labels.get("com.docker.compose.service")
    if proj and "-" in proj and proj.split("-", 1)[0] in ("noboolean", "sunfounder"):
        proj = proj.split("-", 1)[1]
    if proj and svc:
        return proj if svc in ("app", "web", "server") and len(proj) <= 14 else f"{proj}/{svc}"
    names = c.get("Names") or []
    return names[0].lstrip("/") if names else c.get("Id", "")[:8]


class _Sampler:
    def __init__(self):
        self.lock = threading.Lock()
        self.top = []          # [(name, cpu%, mem_bytes)]
        self.running = 0
        self.apps = 0
        self.ready = False
        self.prev = {}         # id -> (cpu_total, system_total)
        t = threading.Thread(target=self.loop, daemon=True, name="oled-containers")
        t.start()

    def stat(self, c):
        st = _docker_get(f"/containers/{c['Id']}/stats?stream=false&one-shot=true", timeout=8) or {}
        cpu = st.get("cpu_stats", {})
        total = cpu.get("cpu_usage", {}).get("total_usage", 0)
        system = cpu.get("system_cpu_usage", 0)
        ncpu = cpu.get("online_cpus") or len(cpu.get("cpu_usage", {}).get("percpu_usage") or []) or 1
        mem = st.get("memory_stats", {})
        mstats = mem.get("stats", {}) or {}
        used = max(0, mem.get("usage", 0) - mstats.get("inactive_file", mstats.get("total_inactive_file", 0)))
        return c, total, system, ncpu, used

    def sweep(self):
        conts = _docker_get("/containers/json") or []
        with ThreadPoolExecutor(max_workers=6) as ex:
            results = list(ex.map(self.stat, conts))
        rows, prev = [], {}
        for c, total, system, ncpu, used in results:
            pct = 0.0
            if c["Id"] in self.prev:
                t0, s0 = self.prev[c["Id"]]
                if system > s0 and total >= t0:
                    pct = (total - t0) / (system - s0) * ncpu * 100
            prev[c["Id"]] = (total, system)
            rows.append((_short_name(c), pct, used))
        rows.sort(key=lambda r: (r[1], r[2]), reverse=True)
        projects = {(c.get("Labels") or {}).get("com.docker.compose.project") or _short_name(c) for c in conts}
        with self.lock:
            self.ready = bool(self.prev)   # CPU% needs two sweeps
            self.prev = prev
            self.top = rows[:3]
            self.running = len(conts)
            self.apps = len(projects)

    def loop(self):
        first = True
        while True:
            try:
                self.sweep()
            except Exception:
                pass
            time.sleep(3 if first else SWEEP)
            first = False


_sampler = None


_f11 = None


def _fit(text, width=62):
    global _f11
    if _f11 is None:
        _f11 = ImageFont.truetype(font, 11)
    if _f11.getlength(text) <= width:
        return text
    while text and _f11.getlength(text + ".") > width:
        text = text[:-1]
    return text + "."


def _mem(b):
    if b >= 1024 ** 3:
        return f"{b / 1024 ** 3:.1f}G"
    return f"{b / 1024 ** 2:.0f}M"


class PageContainers(OLEDPage):
    def main(self, oled, data, config):
        global _sampler
        if _sampler is None:
            _sampler = _Sampler()
        with _sampler.lock:
            top, running, apps, ready = list(_sampler.top), _sampler.running, _sampler.apps, _sampler.ready
        oled.clear()
        oled.draw_text(f"{running} running", 0, 0, size=12, font_path=font)
        oled.draw_text(f"{apps} apps", 127, 0, size=12, font_path=font, align="right")
        oled.draw_bar_graph_horizontal(0, 0, 16, 127, 0)
        if not top:
            oled.draw_text("reading stats..." if running == 0 else "no data", 0, 26, size=12, font_path=font)
            oled.display()
            return
        for i, (name, cpu, mem) in enumerate(top):
            y = 19 + i * 15
            oled.draw_text(_fit(name), 0, y, size=11, font_path=font)
            oled.draw_text(f"{cpu:.0f}%" if ready else "..", 92, y, size=11, font_path=font, align="right")
            oled.draw_text(_mem(mem), 127, y, size=11, font_path=font, align="right")
        oled.display()
