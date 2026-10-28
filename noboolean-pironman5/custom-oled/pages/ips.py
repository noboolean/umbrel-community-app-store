# noboolean: network info page - hostname, LAN/WiFi/Tailscale IPs, gateway, link state
import os
import socket
import time

from pm_auto.libs.utils import get_font
from pm_auto.libs.oled_page import OLEDPage

font = get_font('UbuntuSans-Regular.ttf')
SKIP = ('docker', 'br-', 'veth', 'umbrel-vm', 'lo', 'virbr', 'vnet')
LABELS = (('tailscale', 'TS'), ('wlan', 'WiFi'), ('wl', 'WiFi'), ('eth', 'LAN'), ('en', 'LAN'))


def _read(path, default=''):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return default


def _hostname():
    return _read('/host/etc/hostname') or socket.gethostname()


def _gateway():
    try:
        with open('/proc/net/route') as f:
            for line in f.readlines()[1:]:
                p = line.split()
                if p[1] == '00000000' and int(p[3], 16) & 2:
                    return p[0], socket.inet_ntoa(bytes.fromhex(p[2])[::-1])
    except Exception:
        pass
    return None, None


def _link(iface):
    state = _read(f'/sys/class/net/{iface}/operstate', '?')
    speed = _read(f'/sys/class/net/{iface}/speed')
    if speed and speed.lstrip('-').isdigit() and int(speed) > 0:
        s = int(speed)
        return f"{state} {s // 1000}G" if s >= 1000 and s % 1000 == 0 else f"{state} {s}M"
    return state


def _label(iface):
    for pre, lab in LABELS:
        if iface.startswith(pre):
            return lab
    return iface[:5]


class PageIPs(OLEDPage):
    def __init__(self):
        super().__init__()
        self.offset = 0
        self.cycle_time_start = 0

    def lines(self, data):
        ips = data.get('ips', {}) if isinstance(data.get('ips'), dict) else {}
        ips = {k: v for k, v in ips.items() if not k.startswith(SKIP)}
        gw_if, gw = _gateway()
        out = []
        for iface, ip in ips.items():
            out.append((_label(iface), ip))
        if gw:
            out.append(('GW', gw))
        if gw_if:
            out.append(('Link', f"{gw_if} {_link(gw_if)}"))
        return out

    def main(self, oled, data, config):
        scroll_interval = config.get('scroll_interval', 3)
        oled.clear()
        oled.draw_text(_hostname()[:16], 0, 0, size=13, font_path=font)
        oled.draw_bar_graph_horizontal(0, 0, 16, 127, 0)  # divider line
        rows = self.lines(data)
        if not rows:
            oled.draw_text('DISCONNECTED', 0, 24, size=13, font_path=font)
            oled.display()
            return
        per = 3
        if len(rows) > per and time.time() - self.cycle_time_start >= scroll_interval:
            self.cycle_time_start = time.time()
            self.offset = self.offset + per if self.offset + per < len(rows) else 0
        if self.offset >= len(rows):
            self.offset = 0
        for i, (k, v) in enumerate(rows[self.offset:self.offset + per]):
            y = 19 + i * 15
            oled.draw_text(k, 0, y, size=11, font_path=font)
            oled.draw_text(v, 127, y, size=12, font_path=font, align='right')
        oled.display()
