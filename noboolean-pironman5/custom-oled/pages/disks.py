# noboolean: detailed disk page - one disk per screen (cycles if several):
# name + temp, usage bar + %, used/free/total, read/write throughput, mount point
import glob
import os
import time

from pm_auto.libs.oled_page import OLEDPage
from pm_auto.libs.utils import get_icon, format_bytes, get_font

error_icon = get_icon('icon_error_20.png')
font = get_font('UbuntuSans-Regular.ttf')


def _g(info, *names, default=0):
    for n in names:
        v = getattr(info, n, None)
        if v is not None:
            return v
    return default


def _short(size):
    v, u = format_bytes(size)
    try:
        v = float(v)
        v = f"{v:.0f}" if v >= 100 else f"{v:.1f}".rstrip('0').rstrip('.')
    except Exception:
        pass
    return f"{v}{u[:1]}"


def _rate(bps):
    if bps < 1024:
        return f"{bps:.0f}B"
    if bps < 1024 ** 2:
        return f"{bps / 1024:.0f}K"
    return f"{bps / 1024 ** 2:.1f}M"


def _nvme_temp(dev):
    for p in glob.glob(f'/sys/block/{dev}/device/hwmon*/temp1_input') + glob.glob(f'/sys/block/{dev}/device/device/hwmon/hwmon*/temp1_input'):
        try:
            return int(open(p).read()) / 1000
        except Exception:
            pass
    return None


def _mount(dev):
    best = None
    try:
        with open('/proc/1/mounts') as f:
            for line in f:
                src, mnt = line.split()[:2]
                if src.startswith(f'/dev/{dev}') and not mnt.startswith(('/run', '/mnt', '/var/lib')):
                    if best is None or len(mnt) < len(best):
                        best = mnt
    except Exception:
        pass
    return best or '-'


class PageDisks(OLEDPage):
    def __init__(self):
        super().__init__()
        self.disk_index = 0
        self.disk_num = 0
        self.cycle_time_start = 0
        self.prev = {}   # dev -> (t, read_sectors, write_sectors)
        self.rates = {}

    def io(self, dev):
        try:
            with open('/proc/diskstats') as f:
                for line in f:
                    p = line.split()
                    if p[2] == dev:
                        now, r, w = time.time(), int(p[5]), int(p[9])
                        if dev in self.prev:
                            t0, r0, w0 = self.prev[dev]
                            dt = now - t0
                            if dt >= 0.5:
                                self.rates[dev] = ((r - r0) * 512 / dt, (w - w0) * 512 / dt)
                                self.prev[dev] = (now, r, w)
                        else:
                            self.prev[dev] = (now, r, w)
                        break
        except Exception:
            pass
        return self.rates.get(dev, (0, 0))

    def main(self, oled, data, config):
        scroll_interval = config.get('scroll_interval', 3)
        disks_info = data.get('disks', {}) or {}
        oled.clear()
        if len(disks_info) == 0:
            oled.draw_icon(error_icon, 53, 0, scale=1, invert=False, dither=False, threshold=50)
            oled.draw_text('Disk Detection Error', 0, 22, size=14, font_path=font)
            oled.display()
            return
        if self.disk_num != len(disks_info):
            self.disk_index = 0
            self.cycle_time_start = time.time()
            self.disk_num = len(disks_info)
        if len(disks_info) > 1 and time.time() - self.cycle_time_start >= scroll_interval * 2:
            self.cycle_time_start = time.time()
            self.disk_index = (self.disk_index + 1) % len(disks_info)
        name, info = list(disks_info.items())[self.disk_index]
        dev = os.path.basename(str(name))
        total = _g(info, 'total', '_total')
        used = _g(info, 'used', '_used')
        free = _g(info, 'free', '_free', default=max(0, total - used))
        pct = float(_g(info, 'percent', '_percent'))
        temp = _g(info, 'temperature', '_temperature', default=None)
        if not temp and dev.startswith('nvme'):
            temp = _nvme_temp(dev)
        rd, wr = self.io(dev)

        oled.draw_text(dev[:12], 0, 0, size=12, font_path=font)
        if temp:
            unit = config.get('temperature_unit', 'C')
            t = temp if unit == 'C' else temp * 9 / 5 + 32
            oled.draw_text(f"{int(t)}°{unit}", 127, 0, size=12, font_path=font, align='right')
        oled.draw_bar_graph_horizontal(pct, 0, 16, 96, 6)
        oled.draw_text(f"{pct:.0f}%", 127, 13, size=11, font_path=font, align='right')
        oled.draw_text(f"U {_short(used)}  F {_short(free)}  T {_short(total)}", 0, 25, size=11, font_path=font)
        oled.draw_text(f"R {_rate(rd)}/s  W {_rate(wr)}/s", 0, 38, size=11, font_path=font)
        oled.draw_text(f"on {_mount(dev)}"[:22], 0, 51, size=11, font_path=font)
        oled.display()
