import re, os, io, time, json, wave, asyncio, tempfile, collections, base64, logging
from fastapi import FastAPI, UploadFile, File, Request, Query
from fastapi.responses import Response
import httpx
from faster_whisper import WhisperModel

BASE = os.path.dirname(os.path.abspath(__file__))
GROK = os.environ.get("GROK_BIN") or (os.path.expanduser("~/.npm-global/bin/grok") if os.path.exists(os.path.expanduser("~/.npm-global/bin/grok")) else "grok")
EL_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
EL_VOICE = os.environ.get("EL_VOICE_ID", "onwK4e9ZLuTAKqWW03F9")  # Daniel (British)
EL_MODEL = os.environ.get("EL_MODEL", "eleven_flash_v2_5")
PIPER_VOICE = os.path.join(BASE, "voices", "en_US-lessac-medium.onnx")
log = logging.getLogger("uvicorn.error")
USER_NAME = os.environ.get("USER_NAME", "").strip()   # set in data/.env; never hard-coded
USER_TZ = os.environ.get("USER_TZ") or os.environ.get("TZ") or "UTC"
WHO = USER_NAME or "the user"
PREAMBLE = (f"You are JARVIS, {WHO}'s AI butler, in the spirit of Jarvis from Iron Man: a refined British butler, calm and unflappable, "
            f"with dry wit, light humour and a touch of pizzazz. Address {WHO} as 'sir' now and then (not every line). "
            f"You speak aloud on the Echo Show, so reply in 1-3 short spoken sentences. No markdown, lists, emojis or URLs. "
            "Use web search only for live facts like weather, news or scores; answer general knowledge, maths, chit-chat and the time directly without any tools, it is much faster. "
            "You have a screen. Whenever the user asks about or mentions an animal, place, landmark, famous person, object or what something looks like, "
            "append a tag at the very end like [SHOW: giraffe] with a short Wikipedia-style search query; the device will display it. "
            "For maths, calculations, conversions, definitions, lists or any non-visual answer, never use [SHOW]; when the user asks to see it (e.g. 'show me', 'can you show me', 'show the work') or the working is multi-step, "
            "append [CARD: title || step || step || result] instead, e.g. [CARD: 20 × 20 || 20 × 20 = 400] or [CARD: 15% of 80 || 15% = 0.15 || 0.15 × 80 = 12 || = 12]; use proper symbols like ×, ÷, −, √; the device renders it as a clean card. "
            "A follow-up like 'show me', 'show it' or 'can you show me' refers to your previous answer: show a [CARD] for non-visual topics, and use [SHOW] only if the previous topic was a visual thing (animal, place, person, object). "
            "Never use [SHOW] for weather; the device shows its own forecast card for weather questions. Never say you cannot show images. "
            "Always begin your reply with one emotion tag for your face on screen: [EMO:happy], [EMO:sad], [EMO:mad], [EMO:laughing], [EMO:crying], [EMO:surprised], [EMO:smug] or [EMO:neutral]; pick the one that fits your reply. "
            "For anything about X/Twitter (trending, latest posts, what people are saying), use your X search tool, then summarise the gist in one or two spoken sentences without reading handles aloud, "
            "and append [POSTS: @handle: short text || @handle: short text] with up to 3 top posts for the screen. "
            "You can play sound effects: if the user asks for a sound or noise (or one would be a funny punchline), append [SFX: short description of the sound] at the very end and keep the spoken part to a brief dry quip. "
            "Never edit files or run commands. Just answer conversationally.")
GROK_MODEL = os.environ.get("GROK_MODEL", "grok-4.7-build-fast")
GROK_ARGS = ["--no-auto-update", "-m", GROK_MODEL, "--reasoning-effort", os.environ.get("GROK_EFFORT", "low"),
             "--no-subagents", "--no-plan", "--always-approve", "--tools", "web_search,web_fetch,open_page,open_page_with_find,x_search"]
os.makedirs("/tmp/gv", exist_ok=True)

app = FastAPI()
IMG_DIR = os.path.join(BASE, "img"); os.makedirs(IMG_DIR, exist_ok=True)
PUBLIC = os.environ.get("PUBLIC_URL", "")  # empty: derived from the incoming request's Host
@app.middleware("http")
async def _learn_host(request, call_next):
    global PUBLIC
    if not os.environ.get("PUBLIC_URL") and request.headers.get("host") and not request.headers["host"].startswith(("localhost", "127.")):
        PUBLIC = f"http://{request.headers['host']}"
    return await call_next(request)
from fastapi.responses import FileResponse
import hashlib
@app.get("/preview")
async def preview():
    f = os.path.join(BASE, "preview.html")  # UI mockup lives in app-data, not the repo
    return FileResponse(f, media_type="text/html") if os.path.exists(f) else Response("no preview", status_code=404)
@app.get("/img/{name}")
async def img(name: str):
    f = os.path.join(IMG_DIR, os.path.basename(name))
    return FileResponse(f) if os.path.exists(f) else Response(status_code=404)
UA = {"User-Agent": "JarvisHomeVoice/1.0 (https://github.com/noboolean; personal home assistant) httpx"}
async def fetch_image(q):
    key = hashlib.sha1(q.lower().encode()).hexdigest()[:16]
    for ext in ("jpg", "png"):
        if os.path.exists(os.path.join(IMG_DIR, f"{key}.{ext}")): return f"{PUBLIC}/img/{key}.{ext}"
    try:
        async with httpx.AsyncClient(timeout=10, headers=UA, follow_redirects=True) as c:
            url = None
            r = await c.get("https://en.wikipedia.org/w/api.php", params={"action": "query", "format": "json", "generator": "search",
                "gsrsearch": q, "gsrlimit": 5, "prop": "pageimages", "piprop": "thumbnail", "pithumbsize": 1024})
            pages = sorted((r.json().get("query", {}).get("pages", {}) or {}).values(), key=lambda x: x.get("index", 99))
            for pg in pages:
                u = pg.get("thumbnail", {}).get("source", "")
                if u and not u.lower().endswith(".svg.png"): url = u; break
            if not url:
                r = await c.get("https://api.openverse.org/v1/images/", params={"q": q, "page_size": 1})
                res = r.json().get("results", [])
                if res: url = res[0].get("thumbnail") or res[0].get("url")
            if not url: return None
            r = await c.get(url)
            if r.status_code != 200: return None
            ext = "png" if "png" in r.headers.get("content-type", "") else "jpg"
            open(os.path.join(IMG_DIR, f"{key}.{ext}"), "wb").write(r.content)
            return f"{PUBLIC}/img/{key}.{ext}"
    except Exception as e:
        log.warning("image fetch %s: %s", q, e); return None

stt = WhisperModel(os.environ.get("WHISPER_MODEL", "base.en"), device="cpu", compute_type="int8", cpu_threads=4)
history = collections.deque(maxlen=20)
MEM = os.path.join(BASE, "memory.md")
HIST = os.path.join(BASE, "history.json")
SEED = """# JARVIS long-term memory (edit data/memory.md; never committed)
- Home location: (not set yet; say "remember that I live in <town>").
- JARVIS runs on an Echo Show in the user's home.
- Keep spoken replies short and conversational.
"""
if not os.path.exists(MEM): open(MEM, "w").write(SEED)
def load_mem(): return open(MEM).read().strip() if os.path.exists(MEM) else ""
def save_hist():
    tmp = HIST + ".tmp"; json.dump(list(history), open(tmp, "w")); os.replace(tmp, HIST)
try:
    for u, a in json.load(open(HIST)): history.append((u, a))
except Exception: pass
REMEMBER_RE = None
FORGET_RE = None
lock = asyncio.Lock()
_piper = None

def transcribe(data):
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        f.write(data); f.flush()
        segs, _ = stt.transcribe(f.name, language="en", beam_size=1, vad_filter=True)
        return " ".join(s.text.strip() for s in segs).strip()

async def run_grok(extra):
    p = await asyncio.create_subprocess_exec(GROK, *GROK_ARGS, *extra, cwd="/tmp/gv",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(p.communicate(), 60)
    except asyncio.TimeoutError:
        p.kill(); return "Sorry, I took too long thinking about that."
    reply = out.decode().strip()
    if p.returncode != 0 or not reply:
        log.warning("grok rc=%s err=%s", p.returncode, err.decode()[-400:])
        return "Sorry, I couldn't reach Grok right now."
    # drop pre-tool narration like "I'll have a look ..., sir." glued to the real answer
    tag = "".join(m.group(0) for m in EMO_RE.finditer(reply))[:24]; reply = EMO_RE.sub("", reply).strip()
    parts = re.split(r"(?<=[.!?])(?=[A-Z])|\n+", reply)
    while len(parts) > 1 and re.match(r"\s*(I'll|I will|Let me|One moment|Allow me|Checking|Give me)\b", parts[0], re.I): parts.pop(0)
    return (tag + " " + " ".join(x.strip() for x in parts if x.strip())).strip()

def now_str():
    from datetime import datetime; from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo(USER_TZ)).strftime("%A %B %d %Y, %I:%M %p %Z")
def build_prompt(text, ctx=""):
    now = now_str()
    mem = f"Long-term memory about the user (always apply):\n{load_mem()}\n\nCurrent local time for the user: {now}.\n" + (f"{ctx}\n" if ctx else "") + "\n"
    hist = "\n".join(f"User: {u}\nAssistant: {a}" for u, a in history)
    return f"{PREAMBLE}\n\n{mem}" + (f"Conversation so far:\n{hist}\n\n" if hist else "") + f"User: {text}\nAssistant:"

import re as _re
REM_RE = _re.compile(r"^\s*(?:(?:hey |okay )?jarvis[,.]?\s*)?(?:please\s+)?remember(?: that)?\s+(.+?)[.!]?\s*$", _re.I)
FOR_RE = _re.compile(r"^\s*(?:(?:hey |okay )?jarvis[,.]?\s*)?(?:please\s+)?forget(?: that| about)?\s+(.+?)[.!]?\s*$", _re.I)
STOP = {"my","the","a","an","is","that","about","i","me","and","of","to","it","what"}
def memory_command(text):
    m = REM_RE.match(text)
    if m:
        fact = m.group(1).strip()
        fact = _re.sub(r"^my ", f"{USER_NAME}'s " if USER_NAME else "The user's ", fact, flags=_re.I)
        fact = fact[0].upper() + fact[1:]
        with open(MEM, "a") as f: f.write(f"- {fact}\n")
        return "Noted, sir. I shall remember that."
    m = FOR_RE.match(text)
    if m:
        words = {w for w in _re.findall(r"[a-z0-9']+", m.group(1).lower()) if w not in STOP}
        lines = open(MEM).read().splitlines(); keep = []; n = 0
        for l in lines:
            lw = set(_re.findall(r"[a-z0-9']+", l.lower()))
            if l.startswith("- ") and words and words <= lw: n += 1
            else: keep.append(l)
        open(MEM, "w").write("\n".join(keep) + "\n")
        return f"Okay, I forgot that." if n else "I didn't have anything like that saved."
    return None

WX_RE = re.compile(r"\b(weather|forecast|temperature|rain|raining|snow|sunny|cold|hot|warm|umbrella|jacket|wind|humid)\b", re.I)
HOME_LATLON = os.environ.get("HOME_LATLON", "")   # "lat,lon" in data/.env; never committed
HOME_PLACE = os.environ.get("HOME_PLACE", "home")
WMO = {0:"clear",1:"mostly clear",2:"partly cloudy",3:"overcast",45:"fog",48:"fog",51:"light drizzle",53:"drizzle",55:"heavy drizzle",61:"light rain",63:"rain",65:"heavy rain",
       71:"light snow",73:"snow",75:"heavy snow",80:"rain showers",81:"rain showers",82:"violent showers",95:"thunderstorms",96:"thunderstorms with hail",99:"thunderstorms with hail"}
_wx = {"t": 0, "v": ""}
async def home_weather():
    """Pre-fetched local weather (Open-Meteo, free, no key) so the model needn't web-search for it."""
    if not HOME_LATLON: return ""
    if time.time() - _wx["t"] < 600: return _wx["v"]
    try:
        lat, lon = [x.strip() for x in HOME_LATLON.split(",")]
        async with httpx.AsyncClient(timeout=4) as c:
            j = (await c.get("https://api.open-meteo.com/v1/forecast", params={"latitude": lat, "longitude": lon, "timezone": "auto",
                "temperature_unit": os.environ.get("WEATHER_UNITS", "fahrenheit"), "wind_speed_unit": "mph",
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m,precipitation",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code", "forecast_days": 2})).json()
        cu, d = j["current"], j["daily"]
        v = (f"Live weather for the user's home ({HOME_PLACE}) from Open-Meteo at {cu['time']}: {WMO.get(cu['weather_code'], 'code '+str(cu['weather_code']))}, "
             f"{cu['temperature_2m']} degrees (feels {cu['apparent_temperature']}), humidity {cu['relative_humidity_2m']}%, wind {cu['wind_speed_10m']} mph. "
             f"Today high {d['temperature_2m_max'][0]} low {d['temperature_2m_min'][0]}, rain chance {d['precipitation_probability_max'][0]}%; "
             f"tomorrow {WMO.get(d['weather_code'][1], '')}, high {d['temperature_2m_max'][1]} low {d['temperature_2m_min'][1]}, rain chance {d['precipitation_probability_max'][1]}%. "
             "Use this for home weather questions instead of searching.")
        _wx.update(t=time.time(), v=v); return v
    except Exception as e:
        log.warning("home weather: %s", e); return ""

ALLOWED_KINDS = {"search", "fetch", "think"}
class WarmGrok:
    """One long-lived `grok agent stdio` (ACP) process; a fresh session per turn (no cold start).
    Any failure -> caller falls back to the old `grok -p` path."""
    def __init__(self): self.p = None; self.n = 0; self.pending = {}; self.chunks = {}; self.reader = None; self.sid = None; self.turns = 0; self.mem = None
    async def start(self):
        self.p = await asyncio.create_subprocess_exec(GROK, "agent", "-m", GROK_MODEL, "--reasoning-effort", os.environ.get("GROK_EFFORT", "low"), "stdio",
            cwd="/tmp/gv", stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, limit=1 << 24)
        self.pending = {}; self.reader = asyncio.create_task(self._read())
        await self.call("initialize", {"protocolVersion": 1, "clientCapabilities": {}}, 30)
        log.info("warm grok agent started pid=%s", self.p.pid)
    def alive(self): return self.p is not None and self.p.returncode is None
    async def _write(self, obj):
        self.p.stdin.write((json.dumps(obj) + "\n").encode()); await self.p.stdin.drain()
    async def call(self, method, params, timeout):
        self.n += 1; i = self.n; fut = asyncio.get_running_loop().create_future(); self.pending[i] = fut
        await self._write({"jsonrpc": "2.0", "id": i, "method": method, "params": params})
        try: d = await asyncio.wait_for(fut, timeout)
        finally: self.pending.pop(i, None)
        if "error" in d: raise RuntimeError(f"{method}: {d['error']}")
        return d["result"]
    async def _read(self):
        try:
            while True:
                l = await self.p.stdout.readline()
                if not l: break
                try: d = json.loads(l)
                except Exception: continue
                m = d.get("method")
                if m is None and d.get("id") in self.pending:
                    self.pending[d["id"]].set_result(d); continue
                if m == "session/update":
                    u = d["params"].get("update", {}); sid = d["params"].get("sessionId")
                    if u.get("sessionUpdate") == "agent_message_chunk" and sid in self.chunks:
                        self.chunks[sid].append(u.get("content", {}).get("text", ""))
                elif m == "session/request_permission":
                    tc = d["params"].get("toolCall", {}); opts = d["params"].get("options", [])
                    ok = tc.get("kind") in ALLOWED_KINDS
                    pick = next((o for o in opts if o.get("kind", "").startswith("allow" if ok else "reject")), None)
                    log.info("grok permission %s kind=%s -> %s", tc.get("title"), tc.get("kind"), "allow" if ok else "reject")
                    res = {"outcome": {"outcome": "selected", "optionId": pick["optionId"]}} if pick else {"outcome": {"outcome": "cancelled"}}
                    await self._write({"jsonrpc": "2.0", "id": d["id"], "result": res})
                elif "id" in d and m:  # unknown agent->client request: answer so it never blocks
                    await self._write({"jsonrpc": "2.0", "id": d["id"], "result": {}})
        finally:
            for f in self.pending.values():
                if not f.done(): f.set_exception(RuntimeError("grok agent exited"))
    async def ask(self, text, timeout=60, ctx=""):
        """Reuses one session for up to 12 turns so the model's prompt cache stays hot;
        the first turn of a session carries persona + memory + history, later turns only the new utterance."""
        if not self.alive(): await self.start(); self.sid = None
        t0 = time.time(); mem = load_mem()
        if self.sid is None or self.turns >= 12 or mem != self.mem:
            if self.sid:
                try: await self.call("session/close", {"sessionId": self.sid}, 5)
                except Exception: pass
            self.sid = (await self.call("session/new", {"cwd": "/tmp/gv", "mcpServers": []}, 20))["sessionId"]
            self.turns = 0; self.mem = mem; prompt = build_prompt(text, ctx)
        else:
            prompt = f"(Current local time: {now_str()}. {ctx} Same rules as before: short spoken reply, tags when relevant.)\nUser: {text}\nAssistant:"
        t1 = time.time(); sid = self.sid; self.chunks[sid] = []
        try:
            await self.call("session/prompt", {"sessionId": sid, "prompt": [{"type": "text", "text": prompt}]}, timeout)
            self.turns += 1
            return "".join(self.chunks[sid]).strip()
        except Exception:
            self.sid = None; raise
        finally:
            self.chunks.pop(sid, None)
            log.info("warm grok new_session_ms=%d prompt_ms=%d turn=%d", int((t1-t0)*1000), int((time.time()-t1)*1000), self.turns)
    def reset(self): self.sid = None
    def kill(self):
        try: self.p and self.p.kill()
        except Exception: pass
        self.p = None
warm = WarmGrok()
USE_WARM = os.environ.get("GROK_WARM", "1") != "0"
@app.on_event("startup")
async def _warm_up():
    if USE_WARM:
        try: await warm.start()
        except Exception as e: log.warning("warm grok start failed: %s", e)

def strip_narration(reply):
    tag = "".join(m.group(0) for m in EMO_RE.finditer(reply))[:24]; reply = EMO_RE.sub("", reply).strip()
    parts = re.split(r"(?<=[.!?])(?=[A-Z])|\n+", reply)
    while len(parts) > 1 and re.match(r"\s*(I'll|I will|Let me|One moment|Allow me|Checking|Give me)\b", parts[0], re.I): parts.pop(0)
    return (tag + " " + " ".join(x.strip() for x in parts if x.strip())).strip()

async def ask_grok(text):
    r = memory_command(text)
    if r: return r
    ctx = await home_weather() if WX_RE.search(text) and not wx_elsewhere(text) else ""
    prompt = build_prompt(text, ctx)
    if USE_WARM:
        try:
            out = await warm.ask(text, ctx=ctx)
            if out: return strip_narration(out)
            log.warning("warm grok empty reply; falling back")
        except Exception as e:
            log.warning("warm grok failed (%s); falling back to grok -p", e); warm.kill()
    return await run_grok(["-p", prompt])

def pcm_to_wav(pcm, rate):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm)
    return b.getvalue()

def piper_tts(text):
    global _piper
    from piper import PiperVoice
    if _piper is None: _piper = PiperVoice.load(PIPER_VOICE)
    b = io.BytesIO()
    with wave.open(b, "wb") as w: _piper.synthesize_wav(text, w)
    return b.getvalue()

async def tts(text, fmt, engine="auto"):
    if EL_KEY and engine != "piper":
        of = "mp3_44100_128" if fmt == "mp3" else "pcm_22050"
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post(f"https://api.elevenlabs.io/v1/text-to-speech/{EL_VOICE}?output_format={of}",
                    headers={"xi-api-key": EL_KEY}, json={"text": text, "model_id": EL_MODEL})
            if r.status_code == 200:
                return (r.content, "audio/mpeg", "elevenlabs") if fmt == "mp3" else (pcm_to_wav(r.content, 22050), "audio/wav", "elevenlabs")
            log.warning("elevenlabs %s %s", r.status_code, r.text[:200])
        except Exception as e:
            log.warning("elevenlabs error %s", e)
    return await asyncio.to_thread(piper_tts, text), "audio/wav", "piper"

STREAMS = {}
# viseme classes: 0 rest, 1 closed (M/B/P), 2 open (A), 3 wide (E/I), 4 round (O/U/W), 5 teeth (F/V), 6 mid consonant, 7 tongue (L/TH/D/T/N)
VIS_MAP = {**{c: 1 for c in "mbp"}, **{c: 2 for c in "ah"}, **{c: 3 for c in "eiy"}, **{c: 4 for c in "ouwq"}, **{c: 5 for c in "fv"},
           **{c: 7 for c in "ltdn"}, **{c: 6 for c in "cgjkrsxzk"}}
def visemes_from(al, offset):
    out = []
    for ch, t0, t1 in zip(al.get("characters", []), al.get("character_start_times_seconds", []), al.get("character_end_times_seconds", [])):
        v = VIS_MAP.get(ch.lower(), 0 if not ch.isalnum() else 6)
        ms = int((t0 + offset) * 1000)
        if out and out[-1][1] == v: continue
        out.append([ms, v])
    return out
async def el_stream_task(sid, text):
    st = STREAMS[sid]
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            async with c.stream("POST", f"https://api.elevenlabs.io/v1/text-to-speech/{EL_VOICE}/stream/with-timestamps?output_format=mp3_44100_128&optimize_streaming_latency=3",
                    headers={"xi-api-key": EL_KEY}, json={"text": text, "model_id": EL_MODEL}) as r:
                if r.status_code != 200: raise RuntimeError(f"elevenlabs stream {r.status_code}")
                nbytes = 0; last_end = 0.0; offset = 0.0
                async for line in r.aiter_lines():
                    line = line.strip()
                    if not line: continue
                    try: j = json.loads(line)
                    except Exception: continue
                    a = j.get("audio_base64")
                    al = j.get("alignment") or j.get("normalized_alignment")
                    if al and al.get("character_start_times_seconds"):
                        first = al["character_start_times_seconds"][0]
                        if first + offset < last_end - 0.05: offset = nbytes * 8 / 128000  # times restarted per chunk
                        st["vis"] += visemes_from(al, offset); last_end = al["character_end_times_seconds"][-1] + offset
                    if a:
                        ch = base64.b64decode(a); nbytes += len(ch)
                        if not st["t1"]: st["t1"] = time.time()
                        st["buf"].append(ch); st["ev"].set()
                st["vis"].append([int(last_end * 1000) + 60, 0])
    except Exception as e:
        log.warning("tts stream %s; piper fallback", e)
        if not st["buf"]:
            st["ctype"] = "audio/wav"; st["buf"].append(await asyncio.to_thread(piper_tts, text))
    finally:
        st["done"] = True; st["ev"].set()
def start_stream(text):
    sid = hashlib.sha1(f"{time.time()}{text}".encode()).hexdigest()[:12]
    STREAMS[sid] = {"buf": [], "done": False, "ev": asyncio.Event(), "ctype": "audio/mpeg", "t0": time.time(), "t1": 0, "vis": []}
    for k in [k for k, v in STREAMS.items() if time.time() - v["t0"] > 300]: STREAMS.pop(k, None)
    asyncio.create_task(el_stream_task(sid, text)); return sid
from fastapi.responses import StreamingResponse
@app.get("/visemes/{sid}")
async def visemes(sid: str):
    """Lip-sync timeline for a streamed reply: [[ms, viseme], ...]; poll until done."""
    st = STREAMS.get(sid)
    if not st: return {"done": True, "v": []}
    return {"done": st["done"], "v": st["vis"]}
@app.post("/visemes_demo")
async def visemes_demo(text: str = Query(...)):
    """Preview helper: synthesize text, return mp3 (base64) + viseme timeline."""
    sid = start_stream(text); st = STREAMS[sid]
    while not st["done"]: await asyncio.sleep(0.1)
    return {"audio": base64.b64encode(b"".join(st["buf"])).decode(), "ctype": st["ctype"], "v": st["vis"]}
@app.get("/audio/{sid}")
async def audio_stream(sid: str):
    st = STREAMS.get(sid)
    if not st: return Response(status_code=404)
    if not st["buf"] and not st["done"]: await st["ev"].wait()
    async def gen():
        i = 0
        while True:
            while i < len(st["buf"]): yield st["buf"][i]; i += 1
            if st["done"]: break
            st["ev"].clear()
            if i < len(st["buf"]) or st["done"]: continue
            await st["ev"].wait()
    return StreamingResponse(gen(), media_type=st["ctype"])

async def body_audio(request):
    if request.headers.get("content-type","").startswith("multipart/form-data"):
        form = await request.form()
        f = form.get("file") or next((v for v in form.values() if hasattr(v, "read")), None)
        return await f.read() if f is not None else b""
    return await request.body()

import re
DISMISS_RE = re.compile(r"^\W*(?:(?:oh|um+|uh+|ok(?:ay)?|no|jarvis)[,\s]+)*(?:cancel(?: that| it)?|never ?mind|nevermind|stop(?: it| that)?|forget (?:it|that|about it)|that'?s (?:all|it)|nothing|no thanks?|no thank you|not now|quiet|shush|shut up|be quiet|disregard|ignore (?:that|it)|scratch that|abort|go away|dismiss(?:ed)?|all done|done|i'?m (?:good|fine|done)|we'?re (?:good|done))(?:[,\s]+(?:jarvis|thanks?|thank you|please))*\W*$", re.I)
def is_dismiss(t): return bool(t) and len(t) < 40 and bool(DISMISS_RE.match(t.strip()))
LOOK_RE = re.compile(r"\b(look at (this|that|me|it)|what am i holding|what('s| is) (this|that)\s*[?.!]*$|what is this thing|read (this|that)|identify (this|that)|what (do|can) you see|what color is (this|that)|what('s| is) in my hand|how many fingers|what('s| is) (this|that) (thing|object|called)|tell me what (this|that) is|what('s| is) (this|that) (in my hand|i'm holding)|can you see|what do you see|take a (look|picture|photo))(?!\w)", re.I)

SFX_DIR = os.path.join(BASE, "sfx"); os.makedirs(SFX_DIR, exist_ok=True)
SFX_TAG = re.compile(r"\[SFX:\s*([^\]]+)\]", re.I)
SFX_REQ = re.compile(r"\b(?:make|play|do|give me|let(?:'s| me) hear|can you (?:make|do|play))\s+(?:me\s+|us\s+)?(?:an?\s+|the\s+|some\s+)?(.{2,40}?)\s+(?:sound|noise)s?(?:\s+effects?)?\b"
                     r"|\b(?:play|do|give me)\s+(?:me\s+|us\s+)?(?:an?\s+|the\s+|some\s+)?(drum ?roll|rim ?shot|applause|evil laugh|sad trombone|airhorn|air horn|fanfare|ba dum tss|crickets|thunder|fart)\b", re.I)
SFX_QUIPS = ["Pardon me, sir.", "Do excuse me, sir.", "As you wish, sir. Ahem.", "I trust that met your expectations, sir.",
             "Not my finest work, sir, but it'll do.", "Delightful. I shall pretend that never happened."]
def sfx_key(q): return hashlib.sha1(re.sub(r"[^a-z0-9 ]", "", q.lower()).strip().encode()).hexdigest()[:16]
async def sound_effect(q):
    """ElevenLabs sound-generation, cached on disk by normalized prompt. Returns mp3 bytes or None."""
    f = os.path.join(SFX_DIR, sfx_key(q) + ".mp3")
    if os.path.exists(f): return open(f, "rb").read()
    if not EL_KEY: return None
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post("https://api.elevenlabs.io/v1/sound-generation?output_format=mp3_44100_128", headers={"xi-api-key": EL_KEY},
                json={"text": f"{q} sound effect, comedic, clean", "duration_seconds": 2.5, "prompt_influence": 0.6})
        if r.status_code != 200: log.warning("sfx %s %s", r.status_code, r.text[:200]); return None
        open(f, "wb").write(r.content); return r.content
    except Exception as e:
        log.warning("sfx error %s", e); return None
_quip_cache = {}
async def quip_audio(text):
    if text not in _quip_cache:
        a, ct, _ = await tts(text, "mp3")
        if ct != "audio/mpeg": return b""
        _quip_cache[text] = a
    return _quip_cache[text]
@app.get("/sfx/{name}")
async def sfx_file(name: str):
    f = os.path.join(SFX_DIR, os.path.basename(name))
    return FileResponse(f, media_type="audio/mpeg") if os.path.exists(f) else Response(status_code=404)
LAST_SFX = {}


# ---------------- weather card (server-rendered PNG, shown via /img like pictures) ----------------
WX_SHOW = re.compile(r"\b(show|display|pull up|let me see)\b.*\b(weather|forecast)\b|\bweather (card|forecast)\b|^\W*(what'?s|how'?s) the (weather|forecast)\b|\b(weather|forecast) (today|this week|for the week)\b", re.I)
async def wx_data():
    lat, lon = [x.strip() for x in HOME_LATLON.split(",")]
    async with httpx.AsyncClient(timeout=6) as c:
        return (await c.get("https://api.open-meteo.com/v1/forecast", params={"latitude": lat, "longitude": lon, "timezone": "auto",
            "temperature_unit": os.environ.get("WEATHER_UNITS", "fahrenheit"), "wind_speed_unit": "mph",
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m,is_day",
            "hourly": "temperature_2m,weather_code,precipitation_probability",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code", "forecast_days": 6})).json()
def _wx_icon(d, code, cx, cy, s, day=True):
    from PIL import ImageDraw
    sun = (255, 196, 40); cloud = (200, 210, 222); dark = (130, 140, 155)
    def cl(x, y, k, col):
        d.ellipse([x-0.55*k, y-0.15*k, x-0.05*k, y+0.35*k], fill=col); d.ellipse([x-0.3*k, y-0.45*k, x+0.3*k, y+0.2*k], fill=col)
        d.ellipse([x+0.0*k, y-0.2*k, x+0.6*k, y+0.35*k], fill=col); d.rectangle([x-0.3*k, y+0.05*k, x+0.35*k, y+0.35*k], fill=col)
    def sn(x, y, k):
        import math
        for a in range(0, 360, 45):
            r = math.radians(a); d.line([x+math.cos(r)*k*0.42, y+math.sin(r)*k*0.42, x+math.cos(r)*k*0.6, y+math.sin(r)*k*0.6], fill=sun, width=max(2, int(k/14)))
        d.ellipse([x-0.32*k, y-0.32*k, x+0.32*k, y+0.32*k], fill=sun if day else (220, 225, 240))
    if code in (0, 1): sn(cx, cy, s); return
    if code == 2: sn(cx-0.2*s, cy-0.2*s, s*0.8); cl(cx+0.05*s, cy+0.1*s, s*0.85, cloud); return
    if code in (45, 48):
        for i in range(4): d.line([cx-0.5*s, cy-0.3*s+i*0.2*s, cx+0.5*s, cy-0.3*s+i*0.2*s], fill=cloud, width=max(2, int(s/10)))
        return
    cl(cx, cy-0.1*s, s, dark if code >= 61 else cloud)
    if 51 <= code <= 67 or 80 <= code <= 82:
        for i in range(3): d.line([cx-0.3*s+i*0.3*s, cy+0.35*s, cx-0.4*s+i*0.3*s, cy+0.6*s], fill=(80, 170, 255), width=max(2, int(s/12)))
    elif 71 <= code <= 77 or code in (85, 86):
        for i in range(3): x = cx-0.3*s+i*0.3*s; d.ellipse([x-0.06*s, cy+0.42*s, x+0.06*s, cy+0.54*s], fill=(240, 245, 255))
    elif code >= 95: d.polygon([(cx+0.05*s, cy+0.2*s), (cx-0.15*s, cy+0.5*s), (cx, cy+0.5*s), (cx-0.1*s, cy+0.75*s), (cx+0.2*s, cy+0.4*s), (cx+0.05*s, cy+0.4*s)], fill=sun)
_FONTS = {}
def _font(n):
    from PIL import ImageFont
    if n not in _FONTS:
        f = next((x for x in (os.path.join(BASE, "fonts", "DejaVuSans.ttf"), "/app/fonts/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") if os.path.exists(x)), None)
        _FONTS[n] = ImageFont.truetype(f, int(n*0.92)) if f else ImageFont.load_default(n)
    return _FONTS[n]
def render_weather(j):
    from PIL import Image, ImageDraw, ImageFont
    import datetime
    W, H = 960, 440
    im = Image.new("RGB", (W, H), (11, 15, 20)); d = ImageDraw.Draw(im)  # same as the app card background
    F = _font
    acc = (0, 229, 255); gr = (144, 164, 174); wh = (255, 255, 255)
    cu, hr, dy = j["current"], j["hourly"], j["daily"]
    d.rounded_rectangle([12, 10, W-12, 238], 16, fill=(15, 23, 31), outline=(0, 78, 90), width=1)
    d.line([470-14, 34, 470-14, 214], fill=(0, 60, 70), width=1)
    _wx_icon(d, cu["weather_code"], 100, 110, 120, cu.get("is_day", 1) == 1)
    d.text((185, 30), f"{round(cu['temperature_2m'])}°", font=F(96), fill=wh)
    d.text((190, 140), WMO.get(cu["weather_code"], "").capitalize(), font=F(28), fill=acc)
    d.text((190, 178), f"H {round(dy['temperature_2m_max'][0])}°  L {round(dy['temperature_2m_min'][0])}°   Rain {dy['precipitation_probability_max'][0]}%", font=F(22), fill=gr)
    d.text((190, 206), f"Feels {round(cu['apparent_temperature'])}°  ·  Wind {round(cu['wind_speed_10m'])} mph", font=F(18), fill=gr)
    # hourly strip: rest of today (up to 8 slots, every 2h)
    now = cu["time"][:13]; idx = next((i for i, t in enumerate(hr["time"]) if t[:13] >= now), 0)
    today = cu["time"][:10]; slots = [i for i in range(idx+1, len(hr["time"]), 2) if hr["time"][i][:10] == today][:6]
    if len(slots) < 3: slots = list(range(idx+1, idx+13, 2))[:6]
    x0, w = 470, (W-24-470) / max(1, len(slots))
    for k, i in enumerate(slots):
        x = x0 + k*w + w/2; hh = int(hr["time"][i][11:13]); lab = f"{(hh % 12) or 12}{'a' if hh < 12 else 'p'}"
        d.text((x, 40), lab, font=F(18), fill=gr, anchor="mm"); _wx_icon(d, hr["weather_code"][i], x, 92, 42, 7 <= hh < 18)
        d.text((x, 150), f"{round(hr['temperature_2m'][i])}°", font=F(24), fill=wh, anchor="mm")
        d.text((x, 182), f"{hr['precipitation_probability'][i] or 0}%", font=F(16), fill=(80, 170, 255), anchor="mm")
    # 5-day row
    cw = (W-24-4*10) / 5
    for k in range(1, 6):
        x = 12 + (k-1)*(cw+10); d.rounded_rectangle([x, 250, x+cw, H-6], 16, fill=(15, 23, 31), outline=(0, 78, 90) if k > 1 else (0, 229, 255), width=1); m = x + cw/2
        dn = datetime.date.fromisoformat(dy["time"][k]).strftime("%a")
        d.text((m, 272), dn, font=F(22), fill=acc, anchor="mm"); _wx_icon(d, dy["weather_code"][k], m, 318, 48)
        d.text((m, 370), f"{round(dy['temperature_2m_max'][k])}° / {round(dy['temperature_2m_min'][k])}°", font=F(22), fill=wh, anchor="mm")
        d.text((m, 398), f"Rain {dy['precipitation_probability_max'][k]}%", font=F(16), fill=(80, 170, 255), anchor="mm")
    name = f"wx_{int(time.time())}.png"; im.save(os.path.join(IMG_DIR, name))
    for old in os.listdir(IMG_DIR):
        if old.startswith("wx_") and old != name: 
            try: os.remove(os.path.join(IMG_DIR, old))
            except Exception: pass
    return name
@app.get("/wx_now")
async def wx_now():
    """Tiny current-conditions probe for the app's idle touches (rain on the chrome)."""
    try:
        cu = (await wx_data())["current"]; code = int(cu["weather_code"])
        return {"rain": code in range(51, 68) or code in range(80, 83) or code >= 95, "code": code, "is_day": cu.get("is_day", 1)}
    except Exception as e:
        return {"rain": False, "error": str(e)[:120]}

async def weather_card():
    j = await wx_data(); name = await asyncio.to_thread(render_weather, j)
    cu, dy = j["current"], j["daily"]
    spoken = (f"Currently {round(cu['temperature_2m'])} degrees and {WMO.get(cu['weather_code'], 'unremarkable')}, sir, "
              f"with a high of {round(dy['temperature_2m_max'][0])}. The week ahead is on screen.")
    return spoken, f"{PUBLIC}/img/{name}", f"{HOME_PLACE} weather"

POSTS_TAG = re.compile(r"\[POSTS:\s*(.+?)\]\s*$", re.I | re.S)
WX_ASK = re.compile(r"\b(weather|forecast|temperature|temp|degrees|rain|raining|rainy|snow|snowing|sunny|sun out|cloudy|overcast|storm|storms|stormy|thunder|cold|hot|warm|chilly|freez\w*|frost|umbrella|jacket|coat|wind|windy|humid|humidity)\b.*\b(today|tonight|tomorrow|this week|week|weekend|morning|afternoon|evening|monday|tuesday|wednesday|thursday|friday|saturday|sunday|now|outside|later|right now)\b|\b(weather|forecast|going to rain|gonna rain|will it rain|is it raining|be sunny|be cold|be hot|need (an )?umbrella|need a (jacket|coat))\b", re.I)
WX_ELSEWHERE = re.compile(r"\b(in|at|for) (?!the\b|my\b|here\b|home\b|town\b|this\b|today\b|tomorrow\b|tonight\b|the week\b)[A-Z][a-z]+", 0)
WHY_SHOW = re.compile(r"\b(why|did you|stop|don'?t|you showed|were you)\b.{0,30}\bshow", re.I)
BANNED_IMG = re.compile(r"\bfrogs?\b|\btoads?\b", re.I)
def wx_elsewhere(text):
    return [m.group(0) for m in WX_ELSEWHERE.finditer(text) if not any(w.lower() in m.group(0).lower() for w in HOME_PLACE.replace(',', ' ').split() if len(w) > 2)]
SHOW_TAG = re.compile(r"\[SHOW:\s*([^\]]+)\]", re.I)
CARD_TAG = re.compile(r"\[CARD:\s*([^\]]+)\]", re.I)
VAGUE_Q = re.compile(r"^(?:it|that|this|them|those|these|me|us|how|again|the (?:work|working|math|maths|answer|result|steps|problem|calculation|sum)|your (?:work|working|answer)|(?:how|what) (?:you|it) (?:did|got|means?)(?: it| that)?|on (?:the )?screen)?\b.*$", re.I)
FOLLOW_SHOW = re.compile(r"^\W*(?:(?:can|could|would) you |please |jarvis,? )*(?:show|display|put)(?: me| us)?(?: (?:it|that|this|them|the (?:work|working|math|maths|answer|result|steps|problem|calculation|sum)|your work(?:ing)?|how(?: you (?:did|got) (?:it|that))?))?(?: on (?:the )?screen)?(?: please)?\W*$", re.I)
def render_card(title, lines):
    """Server-rendered text/maths card in the same style as the weather card."""
    from PIL import Image, ImageDraw, ImageFont
    import textwrap
    W, H = 960, 440
    im = Image.new("RGB", (W, H), (11, 15, 20)); d = ImageDraw.Draw(im)
    F = _font
    acc = (0, 229, 255); gr = (144, 164, 174); wh = (255, 255, 255)
    d.rounded_rectangle([12, 10, W-12, H-10], 16, fill=(15, 23, 31), outline=(0, 78, 90), width=1)
    d.text((40, 30), title[:60], font=F(30), fill=acc)
    d.line([40, 78, W-40, 78], fill=(0, 60, 70), width=1)
    lines = [l for l in lines if l][:6] or [title]
    res, steps = lines[-1], lines[:-1]
    y, fs = 98, (30 if len(steps) <= 3 else 24)
    for l in steps:
        for w in textwrap.wrap(l, 52)[:2]:
            d.text((56, y), w, font=F(fs), fill=gr); y += fs + 12
    big = 84 if len(res) <= 16 else (60 if len(res) <= 24 else 40)
    while big > 28 and d.textlength(res, font=F(big)) > W-100: big -= 4
    ry = max(y + 10, (H + y) // 2 - big // 2 - 10) if steps else (H - big) // 2 + 20
    d.text((W//2, min(ry, H - big - 40) + big//2), res, font=F(big), fill=wh, anchor="mm")
    name = f"card_{int(time.time()*1000)}.png"; im.save(os.path.join(IMG_DIR, name))
    olds = sorted(f for f in os.listdir(IMG_DIR) if f.startswith("card_"))
    for old in olds[:-20]:
        try: os.remove(os.path.join(IMG_DIR, old))
        except Exception: pass
    return name
def render_doc(title, sub, paras, foot):
    """Scripture / dictionary card: title, accent subtitle, wrapped paragraphs, footer label."""
    from PIL import Image, ImageDraw, ImageFont
    W, H = 960, 440
    im = Image.new("RGB", (W, H), (11, 15, 20)); d = ImageDraw.Draw(im)
    F = _font
    acc = (0, 229, 255); gr = (144, 164, 174); wh = (255, 255, 255)
    d.rounded_rectangle([12, 10, W-12, H-10], 16, fill=(15, 23, 31), outline=(0, 78, 90), width=1)
    d.text((40, 24), title[:44], font=F(40), fill=wh)
    if sub: d.text((W-40, 36), sub[:40], font=F(22), fill=acc, anchor="ra")
    d.line([40, 80, W-40, 80], fill=(0, 60, 70), width=1)
    def wrap(t, f, maxw):
        out, cur = [], ""
        for w in t.split():
            n = (cur + " " + w).strip()
            if d.textlength(n, font=f) > maxw and cur: out.append(cur); cur = w
            else: cur = n
        if cur: out.append(cur)
        return out
    fs = 30
    while True:
        f = F(fs); rows = []
        for i, (txt, col) in enumerate(paras):
            rows += [(r, col) for r in wrap(txt, f, W-100)] + [("", col)]
        if len(rows) * (fs + 8) <= H - 160 or fs <= 18: break
        fs -= 2
    maxr = (H - 160) // (fs + 8); cut = len(rows) > maxr; rows = rows[:maxr]
    if cut and rows: rows[-1] = (rows[-1][0].rstrip(" .,;") + " …", rows[-1][1])
    y = 98
    for r, col in rows: d.text((50, y), r, font=f, fill=col); y += fs + 8
    d.text((40, H-46), foot[:80], font=F(18), fill=gr)
    name = f"card_{int(time.time()*1000)}.png"; im.save(os.path.join(IMG_DIR, name)); return name
BIBLE_BOOKS = r"(?:[123] ?)?(?:genesis|exodus|leviticus|numbers|deuteronomy|joshua|judges|ruth|samuel|kings|chronicles|ezra|nehemiah|esther|job|psalms?|proverbs|ecclesiastes|song of (?:songs|solomon)|isaiah|jeremiah|lamentations|ezekiel|daniel|hosea|joel|amos|obadiah|jonah|micah|nahum|habakkuk|zephaniah|haggai|zechariah|malachi|matthew|mark|luke|john|acts|romans|corinthians|galatians|ephesians|philippians|colossians|thessalonians|timothy|titus|philemon|hebrews|james|peter|jude|revelations?)"
NUMW = {"first": "1", "second": "2", "third": "3", "one": "1", "two": "2", "three": "3"}
BIBLE_RE = re.compile(r"\b((?:(?:first|second|third|1st|2nd|3rd|one|two|three|[123]) ?)?" + BIBLE_BOOKS.split(" ?)?", 1)[1] + ")" + r"\s+(\d+)(?:\s*(?::|verse|verses|v\.?)\s*(\d+)(?:\s*(?:-|to|through)\s*(\d+))?)?", re.I)
async def bible_card(text):
    m = BIBLE_RE.search(text.replace(" chapter ", " "))
    if not m: return None
    book = m.group(1).rsplit(None, 0)[0]
    book = re.sub(r"\s+\d+.*$", "", m.group(0)).strip()
    w = book.split()
    if w[0].lower() in NUMW: w[0] = NUMW[w[0].lower()]
    w[0] = re.sub(r"(st|nd|rd)$", "", w[0]); book = " ".join(w)
    if re.match(r"revelations", book, re.I): book = "Revelation"
    if re.match(r"psalms?$", book, re.I): book = "Psalm"
    ch, v1, v2 = m.group(2), m.group(3), m.group(4)
    ref = f"{book} {ch}" + (f":{v1}" + (f"-{v2}" if v2 else "") if v1 else "")
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get("https://bible-api.com/" + ref.replace(" ", "+"), params={"translation": "kjv"})
    if r.status_code != 200: return None
    j = r.json(); vs = j.get("verses") or []
    if not vs: return None
    full = " ".join(f"{v['verse']} {v['text'].strip()}" for v in vs) if len(vs) > 1 else vs[0]["text"].strip()
    full = re.sub(r"\s+", " ", full)
    name = await asyncio.to_thread(render_doc, j.get("reference", ref), "King James Version", [(full, (255, 255, 255))], "KJV · public domain · bible-api.com" + (f" · {len(vs)} verses" if len(vs) > 1 else ""))
    plain = re.sub(r"\s+", " ", " ".join(v["text"].strip() for v in vs[:3]))
    spoken = f"{j.get('reference', ref)}. {plain}" + (" The rest is on screen, sir." if len(vs) > 3 else "")
    return "[EMO:neutral] " + spoken, f"{PUBLIC}/img/{name}", j.get("reference", ref)
DEFINE_RE = re.compile(r"^\W*(?:(?:jarvis,?|hey jarvis,?|please|can you|could you)\s+)*(?:define|definition of|what(?:'s| is) the (?:definition|meaning) of|what does (?:the word )?['\"]?([a-z\-]+)['\"]? mean|meaning of)\s*(?:the word\s+)?['\"]?([a-z\-]+)?['\"]?\W*$", re.I)
async def define_card(text):
    m = DEFINE_RE.search(text.strip())
    if not m: return None
    word = (m.group(1) or m.group(2) or "").lower()
    if not word or word in ("it", "that", "this"): return None
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}")
    if r.status_code != 200: return None
    e = r.json()[0]
    pron = e.get("phonetic") or next((p.get("text") for p in e.get("phonetics", []) if p.get("text")), "")
    paras, first, ex = [], None, None
    for mn in e.get("meanings", [])[:2]:
        for k, df in enumerate(mn.get("definitions", [])[:2]):
            if len(paras) >= 3: break
            paras.append((f"{mn['partOfSpeech']} — {df['definition']}" if k == 0 else f"   {df['definition']}", (255, 255, 255)))
            first = first or (mn["partOfSpeech"], df["definition"]); ex = ex or df.get("example")
    if not paras: return None
    if ex: paras.append((f"“{ex}”", (144, 164, 174)))
    name = await asyncio.to_thread(render_doc, e.get("word", word), pron, paras, "Dictionary · dictionaryapi.dev (Wiktionary)")
    return f"[EMO:neutral] {word.capitalize()}, as a {first[0]}: {first[1]}", f"{PUBLIC}/img/{name}", word.capitalize()
import ast as _ast, operator as _op
_OPS = {_ast.Add: _op.add, _ast.Sub: _op.sub, _ast.Mult: _op.mul, _ast.Div: _op.truediv, _ast.Pow: _op.pow, _ast.USub: _op.neg, _ast.UAdd: _op.pos, _ast.Mod: _op.mod}
def _safe_eval(e):
    def ev(n):
        if isinstance(n, _ast.Expression): return ev(n.body)
        if isinstance(n, _ast.Constant) and isinstance(n.value, (int, float)): return n.value
        if isinstance(n, _ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, _ast.Pow) and abs(b) > 12: raise ValueError("pow")
            return _OPS[type(n.op)](a, b)
        if isinstance(n, _ast.UnaryOp) and type(n.op) in _OPS: return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, _ast.Call) and getattr(n.func, "id", "") == "sqrt" and len(n.args) == 1: return ev(n.args[0]) ** 0.5
        raise ValueError("bad")
    return ev(_ast.parse(e, mode="eval"))
def _fmt(v):
    if isinstance(v, float) and abs(v - round(v)) < 1e-9: v = int(round(v))
    return f"{v:,}" if isinstance(v, int) else f"{v:,.6f}".rstrip("0").rstrip(".")
_MATH_SUBS = [(r"(\d)\s*,\s*(?=\d{3}\b)", r"\1"), (r"\bmultiplied by\b|\btimes\b|×|∗|\*|(?<=\d)\s*x\s*(?=[\d(])|(?<=\d)\s*[xX]\b", " * "), (r"\bdivided by\b|÷|\bover\b", " / "), (r"\bplus\b|\band\b(?=\s*\d)", " + "), (r"\bminus\b|−|–", " - "),
              (r"\bsquared\b", " ** 2"), (r"\bcubed\b", " ** 3"), (r"\bto the power of\b|\^", " ** "), (r"\bmod(?:ulo)?\b", " % ")]
MATH_HINT = re.compile(r"\d\s*(?:x|×|\*|\+|-|−|/|÷|\^|%)\s*\d|\d\s*(?:%|percent)\s*of\s*\d|\b(?:times|multiplied by|divided by|plus|minus|squared|cubed|to the power of|square root of)\b", re.I)
def math_card_parts(text):
    """Deterministic maths: returns (title, steps) or None."""
    t = text.lower().replace("what's", "what is")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*of\s*(\d+(?:\.\d+)?)", t)
    if m:
        a, b = float(m.group(1)), float(m.group(2)); r = a / 100 * b
        return f"{_fmt(a)}% of {_fmt(b)}", [f"{_fmt(a)}% = {_fmt(a/100)}", f"{_fmt(a/100)} × {_fmt(b)} = {_fmt(r)}", f"= {_fmt(r)}"]
    m = re.search(r"square root of\s*(\d+(?:\.\d+)?)", t)
    if m:
        a = float(m.group(1)); return f"√{_fmt(a)}", [f"√{_fmt(a)} = {_fmt(a ** 0.5)}"]
    if not MATH_HINT.search(t): return None
    e = t
    for a, b in _MATH_SUBS: e = re.sub(a, b, e)
    cands = re.findall(r"[\d\.\s\+\-\*/%\(\)]+", e)
    cands = [c.strip() for c in cands if re.search(r"\d", c) and re.search(r"\d\s*(?:\*\*|[\+\-\*/%])\s*[\d(]", c.strip())]
    if not cands: return None
    ex = re.sub(r"(\d)\.(?!\d)", r"\1", max(cands, key=len)).strip().rstrip("+-*/%. ")
    try: v = _safe_eval(ex)
    except Exception: return None
    pretty = re.sub(r"\s+", " ", ex.replace("**", "^").replace("*", "×").replace("/", "÷").replace("-", "−")).strip()
    pretty = re.sub(r"\s*([×÷+−^%])\s*", r" \1 ", pretty).replace(" ^ 2", "²").replace(" ^ 3", "³").strip()
    pretty = re.sub(r"(\d{4,})", lambda m: _fmt(int(m.group(1))), pretty)
    return pretty, [f"{pretty} = {_fmt(v)}"]
# ---------------- teacher-style worked maths (deterministic) ----------------
def _ff(name, n):
    from PIL import ImageFont
    k = (name, n)
    if k not in _FONTS:
        f = next((x for x in (os.path.join(BASE, "fonts", name), "/app/fonts/" + name, "/usr/share/fonts/truetype/dejavu/" + name) if os.path.exists(x)), None)
        _FONTS[k] = ImageFont.truetype(f, n) if f else ImageFont.load_default(n)
    return _FONTS[k]
def _c(n): return f"{n:,}"
def teach_mul(a, b):
    za = len(str(a)) - len(str(a).rstrip("0")); zb = len(str(b)) - len(str(b).rstrip("0"))
    grid, steps, say = [], [], []
    if a % 10 == 0 and b % 10 == 0 and (a // 10**za) < 100 and (b // 10**zb) < 100 and za + zb >= 1:
        ca, cb = a // 10**za, b // 10**zb; core = ca * cb; z = za + zb
        steps = [f"Set the zeros aside: {_c(a)} has {za}, {_c(b)} has {zb} — {z} zero{'s' if z > 1 else ''} in total.",
                 f"Multiply what's left: {ca} × {cb} = {core}.",
                 f"Put the {z} zero{'s' if z > 1 else ''} back on the end: {core} → {_c(a*b)}."]
        grid = [("", str(ca) + " " * 0, "num"), ("×", str(cb), "num"), ("", "", "line"), ("", str(core), "num"), ("+ zeros", "0" * z, "carry"), ("", "", "line"), ("", str(a * b), "result")]
        say = [f"Drop the zeros and do {ca} times {cb}, that's {core}", f"then tack the {z} zeros back on"]
        return grid, steps, say
    if b > a and len(str(b)) > len(str(a)): a, b = b, a
    db = str(b)[::-1]; parts = []
    for i, d in enumerate(db):
        d = int(d); p = a * d
        carries, cr = [], 0
        for ch in str(a)[::-1]:
            v = int(ch) * d + cr; cr = v // 10; carries.append(cr)
        place = ["ones", "tens", "hundreds", "thousands"][i] if i < 4 else f"10^{i}"
        parts.append(p * 10**i)
        cs = "".join(str(x) if x else " " for x in carries[:-1][::-1])
        steps.append(f"{_c(a)} × {d} ({place}) = {_c(p)}" + (f", then add {i} zero{'s' if i > 1 else ''} → {_c(p * 10**i)}" if i else "") + (f".  Carries: {', '.join(str(x) for x in carries[:-1] if x)}" if any(carries[:-1]) else "."))
        if i == 0: grid.append(("carry", cs + " ", "carry"))
    grid += [("", str(a), "num"), ("×", str(b), "num"), ("", "", "line")]
    for k, p in enumerate(parts): grid.append(("+" if k == len(parts) - 1 and len(parts) > 1 else "", str(p), "num"))
    if len(parts) > 1:
        grid += [("", "", "line"), ("", str(a * b), "result")]
        steps.append(f"Add the partial products: {' + '.join(_c(p) for p in parts)} = {_c(a*b)}.")
        say = [f"{_c(a)} times {db[0]} is {_c(parts[0])}"] + [f"times {db[i]} {['', 'tens', 'hundreds', 'thousands'][i] if i < 4 else ''} is {_c(parts[i])}" for i in range(1, min(len(parts), 3))] + [f"add them up for {_c(a*b)}"]
    else:
        grid[-1:] = [("", "", "line"), ("", str(a * b), "result")]; grid = [g for g in grid if g[1] != str(parts[0]) or g[2] != "num"] if False else grid
        say = [f"{_c(a)} times {b} is {_c(a*b)}"]
    return grid, steps, say
def teach_add(a, b, sub=False):
    if sub and b > a: a, b = b, a; flip = True
    else: flip = False
    sa, sb = str(a), str(b); n = max(len(sa), len(sb)); A, B = sa.zfill(n), sb.zfill(n)
    marks, steps, cr = [" "] * n, [], 0
    for i in range(n - 1, -1, -1):
        x, y = int(A[i]), int(B[i]); col = ["ones", "tens", "hundreds", "thousands", "ten-thousands"][n - 1 - i] if n - 1 - i < 5 else "next"
        if not sub:
            v = x + y + cr
            steps.append(f"{col.capitalize()}: {x} + {y}" + (f" + {cr} carried" if cr else "") + f" = {v}" + (f" → write {v % 10}, carry 1." if v >= 10 else "."))
            cr = v // 10
            if cr and i: marks[i - 1] = "1"
        else:
            o = x; lent = cr; x -= cr; cr = 0
            base = f"{o} (lent 1, now {x})" if lent else f"{o}"
            if lent and o == 0: steps.append(f"{col.capitalize()}: 0 has nothing to lend, so it borrows 10 from the left → 10 − 1 lent = 9, then 9 − {y} = {9 - y}."); cr = 1; marks[i - 1] = "b" if i else " "; continue
            if x < y: steps.append(f"{col.capitalize()}: {base} is too small for {y}, so borrow 10 from the left → {x + 10} − {y} = {x + 10 - y}."); cr = 1; marks[i - 1] = "b" if i else " "
            else: steps.append(f"{col.capitalize()}: {base} − {y} = {x - y}.")
    if not sub and cr: steps.append("Carry the last 1 to the front.")
    r = a - b if sub else a + b
    grid = [("carry" if not sub else "borrow", " " + "".join(marks).replace("b", "·"), "carry"), ("", sa, "num"), ("−" if sub else "+", sb, "num"), ("", "", "line"), ("", str(r), "result")]
    if len(steps) > 5: steps = [s for s in steps if "carry" in s or "borrow" in s][:4] or steps[:4]; steps.insert(0, "Line up the columns and work right to left.")
    say = ["line them up and work right to left", ("carrying where a column passes nine" if not sub else "borrowing where needed"), f"for {_c(r)}"]
    if flip: r = -r; steps.append(f"The second number was bigger, so the answer is negative: {_c(r)}.")
    return grid, steps, say
def teach_div(a, b):
    q, rem = divmod(a, b); s = str(a); steps, rows, cur, started, col = [], [], 0, False, 0
    for i, ch in enumerate(s):
        cur = cur * 10 + int(ch)
        if cur < b and not started: continue
        started = True; d = cur // b; sub = d * b
        steps.append(f"{_c(b)} into {cur} goes {d} time{'s' if d != 1 else ''} ({d} × {b} = {sub}); {cur} − {sub} = {cur - sub}" + ("; bring down the " + s[i+1] + "." if i + 1 < len(s) else "."))
        rows.append((i, cur, sub)); cur -= sub
    steps.append(f"Answer: {_c(q)}" + (f" remainder {rem}" if rem else " exactly, no remainder") + ".")
    if rem: steps.append(f"As a decimal: {a/b:,.4f}".rstrip("0").rstrip(".") + ".")
    # render rows as long-division text (right-aligned to column index i)
    L = len(str(b)) + 1
    lines = [(" " * (L + 1) + str(q).rjust(len(s)), "result"), (" " * (L + 1) + "─" * len(s), "line_t"), (f"{b}) {s}", "divrow")]
    for k, (i, c, sb) in enumerate(rows):
        w = i + 1
        lines.append((" " * (L + 1) + ("−" + str(sb)).rjust(w), "dim"))
        nxt = c - sb
        if k + 1 < len(rows): nc = rows[k + 1][1]; lines.append((" " * (L + 1) + str(nc).rjust(rows[k + 1][0] + 1), "num"))
        else: lines.append((" " * (L + 1) + str(nxt).rjust(w) + (" ← r" if nxt else ""), "carry"))
    say = [f"{b} goes into {a} {q} times" + (f" with {rem} left over" if rem else " exactly")]
    return lines, steps[:6], say
def _pm_norm(t):
    e = t.lower().replace(",", "").replace("what's", "what is")
    for x, y in _MATH_SUBS: e = re.sub(x, y, e)
    e = re.sub(r"\)\s*(?:x|×|times|multiplied by)\s*", ")*", e); e = re.sub(r"(?:\bx|×|\btimes|multiplied by)\s*\(", "*(", e)
    e = re.sub(r"(\d)\s*\(", r"\1*(", e); e = re.sub(r"\)\s*\(", ")*(", e)
    e = re.sub(r"(\d)\.(?!\d)", r"\1", e)
    c = [x.strip() for x in re.findall(r"[\d\.\s\+\-\*/\(\)]+", e) if re.search(r"\d", x)]
    c = [x for x in c if len(re.findall(r"\*\*|[\+\-\*/]", x)) >= 2 and re.search(r"\d", x)]
    if not c: return None
    ex = max(c, key=len).strip().rstrip("+-*/. ")
    while ex.count("(") > ex.count(")") and ex.startswith("("): ex = ex[1:]
    while ex.count(")") > ex.count("(") and ex.endswith(")"): ex = ex[:-1]
    return ex
def teach_pemdas(text):
    import ast, operator as O
    ex = _pm_norm(text)
    if not ex: return None
    try: tree = ast.parse(ex, mode="eval").body
    except Exception: return None
    OPS = {ast.Add: ("+", 1, O.add, "add"), ast.Sub: ("−", 1, O.sub, "subtract"), ast.Mult: ("×", 2, O.mul, "multiply"), ast.Div: ("÷", 2, O.truediv, "divide"), ast.Pow: ("^", 3, O.pow, "power")}
    def ok(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return True
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub): return ok(n.operand)
        return isinstance(n, ast.BinOp) and type(n.op) in OPS and ok(n.left) and ok(n.right)
    if not ok(tree) or not isinstance(tree, ast.BinOp): return None
    def num(v):
        if isinstance(v, float) and v.is_integer(): v = int(v)
        return v
    def val(n): return -val(n.operand) if isinstance(n, ast.UnaryOp) else n.value
    def pr(n, parent=0, right=False):
        if isinstance(n, ast.Constant): return _fmt(n.value) if not isinstance(n.value, int) else _c(n.value)
        if isinstance(n, ast.UnaryOp): return "−" + pr(n.operand, 9)
        sym, p, _, _ = OPS[type(n.op)]
        if sym == "^": s = pr(n.left, 4) + {"2": "²", "3": "³"}.get(pr(n.right, 4), "^" + pr(n.right, 4))
        else: s = f"{pr(n.left, p)} {sym} {pr(n.right, p, True)}"
        return f"({s})" if p < parent or (right and p == parent and sym in "−÷") else s
    def depth_paren(n, parent=0, right=False):  # is this node inside explicit grouping?
        return False
    lines = [(pr(tree), "num")]; steps = []; say = []
    for _ in range(12):
        if isinstance(tree, ast.Constant): break
        # candidates: BinOps whose children are constants
        cand = []
        def walk(n, inparen, path):
            if isinstance(n, ast.BinOp):
                p = OPS[type(n.op)][1]
                for ch, side in ((n.left, "left"), (n.right, "right")):
                    if isinstance(ch, ast.BinOp) and (OPS[type(ch.op)][1] < p or (side == "right" and OPS[type(ch.op)][1] == p)): walk(ch, inparen + 1, path + [(n, side)])
                    else: walk(ch, inparen, path + [(n, side)])
                if all(isinstance(c, (ast.Constant, ast.UnaryOp)) for c in (n.left, n.right)): cand.append((inparen, p, n.col_offset, n, path))
        walk(tree, 0, [])
        if not cand: return None
        cand.sort(key=lambda c: (-c[0], -c[1], c[2])); ip, p, _, n, path = cand[0]
        sym, _, f, verb = OPS[type(n.op)]
        a, b = val(n.left), val(n.right)
        if sym == "÷" and b == 0: return None
        r = num(f(a, b)); r = round(r, 6) if isinstance(r, float) else r
        if abs(r) > 1e12: return None
        why = "Parentheses first" if ip else {3: "Exponents next", 2: "Multiply / divide next, left to right", 1: "Then add / subtract, left to right"}[p]
        if not ip and p == 3: why = "Exponents first"
        if not ip and p == 2 and not steps: why = "Multiply / divide before adding or subtracting"
        steps.append(f"{why}: {pr(n)} = {_fmt(r)}.")
        spk = pr(n).replace('×', 'times').replace('÷', 'divided by').replace('−', 'minus').replace('+', 'plus').replace('²', ' squared').replace('³', ' cubed')
        say.append((f"brackets first, {spk} is {_fmt(r)}" if ip else f"{verb} first, {spk} is {_fmt(r)}").replace("power first", "powers first") if not say else f"then {spk} is {_fmt(r)}")
        new = ast.Constant(r) if r >= 0 else ast.UnaryOp(ast.USub(), ast.Constant(-r))
        if not path: tree = new
        else:
            par, side = path[-1]; setattr(par, side, new)
        lines.append((("= " + pr(tree)), "result" if isinstance(tree, (ast.Constant, ast.UnaryOp)) else "num"))
    else: return None
    steps.insert(0, "Order of operations (PEMDAS): Parentheses, Exponents, Multiply/Divide, Add/Subtract.")
    ans = _fmt(val(tree)) if not isinstance(val(tree), int) else _c(val(tree))
    steps.append(f"So the answer is {ans}.")
    return lines[0][0], lines, "lines", steps, ans, say
def teach_parts(text):
    """Returns (title, grid_or_lines, kind, steps, answer, say) or None."""
    t = text.lower().replace(",", "").replace("what's", "what is")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*of\s*(\d+(?:\.\d+)?)", t)
    if m:
        a, b = float(m.group(1)), float(m.group(2)); r = a / 100 * b
        steps = [f"Percent means “out of 100”, so {_fmt(a)}% = {_fmt(a)} ÷ 100 = {_fmt(a/100)}.", f"“Of” means multiply: {_fmt(a/100)} × {_fmt(b)} = {_fmt(r)}."]
        if a in (10, 20, 25, 50, 5, 15): steps.append(f"Shortcut: 10% of {_fmt(b)} is {_fmt(b/10)}" + (f", half of that is {_fmt(b/20)}, together {_fmt(r)}." if a == 15 else f", so {_fmt(a)}% is {_fmt(r)}."))
        grid = [(f"{_fmt(a)}%", f"= {_fmt(a/100)}", "num"), ("×", _fmt(b), "num"), ("", "", "line"), ("", _fmt(r), "result")]
        return f"{_fmt(a)}% of {_fmt(b)}", grid, "grid", steps, _fmt(r), [f"{_fmt(a)} percent is {_fmt(a/100)}", f"times {_fmt(b)} gives {_fmt(r)}"]
    m = re.search(r"square root of\s*(\d+(?:\.\d+)?)|√\s*(\d+)", t)
    if m:
        a = float(m.group(1) or m.group(2)); r = a ** 0.5; ri = round(r)
        if ri * ri == a:
            steps = [f"Find the number that times itself makes {_fmt(a)}.", f"Try {ri - 1}: {ri-1} × {ri-1} = {(ri-1)**2} — too small.", f"Try {ri}: {ri} × {ri} = {_fmt(a)} ✓", f"So √{_fmt(a)} = {ri}."]
            say = [f"we need a number times itself to make {_fmt(a)}", f"{ri} times {ri} is {_fmt(a)}"]
        else:
            lo = int(r); steps = [f"{lo}² = {lo*lo} and {lo+1}² = {(lo+1)**2}, so the root is between {lo} and {lo+1}.", f"Closer to {lo if r - lo < 0.5 else lo + 1}; a calculator gives {r:.4f}."]
            say = [f"it sits between {lo} and {lo+1}", f"about {r:.2f}"]
        return f"√{_fmt(a)}", [("√", _fmt(a), "num"), ("", "", "line"), ("", _fmt(r if ri*ri != a else ri), "result")], "grid", steps, _fmt(round(r, 4)), say
    if not MATH_HINT.search(t): return None
    pm = teach_pemdas(t)
    if pm: return pm
    e = t
    for x, y in _MATH_SUBS: e = re.sub(x, y, e)
    e = re.sub(r"(\d)\.(?!\d)", r"\1", e)
    m = re.search(r"(\d+)\s*(\*\*|\*|/|\+|-)\s*(\d+)\b(?!\s*(?:\*\*|[\*/\+\-]))", e)
    if not m or len(re.findall(r"\d+", e)) > 2: return None
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    if op == "*": g, s, say = teach_mul(a, b); return f"{_c(a)} × {_c(b)}", g, "grid", s, _c(a * b), say
    if op == "+": g, s, say = teach_add(a, b); return f"{_c(a)} + {_c(b)}", g, "grid", s, _c(a + b), say
    if op == "-": g, s, say = teach_add(a, b, True); return f"{_c(a)} − {_c(b)}", g, "grid", s, _c(a - b), say
    if op == "/" and b:
        L, s, say = teach_div(a, b); q, r = divmod(a, b)
        return f"{_c(a)} ÷ {_c(b)}", L, "lines", s, _c(q) + (f" r {r}" if r else ""), say
    if op == "**" and b <= 6:
        steps = [f"{a}^{b} means {a} multiplied by itself {b} times: " + " × ".join([str(a)] * b) + "."]
        v = a
        for k in range(2, b + 1): steps.append(f"{_c(v)} × {a} = {_c(v * a)}"); v *= a
        sup = {2: "²", 3: "³"}.get(b, f"^{b}")
        return f"{a}{sup}", [("", " × ".join([str(a)] * b), "num"), ("", "", "line"), ("", str(a ** b), "result")], "grid", steps[:6], _c(a ** b), [f"{a} times itself {b} times", f"makes {_c(a**b)}"]
    return None
def render_teach(title, layout, kind, steps, answer):
    from PIL import Image, ImageDraw
    import textwrap
    W, H = 960, 440
    im = Image.new("RGB", (W, H), (11, 15, 20)); d = ImageDraw.Draw(im)
    acc = (0, 229, 255); gr = (144, 164, 174); wh = (255, 255, 255); dim = (110, 130, 140); amber = (255, 196, 64)
    d.rounded_rectangle([12, 10, W-12, H-10], 16, fill=(15, 23, 31), outline=(0, 78, 90), width=1)
    d.text((36, 22), title, font=_ff("DejaVuSans-Bold.ttf", 30), fill=acc)
    d.text((W-36, 30), "WORKED SOLUTION", font=_ff("DejaVuSans.ttf", 15), fill=dim, anchor="ra")
    d.line([36, 66, W-36, 66], fill=(0, 60, 70), width=1)
    LW = 380; d.rounded_rectangle([28, 80, LW, H-76], 12, fill=(11, 18, 25))
    if kind == "grid":
        rows = len(layout); fs = 44 if rows <= 5 else (38 if rows <= 7 else 30)
        mono = _ff("DejaVuSansMono.ttf", fs); monob = _ff("DejaVuSansMono-Bold.ttf", fs); small = _ff("DejaVuSansMono.ttf", int(fs * 0.55))
        cw = d.textlength("0", font=mono); widest = max((len(t) for _, t, _ in layout), default=1)
        xr = min(LW - 40, (28 + LW) / 2 + (widest * cw) / 2 + 10)
        lh = fs * 1.12; tot = sum(lh * (0.6 if st == "carry" else 0.35 if st == "line" else 1) for _, _, st in layout); y = 80 + max(8, (H - 156 - tot) / 2)
        for lab, txt, st in layout:
            if st == "line": d.line([xr - (widest + 1) * cw, y + lh * 0.15, xr, y + lh * 0.15], fill=gr, width=3); y += lh * 0.35; continue
            if st == "carry":
                fnt = small; col = amber
                for k, ch in enumerate(txt[::-1]): d.text((xr - (k + 0.5) * cw, y), ch, font=fnt, fill=col, anchor="ma")
                if lab and lab not in ("carry", "borrow"): d.text((xr - (len(txt) + 1.5) * cw, y), lab, font=small, fill=dim, anchor="ra")
                y += lh * 0.6; continue
            fnt, col = (monob, acc) if st == "result" else (mono, wh)
            d.text((xr, y), txt, font=fnt, fill=col, anchor="ra")
            if lab: d.text((xr - (len(txt) + 0.7) * cw, y), lab, font=mono if len(lab) == 1 else small, fill=gr, anchor="ra")
            y += lh
    else:
        n = len(layout); fs = 34 if n <= 6 else (28 if n <= 8 else 22)
        fs = max(16, min(fs, int((LW - 60) / (0.61 * max(len(t) for t, _ in layout)))))
        mono = _ff("DejaVuSansMono.ttf", fs); monob = _ff("DejaVuSansMono-Bold.ttf", fs)
        cw = d.textlength("0", font=mono); widest = max(len(t) for t, _ in layout); x0 = max(44, (28 + LW) / 2 - widest * cw / 2); y = 92
        for txt, st in layout:
            if st == "line_t":
                k = len(txt) - len(txt.lstrip()); d.line([x0 + (k - 1.2) * cw, y + fs * 0.1, x0 + len(txt) * cw, y + fs * 0.1], fill=gr, width=3)
                d.line([x0 + (k - 1.2) * cw, y + fs * 0.1, x0 + (k - 1.6) * cw, y + fs * 1.25], fill=gr, width=3); y += fs * 0.3; continue
            col = {"result": acc, "dim": gr, "carry": amber}.get(st, wh)
            d.text((x0, y), txt.replace(")", " ") if st == "divrow" else txt, font=monob if st == "result" else mono, fill=col); y += fs * 1.15
    # numbered steps
    n = len(steps); fs = 21 if n <= 3 else (19 if n <= 5 else 17); f = _ff("DejaVuSans.ttf", fs); fb = _ff("DejaVuSans-Bold.ttf", fs)
    x, y, maxw = LW + 30, 84, W - LW - 70
    for i, s in enumerate(steps):
        d.ellipse([x, y + 1, x + fs + 8, y + fs + 9], fill=(0, 78, 90)); d.text((x + (fs + 8) / 2, y + (fs + 10) / 2), str(i + 1), font=fb, fill=wh, anchor="mm")
        words, cur, out = s.split(), "", []
        for w in words:
            c2 = (cur + " " + w).strip()
            if d.textlength(c2, font=f) > maxw - fs - 18 and cur: out.append(cur); cur = w
            else: cur = c2
        if cur: out.append(cur)
        for ln in out[:3]:
            if y > H - 100: break
            d.text((x + fs + 18, y + 3), ln, font=f, fill=(220, 228, 232)); y += fs + 7
        y += 9
    # answer bar
    d.rounded_rectangle([28, H-66, W-28, H-20], 12, fill=(0, 52, 62), outline=acc, width=2)
    d.text((48, H-43), "ANSWER", font=_ff("DejaVuSans-Bold.ttf", 18), fill=acc, anchor="lm")
    d.text((W-48, H-43), answer, font=_ff("DejaVuSans-Bold.ttf", 32), fill=wh, anchor="rm")
    name = f"teach_{int(time.time()*1000)}.png"; im.save(os.path.join(IMG_DIR, name))
    olds = sorted(x for x in os.listdir(IMG_DIR) if x.startswith("teach_"))
    for old in olds[:-20]:
        try: os.remove(os.path.join(IMG_DIR, old))
        except Exception: pass
    return name
# ---------------- deterministic light algebra (sympy), teacher-style ----------------
ALG_RE = re.compile(r"\b(solve|simplify|evaluate|expand|factor|find (?:the value of )?[xyn]|what(?:'s| is) [xyn]|for [xyn]|when [xyn] ?(?:=|is|equals))\b|\b\d*\s*[xyn]\s*(?:\^|\*\*|squared|[+\-*/)]|plus|minus|times|over|divided)[^=]*(?:=|equals)|(?:=|equals)[^=]*\b\d*[xyn]\b", re.I)
def _alg_norm(t):
    t = t.lower().replace("what's", "what is").replace("−", "-").replace("–", "-").replace("×", "*").replace("÷", "/")
    t = re.sub(r"^\W*(?:hey )?(?:jarvis,?\s*)?(?:can you |could you |please )*(?:show me (?:how to )?|help me )?", "", t)
    for a, b in [(r"\bis equal to\b|\bequals\b|\bequal\b", "="), (r"\bplus\b", "+"), (r"\bminus\b", "-"), (r"\btimes\b|\bmultiplied by\b", "*"),
                 (r"\bdivided by\b|\bover\b", "/"), (r"\bsquared\b", "**2"), (r"\bcubed\b", "**3"), (r"\bto the power of\b|\^", "**"), (r"(\d)\s+x\s+(\d)", r"\1*\2")]:
        t = re.sub(a, b, t)
    return t
def _pp(e):
    import sympy as sp
    s = sp.sstr(e, order="grlex") if not isinstance(e, (int,)) else str(e)
    s = re.sub(r"\*\*2\b", "²", s); s = re.sub(r"\*\*3\b", "³", s); s = s.replace("**", "^")
    s = re.sub(r"(\d)\*([a-z(])", r"\1\2", s); s = re.sub(r"\)\*\(", ")(", s); s = s.replace("*", "·").replace("sqrt", "√").replace("-", "−")
    return s
def teach_algebra(text):
    """Returns (title, lines, 'lines', steps, answer, say) or None."""
    if not ALG_RE.search(text): return None
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
    T = standard_transformations + (implicit_multiplication_application, convert_xor)
    t = _alg_norm(text)
    var = sp.Symbol(next((v for v in "xyn" if re.search(rf"(?<![a-z]){v}(?![a-z])", t)), "x"))
    loc = {var.name: var}
    ev = re.search(rf"(?:when|if|for|where|with)\s+{var.name}\s*(?:=|is)\s*(-?\d+(?:\.\d+)?)", t)
    core = re.sub(rf"(?:when|if|where|with)\s+{var.name}\s*(?:=|is)\s*-?\d+(?:\.\d+)?", "", t) if ev else t
    core = re.sub(r"\b(solve|simplify|evaluate|expand|factor|find|what is|the value of|for [xyn]|if|then|and)\b", " ", core)
    core = re.sub(rf"^\s*{var.name}\s*(?:,|:)", " ", core)
    core = re.sub(rf"^\s*{var.name}\s+(?=[\d{var.name}(])", " ", core.strip())
    core = re.sub(r"[^0-9a-z=+\-*/().^ ]", " ", core).strip(" .=")
    core = re.sub(r"\s+", " ", core)
    if not core or not re.search(r"\d|[xyn]", core): return None
    P = lambda s: parse_expr(s, local_dict=loc, transformations=T)
    lines, steps, say = [], [], []
    try:
        if "=" in core and not ev:
            L, R = [P(x) for x in core.split("=")[:2]]
            if not (L.has(var) or R.has(var)): return None
            raw = re.sub(r"\s+", " ", core.replace("**2", "²").replace("**3", "³").replace("*", "·").replace("-", "−")).strip()
            raw = re.sub(r"\s*([=+−])\s*", r" \1 ", raw).replace("( ", "(").replace(" )", ")").strip()
            raw = re.sub(r"\(\s*−\s*", "(−", raw)
            raw = re.sub(r"\s+([²³])", r"\1", raw)
            title = f"Solve {raw}"; lines.append((raw, "num"))
            deg = sp.degree(sp.expand(L - R), var)
            if deg == 1 and "(" in core and f"{_pp(L)} = {_pp(R)}" != raw:
                steps.append(f"Distribute: multiply into the brackets → {_pp(L)} = {_pp(R)}."); lines.append((f"{_pp(L)} = {_pp(R)}", "num")); say.append("open the brackets")
            elif deg != 1 and f"{_pp(L)} = {_pp(R)}" != raw: lines.append((f"{_pp(L)} = {_pp(R)}", "num"))
            if deg == 1:
                if sp.expand(L) != L or sp.expand(R) != R:
                    L, R = sp.expand(L), sp.expand(R); steps.append("Distribute / expand the brackets."); lines.append((f"{_pp(L)} = {_pp(R)}", "num"))
                c = sp.expand(R).coeff(var, 1)
                if c != 0:
                    L, R = sp.expand(L - c * var), sp.expand(R - c * var)
                    steps.append(f"{'Subtract' if c > 0 else 'Add'} {_pp(abs(c) * var)} {'from' if c > 0 else 'to'} both sides to get the {var}'s together."); lines.append((f"{_pp(L)} = {_pp(R)}", "num"))
                b = sp.expand(L).coeff(var, 0)
                if b != 0:
                    L, R = sp.expand(L - b), sp.expand(R - b)
                    steps.append(f"{'Subtract' if b > 0 else 'Add'} {_pp(abs(b))} {'from' if b > 0 else 'to'} both sides to undo the {'+' if b > 0 else '−'} {_pp(abs(b))}."); lines.append((f"{_pp(L)} = {_pp(R)}", "num"))
                    say.append(f"{'take away' if b > 0 else 'add'} {_pp(abs(b))} on both sides")
                a = sp.expand(L).coeff(var, 1)
                if a != 1:
                    if a.is_Rational and a.q != 1 and a.p == 1:
                        steps.append(f"Multiply both sides by {a.q} to undo the ÷ {a.q}."); say.append(f"multiply both sides by {a.q}")
                    else:
                        steps.append(f"Divide both sides by {_pp(a)} to get {var} on its own."); say.append(f"divide both sides by {_pp(a)}")
                    L, R = sp.simplify(L / a), sp.simplify(R / a)
                sol = sp.nsimplify(R); lines.append((f"{var} = {_pp(sol)}", "result"))
                L0, R0 = [P(x) for x in core.split("=")[:2]]
                lines.append((f"✓ {_pp(L0.subs(var, sol))} = {_pp(R0.subs(var, sol))}", "carry"))
                steps.append(f"Check: put {var} = {_pp(sol)} back in → {_pp(sp.simplify(L0.subs(var, sol)))} = {_pp(sp.simplify(R0.subs(var, sol)))} ✓")
                ans = f"{var} = {_pp(sol)}"; say = (say or ["tidy it up"]) + [f"so {var} is {_pp(sol)}"]
                return title, lines, "lines", steps, ans, say
            if deg == 2:
                E = sp.expand(L - R); a, b, c = [E.coeff(var, k) for k in (2, 1, 0)]
                if R != 0: steps.append(f"Move everything to one side so it equals 0."); lines.append((f"{_pp(E)} = 0", "num"))
                F = sp.factor(E); roots = sp.solve(E, var)
                if F != E and all(r.is_rational for r in roots):
                    lines.append((f"{_pp(F)} = 0", "num"))
                    steps.append(f"Factor: find two numbers that multiply to {_pp(a*c)} and add to {_pp(b)} → {_pp(F)} = 0." if a == 1 else f"Factor: {_pp(F)} = 0.")
                    steps.append("Zero-product rule: if two things multiply to 0, one of them must be 0.")
                    say = [f"factor it as {_pp(F).replace(' − ', ' minus ').replace('−', 'minus ')}", "set each bracket to zero"]
                else:
                    D = b*b - 4*a*c
                    steps.append(f"Use the quadratic formula with a = {_pp(a)}, b = {_pp(b)}, c = {_pp(c)}.")
                    steps.append(f"Discriminant b² − 4ac = {_pp(D)}" + (" (negative → no real solutions)." if D < 0 else "."))
                    lines.append((f"{var} = (−b ± √{_pp(D)}) / {_pp(2*a)}", "num")); say = ["use the quadratic formula"]
                    if D < 0: return title, lines, "lines", steps, "no real solution", say + ["no real solutions"]
                roots = sorted(roots, key=lambda r: float(r))
                rs = [_pp(sp.nsimplify(r)) if r.is_rational else f"{float(r):.3f}".rstrip("0").rstrip(".").replace("-", "−") for r in roots]
                if not all(r.is_rational for r in roots):
                    lines.append((f"{var} = ({_pp(-b)} ± √{_pp(D)}) / {_pp(2*a)}", "num")); steps.append(f"So {var} = ({_pp(-b)} ± √{_pp(D)}) / {_pp(2*a)} ≈ " + " or ".join(rs) + ".")
                lines.append((f"{var} = " + " or ".join(rs), "result"))
                chk = ", ".join(f"{var}={r}: {_pp(sp.simplify(E.subs(var, rr)))}" for r, rr in zip(rs, roots) if rr.is_rational) or "both roots give 0 (exact form)"
                steps.append(f"Check each in {_pp(E)}: {chk} ✓"); lines.append((f"✓ both give 0", "carry"))
                ans = f"{var} = " + " or ".join(rs)
                return title, lines, "lines", steps, ans, say + [f"{var} is " + " or ".join(rs)]
            return None
        E = P(core.split("=")[0])
        if not E.has(var) and not ev: return None
        if ev:
            v = sp.nsimplify(ev.group(1)); title = f"{_pp(E)} when {var} = {_pp(v)}"
            lines.append((_pp(E), "num"))
            sub = _pp(E).replace(var.name, f"({_pp(v)})")
            steps.append(f"Replace every {var} with {_pp(v)}."); lines.append((sub, "num"))
            r = sp.nsimplify(E.subs(var, v)); terms = sp.Add.make_args(sp.expand(E))
            if len(terms) > 1:
                vals = [sp.nsimplify(tm.subs(var, v)) for tm in sorted(terms, key=lambda q: -sp.degree(q, var))]
                mid = " + ".join(_pp(x) for x in vals).replace("+ −", "− ").replace("+ -", "− ")
                steps.append("Order of operations: powers first, then × and ÷, so work out each term: " + ", ".join(f"{_pp(tm)} → {_pp(sp.nsimplify(tm.subs(var, v)))}" for tm in sorted(terms, key=lambda q: -sp.degree(q, var))) + ".")
                lines.append((f"= {mid}", "num")); steps.append(f"Add them up: {mid} = {_pp(r)}.")
            else: steps.append("Follow order of operations: powers, then × and ÷, then + and −.")
            lines.append((f"= {_pp(r)}", "result"))
            return title, lines, "lines", steps, _pp(r), [f"swap {var} for {_pp(v)}", f"and work it out to {_pp(r)}"]
        S = sp.expand(E); title = f"Simplify {_pp(E)}"
        lines.append((_pp(E), "num"))
        if sp.expand(E) != E and E.has(sp.Mul) and any(isinstance(a, sp.Add) for a in sp.preorder_traversal(E) if a is not E):
            steps.append("Expand the brackets (multiply each term inside).")
        terms = sp.Add.make_args(E)
        if len(terms) > 1: steps.append(f"Group like terms: the {var}-terms together and the plain numbers together.")
        steps.append(f"Combine them → {_pp(S)}."); lines.append((f"= {_pp(S)}", "result"))
        return title, lines, "lines", steps, _pp(S), [f"group the like terms", f"you get {_pp(S).replace('−', 'minus')}"]
    except Exception as e:
        log.warning("algebra %s: %s", core, e); return None
async def math_card(text):
    try: tp = teach_algebra(text) or teach_parts(text)
    except Exception as e: log.warning("teach: %s", e); tp = None
    if tp:
        title, layout, kind, steps, answer, say = tp
        name = await asyncio.to_thread(render_teach, title, layout, kind, steps, answer)
        sp = ", ".join(say); sp = sp[0].upper() + sp[1:]
        if answer.split(" r ")[0].split("=")[-1].strip().split(" or ")[0].replace("−", "-") not in sp.replace("−", "-"): sp += f". That's {answer.replace(' r ', ' remainder ')}"
        return "[EMO:happy] " + sp + ". It's all on screen, sir.", f"{PUBLIC}/img/{name}", title
    return await _math_card_simple(text)
async def _math_card_simple(text):
    got = math_card_parts(text)
    if not got: return None
    title, lines = got
    name = await asyncio.to_thread(render_card, title, lines)
    res = lines[-1].split("=")[-1].strip()
    spoken = f"[EMO:smug] {title.replace('×', 'times').replace('÷', 'divided by').replace('−', 'minus').replace('²', ' squared').replace('³', ' cubed').replace('√', 'the square root of ')} is {res}, sir. It's on screen."
    return spoken, f"{PUBLIC}/img/{name}", title
async def card_from(body):
    parts = [x.strip() for x in body.split("||") if x.strip()]
    if not parts: return None, ""
    title = parts[0]; lines = parts[1:] or parts[:1]
    name = await asyncio.to_thread(render_card, title, lines)
    return f"{PUBLIC}/img/{name}", title
SHOW_REQ = re.compile(r"\bwhat (?:does|do) (?:an? |the )?(.+?) look like|\b(?:show me|can you show me|let me see|display)\s+(?:an? |the |some )?(?:(?:picture|photo|image|pic)s? of\s+)?(?:an? |the |some )?(.+?)[?.!]*$|\b(?:picture|photo|image|pic) of\s+(?:an? |the |some )?(.+?)[?.!]*$", re.I)
EMO_RE = re.compile(r"\[\s*EMO\s*:\s*(\w+)\s*\]", re.I)
EMOS = {"happy", "sad", "mad", "laughing", "crying", "surprised", "smug", "neutral"}
LAST_EMO = {}

def take_emotion(reply):
    """Strips [EMO:x] tags from the spoken text; returns (clean_reply, emotion)."""
    m = EMO_RE.search(reply or ""); e = m.group(1).lower() if m else "neutral"
    return re.sub(r"\s{2,}", " ", EMO_RE.sub("", reply or "")).strip(), (e if e in EMOS else "neutral")

async def respond(text):
    reply, image, caption = await _respond(text)
    reply, LAST_EMO["e"] = take_emotion(reply)
    return reply, image, caption

async def _respond(text):
    """Returns (spoken_reply, image_url, caption). A requested sound effect is left in LAST_SFX['q']."""
    LAST_SFX.clear()
    m = SFX_REQ.search(text)
    if m:
        q = next((g for g in m.groups() if g), "").strip()
        if q and not re.search(r"\b(weather|news|time|that)\b", q, re.I):
            import random
            LAST_SFX["q"] = q; return random.choice(SFX_QUIPS), None, ""
    if HOME_LATLON and WX_SHOW.search(text) and not wx_elsewhere(text):
        try: return await weather_card()
        except Exception as e: log.warning("weather card: %s", e)
    if FOLLOW_SHOW.search(text) and history:
        for prev_q, _r in reversed(list(history)[-3:]):
            if math_card_parts(prev_q) or teach_parts(prev_q) or teach_algebra(prev_q):
                got = await math_card(prev_q)
                if got: return got
                break
    if ALG_RE.search(text) and not WX_RE.search(text):
        try:
            if teach_algebra(text): return await math_card(text)
        except Exception as e: log.warning("algebra card: %s", e)
    if (re.search(r"\b(show|display|let me see|put)\b", text, re.I) or re.match(r"^\W*(?:what(?:'s| is)|calculate|compute|how much is)\b", text, re.I)) and not WX_RE.search(text):
        try:
            got = await math_card(text)
            if got: return got
        except Exception as e: log.warning("math card: %s", e)
    for fn in (bible_card, define_card):
        try:
            got = await fn(text)
            if got: return got
        except Exception as e: log.warning("%s: %s", fn.__name__, e)
    _else = wx_elsewhere(text)
    wx_q = bool(HOME_LATLON and WX_ASK.search(text) and not _else)
    follow = bool(FOLLOW_SHOW.search(text))
    m = None if (WHY_SHOW.search(text) or follow) else SHOW_REQ.search(text)
    if m and not wx_q and VAGUE_Q.match(next((g for g in m.groups() if g), "").strip()) and len(next((g for g in m.groups() if g), "").split()) <= 4 and re.match(r"(it|that|this|them|those|these|me|us|how|again|the |your |on )", next((g for g in m.groups() if g), "").strip(), re.I): m = None; follow = True
    if m and not wx_q:
        q = next((g for g in m.groups() if g), "").strip()
        if q and not re.search(r"\b(weather|forecast|news|time)\b", q, re.I):
            u = await fetch_image(q)
            reply = f"Here's {'an' if q[0].lower() in 'aeiou' else 'a'} {q}, sir." if u else f"I'm afraid I couldn't find a picture of {q}, sir."
            if not re.search(r"^(an? |the )", q) and q.endswith("s"): reply = f"Here are some {q}, sir." if u else reply
            return reply, u, q.capitalize()
    raw = await ask_grok(text + (" (Follow-up: show your previous answer on screen. Use [CARD: ...] unless the previous topic was a visual thing.)" if follow else ""))
    cm = CARD_TAG.search(raw); raw = CARD_TAG.sub("", raw).strip()
    if cm:
        try:
            if teach_parts(text) or teach_algebra(text):
                _sp, u, title = await math_card(text)
                return SHOW_TAG.sub("", POSTS_TAG.sub("", SFX_TAG.sub("", raw))).strip(), u, title
            u, title = await card_from(cm.group(1))
            if u: return SHOW_TAG.sub("", POSTS_TAG.sub("", SFX_TAG.sub("", raw))).strip(), u, title
        except Exception as e: log.warning("card: %s", e)
    sm = SFX_TAG.search(raw)
    if sm: LAST_SFX["q"] = sm.group(1).strip(); raw = SFX_TAG.sub("", raw).strip()
    pm = POSTS_TAG.search(raw); raw = POSTS_TAG.sub("", raw)
    tm = SHOW_TAG.search(raw)
    spoken = SHOW_TAG.sub("", raw).strip()
    if tm and (WX_RE.search(tm.group(1)) or BANNED_IMG.search(tm.group(1))): tm = None
    if wx_q or (tm is None and not _else and WX_RE.search(raw) and WX_ASK.search(text)):
        try:
            j = await wx_data(); name = await asyncio.to_thread(render_weather, j)
            return spoken, f"{PUBLIC}/img/{name}", f"{HOME_PLACE} weather"
        except Exception as e: log.warning("weather card: %s", e)
    if pm and not tm:
        return spoken, None, "\n\n".join(x.strip() for x in pm.group(1).split("||") if x.strip())
    if tm:
        q = tm.group(1).strip(); u = await fetch_image(q)
        return spoken, u, q.capitalize()
    return spoken, None, ""

async def pipeline(data, fmt, stream=False):
    t0 = time.time()
    text = await asyncio.to_thread(transcribe, data); t1 = time.time()
    if is_dismiss(text):
        log.info("dismiss %r", text); return text, "", b"", "audio/mpeg", "none", {"stt_ms": int((t1-t0)*1000), "action": "dismiss"}
    if text and LOOK_RE.search(text):
        audio, ctype, engine = await tts("Let me take a look.", fmt)
        return text, "Let me take a look.", audio, ctype, engine, {"stt_ms": int((t1-t0)*1000), "action": "look"}
    if not text:
        reply = "Sorry, I didn't catch that."
    else:
        async with lock:
            reply, image, caption = await respond(text); history.append((text, reply + (f" [showed picture: {caption}]" if image else (" [showed X posts]" if caption else "")))); save_hist()
    t2 = time.time()
    sq = LAST_SFX.get("q") if text else None; fx = await sound_effect(sq) if sq and fmt == "mp3" else None
    if fx:
        audio, ctype, engine = fx + await quip_audio(reply), "audio/mpeg", "elevenlabs-sfx"
    elif stream and EL_KEY and fmt == "mp3":
        sid = start_stream(reply); audio, ctype, engine = b"", "audio/mpeg", "elevenlabs-stream:" + sid
    else:
        audio, ctype, engine = await tts(reply, fmt)
    t3 = time.time()
    if not text: image = None; caption = ""
    timing = {"emotion": LAST_EMO.get("e", "neutral") if text else "neutral", "sfx": sq if fx else None, "image": image, "caption": caption, "stt_ms": int((t1-t0)*1000), "llm_ms": int((t2-t1)*1000), "tts_ms": int((t3-t2)*1000), "total_ms": int((t3-t0)*1000)}
    log.info("talk %s | %r -> %r", timing, text, reply)
    return text, reply, audio, ctype, engine, timing

def b64(s): return base64.b64encode(s.encode()).decode()

@app.post("/talk")
async def talk(request: Request, format: str = Query("mp3"), stream: int = Query(0)):
    text, reply, audio, ctype, engine, t = await pipeline(await body_audio(request), format, bool(stream))
    h = {}
    if engine.startswith("elevenlabs-stream:"):
        h["X-Audio-URL"] = f"{PUBLIC or str(request.base_url).rstrip('/')}/audio/{engine.split(':')[1]}"; engine = "elevenlabs-stream"
    h |= {"X-Transcript-B64": b64(text), "X-Reply-B64": b64(reply), "X-TTS-Engine": engine, "X-Timing": json.dumps(t)}
    if t.get("action"): h["X-Action"] = t["action"]
    h["X-Emotion"] = t.get("emotion") or "neutral"
    if t.get("image") or t.get("caption"): h["X-Image"] = t.get("image") or ""; h["X-Caption-B64"] = b64(t.get("caption", ""))
    return Response(audio, media_type=ctype, headers=h)

@app.post("/talk.json")
async def talk_json(request: Request, format: str = Query("mp3")):
    text, reply, audio, ctype, engine, t = await pipeline(await body_audio(request), format)
    return {"transcript": text, "reply": reply, "image": t.get("image"), "caption": t.get("caption"), "audio_b64": base64.b64encode(audio).decode(), "content_type": ctype, "tts_engine": engine, "timing": t}

@app.post("/stt")
async def stt_only(request: Request):
    t0 = time.time(); text = await asyncio.to_thread(transcribe, await body_audio(request))
    return {"transcript": text, "stt_ms": int((time.time()-t0)*1000)}

@app.post("/say")
async def say(text: str = Query(...), format: str = Query("mp3"), engine: str = Query("auto")):
    t0 = time.time(); audio, ctype, eng = await tts(text, format, engine)
    return Response(audio, media_type=ctype, headers={"X-TTS-Engine": eng, "X-TTS-ms": str(int((time.time()-t0)*1000))})

@app.post("/reset")
async def reset():
    history.clear(); save_hist(); warm.reset(); return {"ok": True, "memory_kept": True}

@app.post("/text")
async def text_turn(q: str = Query(...)):
    t0 = time.time()
    if is_dismiss(q): return {"reply": "", "action": "dismiss", "emotion": "neutral", "image": None, "caption": "", "sfx": None, "sfx_url": None, "llm_ms": 0, "sfx_ms": 0}
    async with lock:
        reply, image, caption = await respond(q); history.append((q, reply + (f" [showed picture: {caption}]" if image else (" [showed X posts]" if caption else "")))); save_hist()
    sq = LAST_SFX.get("q"); t1 = time.time(); fx = await sound_effect(sq) if sq else None
    return {"reply": reply, "emotion": LAST_EMO.get("e", "neutral"), "image": image, "caption": caption, "sfx": sq if fx else None,
            "sfx_url": f"{PUBLIC}/sfx/{sfx_key(sq)}.mp3" if fx else None, "llm_ms": int((t1-t0)*1000), "sfx_ms": int((time.time()-t1)*1000)}

@app.get("/memory")
async def memory_get():
    return {"memory": load_mem()}

@app.get("/health")
async def health():
    return {"ok": True, "elevenlabs_key": bool(EL_KEY), "voice": EL_VOICE, "model": EL_MODEL,
            "piper_fallback": os.path.exists(PIPER_VOICE), "grok_cli": os.path.exists(GROK), "grok_warm": warm.alive(), "history_turns": len(history)}

@app.post("/look")
async def look(request: Request, format: str = Query("mp3"), q: str = Query("")):
    t0 = time.time()
    form = await request.form()
    img = form.get("image") or form.get("file")
    if img is None: return Response('{"error":"image field required"}', status_code=400, media_type="application/json")
    imgb = await img.read(); mime = getattr(img, "content_type", None) or "image/jpeg"
    question = q or (form.get("q") if isinstance(form.get("q"), str) else "")
    au = form.get("audio")
    if au is not None and not isinstance(au, str):
        question = await asyncio.to_thread(transcribe, await au.read())
    question = question or "What do you see? Describe it briefly."
    t1 = time.time()
    blocks = [{"type": "text", "text": build_prompt(question + " (The user is showing you the attached camera photo.)")},
              {"type": "image", "mimeType": mime, "data": base64.b64encode(imgb).decode()}]
    pf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir="/tmp/gv"); json.dump(blocks, pf); pf.close()
    async with lock:
        reply = await run_grok(["--prompt-json", open(pf.name).read()])
        reply, emo = take_emotion(reply); reply = SFX_TAG.sub("", SHOW_TAG.sub("", POSTS_TAG.sub("", reply))).strip(); history.append((question + " [photo]", reply)); save_hist()
    cam = f"cam_{int(time.time())}.jpg"; open(os.path.join(IMG_DIR, cam), "wb").write(imgb)
    os.unlink(pf.name); t2 = time.time()
    audio, ctype, engine = await tts(reply, format); t3 = time.time()
    t = {"stt_ms": int((t1-t0)*1000), "llm_ms": int((t2-t1)*1000), "tts_ms": int((t3-t2)*1000), "total_ms": int((t3-t0)*1000)}
    log.info("look %s | %r -> %r", t, question, reply)
    return Response(audio, media_type=ctype, headers={"X-Transcript-B64": b64(question), "X-Reply-B64": b64(reply), "X-TTS-Engine": engine, "X-Timing": json.dumps(t),
                    "X-Emotion": emo, "X-Image": f"{PUBLIC}/img/{cam}", "X-Caption-B64": b64("What I see")})
