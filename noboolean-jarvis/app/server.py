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
            "Never say you cannot show images. "
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
    parts = re.split(r"(?<=[.!?])(?=[A-Z])|\n+", reply)
    while len(parts) > 1 and re.match(r"\s*(I'll|I will|Let me|One moment|Allow me|Checking|Give me)\b", parts[0], re.I): parts.pop(0)
    return " ".join(x.strip() for x in parts if x.strip())

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
    parts = re.split(r"(?<=[.!?])(?=[A-Z])|\n+", reply)
    while len(parts) > 1 and re.match(r"\s*(I'll|I will|Let me|One moment|Allow me|Checking|Give me)\b", parts[0], re.I): parts.pop(0)
    return " ".join(x.strip() for x in parts if x.strip())

async def ask_grok(text):
    r = memory_command(text)
    if r: return r
    ctx = await home_weather() if WX_RE.search(text) else ""
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
async def el_stream_task(sid, text):
    st = STREAMS[sid]
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            async with c.stream("POST", f"https://api.elevenlabs.io/v1/text-to-speech/{EL_VOICE}/stream?output_format=mp3_44100_128&optimize_streaming_latency=3",
                    headers={"xi-api-key": EL_KEY}, json={"text": text, "model_id": EL_MODEL}) as r:
                if r.status_code != 200: raise RuntimeError(f"elevenlabs stream {r.status_code}")
                async for ch in r.aiter_bytes(4096):
                    if not st["t1"]: st["t1"] = time.time()
                    st["buf"].append(ch); st["ev"].set()
    except Exception as e:
        log.warning("tts stream %s; piper fallback", e)
        if not st["buf"]:
            st["ctype"] = "audio/wav"; st["buf"].append(await asyncio.to_thread(piper_tts, text))
    finally:
        st["done"] = True; st["ev"].set()
def start_stream(text):
    sid = hashlib.sha1(f"{time.time()}{text}".encode()).hexdigest()[:12]
    STREAMS[sid] = {"buf": [], "done": False, "ev": asyncio.Event(), "ctype": "audio/mpeg", "t0": time.time(), "t1": 0}
    for k in [k for k, v in STREAMS.items() if time.time() - v["t0"] > 300]: STREAMS.pop(k, None)
    asyncio.create_task(el_stream_task(sid, text)); return sid
from fastapi.responses import StreamingResponse
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
LOOK_RE = re.compile(r"\b(look at (this|that|me|it)|what am i holding|what('s| is) (this|that) (in my hand|i'm holding)|can you see|what do you see|take a (look|picture|photo))\b", re.I)

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
def render_weather(j):
    from PIL import Image, ImageDraw, ImageFont
    import datetime
    W, H = 960, 424
    im = Image.new("RGB", (W, H), (6, 12, 18)); d = ImageDraw.Draw(im)
    F = lambda n: ImageFont.load_default(n)
    acc = (0, 229, 255); gr = (144, 164, 174); wh = (255, 255, 255)
    cu, hr, dy = j["current"], j["hourly"], j["daily"]
    d.rounded_rectangle([12, 10, W-12, 238], 22, fill=(12, 26, 36))
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
        x = 12 + (k-1)*(cw+10); d.rounded_rectangle([x, 250, x+cw, H-6], 18, fill=(12, 26, 36)); m = x + cw/2
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
async def weather_card():
    j = await wx_data(); name = await asyncio.to_thread(render_weather, j)
    cu, dy = j["current"], j["daily"]
    spoken = (f"Currently {round(cu['temperature_2m'])} degrees and {WMO.get(cu['weather_code'], 'unremarkable')}, sir, "
              f"with a high of {round(dy['temperature_2m_max'][0])}. The week ahead is on screen.")
    return spoken, f"{PUBLIC}/img/{name}", f"{HOME_PLACE} weather"

POSTS_TAG = re.compile(r"\[POSTS:\s*(.+?)\]\s*$", re.I | re.S)
SHOW_TAG = re.compile(r"\[SHOW:\s*([^\]]+)\]", re.I)
SHOW_REQ = re.compile(r"\bwhat (?:does|do) (?:an? |the )?(.+?) look like|\b(?:show me|can you show me|let me see|display)\s+(?:an? |the |some )?(?:(?:picture|photo|image|pic)s? of\s+)?(?:an? |the |some )?(.+?)[?.!]*$|\b(?:picture|photo|image|pic) of\s+(?:an? |the |some )?(.+?)[?.!]*$", re.I)
async def respond(text):
    """Returns (spoken_reply, image_url, caption). A requested sound effect is left in LAST_SFX['q']."""
    LAST_SFX.clear()
    m = SFX_REQ.search(text)
    if m:
        q = next((g for g in m.groups() if g), "").strip()
        if q and not re.search(r"\b(weather|news|time|that)\b", q, re.I):
            import random
            LAST_SFX["q"] = q; return random.choice(SFX_QUIPS), None, ""
    if HOME_LATLON and WX_SHOW.search(text):
        try: return await weather_card()
        except Exception as e: log.warning("weather card: %s", e)
    m = SHOW_REQ.search(text)
    if m:
        q = next((g for g in m.groups() if g), "").strip()
        if q and not re.search(r"\b(weather|forecast|news|time)\b", q, re.I):
            u = await fetch_image(q)
            reply = f"Here's {'an' if q[0].lower() in 'aeiou' else 'a'} {q}, sir." if u else f"I'm afraid I couldn't find a picture of {q}, sir."
            if not re.search(r"^(an? |the )", q) and q.endswith("s"): reply = f"Here are some {q}, sir." if u else reply
            return reply, u, q.capitalize()
    raw = await ask_grok(text)
    sm = SFX_TAG.search(raw)
    if sm: LAST_SFX["q"] = sm.group(1).strip(); raw = SFX_TAG.sub("", raw).strip()
    pm = POSTS_TAG.search(raw); raw = POSTS_TAG.sub("", raw)
    tm = SHOW_TAG.search(raw)
    spoken = SHOW_TAG.sub("", raw).strip()
    if pm and not tm:
        return spoken, None, "\n\n".join(x.strip() for x in pm.group(1).split("||") if x.strip())
    if tm:
        q = tm.group(1).strip(); u = await fetch_image(q)
        return spoken, u, q.capitalize()
    return spoken, None, ""

async def pipeline(data, fmt, stream=False):
    t0 = time.time()
    text = await asyncio.to_thread(transcribe, data); t1 = time.time()
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
    timing = {"sfx": sq if fx else None, "image": image, "caption": caption, "stt_ms": int((t1-t0)*1000), "llm_ms": int((t2-t1)*1000), "tts_ms": int((t3-t2)*1000), "total_ms": int((t3-t0)*1000)}
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
    async with lock:
        reply, image, caption = await respond(q); history.append((q, reply + (f" [showed picture: {caption}]" if image else (" [showed X posts]" if caption else "")))); save_hist()
    sq = LAST_SFX.get("q"); t1 = time.time(); fx = await sound_effect(sq) if sq else None
    return {"reply": reply, "image": image, "caption": caption, "sfx": sq if fx else None,
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
        reply = await run_grok(["--prompt-json", open(pf.name).read()]); history.append((question + " [photo]", reply)); save_hist()
    os.unlink(pf.name); t2 = time.time()
    audio, ctype, engine = await tts(reply, format); t3 = time.time()
    t = {"stt_ms": int((t1-t0)*1000), "llm_ms": int((t2-t1)*1000), "tts_ms": int((t3-t2)*1000), "total_ms": int((t3-t0)*1000)}
    log.info("look %s | %r -> %r", t, question, reply)
    return Response(audio, media_type=ctype, headers={"X-Transcript-B64": b64(question), "X-Reply-B64": b64(reply), "X-TTS-Engine": engine, "X-Timing": json.dumps(t)})
