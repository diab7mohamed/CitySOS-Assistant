"""
citysos_assistant.py - CitySOS bilingual (Arabic / English) assistant with voice support.

Install:
    pip install anthropic fastapi "uvicorn[standard]" python-dotenv python-multipart faster-whisper edge-tts requests

Env:
    GROQ_API_KEY=...              (preferred; keep on the SERVER only)
    GROQ_MODEL=...                (optional)
    ANTHROPIC_API_KEY=...         (optional fallback provider)
    CITYSOS_MODEL=...             (optional model override)
    WHISPER_MODEL=small          (optional: tiny/base/small/medium - medium is better for Egyptian Arabic)

    Local development: copy .env.example to .env and add a newly created API key.

Run:
    uvicorn citysos_assistant:app --host 0.0.0.0 --port 3000
    then open http://localhost:3000  (put index.html next to this file; the mic needs HTTPS or localhost)

Flow:
    text  -> /api/chat   -> Claude -> JSON reply (text only)
    voice -> /api/voice  -> Whisper (speech-to-text) -> Claude -> edge-tts (text-to-speech) -> JSON + audio
    Note: Claude does not take raw audio, so voice is transcribed first.
    The reply follows the user's language and, if they sent voice, comes back as voice too.
"""

import asyncio
import base64
import io
import json
import os
import re
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import httpx

try:
    import edge_tts
except ImportError:
    edge_tts = None

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = (
    os.getenv("CITYSOS_MODEL")
    or os.getenv("GROQ_MODEL")
    or ("openai/gpt-oss-20b" if GROQ_API_KEY else "claude-sonnet-5-5")
)
WHISPER_SIZE = os.getenv("WHISPER_MODEL", "small")
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MB

CATEGORIES = ["automotive", "medical", "pharmacy", "plumbing", "electrical", "home_other", "emergency", "unknown"]

SYSTEM = f"""You are the CitySOS assistant for people in Egypt. Users write or speak in Modern Standard Arabic, Egyptian colloquial Arabic (e.g. "العربية بطلت", "ماسورة مكسورة", "النور فصل"), English, or mixed Arabic-English.
Work directly from the user's original wording. Never translate the message to another language first.
Some messages come from speech-to-text and may contain small recognition mistakes; infer the intended meaning.

Tasks:
1. Decide the dominant language of the latest user message: "ar" or "en" (mixed -> the dominant one). Reply ONLY in that language. If "ar", write natural, warm Egyptian Arabic (عامية مصرية), not stiff formal Arabic. Stay in the same language as earlier turns unless the user switches.
2. Map the problem to exactly one category id from: {", ".join(CATEGORIES)}.
   - automotive: car breakdown, flat tyre, battery, towing, fuel, engine
   - medical: illness, injury, doctor, hospital (never diagnose; advise seeing a professional)
   - pharmacy: medicine, nearest pharmacy
   - plumbing: pipes, leaks, water
   - electrical: power cut, sockets, short circuit, wiring
   - home_other: AC, locks, appliances, painting, carpentry
   - emergency: life-threatening (chest pain, can't breathe, heavy bleeding, fire, gas leak, crime in progress, accident with injuries)
   - unknown: not enough information
3. Answer what the user actually asked. Give short practical first-aid / safety guidance (2-5 short lines). For emergency, lead with Egypt numbers: ambulance 123, police 122, fire 180, plus one safety action.
4. Ask ONE relevant follow-up question (location, car model, severity, ...) unless it is an emergency where the numbers/action come first. Use "" if none is needed.
5. If the message came by voice, keep sentences short and avoid lists, symbols or emojis, because the reply will be read aloud.

Output ONLY compact JSON, no markdown:
{{"lang":"ar|en","category":"<id>","emergency":true|false,"reply":"...","followup":"..."}}"""

# Safety net: if any of these appear, treat as emergency even if the model misses it.
EMERGENCY_PATTERNS = re.compile(
    r"(حريق|ولعت|بيولع|غاز|تسريب غاز|مش قادر أتنفس|مش قادر اتنفس|ضيق تنفس|نزيف|ألم في الصدر|وجع في صدري|"
    r"اغماء|إغماء|مغمي|اختناق|سرقة|بيضربوني|حادثة|fire|gas leak|can'?t breathe|chest pain|bleeding|unconscious|"
    r"choking|robbery|accident)",
    re.IGNORECASE,
)

EMERGENCY_LINE = {
    "ar": "في الطوارئ اتصل فورًا: إسعاف 123، شرطة 122، إطفاء 180.",
    "en": "In an emergency call now: ambulance 123, police 122, fire 180.",
}

ERR = {
    "ar": "حصلت مشكلة مؤقتة. جرّب تاني بعد شوية. ولو الحالة طارئة اتصل: إسعاف 123، شرطة 122، إطفاء 180.",
    "en": "A temporary error occurred. Please try again. If this is an emergency call: ambulance 123, police 122, fire 180.",
}

VOICES = {"ar": "ar-EG-SalmaNeural", "en": "en-US-JennyNeural"}

API_KEY = ANTHROPIC_API_KEY
client = AsyncAnthropic(api_key=API_KEY) if AsyncAnthropic and API_KEY else None
_whisper: Optional[Any] = None


# ---------------------------------------------------------------- helpers
def detect_script(text: str) -> Optional[str]:
    """Cheap hint from character script; the model makes the final call."""
    ar = len(re.findall(r"[\u0600-\u06FF]", text))
    en = len(re.findall(r"[A-Za-z]", text))
    if ar == 0 and en == 0:
        return None
    return "ar" if ar >= en else "en"


def parse_json(text: str) -> dict:
    s = re.sub(r"```json|```", "", text).strip()
    return json.loads(s[s.index("{"): s.rindex("}") + 1])


def clean_history(history: list) -> list:
    """Keep the last 10 turns and make sure the list starts with a user turn."""
    h = history[-10:]
    while h and h[0]["role"] != "user":
        h = h[1:]
    return h


def get_whisper() -> Any:
    global _whisper
    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper is not available or isn't compatible with this Python version. "
            "Install a supported whisper build or disable voice transcription."
        )
    if _whisper is None:
        _whisper = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
    return _whisper


# ---------------------------------------------------------------- voice in / out
def _transcribe_sync(audio: bytes) -> dict:
    segments, info = get_whisper().transcribe(
        io.BytesIO(audio),
        task="transcribe",   # never translate: keep the user's original wording
        language=None,       # auto-detect Arabic / English
        vad_filter=True,
        beam_size=5,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return {"text": text, "language": info.language}


async def transcribe(audio: bytes) -> dict:
    """Speech -> text. Runs in a thread because Whisper is blocking."""
    return await asyncio.to_thread(_transcribe_sync, audio)


async def synthesize(text: str, lang: str) -> bytes:
    """Text -> MP3 bytes using Egyptian Arabic or English neural voices."""
    if edge_tts is None:
        raise RuntimeError("edge-tts is not installed.")
    communicate = edge_tts.Communicate(text, VOICES.get(lang, VOICES["en"]))
    buf = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.extend(chunk["data"])
    return bytes(buf)


# ---------------------------------------------------------------- offline fallback (no AI)
# Used only if the Claude API call fails. Same category ids as the AI path.
KW = [
    ("emergency", EMERGENCY_PATTERNS),
    ("pharmacy", re.compile(r"(صيدلية|صيدليه|دواء|دوا|pharmacy|medicine|drugstore|prescription)", re.I)),
    ("automotive", re.compile(r"(عربية|عربيه|سيارة|ونش|بنزين|كاوتش|موتور|ميكانيك|بطلت|\bcar\b|vehicle|tow|flat tyre|flat tire|engine|battery|mechanic|fuel|stalled|broke down)", re.I)),
    ("plumbing", re.compile(r"(ماسورة|ماسوره|مواسير|مياه|ميه|سباك|تسريب|حنفية|pipe|plumb|leak|water|faucet|drain|toilet)", re.I)),
    ("electrical", re.compile(r"(كهرب|النور|الكهربا|فيشة|ماس كهربائي|سلك|electric|power (cut|outage)|socket|short circuit|wiring|breaker)", re.I)),
    ("medical", re.compile(r"(دكتور|طبيب|مستشفى|عيان|تعبان|حرارة|وجع|ألم|إسعاف|doctor|hospital|sick|fever|pain|injur|ambulance|clinic)", re.I)),
    ("home_other", re.compile(r"(تكييف|مكيف|قفل|مفتاح|دهان|نجار|غسالة|تلاجة|ثلاجة|\bAC\b|air condition|lock|\bkey\b|paint|carpenter|appliance|fridge|washing machine)", re.I)),
]

FB = {
    "ar": {
        "automotive": ("ممكن تكون مشكلة في البطارية أو البنزين أو الموتور. ركن العربية على جنب في مكان آمن، شغّل الإشارات الأربعة، ومتفتحش غطا الموتور وهو سخن.", "العربية فين دلوقتي، وفيه صوت أو لمبة بتنور في العداد؟"),
        "medical": ("لو الحالة مش خطيرة، اتواصل مع دكتور أو أقرب مستشفى. أنا مش بشخّص، بس بوجّهك للخدمة المناسبة.", "إيه الأعراض بالظبط ومن امتى؟"),
        "pharmacy": ("تمام، هدورلك على أقرب صيدلية.", "إنت في أنهي منطقة؟ وتحتاج دوا معين؟"),
        "plumbing": ("اقفل محبس المياه الرئيسي فوراً، وافصل الكهرباء عن أي حاجة قريبة من المياه، ولمّ المياه بفوط.", "الكسر في ماسورة جوه الحيطة ولا ظاهرة؟ وإنت في أنهي منطقة؟"),
        "electrical": ("متلمسش أي أسلاك مكشوفة. افصل القاطع الرئيسي لو فيه ريحة حريق أو شرار.", "النور فاصل في الشقة كلها ولا أوضة واحدة؟"),
        "home_other": ("هوصلك بفني متخصص.", "المشكلة بالظبط إيه، وفي أنهي منطقة؟"),
        "emergency": ("اتصل فوراً: إسعاف 123، شرطة 122، إطفاء 180. ابعد عن مصدر الخطر وخليك مع المصاب.", ""),
        "unknown": ("مش متأكد فهمت المشكلة كويس.", "ممكن توضحلي أكتر؟ هل هي عن العربية، حاجة طبية، ولا صيانة في البيت؟"),
    },
    "en": {
        "automotive": ("Could be the battery, fuel or engine. Pull over safely, turn on your hazard lights, and don't open a hot hood.", "Where are you right now, and are any warning lights or noises showing?"),
        "medical": ("If it isn't an emergency, contact a doctor or the nearest hospital. I can't diagnose, but I'll point you to the right service.", "What are the symptoms, and since when?"),
        "pharmacy": ("Sure, I'll look for the nearest pharmacy.", "Which area are you in, and do you need a specific medicine?"),
        "plumbing": ("Shut the main water valve right away, switch off electricity near the water, and soak up spills with towels.", "Is the pipe exposed or inside a wall? And which area are you in?"),
        "electrical": ("Don't touch exposed wires. Switch off the main breaker if you smell burning or see sparks.", "Is the whole flat without power or just one room?"),
        "home_other": ("I'll connect you with a specialist technician.", "What exactly is the problem, and which area are you in?"),
        "emergency": ("Call now: ambulance 123, police 122, fire 180. Move away from danger and stay with the person.", ""),
        "unknown": ("I'm not sure I understood the problem.", "Could you tell me more? Is it about your car, a medical issue, or home maintenance?"),
    },
}


def local_fallback(text: str, lang: str) -> dict:
    cat = next((k for k, rx in KW if rx.search(text)), "unknown")
    reply, followup = FB[lang][cat]
    return {"category": cat, "emergency": cat == "emergency", "reply": reply, "followup": followup}


# ---------------------------------------------------------------- core chat
async def chat(message: str, history: Optional[list] = None, via_voice: bool = False) -> dict:
    history = history or []
    hint = detect_script(message) or "en"
    tag = f"[detected script: {hint}]" + (" [input: voice]" if via_voice else "")
    try:
        if GROQ_API_KEY:
            async with httpx.AsyncClient(timeout=60.0) as http_client:
                response = await http_client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": MODEL,
                        "messages": [
                            {"role": "system", "content": SYSTEM},
                            *clean_history(history),
                            {"role": "user", "content": f"{message}\n\n{tag}"},
                        ],
                        "temperature": 0.2,
                    },
                )
                response.raise_for_status()
                payload = response.json()
                raw = payload["choices"][0]["message"]["content"]
                j = parse_json(raw)
        elif client is not None:
            res = await client.messages.create(
                model=MODEL,
                max_tokens=700,
                system=SYSTEM,
                messages=[*clean_history(history), {"role": "user", "content": f"{message}\n\n{tag}"}],
            )
            j = parse_json("".join(b.text for b in res.content if b.type == "text"))
        else:
            raise RuntimeError("No LLM API key is set. Set GROQ_API_KEY or ANTHROPIC_API_KEY before using AI responses.")

        lang = "ar" if j.get("lang") == "ar" else "en"
        category = j.get("category") if j.get("category") in CATEGORIES else "unknown"
        emergency = bool(j.get("emergency")) or category == "emergency"
        reply = j.get("reply") or ""
        followup = j.get("followup") or ""

        # Safety net: keyword hit the model missed -> force emergency and add the numbers.
        if not emergency and EMERGENCY_PATTERNS.search(message):
            emergency, category = True, "emergency"
            reply = f"{EMERGENCY_LINE[lang]} {reply}".strip()

        out = {
            "lang": lang,
            "dir": "rtl" if lang == "ar" else "ltr",
            "category": category,
            "emergency": emergency,
            "reply": reply,
            "followup": followup,
            "mode": "ai",
        }
        out["history"] = [
            *history,
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"{reply} {followup}".strip()},
        ]
        return out
    except Exception as e:  # noqa: BLE001
        print("CitySOS AI error:", repr(e))
        j = local_fallback(message, hint)
        note = "(وضع بسيط بدون الذكاء الاصطناعي)" if hint == "ar" else "(basic mode, no AI)"
        reply = f"{j['reply']}\n{note}"
        return {
            "lang": hint,
            "dir": "rtl" if hint == "ar" else "ltr",
            "category": j["category"],
            "emergency": j["emergency"],
            "reply": reply,
            "followup": j["followup"],
            "mode": "local",
            "history": [
                *history,
                {"role": "user", "content": message},
                {"role": "assistant", "content": f"{reply} {j['followup']}".strip()},
            ],
        }


# ---------------------------------------------------------------- API
app = FastAPI(title="CitySOS Assistant")
sessions: dict = {}  # use Redis / DB in production
INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")


@app.get("/")
async def home():
    return FileResponse(INDEX)


@app.post("/api/chat")
async def api_chat(payload: dict):
    session_id, message = payload.get("sessionId"), (payload.get("message") or "").strip()
    if not session_id or not message:
        raise HTTPException(400, "sessionId and message are required")
    r = await chat(message, sessions.get(session_id, []))
    sessions[session_id] = r.pop("history")
    return r


@app.post("/api/voice")
async def api_voice(
    sessionId: str = Form(...),
    audio: UploadFile = File(...),            # webm / ogg / mp3 / wav / m4a
    replyWithVoice: bool = Form(True),        # set false to get text only
):
    data = await audio.read()
    if not data:
        raise HTTPException(400, "empty audio")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "audio too large (max 10 MB)")

    try:
        stt = await transcribe(data)
    except Exception as e:  # noqa: BLE001
        print("STT error:", repr(e))
        if WhisperModel is None:
            raise HTTPException(503, "voice transcription is unavailable in this environment. Please use text chat instead.")
        raise HTTPException(422, "could not read the audio")

    if not stt["text"]:
        lang = "ar" if stt["language"] == "ar" else "en"
        msg = "مسمعتش كلامك كويس، ممكن تعيد تاني؟" if lang == "ar" else "I couldn't hear you clearly. Could you repeat that?"
        return {"transcript": "", "lang": lang, "dir": "rtl" if lang == "ar" else "ltr",
                "category": "unknown", "emergency": False, "reply": msg, "followup": "", "mode": "ai", "audio": None}

    if WhisperModel is None:
        raise HTTPException(503, "voice transcription is unavailable in this environment. Please use text chat instead.")

    r = await chat(stt["text"], sessions.get(sessionId, []), via_voice=True)
    sessions[sessionId] = r.pop("history")
    r["transcript"] = stt["text"]  # show it so the user can spot recognition mistakes

    r["audio"] = None
    if replyWithVoice:
        try:
            spoken = f"{r['reply']} {r['followup']}".strip()
            mp3 = await synthesize(spoken, r["lang"])
            r["audio"] = {"mime": "audio/mpeg", "base64": base64.b64encode(mp3).decode()}
        except Exception as e:  # noqa: BLE001
            print("TTS error:", repr(e))  # fall back to text only
    return r


# ---------------------------------------------------------------- frontend example (JS)
# Record:
#   const rec = new MediaRecorder(await navigator.mediaDevices.getUserMedia({audio:true}));
#   const chunks = []; rec.ondataavailable = e => chunks.push(e.data);
#   rec.onstop = async () => {
#     const fd = new FormData();
#     fd.append("sessionId", sessionId);
#     fd.append("audio", new Blob(chunks, {type: rec.mimeType}), "voice.webm");
#     const r = await (await fetch("/api/voice", {method: "POST", body: fd})).json();
#     // show r.transcript, r.reply + r.followup, set dir = r.dir
#     if (r.audio) new Audio(`data:${r.audio.mime};base64,${r.audio.base64}`).play();
#   };
