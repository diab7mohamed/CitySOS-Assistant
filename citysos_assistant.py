"""
citysos_assistant.py - CitySOS bilingual (Arabic / English) assistant with voice support.
IMPROVED VERSION - fixes: illogical AI replies + poor voice quality.

Main fixes vs the original:
  1. Stronger Groq model by default (gpt-oss-120b instead of the weak 20b).
  2. response_format=json_object so the model is FORCED to return valid JSON
     (the #1 cause of "illogical" replies: bad JSON -> silent local fallback).
  3. One retry with a stricter instruction if JSON parsing fails.
  4. temperature 0 for consistent, deterministic replies.
  5. Speech-to-text uses Groq's hosted whisper-large-v3-turbo API instead of local faster-whisper.
"""

import base64
import json
import os
import re
from typing import Optional

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
from fastapi.responses import FileResponse, Response

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# gpt-oss-20b is too weak for Arabic + strict JSON. 120b (or llama-3.3-70b-versatile)
# follows the JSON schema far more reliably -> logical, on-topic replies.
MODEL = (
    os.getenv("CITYSOS_MODEL")
    or os.getenv("GROQ_MODEL")
    or ("llama-3.3-70b-versatile" if GROQ_API_KEY else "claude-sonnet-5-5")
)
# If the main model fails (renamed / retired / rate limited) we try these in order.
GROQ_FALLBACK_MODELS = [m for m in ("llama-3.3-70b-versatile", "openai/gpt-oss-120b", "llama-3.1-8b-instant") if m != MODEL]
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3-turbo")
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MB

CATEGORIES = ["automotive", "medical", "pharmacy", "plumbing", "electrical", "home_other", "emergency", "unknown"]

SYSTEM = f"""You are CitySOS Assistant, a friendly, empathetic, and capable assistant helping people in Egypt. Speak naturally and warmly, acknowledge the user's concern when appropriate, and guide them through practical next steps one at a time. Use the conversation history to understand references and follow-ups, remember details already shared, and avoid asking again for information the user has provided. Be clear and concise; never claim to have contacted a service or taken an action you cannot actually take.
Users write or speak in Modern Standard Arabic, Egyptian colloquial Arabic (e.g. "العربية بطلت", "ماسورة مكسورة", "النور فصل"), English, or mixed Arabic-English. Work directly from the user's original wording; never translate it first. Speech-to-text may contain small recognition mistakes, so infer the intended meaning from context.

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
3. Answer the user's latest question in context. For multi-step help, give the safest and most useful next step first, then continue one step at a time. Keep advice practical and concise (2-5 short lines). Never diagnose medical conditions; recommend a qualified professional. For emergencies, lead with Egypt numbers: ambulance 123, police 122, fire 180, plus one immediate safety action.
4. Ask at most ONE relevant follow-up question, and only if needed to help with the next step (for example, location, car model, or severity). Do not repeat a question already answered in the conversation. Use "" if none is needed.
5. If the message came by voice, keep sentences short and avoid lists, symbols or emojis, because the reply will be read aloud.

Output ONLY compact JSON, no markdown, no text before or after the JSON:
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

VOICES = {"ar": "ar-EG-ShakirNeural", "en": "en-US-JennyNeural"}

HTTP = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))
API_KEY = ANTHROPIC_API_KEY
client = AsyncAnthropic(api_key=API_KEY) if AsyncAnthropic and API_KEY else None


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
    """Keep the complete conversation and make sure it starts with a user turn."""
    h = history[:]
    while h and h[0]["role"] != "user":
        h = h[1:]
    return h



# ---------------------------------------------------------------- LLM call (with retry)
async def _call_llm(core_messages: list) -> str:
    """Return the raw model output. Raises on HTTP/parse problems."""
    if GROQ_API_KEY:
        messages = [{"role": "system", "content": SYSTEM}, *core_messages]
        last_err = None
        for model in [MODEL, *GROQ_FALLBACK_MODELS]:
            body = {
                "model": model,
                "messages": messages,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            }
            if "gpt-oss" in model:
                body["reasoning_effort"] = "low"   # much faster, still good enough here
            else:
                body["max_tokens"] = 500
            try:
                response = await HTTP.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": "Bearer " + GROQ_API_KEY, "Content-Type": "application/json"},
                    json=body,
                )
                if response.status_code != 200:
                    # THIS line tells you the real reason (bad key, model removed, rate limit...)
                    print(f"[Groq LLM] model={model} status={response.status_code} body={response.text[:300]}")
                    last_err = RuntimeError(f"Groq {response.status_code}")
                    continue
                return response.json()["choices"][0]["message"]["content"]
            except httpx.HTTPError as e:
                print(f"[Groq LLM] model={model} network error: {e!r}")
                last_err = e
        raise last_err or RuntimeError("Groq call failed")
    if client is not None:
        res = await client.messages.create(
            model=MODEL,
            max_tokens=700,
            system=SYSTEM,
            messages=core_messages,
        )
        return "".join(b.text for b in res.content if b.type == "text")
    raise RuntimeError("No LLM API key is set. Set GROQ_API_KEY or ANTHROPIC_API_KEY before using AI responses.")


# ---------------------------------------------------------------- voice in / out
async def transcribe(audio: bytes, filename: str = "voice.webm", content_type: str = "audio/webm") -> dict:
    """Speech -> text via Groq's hosted Whisper."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is required for voice input.")
    r = await HTTP.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": "Bearer " + GROQ_API_KEY},
        files={"file": (filename, audio, content_type)},
        data={
            "model": STT_MODEL,
            "temperature": "0",
            "response_format": "json",
            # Context words help Whisper with Egyptian dialect + mixed Arabic/English.
            "prompt": "مساعد CitySOS بالعامية المصرية: العربية بطلت، ونش، ماسورة مكسورة، النور فصل، صيدلية، إسعاف.",
        },
    )
    if r.status_code != 200:
        print(f"[Groq STT] status={r.status_code} body={r.text[:300]}")
    r.raise_for_status()
    text = (r.json().get("text") or "").strip()
    return {"text": text, "language": detect_script(text) or "ar"}


def clean_for_speech(text: str) -> str:
    text = re.sub(r"[\U00010000-\U0010ffff\u2600-\u27BF\uFE0F]", "", text)   # emojis
    text = re.sub(r"[*_`#>|~]+", " ", text)                                    # markdown
    text = re.sub(r"\(.*?(basic mode|وضع بسيط).*?\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:450]


async def synthesize(text: str, lang: str) -> bytes:
    """Text -> MP3 bytes using Egyptian Arabic or English neural voices."""
    if edge_tts is None:
        raise RuntimeError("edge-tts is not installed.")
    text = clean_for_speech(text)
    if not text:
        raise RuntimeError("nothing to speak")
    communicate = edge_tts.Communicate(text, VOICES.get(lang, VOICES["en"]))
    buf = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.extend(chunk["data"])
    if not buf:
        raise RuntimeError("TTS returned no audio")
    return bytes(buf)


# ---------------------------------------------------------------- offline fallback (no AI)
# Used only if the LLM call fails. Same category ids as the AI path.
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
    core_messages = [*clean_history(history), {"role": "user", "content": f"{message}\n\n{tag}"}]
    try:
        raw = await _call_llm(core_messages)
        try:
            j = parse_json(raw)
        except (json.JSONDecodeError, ValueError, IndexError):
            # One retry with an explicit stricter nudge before giving up.
            raw = await _call_llm([
                *core_messages,
                {"role": "user", "content": "Your previous answer was not valid JSON. Reply with ONLY the compact JSON object now, no markdown, no extra text."},
            ])
            j = parse_json(raw)

        lang = "ar" if j.get("lang") == "ar" else "en"
        category = j.get("category") if j.get("category") in CATEGORIES else "unknown"
        emergency = bool(j.get("emergency")) or category == "emergency"
        reply = (j.get("reply") or "").strip()
        followup = (j.get("followup") or "").strip()
        if not reply:
            raise ValueError("empty reply from model")

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
INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index (1).html")


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
        stt = await transcribe(data, audio.filename or "voice.webm", audio.content_type or "audio/webm")
    except Exception as e:  # noqa: BLE001
        print("STT error:", repr(e))
        raise HTTPException(422, "could not read the audio. Please try again or use text chat instead.")

    if not stt["text"]:
        lang = "ar" if stt["language"] == "ar" else "en"
        msg = "مسمعتش كلامك كويس، ممكن تعيد تاني؟" if lang == "ar" else "I couldn't hear you clearly. Could you repeat that?"
        return {"transcript": "", "lang": lang, "dir": "rtl" if lang == "ar" else "ltr",
                "category": "unknown", "emergency": False, "reply": msg, "followup": "", "mode": "ai", "audio": None}

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


@app.post("/api/tts")
async def api_tts(payload: dict):
    """Text -> MP3. The page calls this AFTER showing the text, so replies appear instantly."""
    text, lang = (payload.get("text") or "").strip(), payload.get("lang", "en")
    if not text:
        raise HTTPException(400, "text is required")
    try:
        mp3 = await synthesize(text, "ar" if lang == "ar" else "en")
    except Exception as e:  # noqa: BLE001
        print("TTS error:", repr(e))
        raise HTTPException(502, "tts failed")
    return Response(content=mp3, media_type="audio/mpeg")
