from fastapi import Request
from app.booking_chat import   handle_booking
import json
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional
from app.routers.appointments import router as appointments_router
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from app.voice_service import get_answer, transcribe_audio


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
LOG_FILE = BASE_DIR / "latency.log"
HISTORY_FILE = BASE_DIR / "latency_history.jsonl"

STATIC_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("voiceassistant")

app = FastAPI(
    title="Voice Assistant API",
    description="Text and voice assistant with latency tracking.",
    version="2.0.0",
)
app.include_router(appointments_router)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

history_lock = Lock()
metrics_lock = Lock()
recent_requests = deque(maxlen=10)
request_count = 0
first_request_total: Optional[float] = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    session_id: str = Field(min_length=1, max_length=100)

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_history(record: dict) -> None:
    with history_lock:
        with HISTORY_FILE.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_recent_history(limit: int = 20) -> list:
    if not HISTORY_FILE.exists():
        return []

    with history_lock:
        with HISTORY_FILE.open("r", encoding="utf-8") as file:
            lines = file.readlines()

    records = []
    for line in lines[-limit:]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return list(reversed(records))


def record_request(
    mode: str,
    stt_seconds: float,
    llm_seconds: float,
    total_seconds: float,
) -> dict:
    global request_count, first_request_total

    with metrics_lock:
        request_count += 1

        if first_request_total is None:
            first_request_total = total_seconds

        recent_requests.append(
            {
                "stt_seconds": stt_seconds,
                "llm_seconds": llm_seconds,
                "total_seconds": total_seconds,
            }
        )

        recent = list(recent_requests)
        count = len(recent)

        averages = {
            "stt_seconds": sum(x["stt_seconds"] for x in recent) / count,
            "llm_seconds": sum(x["llm_seconds"] for x in recent) / count,
            "total_seconds": sum(x["total_seconds"] for x in recent) / count,
        }

        current_count = request_count
        first_total = first_request_total

    metrics = {
        "mode": mode,
        "stt_seconds": round(stt_seconds, 3),
        "llm_seconds": round(llm_seconds, 3),
        # TTS runs in the browser, not on this server.
        "tts_seconds": None,
        "total_seconds": round(total_seconds, 3),
        "first_request_total_seconds": round(first_total, 3),
        "request_count": current_count,
        "average_last_10": {
            key: round(value, 3) for key, value in averages.items()
        },
        "average_sample_count": count,
    }

    logger.info("Request metrics: %s", json.dumps(metrics))
    append_history({"timestamp": utc_now(), **metrics})
    return metrics



def run_answer(
    message: str,
    mode: str,
    session_id: str,
    stt_seconds: float = 0.0,
) -> dict:
    start = time.perf_counter()

    try:
        llm_start = time.perf_counter()

        # Booking conversation gets priority; other messages use the LLM.
        answer = handle_booking(
            message=message,
            session_id=session_id,
            mode=mode,
        )

        if answer is None:
            answer = get_answer(message)

        llm_seconds = time.perf_counter() - llm_start

    except Exception as exc:
        logger.exception("Answer generation failed")
        raise HTTPException(
            status_code=502,
            detail=f"Answer generation failed: {exc}",
        ) from exc

    # Keep the existing latency calculations and metrics schema.
    total_seconds = time.perf_counter() - start + stt_seconds

    metrics = record_request(
        mode=mode,
        stt_seconds=stt_seconds,
        llm_seconds=llm_seconds,
        total_seconds=total_seconds,
    )

    return {
        "answer": answer,
        "metrics": metrics,
    }


@app.get("/")
def home():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="static/index.html not found")
    return FileResponse(index_file)


@app.get("/health")
def health():
    return {"status": "ok", "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")}



@app.post("/chat")
def chat(payload: ChatRequest):
    message = payload.message.strip()
    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty",
        )

    return run_answer(
        message=message,
        mode="text",
        session_id=payload.session_id,
        stt_seconds=0.0,
    )


@app.post("/voice")
async def voice(
    audio: UploadFile = File(...),
    session_id: str = Form(...),
):
    allowed_types = {
        "audio/webm",
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp4",
        "audio/ogg",
        "video/webm",
        "application/octet-stream",
    }

    if audio.content_type and audio.content_type not in allowed_types:
        logger.info("Received audio content type: %s", audio.content_type)

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio recording is empty")

    if len(audio_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file is too large (20 MB max)")

    from io import BytesIO

    audio_stream = BytesIO(audio_bytes)
    audio_stream.name = audio.filename or "recording.webm"

    stt_start = time.perf_counter()
    try:
        transcript = transcribe_audio(audio_stream)
    except Exception as exc:
        logger.exception("Audio transcription failed")
        raise HTTPException(
            status_code=502,
            detail=f"Speech-to-text failed: {exc}",
        ) from exc

    stt_seconds = time.perf_counter() - stt_start

    if not transcript:
        raise HTTPException(
            status_code=400,
            detail="No speech detected. Please speak clearly and try again.",
        )

    result = run_answer(
    message=transcript,
    mode="voice",
    session_id=session_id,
    stt_seconds=stt_seconds,
)
    result["transcript"] = transcript
    return result

@app.post("/browser-tts-log")
async def browser_tts_log(payload: dict):
    """
    Browser sends these events:
    - start_delay: time from asking browser to speak until speech starts
    - playback_complete: duration of browser speech playback
    """
    event = payload.get("event")

    if event not in {"start_delay", "playback_complete"}:
        raise HTTPException(status_code=400, detail="Unknown TTS event")

    try:
        seconds = float(payload.get("seconds", 0))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="seconds must be numeric") from exc

    if seconds < 0 or seconds > 3600:
        raise HTTPException(status_code=400, detail="Invalid TTS duration")

    record = {
        "timestamp": utc_now(),
        "mode": "browser_tts",
        "event": event,
        "seconds": round(seconds, 3),
    }
    logger.info("Browser TTS: %s", json.dumps(record))
    append_history(record)
    return {"status": "logged"}


@app.get("/api/latency-history")
def latency_history(limit: int = 20):
    limit = max(1, min(limit, 100))
    return {"items": get_recent_history(limit)}


@app.get("/api/metrics")
def metrics_summary():
    with metrics_lock:
        recent = list(recent_requests)
        count = request_count
        first_total = first_request_total

    n = len(recent)
    if n:
        averages = {
            "stt_seconds": sum(x["stt_seconds"] for x in recent) / n,
            "llm_seconds": sum(x["llm_seconds"] for x in recent) / n,
            "total_seconds": sum(x["total_seconds"] for x in recent) / n,
        }
    else:
        averages = {
            "stt_seconds": None,
            "llm_seconds": None,
            "total_seconds": None,
        }

    return {
        "request_count": count,
        "first_request_total_seconds": first_total,
        "average_last_10": averages,
        "average_sample_count": n,
        "note": "Browser TTS timings are logged separately; TTS is not executed on the server.",
    }
