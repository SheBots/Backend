# main.py
# FastAPI backend with SSE streaming + explicit startup checks for:
# 1) Local model loading (Transformers)
# 2) Optional RAG service health (if RAG_BASE_URL is set)

import os
import re
import json
import time
import threading
import logging
from dotenv import load_dotenv
from typing import List, Optional, Literal, Dict, Any, Generator

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

# -----------------------------
# Logging setup
# -----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("shebots-backend")

# -----------------------------
# Config (env with safe defaults)
# -----------------------------
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "http://localhost:5173")
MODEL_ID = os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "160"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
RAG_BASE_URL = os.getenv("RAG_BASE_URL", "http://127.0.0.1:8001").rstrip("/")  # e.g., http://localhost:8080

# Device & dtype (robust on CPU-only Windows too)
HAS_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if HAS_CUDA else "cpu"
PREF_DTYPE = os.getenv("TORCH_DTYPE", "float32").lower()
if PREF_DTYPE in {"bf16", "bfloat16"} and HAS_CUDA and torch.cuda.is_bf16_supported():
    DTYPE = torch.bfloat16
elif PREF_DTYPE in {"fp16", "float16"} and HAS_CUDA:
    DTYPE = torch.float16
else:
    DTYPE = torch.float32  # safest on CPU

# -----------------------------
# App & CORS
# -----------------------------
app = FastAPI(title="SheBots Backend (Transformers Streaming)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Global state flags
# -----------------------------
MODEL_READY: bool = False
MODEL_ERROR: Optional[str] = None
RAG_READY: Optional[bool] = True  # None = not checked, True/False after check
RAG_ERROR: Optional[str] = None

tokenizer = None
model = None

# -----------------------------
# Small helpers
# -----------------------------
def banner_line(ch="=", n=70) -> str:
    return ch * n

def redact(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", "[redacted-email]", text, flags=re.I)
    text = re.sub(r"\b(\+?\d[\d \-]{8,}\d)\b", "[redacted-phone]", text)
    return text

def sse(event: Optional[str], data: Dict[str, Any]) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data, ensure_ascii=False)}\n\n"

def build_messages(history: List[Dict[str, str]], user_text: str) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if not any(m.get("role") == "system" for m in history or []):
        msgs.append({"role": "system", "content": "You are a helpful assistant."})
    msgs.extend(history or [])
    msgs.append({"role": "user", "content": user_text})
    return msgs

def check_rag_health(timeout=5.0):
    """Ping RAG /rag/health if RAG_BASE_URL is defined. Sets RAG_READY & RAG_ERROR."""
    global RAG_READY, RAG_ERROR
    if not RAG_BASE_URL:
        RAG_READY = None
        RAG_ERROR = "RAG_BASE_URL not set"
        return
    try:
        import httpx  # optional; only needed if RAG is used
    except Exception as e:
        RAG_READY = False
        RAG_ERROR = f"httpx not installed: {e}"
        return
    try:
        url = f"{RAG_BASE_URL}/rag/health"
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                RAG_READY = bool(data.get("ok"))
                RAG_ERROR = None if RAG_READY else f"RAG unhealthy: {data}"
            else:
                RAG_READY = False
                RAG_ERROR = f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        RAG_READY = False
        RAG_ERROR = str(e)

def log_windows_symlink_tip():
    # Help users on Windows who see HF cache symlink warnings
    if os.name == "nt":
        log.info(
            "Windows tip: enable Developer Mode or run Python as Admin to allow "
            "symlinks for the HuggingFace cache (saves disk space). "
            "To silence the warning, set HF_HUB_DISABLE_SYMLINKS_WARNING=1."
        )

# -----------------------------
# Startup: load model and run checks
# -----------------------------
@app.on_event("startup")
def _load_resources():
    global tokenizer, model, MODEL_READY, MODEL_ERROR

    print(banner_line())
    print("SheBots Backend — Startup")
    print(banner_line("-"))
    print(f"Model: {MODEL_ID}")
    print(f"Device: {DEVICE} | dtype: {DTYPE} | CUDA: {HAS_CUDA}")
    print(f"RAG_BASE_URL: {RAG_BASE_URL or '(not set)'}")
    print(banner_line("-"))

    log_windows_symlink_tip()

    t0 = time.time()
    try:
        # Load tokenizer & model ONCE
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        # Use `dtype=` to avoid deprecation warning for torch_dtype
        # device_map="auto" works with accelerate; safe on single CPU/GPU boxes
        model_kwargs = dict(dtype=DTYPE, device_map="auto")
        # Some older transformers use torch_dtype; keep a fallback if dtype unsupported
        try:
            loaded = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
        except TypeError:
            log.warning("`dtype` not supported by current transformers — falling back to torch_dtype.")
            model_kwargs_fallback = dict(torch_dtype=DTYPE, device_map="auto")
            loaded = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs_fallback)
        model = loaded

        MODEL_READY = True
        MODEL_ERROR = None
        dt = time.time() - t0
        print(f"✅ Model loaded OK in {dt:.2f}s")
    except Exception as e:
        MODEL_READY = False
        MODEL_ERROR = str(e)
        print(f"❌ Model load FAILED: {MODEL_ERROR}")

    # RAG check (non-fatal)
    check_rag_health()
    if RAG_READY is True:
        print("✅ RAG health OK")
    elif RAG_READY is False:
        print(f"❌ RAG health FAIL: {RAG_ERROR}")
    else:
        print("ℹ️  RAG not configured (skipped)")

    print(banner_line())

# -----------------------------
# Schemas
# -----------------------------
Role = Literal["system", "user", "assistant"]

class Message(BaseModel):
    role: Role
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., description="User's latest message")
    history: Optional[List[Message]] = Field(default_factory=list)
    useDocs: Optional[bool] = Field(default=False)

# -----------------------------
# Routes
# -----------------------------
@app.get("/api/health")
async def health():
    return {
        "ok": bool(MODEL_READY),
        "model": MODEL_ID,
        "device": DEVICE,
        "dtype": str(DTYPE).split(".")[-1],
        "model_ready": MODEL_READY,
        "model_error": MODEL_ERROR,
        "rag_base_url": RAG_BASE_URL or None,
        "rag_ready": RAG_READY,
        "rag_error": RAG_ERROR,
    }

@app.post("/api/chat")
async def chat(body: ChatRequest, request: Request):
    if not MODEL_READY:
        return JSONResponse(
            {"error": f"Model not ready: {MODEL_ERROR or 'Unknown error'}"},
            status_code=503,
        )

    user_text = (body.message or "").strip()
    if not user_text:
        return JSONResponse({"error": "Message is required."}, status_code=400)
    if len(user_text) > 2000:
        return JSONResponse({"error": "Message too long (2000 chars max)."}, status_code=413)

    # Warn if RAG requested but not healthy (non-fatal; still answer without RAG)
    if body.useDocs and not RAG_READY:
        log.warning(f"RAG requested but not ready: {RAG_ERROR}")

    # Build messages and template
    msgs = [m.dict() for m in (body.history or [])]
    msgs = build_messages(msgs, redact(user_text))
    prompt_text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([prompt_text], return_tensors="pt").to(DEVICE)

    streamer = TextIteratorStreamer(
        tokenizer, skip_special_tokens=True, skip_prompt=True
    )

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Non-blocking generation
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    async def event_stream() -> Generator[str, None, None]:
        yield sse("start", {"model": MODEL_ID})
        tokens = 0
        try:
            for text in streamer:
                tokens += len(text)
                if await request.is_disconnected():
                    break
                yield sse(None, {"token": text})
            yield sse("end", {"tokensStreamed": tokens})
        except Exception as e:
            yield sse("error", {"error": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
