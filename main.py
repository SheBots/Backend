# main.py — SheBots Backend (GPT-5.1 + Human-friendly RAG Behavior)

import os
import re
import json
import logging
import httpx
from dotenv import load_dotenv
from typing import List, Optional, Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

load_dotenv()

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("backend")

# -----------------------------
# CONFIG
# -----------------------------
RAG_BASE_URL = os.getenv("RAG_BASE_URL", "http://127.0.0.1:8001")
GPT_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_MODEL_NAME = os.getenv("GPT_MODEL_NAME", "gpt-5.1")

USE_GPT_API = bool(GPT_API_KEY)

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "200"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
TOP_P = float(os.getenv("TOP_P", "0.95"))

MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "15000"))
TARGET_CHAR_MIN = int(os.getenv("TARGET_CHAR_MIN", "300"))
TARGET_CHAR_MAX = int(os.getenv("TARGET_CHAR_MAX", "400"))

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="SheBots Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("ALLOWED_ORIGIN", "*")],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_READY = False
MODEL_ERROR = None
RAG_READY = None
RAG_ERROR = None

gpt_client = None

# -----------------------------
# Helpers
# -----------------------------
def redact(text: str):
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[email]", text)
    return re.sub(r"\+?\d[\d\- ]{8,}\d", "[phone]", text)


def sse(event, data: dict):
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data, ensure_ascii=False)}\n\n"


def reduce_rag_context(chunks, max_chars=MAX_CONTEXT_CHARS):
    cleaned = []
    seen = set()
    total = 0

    for c in chunks:
        c = c.strip()
        if not c or c in seen:
            continue

        if total + len(c) > max_chars:
            break

        cleaned.append(c)
        seen.add(c)
        total += len(c)

    return cleaned


def looks_incomplete(text: str) -> bool:
    """
    Heuristic: consider incomplete if it doesn't end with common terminal punctuation
    and has a dangling last sentence fragment.
    """
    if not text:
        return False
    trimmed = text.strip()
    # Ends with a terminal punctuation?
    if re.search(r"[.!?…]$", trimmed):
        return False
    # Ends with obvious cut markers or list starters
    if re.search(r"(?:\b(?:and|or|but|because|so|which)\b$)|[-–:]$", trimmed, re.IGNORECASE):
        return True
    # Too short to judge? Treat as complete.
    if len(trimmed) < 40:
        return False
    # Default: if last line seems truncated (no punctuation in last 100 chars)
    last = trimmed[-120:]
    return not re.search(r"[.!?]$", last)


def check_rag_health():
    global RAG_READY, RAG_ERROR
    try:
        with httpx.Client(timeout=3.0) as c:
            r = c.get(f"{RAG_BASE_URL}/rag/health")
            if r.status_code == 200:
                RAG_READY = r.json().get("ok", False)
                RAG_ERROR = None
            else:
                RAG_READY = False
                RAG_ERROR = f"HTTP {r.status_code}"
    except Exception as e:
        RAG_READY = False
        RAG_ERROR = str(e)


def fetch_rag_context(query: str, k: int = 5):
    if not RAG_READY:
        return []

    url = f"{RAG_BASE_URL}/rag/search"
    params = {"query": query, "k": k}

    try:
        with httpx.Client(timeout=6.0) as c:
            r = c.get(url, params=params)
            if r.status_code != 200:
                r = c.post(url, json=params)

            if r.status_code != 200:
                return []

            docs = []
            for item in r.json().get("results", []):
                txt = item.get("text") or item.get("content") or ""
                if txt:
                    docs.append(txt)

            return docs

    except:
        return []


# -----------------------------
# Startup
# -----------------------------
@app.on_event("startup")
def _startup():
    global MODEL_READY, MODEL_ERROR, gpt_client

    print("=" * 60)
    print("Backend starting…")
    print("=" * 60)

    if USE_GPT_API:
        try:
            gpt_client = OpenAI(api_key=GPT_API_KEY)
            MODEL_READY = True
            print("Using GPT API:", GPT_MODEL_NAME)
        except Exception as e:
            MODEL_READY = False
            MODEL_ERROR = str(e)

    else:
        MODEL_READY = True

    check_rag_health()
    print("RAG:", "OK" if RAG_READY else f"FAIL ({RAG_ERROR})")
    print("=" * 60)


# -----------------------------
# Schemas
# -----------------------------
Role = Literal["system", "user", "assistant"]

class Message(BaseModel):
    role: Role
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = []
    useDocs: Optional[bool] = True


# -----------------------------
# Health
# -----------------------------
@app.get("/api/health")
async def health():
    return {
        "ok": MODEL_READY,
        "model": GPT_MODEL_NAME if USE_GPT_API else None,
        "rag_ready": RAG_READY,
        "rag_error": RAG_ERROR,
    }


# -----------------------------
# CHAT (SSE STREAMING)
# -----------------------------
@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):

    if not MODEL_READY:
        return JSONResponse({"error": MODEL_ERROR}, 503)

    if not RAG_READY:
        return JSONResponse({"error": "RAG service unavailable."}, 503)

    text = req.message.strip()
    if not text:
        return JSONResponse({"error": "Empty message"}, 400)

    # -----------------------------
    # Fetch RAG context
    # -----------------------------
    rag_chunks = reduce_rag_context(fetch_rag_context(text))

    if len(rag_chunks) == 0:
        # FINAL FALLBACK — Completely missing context
        fallback_message = (
            "I couldn’t get enough information about this. "
            "Please try checking the notice section of the website for further information."
        )
        return JSONResponse({"assistant": fallback_message}, 200)

    # -----------------------------
    # Build system prompt
    # -----------------------------
    context = "\n".join(f"- {c}" for c in rag_chunks)

    sysmsg = (
        "You are a helpful academic assistant for the KNU Computer Science department.\n"
        "Use the provided information below to answer accurately, but DO NOT mention sources like 'RAG' or 'documents'.\n"
        "When referencing the information, speak as yourself (e.g., 'According to my information...').\n"
        "Adopt a polite, self-aware tone: if unsure, say 'I might be mistaken—based on my information...' and invite correction.\n"
        "If the provided information clearly lacks relevant details, respond politely without claiming external retrieval.\n"
        "Write your final answer in Korean and aim for a concise response of 300–400 characters.\n\n"
        f"Information:\n{context}"
    )

    messages = [{"role": "system", "content": sysmsg}]
    for h in req.history:
        messages.append({"role": h.role, "content": h.content})
    messages.append({"role": "user", "content": redact(text)})

    # -----------------------------
    # GPT-5.1 Streaming
    # -----------------------------
    async def stream_gpt():
        yield sse("start", {"model": GPT_MODEL_NAME})
        tokens = 0
        full_text = []

        try:
            resp = gpt_client.chat.completions.create(
                model=GPT_MODEL_NAME,
                messages=messages,
                stream=True,
                max_completion_tokens=max(MAX_NEW_TOKENS, 600),
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )

            for chunk in resp:
                if await request.is_disconnected():
                    break

                delta = chunk.choices[0].delta.content
                if delta:
                    tokens += len(delta)
                    full_text.append(delta)
                    yield sse(None, {"token": delta})
            current = "".join(full_text)

            # If looks incomplete, ask the model to continue and stream the remainder (non-streaming for simplicity)
            if looks_incomplete(current):
                continuation = gpt_client.chat.completions.create(
                    model=GPT_MODEL_NAME,
                    messages=messages + [
                        {"role": "assistant", "content": current},
                        {"role": "user", "content": "Please continue and finish your answer."},
                    ],
                    max_completion_tokens=256,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    stream=False,
                )
                extra = continuation.choices[0].message.content or ""
                if extra:
                    tokens += len(extra)
                    # Send the remainder as one token event
                    yield sse(None, {"token": extra})
                    current += extra

            # If too long, rewrite concisely to target Korean character range
            if len(current) > TARGET_CHAR_MAX:
                rewrite = gpt_client.chat.completions.create(
                    model=GPT_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "Rewrite the assistant message in Korean within 300–400 characters, preserving meaning and tone. Avoid mentioning sources; speak as 'my information'."},
                        {"role": "user", "content": current},
                    ],
                    max_completion_tokens=300,
                    temperature=max(0.2, TEMPERATURE - 0.1),
                    top_p=TOP_P,
                    stream=False,
                )
                concise = rewrite.choices[0].message.content or ""
                if concise:
                    yield sse("rewrite", {"original_len": len(current), "new_len": len(concise)})
                    # Emit concise version (client can replace displayed text on 'rewrite' event)
                    yield sse(None, {"token": concise})

            yield sse("end", {"tokens": tokens})

        except Exception as e:
            yield sse("error", {"error": str(e)})

    return StreamingResponse(stream_gpt(), media_type="text/event-stream")


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")), reload=True)
