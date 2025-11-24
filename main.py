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
GPT_API_KEY = os.getenv("GPT_API_KEY")
GPT_MODEL_NAME = os.getenv("GPT_MODEL_NAME", "gpt-5.1")

USE_GPT_API = bool(GPT_API_KEY)

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "200"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
TOP_P = float(os.getenv("TOP_P", "0.95"))

MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))

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
        "You MUST use the following RAG information to answer the user's question as accurately as possible.\n"
        "Even if the information is short, you should still generate a helpful answer.\n"
        "If the RAG context clearly contains no related information, respond politely.\n\n"
        f"RAG Information:\n{context}"
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

        try:
            resp = gpt_client.chat.completions.create(
                model=GPT_MODEL_NAME,
                messages=messages,
                stream=True,
                max_completion_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )

            for chunk in resp:
                if await request.is_disconnected():
                    break

                delta = chunk.choices[0].delta.content
                if delta:
                    tokens += len(delta)
                    yield sse(None, {"token": delta})

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
