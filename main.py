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
GPT_MODEL_NAME = os.getenv("GPT_MODEL_NAME", "gpt-4.1")

USE_GPT_API = bool(GPT_API_KEY)

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "200"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
TOP_P = float(os.getenv("TOP_P", "0.95"))

MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "8000"))
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


def detect_language(text: str) -> Literal["ko", "en", "mixed"]:
    """
    Detect if text is primarily Korean, English, or mixed.
    """
    korean_chars = len(re.findall(r'[가-힣]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    
    total = korean_chars + english_chars
    if total == 0:
        return "en"
    
    korean_ratio = korean_chars / total
    
    if korean_ratio > 0.7:
        return "ko"
    elif korean_ratio < 0.3:
        return "en"
    else:
        return "mixed"


def detect_translation_request(text: str) -> Optional[str]:
    """
    Check if user explicitly requests translation or bilingual response.
    Returns 'both', 'en', 'ko', or None
    """
    text_lower = text.lower()
    
    # Bilingual requests
    if any(phrase in text_lower for phrase in [
        "in both", "양쪽", "둘 다", "both korean and english", "english and korean",
        "both languages", "두 언어"
    ]):
        return "both"
    
    # Translation to English
    if any(phrase in text_lower for phrase in [
        "translate to english", "in english", "영어로", "translate it to english",
        "answer in english"
    ]):
        return "en"
    
    # Translation to Korean
    if any(phrase in text_lower for phrase in [
        "translate to korean", "in korean", "한국어로", "translate it to korean",
        "answer in korean"
    ]):
        return "ko"
    
    return None


async def translate_text(text: str, target_lang: str, gpt_client) -> str:
    """
    Translate text to target language using GPT.
    """
    try:
        prompt = f"Translate the following text to {target_lang}. Only provide the translation, no explanations:\n\n{text}"
        
        response = gpt_client.chat.completions.create(
            model=GPT_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=800,
            temperature=0.3,
            stream=False,
        )
        
        return response.choices[0].message.content or text
    except:
        return text


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


def fetch_rag_context(query: str, k: int = 3):
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
    language: Optional[Literal["ko", "en", "auto"]] = "auto"
    context_provided: Optional[bool] = False  # Track if context was already sent


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
    # Language Detection
    # -----------------------------
    detected_lang = detect_language(text)
    translation_req = detect_translation_request(text)
    
    # Determine response language
    if req.language == "auto":
        response_lang = detected_lang if detected_lang != "mixed" else "ko"
    else:
        response_lang = req.language
    
    # Override if explicit translation request
    if translation_req:
        response_lang = translation_req

    # -----------------------------
    # Smart Context Management (only fetch if needed)
    # -----------------------------
    # Only provide context on first message or if explicitly not provided before
    provide_context = not req.context_provided or len(req.history) == 0
    
    if provide_context:
        rag_chunks = reduce_rag_context(fetch_rag_context(text))

        if len(rag_chunks) == 0:
            # FINAL FALLBACK — Completely missing context
            fallback_message = (
                "I couldn't get enough information about this. "
                "Please try checking the notice section of the website for further information."
            )
            if response_lang == "ko":
                fallback_message = "이에 대한 충분한 정보를 찾을 수 없습니다. 웹사이트의 공지사항을 확인해 주세요."
            elif response_lang == "both":
                fallback_message = (
                    "**Korean:**\n이에 대한 충분한 정보를 찾을 수 없습니다. 웹사이트의 공지사항을 확인해 주세요.\n\n"
                    "**English:**\nI couldn't get enough information about this. Please try checking the notice section of the website for further information."
                )
            
            return JSONResponse({"assistant": fallback_message}, 200)

        # Build context only when providing it
        context = "\n".join(f"- {c}" for c in rag_chunks)
    else:
        context = None

    # -----------------------------
    # Build system prompt based on language and context availability
    # -----------------------------
    if response_lang == "ko" or (response_lang == "mixed" and detected_lang != "en"):
        if provide_context:
            sysmsg = (
                "You are a helpful academic assistant for the KNU Computer Science department.\n"
                "Use ONLY the provided information to answer. DO NOT mention sources.\n"
                "**CRITICAL: Your answer must be 300-400 characters in Korean. Be direct and concise.**\n"
                "Speak as yourself (e.g., '제 정보에 따르면...'). If unsure, say '제 정보가 틀릴 수 있습니다만...'.\n\n"
                f"Information:\n{context}"
            )
        else:
            sysmsg = (
                "You are a helpful academic assistant for the KNU Computer Science department.\n"
                "Use the information provided earlier in the conversation to answer.\n"
                "**CRITICAL: Your answer must be 300-400 characters in Korean. Be direct and concise.**\n"
                "Speak as yourself (e.g., '제 정보에 따르면...'). If unsure, say '제 정보가 틀릴 수 있습니다만...'.\n"
            )
    elif response_lang == "en":
        if provide_context:
            sysmsg = (
                "You are a helpful academic assistant for the KNU Computer Science department.\n"
                "Use ONLY the provided information to answer in English. DO NOT mention sources.\n"
                "**CRITICAL: Your answer must be 150-200 words maximum. Be direct and concise.**\n"
                "Speak as yourself (e.g., 'According to my information...'). If unsure, acknowledge it.\n\n"
                f"Information:\n{context}"
            )
        else:
            sysmsg = (
                "You are a helpful academic assistant for the KNU Computer Science department.\n"
                "Use the information provided earlier in the conversation to answer in English.\n"
                "**CRITICAL: Your answer must be 150-200 words maximum. Be direct and concise.**\n"
                "Speak as yourself (e.g., 'According to my information...'). If unsure, acknowledge it.\n"
            )
    else:  # both languages
        if provide_context:
            sysmsg = (
                "You are a helpful academic assistant for the KNU Computer Science department.\n"
                "Use ONLY the provided information to answer in BOTH Korean and English.\n"
                "**CRITICAL: Format as:**\n"
                "**Korean:** [300-400 characters max]\n\n**English:** [150-200 words max]\n\n"
                "DO NOT mention sources. Be direct and concise.\n\n"
                f"Information:\n{context}"
            )
        else:
            sysmsg = (
                "You are a helpful academic assistant for the KNU Computer Science department.\n"
                "Use the information provided earlier to answer in BOTH Korean and English.\n"
                "**CRITICAL: Format as:**\n"
                "**Korean:** [300-400 characters max]\n\n**English:** [150-200 words max]\n\n"
                "DO NOT mention sources. Be direct and concise.\n"
            )

    messages = [{"role": "system", "content": sysmsg}]
    for h in req.history:
        messages.append({"role": h.role, "content": h.content})
    messages.append({"role": "user", "content": redact(text)})

    # -----------------------------
    # GPT Streaming (optimized - no continuation)
    # -----------------------------
    async def stream_gpt():
        yield sse("start", {"model": GPT_MODEL_NAME, "language": response_lang, "context_provided": provide_context})
        tokens = 0
        full_text = []

        try:
            resp = gpt_client.chat.completions.create(
                model=GPT_MODEL_NAME,
                messages=messages,
                stream=True,
                max_completion_tokens=max(MAX_NEW_TOKENS, 450 if response_lang == "both" else 350),
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

            # Length check adjusted for language - rewrite if too long
            max_length = TARGET_CHAR_MAX if response_lang != "en" else 1200
            if len(current) > max_length:
                rewrite_instruction = (
                    "Rewrite in Korean: 300-400 characters exactly"
                    if response_lang == "ko"
                    else "Rewrite in English: 150-200 words exactly"
                    if response_lang == "en"
                    else "Rewrite bilingual: Korean 300-400 chars, English 150-200 words"
                )
                
                # Only send the assistant's response to rewrite, not the full context
                rewrite = gpt_client.chat.completions.create(
                    model=GPT_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": f"{rewrite_instruction}. Keep meaning, avoid sources."},
                        {"role": "user", "content": current},
                    ],
                    max_completion_tokens=400 if response_lang == "both" else 250,
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