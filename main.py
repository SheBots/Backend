from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import os, re, json, threading, time, logging

from dotenv import load_dotenv
load_dotenv()

"""
Default to a very small CPU-friendly model to work on low-RAM machines.
You can override MODEL_ID in your .env when you want a larger model or use a remote API.
"""
MODEL_ID = os.getenv("MODEL_ID", "distilgpt2")
PORT = int(os.getenv("PORT", 8000))
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "http://localhost:5173")
RAG_BASE_URL = os.getenv("RAG_BASE_URL", "http://localhost:8080")

import httpx

app = FastAPI(title="SheBots Backend")

logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)


# Configure CORS so browser preflight (OPTIONS) requests to /api/chat succeed
origins = [os.getenv("ALLOWED_ORIGIN", "http://localhost:5173"), "http://localhost:3000", "http://127.0.0.1:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    history: list | None = None
    useDocs: bool | None = False


def sse_format(event, data_dict):
    return ("event: {}\n".format(event) if event else "") + f"data: {json.dumps(data_dict, ensure_ascii=False)}\n\n"


# Simple redaction
_EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+")
_PHONE_RE = re.compile(r"\+?\d[\d\s-]{6,}\d")

def redact(text: str) -> str:
    text = _EMAIL_RE.sub("[redacted email]", text)
    text = _PHONE_RE.sub("[redacted phone]", text)
    return text


# Model loading (lazy) - follow provided sample using transformers
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
    import torch
except Exception as e:
    raise RuntimeError("Missing transformers/torch. Install requirements.txt") from e


def get_dtype():
    dtype_str = os.getenv("TORCH_DTYPE", "bfloat16")
    if dtype_str.lower() == "bfloat16":
        if torch.cuda.is_available():
            return torch.bfloat16
        # bf16 not available on CPU
        return torch.float32
    return getattr(torch, dtype_str, torch.float32)

_tokenizer = None
_model = None

def load_model():
    global _tokenizer, _model
    if _model is not None:
        return
    dtype = get_dtype()
    # Prefer CPU by default on low-RAM machines. Set DEVICE_MAP in env to change.
    device_map = os.getenv("DEVICE_MAP", "cpu")
    # read token from .env or environment
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    logger.info(f"Loading model {MODEL_ID} dtype={dtype} device_map={device_map} auth={'yes' if hf_token else 'no'}")

    # Use transformers version to select correct argument names to avoid deprecation errors
    import transformers as _transformers
    import warnings

    # determine major version (fallback to 4 if parsing fails)
    try:
        _tf_major = int((_transformers.__version__ or "4").split('.')[0])
    except Exception:
        _tf_major = 4

    # choose token kw name: transformers v5+ uses `token`, older versions use `use_auth_token`
    _token_kwargs = {}
    if hf_token:
        if _tf_major >= 5:
            _token_kwargs = {"token": hf_token}
        else:
            _token_kwargs = {"use_auth_token": hf_token}

    # choose dtype kw name: v5+ prefers `dtype`, older versions use `torch_dtype`
    _model_kwargs = {"device_map": device_map, "trust_remote_code": True}
    if _tf_major >= 5:
        _model_kwargs.update({"dtype": dtype})
    else:
        _model_kwargs.update({"torch_dtype": dtype})

    # suppress noisy FutureWarnings from transformers/huggingface_hub (targeted)
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, **_token_kwargs)

    # Some small models don't support trust_remote_code or the same dtype args; keep a safe fallback
    try:
        _model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **_model_kwargs, **_token_kwargs)
    except Exception:
        # Retry with minimal kwargs (CPU, no dtype override)
        _model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cpu", trust_remote_code=False, **_token_kwargs)

    # Some tokenizers (and older model bundles) don't provide apply_chat_template; provide a safe helper
    if not hasattr(_tokenizer, "apply_chat_template"):
        def _apply_chat_template(messages, tokenize=False, add_generation_prompt=True):
            # very small compatibility shim to join messages into a single prompt
            parts = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                parts.append(f"[{role}] {content}")
            if add_generation_prompt:
                parts.append("\nAssistant:")
            return "\n".join(parts)
        _tokenizer.apply_chat_template = _apply_chat_template

@app.on_event("startup")
def startup_event():
    # Load model in background thread to not block startup if heavy
    t = threading.Thread(target=load_model, daemon=True)
    t.start()


@app.get("/api/health")
async def health():
    model_ready = (_model is not None) and (_tokenizer is not None)
    return {"ok": True, "model": MODEL_ID, "model_ready": bool(model_ready)}


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    msg = (req.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message is required")
    if len(msg) > 2000:
        raise HTTPException(status_code=400, detail="message too long")

    # redact user message
    redacted_message = redact(msg)

    snippets = []
    if req.useDocs:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{RAG_BASE_URL}/rag/retrieve", params={"q": msg, "k": 5})
                if r.status_code == 200:
                    body = r.json()
                    snippets = body.get("results") or []
        except Exception as e:
            logger.info("RAG service unavailable: %s", e)

    preamble = "You are a helpful assistant."
    if snippets:
        pieces = ["You are a helpful assistant. Use the following web snippets from the Kyungpook National University CS website if relevant. Cite inline as [title](url).\n----\n"]
        for s in snippets:
            text = redact(s.get("text",""))
            title = s.get("title","")
            url = s.get("url","")
            pieces.append(f"{text}\nSource: [{title}]({url})")
        pieces.append("----\n")
        preamble = "\n".join(pieces)

    # small bilingual hint
    system_prefix = "Answer briefly. If the user writes in Korean, answer bilingually (Korean then English).\n"

    full_prompt = system_prefix + preamble + "\nUser: " + redacted_message + "\nAssistant:"

    # Wait for model to be loaded
    load_model()
    if _model is None or _tokenizer is None:
        return JSONResponse({"error":"Model not ready"}, status_code=503)

    inputs = _tokenizer([full_prompt], return_tensors="pt")

    # move inputs to model device if necessary
    try:
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}
    except Exception:
        pass

    streamer = TextIteratorStreamer(_tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        input_ids=inputs.get("input_ids"),
        attention_mask=inputs.get("attention_mask"),
        streamer=streamer,
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", 512)),
        temperature=float(os.getenv("TEMPERATURE", 0.2)),
        top_p=float(os.getenv("TOP_P", 0.95)),
    )

    thread = threading.Thread(target=_model.generate, kwargs=generation_kwargs)
    thread.start()

    async def event_gen():
        yield sse_format("start", {"model": MODEL_ID})
        tokens = 0
        try:
            for text in streamer:
                tokens += len(text)
                yield sse_format(None, {"token": text})
                if await request.is_disconnected():
                    break
            yield sse_format("end", {"tokensStreamed": tokens})
        except Exception as e:
            yield sse_format("error", {"error": str(e)})

    return StreamingResponse(event_gen(), media_type="text/event-stream")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
