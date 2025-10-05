from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import os, re, json, threading, time, logging

from dotenv import load_dotenv
load_dotenv()

MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Llama-3.1-8B")
PORT = int(os.getenv("PORT", 8000))
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "http://localhost:5173")
RAG_BASE_URL = os.getenv("RAG_BASE_URL", "http://localhost:8080")

import httpx

app = FastAPI(title="SheBots Backend")

logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)


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
    device_map = os.getenv("DEVICE_MAP", "auto")
    logger.info(f"Loading model {MODEL_ID} dtype={dtype} device_map={device_map}")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    _model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype, device_map=device_map)


@app.on_event("startup")
def startup_event():
    # Load model in background thread to not block startup if heavy
    t = threading.Thread(target=load_model, daemon=True)
    t.start()


@app.get("/api/health")
async def health():
    return {"ok": True, "model": MODEL_ID}


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
