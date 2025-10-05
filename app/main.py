from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging

try:
    import transformers
    import torch
except Exception:
    # Let users know to install requirements if imports fail
    raise RuntimeError("Missing required packages. Install from requirements.txt")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64
    do_sample: bool = True


app = FastAPI(title="SheBots FastAPI Model Server")

MODEL_ID = os.environ.get("MODEL_ID", "distilgpt2")

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)


def load_pipeline(model_id: str):
    """Load a transformers text-generation pipeline optimized for CPU."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
    pipe = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # CPU
    )
    return pipe


@app.on_event("startup")
def startup_event():
    global pipeline
    logger.info(f"Loading model {MODEL_ID} (this may take a while)...")
    pipeline = load_pipeline(MODEL_ID)
    logger.info("Model loaded and ready")


@app.post("/generate")
def generate(req: GenerateRequest):
    if not req.prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    outputs = pipeline(req.prompt, max_new_tokens=req.max_new_tokens, do_sample=req.do_sample)
    return {"generated_text": outputs[0]["generated_text"]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
