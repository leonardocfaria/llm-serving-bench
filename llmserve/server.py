"""FastAPI server. Env: LLM_MAX_BATCH (1 = naive), LLM_MAX_WAIT_MS, LLM_QUANT (none|int8), LLM_SMALL (tests)."""
import asyncio
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .batcher import Batcher
from .model import TinyGPT, decode, encode, generate, quantize

state = {}


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=256)
    max_new_tokens: int = Field(default=32, ge=1, le=128)


def build_model():
    torch.manual_seed(0)
    small = os.environ.get("LLM_SMALL") == "1"
    model = (TinyGPT(d=64, n_layers=2, n_heads=4) if small else TinyGPT()).eval()
    return quantize(model) if os.environ.get("LLM_QUANT", "none") == "int8" else model


@asynccontextmanager
async def lifespan(app):
    model = build_model()
    state["batcher"] = Batcher(lambda prompts, n: generate(model, prompts, n),
                               max_batch=int(os.environ.get("LLM_MAX_BATCH", "8")),
                               max_wait_ms=float(os.environ.get("LLM_MAX_WAIT_MS", "5")))
    task = asyncio.create_task(state["batcher"].run())
    yield
    task.cancel()


app = FastAPI(title="llm-serving-bench", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    return state["batcher"].stats()


@app.post("/generate")
async def generate_endpoint(req: GenerateRequest):
    # 256 chars can be up to 1024 UTF-8 bytes: keep the last 256 so prompt + 128 new tokens fits ctx 512
    ids = encode(req.prompt)[-256:]
    tokens = await state["batcher"].submit(ids, req.max_new_tokens)
    return {"text": decode(tokens), "tokens": len(tokens)}
