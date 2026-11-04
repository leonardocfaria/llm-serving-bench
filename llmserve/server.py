"""FastAPI server. Env: LLM_MAX_BATCH (1 = naive), LLM_MAX_WAIT_MS, LLM_QUANT (none|int8), LLM_SMALL (tests)."""
import asyncio
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from .batcher import Batcher
from .model import TinyGPT, decode, encode, generate, quantize

state = {}


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=256)
    max_new_tokens: int = Field(default=32, ge=1, le=128)

    @field_validator("prompt")
    @classmethod
    def fits_context(cls, v):  # byte-level tokens: 256 bytes + 128 new tokens <= ctx 512
        if len(v.encode()) > 256:
            raise ValueError("prompt must be at most 256 UTF-8 bytes")
        return v


def build_model():
    torch.manual_seed(0)
    small = os.environ.get("LLM_SMALL") == "1"
    model = (TinyGPT(d=64, n_layers=2, n_heads=4) if small else TinyGPT()).eval()
    return quantize(model) if os.environ.get("LLM_QUANT", "none") == "int8" else model


@asynccontextmanager
async def lifespan(app):
    batcher = state["batcher"] = Batcher(lambda prompts, n: generate(model, prompts, n),
                                         max_batch=int(os.environ.get("LLM_MAX_BATCH", "8")),
                                         max_wait_ms=float(os.environ.get("LLM_MAX_WAIT_MS", "5")))
    # build on the model thread too: all torch work on one thread keeps one OpenMP pool (~2x faster, measured)
    model = await asyncio.get_running_loop().run_in_executor(batcher.executor, build_model)
    task = asyncio.create_task(batcher.run())
    yield
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    batcher.executor.shutdown()


app = FastAPI(title="llm-serving-bench", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    return state["batcher"].stats()


@app.post("/generate")
async def generate_endpoint(req: GenerateRequest):
    tokens = await state["batcher"].submit(encode(req.prompt), req.max_new_tokens)
    return {"text": decode(tokens), "tokens": len(tokens)}
