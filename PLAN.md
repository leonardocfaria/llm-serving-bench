# llm-serving-bench Implementation Plan

## Brainstorm (answered from SPEC + repo evidence)
- **What is it?** A small LLM inference server showing the two levers that matter on CPU: request batching (amortise one weight read across many sequences) and int8 weights. Measured, not claimed.
- **Model:** Hugging Face is blocked (DECISIONS #3), so no distilgpt2 weights. A hand-rolled GPT-2-style decoder (6 layers, d=512, 8 heads, byte vocabulary, ~19 M params) with a fixed seed and random weights. Serving cost depends on shapes, not weight values; text quality is out of scope. No `transformers` dependency is needed.
- **KV cache:** prefill once, then decode one token per step, appending K/V. Left-padding plus a key mask lets prompts of different lengths share a batch. Positions come from the mask so padding does not shift them.
- **Dynamic batching:** an asyncio queue. The batcher takes the first request, waits up to `max_wait_ms` for more (at most `max_batch`), runs one batched `generate` in a worker thread and resolves each future. Naive serving = the same server with `max_batch=1`.
- **Quantisation:** `torch.ao.quantization.quantize_dynamic` on `nn.Linear` (int8 weights, dynamic activations). It is deprecated in favour of torchao but still present, which avoids a new dependency.
- **Non-goals:** continuous (iteration-level) batching, sampling strategies, streaming responses, paged KV.

## Files
- `llmserve/model.py`: `TinyGPT`, `encode/decode`, `generate(model, prompts, max_new_tokens, use_cache=True)` (greedy).
- `llmserve/batcher.py`: `Batcher(run_batch, max_batch, max_wait_ms)` with `submit()` and `run()`.
- `llmserve/server.py`: FastAPI `/generate`, `/stats`, `/health`; env `LLM_MAX_BATCH`, `LLM_MAX_WAIT_MS`, `LLM_QUANT`.
- `bench.py`: starts a server subprocess per config, closed-loop load at several concurrencies, p50/p95, tokens/s, req/s, mean batch size → `results/`.

## Tasks (TDD)
- [x] 1. model: KV-cache output == full recompute; batched (padded) == one-by-one; int8 runs and agrees mostly.
- [x] 2. batcher: concurrent submits are grouped up to max_batch; results routed to the right caller; errors propagate.
- [x] 3. server: TestClient /generate + validation 422.
- [x] 4. bench + results table + plots; README; CI.
