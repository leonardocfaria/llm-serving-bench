# llm-serving-bench

## Goal
Build "LLM inference serving with batching and benchmarks". Serve a small open model (for example distilgpt2 or a tiny Qwen) behind a FastAPI server with dynamic request batching and a KV-cache-aware generation loop; build a load generator measuring p50/p95 latency, tokens/s and throughput vs batch size; compare naive vs batched serving and int8 dynamic quantisation. Keep it CPU-friendly (the cloud box has no GPU): small models and datasets, a fixed random seed, and a documented runtime.

## Definition of done
- [ ] server with dynamic batching works
- [ ] benchmark script produces a results table and plots
- [ ] quantised vs fp32 comparison committed
- [ ] Tests pass with one command; CI workflow runs them
- [ ] README with setup, usage, architecture and results
- [ ] Self code review done
- [ ] Draft PR open

