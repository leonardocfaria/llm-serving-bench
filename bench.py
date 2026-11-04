"""Load generator: starts one server per config, closed-loop clients at several concurrencies.

python bench.py            # full run (~19 M-param model), writes results/
python bench.py --quick    # tiny model, smoke test
"""
import argparse
import asyncio
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

CONFIGS = {  # name: server env
    "naive-fp32": {"LLM_MAX_BATCH": "1"},
    "naive-int8": {"LLM_MAX_BATCH": "1", "LLM_QUANT": "int8"},
    "batched-fp32": {"LLM_MAX_BATCH": "16"},
    "batched-int8": {"LLM_MAX_BATCH": "16", "LLM_QUANT": "int8"},
}
PROMPTS = ["The quick brown fox", "Once upon a time", "def fibonacci(n):", "In 1905, Einstein",
           "Serving models on CPU", "Hello", "A long prompt that keeps going for a while to vary lengths", "x = 1"]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(env: dict, port: int) -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "llmserve.server:app", "--port", str(port), "--log-level", "warning"],
                            env={**os.environ, **env}, cwd=Path(__file__).parent)
    for _ in range(100):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health").status_code == 200:
                return proc
        except httpx.HTTPError:
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError("server did not start")


async def load(url: str, concurrency: int, n_requests: int, max_new: int) -> dict:
    latencies, tokens = [], 0
    counter = iter(range(n_requests))

    async def worker(client):
        nonlocal tokens
        for i in counter:  # shared iterator: closed loop, each worker sends its next request when one returns
            t = time.perf_counter()
            r = await client.post(f"{url}/generate", json={"prompt": PROMPTS[i % len(PROMPTS)], "max_new_tokens": max_new})
            r.raise_for_status()
            latencies.append(time.perf_counter() - t)
            tokens += r.json()["tokens"]

    async with httpx.AsyncClient(timeout=300) as client:
        before = (await client.get(f"{url}/stats")).json()
        t0 = time.perf_counter()
        await asyncio.gather(*(worker(client) for _ in range(concurrency)))
        wall = time.perf_counter() - t0
        after = (await client.get(f"{url}/stats")).json()
    batches = after["batches"] - before["batches"]
    served = after["mean_batch_size"] * after["batches"] - before["mean_batch_size"] * before["batches"]
    q = statistics.quantiles(latencies, n=20)
    return {"concurrency": concurrency, "requests": n_requests, "p50_ms": statistics.median(latencies) * 1000,
            "p95_ms": q[18] * 1000, "tokens_per_s": tokens / wall, "req_per_s": n_requests / wall,
            "mean_batch": served / batches}


def quant_comparison(small: bool) -> dict:
    """Model-level fp32 vs int8: size and greedy-token agreement on the benchmark prompts."""
    import torch
    from llmserve.model import TinyGPT, encode, generate, quantize

    torch.manual_seed(0)
    fp32 = (TinyGPT(d=64, n_layers=2, n_heads=4) if small else TinyGPT()).eval()
    int8 = quantize(fp32)
    prompts = [encode(p) for p in PROMPTS]
    a, b = generate(fp32, prompts, 32), generate(int8, prompts, 32)
    agree = sum(x == y for ra, rb in zip(a, b) for x, y in zip(ra, rb)) / sum(map(len, a))
    first = sum(ra[0] == rb[0] for ra, rb in zip(a, b)) / len(a)
    params = sum(p.numel() for p in fp32.parameters())
    return {"params": params, "fp32_mb": fp32.size_bytes() / 1e6, "int8_mb": int8.size_bytes() / 1e6,
            "token_agreement": agree, "first_token_agreement": first}


def plot(results: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, rows in results.items():
        c = [r["concurrency"] for r in rows]
        axes[0].plot(c, [r["tokens_per_s"] for r in rows], marker="o", label=name)
        axes[1].plot(c, [r["p95_ms"] for r in rows], marker="o", label=name)
    for ax, title in zip(axes, ["throughput (tokens/s)", "p95 latency (ms)"]):
        ax.set(xscale="log", xlabel="concurrent clients", title=title)
        ax.set_xticks([r["concurrency"] for r in rows], [str(r["concurrency"]) for r in rows])
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "bench.png", dpi=120)


def table(results: dict, quant: dict) -> str:
    lines = ["| config | clients | p50 ms | p95 ms | tokens/s | req/s | mean batch |", "|---|---|---|---|---|---|---|"]
    for name, rows in results.items():
        lines += [f"| {name} | {r['concurrency']} | {r['p50_ms']:.0f} | {r['p95_ms']:.0f} | {r['tokens_per_s']:.0f} | "
                  f"{r['req_per_s']:.2f} | {r['mean_batch']:.1f} |" for r in rows]
    lines += ["", "| model | params | size MB | greedy token agreement vs fp32 | first-token agreement |", "|---|---|---|---|---|",
              f"| fp32 | {quant['params'] / 1e6:.1f} M | {quant['fp32_mb']:.1f} | 100 % | 100 % |",
              f"| int8 dynamic | {quant['params'] / 1e6:.1f} M | {quant['int8_mb']:.1f} | {quant['token_agreement']:.0%} | "
              f"{quant['first_token_agreement']:.0%} |"]
    return "\n".join(lines) + "\n"


def main(quick: bool = False, out: str = "results", max_new: int = 32) -> dict:
    out_dir = Path(out)
    out_dir.mkdir(exist_ok=True)
    levels = [1, 4] if quick else [1, 2, 4, 8, 16]
    extra = {"LLM_SMALL": "1"} if quick else {}
    results = {}
    for name, env in CONFIGS.items():
        port = free_port()
        proc = start_server({**env, **extra}, port)
        try:
            url = f"http://127.0.0.1:{port}"
            asyncio.run(load(url, 2, 4, max_new))  # warm-up
            results[name] = [asyncio.run(load(url, c, max(8, 3 * c), max_new)) for c in levels]
            print(name, json.dumps(results[name][-1]))
        finally:
            proc.terminate()
            proc.wait()
    quant = quant_comparison(quick)
    (out_dir / "results.json").write_text(json.dumps({"load": results, "quant": quant, "max_new_tokens": max_new}, indent=2))
    (out_dir / "results.md").write_text(table(results, quant))
    plot(results, out_dir)
    return {"load": results, "quant": quant}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    main(a.quick, a.out)
