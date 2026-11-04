"""Dynamic request batching: collect up to max_batch requests or until max_wait_ms, run them together."""
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor


class Batcher:
    def __init__(self, run_batch, max_batch: int = 8, max_wait_ms: float = 5.0):
        self.run_batch = run_batch  # (prompts, max_new_tokens list) -> outputs, called in a worker thread
        self.max_batch, self.max_wait = max_batch, max_wait_ms / 1000
        self.queue: asyncio.Queue = asyncio.Queue()
        self.batch_sizes: list[int] = []
        # one dedicated model thread: torch's OpenMP pool is per calling thread, so running torch on
        # several threads makes pools compete for the cores (measured: ~2x slower batch-1 latency)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model")

    async def submit(self, prompt, max_new_tokens):
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((prompt, max_new_tokens, fut))
        return await fut

    async def run(self):
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self.queue.get()]
            deadline = loop.time() + self.max_wait
            while len(batch) < self.max_batch and (left := deadline - loop.time()) > 0:
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), left))
                except TimeoutError:
                    break
            self.batch_sizes.append(len(batch))
            try:
                outs = await loop.run_in_executor(self.executor, self.run_batch, [b[0] for b in batch], [b[1] for b in batch])
            except Exception as e:  # one failure fails the whole batch, never hangs callers
                for *_, fut in batch:
                    fut.done() or fut.set_exception(e)
                continue
            for (*_, fut), out in zip(batch, outs):
                fut.done() or fut.set_result(out)  # done() = caller went away (cancelled)

    def stats(self) -> dict:
        n = len(self.batch_sizes)
        return {"batches": n, "mean_batch_size": sum(self.batch_sizes) / n if n else 0.0,
                "histogram": dict(sorted(Counter(self.batch_sizes).items()))}
