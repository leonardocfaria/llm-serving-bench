import asyncio

import pytest

from llmserve.batcher import Batcher


def echo_batch(prompts, max_new):
    return [p + [n] for p, n in zip(prompts, max_new)]


async def _run(batcher, jobs):
    task = asyncio.create_task(batcher.run())
    try:
        return await asyncio.gather(*(batcher.submit(p, n) for p, n in jobs))
    finally:
        task.cancel()


def test_groups_concurrent_requests_and_routes_results():
    b = Batcher(echo_batch, max_batch=4, max_wait_ms=50)
    jobs = [([i], i) for i in range(10)]
    out = asyncio.run(_run(b, jobs))
    assert out == [[i, i] for i in range(10)]
    assert b.batch_sizes == {4: 2, 2: 1}


def test_max_batch_one_is_naive_serving():
    b = Batcher(echo_batch, max_batch=1, max_wait_ms=50)
    asyncio.run(_run(b, [([1], 1), ([2], 2), ([3], 3)]))
    assert b.batch_sizes == {1: 3}


def test_errors_reach_every_caller_in_the_batch():
    def boom(prompts, max_new):
        raise ValueError("bad batch")

    b = Batcher(boom, max_batch=4, max_wait_ms=10)

    async def main():
        task = asyncio.create_task(b.run())
        results = await asyncio.gather(b.submit([1], 1), b.submit([2], 1), return_exceptions=True)
        task.cancel()
        return results

    results = asyncio.run(main())
    assert all(isinstance(r, ValueError) for r in results)
