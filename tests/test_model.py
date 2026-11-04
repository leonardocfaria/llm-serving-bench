import pytest
import torch

from llmserve.model import TinyGPT, decode, encode, generate, quantize

PROMPTS = [encode("hello"), encode("a much longer prompt here"), encode("x")]


def small(dtype=torch.float64):
    torch.manual_seed(0)
    return TinyGPT(d=64, n_layers=2, n_heads=4, ctx=128).to(dtype).eval()


def test_encode_decode_roundtrip():
    assert decode(encode("héllo")) == "héllo"


def test_kv_cache_matches_full_recompute():
    m = small()
    assert generate(m, PROMPTS, 12, use_cache=True) == generate(m, PROMPTS, 12, use_cache=False)


def test_batched_padded_matches_one_by_one():
    m = small()
    batched = generate(m, PROMPTS, [5, 12, 8])
    single = [generate(m, [p], n)[0] for p, n in zip(PROMPTS, [5, 12, 8])]
    assert batched == single
    assert [len(o) for o in batched] == [5, 12, 8]


def test_context_limit_checked_over_whole_batch():
    m = small()  # ctx 128: each request fits alone, but the batch decodes to 120 + 30
    with pytest.raises(ValueError):
        generate(m, [[1] * 120, [1]], [1, 30])


def test_deterministic_with_seed():
    assert generate(small(), PROMPTS, 6) == generate(small(), PROMPTS, 6)


def test_int8_quantized_runs_and_is_smaller():
    m = small(torch.float32)
    q = quantize(m)
    out = generate(q, PROMPTS, 6)
    assert [len(o) for o in out] == [6, 6, 6]
    assert q.size_bytes() < 0.5 * m.size_bytes()
