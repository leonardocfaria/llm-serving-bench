"""GPT-2-style decoder with a KV cache and padded batched greedy generation."""
import io
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 256  # byte-level tokens


def encode(text: str) -> list[int]:
    return list(text.encode())


def decode(ids: list[int]) -> str:
    return bytes(ids).decode(errors="replace")


class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.h = n_heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc, self.out = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x, mask, cache):
        B, T, d = x.shape
        q, k, v = (t.view(B, T, self.h, d // self.h).transpose(1, 2) for t in self.qkv(self.ln1(x)).split(d, -1))
        if cache is not None:
            k, v = torch.cat([cache[0], k], 2), torch.cat([cache[1], v], 2)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))
        return x + self.out(F.gelu(self.fc(self.ln2(x)))), (k, v)


class TinyGPT(nn.Module):
    def __init__(self, d=512, n_layers=6, n_heads=8, ctx=512):
        super().__init__()
        self.ctx = ctx
        self.tok, self.pos = nn.Embedding(VOCAB, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, n_heads) for _ in range(n_layers))
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, VOCAB, bias=False)

    def forward(self, ids, key_mask, cache=None):
        """ids [B,T] new tokens; key_mask [B,S] True for real tokens over past+new; cache: per-layer (k, v)."""
        T, S = ids.shape[1], key_mask.shape[1]
        pos = (key_mask.cumsum(1) - 1).clamp(min=0)[:, -T:]  # left padding does not shift positions
        causal = torch.ones(T, S, dtype=torch.bool, device=ids.device).tril(S - T)
        mask = causal & key_mask[:, None, None, :]
        # every query may see itself, so all-pad rows stay finite (pad keys are never seen by real tokens)
        mask[..., torch.arange(T), torch.arange(S - T, S)] = True
        x = self.tok(ids) + self.pos(pos)
        new_cache = []
        for i, blk in enumerate(self.blocks):
            x, kv = blk(x, mask, None if cache is None else cache[i])
            new_cache.append(kv)
        return self.head(self.ln(x)), new_cache

    def size_bytes(self) -> int:
        buf = io.BytesIO()
        torch.save(self.state_dict(), buf)
        return buf.tell()


@torch.inference_mode()
def generate(model, prompts: list[list[int]], max_new_tokens, use_cache: bool = True) -> list[list[int]]:
    """Greedy decode a batch of prompts of any lengths. max_new_tokens: int or one per prompt."""
    n_new = [max_new_tokens] * len(prompts) if isinstance(max_new_tokens, int) else list(max_new_tokens)
    if any(not p for p in prompts) or max(len(p) + n for p, n in zip(prompts, n_new)) > model.ctx:
        raise ValueError(f"prompts must be non-empty and prompt + new tokens <= {model.ctx}")
    L, steps = max(map(len, prompts)), max(n_new)
    ids = torch.tensor([[0] * (L - len(p)) + p for p in prompts])
    mask = torch.tensor([[False] * (L - len(p)) + [True] * len(p) for p in prompts])
    out = []
    logits, cache = model(ids, mask)
    for step in range(steps):
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        out.append(nxt)
        if step == steps - 1:
            break
        mask = torch.cat([mask, torch.ones_like(nxt, dtype=torch.bool)], 1)
        if use_cache:
            logits, cache = model(nxt, mask, cache)
        else:
            ids = torch.cat([ids, nxt], 1)
            logits, _ = model(ids, mask)
    tokens = torch.cat(out, 1).tolist()
    return [t[:n] for t, n in zip(tokens, n_new)]


def quantize(model: nn.Module) -> nn.Module:
    """int8 dynamic quantisation of every nn.Linear (weights int8, activations quantised per batch)."""
    from torch.ao.quantization import quantize_dynamic

    with warnings.catch_warnings():  # torch.ao is deprecated in favour of torchao; still the stdlib-of-torch path
        warnings.simplefilter("ignore")
        q = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    q.size_bytes = lambda: TinyGPT.size_bytes(q)
    return q
