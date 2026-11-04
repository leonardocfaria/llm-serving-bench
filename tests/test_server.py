from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("LLM_SMALL", "1")
    monkeypatch.setenv("LLM_MAX_BATCH", "4")
    monkeypatch.setenv("LLM_MAX_WAIT_MS", "100")
    from llmserve.server import app

    with TestClient(app) as c:
        yield c


def test_generate_and_validation(client):
    r = client.post("/generate", json={"prompt": "hello", "max_new_tokens": 7})
    assert r.status_code == 200 and r.json()["tokens"] == 7
    assert client.post("/generate", json={"prompt": "", "max_new_tokens": 7}).status_code == 422
    assert client.post("/generate", json={"prompt": "x", "max_new_tokens": 999}).status_code == 422


def test_concurrent_requests_are_batched(client):
    with ThreadPoolExecutor(4) as pool:
        outs = list(pool.map(lambda i: client.post("/generate", json={"prompt": f"p{i}", "max_new_tokens": 4}), range(4)))
    assert all(o.status_code == 200 for o in outs)
    assert client.get("/stats").json()["mean_batch_size"] > 1
