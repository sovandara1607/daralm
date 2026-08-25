"""Tests for the DaraLM FastAPI service (Phase 10, spec section 26).

Uses FastAPI's `TestClient` against a real `ModelService` built from tiny
fixtures (same pattern as `tests/test_generation.py`'s `tiny_model`/
`tiny_tokenizer`) — never a real checkpoint. `app.dependency_overrides`
swaps `get_model_service` for a fixture-backed fake so tests run instantly
and don't depend on whatever happens to be trained on disk.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_model_service
from api.main import app
from api.services.model_service import ModelService
from daralm.model.config import ArchitectureConfig, ModelConfig, TrainingConfig
from daralm.model.transformer import DaraLMTransformer
from daralm.tokenizer.tokenizer import DaraLMTokenizer
from daralm.tokenizer.train import train_sentencepiece
from daralm.utils.device import get_device


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory):
    corpus_dir = tmp_path_factory.mktemp("api_corpus")
    corpus_path = corpus_dir / "corpus.txt"
    lines = ["hello world this is a test sentence", "another different sentence here"] * 30
    corpus_path.write_text("\n".join(lines), encoding="utf-8")
    model_path = train_sentencepiece(
        corpus_path, corpus_dir / "tok", vocab_size=30, model_type="unigram"
    )
    return DaraLMTokenizer.from_pretrained(model_path)


@pytest.fixture(scope="module")
def tiny_model_service(tiny_tokenizer):
    arch = ArchitectureConfig(
        vocab_size=tiny_tokenizer.vocab_size,
        hidden_size=16,
        num_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
    )
    training = TrainingConfig(
        precision="fp32",
        learning_rate=0.0003,
        batch_size=2,
        max_steps=1,
        warmup_steps=0,
        eval_interval=1,
        save_interval=1,
    )
    config = ModelConfig(
        model_name="daralm-api-test", tags=["test"], architecture=arch, training=training
    )
    model = DaraLMTransformer(arch, pad_token_id=tiny_tokenizer.pad_id)
    return ModelService(
        model=model,
        tokenizer=tiny_tokenizer,
        config=config,
        device=get_device(prefer="cpu"),
        checkpoint_step=42,
    )


@pytest.fixture
def client(tiny_model_service):
    app.dependency_overrides[get_model_service] = lambda: tiny_model_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# --- GET /health -----------------------------------------------------------


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["device"] == "cpu"


# --- GET /v1/model -----------------------------------------------------------


def test_model_info_reflects_the_loaded_config(client, tiny_tokenizer):
    response = client.get("/v1/model")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "daralm-api-test"
    assert body["tags"] == ["test"]
    assert body["vocab_size"] == tiny_tokenizer.vocab_size
    assert body["hidden_size"] == 16
    assert body["num_layers"] == 2
    assert body["num_attention_heads"] == 2
    assert body["max_position_embeddings"] == 32
    assert body["parameters"] > 0
    assert body["checkpoint_step"] == 42
    assert body["device"]


# --- POST /v1/tokenize -------------------------------------------------------


def test_tokenize_round_trips_ids_and_pieces(client, tiny_tokenizer):
    response = client.post("/v1/tokenize", json={"text": "hello world"})
    assert response.status_code == 200
    body = response.json()
    expected_ids = tiny_tokenizer.encode("hello world")
    assert body["token_ids"] == expected_ids
    assert body["token_count"] == len(expected_ids)
    assert len(body["tokens"]) == len(body["token_ids"])


def test_tokenize_with_bos_eos_keeps_tokens_and_ids_aligned(client):
    response = client.post(
        "/v1/tokenize", json={"text": "hello", "add_bos": True, "add_eos": True}
    )
    body = response.json()
    assert body["tokens"][0] == "<bos>"
    assert body["tokens"][-1] == "<eos>"
    assert len(body["tokens"]) == len(body["token_ids"])


def test_tokenize_rejects_unknown_fields(client):
    # extra="forbid" on TokenizeRequest — a typo'd field should 422, not be
    # silently ignored (same "fail loudly" discipline as ModelConfig).
    response = client.post("/v1/tokenize", json={"text": "hi", "add_boss": True})
    assert response.status_code == 422


# --- POST /v1/generate --------------------------------------------------------


def test_generate_returns_the_spec_shaped_response(client):
    response = client.post(
        "/v1/generate",
        json={"prompt": "hello", "max_new_tokens": 5, "temperature": 0.8, "top_p": 0.9},
    )
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["generated_text"], str)
    assert len(body["generated_text"]) > 0
    assert isinstance(body["tokens_generated"], int)
    assert body["tokens_generated"] >= 0
    assert body["model"] == "daralm-api-test"


def test_generate_rejects_empty_prompt(client):
    response = client.post("/v1/generate", json={"prompt": ""})
    assert response.status_code == 422


def test_generate_rejects_invalid_temperature(client):
    response = client.post("/v1/generate", json={"prompt": "hi", "temperature": -1.0})
    assert response.status_code == 422


def test_generate_rejects_max_new_tokens_over_the_cap(client):
    response = client.post("/v1/generate", json={"prompt": "hi", "max_new_tokens": 5000})
    assert response.status_code == 422


def test_generate_uses_default_sampling_params_when_omitted(client):
    # The spec's literal example body omits top_k/repetition_penalty/stop_on_eos
    # entirely — confirm those still work via their Field defaults.
    response = client.post("/v1/generate", json={"prompt": "hello world"})
    assert response.status_code == 200


# --- 503 before startup / without a loaded model ------------------------------


def test_get_model_service_raises_503_before_startup():
    # Unit-tests api.dependencies.get_model_service directly against a bare
    # SimpleNamespace standing in for `Request` — deliberately NOT going
    # through TestClient(app) here. A real request against the unmodified
    # `app` object runs api.main's actual `lifespan` on first use (even
    # without entering it as a context manager, in this Starlette version),
    # which would load the real configs/50m.yaml checkpoint from disk —
    # slow, and beside this test's actual point: that get_model_service
    # fails loudly with 503 rather than an unhandled AttributeError when
    # `app.state.model_service` was never set.
    from types import SimpleNamespace

    from fastapi import HTTPException

    fake_request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    with pytest.raises(HTTPException) as exc_info:
        get_model_service(fake_request)
    assert exc_info.value.status_code == 503
