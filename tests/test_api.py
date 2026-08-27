"""Tests for the DaraLM FastAPI service (Phase 10, spec section 26)."""

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


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["device"] == "cpu"


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


def test_tokenize_round_trips_ids_and_pieces(client, tiny_tokenizer):
    response = client.post("/v1/tokenize", json={"text": "hello world"})
    assert response.status_code == 200
    body = response.json()
    expected_ids = tiny_tokenizer.encode("hello world")
    assert body["token_ids"] == expected_ids
    assert body["token_count"] == len(expected_ids)
    assert len(body["tokens"]) == len(body["token_ids"])


def test_tokenize_with_bos_eos_keeps_tokens_and_ids_aligned(client):
    response = client.post("/v1/tokenize", json={"text": "hello", "add_bos": True, "add_eos": True})
    body = response.json()
    assert body["tokens"][0] == "<bos>"
    assert body["tokens"][-1] == "<eos>"
    assert len(body["tokens"]) == len(body["token_ids"])


def test_tokenize_rejects_unknown_fields(client):
    response = client.post("/v1/tokenize", json={"text": "hi", "add_boss": True})
    assert response.status_code == 422


def test_normalize_removes_space_before_khmer_punctuation(client):
    response = client.post("/v1/normalize", json={"text": "ប្រទេសកម្ពុជា ។"})
    assert response.status_code == 200
    body = response.json()
    assert body["normalized_text"] == "ប្រទេសកម្ពុជា។"
    assert body["changed"] is True


def test_normalize_reports_unchanged_when_already_normalized(client):
    response = client.post("/v1/normalize", json={"text": "hello world"})
    body = response.json()
    assert body["normalized_text"] == "hello world"
    assert body["changed"] is False


def test_normalize_arabic_digits_mode(client):
    response = client.post("/v1/normalize", json={"text": "ឆ្នាំ២០២៤", "digits": "arabic"})
    body = response.json()
    assert "2024" in body["normalized_text"]


def test_normalize_rejects_invalid_digits_option(client):
    response = client.post("/v1/normalize", json={"text": "hi", "digits": "latin"})
    assert response.status_code == 422


def test_normalize_rejects_unknown_fields(client):
    response = client.post("/v1/normalize", json={"text": "hi", "typo_field": True})
    assert response.status_code == 422


def test_normalize_works_without_a_loaded_model():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.routes import normalize as normalize_route

    isolated_app = FastAPI()
    isolated_app.include_router(normalize_route.router)
    isolated_client = TestClient(isolated_app)

    response = isolated_client.post("/v1/normalize", json={"text": "hello ។"})
    assert response.status_code == 200
    assert response.json()["normalized_text"] == "hello។"


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
    response = client.post("/v1/generate", json={"prompt": "hello world"})
    assert response.status_code == 200


def test_chat_returns_the_spec_shaped_response(client):
    response = client.post(
        "/v1/chat",
        json={"instruction": "hello", "max_new_tokens": 10, "temperature": 0.8},
    )
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["response"], str)
    assert isinstance(body["tokens_generated"], int)
    assert body["tokens_generated"] >= 0
    assert body["model"] == "daralm-api-test"


def test_chat_response_never_contains_the_assistant_close_marker(client):
    from daralm.data.chat_template import ASSISTANT_CLOSE

    response = client.post("/v1/chat", json={"instruction": "hello world", "max_new_tokens": 30})
    assert ASSISTANT_CLOSE not in response.json()["response"]


def test_chat_rejects_empty_instruction(client):
    response = client.post("/v1/chat", json={"instruction": ""})
    assert response.status_code == 422


def test_chat_rejects_invalid_temperature(client):
    response = client.post("/v1/chat", json={"instruction": "hi", "temperature": -1.0})
    assert response.status_code == 422


def test_chat_rejects_max_new_tokens_over_the_cap(client):
    response = client.post("/v1/chat", json={"instruction": "hi", "max_new_tokens": 5000})
    assert response.status_code == 422


def test_chat_rejects_unknown_fields(client):
    response = client.post("/v1/chat", json={"prompt": "hi"})
    assert response.status_code == 422


def test_chat_uses_default_sampling_params_when_omitted(client):
    response = client.post("/v1/chat", json={"instruction": "hello world"})
    assert response.status_code == 200


def test_metrics_returns_prometheus_text_format(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


def test_metrics_reflects_a_real_request(client):
    client.get("/v1/model")
    response = client.get("/metrics")
    body = response.text
    assert "daralm_requests_total" in body
    assert 'path="/v1/model"' in body


def test_metrics_counts_tokens_generated(client):
    client.post("/v1/generate", json={"prompt": "hi", "max_new_tokens": 5})
    body = client.get("/metrics").text
    assert "daralm_tokens_generated_total" in body
    assert 'endpoint="generate"' in body


def test_root_serves_the_frontend_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "DaraLM" in response.text


def test_get_model_service_raises_503_before_startup():
    from types import SimpleNamespace

    from fastapi import HTTPException

    fake_request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    with pytest.raises(HTTPException) as exc_info:
        get_model_service(fake_request)
    assert exc_info.value.status_code == 503
