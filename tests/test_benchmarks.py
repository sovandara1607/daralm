from __future__ import annotations

import json

import pytest
import torch

from daralm.evaluation.benchmarks import (
    check_overfitting,
    compare_checkpoints,
    evaluate_checkpoint,
    load_history_or_meta,
    summarize_history,
)
from daralm.model.config import ModelConfig
from daralm.model.transformer import DaraLMTransformer
from daralm.tokenizer.tokenizer import DaraLMTokenizer
from daralm.tokenizer.train import train_sentencepiece
from daralm.training.checkpoint import save_checkpoint
from daralm.training.optimizer import build_optimizer
from daralm.training.scheduler import build_scheduler


def _history_improving() -> list[dict]:
    return [
        {"step": 10, "train_loss": 9.0, "lr": 1e-4, "tokens_per_sec": 100.0, "gpu_memory_gb": 0.1},
        {
            "step": 20,
            "train_loss": 7.0,
            "lr": 2e-4,
            "tokens_per_sec": 110.0,
            "gpu_memory_gb": 0.1,
            "val_loss": 7.5,
            "perplexity": 1800.0,
        },
        {
            "step": 30,
            "train_loss": 6.0,
            "lr": 2e-4,
            "tokens_per_sec": 105.0,
            "gpu_memory_gb": 0.1,
            "val_loss": 6.2,
            "perplexity": 490.0,
        },
    ]


def _history_overfitting() -> list[dict]:
    return [
        {
            "step": 10,
            "train_loss": 5.0,
            "lr": 1e-4,
            "tokens_per_sec": 100.0,
            "gpu_memory_gb": 0.1,
            "val_loss": 5.5,
            "perplexity": 244.0,
        },
        {
            "step": 20,
            "train_loss": 3.0,
            "lr": 2e-4,
            "tokens_per_sec": 110.0,
            "gpu_memory_gb": 0.1,
            "val_loss": 5.0,
            "perplexity": 148.0,
        },  # best val
        {
            "step": 30,
            "train_loss": 1.5,
            "lr": 2e-4,
            "tokens_per_sec": 105.0,
            "gpu_memory_gb": 0.1,
            "val_loss": 6.0,
            "perplexity": 403.0,
        },  # val rose from its best -> overfitting
    ]


def test_summarize_history_basic():
    summary = summarize_history(_history_improving())
    assert summary["history_available"] is True
    assert summary["initial_train_loss"] == 9.0
    assert summary["final_train_loss"] == 6.0
    assert summary["final_val_loss"] == 6.2
    assert summary["best_val_loss"] == 6.2


def test_check_overfitting_still_improving():
    summary = summarize_history(_history_improving())
    result = check_overfitting(summary)
    assert "no overfitting" in result["verdict"]
    assert result["final_val_is_best_val"] is True


def test_check_overfitting_detects_val_loss_rising_from_best():
    summary = summarize_history(_history_overfitting())
    result = check_overfitting(summary)
    assert "early overfitting" in result["verdict"]
    assert result["final_val_is_best_val"] is False


def test_check_overfitting_insufficient_data_when_history_unavailable():
    result = check_overfitting({"history_available": False, "note": "no history.json"})
    assert result["verdict"] == "insufficient data"


def test_check_overfitting_insufficient_data_missing_loss_keys():
    result = check_overfitting({"history_available": True, "num_log_events": 0})
    assert result["verdict"] == "insufficient data"


def test_load_history_or_meta_prefers_history_json(tmp_path):
    ckpt_dir = tmp_path / "daralm-x"
    ckpt_dir.mkdir()
    (ckpt_dir / "history.json").write_text(json.dumps(_history_improving()))
    (ckpt_dir / "best").mkdir()
    (ckpt_dir / "best" / "meta.json").write_text(
        json.dumps({"val_loss": 999.0, "perplexity": 999.0})
    )

    summary = load_history_or_meta(ckpt_dir)
    assert summary["history_available"] is True
    assert summary["final_val_loss"] == 6.2  # from history.json, not the meta.json decoy


def test_load_history_or_meta_falls_back_to_meta_json(tmp_path):
    ckpt_dir = tmp_path / "daralm-old"
    ckpt_dir.mkdir()
    (ckpt_dir / "best").mkdir()
    (ckpt_dir / "best" / "meta.json").write_text(
        json.dumps({"step": 300, "val_loss": 7.46, "perplexity": 1737.5})
    )

    summary = load_history_or_meta(ckpt_dir)
    assert summary["history_available"] is False
    assert summary["final_val_loss"] == 7.46
    assert summary["final_perplexity"] == 1737.5


def test_load_history_or_meta_raises_when_neither_exists(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_history_or_meta(tmp_path / "nonexistent")


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory):
    corpus_dir = tmp_path_factory.mktemp("bench_corpus")
    corpus_path = corpus_dir / "corpus.txt"
    lines = ["the quick brown fox jumps over the lazy dog and runs away"] * 60
    corpus_path.write_text("\n".join(lines), encoding="utf-8")
    model_path = train_sentencepiece(
        corpus_path, corpus_dir / "tok", vocab_size=32, model_type="unigram"
    )
    return DaraLMTokenizer.from_pretrained(model_path)


def _make_checkpoint(tmp_path_factory, tmp_path, tokenizer, name, with_history):
    config = ModelConfig(
        model_name=name,
        architecture={
            "vocab_size": tokenizer.vocab_size,
            "hidden_size": 16,
            "num_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 32,
            "max_position_embeddings": 16,
        },
        training={
            "learning_rate": 0.001,
            "batch_size": 2,
            "max_steps": 10,
            "warmup_steps": 1,
            "eval_interval": 5,
            "save_interval": 10,
        },
    )
    model = DaraLMTransformer(config.architecture, pad_token_id=tokenizer.pad_id)
    optimizer = build_optimizer(model, config.training)
    scheduler = build_scheduler(optimizer, config.training)
    tokenizer_file = tmp_path_factory.mktemp(f"tokfile_{name}") / "tok.model"
    tokenizer_file.write_bytes(b"placeholder")

    checkpoint_root = tmp_path / "checkpoints"
    save_checkpoint(
        checkpoint_root, "best", model, optimizer, scheduler, 10, 1000, config, tokenizer_file
    )
    ckpt_dir = checkpoint_root / name

    if with_history:
        history = [
            {
                "step": 5,
                "train_loss": 8.0,
                "lr": 1e-4,
                "tokens_per_sec": 50.0,
                "gpu_memory_gb": 0.05,
            },
            {
                "step": 10,
                "train_loss": 6.0,
                "lr": 1e-4,
                "tokens_per_sec": 55.0,
                "gpu_memory_gb": 0.05,
                "val_loss": 6.5,
                "perplexity": 665.0,
            },
        ]
        (ckpt_dir / "history.json").write_text(json.dumps(history))
    else:
        (ckpt_dir / "best" / "meta.json").write_text(
            json.dumps({"step": 10, "val_loss": 6.5, "perplexity": 665.0})
        )

    return ckpt_dir


def test_evaluate_checkpoint_with_history(tmp_path_factory, tmp_path, tiny_tokenizer):
    ckpt_dir = _make_checkpoint(tmp_path_factory, tmp_path, tiny_tokenizer, "test-model-a", True)
    real_docs = ["the quick brown fox jumps over the lazy dog and runs away quickly"] * 5

    result = evaluate_checkpoint(
        ckpt_dir, tiny_tokenizer, torch.device("cpu"), real_docs, max_new_tokens=5
    )

    assert result["model_name"] == "test-model-a"
    assert result["parameters"] > 0
    assert result["history_summary"]["history_available"] is True
    assert set(result["generations"].keys()) == {"english", "khmer", "mixed"}
    assert result["memorization"]["num_documents_checked"] == 5
    assert 0.0 <= result["memorization"]["avg_match_rate"] <= 1.0
    assert result["memorization"]["match_rate_vs_chance"] is not None


def test_evaluate_checkpoint_without_history_falls_back_gracefully(
    tmp_path_factory, tmp_path, tiny_tokenizer
):
    ckpt_dir = _make_checkpoint(tmp_path_factory, tmp_path, tiny_tokenizer, "test-model-b", False)
    real_docs = ["the quick brown fox jumps over the lazy dog and runs away quickly"] * 3

    result = evaluate_checkpoint(
        ckpt_dir, tiny_tokenizer, torch.device("cpu"), real_docs, max_new_tokens=5
    )

    assert result["history_summary"]["history_available"] is False
    assert result["history_summary"]["final_val_loss"] == 6.5
    assert result["overfitting"]["verdict"] == "insufficient data"


def test_compare_checkpoints_returns_all_models(tmp_path_factory, tmp_path, tiny_tokenizer):
    ckpt_a = _make_checkpoint(tmp_path_factory, tmp_path, tiny_tokenizer, "test-model-c", True)
    ckpt_b = _make_checkpoint(tmp_path_factory, tmp_path, tiny_tokenizer, "test-model-d", False)
    real_docs = ["the quick brown fox jumps over the lazy dog and runs away quickly"] * 3

    report = compare_checkpoints(
        [ckpt_a, ckpt_b], tiny_tokenizer, torch.device("cpu"), real_docs, max_new_tokens=5
    )

    assert set(report["models"].keys()) == {"test-model-c", "test-model-d"}
