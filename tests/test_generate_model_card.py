"""Tests for scripts/generate_model_card.py's render() — a pure function,
so testable without touching real project artifacts on disk.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from daralm.model.config import ModelConfig

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    module_path = REPO_ROOT / "scripts" / "generate_model_card.py"
    spec = importlib.util.spec_from_file_location("generate_model_card", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_model_card"] = module
    spec.loader.exec_module(module)
    return module


def _sample_config() -> ModelConfig:
    return ModelConfig(
        model_name="daralm-test",
        architecture={
            "vocab_size": 1000,
            "hidden_size": 32,
            "num_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 64,
            "max_position_embeddings": 128,
        },
        training={
            "learning_rate": 0.001,
            "batch_size": 4,
            "max_steps": 100,
            "warmup_steps": 10,
            "eval_interval": 20,
            "save_interval": 50,
        },
    )


def test_render_includes_all_spec_section_23_fields():
    module = _load_module()
    card = module.render(_sample_config(), None, None, None, None, None)

    required_headings = [
        "## Architecture",
        "## Languages",
        "## Training Data",
        "## Tokenizer",
        "## Training Configuration",
        "## Training Tokens",
        "## Compute Used",
        "## Evaluation Results",
        "## Known Limitations",
        "## Responsible Use",
        "## License",
    ]
    for heading in required_headings:
        assert heading in card, f"missing section: {heading}"


def test_render_does_not_crash_without_optional_data():
    module = _load_module()
    card = module.render(_sample_config(), None, None, None, None, None)
    assert "No evaluation report found" in card


def test_render_includes_real_architecture_numbers():
    module = _load_module()
    card = module.render(_sample_config(), None, None, None, None, None)
    assert "32" in card  # hidden_size
    assert "1,000" in card  # vocab_size, comma-formatted
    assert "128 tokens" in card  # context window


def test_render_includes_evaluation_data_when_present():
    module = _load_module()
    model_eval = {
        "parameters": 123456,
        "history_summary": {
            "history_available": True,
            "final_val_loss": 5.0,
            "final_perplexity": 148.4,
            "avg_tokens_per_sec": 1000.0,
            "peak_device_memory_gb": 0.5,
        },
        "overfitting": {"verdict": "no overfitting signal — still improving"},
        "memorization": {
            "avg_match_rate": 0.02,
            "num_documents_checked": 10,
            "match_rate_vs_chance": 200.0,
        },
        "generations": {"english": "test output", "khmer": "test", "mixed": "test"},
    }
    card = module.render(_sample_config(), model_eval, None, None, None, None)
    assert "123,456" in card
    assert "5.0000" in card or "5.0" in card
    assert "no overfitting signal" in card


def test_render_reports_real_corpus_token_count_when_available():
    # Regression test for a real bug: the "Training Tokens" and "Known
    # Limitations" sections used to hard-code "~8.25M tokens" / "~3,000
    # Wikipedia documents" — numbers that went silently stale the moment
    # the corpus was expanded (see count_corpus_tokens's docstring).
    module = _load_module()
    corpus_stats = {
        "documents": 15939,
        "characters": 113646816,
        "words": 15021255,
        "khmer_percentage": 49.81,
        "english_percentage": 50.19,
        "split_sizes": {"train": 14345, "val": 796, "test": 798},
        "duplicate_rate_percentage": 0.26,
    }
    card = module.render(_sample_config(), None, corpus_stats, None, None, 26_000_000)
    assert "26,000,000 tokens" in card
    assert "8.25M" not in card
    assert "15,939" in card
    assert "~3,000" not in card


def test_render_handles_missing_corpus_token_count_honestly():
    module = _load_module()
    card = module.render(_sample_config(), None, None, None, None, None)
    assert "unavailable" in card
    assert "8.25M" not in card


def test_count_corpus_tokens_returns_none_for_missing_tokenizer(tmp_path):
    module = _load_module()
    result = module.count_corpus_tokens(tmp_path, tmp_path / "does_not_exist.model")
    assert result is None


def test_count_corpus_tokens_sums_real_tokenized_splits(tmp_path):
    from daralm.tokenizer.train import train_sentencepiece

    module = _load_module()
    corpus_path = tmp_path / "corpus.txt"
    lines = ["hello world this is a test sentence", "another different sentence here"] * 30
    corpus_path.write_text("\n".join(lines), encoding="utf-8")
    model_path = train_sentencepiece(
        corpus_path, tmp_path / "tok", vocab_size=30, model_type="unigram"
    )

    cleaned_dir = tmp_path / "data" / "cleaned"
    cleaned_dir.mkdir(parents=True)
    (cleaned_dir / "train.jsonl").write_text(
        json.dumps({"text": "hello world"}) + "\n", encoding="utf-8"
    )
    (cleaned_dir / "val.jsonl").write_text(
        json.dumps({"text": "this is a test"}) + "\n", encoding="utf-8"
    )

    result = module.count_corpus_tokens(tmp_path / "data", model_path)

    tokenizer = module.DaraLMTokenizer.from_pretrained(model_path)
    expected = len(tokenizer.encode("hello world", add_bos=True, add_eos=True)) + len(
        tokenizer.encode("this is a test", add_bos=True, add_eos=True)
    )
    assert result == expected
    assert result > 0
