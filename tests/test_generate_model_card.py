"""Tests for scripts/generate_model_card.py's render() — a pure function,
so testable without touching real project artifacts on disk.
"""

from __future__ import annotations

import importlib.util
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
    card = module.render(_sample_config(), None, None, None, None)

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
    card = module.render(_sample_config(), None, None, None, None)
    assert "No evaluation report found" in card


def test_render_includes_real_architecture_numbers():
    module = _load_module()
    card = module.render(_sample_config(), None, None, None, None)
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
    card = module.render(_sample_config(), model_eval, None, None, None)
    assert "123,456" in card
    assert "5.0000" in card or "5.0" in card
    assert "no overfitting signal" in card
