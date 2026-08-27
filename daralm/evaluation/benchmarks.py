from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from daralm.evaluation.generation import average_match_rate, check_memorization
from daralm.inference.generator import generate
from daralm.model.config import ModelConfig
from daralm.model.transformer import DaraLMTransformer
from daralm.tokenizer.tokenizer import DaraLMTokenizer
from daralm.training.checkpoint import load_checkpoint
from daralm.utils.logging import get_logger

logger = get_logger(__name__)

FIXED_PROMPTS = {
    "english": "Cambodia is a country in",
    "khmer": "កម្ពុជាជាប្រទេសមួយនៅ",
    "mixed": "ខ្ញុំចង់រៀន machine learning និង",
}


def summarize_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce a Trainer history.json timeline to headline numbers."""
    train_entries = [e for e in history if "train_loss" in e]
    val_entries = [e for e in history if "val_loss" in e]

    summary: dict[str, Any] = {
        "history_available": True,
        "num_log_events": len(train_entries),
        "num_eval_events": len(val_entries),
    }
    if train_entries:
        summary["initial_train_loss"] = train_entries[0]["train_loss"]
        summary["final_train_loss"] = train_entries[-1]["train_loss"]
        tokens_per_sec = [e["tokens_per_sec"] for e in train_entries]
        summary["avg_tokens_per_sec"] = sum(tokens_per_sec) / len(tokens_per_sec)
        summary["max_tokens_per_sec"] = max(tokens_per_sec)
        memory_readings = [
            e["gpu_memory_gb"] for e in train_entries if e.get("gpu_memory_gb") is not None
        ]
        summary["peak_device_memory_gb"] = max(memory_readings) if memory_readings else None
    if val_entries:
        summary["initial_val_loss"] = val_entries[0]["val_loss"]
        summary["final_val_loss"] = val_entries[-1]["val_loss"]
        summary["best_val_loss"] = min(e["val_loss"] for e in val_entries)
        summary["final_perplexity"] = val_entries[-1]["perplexity"]
        summary["best_perplexity"] = min(e["perplexity"] for e in val_entries)
    return summary


def load_history_or_meta(checkpoint_dir: str | Path) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    history_path = checkpoint_dir / "history.json"
    if history_path.exists():
        with history_path.open() as f:
            history = json.load(f)
        return summarize_history(history)

    meta_path = checkpoint_dir / "best" / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Neither history.json nor best/meta.json found under {checkpoint_dir} — "
            "is this a valid checkpoint directory produced by scripts/train.py?"
        )
    with meta_path.open() as f:
        meta = json.load(f)
    return {
        "history_available": False,
        "note": "No history.json (checkpoint predates Phase 6's timeline tracking); "
        "only final step's metrics are available.",
        "final_val_loss": meta.get("val_loss"),
        "final_perplexity": meta.get("perplexity"),
    }


def check_overfitting(history_summary: dict[str, Any]) -> dict[str, Any]:
    """Compare training loss to validation loss."""
    if not history_summary.get("history_available"):
        return {"verdict": "insufficient data", "reason": history_summary.get("note", "")}

    if "final_train_loss" not in history_summary or "final_val_loss" not in history_summary:
        return {"verdict": "insufficient data", "reason": "missing train or val loss entries"}

    gap = history_summary["final_val_loss"] - history_summary["final_train_loss"]
    final_is_best = history_summary["final_val_loss"] <= history_summary["best_val_loss"] + 1e-9

    if not final_is_best:
        verdict = "early overfitting signal — validation loss rose from its best"
    elif gap > 1.0:
        verdict = "large train/val gap, but validation still improving — watch, not yet overfitting"
    else:
        verdict = "no overfitting signal — validation tracking training, still improving"

    return {
        "verdict": verdict,
        "final_train_val_gap": gap,
        "final_val_is_best_val": final_is_best,
    }


def evaluate_checkpoint(
    checkpoint_dir: str | Path,
    tokenizer: DaraLMTokenizer,
    device: torch.device,
    real_documents: list[str],
    max_new_tokens: int = 60,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> dict[str, Any]:
    """Full Phase 8 evaluation for one trained checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    best_dir = checkpoint_dir / "best"
    config = ModelConfig.from_yaml(best_dir / "config.yaml")

    model = DaraLMTransformer(config.architecture, pad_token_id=tokenizer.pad_id)
    load_checkpoint(best_dir, model, tokenizer_path=None)  # tokenizer already verified by caller
    model.to(device)
    model.eval()

    history_summary = load_history_or_meta(checkpoint_dir)
    overfitting = check_overfitting(history_summary)

    generations = {}
    for domain, prompt in FIXED_PROMPTS.items():
        generations[domain] = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    memorization_results = check_memorization(model, tokenizer, real_documents)
    avg_match = average_match_rate(memorization_results)
    chance_baseline = 1.0 / config.architecture.vocab_size

    return {
        "model_name": config.model_name,
        "parameters": model.num_parameters(),
        "architecture": {
            "hidden_size": config.architecture.hidden_size,
            "num_layers": config.architecture.num_layers,
            "num_attention_heads": config.architecture.num_attention_heads,
            "max_position_embeddings": config.architecture.max_position_embeddings,
            "vocab_size": config.architecture.vocab_size,
        },
        "history_summary": history_summary,
        "overfitting": overfitting,
        "generations": generations,
        "memorization": {
            "num_documents_checked": len(memorization_results),
            "avg_match_rate": avg_match,
            "chance_baseline": chance_baseline,
            "match_rate_vs_chance": (avg_match / chance_baseline) if chance_baseline else None,
        },
    }


def compare_checkpoints(
    checkpoint_dirs: list[str | Path],
    tokenizer: DaraLMTokenizer,
    device: torch.device,
    real_documents: list[str],
    **kwargs: Any,
) -> dict[str, Any]:
    results = {}
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_dir = Path(checkpoint_dir)
        logger.info("Evaluating %s", checkpoint_dir)
        result = evaluate_checkpoint(checkpoint_dir, tokenizer, device, real_documents, **kwargs)
        results[result["model_name"]] = result
    return {"models": results}
