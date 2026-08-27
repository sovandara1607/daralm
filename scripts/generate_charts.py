#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import subprocess  # noqa: E402

_real_check_output = subprocess.check_output


def _check_output_skip_macos_font_scan(args, *popen_args, **kwargs):
    if isinstance(args, list) and args[:1] == ["system_profiler"]:
        raise subprocess.CalledProcessError(1, args)
    return _real_check_output(args, *popen_args, **kwargs)


subprocess.check_output = _check_output_skip_macos_font_scan

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless — no display backend needed to write PNGs
import matplotlib.pyplot as plt  # noqa: E402

subprocess.check_output = _real_check_output  # restore — only the import needed patching

from daralm.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

CHECKPOINT_ROOT = Path("checkpoints")


CLASSIFICATION_EPOCHS = [1, 2, 3]
CLASSIFICATION_VAL_ACCURACY = [0.9786, 0.9837, 0.9837]

GRAMMAR_ORIGINAL_STEPS = [50, 100, 150, 200, 250, 300, 350, 400]
GRAMMAR_ORIGINAL_PERPLEXITY = [60.5, 56.1, 53.8, 52.0, 51.1, 50.4, 50.0, 49.7]

GRAMMAR_CER = {
    "Original\n(400 steps, 3K)": 1.2268,
    "Scaled up\n(2400 steps, 12K)": 0.6205,
    "+ No-op subset\n(2400 steps, 12K)": 0.5304,
    "No-op\nbaseline": 0.0205,
}
GRAMMAR_WER = {
    "Original\n(400 steps, 3K)": 1.4216,
    "Scaled up\n(2400 steps, 12K)": 0.8872,
    "+ No-op subset\n(2400 steps, 12K)": 0.7893,
    "No-op\nbaseline": 0.0915,
}

JSON_METRICS_V1 = {"Valid JSON": 1.0, "Schema match": 1.0, "Exact value match": 1.0}
JSON_METRICS_V2 = {"Valid JSON": 1.0, "Schema match": 0.905, "Exact value match": 0.0}
JSON_METRICS_V3 = {"Valid JSON": 0.855, "Schema match": 0.855, "Exact value match": 0.42}

JSON_NOVEL_VOCAB_V2 = {"age": 0.9333, "name": 0.0, "occupation": 0.0, "city": 0.0}
JSON_NOVEL_VOCAB_V3_REALWORDS = {"age": 0.8667, "name": 0.0, "occupation": 0.0, "city": 0.0}

INSTRUCT_PERPLEXITY = {"Original\n(784 examples)": 110.7, "Retrained\n(2,313 examples)": 79.98}


def _load_history(model_name: str) -> list[dict] | None:
    path = CHECKPOINT_ROOT / model_name / "history.json"
    if not path.exists():
        logger.warning("No history.json for %s at %s — skipping", model_name, path)
        return None
    with path.open() as f:
        return json.load(f)


def _val_series(history: list[dict], key: str) -> tuple[list[int], list[float]]:
    steps, values = [], []
    for entry in history:
        if key in entry:
            steps.append(entry["step"])
            values.append(entry[key])
    return steps, values


def chart_pretraining_perplexity(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model_name, label, color in [
        ("daralm-10m", "DaraLM-10M", "#2a78d6"),
        ("daralm-50m", "DaraLM-50M", "#1baf7a"),
        ("daralm-150m", "DaraLM-150M", "#eb6834"),
    ]:
        history = _load_history(model_name)
        if history is None:
            continue
        steps, ppl = _val_series(history, "perplexity")
        if not steps:
            continue
        ax.plot(steps, ppl, marker="o", markersize=3, label=label, color=color, linewidth=2)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation perplexity")
    ax.set_title("Base pretraining: validation perplexity (real history.json data)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "pretraining_perplexity.png", dpi=150)
    plt.close(fig)


def chart_sft_perplexity(output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    configs = [
        ("daralm-50m-instruct", "50M-Instruct", "#2a78d6"),
        ("daralm-50m-grammar", "50M-Grammar\n(scaled-up run)", "#eb6834"),
        ("daralm-50m-json", "50M-JSON\n(v2, disjoint vocab)", "#1baf7a"),
    ]
    for ax, (model_name, label, color) in zip(axes, configs, strict=True):
        history = _load_history(model_name)
        if history is None:
            ax.set_title(f"{label}\n(no history.json found)")
            continue
        steps, ppl = _val_series(history, "perplexity")
        ax.plot(steps, ppl, marker="o", markersize=3, color=color, linewidth=2)
        ax.set_title(label)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Validation perplexity")
        ax.grid(alpha=0.3)

    fig.suptitle("SFT runs: validation perplexity (real history.json data)")
    fig.tight_layout()
    fig.savefig(output_dir / "sft_perplexity.png", dpi=150)
    plt.close(fig)


def chart_classification_accuracy(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(
        CLASSIFICATION_EPOCHS,
        [a * 100 for a in CLASSIFICATION_VAL_ACCURACY],
        marker="o",
        markersize=7,
        color="#2a78d6",
        linewidth=2,
    )
    for epoch, acc in zip(CLASSIFICATION_EPOCHS, CLASSIFICATION_VAL_ACCURACY, strict=True):
        ax.annotate(
            f"{acc:.2%}",
            (epoch, acc * 100),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )
    ax.set_xticks(CLASSIFICATION_EPOCHS)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Khmer/English language classification")
    ax.set_ylim(96, 100)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "classification_accuracy.png", dpi=150)
    plt.close(fig)


def chart_grammar_original_curve(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        GRAMMAR_ORIGINAL_STEPS,
        GRAMMAR_ORIGINAL_PERPLEXITY,
        marker="o",
        color="#d03b3b",
        linewidth=2,
    )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation perplexity")
    ax.set_title("Grammar correction: original run (400 steps, 3K examples)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "grammar_original_curve.png", dpi=150)
    plt.close(fig)


def chart_grammar_cer_wer(output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    chart_specs = [
        (axes[0], GRAMMAR_CER, "Character error rate"),
        (axes[1], GRAMMAR_WER, "Word error rate"),
    ]
    colors = ["#d03b3b", "#eb6834", "#e6c619", "#1baf7a"]
    for ax, data, title in chart_specs:
        labels = list(data.keys())
        values = list(data.values())
        bars = ax.bar(labels, values, color=colors[: len(labels)])
        ax.bar_label(bars, fmt="%.3f")
        ax.set_title(title)
        ax.set_ylabel("Error rate (lower is better)")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Grammar correction: three real attempts, real held-out evaluation")
    fig.tight_layout()
    fig.savefig(output_dir / "grammar_cer_wer.png", dpi=150)
    plt.close(fig)


def chart_json_v1_vs_v2(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = list(JSON_METRICS_V1.keys())
    x = range(len(labels))
    width = 0.26
    v1_vals = [JSON_METRICS_V1[k] * 100 for k in labels]
    v2_vals = [JSON_METRICS_V2[k] * 100 for k in labels]
    v3_vals = [JSON_METRICS_V3[k] * 100 for k in labels]
    bars1 = ax.bar(
        [i - width for i in x], v1_vals, width, label="v1 (shared vocab)", color="#d03b3b"
    )
    bars2 = ax.bar([i for i in x], v2_vals, width, label="v2 (disjoint vocab)", color="#eb6834")
    bars3 = ax.bar(
        [i + width for i in x], v3_vals, width, label="v3 (unbounded random)", color="#1baf7a"
    )
    for bars in (bars1, bars2, bars3):
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Structured JSON generation: dataset v1 vs v2 vs v3 (real, official eval)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "json_v1_vs_v2.png", dpi=150)
    plt.close(fig)


def chart_json_novel_vocab_breakdown(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = JSON_NOVEL_VOCAB_V3_REALWORDS
    labels = list(data.keys())
    values = [v * 100 for v in data.values()]
    colors = ["#1baf7a" if v > 0.5 else "#d03b3b" for v in data.values()]
    bars = ax.bar(labels, values, color=colors)
    ax.bar_label(bars, fmt="%.0f%%")
    ax.set_ylabel("Correct (%)")
    ax.set_title("Structured JSON v3: real-word distribution-shift check (n=30)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "json_novel_vocab_breakdown.png", dpi=150)
    plt.close(fig)


def chart_instruct_perplexity_comparison(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels = list(INSTRUCT_PERPLEXITY.keys())
    values = list(INSTRUCT_PERPLEXITY.values())
    bars = ax.bar(labels, values, color=["#d03b3b", "#eb6834"])
    ax.bar_label(bars, fmt="%.1f")
    ax.set_ylabel("Final validation perplexity (lower is better)")
    ax.set_title("DaraLM-50M-Instruct validation perplexity")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "instruct_perplexity_comparison.png", dpi=150)
    plt.close(fig)


def chart_final_perplexity_summary(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    model_names = [
        "daralm-10m",
        "daralm-50m",
        "daralm-150m",
        "daralm-50m-instruct",
        "daralm-50m-grammar",
        "daralm-50m-json",
    ]
    labels, values = [], []
    for model_name in model_names:
        history = _load_history(model_name)
        if history is None:
            continue
        _, ppl = _val_series(history, "perplexity")
        if not ppl:
            continue
        labels.append(model_name.replace("daralm-", ""))
        values.append(ppl[-1])

    bars = ax.bar(labels, values, color="#2a78d6")
    ax.bar_label(bars, fmt="%.1f")
    ax.set_ylabel("Final validation perplexity (lower is better)")
    ax.set_title("All trained models: final val perplexity (real history.json data)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)  # after tight_layout, which would reset it
    fig.text(
        0.5,
        0.02,
        "Not directly comparable: base pretraining vs. narrow SFT tasks have very different "
        "label entropy — a low SFT perplexity reflects task narrowness, not model quality.",
        ha="center",
        fontsize=8,
        style="italic",
        color="#666666",
    )
    fig.savefig(output_dir / "final_perplexity_summary.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("charts"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    chart_fns = [
        chart_pretraining_perplexity,
        chart_sft_perplexity,
        chart_classification_accuracy,
        chart_grammar_original_curve,
        chart_grammar_cer_wer,
        chart_json_v1_vs_v2,
        chart_json_novel_vocab_breakdown,
        chart_instruct_perplexity_comparison,
        chart_final_perplexity_summary,
    ]
    for fn in chart_fns:
        fn(args.output_dir)
        logger.info("Wrote %s", fn.__name__)

    print()
    print(f"=== Wrote {len(chart_fns)} charts to {args.output_dir}/ ===")
    for path in sorted(args.output_dir.glob("*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
