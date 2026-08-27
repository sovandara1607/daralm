#!/usr/bin/env python
"""Generate MODEL_CARD.md from real project artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daralm.model.config import ModelConfig  # noqa: E402
from daralm.tokenizer.tokenizer import DaraLMTokenizer  # noqa: E402
from daralm.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="daralm-50m", help="Model name, matching a config file")
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Exact data snapshot used by this checkpoint. Omit rather than using newer data.",
    )
    parser.add_argument(
        "--evaluation-report", type=Path, default=Path("experiments/evaluation_report.json")
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("checkpoints/tokenizer/unigram.model"),
        help="Used to compute the real corpus token count — never hard-coded (see "
        "count_corpus_tokens's docstring for the bug this replaced).",
    )
    parser.add_argument("--output", type=Path, default=Path("MODEL_CARD.md"))
    return parser.parse_args()


def count_corpus_tokens(data_dir: Path, tokenizer_path: Path) -> int | None:
    """Real total token count across the cleaned train/val/test splits."""
    if not tokenizer_path.exists():
        return None
    tokenizer = DaraLMTokenizer.from_pretrained(tokenizer_path)
    total = 0
    for split in ("train", "val", "test"):
        path = data_dir / "cleaned" / f"{split}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                total += len(tokenizer.encode(record["text"], add_bos=True, add_eos=True))
    return total


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def render(
    config: ModelConfig,
    model_eval: dict | None,
    corpus_stats: dict | None,
    manifest: dict | None,
    tokenizer_eval: dict | None,
    corpus_tokens: int | None,
) -> str:
    arch = config.architecture
    train = config.training
    lines: list[str] = []

    lines.append(f"# Model Card: {config.model_name.upper()}")
    lines.append("")
    lines.append(
        "> Part of the DaraLM project — a from-scratch decoder-only Transformer for "
        "Khmer + English, built as an educational/portfolio exercise. **Not a production "
        "or commercial-grade language model.** See the repo README for the full project."
    )
    lines.append("")

    lines.append("## Architecture")
    lines.append("")
    lines.append("Decoder-only Transformer (GPT-style), implemented from scratch:")
    lines.append("- Pre-normalization with RMSNorm")
    lines.append("- Causal multi-head self-attention with Rotary Position Embeddings (RoPE)")
    lines.append("- GELU feed-forward network")
    lines.append("- Weight-tied token embedding / LM head")
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append(f"| Hidden size | {arch.hidden_size} |")
    lines.append(f"| Layers | {arch.num_layers} |")
    lines.append(f"| Attention heads | {arch.num_attention_heads} (head_dim={arch.head_dim}) |")
    lines.append(f"| Feed-forward size | {arch.intermediate_size} |")
    lines.append(f"| Context window | {arch.max_position_embeddings} tokens |")
    lines.append(f"| Vocabulary | {arch.vocab_size:,} tokens |")
    if model_eval:
        lines.append(f"| Parameters | {model_eval['parameters']:,} |")
    lines.append("")

    lines.append("## Languages")
    lines.append("")
    lines.append("Khmer and English, trained on a roughly balanced bilingual corpus.")
    if corpus_stats:
        lines.append(
            f"Training corpus was {corpus_stats['khmer_percentage']}% Khmer, "
            f"{corpus_stats['english_percentage']}% English by document count."
        )
    lines.append("")

    lines.append("## Training Data")
    lines.append("")
    if corpus_stats:
        lines.append(
            f"- {corpus_stats['documents']:,} documents, {corpus_stats['characters']:,} "
            f"characters, {corpus_stats['words']:,} words after cleaning and deduplication"
        )
        lines.append(
            f"- Split: {corpus_stats['split_sizes']['train']:,} train / "
            f"{corpus_stats['split_sizes']['val']:,} val / "
            f"{corpus_stats['split_sizes']['test']:,} test documents"
        )
        lines.append(f"- Duplicate rate: {corpus_stats['duplicate_rate_percentage']}%")
    if manifest:
        lines.append("- Sources:")
        for source in manifest["sources"]:
            lines.append(
                f"  - **{source['source']}**: {source['hf_dataset']} "
                f"({source['hf_config']}), license: {source['license']}, "
                f"{source['documents_fetched']:,} documents fetched"
            )
    lines.append(
        "\nNo instruction-tuning or preference data — this is a base language model, "
        "trained purely on causal next-token prediction over raw Wikipedia text."
    )
    lines.append("")

    lines.append("## Tokenizer")
    lines.append("")
    lines.append(
        f"SentencePiece Unigram, {arch.vocab_size:,} vocabulary, trained on the same corpus "
        "(Phase 2). Special tokens: `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3`."
    )
    if tokenizer_eval:
        recommendation = tokenizer_eval.get("recommendation", "")
        lines.append(f"\nSelected over BPE based on measured evaluation: {recommendation}")
    lines.append("")

    lines.append("## Training Configuration")
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append(f"| Optimizer | {train.optimizer} |")
    lines.append(f"| Learning rate | {train.learning_rate} (cosine decay after warmup) |")
    lines.append(f"| Weight decay | {train.weight_decay} |")
    lines.append(f"| Precision | {train.precision} |")
    batch_desc = f"{train.batch_size} × {train.gradient_accumulation_steps}"
    lines.append(f"| Batch size (micro / accumulation) | {batch_desc} |")
    lines.append(f"| Max steps (this checkpoint) | {train.max_steps} |")
    lines.append(f"| Warmup steps | {train.warmup_steps} |")
    lines.append(f"| Seed | {train.seed} |")
    lines.append("")

    tokens_trained = (
        train.batch_size
        * train.gradient_accumulation_steps
        * arch.max_position_embeddings
        * train.max_steps
    )
    effective_batch = train.batch_size * train.gradient_accumulation_steps
    lines.append("## Training Tokens")
    lines.append("")
    if corpus_tokens:
        passes = tokens_trained / corpus_tokens
        if passes < 1:
            pass_desc = "well under one full pass"
        elif passes < 2:
            pass_desc = "a bit over one full pass"
        else:
            pass_desc = "multiple passes"
        lines.append(
            f"~{tokens_trained:,} tokens seen ({effective_batch} sequences × "
            f"{arch.max_position_embeddings} tokens/sequence × {train.max_steps} optimizer "
            f"steps). The training corpus is ~{corpus_tokens:,} tokens at this tokenizer's "
            f"real, measured compression rate (not estimated) — so this checkpoint has seen "
            f"roughly {passes:.2f}x the corpus, {pass_desc}."
        )
    else:
        lines.append(
            f"~{tokens_trained:,} tokens seen ({effective_batch} sequences × "
            f"{arch.max_position_embeddings} tokens/sequence × {train.max_steps} optimizer "
            "steps). Corpus token count unavailable (tokenizer not found) — re-run with "
            "--tokenizer pointing at a valid model file for a full picture."
        )
    lines.append("")

    lines.append("## Compute Used")
    lines.append("")
    if model_eval and model_eval["history_summary"].get("history_available"):
        hist = model_eval["history_summary"]
        lines.append("- Device: Apple Silicon (MPS) — a consumer laptop, not a training cluster")
        lines.append(f"- Throughput: avg {hist['avg_tokens_per_sec']:.0f} tokens/sec")
        lines.append(f"- Peak device memory: {hist['peak_device_memory_gb']:.2f} GB")
        lines.append(
            "- Batch size was set empirically after direct benchmarking found a severe "
            "MPS-backend performance cliff at larger batch sizes for this model size "
            "(see README Phase 7) — not a guess."
        )
    lines.append("")

    lines.append("## Evaluation Results")
    lines.append("")
    if model_eval:
        hist = model_eval["history_summary"]
        lines.append(
            f"- Final validation loss: {hist.get('final_val_loss', 'n/a'):.4f}, "
            f"perplexity: {hist.get('final_perplexity', 'n/a'):.1f}"
        )
        lines.append(f"- Overfitting check: {model_eval['overfitting']['verdict']}")
        mem = model_eval["memorization"]
        lines.append(
            f"- Memorization check (spec section 15): {mem['avg_match_rate']:.4f} average "
            f"token-match rate over {mem['num_documents_checked']} real training documents "
            f"— {mem['match_rate_vs_chance']:.0f}x the random-chance baseline, but still low "
            "in absolute terms; no concerning verbatim memorization detected at this scale/"
            "step count."
        )
        lines.append("- Fixed-prompt generation samples (temperature=0.8, top-p=0.9):")
        for domain, text in model_eval["generations"].items():
            lines.append(f"  - **{domain}**: `{text[:150]}{'...' if len(text) > 150 else ''}`")
    else:
        lines.append(
            "*No evaluation report found — run `scripts/evaluate.py` before regenerating "
            "this card.*"
        )
    lines.append("")

    lines.append("## Known Limitations")
    lines.append("")
    lines.append(
        "- **Not fluent.** At this step count, generation is grammatically fragmentary in "
        "both languages — real words and some real morphology/particles, not coherent "
        "sentences or paragraphs. This is expected at this scale, not a bug."
    )
    corpus_doc_desc = f"~{corpus_stats['documents']:,}" if corpus_stats else "a small number of"
    lines.append(
        f"- **Small corpus.** {corpus_doc_desc} Wikipedia documents is a tiny fraction of what "
        "production LLMs train on. Facts, if any appear, should not be trusted."
    )
    lines.append(
        "- **Language drift.** Generation does not reliably stay in the prompt's language "
        "for its full length — an English prompt can drift into Khmer partway through."
    )
    lines.append(
        "- **No instruction-following.** This is a base model; it continues text, it does "
        "not follow instructions or answer questions reliably."
    )
    lines.append(
        "- **No safety tuning.** No RLHF, no content filtering, no red-teaming has been performed."
    )
    lines.append("")

    lines.append("## Responsible Use")
    lines.append("")
    lines.append(
        "This model is a **learning artifact**, built to understand the LLM training "
        "pipeline end to end. It is not evaluated or intended for factual, medical, legal, "
        "financial, or safety-critical use of any kind. Outputs should not be presented to "
        "end users as authoritative."
    )
    lines.append("")

    lines.append("## License")
    lines.append("")
    lines.append(
        "Code: MIT (see repository LICENSE). Training data: Wikipedia text under "
        "CC BY-SA 4.0 / GFDL — see `data/raw/MANIFEST.json` for exact sources. Model "
        "weights inherit the CC BY-SA 4.0 share-alike terms of the training data."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    config_path = args.configs_dir / f"{args.model.removeprefix('daralm-')}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config found at {config_path}")
    config = ModelConfig.from_yaml(config_path)

    eval_report = load_json(args.evaluation_report)
    model_eval = eval_report["models"].get(args.model) if eval_report else None
    if model_eval is None:
        logger.warning(
            "No evaluation data for %s in %s — run scripts/evaluate.py first for full results",
            args.model,
            args.evaluation_report,
        )

    corpus_stats = None
    manifest = None
    corpus_tokens = None
    if args.data_dir is not None:
        corpus_stats = load_json(args.data_dir / "cleaned" / "stats.json")
        manifest = load_json(args.data_dir / "raw" / "MANIFEST.json")
        corpus_tokens = count_corpus_tokens(args.data_dir, args.tokenizer)
        if corpus_tokens is None:
            logger.warning("Tokenizer not found at %s — corpus token count omitted", args.tokenizer)
    else:
        logger.warning(
            "No --data-dir supplied; omitting corpus claims to avoid attributing newer data "
            "to an older checkpoint"
        )
    tokenizer_eval = load_json(args.checkpoints_dir / "tokenizer" / "evaluation_report.json")

    card = render(config, model_eval, corpus_stats, manifest, tokenizer_eval, corpus_tokens)
    args.output.write_text(card, encoding="utf-8")
    print(f"Model card written to: {args.output}")


if __name__ == "__main__":
    main()
