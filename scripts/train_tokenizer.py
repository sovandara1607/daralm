#!/usr/bin/env python
"""Train and evaluate DaraLM's Khmer+English tokenizer (Phase 2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daralm.data.loader import load_jsonl  # noqa: E402
from daralm.tokenizer.evaluation import (  # noqa: E402
    evaluate_tokenizer,
    load_domain_examples,
    show_example_tokenization,
)
from daralm.tokenizer.tokenizer import DaraLMTokenizer  # noqa: E402
from daralm.tokenizer.train import build_corpus_file, train_sentencepiece  # noqa: E402
from daralm.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

MANUAL_EXAMPLE_TEXTS = [
    "Cambodia is a country in Southeast Asia.",
    "កម្ពុជាជាប្រទេសមួយនៅអាស៊ីអាគ្នេយ៍។",
    "ខ្ញុំចង់រៀន machine learning និង deep learning។",
    "https://huggingface.co/datasets/wikimedia/wikipedia",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned-dir", type=Path, default=Path("data/cleaned"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/tokenizer"))
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=16000,
        help="Shared vocab size for both BPE and Unigram (spec range: 16,000-32,000)",
    )
    return parser.parse_args()


def recommend(bpe_results: dict, unigram_results: dict) -> str:
    """Pick a winner from measured evaluation numbers, with a stated reason."""
    domains = list(bpe_results.keys())

    def score(results: dict) -> float:
        avg_compression = sum(results[d]["avg_chars_per_token"] for d in domains) / len(domains)
        avg_unk = sum(results[d]["unk_rate_percentage"] for d in domains) / len(domains)
        return avg_compression - avg_unk

    bpe_score = score(bpe_results)
    unigram_score = score(unigram_results)
    winner = "bpe" if bpe_score >= unigram_score else "unigram"
    return (
        f"{winner} (score {max(bpe_score, unigram_score):.3f} vs "
        f"{min(bpe_score, unigram_score):.3f} — score = avg chars/token compression "
        f"across domains, penalized by avg unk rate %)"
    )


def print_comparison(bpe_results: dict, unigram_results: dict) -> None:
    print()
    print("=== Tokenizer Comparison ===")
    header = f"{'domain':<20}{'metric':<28}{'bpe':>12}{'unigram':>12}"
    print(header)
    print("-" * len(header))
    for domain in bpe_results:
        for metric in bpe_results[domain]:
            if metric == "num_examples":
                continue
            b = bpe_results[domain][metric]
            u = unigram_results[domain][metric]
            print(f"{domain:<20}{metric:<28}{b:>12}{u:>12}")


def main() -> None:
    args = parse_args()

    corpus_path = args.output_dir / "corpus.txt"
    build_corpus_file(args.cleaned_dir / "train.jsonl", corpus_path)

    bpe_model_path = train_sentencepiece(
        corpus_path, args.output_dir / "bpe", args.vocab_size, model_type="bpe"
    )
    unigram_model_path = train_sentencepiece(
        corpus_path, args.output_dir / "unigram", args.vocab_size, model_type="unigram"
    )

    bpe_tokenizer = DaraLMTokenizer.from_pretrained(bpe_model_path)
    unigram_tokenizer = DaraLMTokenizer.from_pretrained(unigram_model_path)

    # Sample real Khmer/English text from the cleaned corpus for domain eval
    khmer_records = [
        r for r in load_jsonl(args.cleaned_dir / "val.jsonl") if r["language"] == "km"
    ] or [r for r in load_jsonl(args.cleaned_dir / "train.jsonl") if r["language"] == "km"]
    english_records = [
        r for r in load_jsonl(args.cleaned_dir / "val.jsonl") if r["language"] == "en"
    ] or [r for r in load_jsonl(args.cleaned_dir / "train.jsonl") if r["language"] == "en"]
    domain_examples = load_domain_examples(khmer_records, english_records)

    bpe_results = evaluate_tokenizer(bpe_tokenizer, domain_examples)
    unigram_results = evaluate_tokenizer(unigram_tokenizer, domain_examples)

    print_comparison(bpe_results, unigram_results)
    recommendation = recommend(bpe_results, unigram_results)
    print()
    print(f"Recommended: {recommendation}")

    print()
    print("=== Manual Tokenization Examples (Unigram) ===")
    manual_examples = []
    for text in MANUAL_EXAMPLE_TEXTS:
        example = show_example_tokenization(unigram_tokenizer, text)
        manual_examples.append(example)
        print(f"\nText:   {example['text']}")
        print(f"Pieces ({example['num_pieces']}): {example['pieces']}")

    report = {
        "vocab_size": args.vocab_size,
        "bpe": bpe_results,
        "unigram": unigram_results,
        "recommendation": recommendation,
        "manual_examples": manual_examples,
    }
    report_path = args.output_dir / "evaluation_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nFull evaluation report written to: {report_path}")


if __name__ == "__main__":
    main()
