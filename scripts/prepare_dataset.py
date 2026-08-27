#!/usr/bin/env python
"""Build the DaraLM raw + cleaned dataset from public Khmer/English sources."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daralm.data.dataset import split_dataset  # noqa: E402
from daralm.data.deduplication import deduplicate  # noqa: E402
from daralm.data.loader import fetch_wikipedia_sample, load_jsonl, save_jsonl  # noqa: E402
from daralm.data.preprocessing import compute_corpus_stats, process_document  # noqa: E402
from daralm.utils.logging import get_logger  # noqa: E402
from daralm.utils.seed import set_seed  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--languages", nargs="+", default=["km", "en"], help="Wikipedia language codes to fetch"
    )
    parser.add_argument(
        "--docs-per-language", type=int, default=1000, help="Documents to fetch per language"
    )
    parser.add_argument(
        "--min-chars", type=int, default=200, help="Skip raw articles shorter than this"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="Root data/ dir")
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Reuse existing data/raw/*.jsonl instead of re-fetching from Hugging Face",
    )
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def fetch_raw(languages: list[str], n_docs: int, min_chars: int, raw_dir: Path) -> list[dict]:
    all_records: list[dict] = []
    manifest_entries = []

    for language in languages:
        records, manifest_entry = fetch_wikipedia_sample(language, n_docs, min_chars=min_chars)
        save_jsonl(records, raw_dir / f"wikipedia_{language}.jsonl")
        all_records.extend(records)
        manifest_entries.append(manifest_entry)

    manifest = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "sources": manifest_entries,
    }
    manifest_path = raw_dir / "MANIFEST.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info("Wrote manifest for %d source(s) to %s", len(manifest_entries), manifest_path)
    return all_records


def load_existing_raw(languages: list[str], raw_dir: Path) -> list[dict]:
    all_records: list[dict] = []
    for language in languages:
        path = raw_dir / f"wikipedia_{language}.jsonl"
        records = load_jsonl(path)
        logger.info("Loaded %d existing raw records from %s", len(records), path)
        all_records.extend(records)
    return all_records


def clean_and_filter(raw_records: list[dict]) -> tuple[list[dict], dict[str, int], list[dict]]:
    """Run every raw record through cleaning + quality filtering."""
    kept: list[dict] = []
    reason_counts: dict[str, int] = {}
    filtered_examples: list[dict] = []

    for record in raw_records:
        cleaned, reason = process_document(record)
        if cleaned is not None:
            kept.append(cleaned)
            continue
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if len(filtered_examples) < 20:
            filtered_examples.append(
                {
                    "reason": reason,
                    "source": record.get("source", "unknown"),
                    "text_preview": (record.get("text") or "")[:200],
                }
            )

    return kept, reason_counts, filtered_examples


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    raw_dir = args.output_dir / "raw"
    cleaned_dir = args.output_dir / "cleaned"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_fetch:
        logger.info("Skipping fetch; loading existing raw data from %s", raw_dir)
        raw_records = load_existing_raw(args.languages, raw_dir)
    else:
        logger.info("Fetching %d docs/language for %s", args.docs_per_language, args.languages)
        raw_records = fetch_raw(args.languages, args.docs_per_language, args.min_chars, raw_dir)

    total_seen = len(raw_records)
    logger.info("Total raw documents: %d", total_seen)

    cleaned_records, reason_counts, filtered_examples = clean_and_filter(raw_records)
    logger.info("Kept %d / %d after cleaning + filtering", len(cleaned_records), total_seen)

    deduped_records, duplicate_count = deduplicate(cleaned_records)
    logger.info(
        "Removed %d duplicate(s); %d unique documents remain", duplicate_count, len(deduped_records)
    )

    splits = split_dataset(
        deduped_records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    for split_name, split_records in splits.items():
        split_path = cleaned_dir / f"{split_name}.jsonl"
        save_jsonl(split_records, split_path)
        logger.info("Wrote %d records to %s", len(split_records), split_path)

    stats = compute_corpus_stats(
        deduped_records, duplicate_count=duplicate_count, total_seen=total_seen
    )
    stats["split_sizes"] = {name: len(records) for name, records in splits.items()}
    stats["filtered_reason_counts"] = reason_counts
    stats_path = cleaned_dir / "stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    quality_report = {
        "filtered_reason_counts": reason_counts,
        "sample_filtered_documents": filtered_examples,
    }
    quality_report_path = cleaned_dir / "quality_report.json"
    with quality_report_path.open("w", encoding="utf-8") as f:
        json.dump(quality_report, f, ensure_ascii=False, indent=2)

    lang_pcts = (
        f"{stats['khmer_percentage']} / {stats['english_percentage']} / {stats['mixed_percentage']}"
    )
    print()
    print("=== Dataset Preparation Summary ===")
    print(f"Raw documents fetched:      {total_seen}")
    print(f"Dropped by quality filter:  {sum(reason_counts.values())} {reason_counts or ''}")
    print(f"Duplicates removed:         {duplicate_count}")
    print(f"Final unique documents:     {len(deduped_records)}")
    print(f"Split sizes:                {stats['split_sizes']}")
    print(f"Khmer / English / Mixed %:  {lang_pcts}")
    print(f"Average document length:    {stats['average_document_chars']} chars")
    print(f"Duplicate rate:             {stats['duplicate_rate_percentage']}%")
    print(f"Stats written to:           {stats_path}")
    print(f"Quality report written to:  {quality_report_path}")


if __name__ == "__main__":
    main()
