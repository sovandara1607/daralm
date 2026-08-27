#!/usr/bin/env python
"""Merge verified Common Crawl Khmer records into existing cleaned splits."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daralm.data.dataset import split_dataset  # noqa: E402
from daralm.data.deduplication import document_hash  # noqa: E402
from daralm.data.loader import load_jsonl, save_jsonl  # noqa: E402
from daralm.data.preprocessing import compute_corpus_stats, process_document  # noqa: E402


def prepare_additions(
    raw_records: list[dict[str, Any]], existing_splits: dict[str, list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Clean raw records and reject duplicates across every existing split."""
    seen = {
        document_hash(record["text"]) for records in existing_splits.values() for record in records
    }
    additions: list[dict[str, Any]] = []
    report = {"raw": len(raw_records), "filtered": 0, "duplicates": 0}
    for raw in raw_records:
        cleaned, _ = process_document(raw)
        if cleaned is None:
            report["filtered"] += 1
            continue
        digest = document_hash(cleaned["text"])
        if digest in seen:
            report["duplicates"] += 1
            continue
        seen.add(digest)
        additions.append(
            {
                **cleaned,
                "url": raw.get("url"),
                "crawl": raw.get("crawl"),
            }
        )
    report["accepted"] = len(additions)
    return additions, report


def merge_splits(
    existing_splits: dict[str, list[dict[str, Any]]],
    additions: list[dict[str, Any]],
    *,
    seed: int = 42,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    new_splits = split_dataset(additions, seed=seed)
    merged = {
        name: [*existing_splits[name], *new_splits[name]] for name in ("train", "val", "test")
    }
    return merged, {name: len(records) for name, records in new_splits.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/commoncrawl_km.jsonl"))
    parser.add_argument("--cleaned-dir", type=Path, default=Path("data/cleaned"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    names = ("train", "val", "test")
    existing = {name: load_jsonl(args.cleaned_dir / f"{name}.jsonl") for name in names}
    additions, quality = prepare_additions(load_jsonl(args.raw), existing)
    if not additions:
        raise RuntimeError("No new Common Crawl documents survived cleaning and deduplication")
    merged, added_by_split = merge_splits(existing, additions, seed=args.seed)

    print(f"Accepted {quality['accepted']} / {quality['raw']} Common Crawl documents")
    print(f"Rejected: filtered={quality['filtered']}, duplicates={quality['duplicates']}")
    print(f"Additions by split: {added_by_split}")
    print(f"Final split sizes: {dict((name, len(merged[name])) for name in names)}")
    if args.dry_run:
        print("Dry run: no files changed")
        return

    temporary: dict[str, Path] = {}
    for name in names:
        temp_path = args.cleaned_dir / f".{name}.jsonl.commoncrawl-merge.tmp"
        save_jsonl(merged[name], temp_path)
        temporary[name] = temp_path
    for name in names:
        temporary[name].replace(args.cleaned_dir / f"{name}.jsonl")

    all_records = [record for name in names for record in merged[name]]
    stats_path = args.cleaned_dir / "stats.json"
    previous_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats = compute_corpus_stats(all_records)
    stats["duplicate_rate_percentage"] = previous_stats.get("duplicate_rate_percentage", 0.0)
    stats["filtered_reason_counts"] = previous_stats.get("filtered_reason_counts", {})
    stats["split_sizes"] = {name: len(merged[name]) for name in names}
    stats["note"] = (
        "Token-based stats require a trained tokenizer. Duplicate rate is retained from the "
        "original Wikipedia preparation; see merges for Common Crawl-specific rejection counts."
    )
    stats["merges"] = [
        *previous_stats.get("merges", []),
        {
            "merged_at": datetime.now(timezone.utc).isoformat(),
            "source": str(args.raw),
            "seed": args.seed,
            "quality": quality,
            "added_by_split": added_by_split,
        },
    ]
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest_path = args.cleaned_dir / "MANIFEST_commoncrawl_merge.json"
    manifest = {
        "merged_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(args.raw),
        "commoncrawl_manifest": "data/raw/MANIFEST_commoncrawl_km.json",
        "seed": args.seed,
        "quality": quality,
        "added_by_split": added_by_split,
        "final_split_sizes": stats["split_sizes"],
        "method": "Preserved existing splits; split only new records 90/5/5 and appended them.",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Merge manifest: {manifest_path}")


if __name__ == "__main__":
    main()
