#!/usr/bin/env python
"""Build the Phase 9 instruction-tuning dataset: English (Alpaca) + hand-authored Khmer.

Same discipline as Phase 1's `prepare_dataset.py`: fetch -> clean -> dedupe
-> split -> report, with every source's license recorded, not just
downloaded and forgotten (spec section 30).

No large public Khmer instruction-tuning dataset was readily available, so
the Khmer half of this dataset (`data/instructions/khmer_handauthored.jsonl`)
is hand-written for this project — 25 short factual instruction/response
pairs, clearly labeled `"source": "handauthored"` in every record and in
the manifest, not presented as if it were sourced data.

Usage:
    python scripts/prepare_instruction_dataset.py --n-english 300
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daralm.data.cleaner import clean_text  # noqa: E402
from daralm.data.dataset import split_dataset  # noqa: E402
from daralm.data.deduplication import deduplicate  # noqa: E402
from daralm.data.loader import fetch_alpaca_sample, load_jsonl, save_jsonl  # noqa: E402
from daralm.utils.logging import get_logger  # noqa: E402
from daralm.utils.seed import set_seed  # noqa: E402

logger = get_logger(__name__)

MIN_INSTRUCTION_CHARS = 4
MIN_RESPONSE_CHARS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-english", type=int, default=300, help="Alpaca examples to fetch")
    parser.add_argument(
        "--khmer-file",
        type=Path,
        default=Path("data/instructions/khmer_handauthored.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clean_examples(records: list[dict]) -> tuple[list[dict], int]:
    """Light cleaning pass: normalize whitespace/unicode, drop empty/too-short pairs.

    Instruction examples are short and structured (unlike Phase 1's long
    documents), so the full document-quality pipeline (language detection,
    URL-spam heuristics, etc.) doesn't apply here — just the parts that do:
    Khmer-safe Unicode normalization and a floor on both instruction and
    response length.
    """
    cleaned = []
    dropped = 0
    for record in records:
        instruction = clean_text(record.get("instruction", ""))
        response = clean_text(record.get("response", ""))
        if len(instruction) < MIN_INSTRUCTION_CHARS or len(response) < MIN_RESPONSE_CHARS:
            dropped += 1
            continue
        cleaned.append(
            {
                "instruction": instruction,
                "response": response,
                "language": record.get("language", "unknown"),
                "source": record.get("source", "unknown"),
            }
        )
    return cleaned, dropped


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    raw_dir = args.output_dir / "raw"
    cleaned_dir = args.output_dir / "cleaned"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    english_records, english_manifest = fetch_alpaca_sample(args.n_english)
    save_jsonl(english_records, raw_dir / "instructions_alpaca.jsonl")

    if not args.khmer_file.exists():
        raise FileNotFoundError(
            f"Khmer instruction file not found: {args.khmer_file}. This project ships a "
            "hand-authored one at data/instructions/khmer_handauthored.jsonl."
        )
    khmer_records = load_jsonl(args.khmer_file, required_field="instruction")
    khmer_manifest = {
        "source": "handauthored",
        "language": "km",
        "description": "Hand-written for this project (no suitable public Khmer "
        "instruction-tuning dataset was available) — not sourced from an external corpus.",
        "examples": len(khmer_records),
    }
    logger.info("Loaded %d hand-authored Khmer examples", len(khmer_records))

    manifest = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "sources": [english_manifest, khmer_manifest],
    }
    manifest_path = raw_dir / "MANIFEST_instructions.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info("Wrote manifest to %s", manifest_path)

    all_records = english_records + khmer_records
    total_seen = len(all_records)

    cleaned_records, dropped = clean_examples(all_records)
    logger.info("Kept %d / %d after cleaning", len(cleaned_records), total_seen)

    deduped_records, duplicate_count = deduplicate(
        [{"text": f"{r['instruction']}\n{r['response']}", **r} for r in cleaned_records]
    )
    # `deduplicate` expects a "text" field (it's generic over any record with
    # one); strip that synthetic field back out before saving.
    for record in deduped_records:
        record.pop("text", None)
    logger.info(
        "Removed %d duplicate(s); %d unique examples remain", duplicate_count, len(deduped_records)
    )

    splits = split_dataset(
        deduped_records, train_ratio=args.train_ratio, val_ratio=args.val_ratio,
        test_ratio=1.0 - args.train_ratio - args.val_ratio, seed=args.seed,
    )
    for split_name, split_records in splits.items():
        if not split_records:
            continue
        path = cleaned_dir / f"instructions_{split_name}.jsonl"
        save_jsonl(split_records, path)
        logger.info("Wrote %d examples to %s", len(split_records), path)

    print()
    print("=== Instruction Dataset Preparation Summary ===")
    print(
        f"English (Alpaca):    {len(english_records)} examples, "
        f"license: {english_manifest['license']}"
    )
    print(f"Khmer (handauthored): {len(khmer_records)} examples")
    print(f"Total fetched:       {total_seen}")
    print(f"Dropped (too short): {dropped}")
    print(f"Duplicates removed:  {duplicate_count}")
    print(f"Final unique:        {len(deduped_records)}")
    print(f"Split sizes:         { {k: len(v) for k, v in splits.items()} }")


if __name__ == "__main__":
    main()
