#!/usr/bin/env python
"""Fetch locally-verified Khmer pages from Common Crawl's URL index."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daralm.data.commoncrawl import fetch_indexed_khmer_sample  # noqa: E402
from daralm.data.loader import save_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=int, default=100)
    parser.add_argument("--crawl")
    parser.add_argument("--url-pattern", action="append", dest="patterns")
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-record-mb", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    records, manifest = fetch_indexed_khmer_sample(
        args.docs,
        crawl=args.crawl,
        url_patterns=args.patterns,
        min_chars=args.min_chars,
        max_record_bytes=args.max_record_mb * 1024 * 1024,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "commoncrawl_km.jsonl"
    manifest_path = args.output_dir / "MANIFEST_commoncrawl_km.json"
    save_jsonl(records, data_path)
    manifest["fetched_at"] = datetime.now(timezone.utc).isoformat()
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    print(f"Wrote {len(records):,} verified Khmer documents to {data_path}")
    print(f"Downloaded {manifest['compressed_bytes_downloaded'] / 1024 / 1024:.2f} MiB")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
