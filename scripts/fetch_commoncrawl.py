#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daralm.data.commoncrawl import fetch_commoncrawl_sample  # noqa: E402
from daralm.data.loader import save_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", choices=["km", "en"], default=["km", "en"])
    parser.add_argument("--docs-per-language", type=int, default=100)
    parser.add_argument("--crawl", help="Crawl ID, e.g. CC-MAIN-2026-25; defaults to latest")
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-wet-files", type=int, default=3)
    parser.add_argument("--max-download-mb", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, manifest = fetch_commoncrawl_sample(
        args.languages,
        args.docs_per_language,
        crawl=args.crawl,
        min_chars=args.min_chars,
        max_wet_files=args.max_wet_files,
        max_download_bytes=args.max_download_mb * 1024 * 1024,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "commoncrawl.jsonl"
    manifest_path = args.output_dir / "MANIFEST_commoncrawl.json"
    save_jsonl(records, data_path)
    manifest["fetched_at"] = datetime.now(timezone.utc).isoformat()
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print(f"Wrote {len(records):,} documents to {data_path}")
    print(f"Language counts: {manifest['documents_fetched']}")
    print(f"Downloaded: {manifest['compressed_bytes_downloaded'] / 1024 / 1024:.1f} MiB")
    print(f"Manifest: {manifest_path}")
    if any(count < args.docs_per_language for count in manifest["documents_fetched"].values()):
        print("Some quotas were not met; increase --max-wet-files and --max-download-mb.")


if __name__ == "__main__":
    main()
