"""Bounded Common Crawl WET ingestion for DaraLM."""

from __future__ import annotations

import gzip
import io
import json
import random
import zlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

import httpx

from daralm.data.cleaner import detect_language
from daralm.data.deduplication import deduplicate
from daralm.utils.logging import get_logger

logger = get_logger(__name__)

DATA_BASE = "https://data.commoncrawl.org"
COLLECTIONS_URL = "https://index.commoncrawl.org/collinfo.json"
USER_AGENT = "DaraLM-CommonCrawl/0.1 (educational Khmer language-model research)"
LANGUAGE_CODES = {"km": "khm", "en": "eng"}
DEFAULT_KHMER_PATTERNS = [
    "nlc.gov.kh/*",
    "info.gov.kh/*",
    "nea.gov.kh/*",
    "kampucheathmey.com/*",
    "phnompenhpost.com/khmer/*",
    "cambodiadaily.com/khmer/*",
]


class DownloadLimitReached(RuntimeError):
    """Raised internally when the configured compressed-byte budget is spent."""


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hidden_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg", "template"}:
            self.hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg", "template"} and self.hidden_depth:
            self.hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden_depth and data.strip():
            self.parts.append(data.strip())


def html_to_text(html: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(html)
    return "\n".join(parser.parts)


def remove_repeated_boilerplate(
    records: list[dict[str, Any]], *, threshold: float = 0.6, min_group_size: int = 3
) -> tuple[list[dict[str, Any]], int]:
    """Remove lines repeated across most pages from the same domain."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[urlparse(record.get("url") or "").netloc].append(record)

    cleaned: list[dict[str, Any]] = []
    removed_lines = 0
    for group in groups.values():
        if len(group) < min_group_size:
            cleaned.extend(group)
            continue
        document_frequency: Counter[str] = Counter()
        for record in group:
            document_frequency.update(set(record["text"].splitlines()))
        repeated = {
            line
            for line, count in document_frequency.items()
            if line.strip() and count / len(group) >= threshold
        }
        for record in group:
            lines = record["text"].splitlines()
            unique_lines = [line for line in lines if line not in repeated]
            removed_lines += len(lines) - len(unique_lines)
            cleaned.append({**record, "text": "\n".join(unique_lines).strip()})
    return cleaned, removed_lines


def extract_warc_html(record: bytes) -> str:
    """Extract visible text from one gzip-compressed WARC response record."""
    uncompressed = gzip.decompress(record)
    _, _, warc_payload = uncompressed.partition(b"\r\n\r\n")
    if not warc_payload:
        _, _, warc_payload = uncompressed.partition(b"\n\n")
    http_headers, separator, body = warc_payload.partition(b"\r\n\r\n")
    if not separator:
        http_headers, _, body = warc_payload.partition(b"\n\n")
    headers_text = http_headers.decode("latin-1", errors="replace").lower()
    if "content-encoding: gzip" in headers_text:
        body = gzip.decompress(body)
    return html_to_text(body.decode("utf-8", errors="replace"))


def iter_wet_records(chunks: Iterable[bytes]) -> Iterator[tuple[dict[str, str], bytes]]:
    """Parse decompressed WARC/WET byte chunks without loading a file at once."""
    buffer = bytearray()
    for chunk in chunks:
        buffer.extend(chunk)
        while True:
            start = buffer.find(b"WARC/")
            if start < 0:
                # Retain enough bytes to match a split ``WARC/`` marker
                if len(buffer) > 5:
                    del buffer[:-5]
                break
            if start:
                del buffer[:start]

            separator = b"\r\n\r\n"
            header_end = buffer.find(separator)
            if header_end < 0:
                separator = b"\n\n"
                header_end = buffer.find(separator)
            if header_end < 0:
                break

            header_blob = bytes(buffer[:header_end]).decode("utf-8", errors="replace")
            headers: dict[str, str] = {}
            for line in header_blob.splitlines()[1:]:
                if ":" in line:
                    key, value = line.split(":", 1)
                    headers[key.strip().lower()] = value.strip()
            try:
                content_length = int(headers["content-length"])
            except (KeyError, ValueError):
                del buffer[: header_end + len(separator)]
                continue

            payload_start = header_end + len(separator)
            payload_end = payload_start + content_length
            if len(buffer) < payload_end:
                break
            payload = bytes(buffer[payload_start:payload_end])
            del buffer[:payload_end]
            yield headers, payload


def _decompressed_chunks(compressed_chunks: Iterable[bytes]) -> Iterator[bytes]:
    """Decompress a WET stream, including its concatenated gzip members."""
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    for chunk in compressed_chunks:
        remaining = chunk
        while remaining:
            data = decompressor.decompress(remaining)
            if data:
                yield data
            if not decompressor.eof:
                break
            remaining = decompressor.unused_data
            decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    tail = decompressor.flush()
    if tail:
        yield tail


def _record_language(headers: dict[str, str], text: str) -> str | None:
    detected = detect_language(text)
    if detected in LANGUAGE_CODES:
        return detected

    cc_languages = {
        code.strip() for code in headers.get("warc-identified-content-language", "").split(",")
    }
    if cc_languages.intersection(LANGUAGE_CODES.values()):
        logger.debug(
            "Rejecting record whose Common Crawl language metadata is not script-verifiable: %s",
            sorted(cc_languages),
        )
    return None


def _latest_crawl(client: httpx.Client) -> str:
    response = client.get(COLLECTIONS_URL)
    response.raise_for_status()
    collections = response.json()
    if not collections:
        raise RuntimeError("Common Crawl returned an empty collection list")
    return str(collections[0]["id"])


def _wet_paths(client: httpx.Client, crawl: str) -> list[str]:
    response = client.get(f"{DATA_BASE}/crawl-data/{crawl}/wet.paths.gz")
    response.raise_for_status()
    with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as archive:
        return [line.decode().strip() for line in archive if line.strip()]


def fetch_commoncrawl_sample(
    languages: list[str],
    docs_per_language: int,
    *,
    crawl: str | None = None,
    min_chars: int = 200,
    max_wet_files: int = 3,
    max_download_bytes: int = 300 * 1024 * 1024,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch a quota-limited multilingual sample from Common Crawl WET files."""
    unsupported = sorted(set(languages) - LANGUAGE_CODES.keys())
    if unsupported:
        raise ValueError(f"Unsupported language(s): {', '.join(unsupported)}")
    if docs_per_language <= 0 or max_wet_files <= 0 or max_download_bytes <= 0:
        raise ValueError("Document, file, and byte limits must all be positive")

    headers = {"User-Agent": USER_AGENT}
    counts = dict.fromkeys(languages, 0)
    records: list[dict[str, Any]] = []
    downloaded_bytes = 0
    files_scanned: list[str] = []

    with httpx.Client(headers=headers, timeout=60.0, follow_redirects=True) as client:
        selected_crawl = crawl or _latest_crawl(client)
        paths = _wet_paths(client, selected_crawl)
        random.Random(seed).shuffle(paths)

        for wet_path in paths[:max_wet_files]:
            if all(counts[language] >= docs_per_language for language in languages):
                break
            files_scanned.append(wet_path)
            logger.info("Scanning Common Crawl WET file %s", wet_path)
            with client.stream("GET", f"{DATA_BASE}/{wet_path}") as response:
                response.raise_for_status()

                def limited_chunks() -> Iterator[bytes]:
                    nonlocal downloaded_bytes
                    for chunk in response.iter_bytes():
                        if downloaded_bytes + len(chunk) > max_download_bytes:
                            raise DownloadLimitReached
                        downloaded_bytes += len(chunk)
                        yield chunk

                try:
                    parsed = iter_wet_records(_decompressed_chunks(limited_chunks()))
                    for record_headers, payload in parsed:
                        if record_headers.get("warc-type") != "conversion":
                            continue
                        text = payload.decode("utf-8", errors="replace").strip()
                        if len(text) < min_chars:
                            continue
                        language = _record_language(record_headers, text)
                        if language not in counts or counts[language] >= docs_per_language:
                            continue
                        records.append(
                            {
                                "text": text,
                                "language": language,
                                "source": "commoncrawl",
                                "url": record_headers.get("warc-target-uri"),
                                "crawl": selected_crawl,
                            }
                        )
                        counts[language] += 1
                except DownloadLimitReached:
                    logger.warning("Stopped at the configured download limit")
                    break

    manifest = {
        "source": "commoncrawl",
        "crawl": selected_crawl,
        "source_url": "https://commoncrawl.org/",
        "data_base_url": DATA_BASE,
        "format": "WET (WARC Encapsulated Text)",
        "license": (
            "Common Crawl provides open access to the archive, but does not grant a blanket "
            "license to page content. Copyright and usage rights remain with each source site; "
            "preserve page URLs and review rights before redistribution or commercial use."
        ),
        "languages_requested": languages,
        "documents_requested_per_language": docs_per_language,
        "documents_fetched": counts,
        "min_chars_filter": min_chars,
        "max_wet_files": max_wet_files,
        "wet_files_scanned": files_scanned,
        "compressed_bytes_downloaded": downloaded_bytes,
        "seed": seed,
    }
    return records, manifest


def fetch_indexed_khmer_sample(
    n_docs: int,
    *,
    crawl: str | None = None,
    url_patterns: list[str] | None = None,
    min_chars: int = 200,
    max_record_bytes: int = 5 * 1024 * 1024,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch Khmer pages via targeted CDX queries and WARC byte ranges."""
    if n_docs <= 0 or max_record_bytes <= 0:
        raise ValueError("Document and byte limits must be positive")
    patterns = url_patterns or DEFAULT_KHMER_PATTERNS
    records: list[dict[str, Any]] = []
    seen_digests: set[str] = set()
    queried: list[str] = []
    rejected = 0
    downloaded_bytes = 0

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60.0) as client:
        selected_crawl = crawl or _latest_crawl(client)
        index_url = f"https://index.commoncrawl.org/{selected_crawl}-index"
        for pattern in patterns:
            if len(records) >= n_docs:
                break
            queried.append(pattern)
            params = [
                ("url", pattern),
                ("output", "json"),
                ("filter", "=status:200"),
                ("filter", "languages:khm"),
                ("pageSize", "1"),
            ]
            logger.info("Querying Common Crawl index for %s", pattern)
            with client.stream("GET", index_url, params=params) as response:
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                for line in response.iter_lines():
                    if len(records) >= n_docs:
                        break
                    if not line.strip():
                        continue
                    capture = json.loads(line)
                    digest = str(capture.get("digest", ""))
                    length = int(capture["length"])
                    if digest in seen_digests or length > max_record_bytes:
                        continue
                    seen_digests.add(digest)
                    offset = int(capture["offset"])
                    archive_url = f"{DATA_BASE}/{capture['filename']}"
                    record_response = client.get(
                        archive_url,
                        headers={"Range": f"bytes={offset}-{offset + length - 1}"},
                    )
                    record_response.raise_for_status()
                    downloaded_bytes += len(record_response.content)
                    try:
                        text = extract_warc_html(record_response.content).strip()
                    except (EOFError, OSError, zlib.error):
                        rejected += 1
                        continue
                    if len(text) < min_chars or detect_language(text) not in {"km", "mixed"}:
                        rejected += 1
                        continue
                    records.append(
                        {
                            "text": text,
                            "language": "km",
                            "source": "commoncrawl",
                            "url": capture.get("url"),
                            "crawl": selected_crawl,
                        }
                    )

    records, boilerplate_lines_removed = remove_repeated_boilerplate(records)
    records = [
        record
        for record in records
        if len(record["text"]) >= min_chars and detect_language(record["text"]) in {"km", "mixed"}
    ]
    records, duplicates_removed = deduplicate(records)
    manifest = {
        "source": "commoncrawl",
        "crawl": selected_crawl,
        "source_url": "https://commoncrawl.org/",
        "strategy": "CDX language-filtered WARC byte-range retrieval",
        "license": (
            "Common Crawl provides open access to the archive, but page copyrights and usage "
            "rights remain with each source site. Preserve URLs and review rights before "
            "redistribution or commercial use."
        ),
        "language": "km",
        "documents_requested": n_docs,
        "documents_fetched": len(records),
        "records_rejected_after_local_verification": rejected,
        "documents_after_boilerplate_removal": len(records),
        "boilerplate_lines_removed": boilerplate_lines_removed,
        "duplicates_removed_after_boilerplate_cleaning": duplicates_removed,
        "url_patterns_queried": queried,
        "compressed_bytes_downloaded": downloaded_bytes,
        "max_record_bytes": max_record_bytes,
        "min_chars_filter": min_chars,
    }
    return records, manifest
