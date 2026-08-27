from __future__ import annotations

from scripts.merge_commoncrawl_dataset import merge_splits, prepare_additions


def _record(text: str, language: str = "en") -> dict:
    return {"text": text, "language": language, "source": "test"}


def test_prepare_additions_filters_existing_duplicates_and_preserves_provenance():
    existing = {
        "train": [_record("An existing document with enough text to be valid.")],
        "val": [],
        "test": [],
    }
    raw = [
        {
            "text": "An existing document with enough text to be valid!",
            "language": "en",
            "source": "commoncrawl",
            "url": "https://duplicate.test",
            "crawl": "CC-MAIN-TEST",
        },
        {
            "text": "អត្ថបទខ្មែរថ្មីមួយដែលមានប្រវែងគ្រប់គ្រាន់សម្រាប់ការសាកល្បង។",
            "language": "km",
            "source": "commoncrawl",
            "url": "https://new.test",
            "crawl": "CC-MAIN-TEST",
        },
    ]
    additions, report = prepare_additions(raw, existing)
    assert report == {"raw": 2, "filtered": 0, "duplicates": 1, "accepted": 1}
    assert additions[0]["url"] == "https://new.test"
    assert additions[0]["crawl"] == "CC-MAIN-TEST"


def test_merge_splits_preserves_existing_records_and_splits_additions():
    existing = {
        "train": [_record("existing train")],
        "val": [_record("existing val")],
        "test": [_record("existing test")],
    }
    additions = [_record(f"new document {index}") for index in range(100)]
    merged, counts = merge_splits(existing, additions, seed=42)
    assert counts == {"train": 90, "val": 5, "test": 5}
    assert merged["train"][0]["text"] == "existing train"
    assert merged["val"][0]["text"] == "existing val"
    assert merged["test"][0]["text"] == "existing test"
