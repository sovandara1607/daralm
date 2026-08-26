"""Tests for the data pipeline: loader, preprocessing, dedup, split, packing."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from daralm.data.chat_template import format_example
from daralm.data.dataset import (
    ClassificationDataset,
    InstructionDataset,
    PackedTokenDataset,
    TextDataset,
    split_dataset,
)
from daralm.data.deduplication import deduplicate, document_hash
from daralm.data.loader import load_jsonl, save_jsonl
from daralm.data.preprocessing import compute_corpus_stats, process_document
from daralm.tokenizer.tokenizer import DaraLMTokenizer
from daralm.tokenizer.train import train_sentencepiece

# --- loader ---------------------------------------------------------------


def test_save_and_load_jsonl_roundtrip(tmp_path):
    records = [
        {"text": "hello world", "language": "en", "source": "test"},
        {"text": "កម្ពុជា", "language": "km", "source": "test"},
    ]
    path = tmp_path / "docs.jsonl"
    save_jsonl(records, path)
    loaded = load_jsonl(path)
    assert loaded == records


def test_load_jsonl_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_jsonl(tmp_path / "does_not_exist.jsonl")


def test_load_jsonl_skips_malformed_lines(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text(
        '{"text": "good", "language": "en", "source": "s"}\n'
        "not valid json\n"
        '{"language": "en", "source": "s"}\n'  # missing "text"
        '{"text": "also good", "language": "en", "source": "s"}\n',
        encoding="utf-8",
    )
    records = load_jsonl(path)
    assert len(records) == 2
    assert records[0]["text"] == "good"
    assert records[1]["text"] == "also good"


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text('{"text": "a", "language": "en", "source": "s"}\n\n\n', encoding="utf-8")
    assert len(load_jsonl(path)) == 1


# --- preprocessing ----------------------------------------------------------


def test_process_document_keeps_valid_document():
    record = {"text": "Cambodia is a country in Southeast Asia. " * 3, "source": "test"}
    cleaned, reason = process_document(record)
    assert reason is None
    assert cleaned is not None
    assert cleaned["language"] == "en"
    assert cleaned["source"] == "test"


def test_process_document_filters_empty_text():
    cleaned, reason = process_document({"text": "", "source": "test"})
    assert cleaned is None
    assert reason == "empty"


def test_process_document_filters_too_short():
    cleaned, reason = process_document({"text": "hi", "source": "test"})
    assert cleaned is None
    assert reason == "too_short"


def test_process_document_filters_corrupted_encoding():
    record = {"text": "�" * 100, "source": "test"}
    cleaned, reason = process_document(record)
    assert cleaned is None
    assert reason == "corrupted_encoding"


def test_process_document_preserves_explicit_language():
    record = {
        "text": "Cambodia is great and text is long enough. " * 3,
        "language": "en",
        "source": "test",
    }
    cleaned, _ = process_document(record)
    assert cleaned["language"] == "en"


def test_compute_corpus_stats_basic():
    records = [
        {"text": "hello world", "language": "en", "source": "s"},
        {"text": "កម្ពុជា", "language": "km", "source": "s"},
    ]
    stats = compute_corpus_stats(records, duplicate_count=1, total_seen=3)
    assert stats["documents"] == 2
    assert stats["khmer_percentage"] == 50.0
    assert stats["english_percentage"] == 50.0
    assert stats["duplicate_rate_percentage"] == pytest.approx(33.33, abs=0.01)
    assert stats["tokens"] is None  # no tokenizer until Phase 2


def test_compute_corpus_stats_empty_records():
    stats = compute_corpus_stats([])
    assert stats["documents"] == 0
    assert stats["average_document_chars"] == 0.0


# --- deduplication ----------------------------------------------------------


def test_document_hash_ignores_case_punctuation_and_whitespace():
    assert document_hash("Hello,  World!") == document_hash("hello world")


def test_deduplicate_removes_exact_duplicates():
    records = [
        {"text": "Hello world", "language": "en", "source": "s"},
        # near-identical to the record above after hash normalization
        {"text": "hello   world!!", "language": "en", "source": "s"},
        {"text": "Completely different text here", "language": "en", "source": "s"},
    ]
    kept, duplicates = deduplicate(records)
    assert len(kept) == 2
    assert duplicates == 1


def test_deduplicate_empty_list():
    kept, duplicates = deduplicate([])
    assert kept == []
    assert duplicates == 0


# --- dataset split ------------------------------------------------------


def _make_records(n: int) -> list[dict]:
    return [{"text": f"document {i}", "language": "en", "source": "s"} for i in range(n)]


def test_split_dataset_ratios():
    records = _make_records(100)
    splits = split_dataset(records, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=1)
    assert len(splits["train"]) == 80
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10


def test_split_dataset_no_overlap_and_full_coverage():
    records = _make_records(50)
    splits = split_dataset(records, seed=7)
    all_texts = [r["text"] for r in splits["train"] + splits["val"] + splits["test"]]
    assert sorted(all_texts) == sorted(r["text"] for r in records)
    assert len(set(all_texts)) == 50


def test_split_dataset_invalid_ratios_raises():
    with pytest.raises(ValueError):
        split_dataset(_make_records(10), train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


def test_split_dataset_empty_records_raises():
    with pytest.raises(ValueError):
        split_dataset([])


def test_split_dataset_is_reproducible_with_same_seed():
    records = _make_records(50)
    splits_a = split_dataset(records, seed=123)
    splits_b = split_dataset(records, seed=123)
    assert [r["text"] for r in splits_a["train"]] == [r["text"] for r in splits_b["train"]]


# --- TextDataset --------------------------------------------------------


def test_text_dataset_len_and_getitem():
    records = _make_records(5)
    ds = TextDataset(records)
    assert len(ds) == 5
    assert ds[0]["text"] == "document 0"


def test_text_dataset_from_jsonl(tmp_path):
    records = _make_records(3)
    path = tmp_path / "docs.jsonl"
    save_jsonl(records, path)
    ds = TextDataset.from_jsonl(path)
    assert len(ds) == 3


# --- PackedTokenDataset ---------------------------------------------------


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory):
    corpus_dir = tmp_path_factory.mktemp("corpus")
    corpus_path = corpus_dir / "corpus.txt"
    lines = ["hello world this is a test sentence", "another different sentence here"] * 30
    corpus_path.write_text("\n".join(lines), encoding="utf-8")
    model_path = train_sentencepiece(
        corpus_path, corpus_dir / "tok", vocab_size=30, model_type="unigram"
    )
    return DaraLMTokenizer.from_pretrained(model_path)


def _text_records(n: int) -> list[dict]:
    return [
        {"text": "hello world this is a test sentence.", "language": "en", "source": "s"}
        for _ in range(n)
    ]


def test_packed_dataset_item_shape_and_length(tiny_tokenizer):
    records = _text_records(50)
    block_size = 16
    ds = PackedTokenDataset(records, tiny_tokenizer, block_size=block_size)
    assert len(ds) == ds.total_tokens // block_size
    assert len(ds) > 0
    item = ds[0]
    assert item.shape == (block_size,)
    assert item.dtype.is_floating_point is False  # integer token IDs


def test_packed_dataset_blocks_are_contiguous_non_overlapping(tiny_tokenizer):
    records = _text_records(50)
    block_size = 8
    ds = PackedTokenDataset(records, tiny_tokenizer, block_size=block_size)
    # Concatenating consecutive blocks should reproduce a contiguous slice
    # of the underlying token stream — i.e. no gaps, no overlap.
    block0 = ds[0]
    block1 = ds[1]
    manual_slice = ds._tokens[block_size : 2 * block_size]  # noqa: SLF001
    assert torch.equal(block1, manual_slice)
    assert block0.numel() == block1.numel() == block_size


def test_packed_dataset_contains_bos_eos_tokens(tiny_tokenizer):
    # With short repeated documents, <bos>/<eos> should appear frequently
    # in the packed stream (each document is wrapped in them).
    records = _text_records(50)
    ds = PackedTokenDataset(records, tiny_tokenizer, block_size=16)
    flat = ds._tokens.tolist()  # noqa: SLF001
    assert tiny_tokenizer.bos_id in flat
    assert tiny_tokenizer.eos_id in flat


def test_packed_dataset_rejects_non_positive_block_size(tiny_tokenizer):
    with pytest.raises(ValueError):
        PackedTokenDataset(_text_records(10), tiny_tokenizer, block_size=0)


def test_packed_dataset_rejects_too_short_corpus(tiny_tokenizer):
    with pytest.raises(ValueError):
        PackedTokenDataset(_text_records(1), tiny_tokenizer, block_size=100_000)


def test_packed_dataset_index_out_of_range_raises(tiny_tokenizer):
    ds = PackedTokenDataset(_text_records(50), tiny_tokenizer, block_size=8)
    with pytest.raises(IndexError):
        _ = ds[len(ds)]


def test_packed_dataset_works_with_dataloader(tiny_tokenizer):
    ds = PackedTokenDataset(_text_records(50), tiny_tokenizer, block_size=8)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    assert batch.shape == (4, 8)


# --- InstructionDataset (Phase 9) -------------------------------------


def _instruction_examples(n: int) -> list[dict]:
    return [
        {
            "instruction": f"question number {i}",
            "response": f"answer number {i}",
            "language": "en",
            "source": "test",
        }
        for i in range(n)
    ]


def test_instruction_dataset_item_shapes(tiny_tokenizer):
    ds = InstructionDataset(_instruction_examples(5), tiny_tokenizer, block_size=64)
    assert len(ds) == 5
    input_ids, labels = ds[0]
    assert input_ids.shape == (64,)
    assert labels.shape == (64,)
    assert input_ids.dtype == torch.long
    assert labels.dtype == torch.long


def test_instruction_dataset_masks_prompt_and_supervises_response(tiny_tokenizer):
    example = {"instruction": "what is this", "response": "an answer"}
    prompt_text, _ = format_example(example["instruction"], example["response"])
    prompt_len = len(tiny_tokenizer.encode(prompt_text, add_bos=True, add_eos=False))

    ds = InstructionDataset([example], tiny_tokenizer, block_size=100)
    input_ids, labels = ds[0]

    # Every prompt-region label must be the pad/ignore_index — masked out.
    assert (labels[:prompt_len] == tiny_tokenizer.pad_id).all()
    # At least one response-region label must be a real (non-pad) token —
    # otherwise nothing would be supervised at all.
    response_region = labels[prompt_len:]
    assert (response_region != tiny_tokenizer.pad_id).any()
    # Response-region labels must equal the corresponding input_ids (the
    # model is being taught to predict exactly what's already there).
    assert torch.equal(input_ids[prompt_len:], labels[prompt_len:])


def test_instruction_dataset_padding_is_masked(tiny_tokenizer):
    ds = InstructionDataset(_instruction_examples(1), tiny_tokenizer, block_size=100)
    input_ids, labels = ds[0]
    # Whatever tail is padding in input_ids must also be masked in labels.
    pad_positions = input_ids == tiny_tokenizer.pad_id
    assert (labels[pad_positions] == tiny_tokenizer.pad_id).all()


def test_instruction_dataset_skips_examples_exceeding_block_size(tiny_tokenizer):
    long_example = {
        "instruction": "word " * 200,
        "response": "another word " * 200,
        "language": "en",
        "source": "test",
    }
    short_example = _instruction_examples(1)[0]
    ds = InstructionDataset([long_example, short_example], tiny_tokenizer, block_size=64)
    assert len(ds) == 1  # only the short one survives


def test_instruction_dataset_raises_if_all_examples_too_long(tiny_tokenizer):
    long_example = {
        "instruction": "word " * 200,
        "response": "another word " * 200,
        "language": "en",
        "source": "test",
    }
    with pytest.raises(ValueError):
        InstructionDataset([long_example], tiny_tokenizer, block_size=32)


def test_instruction_dataset_rejects_non_positive_block_size(tiny_tokenizer):
    with pytest.raises(ValueError):
        InstructionDataset(_instruction_examples(1), tiny_tokenizer, block_size=0)


def test_instruction_dataset_works_with_dataloader(tiny_tokenizer):
    ds = InstructionDataset(_instruction_examples(4), tiny_tokenizer, block_size=64)
    loader = DataLoader(ds, batch_size=2)
    batch = next(iter(loader))
    # Default collation of 2-tuples -> a 2-element list of stacked tensors.
    assert isinstance(batch, list)
    assert len(batch) == 2
    assert batch[0].shape == (2, 64)
    assert batch[1].shape == (2, 64)


# --- ClassificationDataset --------------------------------------------------


def _classification_records(n: int, labels: tuple[str, ...] = ("en", "km")) -> list[dict]:
    return [
        {"text": "hello world this is a test sentence.", "label": labels[i % len(labels)]}
        for i in range(n)
    ]


def test_classification_dataset_item_shapes(tiny_tokenizer):
    ds = ClassificationDataset(
        _classification_records(4), tiny_tokenizer, block_size=32, label_list=["en", "km"]
    )
    input_ids, label = ds[0]
    assert input_ids.shape == (32,)
    assert label.shape == ()
    assert label.dtype == torch.long


def test_classification_dataset_label2id_follows_label_list_order(tiny_tokenizer):
    # label_list order defines label2id — "km" is index 0, "en" is index 1,
    # deliberately not alphabetical, to prove the mapping follows the
    # given list rather than some implicit sort.
    records = [
        {"text": "hello world this is a test sentence.", "label": "km"},
        {"text": "another different sentence here.", "label": "en"},
    ]
    ds = ClassificationDataset(records, tiny_tokenizer, block_size=32, label_list=["km", "en"])
    assert ds.label2id == {"km": 0, "en": 1}
    _, label0 = ds[0]
    _, label1 = ds[1]
    assert label0.item() == 0  # "km"
    assert label1.item() == 1  # "en"


def test_classification_dataset_pads_short_examples(tiny_tokenizer):
    ds = ClassificationDataset(
        [{"text": "short", "label": "en"}], tiny_tokenizer, block_size=32, label_list=["en", "km"]
    )
    input_ids, _ = ds[0]
    assert (input_ids == tiny_tokenizer.pad_id).any()  # real padding present


def test_classification_dataset_truncates_long_examples_instead_of_dropping(tiny_tokenizer):
    # Unlike InstructionDataset, over-long examples are truncated, not
    # skipped — the dataset should still have every record, just shortened.
    long_text = "hello world this is a test sentence."  # 24 tokens, measured
    ds = ClassificationDataset(
        [{"text": long_text, "label": "en"}], tiny_tokenizer, block_size=8, label_list=["en", "km"]
    )
    assert len(ds) == 1  # kept, not dropped
    input_ids, _ = ds[0]
    assert input_ids.shape == (8,)


def test_classification_dataset_rejects_unrecognized_label(tiny_tokenizer):
    records = [{"text": "hello world this is a test sentence.", "label": "fr"}]
    with pytest.raises(ValueError, match="Unrecognized label"):
        ClassificationDataset(records, tiny_tokenizer, block_size=32, label_list=["en", "km"])


def test_classification_dataset_rejects_fewer_than_two_labels(tiny_tokenizer):
    with pytest.raises(ValueError):
        ClassificationDataset(
            _classification_records(2, labels=("en",)),
            tiny_tokenizer,
            block_size=32,
            label_list=["en"],
        )


def test_classification_dataset_rejects_non_positive_block_size(tiny_tokenizer):
    with pytest.raises(ValueError):
        ClassificationDataset(
            _classification_records(1), tiny_tokenizer, block_size=0, label_list=["en", "km"]
        )


def test_classification_dataset_rejects_empty_records(tiny_tokenizer):
    with pytest.raises(ValueError):
        ClassificationDataset([], tiny_tokenizer, block_size=32, label_list=["en", "km"])


def test_classification_dataset_works_with_dataloader(tiny_tokenizer):
    ds = ClassificationDataset(
        _classification_records(4), tiny_tokenizer, block_size=32, label_list=["en", "km"]
    )
    loader = DataLoader(ds, batch_size=2)
    batch = next(iter(loader))
    assert isinstance(batch, list)
    assert len(batch) == 2
    assert batch[0].shape == (2, 32)  # input_ids
    assert batch[1].shape == (2,)  # labels
