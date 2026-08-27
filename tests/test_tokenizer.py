"""Tests for daralm.tokenizer — training, wrapper, and evaluation."""

from __future__ import annotations

import pytest

from daralm.tokenizer.evaluation import compute_metrics, load_domain_examples
from daralm.tokenizer.tokenizer import DaraLMTokenizer
from daralm.tokenizer.train import BOS_ID, EOS_ID, PAD_ID, UNK_ID, train_sentencepiece

_KHMER_LINES = [
    "កម្ពុជាជាប្រទេសមួយនៅអាស៊ីអាគ្នេយ៍។",
    "ភ្នំពេញជារាជធានីនៃប្រទេសកម្ពុជា។",
    "ខ្ញុំចង់រៀនភាសាខ្មែរ។",
] * 20
_ENGLISH_LINES = [
    "Cambodia is a country in Southeast Asia.",
    "Phnom Penh is the capital of Cambodia.",
    "I want to learn the Khmer language.",
] * 20


@pytest.fixture(scope="module")
def tiny_corpus_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("corpus") / "corpus.txt"
    with path.open("w", encoding="utf-8") as f:
        for line in _KHMER_LINES + _ENGLISH_LINES:
            f.write(line + "\n")
    return path


_TINY_VOCAB_SIZE = 80


@pytest.fixture(scope="module")
def bpe_tokenizer(tiny_corpus_path, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("bpe_model")
    model_path = train_sentencepiece(
        tiny_corpus_path, out_dir / "bpe", vocab_size=_TINY_VOCAB_SIZE, model_type="bpe"
    )
    return DaraLMTokenizer.from_pretrained(model_path)


@pytest.fixture(scope="module")
def unigram_tokenizer(tiny_corpus_path, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("unigram_model")
    model_path = train_sentencepiece(
        tiny_corpus_path, out_dir / "unigram", vocab_size=_TINY_VOCAB_SIZE, model_type="unigram"
    )
    return DaraLMTokenizer.from_pretrained(model_path)


def test_train_sentencepiece_produces_model_file(bpe_tokenizer):
    assert bpe_tokenizer.vocab_size == _TINY_VOCAB_SIZE


def test_special_token_ids_are_fixed_convention(bpe_tokenizer, unigram_tokenizer):
    for tok in (bpe_tokenizer, unigram_tokenizer):
        assert tok.pad_id == PAD_ID == 0
        assert tok.unk_id == UNK_ID == 1
        assert tok.bos_id == BOS_ID == 2
        assert tok.eos_id == EOS_ID == 3


def test_from_pretrained_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DaraLMTokenizer.from_pretrained(tmp_path / "does_not_exist.model")


def test_encode_returns_nonempty_ids_for_khmer(bpe_tokenizer):
    ids = bpe_tokenizer.encode(_KHMER_LINES[0])
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)


def test_encode_returns_nonempty_ids_for_english(bpe_tokenizer):
    ids = bpe_tokenizer.encode(_ENGLISH_LINES[0])
    assert len(ids) > 0


def test_encode_decode_roundtrip_on_in_vocabulary_text(unigram_tokenizer):
    text = _KHMER_LINES[0]
    decoded = unigram_tokenizer.decode(unigram_tokenizer.encode(text))
    assert decoded == text


def test_encode_empty_string(bpe_tokenizer):
    assert bpe_tokenizer.encode("") == []


def test_decode_empty_list(bpe_tokenizer):
    assert bpe_tokenizer.decode([]) == ""


def test_encode_with_bos_eos_wraps_ids(bpe_tokenizer):
    ids = bpe_tokenizer.encode(_ENGLISH_LINES[0], add_bos=True, add_eos=True)
    assert ids[0] == bpe_tokenizer.bos_id
    assert ids[-1] == bpe_tokenizer.eos_id


def test_decode_strips_special_tokens(bpe_tokenizer):
    ids = bpe_tokenizer.encode(_ENGLISH_LINES[0], add_bos=True, add_eos=True)
    decoded = bpe_tokenizer.decode(ids)
    assert "<bos>" not in decoded
    assert "<eos>" not in decoded


def test_tokenize_returns_piece_strings(bpe_tokenizer):
    pieces = bpe_tokenizer.tokenize(_ENGLISH_LINES[0])
    assert isinstance(pieces, list)
    assert all(isinstance(p, str) for p in pieces)
    assert len(pieces) > 0


def test_unseen_script_produces_unk_or_fallback_pieces(bpe_tokenizer):
    ids = bpe_tokenizer.encode("これは日本語です")
    assert len(ids) > 0  # SentencePiece always produces *something*, even if it's all <unk>


def test_compute_metrics_basic(bpe_tokenizer):
    metrics = compute_metrics(bpe_tokenizer, _ENGLISH_LINES[:3])
    assert metrics["num_examples"] == 3
    assert metrics["avg_tokens_per_sentence"] > 0
    assert metrics["avg_chars_per_token"] > 0
    assert 0.0 <= metrics["unk_rate_percentage"] <= 100.0


def test_compute_metrics_empty_list(bpe_tokenizer):
    metrics = compute_metrics(bpe_tokenizer, [])
    assert metrics["num_examples"] == 0
    assert metrics["avg_tokens_per_sentence"] == 0.0


def test_load_domain_examples_has_four_domains():
    khmer_records = [{"text": t, "language": "km", "source": "s"} for t in _KHMER_LINES]
    english_records = [{"text": t, "language": "en", "source": "s"} for t in _ENGLISH_LINES]
    domains = load_domain_examples(khmer_records, english_records)
    assert set(domains.keys()) == {"khmer", "english", "mixed", "code_numbers_urls"}
    assert len(domains["khmer"]) > 0
    assert len(domains["english"]) > 0
    assert len(domains["mixed"]) > 0
    assert len(domains["code_numbers_urls"]) > 0
