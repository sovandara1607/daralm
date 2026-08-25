"""Train a SentencePiece tokenizer on the cleaned DaraLM corpus.

We use SentencePiece rather than training raw Hugging Face `tokenizers`
BPE/WordPiece trainers for one Khmer-specific reason: SentencePiece treats
the input as a raw Unicode stream and does not assume whitespace marks word
boundaries. Khmer text does not reliably use spaces between words the way
English does (spaces in Khmer text mark clause/phrase boundaries, not word
boundaries) — a tokenizer pipeline built around a whitespace
pre-tokenizer would silently mis-segment most Khmer text before subword
merging even begins. SentencePiece sidesteps this entirely, and — usefully
for the comparison the project spec asks for — a single trainer interface
covers both algorithms we need to compare (`model_type="bpe"` and
`model_type="unigram"`), trained on the exact same corpus and vocab size.

Only the `train` split is used to fit the tokenizer, never `val`/`test` —
the tokenizer's vocabulary is as much a part of "the model" as its weights,
so holding out validation/test data from it mirrors the same discipline
used for the model itself later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import sentencepiece as spm

from daralm.data.loader import load_jsonl
from daralm.utils.logging import get_logger

logger = get_logger(__name__)

# Special token IDs, fixed so every trained tokenizer (BPE or Unigram) uses
# the same convention and Phase 3 can rely on it without inspecting the model.
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
PAD_PIECE, UNK_PIECE, BOS_PIECE, EOS_PIECE = "<pad>", "<unk>", "<bos>", "<eos>"

# SentencePiece's own recommended value for languages with large/rich
# character sets (originally suggested for CJK; Khmer's many combining
# vowel signs and subscript consonants put it in the same category as
# English's compact ~26-letter alphabet). Leaves the rarest ~0.05% of
# characters uncovered -> they map to <unk> instead of each claiming a
# scarce vocab slot for a handful of occurrences.
CHARACTER_COVERAGE = 0.9995


# SentencePiece silently *skips* any training line longer than this many
# bytes (its own default is 4192) rather than erroring — the kind of silent
# data loss this project explicitly does not want. We split each document
# into its existing paragraph/line breaks instead of flattening it to one
# line, both to avoid tripping that limit and because SentencePiece expects
# many shorter lines, not one huge line per document.
_MAX_LINE_CHARS = 4000


def build_corpus_file(cleaned_split_path: str | Path, output_path: str | Path) -> Path:
    """Explode a cleaned JSONL split into a SentencePiece training corpus.

    Each document is split on its own line breaks (paragraphs/sections, as
    already preserved by `daralm.data.cleaner.collapse_whitespace`) rather
    than flattened into a single line — a single very long line is silently
    dropped by SentencePiece's default `max_sentence_length`, which would
    otherwise mean most of a Wikipedia-article-sized corpus never reaches
    the trainer at all. Any individual line still over `_MAX_LINE_CHARS` is
    further chopped at that length as a last-resort safeguard.
    """
    records = load_jsonl(cleaned_split_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines_written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            for raw_line in record["text"].splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                for start in range(0, len(line), _MAX_LINE_CHARS):
                    chunk = line[start : start + _MAX_LINE_CHARS].strip()
                    if chunk:
                        f.write(chunk + "\n")
                        lines_written += 1

    logger.info(
        "Wrote %d line(s) from %d document(s) to corpus file %s",
        lines_written,
        len(records),
        output_path,
    )
    return output_path


def train_sentencepiece(
    corpus_path: str | Path,
    model_prefix: str | Path,
    vocab_size: int,
    model_type: Literal["bpe", "unigram"],
) -> Path:
    """Train one SentencePiece model and return its .model path.

    `byte_fallback` is deliberately OFF: with it on, SentencePiece can
    represent any input byte sequence and the true unknown-token rate would
    be ~0% by construction, which would make the "unknown token rate"
    evaluation metric the project spec asks for meaningless. Leaving it off
    means rare/out-of-corpus characters genuinely map to <unk>, giving that
    metric something real to measure — at the cost of not being
    production-hardened against arbitrary input, which is a reasonable
    trade for this evaluation phase and worth revisiting later.
    """
    model_prefix = Path(model_prefix)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Training SentencePiece model_type=%s vocab_size=%d from %s",
        model_type,
        vocab_size,
        corpus_path,
    )
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=CHARACTER_COVERAGE,
        byte_fallback=False,
        # SentencePiece's line-length cutoff is measured in bytes, not
        # characters — Khmer characters are 3 bytes each in UTF-8, so our
        # own `_MAX_LINE_CHARS`-based chunking (measured in characters)
        # doesn't guarantee every line clears the *byte* limit. Raise the
        # ceiling well above worst case (4000 chars * 3 bytes) instead of
        # leaving a small, silent residual data loss.
        max_sentence_length=16384,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_piece=PAD_PIECE,
        unk_piece=UNK_PIECE,
        bos_piece=BOS_PIECE,
        eos_piece=EOS_PIECE,
    )
    model_path = model_prefix.with_suffix(".model")
    logger.info("Wrote trained model to %s", model_path)
    return model_path
