"""Train a SentencePiece tokenizer on the cleaned DaraLM corpus."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import sentencepiece as spm

from daralm.data.loader import load_jsonl
from daralm.utils.logging import get_logger

logger = get_logger(__name__)

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
PAD_PIECE, UNK_PIECE, BOS_PIECE, EOS_PIECE = "<pad>", "<unk>", "<bos>", "<eos>"

# Preserve nearly all characters in the multilingual corpus.
CHARACTER_COVERAGE = 0.9995


_MAX_LINE_CHARS = 4000


def build_corpus_file(cleaned_split_path: str | Path, output_path: str | Path) -> Path:
    """Explode a cleaned JSONL split into a SentencePiece training corpus."""
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
    """Train one SentencePiece model and return its .model path."""
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
        # SentencePiece's line-length cutoff is measured in bytes, not characters.
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
