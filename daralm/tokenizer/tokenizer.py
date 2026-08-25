"""DaraLMTokenizer — a thin, stable wrapper around a trained SentencePiece model.

The rest of the codebase (data packing in Phase 3, generation in Phase 10,
...) should depend on this class, not on `sentencepiece.SentencePieceProcessor`
directly. That indirection is deliberate: it keeps the specific tokenizer
backend (SentencePiece today) an implementation detail that could be swapped
later without touching every call site, and it's the one place that encodes
DaraLM's special-token conventions (see `daralm.tokenizer.train` for the
fixed IDs).
"""

from __future__ import annotations

from pathlib import Path

import sentencepiece as spm

from daralm.tokenizer.train import BOS_ID, EOS_ID, PAD_ID, UNK_ID


class DaraLMTokenizer:
    """Loads a trained SentencePiece model and exposes encode/decode.

    Example:
        tokenizer = DaraLMTokenizer.from_pretrained("checkpoints/tokenizer/unigram.model")
        ids = tokenizer.encode("Cambodia is a country in Southeast Asia.")
        text = tokenizer.decode(ids)
    """

    def __init__(self, processor: spm.SentencePieceProcessor) -> None:
        self._sp = processor

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> DaraLMTokenizer:
        """Load a tokenizer from a trained `.model` file.

        Raises `FileNotFoundError` if the path doesn't exist — fail loudly
        rather than let SentencePiece's own (less clear) native error surface.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
        processor = spm.SentencePieceProcessor(model_file=str(model_path))
        return cls(processor)

    @property
    def vocab_size(self) -> int:
        return self._sp.vocab_size()

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    def tokenize(self, text: str) -> list[str]:
        """Split `text` into subword piece strings (for inspection/evaluation)."""
        return self._sp.encode(text, out_type=str)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode `text` into token IDs, optionally wrapped with <bos>/<eos>."""
        ids = self._sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id, *ids]
        if add_eos:
            ids = [*ids, self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back into text. <pad>/<bos>/<eos> are dropped automatically."""
        ids = [i for i in ids if i not in (self.pad_id, self.bos_id, self.eos_id)]
        return self._sp.decode(ids)
