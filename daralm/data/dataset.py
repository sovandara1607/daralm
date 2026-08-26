"""Train/validation/test splitting, plus raw-text and tokenized dataset wrappers.

This covers the last two stages of the pipeline in spec section 6:

    ... -> Dedup -> Split -> Tokenizer -> Token IDs -> Sequence Packing -> Training Dataset

`split_dataset` runs the split. `TextDataset` is a thin wrapper over cleaned
records for iteration/inspection. `PackedTokenDataset` (Phase 4) is the
tokenized, sequence-packed `torch.utils.data.Dataset` an actual training
`DataLoader` consumes.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset

from daralm.data.chat_template import format_example
from daralm.data.loader import load_jsonl
from daralm.utils.logging import get_logger

if TYPE_CHECKING:
    from daralm.tokenizer.tokenizer import DaraLMTokenizer

logger = get_logger(__name__)

_RATIO_TOLERANCE = 1e-6


def split_dataset(
    records: list[dict[str, Any]],
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Shuffle and split `records` into train/val/test sets.

    Splitting happens *after* deduplication so no document (or a near-copy
    of it) can leak across splits, and *before* tokenization, so the split
    boundary is defined on documents, not on token sequences that might
    otherwise straddle a boundary.

    Args:
        records: Cleaned, deduplicated records to split.
        train_ratio, val_ratio, test_ratio: Must sum to 1.0.
        seed: Shuffle seed, recorded with the experiment for reproducibility.

    Raises:
        ValueError: if the ratios don't sum to 1.0, or `records` is empty.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > _RATIO_TOLERANCE:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must sum to 1.0, got {total_ratio}"
        )
    if not records:
        raise ValueError("Cannot split an empty dataset")

    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


class TextDataset:
    """A minimal in-memory wrapper over cleaned {"text","language","source"} records.

    Not a `torch.utils.data.Dataset` (no tokenization has happened yet) —
    just enough to load a cleaned split from disk and iterate/index into it
    for inspection, stats, or feeding into the tokenizer trainer (Phase 2).
    """

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._records[index]

    @classmethod
    def from_jsonl(cls, path: str | Path) -> TextDataset:
        return cls(load_jsonl(path))


class PackedTokenDataset(Dataset):
    """Tokenized, sequence-packed dataset for causal LM training.

    Every document is tokenized and wrapped in `<bos> ... <eos>`, and all
    documents are concatenated into one long token stream, which is then
    cut into non-overlapping `block_size`-length chunks. This is "sequence
    packing": rather than padding every document up to the batch's longest
    one (wasteful — most of our Wikipedia articles are far longer than
    `block_size` anyway, and short documents would be mostly padding), we
    pack tokens from many documents back to back so every training example
    is 100% real content, no padding at all.

    One consequence worth knowing: a single packed example can span the
    boundary between two unrelated documents (document A's `<eos>`
    immediately followed by document B's `<bos>`). The model has no special
    signal for this beyond the token IDs themselves — this is standard
    practice for LLM pretraining (GPT-2/3, LLaMA all pack this way), not an
    oversight.

    Each item is a single `block_size`-length tensor, not a separate
    (input, label) pair — `daralm.model.transformer.DaraLMTransformer.forward`
    shifts internally when called as `model(batch, labels=batch)` (the same
    convention Hugging Face's `labels=input_ids` uses). The trade-off: the
    very last position in each block has no next-token target and
    contributes no loss (its target would be the first token of the
    *next* block, which we deliberately don't fetch, to keep blocks
    non-overlapping and the API simple) — a small, known inefficiency
    (1/block_size of positions), not a bug.

    At this project's current corpus scale (~2,700 training documents),
    packing happens in memory at construction time. Pre-tokenizing once and
    caching the token stream to `data/tokenized/` is a natural upgrade once
    corpus size makes re-tokenizing on every run too slow — not needed yet,
    so not built yet.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: DaraLMTokenizer,
        block_size: int,
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        token_stream: list[int] = []
        for record in records:
            token_stream.extend(tokenizer.encode(record["text"], add_bos=True, add_eos=True))

        if len(token_stream) < block_size:
            raise ValueError(
                f"Token stream has only {len(token_stream)} tokens, fewer than "
                f"block_size ({block_size}) — cannot form even one training example. "
                "Provide more records or a smaller block_size."
            )

        self._tokens = torch.tensor(token_stream, dtype=torch.long)
        self._block_size = block_size
        self._num_blocks = len(self._tokens) // block_size

    def __len__(self) -> int:
        return self._num_blocks

    def __getitem__(self, index: int) -> torch.Tensor:
        if not 0 <= index < self._num_blocks:
            raise IndexError(f"index {index} out of range for dataset of length {self._num_blocks}")
        start = index * self._block_size
        end = start + self._block_size
        return self._tokens[start:end]

    @property
    def total_tokens(self) -> int:
        """Total tokens in the packed stream (including any leftover, unused tail)."""
        return len(self._tokens)


class InstructionDataset(Dataset):
    """Tokenized instruction/response pairs for supervised fine-tuning (Phase 9).

    Unlike `PackedTokenDataset`, examples are *not* packed together — each
    instruction/response pair is padded on its own up to `block_size`
    rather than concatenated with its neighbors. Packing base-pretraining
    documents back to back is fine because next-token prediction doesn't
    care about document boundaries; packing instruction examples together
    would actively teach the model that one conversation's `<eos>` is
    immediately followed by an unrelated next conversation's `<user>` turn
    — exactly the pattern instruction-tuning should discourage, not
    reinforce.

    Loss masking is the other real difference from base pretraining: each
    item is an `(input_ids, labels)` pair, not one tensor. `labels` holds
    `pad_token_id` (cross-entropy's `ignore_index`) at every *prompt*
    position — the model isn't scored on predicting the instruction it was
    given, only on generating a good response to it — and the real token
    ID at every *response* position. This needs zero changes to
    `DaraLMTransformer.forward`, which already accepts `input_ids` and
    `labels` as independent tensors; base pretraining just always happens
    to call it with `labels is input_ids`.
    """

    def __init__(
        self,
        examples: list[dict[str, Any]],
        tokenizer: DaraLMTokenizer,
        block_size: int,
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        pad_id = tokenizer.pad_id
        self._input_ids: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []
        skipped_too_long = 0

        for example in examples:
            prompt_text, response_text = format_example(example["instruction"], example["response"])
            prompt_ids = tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
            response_ids = tokenizer.encode(response_text, add_bos=False, add_eos=True)
            total_len = len(prompt_ids) + len(response_ids)

            if total_len > block_size:
                # Dropped, not truncated: truncating could cut the prompt
                # mid-sentence or the response before its <eos>, either of
                # which would corrupt what the model is being taught. With
                # a small dataset we can afford to just skip the outlier.
                skipped_too_long += 1
                continue

            pad_len = block_size - total_len
            input_ids = prompt_ids + response_ids + [pad_id] * pad_len
            labels = [pad_id] * len(prompt_ids) + response_ids + [pad_id] * pad_len

            self._input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            self._labels.append(torch.tensor(labels, dtype=torch.long))

        if skipped_too_long:
            logger.warning(
                "Skipped %d example(s) exceeding block_size (%d)", skipped_too_long, block_size
            )
        if not self._input_ids:
            raise ValueError(
                "No examples fit within block_size — every example was skipped. "
                "Provide shorter examples or a larger block_size."
            )

    def __len__(self) -> int:
        return len(self._input_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._input_ids[index], self._labels[index]


class ClassificationDataset(Dataset):
    """Tokenized `{"text", "label"}` pairs for training a
    `daralm.model.classification_head.ClassificationHead` on top of a
    `DaraLMTransformer` backbone — the roadmap's first classification-family
    capability (`ROADMAP_NLP_PLATFORM.md`).

    Two real differences from `InstructionDataset`, both deliberate:

    1. **Truncated, not skipped, when longer than `block_size`.** SFT
       drops over-long examples because truncating could cut a response
       before its `<eos>` — actively corrupting what the model is taught
       to generate. Classification has no such risk: using only the first
       `block_size` tokens of a long document still gives the classifier
       real signal to pool from (the last-real-token pooling strategy in
       `ClassificationHead` just pools from wherever the truncated
       sequence ends instead of the true document end) — the standard
       practice for text classifiers, not a shortcut specific to this
       project.
    2. **Explicit `label2id`, built from `label_list`, not discovered from
       the data.** An unrecognized label raises immediately (`ValueError`)
       rather than silently becoming a new, unplanned class — the same
       "fail loudly on a typo" discipline `ModelConfig`'s `extra="forbid"`
       already applies to configs, applied here to label spelling.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: DaraLMTokenizer,
        block_size: int,
        label_list: list[str],
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if len(label_list) < 2:
            raise ValueError(f"label_list must have at least 2 classes, got {label_list}")

        self.label_list = list(label_list)
        self.label2id = {label: idx for idx, label in enumerate(self.label_list)}

        pad_id = tokenizer.pad_id
        self._input_ids: list[torch.Tensor] = []
        self._label_ids: list[int] = []
        truncated_count = 0

        for record in records:
            label = record["label"]
            if label not in self.label2id:
                raise ValueError(
                    f"Unrecognized label {label!r} — not in label_list {self.label_list}. "
                    "Every record's label must be one of the configured classes."
                )

            ids = tokenizer.encode(record["text"], add_bos=True, add_eos=True)
            if len(ids) > block_size:
                ids = ids[:block_size]
                truncated_count += 1
            else:
                ids = ids + [pad_id] * (block_size - len(ids))

            self._input_ids.append(torch.tensor(ids, dtype=torch.long))
            self._label_ids.append(self.label2id[label])

        if truncated_count:
            logger.warning(
                "Truncated %d example(s) exceeding block_size (%d)", truncated_count, block_size
            )
        if not self._input_ids:
            raise ValueError("No records provided — cannot build an empty ClassificationDataset.")

    def __len__(self) -> int:
        return len(self._input_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._input_ids[index], torch.tensor(self._label_ids[index], dtype=torch.long)
