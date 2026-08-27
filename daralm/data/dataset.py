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
    """Shuffle and split `records` into train/val/test sets."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > _RATIO_TOLERANCE:
        raise ValueError(f"train_ratio + val_ratio + test_ratio must sum to 1.0, got {total_ratio}")
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
    """Tokenized, sequence-packed dataset for causal LM training."""

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
                # Do not truncate a supervised response mid-answer.
                skipped_too_long += 1
                continue

            pad_len = block_size - total_len
            input_ids = prompt_ids + response_ids + [pad_id] * pad_len
            # Ignore the prompt and padding when computing response loss.
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
