"""Tokenizer evaluation: fragmentation, compression, and unknown-token rate."""

from __future__ import annotations

from typing import Any

from daralm.tokenizer.tokenizer import DaraLMTokenizer

MIXED_EXAMPLES = [
    "Cambodia (កម្ពុជា) is a country in Southeast Asia.",
    "ខ្ញុំចង់រៀន machine learning និង deep learning។",
    "The capital of Cambodia is ភ្នំពេញ (Phnom Penh).",
    "សូមអរគុណ for your help, thank you ណាស់!",
]

CODE_NUMBERS_URLS_EXAMPLES = [
    "def train_model(lr=0.0003, batch_size=32): return model.fit(X, y)",
    "Visit https://huggingface.co/datasets/wikimedia/wikipedia for the source data.",
    "The model has 42,105,344 parameters and trains at a learning rate of 3e-4.",
    "import torch; x = torch.tensor([1, 2, 3]); print(x.sum() / len(x))",
]


def load_domain_examples(
    khmer_records: list[dict[str, Any]],
    english_records: list[dict[str, Any]],
    n_samples: int = 20,
) -> dict[str, list[str]]:
    """Assemble the four evaluation domains."""

    def sample_texts(records: list[dict[str, Any]], max_chars: int = 500) -> list[str]:
        return [r["text"][:max_chars] for r in records[:n_samples] if r["text"].strip()]

    return {
        "khmer": sample_texts(khmer_records),
        "english": sample_texts(english_records),
        "mixed": MIXED_EXAMPLES,
        "code_numbers_urls": CODE_NUMBERS_URLS_EXAMPLES,
    }


def compute_metrics(tokenizer: DaraLMTokenizer, texts: list[str]) -> dict[str, Any]:
    """Compute tokenizer quality metrics over a list of example texts."""
    if not texts:
        return {
            "num_examples": 0,
            "avg_tokens_per_sentence": 0.0,
            "avg_chars_per_token": 0.0,
            "unk_rate_percentage": 0.0,
            "avg_tokens_per_whitespace_unit": 0.0,
            "avg_tokens_per_char": 0.0,
        }

    total_tokens = 0
    total_chars = 0
    total_unk = 0
    total_whitespace_units = 0

    for text in texts:
        ids = tokenizer.encode(text)
        total_tokens += len(ids)
        total_chars += len(text)
        total_unk += sum(1 for i in ids if i == tokenizer.unk_id)
        total_whitespace_units += max(len(text.split()), 1)

    return {
        "num_examples": len(texts),
        "avg_tokens_per_sentence": round(total_tokens / len(texts), 2),
        "avg_chars_per_token": round(total_chars / total_tokens, 3) if total_tokens else 0.0,
        "unk_rate_percentage": round(100 * total_unk / total_tokens, 3) if total_tokens else 0.0,
        "avg_tokens_per_whitespace_unit": round(total_tokens / total_whitespace_units, 3),
        "avg_tokens_per_char": round(total_tokens / total_chars, 3) if total_chars else 0.0,
    }


def evaluate_tokenizer(
    tokenizer: DaraLMTokenizer, domain_examples: dict[str, list[str]]
) -> dict[str, dict[str, Any]]:
    """Run `compute_metrics` across every evaluation domain."""
    return {domain: compute_metrics(tokenizer, texts) for domain, texts in domain_examples.items()}


def show_example_tokenization(tokenizer: DaraLMTokenizer, text: str) -> dict[str, Any]:
    """Return a human-readable breakdown of how `text` gets tokenized."""
    pieces = tokenizer.tokenize(text)
    ids = tokenizer.encode(text)
    return {"text": text, "num_pieces": len(pieces), "pieces": pieces, "ids": ids}
