"""Regression test for the Phase 5 sanity gate: can the model overfit at all?

Runs the same procedure as `scripts/overfit_test.py` (train on a tiny fixed
set, verify loss collapses and greedy generation reproduces it) at a much
smaller synthetic scale, so it runs in well under a second as part of the
normal test suite. This is a permanent trip-wire: if a future change to the
model/trainer breaks the ability to overfit, this test catches it —
matching spec section 19's framing that this capability is a precondition
for everything after it, not a one-off manual check.
"""

from __future__ import annotations

import torch

from daralm.data.dataset import PackedTokenDataset
from daralm.evaluation.generation import average_match_rate, check_memorization
from daralm.evaluation.perplexity import compute_perplexity
from daralm.model.config import ModelConfig
from daralm.model.transformer import DaraLMTransformer
from daralm.tokenizer.tokenizer import DaraLMTokenizer
from daralm.tokenizer.train import train_sentencepiece
from daralm.training.trainer import Trainer

# A handful of short, distinct sentences — enough variety that memorizing
# them isn't trivially the same as memorizing one repeated string, but few
# and short enough that a 2-layer, 32-dim model can plausibly memorize them
# within a few hundred steps.
_SENTENCES = [
    "the quick brown fox jumps over the lazy dog",
    "a journey of a thousand miles begins with a single step",
    "all that glitters is not gold in this world",
    "actions speak louder than words in every language",
    "practice makes perfect when you never give up",
]


def _build_tokenizer(tmp_path_factory) -> DaraLMTokenizer:
    corpus_dir = tmp_path_factory.mktemp("overfit_corpus")
    corpus_path = corpus_dir / "corpus.txt"
    corpus_path.write_text("\n".join(_SENTENCES * 20), encoding="utf-8")
    model_path = train_sentencepiece(
        corpus_path, corpus_dir / "tok", vocab_size=48, model_type="unigram"
    )
    return DaraLMTokenizer.from_pretrained(model_path)


def test_model_can_overfit_a_tiny_fixed_dataset(tmp_path_factory, tmp_path):
    tokenizer = _build_tokenizer(tmp_path_factory)
    records = [
        {"text": sentence, "language": "en", "source": "test"}
        for sentence in _SENTENCES
        for _ in range(10)  # repeat so packing yields a reasonable number of blocks
    ]

    config = ModelConfig(
        model_name="test-overfit",
        architecture={
            "vocab_size": tokenizer.vocab_size,
            "hidden_size": 32,
            "num_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 64,
            "max_position_embeddings": 16,
        },
        training={
            "learning_rate": 0.005,
            "batch_size": 4,
            "max_steps": 300,
            "warmup_steps": 20,
            "eval_interval": 100,
            "save_interval": 300,
            "log_interval": 100,
        },
    )

    block_size = config.architecture.max_position_embeddings
    dataset = PackedTokenDataset(records, tokenizer, block_size=block_size)
    model = DaraLMTransformer(config.architecture, pad_token_id=tokenizer.pad_id)

    device = torch.device("cpu")
    trainer = Trainer(
        config=config,
        model=model,
        train_dataset=dataset,
        val_dataset=dataset,  # same set on purpose — see module docstring
        device=device,
        checkpoint_dir=tmp_path / "checkpoints",
        tokenizer_path=tmp_path_factory.mktemp("tok2") / "placeholder.model",
    )
    # save_checkpoint would fingerprint tokenizer_path's bytes; this test
    # never saves (save_interval == max_steps is set but train() will still
    # save once at the end) — give it a real, if content-irrelevant, file.
    (trainer.tokenizer_path).write_bytes(b"placeholder")

    initial_loss, _ = compute_perplexity(model, trainer.val_loader, device)
    trainer.train()
    final_loss, _ = compute_perplexity(model, trainer.val_loader, device)

    assert final_loss < initial_loss * 0.5, (
        f"Expected loss to drop by more than half when overfitting a tiny fixed "
        f"dataset; got initial={initial_loss:.3f} final={final_loss:.3f}. If this "
        f"fails, assume a bug in the model/trainer before scaling up (spec section 19)."
    )

    results = check_memorization(model, tokenizer, _SENTENCES, min_tokens=3)
    match_rate = average_match_rate(results)
    assert match_rate > 0.3, (
        f"Expected the overfit model's greedy continuations to substantially "
        f"reproduce the tiny training set; got average match rate {match_rate:.3f}."
    )
