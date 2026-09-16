"""Tests for daralm.training — optimizer, scheduler, and the Trainer loop."""

from __future__ import annotations

import json

import pytest
import torch

from daralm.data.dataset import InstructionDataset, PackedTokenDataset
from daralm.model.config import ModelConfig
from daralm.model.transformer import DaraLMTransformer
from daralm.tokenizer.tokenizer import DaraLMTokenizer
from daralm.tokenizer.train import train_sentencepiece
from daralm.training.optimizer import build_optimizer
from daralm.training.scheduler import build_scheduler
from daralm.training.trainer import Trainer


def _tiny_config(**training_overrides) -> ModelConfig:
    training = {
        "learning_rate": 0.01,
        "batch_size": 2,
        "max_steps": 20,
        "warmup_steps": 4,
        "eval_interval": 10,
        "save_interval": 10,
        "log_interval": 5,
    }
    training.update(training_overrides)
    return ModelConfig(
        model_name="test-trainer-model",
        architecture={
            "vocab_size": 30,
            "hidden_size": 16,
            "num_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 32,
            "max_position_embeddings": 16,
        },
        training=training,
    )


def test_optimizer_splits_decay_and_no_decay_params():
    config = _tiny_config()
    model = DaraLMTransformer(config.architecture)
    optimizer = build_optimizer(model, config.training)
    assert len(optimizer.param_groups) == 2
    decayed, not_decayed = optimizer.param_groups
    assert decayed["weight_decay"] == config.training.weight_decay
    assert not_decayed["weight_decay"] == 0.0
    no_decay_param_ids = {id(p) for p in not_decayed["params"]}
    for name, param in model.named_parameters():
        if "norm" in name:
            assert id(param) in no_decay_param_ids


def test_optimizer_unknown_type_raises():
    config = _tiny_config()
    config.training.optimizer = "not_a_real_optimizer"
    model = DaraLMTransformer(config.architecture)
    with pytest.raises(ValueError):
        build_optimizer(model, config.training)


def test_scheduler_warmup_increases_lr_linearly():
    config = _tiny_config(warmup_steps=4, max_steps=20)
    model = DaraLMTransformer(config.architecture)
    optimizer = build_optimizer(model, config.training)
    scheduler = build_scheduler(optimizer, config.training)

    lrs = []
    for _ in range(4):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    assert all(lrs[i] < lrs[i + 1] for i in range(len(lrs) - 1))


def test_scheduler_reaches_peak_lr_after_warmup():
    config = _tiny_config(warmup_steps=4, max_steps=20)
    model = DaraLMTransformer(config.architecture)
    optimizer = build_optimizer(model, config.training)
    scheduler = build_scheduler(optimizer, config.training)
    peak_lr = config.training.learning_rate

    for _ in range(4):
        scheduler.step()
    assert abs(scheduler.get_last_lr()[0] - peak_lr) < 1e-9


def test_scheduler_decays_toward_min_lr_ratio():
    config = _tiny_config(warmup_steps=2, max_steps=20)
    model = DaraLMTransformer(config.architecture)
    optimizer = build_optimizer(model, config.training)
    scheduler = build_scheduler(optimizer, config.training)
    peak_lr = config.training.learning_rate
    min_lr = peak_lr * config.training.min_lr_ratio

    for _ in range(20):
        scheduler.step()
    final_lr = scheduler.get_last_lr()[0]
    assert final_lr < peak_lr
    assert final_lr >= min_lr - 1e-9


@torch.no_grad()
def _make_tiny_tokenizer(tmp_path_factory):
    corpus_dir = tmp_path_factory.mktemp("trainer_corpus")
    corpus_path = corpus_dir / "corpus.txt"
    lines = ["the quick brown fox jumps over the lazy dog"] * 100
    corpus_path.write_text("\n".join(lines), encoding="utf-8")
    model_path = train_sentencepiece(
        corpus_path, corpus_dir / "tok", vocab_size=32, model_type="unigram"
    )
    return DaraLMTokenizer.from_pretrained(model_path)


def test_trainer_runs_and_decreases_loss(tmp_path_factory, tmp_path):
    tokenizer = _make_tiny_tokenizer(tmp_path_factory)
    records = [
        {"text": "the quick brown fox jumps over the lazy dog", "language": "en", "source": "s"}
        for _ in range(80)
    ]
    train_records, val_records = records[:60], records[60:]

    config = _tiny_config(
        max_steps=15, warmup_steps=2, eval_interval=5, save_interval=15, log_interval=5
    )
    config.architecture = config.architecture.model_copy(
        update={"vocab_size": tokenizer.vocab_size}
    )

    block_size = 8
    train_dataset = PackedTokenDataset(train_records, tokenizer, block_size=block_size)
    val_dataset = PackedTokenDataset(val_records, tokenizer, block_size=block_size)

    tokenizer_file = tmp_path_factory.mktemp("tokfile") / "tok.model"
    tokenizer_file.write_bytes(b"dummy-bytes-for-fingerprint")

    model = DaraLMTransformer(config.architecture, pad_token_id=tokenizer.pad_id)
    trainer = Trainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path / "checkpoints",
        tokenizer_path=tokenizer_file,
    )

    first_loss, _ = trainer._train_step(iter(trainer.train_loader))  # noqa: SLF001
    for _ in range(10):
        loss, _ = trainer._train_step(iter(trainer.train_loader))  # noqa: SLF001
    last_loss = loss

    assert last_loss < first_loss


def test_trainer_enables_gradient_checkpointing_when_configured(tmp_path_factory, tmp_path):
    tokenizer = _make_tiny_tokenizer(tmp_path_factory)
    records = [
        {"text": "the quick brown fox jumps over the lazy dog", "language": "en", "source": "s"}
        for _ in range(80)
    ]
    train_records, val_records = records[:60], records[60:]

    config = _tiny_config(
        max_steps=15,
        warmup_steps=2,
        eval_interval=5,
        save_interval=15,
        log_interval=5,
        gradient_checkpointing=True,
    )
    config.architecture = config.architecture.model_copy(
        update={"vocab_size": tokenizer.vocab_size}
    )

    block_size = 8
    train_dataset = PackedTokenDataset(train_records, tokenizer, block_size=block_size)
    val_dataset = PackedTokenDataset(val_records, tokenizer, block_size=block_size)

    tokenizer_file = tmp_path_factory.mktemp("tokfile-gc") / "tok.model"
    tokenizer_file.write_bytes(b"dummy-bytes-for-fingerprint")

    model = DaraLMTransformer(config.architecture, pad_token_id=tokenizer.pad_id)
    assert model.gradient_checkpointing is False  # off until the Trainer turns it on

    trainer = Trainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path / "checkpoints",
        tokenizer_path=tokenizer_file,
    )

    assert trainer.model.gradient_checkpointing is True

    first_loss, _ = trainer._train_step(iter(trainer.train_loader))  # noqa: SLF001
    for _ in range(10):
        loss, _ = trainer._train_step(iter(trainer.train_loader))  # noqa: SLF001
    assert loss < first_loss  # still trains normally end-to-end, just recomputing on backward


def test_trainer_compile_preserves_checkpoint_compatibility(tmp_path_factory, tmp_path):
    """torch.compile(model) (not used here) prefixes state_dict keys with
    '_orig_mod.' and changes the model's wrapper type — either would silently
    break save_checkpoint/load_checkpoint. Trainer must use in-place
    model.compile() instead, which changes neither."""
    tokenizer = _make_tiny_tokenizer(tmp_path_factory)
    records = [
        {"text": "the quick brown fox jumps over the lazy dog", "language": "en", "source": "s"}
        for _ in range(80)
    ]
    train_records, val_records = records[:60], records[60:]

    config = _tiny_config(
        max_steps=15, warmup_steps=2, eval_interval=5, save_interval=15, log_interval=5,
        compile=True,
    )
    config.architecture = config.architecture.model_copy(
        update={"vocab_size": tokenizer.vocab_size}
    )

    block_size = 8
    train_dataset = PackedTokenDataset(train_records, tokenizer, block_size=block_size)
    val_dataset = PackedTokenDataset(val_records, tokenizer, block_size=block_size)

    tokenizer_file = tmp_path_factory.mktemp("tokfile-compile") / "tok.model"
    tokenizer_file.write_bytes(b"dummy-bytes-for-fingerprint")

    model = DaraLMTransformer(config.architecture, pad_token_id=tokenizer.pad_id)
    trainer = Trainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path / "checkpoints",
        tokenizer_path=tokenizer_file,
    )

    assert type(trainer.model) is DaraLMTransformer
    assert not any(k.startswith("_orig_mod.") for k in trainer.model.state_dict())

    first_loss, _ = trainer._train_step(iter(trainer.train_loader))  # noqa: SLF001
    for _ in range(5):
        loss, _ = trainer._train_step(iter(trainer.train_loader))  # noqa: SLF001
    assert loss < first_loss

    trainer._save("best", val_loss=1.0, perplexity=2.7)  # noqa: SLF001 — must not raise


def test_trainer_full_run_writes_checkpoints(tmp_path_factory, tmp_path):
    tokenizer = _make_tiny_tokenizer(tmp_path_factory)
    records = [
        {"text": "the quick brown fox jumps over the lazy dog", "language": "en", "source": "s"}
        for _ in range(80)
    ]
    train_records, val_records = records[:60], records[60:]

    config = _tiny_config(
        max_steps=10, warmup_steps=2, eval_interval=5, save_interval=5, log_interval=100
    )
    config.architecture = config.architecture.model_copy(
        update={"vocab_size": tokenizer.vocab_size}
    )

    block_size = 8
    train_dataset = PackedTokenDataset(train_records, tokenizer, block_size=block_size)
    val_dataset = PackedTokenDataset(val_records, tokenizer, block_size=block_size)
    model = DaraLMTransformer(config.architecture, pad_token_id=tokenizer.pad_id)

    tokenizer_file = tmp_path_factory.mktemp("tokfile2") / "tok.model"
    tokenizer_file.write_bytes(b"dummy-bytes-for-fingerprint")

    checkpoint_root = tmp_path / "checkpoints"
    trainer = Trainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=torch.device("cpu"),
        checkpoint_dir=checkpoint_root,
        tokenizer_path=tokenizer_file,
    )
    final_state = trainer.train()

    assert final_state.step == 10
    assert (checkpoint_root / config.model_name / "step-5").exists()
    assert (checkpoint_root / config.model_name / "step-10").exists()
    assert (checkpoint_root / config.model_name / "best").exists()


def test_trainer_records_and_writes_history(tmp_path_factory, tmp_path):
    tokenizer = _make_tiny_tokenizer(tmp_path_factory)
    records = [
        {"text": "the quick brown fox jumps over the lazy dog", "language": "en", "source": "s"}
        for _ in range(80)
    ]
    train_records, val_records = records[:60], records[60:]

    config = _tiny_config(
        max_steps=10, warmup_steps=2, eval_interval=5, save_interval=5, log_interval=2
    )
    config.architecture = config.architecture.model_copy(
        update={"vocab_size": tokenizer.vocab_size}
    )

    block_size = 8
    train_dataset = PackedTokenDataset(train_records, tokenizer, block_size=block_size)
    val_dataset = PackedTokenDataset(val_records, tokenizer, block_size=block_size)
    model = DaraLMTransformer(config.architecture, pad_token_id=tokenizer.pad_id)

    tokenizer_file = tmp_path_factory.mktemp("tokfile3") / "tok.model"
    tokenizer_file.write_bytes(b"dummy-bytes-for-fingerprint")

    checkpoint_root = tmp_path / "checkpoints"
    trainer = Trainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=torch.device("cpu"),
        checkpoint_dir=checkpoint_root,
        tokenizer_path=tokenizer_file,
    )
    trainer.train()

    assert len(trainer.history) == 6
    entries_with_val = [e for e in trainer.history if "val_loss" in e]
    assert len(entries_with_val) == 2
    entries_with_train = [e for e in trainer.history if "train_loss" in e]
    assert len(entries_with_train) == 5
    for entry in entries_with_val:
        assert entry["perplexity"] > 0

    history_path = checkpoint_root / config.model_name / "history.json"
    assert history_path.exists()
    with history_path.open() as f:
        loaded = json.load(f)
    assert loaded == trainer.history


def test_trainer_handles_instruction_dataset_batches(tmp_path_factory, tmp_path):
    tokenizer = _make_tiny_tokenizer(tmp_path_factory)
    examples = [
        {
            "instruction": "the quick brown fox",
            "response": "jumps over the lazy dog",
            "language": "en",
            "source": "test",
        }
        for _ in range(40)
    ]
    train_examples, val_examples = examples[:30], examples[30:]

    config = _tiny_config(
        max_steps=10, warmup_steps=2, eval_interval=5, save_interval=10, log_interval=5
    )
    config.architecture = config.architecture.model_copy(
        update={"vocab_size": tokenizer.vocab_size, "max_position_embeddings": 128}
    )

    train_dataset = InstructionDataset(train_examples, tokenizer, block_size=128)
    val_dataset = InstructionDataset(val_examples, tokenizer, block_size=128)
    model = DaraLMTransformer(config.architecture, pad_token_id=tokenizer.pad_id)

    tokenizer_file = tmp_path_factory.mktemp("sft_tokfile") / "tok.model"
    tokenizer_file.write_bytes(b"placeholder")

    trainer = Trainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path / "checkpoints",
        tokenizer_path=tokenizer_file,
    )

    first_loss, _ = trainer._train_step(iter(trainer.train_loader))  # noqa: SLF001
    for _ in range(10):
        last_loss, _ = trainer._train_step(iter(trainer.train_loader))  # noqa: SLF001

    assert last_loss < first_loss

    val_loss = trainer._run_validation()  # noqa: SLF001
    assert val_loss > 0 and val_loss < 20  # sane, finite loss — not NaN/inf
