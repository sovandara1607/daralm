"""The training loop.

Implements the professional-training-loop checklist from spec section 11:
mixed precision, gradient accumulation, gradient clipping, periodic
validation, checkpointing, resume, LR scheduling, and the exact log line
format the spec asks for. Deliberately does NOT implement gradient
checkpointing, FlashAttention, or distributed training — those are later,
explicitly-staged optimizations (spec section 20), not baseline-correctness
concerns.

Why gradient accumulation works: a single micro-batch of `batch_size`
sequences may be all that fits in memory, but a larger *effective* batch
size (`batch_size * gradient_accumulation_steps`) often trains better —
more sequences' gradients averaged together means a less noisy update.
Accumulation simulates that larger batch by summing (technically,
averaging) gradients across several forward/backward passes *before*
calling `optimizer.step()` once, rather than stepping after every
micro-batch — mathematically equivalent to one big batch, at the cost of
extra forward/backward passes instead of extra memory.

Why validation loss matters (not just training loss): training loss only
tells you the model is fitting the data it's *seen*. Validation loss (on
held-out data, from `daralm.data.dataset.split_dataset`) is the only signal
in this loop for whether the model is learning something that generalizes,
versus memorizing the training set — the gap between the two is literally
the definition of overfitting.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from daralm.data.dataset import InstructionDataset, PackedTokenDataset
from daralm.model.config import ModelConfig
from daralm.model.transformer import DaraLMTransformer
from daralm.training.checkpoint import save_checkpoint, write_meta_json
from daralm.training.optimizer import build_optimizer
from daralm.training.scheduler import build_scheduler
from daralm.utils.logging import get_logger

logger = get_logger(__name__)

# Precision -> the torch dtype used inside the autocast region during the
# forward pass. fp32 uses no autocast at all (full precision throughout).
_AUTOCAST_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16}


def _gpu_memory_gb(device: torch.device) -> float | None:
    """Current allocated device memory in GB, or None if not measurable.

    CUDA and MPS expose this through different (non-interchangeable) APIs;
    CPU has no meaningful "GPU memory" at all. Returning None (rather than
    a string like the log line used to) lets callers decide how to display
    or aggregate it, rather than baking formatting in here.
    """
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024**3
    if device.type == "mps":
        # torch.mps has no "max/peak" counter like CUDA's — only current
        # allocation. Still a genuinely useful throughput/memory signal on
        # Apple Silicon, which this whole project has been developed and
        # trained on; unlike CUDA there's no equivalent of nvidia-smi's
        # utilization percentage available through PyTorch here at all.
        return torch.mps.current_allocated_memory() / 1024**3
    return None


def _unpack_batch(batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (input_ids, labels) regardless of which dataset produced `batch`.

    `PackedTokenDataset` (base pretraining) yields a single tensor —
    `labels is input_ids`, matching `DaraLMTransformer.forward`'s default
    causal-LM convention. `InstructionDataset` (Phase 9 SFT) yields
    `(input_ids, labels)` pairs with the prompt portion of `labels` masked
    out — PyTorch's default DataLoader collation turns a dataset of 2-tuples
    into a 2-element list of stacked tensors, which is exactly what this
    branch expects.
    """
    if isinstance(batch, (list, tuple)):
        input_ids, labels = batch
        return input_ids, labels
    return batch, batch


@dataclass
class TrainerState:
    """Everything needed to resume — populated at construction or on `load_checkpoint`."""

    step: int = 0
    tokens_processed: int = 0
    best_val_loss: float = float("inf")


class Trainer:
    """Owns the model, optimizer, scheduler, and the training/eval loop for one run."""

    def __init__(
        self,
        config: ModelConfig,
        model: DaraLMTransformer,
        train_dataset: PackedTokenDataset | InstructionDataset,
        val_dataset: PackedTokenDataset | InstructionDataset,
        device: torch.device,
        checkpoint_dir: str | Path,
        tokenizer_path: str | Path,
    ) -> None:
        self.config = config
        self.training_config = config.training
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.tokenizer_path = tokenizer_path

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.training_config.batch_size, shuffle=True, drop_last=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.training_config.batch_size, shuffle=False, drop_last=True
        )

        self.optimizer = build_optimizer(self.model, self.training_config)
        self.scheduler = build_scheduler(self.optimizer, self.training_config)

        precision = self.training_config.precision
        self.autocast_dtype = _AUTOCAST_DTYPE.get(precision)
        # GradScaler is only needed for fp16 (its narrow exponent range can
        # under/overflow small gradients); bf16 has fp32's exponent range
        # and doesn't need one, and fp32 obviously doesn't either.
        self.grad_scaler = torch.amp.GradScaler(enabled=(precision == "fp16"))

        self.state = TrainerState()
        # A structured, machine-readable timeline of the run — one entry per
        # log_interval/eval_interval event — written to history.json
        # alongside checkpoints. This is what Phase 6's analysis script
        # reads to report on training/validation loss, tokens/sec, and
        # memory over the course of a run, rather than re-parsing log text.
        self.history: list[dict] = []

    def _autocast(self):
        if self.autocast_dtype is None:
            return torch.autocast(device_type=self.device.type, enabled=False)
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

    def _run_validation(self) -> float:
        """Average loss over the full validation set. Model is returned to train mode after."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids, labels = _unpack_batch(batch)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                with self._autocast():
                    output = self.model(input_ids, labels=labels)
                total_loss += output.loss.item()
                num_batches += 1
        self.model.train()
        if num_batches == 0:
            raise RuntimeError(
                "Validation dataset produced zero batches — likely too few tokens for even "
                "one batch at the configured batch_size/block_size."
            )
        return total_loss / num_batches

    def _train_step(self, batch_iter) -> tuple[float, int]:
        """Run one optimizer step (across `gradient_accumulation_steps` micro-batches).

        Returns (loss_for_logging, tokens_processed_this_step).
        """
        self.optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        tokens_this_step = 0
        accumulation_steps = self.training_config.gradient_accumulation_steps

        for _ in range(accumulation_steps):
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(self.train_loader)
                batch = next(batch_iter)
            input_ids, labels = _unpack_batch(batch)
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            tokens_this_step += input_ids.numel()

            with self._autocast():
                output = self.model(input_ids, labels=labels)
                loss = output.loss / accumulation_steps

            self.grad_scaler.scale(loss).backward()
            accumulated_loss += loss.item()

        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.gradient_clip)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.scheduler.step()

        return accumulated_loss, tokens_this_step

    def train(self) -> TrainerState:
        """Run the full training loop from `self.state.step` to `max_steps`."""
        max_steps = self.training_config.max_steps
        logger.info(
            "Starting training: model=%s device=%s max_steps=%d (resuming from step=%d)",
            self.config.model_name,
            self.device,
            max_steps,
            self.state.step,
        )

        self.model.train()
        batch_iter = iter(self.train_loader)
        step_start_time = time.monotonic()

        while self.state.step < max_steps:
            train_loss, tokens_this_step = self._train_step(batch_iter)
            self.state.step += 1
            self.state.tokens_processed += tokens_this_step

            if self.state.step % self.training_config.log_interval == 0:
                elapsed = time.monotonic() - step_start_time
                tokens_per_sec = tokens_this_step * self.training_config.log_interval / max(
                    elapsed, 1e-9
                )
                gpu_memory_gb = _gpu_memory_gb(self.device)
                gpu_memory_str = f"{gpu_memory_gb:.2f}GB" if gpu_memory_gb is not None else "n/a"
                lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    "step=%d train_loss=%.4f lr=%.2e tokens/sec=%.0f gpu_memory=%s",
                    self.state.step,
                    train_loss,
                    lr,
                    tokens_per_sec,
                    gpu_memory_str,
                )
                self.history.append(
                    {
                        "step": self.state.step,
                        "tokens_processed": self.state.tokens_processed,
                        "train_loss": train_loss,
                        "lr": lr,
                        "tokens_per_sec": tokens_per_sec,
                        "gpu_memory_gb": gpu_memory_gb,
                    }
                )
                step_start_time = time.monotonic()

            if self.state.step % self.training_config.eval_interval == 0:
                val_loss = self._run_validation()
                perplexity = math.exp(min(val_loss, 20))  # cap to avoid inf from a bad early step
                logger.info(
                    "step=%d val_loss=%.4f perplexity=%.1f", self.state.step, val_loss, perplexity
                )
                # Attach to the most recent history entry if it's this same
                # step (log_interval and eval_interval often coincide);
                # otherwise append a val-only entry.
                if self.history and self.history[-1]["step"] == self.state.step:
                    self.history[-1]["val_loss"] = val_loss
                    self.history[-1]["perplexity"] = perplexity
                else:
                    self.history.append(
                        {"step": self.state.step, "val_loss": val_loss, "perplexity": perplexity}
                    )
                if val_loss < self.state.best_val_loss:
                    self.state.best_val_loss = val_loss
                    self._save("best", val_loss, perplexity)

            if self.state.step % self.training_config.save_interval == 0:
                val_loss_for_meta = self.state.best_val_loss
                self._save(f"step-{self.state.step}", val_loss_for_meta, None)
                self._write_history()

        self._write_history()
        logger.info("Training complete at step=%d", self.state.step)
        return self.state

    def _write_history(self) -> None:
        """Write the run's metrics timeline to history.json alongside checkpoints."""
        history_dir = self.checkpoint_dir / self.config.model_name
        history_dir.mkdir(parents=True, exist_ok=True)
        with (history_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def _save(self, tag: str, val_loss: float | None, perplexity: float | None) -> None:
        checkpoint_path = save_checkpoint(
            self.checkpoint_dir,
            tag,
            self.model,
            self.optimizer,
            self.scheduler,
            self.state.step,
            self.state.tokens_processed,
            self.config,
            self.tokenizer_path,
        )
        write_meta_json(
            checkpoint_path,
            step=self.state.step,
            tokens_processed=self.state.tokens_processed,
            val_loss=val_loss,
            perplexity=perplexity,
        )
