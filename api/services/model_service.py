"""ModelService — owns one loaded model + tokenizer and does the actual work.

This is the layer between HTTP (api.routes) and the library code every
earlier phase already built and tested (`daralm.model.transformer`,
`daralm.tokenizer.tokenizer`, `daralm.inference.generator`). Routes never
touch `DaraLMTransformer`/`DaraLMTokenizer` directly — they go through a
`ModelService` instance, for two reasons:

1. **Testability.** Loading a real checkpoint (tens of MB, seconds of I/O)
   on every test would make the API test suite slow and would tie every
   test's correctness to whatever happens to be trained on disk right now.
   `ModelService.__init__` accepts an already-constructed model/tokenizer
   directly, so tests build tiny fixtures (same pattern as
   `tests/test_generation.py`) and never touch a real checkpoint.
   `ModelService.from_checkpoint` is the *only* place real disk I/O
   happens, used solely by `api.main`'s startup.
2. **One process-wide model, loaded once.** Loading a checkpoint per
   request would be needlessly slow and would defeat the point of a long-
   running service — `api.main` builds exactly one `ModelService` at
   startup and every request reuses it.
"""

from __future__ import annotations

from pathlib import Path

import torch
from starlette.concurrency import run_in_threadpool

from daralm.inference.generator import generate, generate_chat
from daralm.model.config import ModelConfig
from daralm.model.transformer import DaraLMTransformer
from daralm.tokenizer.tokenizer import DaraLMTokenizer
from daralm.training.checkpoint import load_checkpoint
from daralm.utils.device import get_device, get_device_name
from daralm.utils.logging import get_logger

logger = get_logger(__name__)


class ModelService:
    """Wraps one loaded `DaraLMTransformer` + `DaraLMTokenizer` pair."""

    def __init__(
        self,
        model: DaraLMTransformer,
        tokenizer: DaraLMTokenizer,
        config: ModelConfig,
        device: torch.device,
        checkpoint_step: int | None = None,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()  # generation must never see training-mode dropout
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.checkpoint_step = checkpoint_step

    @classmethod
    def from_checkpoint(
        cls,
        config_path: str | Path,
        checkpoint_dir: str | Path,
        tokenizer_path: str | Path,
    ) -> ModelService:
        """Load a model for serving: config -> tokenizer -> weights.

        Fails loudly (via the existing `ModelConfig.from_yaml` / tokenizer
        `FileNotFoundError` / `load_checkpoint`'s own checks) rather than
        starting a service that would 500 on the first real request — a
        bad config, missing tokenizer, or vocab-size mismatch is a startup
        error, not a runtime one.
        """
        config = ModelConfig.from_yaml(config_path)
        tokenizer = DaraLMTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.vocab_size != config.architecture.vocab_size:
            raise ValueError(
                f"Tokenizer vocab_size ({tokenizer.vocab_size}) does not match "
                f"{config_path}'s architecture.vocab_size ({config.architecture.vocab_size})."
            )

        device = get_device()
        model = DaraLMTransformer(config.architecture, pad_token_id=tokenizer.pad_id)
        # map_location deliberately left at load_checkpoint's default ("cpu"),
        # matching scripts/train.py and scripts/train_sft.py — torch.set_rng_state
        # only accepts a CPU ByteTensor, so loading the checkpoint's saved RNG
        # state directly onto an MPS/CUDA device crashes. `model.load_state_dict`
        # copies values in place regardless of the source tensor's device, so
        # the weights still end up on `device` once `ModelService.__init__`
        # calls `model.to(device)` below — this is a load-then-move ordering,
        # not a "load straight to the target device" one.
        checkpoint_info = load_checkpoint(checkpoint_dir, model, tokenizer_path=tokenizer_path)
        logger.info(
            "Loaded %s from %s (step=%d) on %s",
            config.model_name,
            checkpoint_dir,
            checkpoint_info["step"],
            get_device_name(device),
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
            checkpoint_step=checkpoint_info["step"],
        )

    def model_info(self) -> dict:
        """Everything GET /v1/model reports — read from the config/model already in memory."""
        arch = self.config.architecture
        return {
            "model_name": self.config.model_name,
            "tags": self.config.tags,
            "vocab_size": arch.vocab_size,
            "hidden_size": arch.hidden_size,
            "num_layers": arch.num_layers,
            "num_attention_heads": arch.num_attention_heads,
            "max_position_embeddings": arch.max_position_embeddings,
            "parameters": self.model.num_parameters(),
            "checkpoint_step": self.checkpoint_step,
            "device": get_device_name(self.device),
        }

    def tokenize(self, text: str, add_bos: bool, add_eos: bool) -> dict:
        """Token IDs plus their human-readable subword pieces, kept aligned.

        `DaraLMTokenizer.tokenize()` (piece strings) and `.encode()` (IDs)
        don't naturally align once `add_bos`/`add_eos` are requested —
        `encode()` prepends/appends the special-token ID, but the raw piece
        list from SentencePiece never included those tokens to begin with.
        Prepending/appending the literal `<bos>`/`<eos>` strings here keeps
        `tokens` and `token_ids` the same length and index-aligned, which
        matters for a debugging endpoint whose whole point is inspecting
        exactly what the model will see.
        """
        pieces = self.tokenizer.tokenize(text)
        ids = self.tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
        if add_bos:
            pieces = ["<bos>", *pieces]
        if add_eos:
            pieces = [*pieces, "<eos>"]
        return {"token_ids": ids, "tokens": pieces, "token_count": len(ids)}

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        stop_on_eos: bool,
    ) -> dict:
        """Run generation off the event loop.

        `generate()` is synchronous, CPU/MPS-bound PyTorch work — calling
        it directly inside an `async def` route would block the whole
        event loop (and therefore every other concurrent request) for the
        entire generation. `run_in_threadpool` runs it in a worker thread
        instead, which is the standard fix for exactly this class of
        problem (any synchronous, blocking call inside async code) rather
        than something specific to model inference.
        """
        prompt_len = len(self.tokenizer.encode(prompt, add_bos=True, add_eos=False))
        full_text = await run_in_threadpool(
            generate,
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            stop_on_eos,
        )
        full_ids = self.tokenizer.encode(full_text, add_bos=True, add_eos=False)
        generated_len = len(full_ids) - prompt_len
        return {
            "generated_text": full_text,
            # Not exactly max_new_tokens when stop_on_eos triggers early, or
            # when re-tokenizing the decoded text doesn't perfectly round-trip
            # to the same ID count SentencePiece produced during generation
            # (whitespace normalization can shift token boundaries slightly)
            # — an honest re-measurement, not the requested cap echoed back.
            "tokens_generated": max(generated_len, 0),
        }

    async def chat(
        self,
        instruction: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> dict:
        """Run chat-templated generation off the event loop — same
        `run_in_threadpool` reasoning as `generate()` above.

        Unlike `generate()`, no prompt-length subtraction is needed:
        `generate_chat()` already returns only the assistant's response
        text (the chat-template prompt is stripped before it's returned),
        so `tokens_generated` is a direct encode-and-count, not a diff.
        """
        response_text = await run_in_threadpool(
            generate_chat,
            self.model,
            self.tokenizer,
            instruction,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
        )
        tokens_generated = len(self.tokenizer.encode(response_text, add_bos=False, add_eos=False))
        return {"response": response_text, "tokens_generated": tokens_generated}
