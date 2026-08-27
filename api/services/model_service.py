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
        """Load a model for serving: config -> tokenizer -> weights."""
        config = ModelConfig.from_yaml(config_path)
        tokenizer = DaraLMTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.vocab_size != config.architecture.vocab_size:
            raise ValueError(
                f"Tokenizer vocab_size ({tokenizer.vocab_size}) does not match "
                f"{config_path}'s architecture.vocab_size ({config.architecture.vocab_size})."
            )

        device = get_device()
        model = DaraLMTransformer(config.architecture, pad_token_id=tokenizer.pad_id)
        # map_location deliberately left at load_checkpoint's default ("cpu").
        checkpoint_path = Path(checkpoint_dir)
        release_path = (
            checkpoint_path / "pytorch_model.pt" if checkpoint_path.is_dir() else checkpoint_path
        )
        if (checkpoint_path / "checkpoint.pt").exists():
            # Resumable training checkpoint: retain the strict tokenizer fingerprint.
            checkpoint_info = load_checkpoint(checkpoint_path, model, tokenizer_path=tokenizer_path)
        elif release_path.name == "pytorch_model.pt" and release_path.exists():
            release = torch.load(release_path, map_location="cpu", weights_only=True)
            model.load_state_dict(release["model_state_dict"])
            checkpoint_info = {
                "step": release.get("step"),
                "tokens_processed": release.get("tokens_processed", 0),
                "config": release.get("config", {}),
            }
            logger.info(
                "Loaded inference release from %s (step=%s)",
                release_path,
                checkpoint_info["step"],
            )
        else:
            raise FileNotFoundError(
                f"No checkpoint.pt or pytorch_model.pt found at {checkpoint_path}"
            )
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
        """Token IDs plus their human-readable subword pieces, kept aligned."""
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
        """Run generation off the event loop."""
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
            # EOS and decoded-text retokenization can reduce this count.
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
