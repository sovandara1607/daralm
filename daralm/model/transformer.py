"""DaraLMTransformer — the full decoder-only Transformer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from daralm.model.config import ArchitectureConfig
from daralm.model.embeddings import RotaryEmbedding, TokenEmbedding
from daralm.model.normalization import RMSNorm
from daralm.model.transformer_block import TransformerBlock


@dataclass
class DaraLMOutput:
    """Return type of `DaraLMTransformer.forward`."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None


class DaraLMTransformer(nn.Module):
    def __init__(self, config: ArchitectureConfig, pad_token_id: int = 0) -> None:
        super().__init__()
        self.config = config
        self.pad_token_id = pad_token_id

        self.token_embedding = TokenEmbedding(config.vocab_size, config.hidden_size)
        self.rotary_embedding = RotaryEmbedding(
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Off by default; Trainer flips this on when TrainingConfig.gradient_checkpointing
        # is set (see gradient_checkpointing_enable/disable below and Model Optimization
        # Stage 3). Never combined with use_cache — training never passes use_cache=True.
        self.gradient_checkpointing = False

        self._init_weights()

        # Share input and output embeddings.
        self.lm_head.weight = self.token_embedding.embedding.weight

    def _init_weights(self) -> None:
        """Initialize weights the way GPT-2/nanoGPT do — not PyTorch's defaults."""
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

        residual_std = std / math.sqrt(2 * self.config.num_layers)
        # Scale residual outputs to keep variance stable as depth grows.
        for name, param in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=residual_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_hidden_states: bool = False,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> DaraLMOutput:
        batch_size, seq_len = input_ids.shape
        past_len = past_key_values[0][0].size(2) if past_key_values is not None else 0
        if past_len + seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"past_len + seq_len ({past_len + seq_len}) exceeds "
                f"max_position_embeddings ({self.config.max_position_embeddings})"
            )

        x = self.token_embedding(input_ids)
        cos, sin = self.rotary_embedding(seq_len, device=input_ids.device, offset=past_len)

        present_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                # use_cache is always False here (checkpointing only applies during
                # training, and Trainer never requests a KV cache) — recomputing
                # this block's forward pass on the backward pass is what buys back
                # the activation memory it isn't storing.
                x, present = torch.utils.checkpoint.checkpoint(
                    block, x, cos, sin, layer_past, use_cache, use_reentrant=False
                )
            else:
                x, present = block(x, cos, sin, layer_past, use_cache)
            if use_cache:
                present_key_values.append(present)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Position i predicts the token at i + 1.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.pad_token_id,
            )

        return DaraLMOutput(
            logits=logits,
            loss=loss,
            hidden_states=x if return_hidden_states else None,
            past_key_values=present_key_values if use_cache else None,
        )

    def gradient_checkpointing_enable(self) -> None:
        """Recompute each TransformerBlock's forward pass during backward instead of
        storing its activations — lower peak memory, slower per-step (Stage 3)."""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def num_parameters(self, exclude_tied: bool = True) -> int:
        """Count trainable parameters."""
        if exclude_tied:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(
            p.numel() for _, p in self.named_parameters(remove_duplicate=False) if p.requires_grad
        )
