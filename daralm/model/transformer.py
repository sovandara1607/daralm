"""DaraLMTransformer — the full decoder-only Transformer.

    Token IDs
      -> Token Embedding
      -> [ Transformer Block ] x num_layers   (RoPE applied inside each block's attention)
      -> Final RMSNorm
      -> LM Head (weight-tied to the token embedding)
      -> Logits

Weight tying: `lm_head.weight` and `token_embedding.embedding.weight` are
the *same* `nn.Parameter` object, not merely initialized to equal values.
Intuitively, the token embedding answers "given this token, what vector
represents it", and the LM head answers "given this vector, how much does
it look like each token" — the same question, asked in opposite
directions, so sharing one matrix for both is a natural inductive bias, and
it removes a `vocab_size * hidden_size` block of parameters (often a
sizeable fraction of a small model's total) that would otherwise be
learned twice. This is exactly the assumption `scripts/inspect_model_config.py`
has used for its parameter estimate since Phase 0 ("LM head adds 0 extra
params — tied to token embedding").

Causal LM loss: given one sequence of token IDs, the model's logits at
position i predict the token at position i+1. So for input "A B C D E",
logits[:-1] (predictions after seeing "A", "AB", "ABC", "ABCD") are compared
against input_ids[1:] ("B C D E") — no separate "labels" argument needs to
be constructed by the caller; the shift happens inside `forward`. Padding
positions are excluded from the loss via `ignore_index`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
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
    # One (key, value) tuple per layer when `use_cache=True`, else None —
    # see `forward`'s docstring and `daralm.inference.generator` for how
    # this is actually used to speed up autoregressive generation.
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None


class DaraLMTransformer(nn.Module):
    """A GPT-style decoder-only Transformer, fully defined by an `ArchitectureConfig`.

    Nothing about the architecture's shape is hard-coded here — instantiate
    the same class from `configs/tiny.yaml`, `configs/50m.yaml`, or any
    future size, and every dimension follows from the config.
    """

    def __init__(self, config: ArchitectureConfig, pad_token_id: int = 0) -> None:
        super().__init__()
        self.config = config
        # Matches the fixed special-token convention established in Phase 2
        # (daralm.tokenizer.train.PAD_ID = 0). Kept as a plain constructor
        # default here, rather than importing the tokenizer package, so the
        # model has no dependency on the tokenizer module.
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

        self._init_weights()

        # Weight tying: point the LM head at the *same* Parameter tensor as
        # the token embedding, rather than a separately-learned one. Done
        # *after* `_init_weights` so there is exactly one clean
        # initialization, not two competing ones for the same tensor.
        self.lm_head.weight = self.token_embedding.embedding.weight

    def _init_weights(self) -> None:
        """Initialize weights the way GPT-2/nanoGPT do — not PyTorch's defaults.

        This matters more than it might look: PyTorch's default `nn.Linear`
        init (Kaiming-uniform, scaled by fan-in) and `nn.Embedding` init
        (unit normal) were not designed with an 8-layer residual stack in
        mind. Left alone, they produce a model whose *initial* logits
        already have huge variance — verified directly: before this method
        existed, a freshly-constructed model's loss on random input was
        ~127 for DaraLM-Tiny and ~510 for DaraLM-50M, when an untrained
        model predicting uniformly over the vocabulary should start around
        `ln(vocab_size)` (~9.7 at vocab_size=16,000). That's not a stylistic
        preference, it's the model being numerically unable to start
        learning sensibly.

        Two rules, both standard since GPT-2:
        1. All `nn.Linear`/`nn.Embedding` weights ~ Normal(0, 0.02).
        2. Each block's two "output" projections — attention's `o_proj`
           (mixes attention heads back together) and the FFN's `down_proj`
           (projects back down to hidden_size) — are the two places, once
           per block, where a sub-layer's result is about to be added into
           the residual stream. With `num_layers` blocks all adding into
           that same stream, its variance would otherwise grow roughly
           linearly with depth. Scaling just these two projections' init
           by an extra `1 / sqrt(2 * num_layers)` keeps the residual
           stream's scale roughly independent of depth, so a 2-layer and a
           50-layer DaraLM both start training from a comparable place.
        """
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

        residual_std = std / math.sqrt(2 * self.config.num_layers)
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
        """Args:
            input_ids: (batch, seq_len) token IDs — the *new* tokens only
                when `past_key_values` is given (e.g. just the one
                newly-sampled token during cached generation), not the
                full sequence so far.
            labels: (batch, seq_len) token IDs to compute loss against — pass
                `input_ids` itself for standard causal LM training. If None,
                only logits are returned (e.g. for inference).
            return_hidden_states: if True, also return the post-final-norm,
                pre-LM-head hidden states (batch, seq_len, hidden_size) —
                what a task head (e.g. `daralm.model.classification_head`)
                pools over, rather than the vocab-sized logits every
                existing caller (causal-LM training, `generate()`) uses.
                Default False and additive-only: every existing call site
                is unaffected, `hidden_states` stays `None` unless asked
                for.
            past_key_values: one (key, value) tuple per layer, from a prior
                call with `use_cache=True`, or None for a normal
                full-sequence forward pass (unchanged default behavior —
                every pre-existing call site passes neither of these two
                new arguments and sees no behavior change at all).
            use_cache: if True, also return this call's per-layer (key,
                value) pairs (concatenated with `past_key_values` if given)
                for the caller to pass back in on the next step. See
                `daralm.model.attention.CausalSelfAttention.forward`'s
                docstring for why this turns generation from O(n^2) into
                O(n) total work.

        Returns:
            DaraLMOutput(logits of shape (batch, seq_len, vocab_size), loss,
            hidden_states of shape (batch, seq_len, hidden_size) or None,
            past_key_values: list of per-layer (key, value) or None).
        """
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
            x, present = block(x, cos, sin, layer_past, use_cache)
            if use_cache:
                present_key_values.append(present)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
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

    def num_parameters(self, exclude_tied: bool = True) -> int:
        """Count trainable parameters.

        `exclude_tied=True` (default) counts the tied embedding/LM-head
        matrix once, matching how `scripts/inspect_model_config.py`
        estimates parameters (and how the model is actually stored — it's
        one tensor, not two, in `state_dict()`'s underlying storage,
        though `state_dict()` will list it under both keys).

        Note: `self.parameters()` (and `self.named_parameters()` with its
        default `remove_duplicate=True`) already de-duplicates shared
        Parameter objects — so naively summing `p.numel()` over it always
        gives the tied-once count. To report the untied count (each tied
        weight counted once per name it appears under) we have to opt back
        into the duplicates explicitly via `named_parameters(remove_duplicate=False)`.
        """
        if exclude_tied:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(
            p.numel()
            for _, p in self.named_parameters(remove_duplicate=False)
            if p.requires_grad
        )
