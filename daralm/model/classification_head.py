"""ClassificationHead — the model-family design from `ROADMAP_NLP_PLATFORM.md`
Phase 3: "a small linear/MLP head on top of the frozen (or lightly
fine-tuned) final hidden state, not a new full checkpoint per task."

This is deliberately a separate, small module rather than a change to
`DaraLMTransformer` itself — the backbone stays exactly what it always
was (a causal language model); this wraps it for a different job. A
`DaraLMTransformer` plus a `ClassificationHead` is not a new model class,
it's a composition — the same backbone can be reused across every
classification-family task (language detection, sentiment, intent, ...)
in the roadmap, each with its own small head.

Pooling strategy: last-real-token, not mean-pooling. This is a
decoder-only, causally-masked model — position i's hidden state has only
ever attended to positions <= i, so the *last* real (non-padding) token's
hidden state is the only one that has "seen" the entire input. Mean-
pooling would average in positions that saw a truncated prefix of the
sequence, diluting signal from earlier positions with less context than
later ones — the wrong pooling strategy for a causal model, even though
it's the right one for a bidirectional encoder like BERT.
"""

from __future__ import annotations

import torch
from torch import nn


class ClassificationHead(nn.Module):
    """A linear classifier on top of a `DaraLMTransformer`'s pooled hidden state.

    Not a subclass of `DaraLMTransformer` and doesn't own one — callers
    pass hidden states in directly (typically
    `backbone(input_ids, return_hidden_states=True).hidden_states`), so
    this class has no opinion about which backbone produced them, whether
    it's frozen, or how it was loaded. That separation is what makes "freeze
    the backbone, train only this head" (a linear probe) and "fine-tune
    both together" both possible without two different head implementations.
    """

    def __init__(self, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}")
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor, pad_token_id: int
    ) -> torch.Tensor:
        """Args:
            hidden_states: (batch, seq_len, hidden_size) — a backbone's
                post-final-norm, pre-LM-head output.
            input_ids: (batch, seq_len) — the same token IDs the backbone
                was run on, used only to find each sequence's last
                non-padding position (never fed into the classifier itself).
            pad_token_id: which token ID marks padding.

        Returns:
            (batch, num_classes) raw logits — pass through
            `torch.nn.functional.cross_entropy` for loss, or
            `.argmax(dim=-1)` for a predicted class.
        """
        is_real_token = input_ids != pad_token_id
        # cumsum is non-decreasing and flat across any trailing padding, so
        # its first (argmax's tie-breaking rule) maximum is exactly the
        # last real-token position. Verified for the one edge case that
        # matters: an all-padding row (shouldn't occur for any real
        # example — every real text produces at least one token) yields
        # an all-zero cumsum, and argmax of that is index 0 — a defined,
        # harmless fallback (pools from the first, still-padding, position)
        # rather than an out-of-bounds crash, not a claim that it's a
        # meaningful classification for that row.
        last_real_index = is_real_token.to(torch.long).cumsum(dim=1).argmax(dim=1)

        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        pooled = hidden_states[batch_indices, last_real_index]  # (batch, hidden_size)

        return self.classifier(pooled)
