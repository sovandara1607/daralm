from __future__ import annotations

import torch
from torch import nn


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}")
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor, pad_token_id: int
    ) -> torch.Tensor:
        is_real_token = input_ids != pad_token_id
        # Locate the final non-padding token in each sequence.
        last_real_index = is_real_token.to(torch.long).cumsum(dim=1).argmax(dim=1)

        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        pooled = hidden_states[batch_indices, last_real_index]  # (batch, hidden_size)

        return self.classifier(pooled)
