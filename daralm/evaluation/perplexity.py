from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from daralm.model.transformer import DaraLMTransformer


@torch.no_grad()
def compute_perplexity(
    model: DaraLMTransformer, dataloader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Average loss and perplexity over every batch in `dataloader`."""
    was_training = model.training
    model.eval()

    total_loss = 0.0
    num_batches = 0
    for batch in dataloader:
        batch = batch.to(device)
        output = model(batch, labels=batch)
        total_loss += output.loss.item()
        num_batches += 1

    model.train(was_training)

    if num_batches == 0:
        raise ValueError("dataloader produced zero batches — nothing to evaluate")

    avg_loss = total_loss / num_batches
    perplexity = math.exp(min(avg_loss, 20))  # cap: exp(20) already ~5e8, avoids inf on a bad model
    return avg_loss, perplexity
