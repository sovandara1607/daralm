"""Perplexity: the standard way to turn a loss number into something interpretable.

Cross-entropy loss is measured in nats (natural-log units) and doesn't have
an intuitive scale on its own — "loss 4.2" means little by itself.
Perplexity = exp(loss) rescales it into "the model was, on average, as
uncertain as if it were choosing uniformly among this many tokens" — a
perplexity of 80 means the model's average confusion at each position was
comparable to guessing uniformly among 80 options. Lower is better; a
perfect model scores 1.0 (completely certain, always right).
"""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from daralm.model.transformer import DaraLMTransformer


@torch.no_grad()
def compute_perplexity(
    model: DaraLMTransformer, dataloader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Average loss and perplexity over every batch in `dataloader`.

    Returns (avg_loss, perplexity). Puts the model in eval mode for the
    duration (disables dropout) and restores its prior mode afterward,
    rather than assuming the caller already did so.
    """
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
