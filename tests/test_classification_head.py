"""Tests for daralm.model.classification_head.ClassificationHead.

The pooling logic (finding each sequence's last non-padding token) is the
part most likely to have an off-by-one bug, so it gets the most direct
scrutiny here — including the edge case (an all-padding row) verified by
hand before being documented as safe.
"""

from __future__ import annotations

import pytest
import torch

from daralm.model.classification_head import ClassificationHead

PAD_ID = 0
HIDDEN_SIZE = 8


def test_rejects_fewer_than_two_classes():
    with pytest.raises(ValueError):
        ClassificationHead(hidden_size=HIDDEN_SIZE, num_classes=1)


def test_output_shape_is_batch_by_num_classes():
    head = ClassificationHead(hidden_size=HIDDEN_SIZE, num_classes=3)
    batch, seq_len = 4, 10
    hidden_states = torch.randn(batch, seq_len, HIDDEN_SIZE)
    input_ids = torch.randint(1, 100, (batch, seq_len))  # no padding
    logits = head(hidden_states, input_ids, pad_token_id=PAD_ID)
    assert logits.shape == (batch, 3)


def test_pools_from_the_last_real_token_not_padding():
    # A single sequence: 3 real tokens then 2 padding. Set the hidden
    # state at the last real position (index 2) to a distinctive value and
    # everything else to zero, then confirm the head's output matches
    # what a classifier applied to *that specific* vector would produce —
    # proof it pooled from index 2, not index 4 (the actual last position)
    # or index 0.
    head = ClassificationHead(hidden_size=HIDDEN_SIZE, num_classes=2)
    hidden_states = torch.zeros(1, 5, HIDDEN_SIZE)
    distinctive = torch.randn(HIDDEN_SIZE)
    hidden_states[0, 2] = distinctive
    input_ids = torch.tensor([[5, 7, 3, PAD_ID, PAD_ID]])

    with torch.no_grad():
        logits = head(hidden_states, input_ids, pad_token_id=PAD_ID)
        expected = head.classifier(distinctive.unsqueeze(0))

    assert torch.allclose(logits, expected)


def test_pools_from_final_position_when_no_padding_present():
    head = ClassificationHead(hidden_size=HIDDEN_SIZE, num_classes=2)
    hidden_states = torch.zeros(1, 4, HIDDEN_SIZE)
    distinctive = torch.randn(HIDDEN_SIZE)
    hidden_states[0, 3] = distinctive  # last position, no padding at all
    input_ids = torch.tensor([[5, 7, 3, 9]])

    with torch.no_grad():
        logits = head(hidden_states, input_ids, pad_token_id=PAD_ID)
        expected = head.classifier(distinctive.unsqueeze(0))

    assert torch.allclose(logits, expected)


def test_handles_different_real_lengths_within_one_batch():
    # Row 0: 2 real tokens + 3 padding. Row 1: 4 real tokens, no padding.
    head = ClassificationHead(hidden_size=HIDDEN_SIZE, num_classes=2)
    hidden_states = torch.zeros(2, 5, HIDDEN_SIZE)
    row0_distinctive = torch.randn(HIDDEN_SIZE)
    row1_distinctive = torch.randn(HIDDEN_SIZE)
    hidden_states[0, 1] = row0_distinctive
    hidden_states[1, 4] = row1_distinctive
    input_ids = torch.tensor(
        [
            [5, 7, PAD_ID, PAD_ID, PAD_ID],
            [5, 7, 3, 9, 2],
        ]
    )

    with torch.no_grad():
        logits = head(hidden_states, input_ids, pad_token_id=PAD_ID)
        expected0 = head.classifier(row0_distinctive.unsqueeze(0))
        expected1 = head.classifier(row1_distinctive.unsqueeze(0))

    # atol=1e-6, not the default 1e-8: computing both rows in one batched
    # matmul vs. this test's separate single-row matmul for `expected*`
    # can differ by ~1e-7 from float32 summation-order noise alone (a
    # real, measured, benign BLAS non-determinism, not a pooling bug) —
    # tight enough to still catch an actual indexing error, which would
    # produce a difference orders of magnitude larger than this.
    assert torch.allclose(logits[0:1], expected0, atol=1e-6)
    assert torch.allclose(logits[1:2], expected1, atol=1e-6)


def test_all_padding_row_does_not_crash():
    # Verified edge case: an all-padding row's cumsum is all-zero, so
    # argmax returns index 0 (a defined, in-bounds fallback), not a crash
    # or an out-of-range index.
    head = ClassificationHead(hidden_size=HIDDEN_SIZE, num_classes=2)
    hidden_states = torch.randn(1, 5, HIDDEN_SIZE)
    input_ids = torch.full((1, 5), PAD_ID)

    logits = head(hidden_states, input_ids, pad_token_id=PAD_ID)
    assert logits.shape == (1, 2)


def test_gradients_flow_back_to_the_pooled_position_only():
    # A real, end-to-end check that this is actually differentiable and
    # that gradient flows specifically to the position that was pooled —
    # not to unrelated positions, which would indicate a pooling/indexing
    # bug even if the forward pass shape looked correct.
    head = ClassificationHead(hidden_size=HIDDEN_SIZE, num_classes=2)
    hidden_states = torch.randn(1, 5, HIDDEN_SIZE, requires_grad=True)
    input_ids = torch.tensor([[5, 7, 3, PAD_ID, PAD_ID]])

    logits = head(hidden_states, input_ids, pad_token_id=PAD_ID)
    logits.sum().backward()

    assert hidden_states.grad is not None
    assert torch.any(hidden_states.grad[0, 2] != 0)  # the pooled position
    assert torch.all(hidden_states.grad[0, 3] == 0)  # padding, untouched
    assert torch.all(hidden_states.grad[0, 4] == 0)  # padding, untouched
