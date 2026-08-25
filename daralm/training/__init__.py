"""DaraLM training: optimizer, LR scheduler, checkpointing, and the training loop.

Phase 4. `trainer.py`'s `Trainer` class owns the loop; `optimizer.py` and
`scheduler.py` build its two moving parts; `checkpoint.py` handles
save/resume. Distributed training (`distributed.py`) is intentionally not
built yet — explicitly a later-phase optimization (spec section 20), not a
baseline-correctness concern.
"""
