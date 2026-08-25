"""DaraLM evaluation: perplexity, generation quality, benchmarks.

`perplexity.py` and `generation.py` (memorization-checking) got a head
start in Phase 5, which needed both to run the overfitting sanity test.
Phase 8's `benchmarks.py` builds on them: fixed cross-checkpoint prompts,
the overfitting (train-vs-val) check, and a real-corpus memorization check,
applied uniformly across every trained model (spec section 15, in full).
"""
