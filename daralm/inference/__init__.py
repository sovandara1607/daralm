"""DaraLM inference: text generation and sampling strategies.

Phase 4 (basic autoregressive generation, needed to verify the tiny-model
pipeline end to end) implements this, ahead of Phase 10 which is the
FastAPI *serving* layer around it — `generator.py`/`sampling.py` are the
library code Phase 10's API will call, not duplicated by it.
"""
