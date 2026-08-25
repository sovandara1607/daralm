"""DaraLM tokenizer training and evaluation.

Phase 2. SentencePiece-backed (see `train.py` for why). `tokenizer.py` is
the stable wrapper the rest of the codebase should depend on;
`evaluation.py` measures fragmentation/compression/unk-rate across Khmer,
English, mixed, and code/numbers/URL text.
"""
