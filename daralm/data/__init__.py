"""DaraLM data pipeline: loading, cleaning, deduplication, dataset assembly.

Phase 1. Raw text -> `loader` -> `preprocessing` (clean + filter) ->
`deduplication` -> `dataset` (train/val/test split). Tokenization and
sequence packing are out of scope here — see Phase 2.
"""
