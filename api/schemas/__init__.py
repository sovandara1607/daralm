"""Pydantic request/response models for the DaraLM API.

Every route validates its input against one of these before touching the
model — the same "fail loudly" discipline `daralm.model.config` applies to
YAML configs, applied here to HTTP requests: a malformed request gets a
clear 422 with the exact field that's wrong, not a stack trace from deep
inside generation.
"""
