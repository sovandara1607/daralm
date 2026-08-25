"""DaraLM inference API — Phase 10, spec section 26.

Wraps `daralm.inference.generator`/`daralm.tokenizer.tokenizer` (already
built and tested in Phases 2-9) behind a FastAPI service, per the
architecture the spec lays out:

    Client -> FastAPI -> Model Service -> Tokenizer -> DaraLM -> Generation -> Response

No model logic lives in this package — `api.services.model_service` is a
thin adapter over the existing library code, and `api.routes` are thinner
still (parse request, call the service, shape the response). The actual
Transformer, tokenizer, and generation loop are exactly the ones every
earlier phase already tested; this phase's only new code is the serving
layer around them.
"""
