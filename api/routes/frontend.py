"""GET / — serves the static frontend (`web/index.html`), spec section 27.

A single self-contained HTML file (inline CSS/JS, no build step, no
Node.js toolchain) rather than a Next.js app — see README's "Production
architecture" section for why: it talks to the same-origin `/v1/chat`,
`/v1/generate`, `/v1/tokenize` endpoints via `fetch`, needs no separate
dev/build/deploy story, and ships in the same Docker image as the API
with a single extra `COPY`.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(tags=["frontend"])

# Repo layout: api/routes/frontend.py -> ../../web/index.html
_INDEX_HTML = Path(__file__).resolve().parent.parent.parent / "web" / "index.html"


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_INDEX_HTML)
