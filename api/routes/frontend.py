"""GET / — serves the static frontend (`web/index.html`), spec section 27."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(tags=["frontend"])

_INDEX_HTML = Path(__file__).resolve().parent.parent.parent / "web" / "index.html"


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_INDEX_HTML)
