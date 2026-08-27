"""Chat template for instruction/response formatting."""

from __future__ import annotations

USER_OPEN = "<user>"
USER_CLOSE = "</user>"
ASSISTANT_OPEN = "<assistant>"
ASSISTANT_CLOSE = "</assistant>"


def format_prompt(instruction: str) -> str:
    return f"{USER_OPEN}\n{instruction.strip()}\n{USER_CLOSE}\n{ASSISTANT_OPEN}\n"


def format_response(response: str) -> str:
    """The assistant-turn text, including its closing marker."""
    return f"{response.strip()}\n{ASSISTANT_CLOSE}"


def format_example(instruction: str, response: str) -> tuple[str, str]:
    """A full training example, split into (prompt_text, response_text)."""
    return format_prompt(instruction), format_response(response)
