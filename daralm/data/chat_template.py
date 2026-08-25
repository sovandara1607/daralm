"""Chat template for instruction/response formatting — spec section 25.

    <bos>
    <user>
    What is machine learning?
    </user>
    <assistant>
    Machine learning is...
    </assistant>
    <eos>

`<bos>`/`<eos>` are the tokenizer's own special tokens, added via
`tokenizer.encode(..., add_bos=..., add_eos=...)` — not literal text.
`<user>`, `</user>`, `<assistant>`, `</assistant>` ARE literal text,
encoded as ordinary subword pieces, deliberately *not* new special tokens.

Why not add them as real special tokens: doing so would mean retraining
the tokenizer (a new vocab_size, a differently-shaped embedding matrix)
and would invalidate every Base checkpoint trained so far. Spec section 24
says "keep Base and Instruct models separate" specifically meaning Instruct
is fine-tuned *from* Base's exact weights — same vocab, same architecture,
same embedding matrix. Plain-text markers cost a handful of extra tokens
per turn (a few subword pieces each) instead of one atomic token, but need
zero changes to the tokenizer or the Base checkpoint's shape.
"""

from __future__ import annotations

USER_OPEN = "<user>"
USER_CLOSE = "</user>"
ASSISTANT_OPEN = "<assistant>"
ASSISTANT_CLOSE = "</assistant>"


def format_prompt(instruction: str) -> str:
    """The user-turn text, through the point where the assistant should start responding.

    Used both to build training examples and, at inference time, to wrap a
    user message before generation (see `daralm.inference.generator.generate_chat`).
    """
    return f"{USER_OPEN}\n{instruction.strip()}\n{USER_CLOSE}\n{ASSISTANT_OPEN}\n"


def format_response(response: str) -> str:
    """The assistant-turn text, including its closing marker."""
    return f"{response.strip()}\n{ASSISTANT_CLOSE}"


def format_example(instruction: str, response: str) -> tuple[str, str]:
    """A full training example, split into (prompt_text, response_text).

    The split point is exactly where loss masking should begin — see
    `daralm.data.dataset.InstructionDataset`, which tokenizes each half
    separately for exactly this reason.
    """
    return format_prompt(instruction), format_response(response)
