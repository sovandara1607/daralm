#!/usr/bin/env python
"""The Phase 5 sanity gate: can DaraLM-Tiny overfit a tiny, fixed dataset?."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daralm.data.dataset import PackedTokenDataset  # noqa: E402
from daralm.data.loader import load_jsonl  # noqa: E402
from daralm.evaluation.generation import average_match_rate, check_memorization  # noqa: E402
from daralm.evaluation.perplexity import compute_perplexity  # noqa: E402
from daralm.model.config import ModelConfig  # noqa: E402
from daralm.model.transformer import DaraLMTransformer  # noqa: E402
from daralm.tokenizer.tokenizer import DaraLMTokenizer  # noqa: E402
from daralm.training.trainer import Trainer  # noqa: E402
from daralm.utils.device import get_device, get_device_name  # noqa: E402
from daralm.utils.logging import get_logger  # noqa: E402
from daralm.utils.seed import set_seed  # noqa: E402

logger = get_logger(__name__)


def _is_suitable_for_overfit_test(
    text: str, tokenizer: DaraLMTokenizer, max_unk_ratio: float = 0.02, max_pipe_ratio: float = 0.01
) -> bool:
    if not text:
        return False
    if text.count("|") / len(text) > max_pipe_ratio:
        return False
    ids = tokenizer.encode(text)
    if not ids:
        return False
    unk_ratio = sum(1 for i in ids if i == tokenizer.unk_id) / len(ids)
    return unk_ratio <= max_unk_ratio


MIN_LOSS_DROP_RATIO = 0.7
MIN_MATCH_RATE_VS_CHANCE_MULTIPLE = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tiny.yaml"))
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("checkpoints/tokenizer/unigram.model")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/cleaned"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Separate from the real training checkpoints, to avoid mixing this sanity run in",
    )
    parser.add_argument(
        "--n-examples", type=int, default=300, help="Number of documents to overfit"
    )
    parser.add_argument(
        "--max-doc-chars",
        type=int,
        default=2000,
        help=(
            "Only sample from documents under this length. Caps total token volume so the "
            "set stays genuinely overfittable, without literally taking the N *shortest* "
            "documents — on this corpus those cluster almost entirely into one narrow genre "
            "(Cambodian administrative geography stubs, all sharing a near-identical sentence "
            "template), which breaks the memorization check: the model can learn the shared "
            "template perfectly while still greedily guessing a different, equally-plausible "
            "place name than the one specific document being checked. Random sampling below "
            "a length cap keeps the set small while avoiding that genre bias."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--n-memorization-samples", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ModelConfig.from_yaml(args.config)
    config.model_name = f"{config.model_name}-overfit-test"
    config.training = config.training.model_copy(
        update={
            "max_steps": args.max_steps,
            "warmup_steps": args.warmup_steps,
            "eval_interval": max(args.max_steps // 6, 1),
            "save_interval": args.max_steps,  # only save once, at the end
            "log_interval": max(args.max_steps // 15, 1),
        }
    )
    set_seed(config.training.seed)

    tokenizer = DaraLMTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.vocab_size != config.architecture.vocab_size:
        raise ValueError(
            f"Tokenizer vocab_size ({tokenizer.vocab_size}) != config vocab_size "
            f"({config.architecture.vocab_size})"
        )

    all_records = load_jsonl(args.data_dir / "train.jsonl")
    length_filtered = [r for r in all_records if len(r["text"]) <= args.max_doc_chars]
    candidates = [r for r in length_filtered if _is_suitable_for_overfit_test(r["text"], tokenizer)]
    logger.info(
        "%d/%d documents under %d chars pass the clean-prose filter",
        len(candidates),
        len(length_filtered),
        args.max_doc_chars,
    )
    if len(candidates) < args.n_examples:
        raise ValueError(
            f"Only {len(candidates)} suitable documents available, fewer than "
            f"--n-examples={args.n_examples}. Lower --n-examples or raise --max-doc-chars."
        )
    overfit_records = random.Random(config.training.seed).sample(candidates, args.n_examples)
    lengths = [len(r["text"]) for r in overfit_records]
    logger.info(
        "Selected %d documents (%d-%d chars, randomly sampled from %d under %d chars) to overfit",
        len(overfit_records),
        min(lengths),
        max(lengths),
        len(candidates),
        args.max_doc_chars,
    )

    block_size = config.architecture.max_position_embeddings
    overfit_dataset = PackedTokenDataset(overfit_records, tokenizer, block_size=block_size)
    logger.info("Packed into %d block(s) of size %d", len(overfit_dataset), block_size)

    device = get_device()
    logger.info("Using device: %s", get_device_name(device))

    model = DaraLMTransformer(config.architecture, pad_token_id=tokenizer.pad_id)

    trainer = Trainer(
        config=config,
        model=model,
        train_dataset=overfit_dataset,
        val_dataset=overfit_dataset,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        tokenizer_path=args.tokenizer,
    )

    initial_loss, initial_perplexity = compute_perplexity(model, trainer.val_loader, device)
    logger.info(
        "Initial loss=%.4f perplexity=%.1f (before training)", initial_loss, initial_perplexity
    )

    trainer.train()

    final_loss, final_perplexity = compute_perplexity(model, trainer.val_loader, device)
    loss_drop_ratio = 1.0 - (final_loss / initial_loss)
    logger.info(
        "Final loss=%.4f perplexity=%.1f (dropped %.1f%% from initial)",
        final_loss,
        final_perplexity,
        loss_drop_ratio * 100,
    )

    memorization_texts = [r["text"] for r in overfit_records[: args.n_memorization_samples]]
    memorization_results = check_memorization(model, tokenizer, memorization_texts)
    avg_match = average_match_rate(memorization_results)

    print()
    print("=== Memorization check (greedy continuation vs. real continuation) ===")
    for result in memorization_results[:5]:
        print(f"  match_rate={result['token_match_rate']:.2f}  prompt={result['prompt_preview']!r}")
        print(f"    real:      {result['reference_preview']!r}")
        print(f"    generated: {result['generated_preview']!r}")
    print(f"Average token match rate across {len(memorization_results)} documents: {avg_match:.3f}")

    chance_match_rate = 1.0 / config.architecture.vocab_size
    min_match_rate = chance_match_rate * MIN_MATCH_RATE_VS_CHANCE_MULTIPLE
    passed = loss_drop_ratio >= MIN_LOSS_DROP_RATIO and avg_match >= min_match_rate

    print()
    print("=== Phase 5 Sanity Test Verdict ===")
    print(f"Initial loss:         {initial_loss:.4f} (perplexity {initial_perplexity:.1f})")
    print(f"Final loss:           {final_loss:.4f} (perplexity {final_perplexity:.1f})")
    print(f"Loss drop ratio:      {loss_drop_ratio:.1%}  (threshold: >= {MIN_LOSS_DROP_RATIO:.0%})")
    print(
        f"Avg token match rate: {avg_match:.4f}  (threshold: >= {min_match_rate:.4f}, "
        f"i.e. {MIN_MATCH_RATE_VS_CHANCE_MULTIPLE}x the {chance_match_rate:.5f} chance baseline)"
    )
    print()
    if passed:
        print("PASS — the model can overfit a tiny dataset.")
        print("Safe to proceed to Phase 6 (DaraLM-10M).")
    else:
        print("FAIL — the model could not sufficiently overfit.")
        print("Do NOT proceed to larger training — per spec section 19, assume a pipeline bug.")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
