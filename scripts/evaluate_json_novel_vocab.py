#!/usr/bin/env python

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daralm.evaluation.json_validity import (  # noqa: E402
    is_valid_json,
    parse_json_or_none,
    schema_matches,
)
from daralm.inference.generator import generate_chat  # noqa: E402
from daralm.model.config import ModelConfig  # noqa: E402
from daralm.model.transformer import DaraLMTransformer  # noqa: E402
from daralm.tokenizer.tokenizer import DaraLMTokenizer  # noqa: E402
from daralm.training.checkpoint import load_checkpoint  # noqa: E402
from daralm.utils.device import get_device, get_device_name  # noqa: E402
from daralm.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

SCHEMA_KEYS = ("name", "age", "occupation", "city")

REAL_NAMES = [
    "Alexander Hamilton",
    "Marie Curie",
    "Nelson Mandela",
    "Frida Kahlo",
    "Leonardo da Vinci",
    "Rosa Parks",
    "Charles Darwin",
    "Amelia Earhart",
    "Winston Churchill",
    "Ada Lovelace",
]
REAL_OCCUPATIONS = [
    "software developer",
    "graphic designer",
    "civil engineer",
    "data scientist",
    "social worker",
    "flight attendant",
    "marine biologist",
    "urban planner",
    "sound engineer",
    "yoga instructor",
]
REAL_CITIES = [
    "Vancouver",
    "Amsterdam",
    "Melbourne",
    "Barcelona",
    "Montreal",
    "Auckland",
    "Edinburgh",
    "Stockholm",
    "Zurich",
    "Kyoto",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("checkpoints/tokenizer/unigram.model")
    )
    parser.add_argument("--n-examples", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--seed", type=int, default=99)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ModelConfig.from_yaml(args.config)
    tokenizer = DaraLMTokenizer.from_pretrained(args.tokenizer)
    device = get_device()
    logger.info("Using device: %s", get_device_name(device))

    model = DaraLMTransformer(config.architecture, pad_token_id=tokenizer.pad_id)
    load_checkpoint(args.checkpoint, model, tokenizer_path=args.tokenizer)
    model.to(device)
    model.eval()

    rng = random.Random(args.seed)
    n = args.n_examples
    valid_count, schema_count = 0, 0
    age_correct, name_correct, occ_correct, city_correct = 0, 0, 0, 0

    for i in range(n):
        name = rng.choice(REAL_NAMES)
        age = rng.randint(18, 90)
        occupation = rng.choice(REAL_OCCUPATIONS)
        city = rng.choice(REAL_CITIES)
        text = f"{name} is a {age}-year-old {occupation} living in {city}."
        instruction = (
            "Extract the following fields as a JSON object with keys "
            f'"name", "age", "occupation", "city": {text}'
        )

        generated = generate_chat(
            model, tokenizer, instruction, max_new_tokens=args.max_new_tokens, temperature=0.0
        )
        if i < 3:
            logger.info("Example %d\n  input: %s\n  generated: %s", i, text, generated)

        if not is_valid_json(generated):
            continue
        valid_count += 1
        parsed = parse_json_or_none(generated)
        if not schema_matches(parsed, SCHEMA_KEYS):
            continue
        schema_count += 1

        if parsed.get("age") == age:
            age_correct += 1
        if parsed.get("name") == name:
            name_correct += 1
        if parsed.get("occupation") == occupation:
            occ_correct += 1
        if parsed.get("city") == city:
            city_correct += 1

    print()
    print("=== Structured JSON: Real-Word Distribution-Shift Check ===")
    print(f"Examples: {n}")
    print(f"Valid-JSON rate:   {valid_count / n:.2%}")
    print(f"Schema-match rate: {schema_count / n:.2%}")
    print(f"age correct:        {age_correct}/{n} = {age_correct / n:.2%}")
    print(f"name correct:       {name_correct}/{n} = {name_correct / n:.2%}")
    print(f"occupation correct: {occ_correct}/{n} = {occ_correct / n:.2%}")
    print(f"city correct:       {city_correct}/{n} = {city_correct / n:.2%}")


if __name__ == "__main__":
    main()
