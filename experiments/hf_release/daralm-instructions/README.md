---
license: cc-by-nc-4.0
language:
  - km
  - en
task_categories:
  - text-generation
tags:
  - instruction-tuning
  - khmer
  - english
configs:
  - config_name: default
    data_files:
      - split: train
        path: "train.jsonl"
      - split: validation
        path: "validation.jsonl"
      - split: test
        path: "test.jsonl"
---

# DaraLM Instructions

The 923-example dataset used for the published
[`DaraLM-50M-Instruct`](https://huggingface.co/daraa1607/daralm-50m-instruct)
checkpoint. It contains 784 train, 138 validation, and 1 test record.

| Source | Examples | Language | Terms |
|---|---:|---|---|
| [`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca) | 499 | English | CC BY-NC 4.0 |
| [`saillab/alpaca_khmer_taco`](https://huggingface.co/datasets/saillab/alpaca_khmer_taco) | 399 | Khmer | No explicit source license; treat as non-commercial and verify before reuse |
| DaraLM hand-authored examples | 25 | Khmer | MIT |

Two of 925 fetched records were removed during cleaning. Exact duplicates were
removed before deterministic splitting with seed 42.

## Format

```json
{"instruction": "What is the capital of France?", "response": "Paris.", "language": "en", "source": "alpaca"}
```

Chat markers are not stored in the dataset; the training pipeline applies them
at runtime.

## Limitations

- The dataset is non-commercial because most records are Alpaca-derived.
- The Khmer source was machine translated and was not independently audited for
  fluency or accuracy.
- The dataset is far too small to produce reliable general instruction following
  by itself.
- The single-record test split is not a meaningful standalone benchmark.

## License

Published as CC BY-NC 4.0. The Khmer source has no explicit license declaration,
so users should obtain clarification from its author before redistribution or
commercial use.
