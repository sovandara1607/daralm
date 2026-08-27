# Model Card: DaraLM-150M Base

This card describes the 300-step DaraLM-150M experiment. It is an educational,
severely undertrained model and is not packaged as a production release.

## Model details

| Field | Value |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameters | 149,235,072 |
| Hidden size | 896 |
| Layers | 14 |
| Attention heads | 14 |
| Feed-forward size | 3,584 |
| Context window | 1,024 tokens |
| Vocabulary | 16,000 SentencePiece Unigram tokens |
| Languages | Khmer and English |

The implementation uses pre-normalization with RMSNorm, causal self-attention,
RoPE, GELU feed-forward layers, and tied input/output embeddings.

## Training

- 15,939 cleaned and deduplicated Wikipedia documents: 14,345 train, 796
  validation, and 798 test.
- Approximately 28.94 million corpus tokens.
- Roughly balanced Khmer and English document counts.
- Wikipedia source license: CC BY-SA 4.0 / GFDL.

| Training field | Value |
|---|---|
| Objective | Causal next-token prediction |
| Optimizer | AdamW |
| Learning rate | 2e-4 with warmup and cosine decay |
| Effective batch | 2 sequences |
| Precision | bf16 |
| Steps | 300 |
| Tokens processed | 614,400 |
| Device | Apple Silicon MPS |

The model saw only about 2% of one corpus pass. Its larger parameter count did
not compensate for the much smaller training budget.

## Evaluation

- Final validation loss: 7.0313.
- Final validation perplexity: 1,131.5.
- Validation was still improving at the final step.
- At the same step count, DaraLM-50M reached perplexity 573.6 and produced more
  coherent samples.

The result means the 150M run was undertrained, not that larger models are
intrinsically worse.

## Limitations

- Generation is fragmentary and factually unreliable.
- It performs worse than the more thoroughly trained 50M checkpoint.
- Khmer/English language drift occurs.
- It does not follow instructions.
- No safety tuning, preference tuning, content filtering, or red-teaming was
  performed.

Do not use this model for factual, medical, legal, financial, or safety-critical
decisions.

## License

Code is MIT licensed. Model weights inherit applicable CC BY-SA 4.0 / GFDL
terms from the Wikipedia training data.
