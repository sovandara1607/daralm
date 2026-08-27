# Model Card: DaraLM-50M Base

This card describes the 1,500-step DaraLM-50M base checkpoint packaged in
`experiments/hf_release/daralm-50m/`. It is an educational model, not a
production language model.

## Model details

| Field | Value |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameters | 33,366,528 |
| Hidden size | 512 |
| Layers | 8 |
| Attention heads | 8 |
| Feed-forward size | 2,048 |
| Context window | 1,024 tokens |
| Vocabulary | 16,000 SentencePiece Unigram tokens |
| Languages | Khmer and English |

The implementation uses pre-normalization with RMSNorm, causal self-attention,
RoPE, GELU feed-forward layers, and tied input/output embeddings.

## Training

This checkpoint trained on the earlier Wikipedia snapshot used by the original
50M run:

- 2,990 cleaned and deduplicated documents: 2,691 train, 149 validation, and
  150 test.
- 1,500 Khmer and 1,500 English articles were fetched before filtering.
- Approximately 8.25 million corpus tokens.
- Wikipedia source license: CC BY-SA 4.0 / GFDL.

The repository’s current cleaned corpus is larger and now includes Common Crawl
records. This checkpoint has not learned that newer data.

| Training field | Value |
|---|---|
| Objective | Causal next-token prediction |
| Optimizer | AdamW |
| Learning rate | 3e-4 with warmup and cosine decay |
| Effective batch | 16 sequences |
| Precision | bf16 |
| Steps | 1,500 |
| Tokens processed | 24,576,000 |
| Device | Apple Silicon MPS |

The run processed roughly three corpus-equivalents. The packaged
`pytorch_model.pt` contains inference weights, not optimizer or scheduler state.

## Evaluation

- Final validation loss: 5.2041.
- Final validation perplexity: 182.0.
- Validation was still improving at the final step.
- Generation became more sentence-like than the 300-step checkpoint, but
  remained repetitive and factually unreliable.

Example English continuation:

> Cambodia is a country in the western Sea of the Middle East...

This sample demonstrates surface fluency, not factual knowledge.

## Limitations

- It is a base completion model, not an instruction-following model.
- Outputs can sound plausible while being wrong.
- Khmer quality is weaker and language drift can occur.
- The corpus is extremely small compared with production LLM training data.
- No safety tuning, preference tuning, content filtering, or red-teaming was
  performed.

Do not use this model for factual, medical, legal, financial, or safety-critical
decisions.

## License

Code is MIT licensed. The checkpoint inherits applicable CC BY-SA 4.0 / GFDL
terms from its Wikipedia training data. See the source manifests for provenance.
