# Model Card: DaraLM-50M-Instruct

This card describes the 300-step checkpoint packaged in
`experiments/hf_release/daralm-50m-instruct/`. It is a supervised fine-tune of
DaraLM-50M Base and is not suitable for production use.

A later 900-step experiment used 2,313 training examples and reached validation
perplexity 79.98, but manual generation did not improve meaningfully. That later
experiment is not the packaged checkpoint described here.

## Model details

| Field | Value |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameters | 33,366,528 |
| Hidden size | 512 |
| Layers | 8 |
| Attention heads | 8 |
| Context window | 1,024 tokens |
| Vocabulary | 16,000 SentencePiece Unigram tokens |
| Base checkpoint | DaraLM-50M at step 1,500 |

Only model weights were inherited from Base. SFT used a fresh optimizer and
scheduler.

## Chat format

```text
<bos><user>
{instruction}
</user>
<assistant>
{response}
</assistant><eos>
```

The user and assistant markers are ordinary text, not tokenizer special tokens.
Because the Wikipedia-trained tokenizer has poor `<`/`>` coverage, decoded
markers can contain the unknown-character glyph `⁇`.

## Training data

The packaged checkpoint used 923 instruction/response pairs: 784 train, 138
validation, and 1 test.

| Source | Examples | Language | Terms |
|---|---:|---|---|
| `tatsu-lab/alpaca` | 499 | English | CC BY-NC 4.0 |
| `saillab/alpaca_khmer_taco` | 399 | Khmer | Treated as Alpaca-derived CC BY-NC 4.0 |
| Hand-authored DaraLM examples | 25 | Khmer | MIT |

Two of 925 fetched records were removed during cleaning. The Khmer translation quality
has not been independently audited.

Examples were padded individually to 768 tokens. Only response tokens
contributed to loss; prompt and padding positions were masked.

## Training

| Training field | Value |
|---|---|
| Objective | Response-only supervised fine-tuning |
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Effective batch | 16 sequences |
| Precision | bf16 |
| Steps | 300 |
| Tokens processed | 3,686,400 |

Three SFT attempts informed this checkpoint:

1. A small dataset on the 300-step Base reached perplexity 194.4.
2. More instruction data on the same undertrained Base worsened it to 290.4.
3. Continuing Base pretraining to step 1,500 before SFT improved it to 110.7.

The main lesson was that additional SFT data could not replace adequate base
pretraining.

## Evaluation

| Step | Validation loss | Perplexity |
|---:|---:|---:|
| 50 | 4.913 | 136.1 |
| 150 | 4.753 | 115.9 |
| 300 | 4.707 | 110.7 |

English responses became longer and more grammatical, but remained repetitive
and factually wrong. Khmer responses often stayed short or collapsed toward the
assistant closing marker. Lower perplexity should not be interpreted as factual
accuracy or reliable instruction following.

## Limitations

- Fluent-looking answers are frequently fabricated.
- Khmer output is substantially weaker than English output.
- Chat markers can render with `⁇` because of tokenizer coverage.
- The dataset is small and partly machine translated.
- No preference tuning, safety tuning, content filtering, or red-teaming was
  performed.

Do not use this model for factual, medical, legal, financial, or safety-critical
decisions.

## License

Code is MIT licensed. The weights combine Wikipedia-derived CC BY-SA 4.0 / GFDL
terms with Alpaca-derived CC BY-NC 4.0 terms and must not be used commercially.
