# Model Card: DARALM-150M

> Part of the DaraLM project — a from-scratch decoder-only Transformer for Khmer + English, built as an educational/portfolio exercise. **Not a production or commercial-grade language model.** See the repo README for the full project.

## Architecture

Decoder-only Transformer (GPT-style), implemented from scratch:
- Pre-normalization with RMSNorm
- Causal multi-head self-attention with Rotary Position Embeddings (RoPE)
- GELU feed-forward network
- Weight-tied token embedding / LM head

| | |
|---|---|
| Hidden size | 896 |
| Layers | 14 |
| Attention heads | 14 (head_dim=64) |
| Feed-forward size | 3584 |
| Context window | 1024 tokens |
| Vocabulary | 16,000 tokens |
| Parameters | 149,235,072 |

## Languages

Khmer and English, trained on a roughly balanced bilingual corpus.
Training corpus was 49.81% Khmer, 50.19% English by document count.

## Training Data

- 15,939 documents, 113,646,816 characters, 15,021,255 words after cleaning and deduplication
- Split: 14,345 train / 796 val / 798 test documents
- Duplicate rate: 0.26%
- Sources:
  - **wikipedia-km**: wikimedia/wikipedia (20231101.km), license: CC BY-SA 4.0 / GFDL (Wikimedia Foundation), 8,000 documents fetched
  - **wikipedia-en**: wikimedia/wikipedia (20231101.en), license: CC BY-SA 4.0 / GFDL (Wikimedia Foundation), 8,000 documents fetched

No instruction-tuning or preference data — this is a base language model, trained purely on causal next-token prediction over raw Wikipedia text.

## Tokenizer

SentencePiece Unigram, 16,000 vocabulary, trained on the same corpus (Phase 2). Special tokens: `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3`.

Selected over BPE based on measured evaluation: unigram (score 3.474 vs 3.450 — score = avg chars/token compression across domains, penalized by avg unk rate %)

## Training Configuration

| | |
|---|---|
| Optimizer | adamw |
| Learning rate | 0.0002 (cosine decay after warmup) |
| Weight decay | 0.1 |
| Precision | bf16 |
| Batch size (micro / accumulation) | 2 × 1 |
| Max steps (this checkpoint) | 300 |
| Warmup steps | 20 |
| Seed | 42 |

## Training Tokens

~614,400 tokens seen (2 sequences × 1024 tokens/sequence × 300 optimizer steps). The training corpus is ~28,938,878 tokens at this tokenizer's real, measured compression rate (not estimated) — so this checkpoint has seen roughly 0.02x the corpus, well under one full pass.

## Compute Used

- Device: Apple Silicon (MPS) — a consumer laptop, not a training cluster
- Throughput: avg 1509 tokens/sec
- Peak device memory: 2.55 GB
- Batch size was set empirically after direct benchmarking found a severe MPS-backend performance cliff at larger batch sizes for this model size (see README Phase 7) — not a guess.

## Evaluation Results

- Final validation loss: 7.0313, perplexity: 1131.5
- Overfitting check: no overfitting signal — validation tracking training, still improving
- Memorization check (spec section 15): 0.0573 average token-match rate over 30 real training documents — 917x the random-chance baseline, but still low in absolute terms; no concerning verbatim memorization detected at this scale/step count.
- Fixed-prompt generation samples (temperature=0.8, top-p=0.9):
  - **english**: `Cambodia is a country in married. The other laws of the clubs that the southwests. This sends. He in the Sign of the first in agerla, the unable of th...`
  - **khmer**: `កម្ពុជាជាប្រទេសមួយនៅ័បានខាងត្បូង វាធ្វើឿង រាជនោះ របបលោកខ្មែរក្រហមទៅរការបាន សាល និងស្ស្លាប់៥ក្នុងប្រពៃណីខ្បទដឹងខឥការ ទេអ្នកលើោថ្មីរស់នៅ នេះ ថែមទៀត ចំពោ...`
  - **mixed**: `ខ្ញុំចង់រៀន machine learning និងក ១៤ឱ្យបិកkaនៃតើ រ ដែលថៃបំពេញរtiសបុ ចង់ មកមាសខាងជើងហម៉ាបានលែងលុះប្រាសាទ ។ក្នុងរប្រជំងនេះ នដល់គឺក៏ ដាច់ បុត្រឃើញយករកលើហ...`

## Known Limitations

- **Not fluent.** At this step count, generation is grammatically fragmentary in both languages — real words and some real morphology/particles, not coherent sentences or paragraphs. This is expected at this scale, not a bug.
- **Small corpus.** ~15,939 Wikipedia documents is a tiny fraction of what production LLMs train on. Facts, if any appear, should not be trusted.
- **Language drift.** Generation does not reliably stay in the prompt's language for its full length — an English prompt can drift into Khmer partway through.
- **No instruction-following.** This is a base model; it continues text, it does not follow instructions or answer questions reliably.
- **No safety tuning.** No RLHF, no content filtering, no red-teaming has been performed.

## Responsible Use

This model is a **learning artifact**, built to understand the LLM training pipeline end to end. It is not evaluated or intended for factual, medical, legal, financial, or safety-critical use of any kind. Outputs should not be presented to end users as authoritative.

## License

Code: MIT (see repository LICENSE). Training data: Wikipedia text under CC BY-SA 4.0 / GFDL — see `data/raw/MANIFEST.json` for exact sources. Model weights inherit the CC BY-SA 4.0 share-alike terms of the training data.
