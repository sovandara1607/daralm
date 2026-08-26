# Model Card: DARALM-50M

> Part of the DaraLM project — a from-scratch decoder-only Transformer for Khmer + English, built as an educational/portfolio exercise. **Not a production or commercial-grade language model.** See the repo README for the full project.

## Architecture

Decoder-only Transformer (GPT-style), implemented from scratch:
- Pre-normalization with RMSNorm
- Causal multi-head self-attention with Rotary Position Embeddings (RoPE)
- GELU feed-forward network
- Weight-tied token embedding / LM head

| | |
|---|---|
| Hidden size | 512 |
| Layers | 8 |
| Attention heads | 8 (head_dim=64) |
| Feed-forward size | 2048 |
| Context window | 1024 tokens |
| Vocabulary | 16,000 tokens |
| Parameters | 33,366,528 |

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
| Learning rate | 0.0003 (cosine decay after warmup) |
| Weight decay | 0.1 |
| Precision | bf16 |
| Batch size (micro / accumulation) | 4 × 4 |
| Max steps (this checkpoint) | 1500 |
| Warmup steps | 20 |
| Seed | 42 |

## Training Tokens

~24,576,000 tokens seen (16 sequences × 1024 tokens/sequence × 1500 optimizer steps). The training corpus is ~28,938,878 tokens at this tokenizer's real, measured compression rate (not estimated) — so this checkpoint has seen roughly 0.85x the corpus, well under one full pass.

## Compute Used

- Device: Apple Silicon (MPS) — a consumer laptop, not a training cluster
- Throughput: avg 6190 tokens/sec
- Peak device memory: 0.51 GB
- Batch size was set empirically after direct benchmarking found a severe MPS-backend performance cliff at larger batch sizes for this model size (see README Phase 7) — not a guess.

## Evaluation Results

- Final validation loss: 5.2041, perplexity: 182.0
- Overfitting check: no overfitting signal — validation tracking training, still improving
- Memorization check (spec section 15): 0.0327 average token-match rate over 30 real training documents — 523x the random-chance baseline, but still low in absolute terms; no concerning verbatim memorization detected at this scale/step count.
- Fixed-prompt generation samples (temperature=0.8, top-p=0.9):
  - **english**: `Cambodia is a country in the western Sea of the Middle East. The name of the Assyrian period is: The Name of the India of the Northern Ireland is the ...`
  - **khmer**: `កម្ពុជាជាប្រទេសមួយនៅសិង្ហបុរី។ ព្រះនាមស.ក. ១៣) ៦៣។ ការបែងចែកឡើងវិញនៅឆ្នាំ ១៩២៥។ ៣៨ នាក់ ខែមករា ប្រទេសកម្ពុជាត្រូវត្រូវធ្វើឡើងក្នុងកម្ពុជា ដែលកងទ័ពប្រទ...`
  - **mixed**: `ខ្ញុំចង់រៀន machine learning និងមហាវលល។ កិច្ចកិច្ចមនុស្សនេះហើយ គឺជាសកលស្តីពីការតំណាង។ ក្រុមប្រឹក្សាឃុំ សង្កាត់ក្លាក្លានោះ អ្នកសង្កេតការណ៍ត្រូវសម្លាប់ ...`

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
