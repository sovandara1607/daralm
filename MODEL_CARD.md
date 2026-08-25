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
Training corpus was 49.83% Khmer, 50.17% English by document count.

## Training Data

- 2,990 documents, 37,945,756 characters, 5,409,122 words after cleaning and deduplication
- Split: 2,691 train / 149 val / 150 test documents
- Duplicate rate: 0.07%
- Sources:
  - **wikipedia-km**: wikimedia/wikipedia (20231101.km), license: CC BY-SA 4.0 / GFDL (Wikimedia Foundation), 1,500 documents fetched
  - **wikipedia-en**: wikimedia/wikipedia (20231101.en), license: CC BY-SA 4.0 / GFDL (Wikimedia Foundation), 1,500 documents fetched

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
| Max steps (this checkpoint) | 300 |
| Warmup steps | 20 |
| Seed | 42 |

## Training Tokens

~4,915,200 tokens seen (16 sequences × 1024 tokens/sequence × 300 optimizer steps). The training corpus (Phase 1) is ~8.25M tokens at this tokenizer's compression rate, so this checkpoint has seen well under a handful of full passes over it — not enough for the memorization check below to be a strong test.

## Compute Used

- Device: Apple Silicon (MPS) — a consumer laptop, not a training cluster
- Throughput: avg 6867 tokens/sec
- Peak device memory: 0.53 GB
- Batch size was set empirically after direct benchmarking found a severe MPS-backend performance cliff at larger batch sizes for this model size (see README Phase 7) — not a guess.

## Evaluation Results

- Final validation loss: 6.3519, perplexity: 573.6
- Overfitting check: no overfitting signal — validation tracking training, still improving
- Memorization check (spec section 15): 0.0280 average token-match rate over 30 real training documents — 448x the random-chance baseline, but still low in absolute terms; no concerning verbatim memorization detected at this scale/step count.
- Fixed-prompt generation samples (temperature=0.8, top-p=0.9):
  - **english**: `Cambodia is a country in សរសេរ ព្រះ្រៈ អតីតកាលន ស្ថិតនៅក្នុងបតិបង្កើតរសមខ្លួនស្កាង ពួកនាំណ្នា  ហេតុនេះ លោក សង្គ្រាមឫ ឪពុក ជាការ។ កសាងទាំងឡាយដែលមាន។ ថវ...`
  - **khmer**: `កម្ពុជាជាប្រទេសមួយនៅ ប្រទេសបុរាណ ក្រុងទុកជាឡើងនឹងប្រអ្នកពីសំរាប់ ក្នុងករណីងខេត្ត ប់ធ៍ សេចក្ដី ឬ និង ប៉ា មហាសមុទ្រជានៅ១៣ អាង ត្រីពីរភូមិាថា ពួកពេល ខាងក...`
  - **mixed**: `ខ្ញុំចង់រៀន machine learning និង ជ្រុងហែន ស្វាប៉ូែ ផែនការវត្តតាមអានិងពុំស្ទឹងញ ទទី១ បរិភោគយាយ ឯ អ្នកសមជាអ្នកកិត្តិយសពាក្យ ហើយចុងអង្ទឹក ធម្មសសសក ការមាន...`

## Known Limitations

- **Not fluent.** At this step count, generation is grammatically fragmentary in both languages — real words and some real morphology/particles, not coherent sentences or paragraphs. This is expected at this scale, not a bug.
- **Small corpus.** ~3,000 Wikipedia documents is a tiny fraction of what production LLMs train on. Facts, if any appear, should not be trusted.
- **Language drift.** Generation does not reliably stay in the prompt's language for its full length — an English prompt can drift into Khmer partway through.
- **No instruction-following.** This is a base model; it continues text, it does not follow instructions or answer questions reliably.
- **No safety tuning.** No RLHF, no content filtering, no red-teaming has been performed.

## Responsible Use

This model is a **learning artifact**, built to understand the LLM training pipeline end to end. It is not evaluated or intended for factual, medical, legal, financial, or safety-critical use of any kind. Outputs should not be presented to end users as authoritative.

## License

Code: MIT (see repository LICENSE). Training data: Wikipedia text under CC BY-SA 4.0 / GFDL — see `data/raw/MANIFEST.json` for exact sources. Model weights inherit the CC BY-SA 4.0 share-alike terms of the training data.
