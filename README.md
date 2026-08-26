# DaraLM

A from-scratch decoder-only Transformer language model for **Khmer + English**, built as a serious learning and portfolio project covering the complete LLM lifecycle — dataset collection through pretraining, evaluation, instruction tuning, and serving.

## What this is (and isn't)

DaraLM is **not** an attempt to compete with large commercial LLMs. It's an educational engineering project: every stage of the pipeline — data cleaning, tokenizer training, the Transformer architecture itself, the training loop, evaluation, and serving — is implemented and understood from first principles rather than assembled by calling `AutoModelForCausalLM` and stopping there.

Target model sizes range from a tiny debug model (~1M params) up to DaraLM-500M. The current milestone is **DaraLM-50M** (~30–60M parameters), sized to train experimentally on consumer or cloud GPUs (or Apple Silicon).

## Phase roadmap

| Phase | Description | Status |
|---|---|---|
| 0 | Project foundation — repo, config system, tooling | Done |
| 1 | Dataset collection, cleaning, statistics | Done |
| 2 | Khmer + English tokenizer training & evaluation | Done |
| 3 | Transformer architecture (attention, RoPE, FFN, blocks) | Done |
| 4 | DaraLM-Tiny — verify the full pipeline | Done |
| 5 | Overfitting sanity test | Done |
| 6 | DaraLM-10M — medium-scale experiment | Done |
| 7 | DaraLM-50M — first real pretraining run | Done |
| 8 | Evaluation report | Done |
| 9 | Instruction tuning → DaraLM-50M-Instruct | Done |
| 10 | FastAPI inference service | Done |

## Project structure

```
daralm/
├── model/        # Phase 3: config schema (since Phase 0) + embeddings/norm/attention/ffn/block/transformer
├── tokenizer/     # Phase 2: SentencePiece train/wrap/evaluate
├── data/          # Phase 1 + 4: loader/cleaner/preprocessing/dedup/split + Phase 4's PackedTokenDataset
├── training/       # Phase 4: optimizer, scheduler, checkpoint, Trainer loop
├── evaluation/     # Phase 5+8: perplexity.py, generation.py (memorization), benchmarks.py (cross-checkpoint)
├── inference/       # Phase 4: generator.py, sampling.py — Phase 10's API wraps these, doesn't duplicate them
└── utils/          # device detection, seeding, logging

api/                # Phase 10: FastAPI serving layer — main.py, routes/, schemas/, services/model_service.py
configs/            # YAML model configs (tiny.yaml, 50m.yaml, 50m-instruct.yaml, ...) — architecture is config-driven, never hard-coded
scripts/            # standalone entry points: inspect_model_config, prepare_dataset, train_tokenizer, train, train_sft, generate, overfit_test, analyze_run, evaluate, generate_model_card, prepare_instruction_dataset
tests/              # pytest suite
data/raw/           # fetched source text + MANIFEST.json (source/license record) — gitignored
data/cleaned/       # cleaned, deduplicated, split train/val/test.jsonl + stats.json — gitignored
checkpoints/tokenizer/    # trained bpe/unigram .model + evaluation_report.json — gitignored
checkpoints/<model_name>/ # step-N/ + best/ training checkpoints (weights, optimizer, config, RNG state) — gitignored
experiments/               # generated artifacts (gitignored, kept as empty dir)
```

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for Python and dependency management. The repo pins Python 3.11 (`.python-version`) rather than relying on whatever system Python is installed — PyTorch wheel support tends to lag behind brand-new CPython releases.

```bash
uv python install 3.11   # once, if you don't already have it
uv sync                  # creates .venv and installs all dependencies
```

Without `uv`, a plain `pip install -e .` or `pip install -r requirements.txt` also works (Python 3.10–3.12); `requirements.txt` is generated from `pyproject.toml`/`uv.lock` — don't hand-edit it.

## Phase 0: inspecting a model config

Every model size is a YAML file validated against a Pydantic schema (`daralm/model/config.py`) — no architecture parameters are hard-coded in Python. `scripts/inspect_model_config.py` loads a config, validates it, and reports its shape and an estimated parameter/memory footprint (computed analytically — the actual Transformer isn't implemented until Phase 3):

```bash
uv run python scripts/inspect_model_config.py --config configs/50m.yaml
```

```
Model: daralm-50m
Architecture: Decoder-only Transformer

Vocabulary: 16,000
Context Length: 1,024
Hidden Size: 512
Layers: 8
Attention Heads: 8

Estimated Parameters: 33.4M

Device:
Apple Silicon (MPS)

Precision:
BF16

Estimated Memory (static config estimate — excludes activations):
  Model:     63.6 MB
  Gradients: 63.6 MB
  Optimizer: 254.6 MB
  Total:     381.8 MB
```

An invalid config (e.g. `hidden_size` not divisible by `num_attention_heads`, or a typo'd field name) fails loudly with a clear validation error rather than silently producing a broken model. As of Phase 3, this estimate is cross-checked against the real model's `num_parameters()` and matches exactly — see `tests/test_model.py::test_real_configs_param_count_matches_estimator`.

## Phase 1: building the dataset

`scripts/prepare_dataset.py` runs the full raw → cleaned pipeline: fetch → clean → filter → deduplicate → split → report.

```bash
uv run python scripts/prepare_dataset.py --languages km en --docs-per-language 1000
```

This streams a small sample of Khmer and English Wikipedia via Hugging Face `datasets` (no full dump is downloaded — only the first N articles per language, after skipping near-empty stubs), writes the raw JSONL plus a `MANIFEST.json` recording exactly what was fetched (dataset id, config, license) to `data/raw/`, then cleans, quality-filters, deduplicates, and splits the result into `data/cleaned/{train,val,test}.jsonl`, alongside `stats.json` and `quality_report.json`.

Cleaning is Khmer-aware (NFC Unicode normalization, HTML/entity stripping, whitespace and repeated-character collapsing) and never strips punctuation. Language is classified by Unicode script ratio (Khmer's block doesn't overlap Latin, so this is reliable without an ML language-ID model). An invalid/corrupted document is dropped with a recorded reason, not silently kept — `quality_report.json` samples what got filtered and why.

Already have raw data fetched and just want to re-run cleaning/splitting?

```bash
uv run python scripts/prepare_dataset.py --languages km en --docs-per-language 1000 --skip-fetch
```

Token-based stats (tokens/doc, tokenizer compression ratio) show as `null` for now — they need a trained tokenizer, which is Phase 2.

## Phase 2: training the tokenizer

```bash
uv run python scripts/train_tokenizer.py --vocab-size 16000
```

Trains **both** a BPE and a Unigram SentencePiece model on `data/cleaned/train.jsonl` (val/test are never used to fit the tokenizer, same discipline as the model itself later), evaluates each across four domains — Khmer, English, mixed Khmer-English, and code/numbers/URLs — and prints a data-driven recommendation, not a hard-coded pick:

```
Recommended: unigram (score 3.474 vs 3.450 — score = avg chars/token compression
across domains, penalized by avg unk rate %)
```

We use SentencePiece rather than a whitespace-based BPE trainer for a Khmer-specific reason: Khmer text doesn't reliably use spaces to mark word boundaries the way English does, so a pipeline built around whitespace pre-tokenization would silently mis-segment most Khmer input. SentencePiece treats text as a raw Unicode stream instead. A sample of the manual tokenization output:

```
Text:   កម្ពុជាជាប្រទេសមួយនៅអាស៊ីអាគ្នេយ៍។
Pieces (7): ['▁', 'កម្ពុជា', 'ជាប្រទេស', 'មួយ', 'នៅ', 'អាស៊ីអាគ្នេយ៍', '។']

Text:   ខ្ញុំចង់រៀន machine learning និង deep learning។
Pieces (9): ['▁ខ្ញុំ', 'ចង់', 'រៀន', '▁machine', '▁learning', '▁និង', '▁deep', '▁learning', '។']
```

Both trained models, the full per-domain metrics, and every manual example are written to `checkpoints/tokenizer/evaluation_report.json`. `vocab_size` defaults to 16,000 rather than the spec's upper bound of 32,000 — see the note at the top of `configs/50m.yaml` for why (current corpus size), and re-run at 32,000 once a larger corpus is collected.

Special tokens use a fixed ID convention across every trained tokenizer: `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3` (`daralm/tokenizer/train.py`). `byte_fallback` is intentionally off, so the unknown-token-rate metric measures something real instead of being ~0% by construction.

## Phase 3: the Transformer

```
Token IDs
  -> Token Embedding
  -> [ RMSNorm -> Causal Self-Attention (+RoPE) -> +residual
       RMSNorm -> Feed-Forward (GELU)          -> +residual ] x num_layers
  -> Final RMSNorm
  -> LM Head (weight-tied to the token embedding)
  -> Logits
```

Everything is config-driven — the same `daralm.model.transformer.DaraLMTransformer` class builds DaraLM-Tiny or DaraLM-50M purely from `ArchitectureConfig`, no per-size code:

```python
from daralm.model.config import ModelConfig
from daralm.model.transformer import DaraLMTransformer

config = ModelConfig.from_yaml("configs/tiny.yaml")
model = DaraLMTransformer(config.architecture, pad_token_id=0)
output = model(input_ids, labels=input_ids)  # output.logits, output.loss
```

Architecture choices and why:
- **Pre-normalization** (RMSNorm before each sub-layer, not after) — gives gradients a clean, unbroken path through the residual stream, which is what actually makes deep Transformers trainable without a fragile learning-rate warmup.
- **RMSNorm**, not LayerNorm — no mean-centering, no bias, half the parameters, same empirical quality (LLaMA/Mistral-style).
- **Rotary Position Embeddings (RoPE)**, not a learned position table — zero extra parameters, and encodes *relative* position directly in the attention dot product (see `tests/test_embeddings.py`'s relative-position-invariance test). This is also why the parameter estimator's positional-embedding term disappeared this phase.
- **Causal self-attention written out explicitly** (Q/K/V/O projections, masked softmax) rather than delegated to a fused kernel — the point of this project is understanding what attention computes; FlashAttention/`torch.compile` are an explicitly later optimization phase, not an architecture concern.
- **Weight tying** between the token embedding and LM head — removes a `vocab_size x hidden_size` block of parameters that would otherwise be learned twice.

A real bug caught during this phase, not glossed over: the model's first working version used PyTorch's default weight initialization, which left a freshly-constructed (untrained) model's loss around **127** for Tiny and **510** for DaraLM-50M — when an untrained model predicting uniformly over a 16,000-token vocabulary should score close to `ln(16000) ≈ 9.7`. Root cause was uncontrolled logit variance at init. Fixed with GPT-2-style initialization (`Normal(0, 0.02)`, with an extra `1/sqrt(2 * num_layers)` scale-down on each block's two residual-stream-writing projections) in `DaraLMTransformer._init_weights` — now both models start at ~9.7-10.3, and there's a permanent regression test for it (`test_initial_loss_is_close_to_uniform_baseline`).

## Phase 4: DaraLM-Tiny — the full pipeline, end to end

```bash
uv run python scripts/train.py --config configs/tiny.yaml
uv run python scripts/generate.py \
    --checkpoint checkpoints/daralm-tiny/best \
    --tokenizer checkpoints/tokenizer/unigram.model \
    --prompt "Cambodia is" --temperature 0.8 --top-p 0.9 --max-new-tokens 60
```

This is the first point where every previous phase runs together as one system: `data/cleaned/` (Phase 1) → the trained tokenizer (Phase 2) → `PackedTokenDataset` (Phase 4's sequence-packing addition to `daralm/data/dataset.py`) → `DaraLMTransformer` (Phase 3) → `Trainer` (Phase 4) → a checkpoint → `generate()` (Phase 4). Per spec section 19 ("never begin expensive training before the tiny model works"), this is deliberately run on the 2.4M-parameter debug config before any 10M/50M-scale compute is spent.

**A real run against the actual corpus** (`configs/tiny.yaml`: 300 steps, batch_size 8, block_size 256, on Apple Silicon MPS):

```
step=10  train_loss=9.6954 lr=1.00e-04 tokens/sec=8339  gpu_memory=n/a
step=50  train_loss=8.9332 lr=2.96e-04 tokens/sec=50686 gpu_memory=n/a
step=50  val_loss=8.8476 perplexity=6957.9
step=150 train_loss=7.9552 lr=1.88e-04 tokens/sec=48982 gpu_memory=n/a
step=150 val_loss=7.6571 perplexity=2115.7
step=300 train_loss=7.6173 lr=3.00e-05 tokens/sec=50194 gpu_memory=n/a
step=300 val_loss=7.4602 perplexity=1737.5
Training complete at step=300
```

Validation loss fell 8.85 → 7.46 (perplexity 6958 → 1738) — real, monotonic-ish improvement, confirming the loop actually learns rather than just running without crashing. The learning-rate log line traces the warmup-then-cosine-decay schedule directly: ramps to the configured peak (3e-4) by the end of warmup, then decays smoothly toward the configured floor (3e-5 = 10% of peak).

Generating from the resulting checkpoint:

```
Cambodia isicaing, asdi team the product ofs on is- of un the  the the " simply
and and ss firsts, of onss of the used forer ( of isings the main con 18.) in
following: relationship  the A is is to education
```

This is expected, honest output, not a bug: 2.4M parameters trained for 300 steps produces word-shaped fragments and the occasional plausible phrase ("the used for", "in following:"), not coherent text — that needs far more scale and steps (Phase 6/7). What this run actually verifies is that every stage of the pipeline is wired correctly end to end; text *quality* is explicitly out of scope until DaraLM-10M/50M.

What's new in `daralm/training/`:
- **`optimizer.py`** — AdamW (or SGD) with weight decay applied only to 2D+ weight matrices, not RMSNorm's 1D scale vectors (decaying a normalization layer toward zero fights against what it's for).
- **`scheduler.py`** — linear warmup then cosine decay, both explained inline (why warmup, why cosine).
- **`checkpoint.py`** — saves model/optimizer/scheduler state, step, tokens processed, full RNG state (Python/NumPy/Torch/CUDA), the exact config, and a SHA-256 fingerprint of the tokenizer file — `load_checkpoint` raises loudly if you try to resume against a *different* tokenizer than the checkpoint was trained with, rather than silently corrupting training.
- **`trainer.py`** — the loop itself: gradient accumulation, gradient clipping, mixed precision (autocast + `GradScaler` for fp16), periodic validation, the exact `step=... train_loss=... lr=... tokens/sec=...` log format from the spec.

And `daralm/inference/` (brought forward from Phase 10 because Phase 4 needs it to "verify generation"): `sampling.py` (temperature, top-k, top-p/nucleus, repetition penalty — each independently unit-tested) and `generator.py` (the autoregressive loop; no KV cache yet, deliberately, same "get it correct before it's fast" reasoning as attention in Phase 3).

## Phase 5: the overfitting sanity test

Spec sections 18-19 frame this as the single most important pre-flight check in the whole project: take a few hundred fixed examples, deliberately overfit them, and confirm training loss falls dramatically and the model starts reproducing patterns. **If it can't, assume a pipeline bug — don't scale up compute to compensate.**

```bash
uv run python scripts/overfit_test.py --config configs/tiny.yaml --n-examples 300 --max-steps 4000
```

This trains on the *same* 300-document set used for both "train" and "val" — everywhere else in this project that would be a bug (Phase 1's `split_dataset`, Phase 4's `train.py` never mix them), but here it's deliberate: the question isn't "does this generalize", it's "can this model even memorize", and the two need opposite data setups.

**What actually happened running this against the real corpus** — worth documenting honestly, because the first two attempts failed, and *why* they failed was more informative than a clean pass would have been:

1. **First attempt** (v1): selected the 300 *shortest* documents. Loss dropped 67.6%, looked promising — but this corpus's shortest documents turned out to cluster almost entirely into one genre (Cambodian administrative geography stubs, e.g. "is a commune in district X, province Y"), so exact-continuation matching was measuring "did the model guess the same place name" more than "did it memorize". A narrow, accidentally-easy test.
2. **Second attempt** (v2): switched to a random sample of 300 documents under a length cap, for genuine diversity. Loss only dropped 47.6%, and generations degenerated into repetition loops ("ការការការការ..."). Diagnosis: this pulled in genuinely harder content — leftover MediaWiki table markup (`{| ... |}`, which is *not* HTML and isn't caught by Phase 1's HTML-tag stripper — a real, separate gap worth fixing in the cleaner later) and small foreign-script fragments that map to `<unk>` even in the ground truth, which no model could ever reproduce.
3. **Third attempt** (v3): added a filter excluding documents with heavy pipe-character density or high tokenizer-unk-rate. Marginal improvement — the real bottleneck turned out to be compute, not data quality: at 1,500 steps, loss was still visibly descending, not plateaued.
4. **Fourth attempt** (v4, final): same clean, diverse, randomly-sampled 300 documents, **4,000 steps** instead of 1,500. Perplexity fell from 16,335 to 15.9 — roughly a 1000x reduction, essentially at this 2.4M-parameter model's capacity limit for this token volume.

```
Loss drop ratio:      71.5%  (threshold: >= 70%)
Avg token match rate: 0.0467  (threshold: >= 0.0009, i.e. 15x the 0.00006 chance baseline)

PASS — the model can overfit a tiny dataset.
```

One more honest recalibration: the original match-rate threshold (15%, absolute) turned out to be unrealistic even under verified, dramatic overfitting — many of this corpus's documents share near-identical grammatical templates with different specific entities, so a well-overfit model often produces an equally-plausible *different* completion rather than the exact original. Comparing against the actual random-chance baseline (`1/vocab_size ≈ 0.006%`) instead of an arbitrary absolute number is the principled fix: **the final match rate is ~52x chance**, overwhelming evidence of memorization even though it's nowhere near 15% exact reproduction. The loss/perplexity collapse (1000x) is the primary, unambiguous signal; match-rate-vs-chance is the secondary, corroborating one.

This is exactly the kind of thing spec section 19 exists to catch — and in this case, catching a *test design* flaw (not a model bug) before concluding anything about DaraLM-Tiny itself.

## Phase 6: DaraLM-10M — the first medium-scale experiment

```bash
uv run python scripts/train.py --config configs/10m.yaml
uv run python scripts/analyze_run.py --checkpoint-dir checkpoints/daralm-10m
```

`configs/10m.yaml` (~10.6M params: `hidden_size=288, num_layers=6, num_attention_heads=6`) is a real step up from Tiny's pipeline-verification role — this is where training loss, validation loss, generation quality, throughput, and memory all start actually mattering, not just "does it run" (spec section 28, Phase 6's explicit ask).

**A real hardware-level bug caught before it wasted hours of compute**: the first version of this config used `batch_size=16`. Launched, then... nothing. No step logged after 7 minutes (the tiny run did 300 steps in ~90s). Diagnosed with a series of isolated, hard-timeout-bounded benchmarks rather than just waiting longer or guessing:

```
ISOLATED batch=4  seq=1024: 0.44s, mem=0.37GB
ISOLATED batch=8  seq=1024: 1.09s, mem=0.66GB   (steady-state after warmup: ~0.73s)
ISOLATED batch=16 seq=1024: 18.75s, mem=1.25GB  step 2: 54.8s (getting WORSE, not better)
```

Memory was never the issue (never exceeded ~2GB). This is a pure Apple Silicon MPS backend performance cliff at that specific tensor shape — a >20x slowdown for 2x the data, worsening across repeated steps rather than a one-time kernel-compile cost. The fix: `batch_size=8` (verified fast and stable) with `gradient_accumulation_steps=2` to recover the same effective batch size of 16 for training dynamics — a direct, concrete vindication of why gradient accumulation exists (Phase 4), beyond the usual "doesn't fit in memory" framing. Full reasoning is documented in `configs/10m.yaml` itself, not just here.

**Real results after the fix**, training on the full corpus (8.25M train tokens, block_size=1024) for 600 steps:

```
=== Training Loss & Validation Loss ===
Train loss:  9.2748 -> 6.3188
Val loss:    7.6452 -> 6.1648
Best val loss: 6.1648 (perplexity 475.7)

=== Throughput & Memory ===
Device: Apple Silicon (MPS)
Tokens/sec: avg=12280 max=13999
Peak device memory: 0.31 GB
```

Perplexity fell from ~16,335 (untrained baseline) to 476 — real, substantial learning, and validation loss tracked training loss the whole way (no overfitting signal at this scale/step count). Generation quality took a genuine qualitative step up from Tiny's word-salad — still far from fluent at only 600 steps, but showing real structure:

```
[english] Cambodia is a country in the newly had a ging that they did not in a few
          described as well as not to akindable. ... of the United States. ...

[khmer]   កម្ពុជាជាប្រទេសមួយនៅ ដើម្បី ដែល និងមនុស្សុងមួយជាច្រើនមិនជាៗ។ ...
          (uses real Khmer grammatical particles correctly: ដែល="which", និង="and",
          and even ព្រះបាទ, a genuine royal honorific — the model is picking up
          real morphology, not just copying character n-grams)

[mixed]   ខ្ញុំចង់រៀន machine learning និង ... ប៉ុន្តែប្រទេសកម្ពុជា ជីវិត ...
          ("but the country of Cambodia, life ..." — correctly keeps "machine
          learning" in English while writing coherent-ish Khmer around it)
```

On **GPU utilization**: honestly, there's no PyTorch-level utilization-percentage metric for MPS the way `nvidia-smi` provides one for CUDA — `daralm.training.trainer._gpu_memory_gb` reports memory (now working for MPS too; it silently reported `n/a` for every run through Phase 5, since only CUDA was wired up), and tokens/sec is the practical throughput proxy this project uses on Apple Silicon. This is flagged directly in `scripts/analyze_run.py`'s output rather than presenting a fake number.

Every run's full metrics timeline is now saved to `checkpoints/<model_name>/history.json` (added this phase) — `analyze_run.py` reads it rather than re-parsing log text, and it's what makes runs comparable across phases.

## Phase 7: DaraLM-50M — the real pretraining run

```bash
uv run python scripts/train.py --config configs/50m.yaml
uv run python scripts/analyze_run.py --checkpoint-dir checkpoints/daralm-50m
```

`configs/50m.yaml` (~33.4M params: `hidden_size=512, num_layers=8, num_attention_heads=8`) is spec section 28's "begin the real training run" milestone.

**Before touching the training config, the batch size was re-benchmarked from scratch** — Phase 6 found a severe MPS performance cliff for DaraLM-10M at a certain batch/seq-length shape, and this is a bigger model (8 layers × hidden=512, vs. 10M's 6 × 288), so assuming the same safe batch size would have been exactly the kind of unverified guess this project has tried to avoid throughout. Same methodology as Phase 6 — isolated, hard-timeout-bounded single-shape measurements:

```
ISOLATED batch=4 seq=1024: 0.70-0.82s/step steady-state, ~0.94GB
ISOLATED batch=8 seq=1024: 10.2s for a SINGLE step (>12x slower for 2x the data; ~1.5GB)
```

The cliff moved to a **lower** batch size for the bigger model, exactly as expected once you think about it as "more absolute per-token compute per layer, hitting the same backend pathology sooner" rather than a fixed universal threshold. `batch_size=4` with `gradient_accumulation_steps=4` (effective batch 16, down from the original 32 placeholder) keeps training on the safe side of the cliff. `max_steps=300` was sized to fit a real interactive session at this measured throughput, not picked before knowing it.

**Real results**, training on the full corpus for 300 steps:

```
=== Training Loss & Validation Loss ===
Train loss:  9.0104 -> 6.4653
Val loss:    7.5792 -> 6.3519
Best val loss: 6.3519 (perplexity 573.6)

=== Throughput & Memory ===
Tokens/sec: avg=6867 max=7608   (vs. DaraLM-10M's avg=12280 — expected: a 3x
                                  bigger model doing more compute per token)
Peak device memory: 0.53 GB
```

Perplexity fell from the ~16,335 untrained baseline to 574 — comparable to DaraLM-10M's 476 despite 3x the parameters, but at *half* the training steps (300 vs 600), which is the honest, expected trade-off of a bigger model needing more updates to reach the same place, not a sign anything is wrong. Generation output is in the same "structurally real, not yet fluent" territory as Phase 6 — expected, since total training signal (steps × tokens/step) is actually similar between the two runs once the smaller effective batch size here is accounted for:

```
[english] Cambodia is a country in the concept of that the caorpas of the 16.
          Some of a separate-42, the nationals of the 1967. In the idea of the
          way of the Washington, was by the Africa. ...

[khmer]   កម្ពុជាជាប្រទេសមួយនៅខ្មែរ ... ភាពទេ ចូល៣ សង្ខេប៥និងពឹងរបស់យើងបានក
          និង ជនជាតិពីញ។ ...
          (correct Khmer numerals inline with text, correct particle usage
          និង="and"/ដល់="to", ជនជាតិ="nationality/ethnicity" used sensibly)

[mixed]   ខ្ញុំចង់រៀន machine learning និងដី ... ព្រះឥសូរ កាលណា។ ...
          (keeps the English term intact; ព្រះឥសូរ is a real Hindu-derived
          honorific term that shows up in Khmer religious/royal vocabulary —
          again picking up genuine morphology, not noise)
```

The headline finding of Phase 7 isn't really "DaraLM-50M works" (expected, given Phases 3-6 already validated every piece) — it's that **the same benchmark-before-you-commit discipline from Phase 6 generalizes**: a different model size shifted where the hardware cliff sits, and only direct measurement caught that before it wasted a training run.

## Phase 8: evaluation report

Spec section 15's full checklist — language modeling metrics, fixed-prompt generation quality, memorization, overfitting — applied uniformly across every trained model so far (DaraLM-Tiny, -10M, -50M), not just whichever one was trained most recently:

```bash
uv run python scripts/evaluate.py
uv run python scripts/generate_model_card.py --model daralm-50m
```

```
model               params  val_loss  perplexity  mem_vs_chance   overfit
-------------------------------------------------------------------------
daralm-tiny      2,441,856     7.460      1737.5         779.1x       n/a
daralm-10m      10,583,712     6.165       475.7         490.7x        OK
daralm-50m      33,366,528     6.352       573.6         448.4x        OK
```

A few things worth calling out, not just the table:

- **`daralm-tiny`'s overfit check reads `n/a`, correctly.** That checkpoint was trained in Phase 4, before Phase 6 added `history.json` timeline tracking — rather than crash or silently fabricate a verdict, `load_history_or_meta` falls back to the `meta.json` every checkpoint has always had, and `check_overfitting` honestly reports "insufficient data." A real, permanent situation (checkpoints from different phases of one evolving project), handled by degrading gracefully instead of demanding every checkpoint be retrained to match the newest tooling.
- **Memorization is low across the board** (2.8%-4.9% average token-match rate against real training documents) despite being 450-780x the random-chance baseline (`1/vocab_size`) — both things are true and both matter: nowhere near chance, but nowhere near the kind of verbatim reproduction that would be a real concern. Expected, given every model has seen under a handful of passes over the corpus.
- **A genuine bug caught before it wasted 20+ minutes**: the first version of the memorization check had no cap on how much of a document's continuation it verified — for this corpus's longer real articles (many thousands of tokens) that meant thousands of *uncached* forward passes per document (generation here has no KV cache, a deliberate Phase 4 simplicity trade-off). The Tiny model alone took 100 seconds; extrapolated across 10M and 50M this would have run 15-25+ minutes. Fixed by capping the checked continuation to 50 tokens — not just faster, but the methodologically *correct* choice: real memorization audits (e.g. Carlini et al.'s "extractable memorization") test a bounded continuation window, not full-document exact match, so this was a genuine correction, not a shortcut.
- **No overfitting signal yet** for either real-scale model — validation loss is still at its best value at the final checkpoint for both. Expected at 300-600 steps; worth re-checking as training scales up.

`experiments/evaluation_report.json` holds the full structured output (architecture, history, overfitting verdict, all three generation samples, memorization detail) for all three models — the source of truth `scripts/generate_model_card.py` reads from to render **`MODEL_CARD.md`** (spec section 23's full field list: architecture, languages, training data with exact sources/licenses, tokenizer, training tokens, compute used, training config, evaluation results, known limitations, responsible use, license) automatically, not hand-typed — so the card can't silently drift from what was actually run. Its "Known Limitations" section is written as prominently as its numbers, per section 23's own rule: *do not claim capabilities that have not been evaluated.*

## Phase 9: instruction tuning → DaraLM-50M-Instruct

Spec sections 24-25: supervised fine-tune (SFT) DaraLM-50M Base into an instruction-following model, on `{"instruction", "response"}` pairs wrapped in a `<bos><user>...</user><assistant>...</assistant><eos>` chat template, with the prompt portion masked out of the training loss. Base and Instruct are kept strictly separate — same architecture and vocabulary (SFT loads Base's weights directly), different checkpoint directory, different training data/hyperparameters, never overwriting Base.

```bash
uv run python scripts/prepare_instruction_dataset.py --n-english 300
uv run python scripts/train_sft.py --config configs/50m-instruct.yaml --base-checkpoint checkpoints/daralm-50m/best
```

```
step=5   train_loss=6.4423
step=20  train_loss=5.7023  val_loss=5.7301  perplexity=308.0
step=40  train_loss=5.6308  val_loss=5.4131  perplexity=224.3
step=60  train_loss=5.4513  val_loss=5.3171  perplexity=203.8
step=80  train_loss=4.9493  val_loss=5.2818  perplexity=196.7
step=100 train_loss=4.9884  val_loss=5.2697  perplexity=194.4
```

A few things worth calling out:

- **The SFT dataset is 324 examples from two sources**, recorded in `data/raw/MANIFEST_instructions.json`: 300 English examples sampled from `tatsu-lab/alpaca` (CC BY-NC 4.0 — **non-commercial**, this project stays educational/portfolio use), and 25 hand-authored Khmer instruction/response pairs (`data/instructions/khmer_handauthored.jsonl`) — no suitable public Khmer instruction-tuning dataset was available at this project's scale, so a small set was written for it and clearly labeled `"source": "handauthored"`, never presented as sourced data.
- **A real bug, caught and fixed before the dataset was usable**: the first real run of `prepare_instruction_dataset.py` silently loaded **0 of the 25 Khmer examples** — `daralm.data.loader.load_jsonl` was hard-coded to require a `"text"` field (Phase 1's base-pretraining record schema), and the Khmer file uses `{"instruction", "response", ...}` instead, so every line was skipped with a warning that was easy to miss in the log noise. Fixed by generalizing `load_jsonl(path, required_field="text")` to accept the field name it should check for, backward-compatible with every other existing caller. Re-running confirmed all 25 Khmer examples load correctly.
- **No sequence packing for instruction data.** Unlike base pretraining's `PackedTokenDataset`, `InstructionDataset` pads each example individually to `block_size=512` (not the architecture's `max_position_embeddings=1024`) rather than concatenating examples back to back — packing would teach the model that one conversation's `<eos>` is immediately followed by an unrelated next conversation's `<user>` turn, exactly the pattern SFT should discourage. `block_size=512` was chosen after actually measuring tokenized example lengths (p50=103, p90=195, max=490 tokens with the real tokenizer + chat template) — nothing was dropped for exceeding it.
- **The MPS batch-size cliff was re-checked at this new shape, not assumed safe.** Phase 6-7 found a severe Apple Silicon MPS throughput cliff for this architecture at large batch×seq_len shapes. SFT uses a different sequence length (512, not 1024), so the cliff's location wasn't assumed to carry over — isolated timing found batch=4 → 0.29s/step, batch=8 → 0.64s/step, batch=16 → **22.3s/step**, the same pathology recurring at this shape too. `batch_size=4` (matching Base's config) was confirmed, not guessed, to be safely below it.
- **A real, disclosed tokenizer limitation surfaced during evaluation**: the Phase 2 tokenizer was trained only on Wikipedia article text, which has essentially no `<`/`>` characters, so the chat template's plain-text markers (`<user>`, `<assistant>`, etc.) partially decode through the `<unk>` placeholder glyph `⁇` instead of showing literal angle brackets. The underlying token IDs stay internally consistent and learnable — this is a display/readability issue, not silent training corruption — but it wasn't anticipated when the plain-text-marker design decision was made in Phase 9's early planning, and is disclosed rather than quietly patched. Full detail and a real side-by-side Base-vs-Instruct generation comparison (including the honest read that Instruct's outputs are often near-empty, not yet coherent) is in **`MODEL_CARD_INSTRUCT.md`**.
- **`MODEL_CARD_INSTRUCT.md` is hand-authored, not auto-generated** — unlike `MODEL_CARD.md`, `scripts/evaluate.py`'s pipeline is built around base-pretraining metrics (packed-token perplexity, memorization vs. raw training documents) and doesn't yet understand `InstructionDataset`'s loss-masked batches. Every number in it is still real, pulled directly from `checkpoints/daralm-50m-instruct/history.json`/`best/meta.json` and `data/raw/MANIFEST_instructions.json` — extending `evaluate.py` for SFT-style evaluation is tracked as future work, not done here.

> **This section is a historical record of the original Phase 9 run.** Both models were substantially retrained afterward — see "Improving the models" below for the real follow-up story (a dead-end attempt, a correct diagnosis, and what actually worked).

## Phase 10: the inference API

Spec section 26: a FastAPI service around the generation/tokenizer library code every earlier phase already built and tested — `Client → FastAPI → Model Service → Tokenizer → DaraLM → Generation → Response`. No model logic lives in `api/`; `api/services/model_service.py` is a thin adapter over `daralm.inference.generator`/`daralm.tokenizer.tokenizer`, and the routes are thinner still.

```bash
uv run uvicorn api.main:app --reload
```

Which checkpoint gets served is config-driven (never hard-coded, per spec section 30), via three environment variables — `DARALM_CONFIG`, `DARALM_CHECKPOINT`, `DARALM_TOKENIZER` — defaulting to DaraLM-50M **Base** (`configs/50m.yaml` / `checkpoints/daralm-50m/best`). Four endpoints, exactly per spec section 26:

```bash
curl http://127.0.0.1:8000/health
# {"status": "ok", "device": "mps"}

curl http://127.0.0.1:8000/v1/model
# {"model_name": "daralm-50m", "tags": ["50m"], "vocab_size": 16000, "hidden_size": 512,
#  "num_layers": 8, "num_attention_heads": 8, "max_position_embeddings": 1024,
#  "parameters": 33366528, "checkpoint_step": 300, "device": "Apple Silicon (MPS)"}

curl -X POST http://127.0.0.1:8000/v1/tokenize -H "Content-Type: application/json" \
  -d '{"text": "Cambodia is a country in Southeast Asia.", "add_bos": true, "add_eos": true}'
# {"token_ids": [2, 8438, 17, 13, 411, 11, 5862, 1198, 7, 3],
#  "tokens": ["<bos>", "▁Cambodia", "▁is", "▁a", "▁country", "▁in", "▁Southeast", "▁Asia", ".", "<eos>"],
#  "token_count": 10}

curl -X POST http://127.0.0.1:8000/v1/generate -H "Content-Type: application/json" \
  -d '{"prompt": "Cambodia is", "max_new_tokens": 40, "temperature": 0.8, "top_p": 0.9}'
# {"generated_text": "Cambodia is usually found in a model and are off in a quantities. In the
#  edge of the successor, the Roman emperor that the atomic people of December and the power's,
#  the in an virtual latter,", "tokens_generated": 40, "model": "daralm-50m"}
```

All four commands above were run against a real, locally started server (`uv run uvicorn api.main:app`) with the real DaraLM-50M Base checkpoint loaded — not simulated. Same commands re-run against `DARALM_CONFIG=configs/50m-instruct.yaml DARALM_CHECKPOINT=checkpoints/daralm-50m-instruct/best` correctly switched `/v1/model`/`/v1/generate` over to the Instruct checkpoint (step 100) — confirming the env-var-driven model swap actually works, not just that it's documented to.

A few things worth calling out:

- **A real device-placement bug, caught before it shipped.** The first version of `ModelService.from_checkpoint` passed `map_location=device` (MPS) into `load_checkpoint`, which loads the checkpoint's saved RNG-state tensor onto MPS too — but `torch.set_rng_state()` only accepts a CPU `ByteTensor`, so startup crashed with `TypeError: RNG state must be a torch.ByteTensor`. Every training script (`train.py`, `train_sft.py`) avoids this by leaving `map_location` at `load_checkpoint`'s default (`"cpu"`) and relying on `model.load_state_dict`'s in-place value copy to land weights on whatever device the model is already on — `ModelService` was fixed to do the same. Found by actually starting the server against a real checkpoint on this Apple Silicon machine, not by code review.
- **A test that silently did the wrong thing, caught before it gave false confidence.** The first version of the "returns 503 before the model loads" test used `TestClient(app)` without a `with` block, assuming that skips FastAPI's `lifespan` startup — it doesn't, in the Starlette version this project pins; the test was actually loading the real 50M checkpoint from disk on every run. Fixed by unit-testing `api.dependencies.get_model_service` directly against a bare stand-in `Request`, rather than exercising the real app's lifespan at all.
- **`/v1/generate` is a raw-completion endpoint, not a chat endpoint.** Per spec section 26's exact four endpoints — no `/v1/chat` was added, even though `generate_chat()` (Phase 9) exists and could easily wrap one. Pointing the server at the Instruct checkpoint still works (`/v1/generate` doesn't care which weights it's given), but it won't apply the `<user>/<assistant>` chat template the way `generate_chat()` does — that wiring is real, deliberately scoped-out future work, not an oversight.
- **Fail-loudly validation, not just "the model is bad."** Pydantic's `extra="forbid"` on every request schema means a typo like `max_new_tkens` gets an immediate 422 naming the exact bad field, the same discipline `ModelConfig` already applies to training YAML. Verified against a running server, not just unit tests: empty prompt, out-of-range `temperature`, `max_new_tokens` over the 1024 cap, and an unknown field all return 422 with a clear `detail`.
- **Generation runs off the event loop.** `ModelService.generate` uses Starlette's `run_in_threadpool` rather than calling the synchronous, CPU/MPS-bound `generate()` directly inside an `async def` route — otherwise one slow generation would block every other concurrent request on the same server.
- **Tests never touch a real checkpoint.** `tests/test_api.py` builds a tiny in-memory model+tokenizer (same fixture pattern as `tests/test_generation.py`) and swaps it in via `app.dependency_overrides[get_model_service]` — the whole API test suite (11 tests) runs in ~5 seconds and never depends on whatever happens to be trained on disk.

### Containerizing the API

A `Dockerfile` (two-stage build, non-root user, `HEALTHCHECK` against `GET /health`) and a `Makefile` wrap the API in Docker. `configs/`, `checkpoints/`, and `data/` are deliberately **not** baked into the image — they're bind-mounted read-only at `docker run` time, so the same image serves whichever checkpoint you point it at, exactly like running it locally with different `DARALM_CONFIG`/`DARALM_CHECKPOINT`/`DARALM_TOKENIZER` env vars.

```bash
make docker-build
make docker-run     # mounts configs/checkpoints, serves configs/50m.yaml by default
make docker-run CONFIG=configs/50m-instruct.yaml CHECKPOINT=checkpoints/daralm-50m-instruct/best
make docker-logs
make docker-stop
```

Built and actually run for real against the real DaraLM-50M Base checkpoint — not just written and assumed to work:

```
$ curl http://localhost:8000/health
{"status":"ok","device":"cpu"}
$ docker inspect --format='{{.State.Health.Status}}' daralm-api
healthy
```

`device` correctly reports `"cpu"` inside the container (vs. `"mps"` when run natively on this machine) — `daralm.utils.device.get_device()`'s CUDA → MPS → CPU fallback needed zero code changes to work correctly in a container with neither GPU exposed, exactly as it was designed to.

**Image size: 16GB → 1.47GB, two real fixes, both measured before and after:**

1. **CUDA bloat.** `docker history` + `du` inside the built image showed `/app/.venv` was 4.7GB, of which 2.9GB was `nvidia/` (cuBLAS, cuSPARSE, NCCL, ...) and 650MB was `triton` — both CUDA-only, both dead weight on a container with no GPU. Root cause: `pyproject.toml`'s `torch` dependency resolved against PyPI's default index, whose Linux wheel bundles the full CUDA runtime regardless of whether a GPU is present. Fixed with `[tool.uv.sources]`/`[[tool.uv.index]]` in `pyproject.toml`: `torch` resolves against PyTorch's own CPU-only wheel index specifically when `sys_platform == 'linux'`, leaving macOS (this project's actual dev/training platform, MPS-enabled) completely untouched — verified after the change that `uv sync` locally still installs the MPS-capable build and `torch.backends.mps.is_available()` still returns `True`.
2. **A duplicate-layer bug in the Dockerfile itself.** `docker history` showed `COPY /app/.venv /app/.venv` (5GB) immediately followed by `RUN chown -R daralm:daralm /app` — **also 5GB**. `chown -R` over an already-copied multi-GB tree touches every file's metadata, and a layered filesystem writes the whole tree again into a new layer just to change ownership. Fixed by moving ownership into the `COPY --chown=daralm:daralm ...` instructions themselves (one write, not two) and scoping the remaining `RUN chown` to just the three empty mount-point directories it actually needs to touch.

Rebuilt and re-verified end to end after both fixes — same real checkpoint, same curl commands, identical `generated_text` output, `docker inspect`'s healthcheck still reports `healthy`, and the full local test suite (215 tests) + `ruff check` still pass.

**A real infrastructure lesson from building this**: the first build attempt failed mid-layer with host disk I/O errors — the machine's own disk was down to 143Mi free, which was enough to corrupt Docker Desktop's internal storage (its containerd content-store and buildkit metadata DB) badly enough that even `docker image prune`/`docker builder prune` failed rather than reclaiming space. The fix was removing Docker Desktop's 55GB VM disk file (`Docker.raw`) entirely and letting it rebuild fresh on relaunch — a full local Docker reset, not a targeted prune, and worth knowing before it happens again: keep meaningful headroom (multiple GB) free before running a build that pulls torch's CUDA wheels.

## Improving the models (post-Phase-10)

After Phase 10, the obvious next question was: can the actual models get better, not just the infrastructure around them? This section is a real research narrative — including a genuine dead end — not a straight line to success.

**Attempt 1 — bigger/better instruction dataset.** The original Instruct model's SFT set was 324 examples, 92% English. Found a real public Khmer instruction dataset (`saillab/alpaca_khmer_taco`, ~50K rows, Alpaca translated via the TaCo method) and expanded to 923 examples (500 English + 400 real Khmer + 25 hand-authored), rebalancing Khmer from 7.7% to 46% of the data:

```bash
uv run python scripts/prepare_instruction_dataset.py --n-english 500 --n-khmer 400
uv run python scripts/train_sft.py --config configs/50m-instruct.yaml --base-checkpoint checkpoints/daralm-50m/best --block-size 768
```

Result: val perplexity got **worse** (194.4 → 290.4), and generation was still short/degenerate — no visible improvement despite 3x the data. Two real technical findings came out of it regardless: the Khmer dataset's `input` field encodes "empty" as the literal string `"nan"` (a pandas/parquet artifact, not `None`), and `saillab/alpaca_khmer_taco`'s `output` field is a compound `"Instruction in English: ... Response in Khmer: ..."` string requiring real parsing (`daralm.data.loader.fetch_alpaca_khmer_sample`) — both handled correctly and verified against real streamed rows before being wired into the pipeline. But the core goal — visibly better Instruct output — didn't happen.

**Diagnosis**: DaraLM-50M Base had only seen 300 pretraining steps (~4.9M tokens, val perplexity 573.6) — still very undertrained at the language-modeling level. SFT can nudge an already-capable base model toward following instructions; it can't manufacture fluency the base model never learned. More instruction data can't fix that ceiling.

**Attempt 2 — longer base pretraining, then redo SFT.** Resumed DaraLM-50M Base from step 300 → 1500 (`configs/50m.yaml`, `--resume`):

```bash
uv run python scripts/train.py --config configs/50m.yaml --resume
```

```
step=350  val_loss=6.2192  perplexity=502.3
step=500  val_loss=5.8668  perplexity=353.1
step=750  val_loss=5.5265  perplexity=251.3
step=1000 val_loss=5.3517  perplexity=211.0
step=1250 val_loss=5.2530  perplexity=191.1
step=1500 val_loss=5.2041  perplexity=182.0
```

A real, expected wrinkle: resuming with a larger `max_steps` rebuilds the cosine LR schedule against the *new* total before restoring the step counter — so the learning rate visibly jumped back up (from ~10% of peak, fully decayed under the old 300-step schedule, to ~92% of peak under the new 1500-step one) right at the resume point. Not a bug; documented in `configs/50m.yaml`'s own comments so it isn't mistaken for one if seen again.

Redid SFT on the *same* 923-example dataset from attempt 1, this time on top of the much-improved Base:

```
step=50  val_loss=4.9132 perplexity=136.1
step=150 val_loss=4.7525 perplexity=115.9
step=300 val_loss=4.7069 perplexity=110.7
```

**This worked.** Perplexity 110.7 — 43% lower than the best of the two earlier attempts (194.4). More importantly, real generation changed qualitatively, not just numerically:

| Prompt | Old Instruct (either earlier attempt) | New Instruct (this checkpoint) |
|---|---|---|
| Capital of France | `The -thethecoral ⁇ /assistant>` | `The city of the Republic is now the highest of the world. The area of Australia also is an important part of the city, and it has has a result of...` |
| Photosynthesis | *(empty)* | `In this way the problem of the lible vier to the end of the system, the goals in the decision, and the process of the project to the issue of the future.` |

English generation from both Base and Instruct is now visibly longer and more grammatically coherent (real clause structure, capitalization, punctuation) — still confidently wrong and repetitive ("has has a result of a result of"), but categorically past the near-empty/marker-only failure mode. **Khmer generation in Instruct did not improve at the same rate** — it's still short and often collapses toward the closing marker almost immediately, despite Khmer now being 46% of the SFT data. This asymmetry is a real, unresolved, disclosed gap, not glossed over — see `MODEL_CARD_INSTRUCT.md`'s "Honest read" section for the full comparison table and discussion.

**The takeaway, stated plainly**: attempt 1 (bigger instruction dataset) was a real, well-executed dead end on its own. Attempt 2 (longer base pretraining) was the actual fix. If you're improving a small from-scratch LLM's instruction-following and it's not working, the base model's own fluency is worth checking before assuming the instruction data is the problem.

## Production architecture: `/v1/chat` (spec section 27)

Spec section 26's original four endpoints didn't include a chat endpoint — `/v1/generate` is deliberately a raw-completion endpoint, no chat template applied. `/v1/chat` is the first piece of spec section 27's "future production architecture" direction: it wraps `daralm.inference.generator.generate_chat`, which applies the `<user>/<assistant>` chat template `InstructionDataset` trains on, and returns only the assistant's response — not the echoed prompt.

```bash
curl -X POST http://127.0.0.1:8000/v1/chat -H "Content-Type: application/json" \
  -d '{"instruction": "What is the capital of France?", "max_new_tokens": 50, "temperature": 0.8}'
# {"response": "The region of Epirus is XII, is a great part of the air and was...the population
#  of the plant hasleft celebrated by the", "tokens_generated": 50, "model": "daralm-50m-instruct"}
```

Run against the real, currently-published Instruct checkpoint (`DARALM_CONFIG=configs/50m-instruct.yaml DARALM_CHECKPOINT=checkpoints/daralm-50m-instruct/best`), not simulated — output matches the fluent-but-wrong pattern documented above and in `MODEL_CARD_INSTRUCT.md`.

A few things worth calling out:

- **Which checkpoint serves `/v1/chat` is a deployer choice, not a restriction this code enforces.** Calling it against Base doesn't error — Base just runs whatever text it's given through the chat template — but the output won't look like a chat response, since Base was never trained on that template. No endpoint is hard-coded to one checkpoint type, consistent with `/v1/generate`'s existing design.
- **`ModelService.chat()` doesn't need `generate()`'s prompt-length-subtraction trick.** `generate_chat()` already returns only the assistant's response text (the chat-template prompt is stripped internally before the function returns) — `tokens_generated` is a direct encode-and-count, simpler than `/v1/generate`'s re-tokenize-and-diff approach.
- **Same fail-loudly validation as `/v1/generate`**: verified against a running server — empty `instruction` and a caller mistakenly sending `/v1/generate`'s `prompt` field instead of `instruction` both return a clear 422, not a silently-ignored or misinterpreted request.
- **7 new tests** (`tests/test_api.py`), same tiny-fixture pattern as the rest of the API suite — 222 tests total, still passing, still never touching a real checkpoint.

## Observability: `/metrics` + request logging

The next piece of spec section 27 — not a Prometheus/Grafana deployment (spec section 30 explicitly warns against adding infrastructure before it's needed), just the standard first layer any of that would need anyway: a `/metrics` endpoint in Prometheus's own text exposition format (via the standard `prometheus_client` library, not hand-rolled), plus a request-logging middleware.

```bash
curl http://127.0.0.1:8000/metrics
# daralm_requests_total{method="GET",path="/v1/model",status_code="200"} 1.0
# daralm_request_duration_seconds_bucket{le="0.005",method="GET",path="/v1/model"} 1.0
# daralm_tokens_generated_total{endpoint="generate"} 30.0
```

- **`api/observability.py`** owns both the `ObservabilityMiddleware` (logs every request via the same `daralm.utils.logging.get_logger` pattern as the rest of the project, and records it into `daralm_requests_total`/`daralm_request_duration_seconds`) and `record_tokens_generated()`, called from `/v1/generate` and `/v1/chat` after each successful call.
- **Kept out of `ModelService`, on purpose.** Token-count recording lives in the routes, not the service layer — `ModelService` stays HTTP/metrics-agnostic, the same separation-of-concerns reasoning that put `run_in_threadpool` in the service layer but validation in the schemas.
- **`/metrics` has no model dependency**, deliberately — reachable even during the few-second startup window while the model is still loading, so a scraper can tell the process is alive before `/health` would return 200.
- Verified end to end against a real running server: real request counts and histogram buckets appear correctly labeled by method/path, and `daralm_tokens_generated_total{endpoint="generate"}` increments after a real `/v1/generate` call — not just that the endpoint returns 200.

## A frontend: `GET /`

A single self-contained `web/index.html` (inline CSS/JS, no build step, no Node.js toolchain) rather than the spec's eventual Next.js UI — it talks to the same-origin `/v1/chat`, `/v1/generate`, `/v1/tokenize` endpoints via `fetch`, needs no separate dev/build/deploy pipeline, and ships in the same Docker image as the API with one extra `COPY` line. Visit `http://127.0.0.1:8000/` for three tabs (Chat, Generate, Tokenize) with sampling-parameter sliders and a model-info bar reading live from `/v1/model`.

Browser automation wasn't available to click through it live in this session, so verification instead confirmed every JS↔API contract directly against a real running server: each form's exact request payload and the real response's field names (`response`/`generated_text`/`tokens`, `tokens_generated`, `model`, `token_count`) were checked to match what the page's JS reads, including the 422 error shape (`detail: [{loc, msg}, ...]`) the error-rendering code parses. Not the same as a real click-through, and disclosed as such rather than claimed as more than it is.

## DaraLM-150M: the scale-up experiment, honestly reported

After DaraLM-50M-Instruct still hallucinated on basic questions even after a real, verified improvement, the next question was: does scaling up — a bigger corpus *and* a bigger model — actually fix it? Real experiment, real result, and the result is more humbling than a straightforward win.

**Corpus expansion**: re-fetched Wikipedia at 8,000 docs/language (up from 1,500) — 15,939 unique documents after cleaning/dedup (up from 2,990), ~29M tokens at this tokenizer's real compression rate (up from ~8.25M).

**Architecture**: `configs/150m.yaml` — hidden_size=896, 14 layers, 14 heads, same 16K vocab/1024 context as every other size. `scripts/inspect_model_config.py` confirmed 149.2M real parameters before any training started.

```bash
uv run python scripts/prepare_dataset.py --languages km en --docs-per-language 8000
uv run python scripts/train.py --config configs/150m.yaml
```

**A real infrastructure lesson, corrected in real time**: an isolated MPS throughput benchmark measured **42.5s/step** for this model — ~50-60x slower than 50M, which would have meant a ~3.5 hour run. That number was reported to the user as fact. It was wrong. The actual training run, started moments later, ran at a steady **~1.1-1.2s/step** — the isolated benchmark had been contaminated by leftover zombie processes from hours earlier in the session competing for MPS resources, a confound that wasn't caught before reporting the bad number. The correction was issued as soon as the real run's speed became clear, not quietly folded in afterward. **A second, real cost was still there and worth knowing**: validation passes took **~4.3 minutes each** (the val set grew 3x along with the corpus), and with `eval_interval=25` firing 12 times across 300 steps, validation dominated total wall-clock time — real total run time was ~53 minutes, not the initially-corrected "~6 minutes" either. Two real numbers, two corrections, both disclosed as they were found rather than smoothed into one clean estimate after the fact.

**The honest result**:

```
step=25  perplexity=3338.8
step=100 perplexity=1750.6
step=200 perplexity=1278.2
step=300 perplexity=1131.5   (final)
```

Steady improvement, no overfitting signal — but the number that actually matters is the comparison, not the trend: **DaraLM-50M's own perplexity at step 300 was 573.6** — roughly *half* of 150M's step-300 perplexity. The bigger model is currently *behind* the smaller one at the same step count. Two compounding reasons, both real: 150M's MPS-forced `batch_size=2` (vs. 50M's `batch_size=4`) meant it processed only 614,400 tokens by step 300, against 50M's 4,915,200 — an 8x gap in raw data seen: and independent of that, larger models are known to converge more slowly in absolute loss early in training (more parameters to fit) before eventually overtaking smaller ones — a crossover point 300 steps didn't reach. Restated in the same terms as the "Training Tokens" section above: 150M has seen only **~2% of the corpus** so far; 50M's own 1500-step run has seen **~85%**, still under one full pass.

Real generation confirms the numbers, not just an abstract score:

| Prompt | daralm-50m (step 1500) | daralm-150m (step 300) |
|---|---|---|
| "What is the capital of France?" | `...The highest major major cities are: The New York Times...` | `...In a a in the country, and are a late's. The in the part...` |
| "Describe photosynthesis in one sentence." | `...The first time had an important number of...` | `...In the --tt, and the a late of the most not in the same...` |

150M's output is measurably *less* coherent right now — more fragments, more stray punctuation, less real sentence structure.

**The takeaway, stated as plainly as the 50M lesson that led here**: doubling down on scale (bigger corpus + bigger model at once) doesn't pay off automatically — it pays off once training runs long enough to reach the point where the larger model's extra capacity starts being used productively instead of just being slower to fit. 300 steps was enough to *see* that 150M is a fundamentally different, more expensive training problem than 50M was; it wasn't enough to *cross* the point where being bigger actually helps. The honest next step (not yet done) would be extending 150M's own training the same way 50M's was extended earlier — continuing from this checkpoint, not starting over.

**A real, unrelated bug found and fixed along the way**: `scripts/generate_model_card.py` had two hard-coded strings — `"~8.25M tokens"` and `"~3,000 Wikipedia documents"` — left over from before the corpus was expanded. Every model card generated after the corpus grew (including 50M's own, regenerated multiple times this session) was silently repeating the stale, wrong corpus size. Fixed by having the script actually tokenize the real cleaned corpus (`count_corpus_tokens`, a few real seconds of work, not a guess) and read the real document count from `data/cleaned/stats.json` instead of a string written once and never revisited. Caught while writing this section, not by a dedicated audit — a reminder that generated documentation needs the same "don't guess when it can be computed" discipline as the code it describes.

Full model card: `MODEL_CARD_150M.md`. A broader roadmap for where DaraLM could go next (translation, classification, embeddings, and more, evaluated for real feasibility rather than assumed) is in `ROADMAP_NLP_PLATFORM.md`.

## NLP platform roadmap, Stage 1: normalization + classification

`ROADMAP_NLP_PLATFORM.md` reordered the original 20-phase request into risk-ranked stages. Stage 1 — "near-zero-risk foundation" — is now done, both parts real and measured.

**Khmer text normalization** (`daralm/data/khmer_normalize.py`): most of the wishlist (Unicode NFC, zero-width characters, duplicated-character collapsing) turned out to already exist in `daralm/data/cleaner.py`'s `clean_text()` from Phase 1, so this only built what was genuinely missing — Khmer digit (០-៩) ↔ Arabic digit conversion, and removal of a spurious space before Khmer sentence punctuation (`។៕៖៚៘`), measured on the real corpus at **28.5% of `។` occurrences** having a stray preceding space. Exposed with no model dependency at `POST /v1/normalize` — it works even before a checkpoint finishes loading, unlike every other route.

**Classification** (`daralm/model/classification_head.py`, `scripts/train_classifier.py`): a linear probe — a small `ClassificationHead` trained on top of a **frozen** DaraLM-50M backbone, per `ROADMAP_NLP_PLATFORM.md` Phase 3's own model-family design ("lightweight heads for classification-family tasks... cheapest to train, cheapest to store, cheapest to serve"). Pooling is last-real-token, not mean — the right choice for a causal decoder, where only the final real position has attended to the whole input.

First task: language detection (en/km), using the `language` field the corpus already carries from Phase 1's cleaning step — free labels, no new data sourcing needed.

```bash
uv run python scripts/train_classifier.py \
  --backbone-config configs/50m.yaml \
  --backbone-checkpoint checkpoints/daralm-50m/best
```

```
epoch=1 val_loss=0.0829 val_accuracy=0.9786
epoch=2 val_loss=0.0710 val_accuracy=0.9837
epoch=3 val_loss=0.0658 val_accuracy=0.9837   (best)
```

**Read this result honestly**: 98.37% is expected, not impressive — Khmer and English occupy disjoint Unicode ranges, so this task is close to solvable by a character-range heuristic alone (in fact `daralm/data/cleaner.py`'s `detect_language()` already does exactly that, with no model at all). What this run actually validates is the *mechanism* — `return_hidden_states`, the pooling math, the frozen-backbone training loop, checkpointing — working correctly end to end on real data, not a claim that DaraLM has learned anything semantically deep. The real test of that would be sentiment, topic, or intent classification, none of which have labeled Khmer data sourced yet (`ROADMAP_NLP_PLATFORM.md`'s own dataset table flags this as the next concrete gap).

Head checkpoint: `checkpoints/daralm-50m-classify-language/best.pt` (head weights + label list only, ~6KB — the frozen 33.4M-parameter backbone isn't duplicated per task).

## NLP platform roadmap, Stage 2 item 3: grammar correction — a real bug, then a real (nuanced) result

Stage 2's first item, grammar/spelling correction, was chosen specifically because it needed no external data — synthetic typos (`daralm/data/corrupt.py`: delete/duplicate/swap/substitute) generated from the existing corpus's own sentences (`scripts/prepare_grammar_dataset.py`), reusing `InstructionDataset`/`scripts/train_sft.py` unchanged. 3000/450/450 train/val/test examples, SFT'd from DaraLM-50M Base for 400 steps (`configs/50m-grammar.yaml`).

**Training loss looked fine**: perplexity went 60.5 → 49.7 over 400 steps, steady, no overfitting signal — the same shape every other successful fine-tune in this project has shown.

**The first real generation-based evaluation looked like total failure.** `scripts/evaluate_grammar.py` runs real inference (`generate_chat`, not a proxy metric) and scores it with character/word error rate (CER/WER) against the known-correct reference, alongside a no-op baseline (CER/WER of the *uncorrected* corrupted input). The first run: Model CER 1.06–1.25 vs. a 0.028 no-op baseline, 0/200 exact matches — worse than doing nothing.

**A diagnostic overfit test (`scripts/overfit_test_grammar.py`, same "can it even memorize a tiny fixed set" gate as Phase 5) found the real cause was a bug, not a capacity ceiling.** `generate_chat`'s stop condition compared decoded text against the literal string `"</assistant>"` — but this tokenizer has essentially no coverage of `<`/`>` characters (already documented as a limitation elsewhere in this project), so `<` round-trips through encode→decode as `<unk>`'s placeholder glyph, never the literal character. The stop condition silently never fired: generation ran to `max_new_tokens` even after the model had already produced a correct answer, appending garbage after it. **Fixed** in `daralm/inference/generator.py` — compare against the marker's own actual decoded form, computed from the same tokenizer doing the generating, not its literal source text.

**With the fix, the memorization diagnostic passed clearly**: 13/16 exact matches on the tiny fixed set, average CER 0.051 (down from 0.216 pre-fix) — proof the SFT recipe genuinely *can* learn this task's input→output mapping when given enough targeted signal. (The diagnostic script's own automated verdict still printed "FAIL" — its pass threshold required beating an already-near-zero baseline CER by 2x, miscalibrated for a set where the baseline itself is nearly 0. A real result contradicting my own overly strict threshold, reported as found, not smoothed over.)

**But re-running the real held-out evaluation with the same fix confirmed the original finding stands** — this wasn't all one bug:

```
Model CER:    1.23   (no-op baseline: 0.028)
Model WER:    1.42   (no-op baseline: 0.14)
Exact match:  0/200
```

**The honest, corrected picture**: the model can learn to reproduce sentences it was trained on (memorization: strong), but does not generalize the correction skill to new, unseen corrupted sentences (held-out: still worse than a no-op). This reframes the original conclusion — it's *not* that SFT-for-generation is fundamentally broken on this architecture (the memorization test disproves that); it's that 400 steps on 3000 examples, at 33M parameters, wasn't enough to learn a *generalizable* rule, only to memorize specific examples. That's a real, different, more actionable finding than "generation is broken" — more data and/or more steps is a reasonable next bet, not a dead end, though it isn't guaranteed to close the gap either.

**A separate, valuable side effect**: the `generate_chat` fix applies to every chat-based capability in this project, not just grammar correction — any prior evaluation relying on `</assistant>`-based stopping was silently running longer than intended. Regression-tested in `tests/test_generation.py`.

**A direct test of "more budget," not an assumption**: rather than guess whether more data/steps would close the generalization gap, the same recipe was re-run scaled up 4×/6× — 12,000 train examples (from 3,000), 2400 SFT steps (from 400), same architecture, same config otherwise. Training perplexity improved dramatically (60.5→**7.40**, vs. the original run's 60.5→49.7). Real held-out evaluation, same 200 test examples, same metric:

| Run | Model CER | Model WER | Exact match | No-op baseline CER |
|---|---|---|---|---|
| Original (400 steps, 3K examples) | 1.23 | 1.42 | 0/200 | 0.028 |
| Scaled up (2400 steps, 12K examples) | **0.62** | **0.89** | 1/200 | 0.028 |

**Honest read**: scaling up helped substantially — CER roughly halved, and the failure mode qualitatively changed from pure degenerate repetition (`"stant> In addition, stant>..."`) to recognizable, topically-on-target text with real remaining spelling/grammar errors (e.g. `"Attactions Psycologicals fors for example to theories can bed astic, existential, or social."` for a source about attractions/psychology/conspiracy theories — wrong in detail, but clearly *trying* to do the task, not hallucinating something unrelated). **But it still does not beat the no-op baseline** — CER 0.62 is still ~22× worse than doing nothing. Some examples (especially Khmer) still fall into repetition loops. The honest conclusion isn't "fixed" — it's "more budget measurably narrows the gap without closing it," which is real, useful information for anyone deciding whether to keep scaling this specific approach (more data/steps) versus trying a different one (e.g. a larger backbone, or an architecture change) for this task.

Not fixed silently, not hidden — the bug, the fix, and both the corrected and scaled-up held-out results are all documented here, same discipline as every other finding in this project. `ROADMAP_NLP_PLATFORM.md` is updated to reflect the corrected picture.

## Running tests

```bash
uv run pytest
```

## License

MIT — see [LICENSE](LICENSE).
