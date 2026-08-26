

# DaraLM → Khmer NLP Platform: Feasibility Analysis & Roadmap

> Written before any of this is implemented. Phase 1 is a real inspection of the codebase as it exists today (verified via direct commands, not recalled from memory). Phase 2 is an honest feasibility assessment — including deliberate pushback on parts of the original 16-capability request that this project's own results this session make unrealistic at the current scale. Nothing here is implemented yet; this is the "before coding" deliverable, per the request that spawned it.

---

## Phase 1: Current Architecture (verified against the live codebase)

**Model architecture** — decoder-only Transformer, implemented from scratch (`daralm/model/`): pre-norm RMSNorm, causal multi-head self-attention with RoPE, GELU feed-forward, weight-tied embeddings. No `AutoModelForCausalLM`, no third-party model code.

**Tokenizer** — SentencePiece Unigram, **16,000 vocabulary**, one shared tokenizer across every model size. Special tokens: `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3`. Trained on the Phase 1 Wikipedia corpus — has essentially no coverage of `<`/`>` characters (a real, disclosed limitation already documented in `MODEL_CARD_INSTRUCT.md`: chat template markers partially decode through `<unk>`).

**Model family currently trained** (`checkpoints/`):

| Model | Params | Hidden | Layers | Heads | Context | Status |
|---|---|---|---|---|---|---|
| daralm-tiny | 2.4M | 128 | 2 | 4 | 256 | Done (pipeline verification) |
| daralm-10m | 10.6M | 288 | 6 | 6 | 1024 | Done |
| daralm-50m | 33.4M | 512 | 8 | 8 | 1024 | Done — Base, perplexity 182.0 |
| daralm-50m-instruct | 33.4M | 512 | 8 | 8 | 1024 | Done — Instruct, perplexity 110.7 |
| daralm-150m | 149.2M | 896 | 14 | 14 | 1024 | **Training now**, in progress this session |

**Context length**: 1024 tokens, all sizes — a real constraint for document QA/RAG (Phase 10 of the original request) and long-document summarization.

**Training pipeline** (`daralm/training/`): mixed precision (bf16/fp16/fp32 via `torch.autocast`), gradient accumulation, gradient clipping, checkpointing with full resumability (model + optimizer + scheduler + RNG state), cosine LR schedule with warmup. `Trainer` already supports two batch shapes transparently (`_unpack_batch`): plain tensors for base pretraining (`PackedTokenDataset`), and `(input_ids, labels)` tuples for loss-masked SFT (`InstructionDataset`) — **this dual-format support is directly reusable for several of the requested capabilities** (see Phase 2).

**Inference pipeline** (`daralm/inference/`): `generate()` (raw completion) and `generate_chat()` (wraps `daralm/data/chat_template.py`'s plain-text `<user>/<assistant>` markers). No KV cache — every new token recomputes the full sequence (a documented, deliberate "correct before fast" simplicity tradeoff from early in the project). Sampling: temperature, top-k, top-p, repetition penalty, all independently unit-tested.

**Checkpoint structure**: `checkpoints/<model_name>/{best,step-N}/checkpoint.pt` (model + optimizer + scheduler + RNG state + config.yaml + tokenizer fingerprint) plus `meta.json` and `history.json`. `daralm/training/checkpoint.py`'s `load_checkpoint`/`save_checkpoint` are generic — not tied to causal-LM training specifically.

**Dataset format**: base pretraining uses `{"text", "language", "source"}` JSONL, packed into contiguous `block_size` chunks (`PackedTokenDataset`). Instruction data uses `{"instruction", "response", "language", "source"}` JSONL, individually padded with loss masking (`InstructionDataset`) — **the loss-masking mechanism here is exactly what most of the requested capabilities (classification, extraction, translation) would reuse**, since they're all "mask the prompt, supervise the target" in the same shape.

**Evaluation system** (`daralm/evaluation/`, `scripts/evaluate.py`): perplexity, fixed-prompt generation quality, memorization-vs-training-data, overfitting checks — run uniformly across every base-pretrained checkpoint. **Does not currently evaluate SFT/instruction models** (documented gap — `MODEL_CARD_INSTRUCT.md` is hand-authored specifically because of this) and has no support at all for classification/translation/extraction-style metrics (accuracy, BLEU, F1, etc.) — this is real, net-new work, not an extension of an existing eval harness.

**CLI / scripts** (`scripts/`): one script per pipeline stage (`prepare_dataset.py`, `train_tokenizer.py`, `train.py`, `train_sft.py`, `evaluate.py`, `generate.py`, `inspect_model_config.py`, `generate_model_card.py`, `overfit_test.py`, `analyze_run.py`, `prepare_instruction_dataset.py`) — no unified CLI entry point or `daralm` console script.

**Configuration system** (`daralm/model/config.py`): Pydantic v2, `extra="forbid"` (typos fail loudly), cross-field validation (head-count divisibility, RoPE even-head-dim, warmup-vs-max-steps). One YAML per model size — this pattern scales cleanly to new task-specific configs.

**API/server** (`api/`): FastAPI, config-driven model loading via env vars (`DARALM_CONFIG`/`DARALM_CHECKPOINT`/`DARALM_TOKENIZER`), one loaded model per process. Endpoints today: `GET /health`, `GET /v1/model`, `POST /v1/tokenize`, `POST /v1/generate`, `POST /v1/chat`, `GET /metrics` (Prometheus), `GET /` (static frontend). **Single-model-per-process is a real architectural fact to reckon with** — the "DaraLM Tools" `/v1/translate`, `/v1/classify`, etc. endpoints in the original request assume either one model that does everything, or a router in front of several loaded models; today's `ModelService` loads exactly one.

**Tests**: 226 tests, `pytest`, tiny-fixture pattern throughout (real tokenizer/model objects built from tiny synthetic corpora, never loading real checkpoints in unit tests) — this pattern is directly reusable for every new capability's tests.

**Device support**: CUDA → MPS → CPU auto-detect (`daralm/utils/device.py`), Apple Silicon is this project's actual primary dev/training platform. A real, hard-won, recurring lesson this session: **MPS throughput does not scale predictably with model size or batch size** — every model size so far has needed its own direct measurement (10M/50M found a batch-size cliff; 150M today found the cliff moved to model size itself, plus a benchmark contaminated by leftover process contention that gave a false 42.5s/step reading before the real ~1.1s/step was measured from the actual training run). **This means: any new capability that changes the model's effective batch shape (longer sequences for RAG context, wider batches for embedding training) needs its own fresh MPS measurement — never assume a prior size's numbers carry over.**

---

## Phase 2: Capability Feasibility — including honest pushback

Before the table: the single most important fact from this session's own results has to inform this analysis, or the table below would be dishonest. **DaraLM-50M-Instruct, after being retrained on a 5x-improved Base model, still produces fluent-sounding, factually empty output on basic single-turn Q&A** ("The city of the Republic is now the highest of the world" — grammatically real, semantically nothing). That's the *simplest* possible generative task this architecture supports. Several of the 16 requested capabilities (structured JSON, tool calling, NL-to-SQL) require **strictly harder** reliability — not just fluent text, but *exactly correct, machine-parseable* output, every time. Recommending those as nearer-term work than they honestly are would repeat the same overclaiming this project has explicitly committed to avoiding (`MODEL_CARD_INSTRUCT.md`, `MODEL_CARD.md`'s "Known Limitations" sections, the whole "don't claim untested capabilities" principle this project has followed since Phase 0).

| Capability | Feasibility | Training Needed | Architecture Change | Data Needed | Honest note |
|---|---|---|---|---|---|
| **Khmer text normalization** | **High** | No | None — pure preprocessing | None | Not a model capability at all — a text-processing utility (Unicode/zero-width/spacing normalization) that runs *before* the tokenizer. Zero risk, immediately useful, and several other items on this list (OCR correction, grammar correction, translation) implicitly depend on having this first. |
| **Classification** | **High** | Yes | Small — classification head on final hidden state, OR reuse existing generative format | Labeled Khmer/English examples (hundreds–low thousands per task, not millions) | The *least* generation-reliability-dependent item on this list — doesn't need the model to produce long correct text, just a well-separated decision boundary. Realistic starting point. |
| **Language detection** | **High** | Minimal | None | Small labeled set (or none — could bootstrap from existing `language` field already in every dataset record) | A trivial special case of classification; nearly free given existing data already carries `language` labels. |
| **Khmer↔English translation** | **Medium** | Yes | Minimal (reuse `InstructionDataset`'s loss-masking shape) | A real parallel corpus — **not currently in this project**; would need sourcing/vetting like every other dataset here (license, quality) | Direct generation task — inherits the demonstrated hallucination/fluency-without-accuracy problem. Expect a real fine-tune to produce translations that are grammatically plausible but frequently wrong at this parameter count/data scale, exactly like current chat output. Worth attempting with honest expectations, not worth promising quality. |
| **Grammar/spelling correction** | **Medium-High** | Yes | Minimal (same shape as translation) | **None needed externally** — synthetic corrupted/correct pairs can be generated directly from the existing Wikipedia corpus (delete/duplicate characters, mangle spacing) | One of the more realistic generation tasks on this list specifically *because* training data is free and the task is narrow (correct-a-known-error, not generate-open-ended-content) — a genuinely good early candidate. |
| **Embeddings / semantic search** | **Medium** | Yes (contrastive) | Real — pooling strategy + projection layer, new training objective entirely | Contrastive pairs (query/positive/negative) — not currently in this project | Architecturally straightforward to bolt on; quality is a completely open question until measured — a causal LM's hidden states were never optimized to be good embeddings, and this project has no evidence either way yet. Requires new eval infra (Recall@K, MRR) this project doesn't have. |
| **Reranking** | **Medium** | Yes | Small (scoring head or reuse classification-style head) | Query-document relevance pairs — not currently in this project | Similar risk profile to classification (doesn't need long correct generation) but needs new data this project hasn't sourced. |
| **Document QA / RAG** | **Split** | Yes (for the generation half) | Real (retrieval is genuinely new infrastructure) | Depends on embeddings/reranking above | The retrieval half (vector DB, chunking, top-K retrieval) is standard infrastructure, independent of model quality, and can be built and tested on its own. The generation half inherits every hallucination problem already documented — a retrieved-context answer will still confabulate. Keep these two halves honestly separate in any status reporting. |
| **Named Entity Recognition / info extraction** | **Low-Medium** | Yes | Moderate — reliable structured output at the level NER needs | Labeled Khmer NER data — scarce/nonexistent for Khmer specifically, would likely need synthetic generation + human review | Requires the model to reliably identify *and* correctly format spans — a harder reliability bar than classification. Realistic only after the model demonstrates it can hit close to 100% valid-JSON rate on *something* simpler first (see structured output row). |
| **Summarization** | **Low-Medium** | Yes | Minimal | Summarization pairs — scarce for Khmer, would need sourcing or synthetic generation | Open-ended long-form generation — the hardest fluency-without-accuracy failure mode to hide, since a bad summary is exactly "fluent nonsense" dressed as a summary. Needs a real dataset this project doesn't have. |
| **Structured JSON generation** | **Low** | Yes | None architecturally, but real engineering (schema validation, retry/repair) | Instruction-style examples with strict JSON targets | This is the actual prerequisite for NER, tool calling, and NL-to-SQL below — worth building and *measuring* (valid-JSON rate) before attempting any of those three, as an honest go/no-go gate. |
| **Tool / function calling** | **Not currently realistic** | Yes, extensively | Moderate (schema selection + argument extraction) | Structured tool-call examples | Depends entirely on structured-output reliability this project has not yet demonstrated at any level. Attempting this before structured JSON generation is proven out would be building on an unverified foundation — exactly what this project's engineering principles argue against. |
| **Intent detection** | **High** | Yes | None — this is classification with a different label set | Labeled intent examples | Not architecturally distinct from classification above; grouping these under one task head is the right design, not two separate efforts. |
| **Natural-language-to-SQL** | **Not currently realistic** | Yes, extensively | Moderate | Khmer NL→SQL pairs — essentially nonexistent publicly, would need significant synthetic generation | Needs both real translation ability (Khmer→intent) *and* perfect structured output (valid SQL) simultaneously — compounds two of the least-proven capabilities on this list. The safety net (read-only, schema allowlist, no DROP/DELETE/UPDATE/INSERT) described in the original request is correct and necessary *regardless* of model quality, but doesn't make the generation itself more reliable. |
| **OCR post-correction** | **Better handled using a separate model / not now** | Yes | Minimal (same shape as grammar correction) | OCR-corrupted/correct pairs — needs an actual Khmer OCR system as a prerequisite, which this project doesn't have and wasn't asked to build | Correctly scoped in the original request as "don't build OCR inside DaraLM" — but that also means this depends on external infrastructure not yet in place. Lowest priority until that dependency exists. |
| **Synthetic Khmer training-data generation** | **Medium** | No new training (uses existing models) | None | None — this *produces* data, doesn't consume it | Genuinely useful as a *tool* to help build datasets for the harder items above (translation, NER, summarization) — but synthetic data from a model that already hallucinates needs human review before being trusted as training data for anything else, or errors compound. |

### The honest bottom line

Of 16 requested capabilities: **2 need no model training at all** (normalization, and arguably synthetic-data-generation-as-a-tool), **4 are realistic near-term wins** (classification, language detection, intent detection, grammar correction), **1 is a good measurable gate to build before anything downstream** (structured JSON generation), and **9 either need data this project doesn't have, need reliability this project hasn't demonstrated at any level yet, or both**. Building all 16 "at once" was already ruled out in the request itself — the analysis above is the concrete version of *why*, not just *that*.

---

## Phase 3: Model Family — Recommended Design

**Not 8 separate full checkpoints.** At 149M params, a full checkpoint is ~570MB (state dict + optimizer + scheduler); 8 of those is untenable to train, store, and keep synchronized with a moving Base, and most of the 16 capabilities don't need separate full pretraining runs to work.

Recommended structure, cheapest-to-most-expensive:

1. **Shared Base + Instruct** (already exist) — the foundation every task-specific variant builds from, exactly as Base/Instruct are already kept strictly separate.
2. **Task-specific fine-tunes for generation tasks** (translation, grammar correction, summarization) — small, full fine-tunes from Base, same pattern as Instruct's own SFT (`InstructionDataset`'s loss-masking already generalizes to these — different data, identical mechanism). Each gets its own checkpoint directory, same as `daralm-50m-instruct`, never overwriting Base.
3. **Lightweight heads for classification-family tasks** (classification, intent, language detection, reranking) — a small linear/MLP head on top of the frozen (or lightly fine-tuned) final hidden state, *not* a new full checkpoint per task. Cheapest to train, cheapest to store, cheapest to serve (one shared backbone, several small heads loaded alongside it).
4. **A genuinely new component for embeddings** — pooling + projection layer + contrastive training objective. Doesn't fit the "head on frozen backbone" pattern as cleanly since embedding quality typically benefits from fine-tuning the backbone too; treat as its own small training effort, evaluated independently before deciding whether to keep it a separate checkpoint or merge back.
5. **LoRA adapters — deliberately not recommended yet.** LoRA's main value is cheaply supporting *many* task variants from one frozen backbone without full-fine-tune storage/compute cost per task. At 149M params, a full fine-tune is already cheap enough (150M SFT run this session: 300 steps in ~6 minutes of actual training compute) that LoRA's cost savings don't outweigh the real complexity it adds (new dependency, new training code path, an extra abstraction layer over something this project has kept deliberately simple throughout). Revisit if/when the model family grows past what full fine-tunes can comfortably manage.

```
checkpoints/
├── daralm-50m/                  # Base (existing)
├── daralm-50m-instruct/         # Instruct (existing)
├── daralm-150m/                 # Base, larger (training now)
├── daralm-150m-instruct/        # Instruct, larger (future, mirrors existing pattern)
├── daralm-150m-translate/       # full fine-tune, Phase 4
├── daralm-150m-grammar/         # full fine-tune, later
└── daralm-150m-heads/
    ├── classify.pt              # lightweight head, shared backbone
    ├── intent.pt                # lightweight head, shared backbone
    └── rerank.pt                # lightweight head, shared backbone
```

---

## Recommended Datasets (per capability, only for the near-term-realistic set)

- **Classification / intent / language detection**: no single canonical Khmer source identified yet — would need to either construct from existing corpus (e.g. topic labels from Wikipedia categories, already latent in the fetched data) or source/vet a labeled set the same way `alpaca_khmer_taco` was found and vetted this session (license-checked, quality-sampled before committing).
- **Grammar/spelling correction**: self-supervised from the existing `data/cleaned/train.jsonl` corpus (25.9M tokens) via synthetic corruption — no external sourcing needed, lowest-friction dataset on this whole list.
- **Translation**: needs a real Khmer-English parallel corpus, sourced and license-checked with the same discipline as every dataset in this project (`data/raw/MANIFEST*.json` pattern) — not yet identified, first concrete task if this capability is prioritized.

## Recommended Evaluation Metrics (per capability)

- **Classification/intent/language detection**: accuracy, precision/recall/F1 per class, confusion matrix — standard, no new infrastructure beyond what a normal classification eval needs.
- **Grammar correction**: character error rate (CER), word error rate (WER) against the known-correct synthetic target — directly measurable since the ground truth is the un-corrupted source text.
- **Translation**: BLEU and chrF at minimum (both have mature, well-tested implementations); COMET only if a working reference implementation is confirmed compatible with this project's dependency constraints (not yet verified) — plus manual Khmer review, since automatic MT metrics are known to be unreliable for lower-resource languages.
- **Structured JSON generation**: valid-JSON rate (parse success / total attempts) as the *first* metric to establish before any downstream schema-correctness metric matters at all.

## Proposed Folder Structure

```
eval/
├── classification/
├── generation/        # existing perplexity/memorization/overfitting checks, unchanged
├── grammar_correction/
└── translation/        # only populated once/if that capability is actually started
```

Deliberately not scaffolding empty directories for capabilities not being built yet (`embedding/`, `reranking/`, `qa/`, `tool_calling/` from the original request) — matches this project's own "avoid unnecessary complexity" / "do not add infrastructure before it's needed" principles, already applied consistently through Phases 0-10.

---

## Prioritized Roadmap (revised from the original request's ordering)

The original request's Stage 1 led with **Translation**. Given the feasibility analysis above, that's reordered — translation is real work with real data-sourcing risk and inherits the demonstrated hallucination problem; the items below it are lower-risk, faster to get a genuine measured win from, and some directly de-risk what comes after.

**Stage 1 — near-zero-risk foundation — ✅ done, both measured**
1. ✅ Khmer text normalization — `daralm/data/khmer_normalize.py`, `POST /v1/normalize`. Real gap found and fixed: 28.5% of `។` occurrences in the corpus had a stray preceding space.
2. ✅ Classification + intent detection + language detection — `daralm/model/classification_head.py`, `scripts/train_classifier.py`. First real result: 98.37% val accuracy on en/km language detection (frozen DaraLM-50M backbone, linear probe). Honestly caveated in `README.md`: this task is near-trivial (disjoint Unicode scripts) and validates the mechanism, not semantic understanding — sentiment/topic/intent classification remains the real test and still needs labeled Khmer data sourced (see Recommended Datasets above).

**Stage 2 — real fine-tunes, honest expectations**
3. ⚠️ **Grammar/spelling correction — done: a real bug found+fixed, then a real scaled-up test.** `scripts/prepare_grammar_dataset.py` + `scripts/train_sft.py` (configs/50m-grammar.yaml). First run (400 steps, 3,000 examples): training perplexity improved steadily (60.5→49.7), but real generation-based evaluation (`scripts/evaluate_grammar.py`) looked like total failure (CER 1.06-1.25 vs a 0.028 no-op baseline). A memorization diagnostic (`scripts/overfit_test_grammar.py`) traced part of this to a real bug in `generate_chat`'s stop condition (comparing against the literal `"</assistant>"` string, which this tokenizer's `<`/`>` coverage gap means never actually appears in decoded output) — fixed in `daralm/inference/generator.py`. With the fix: strong memorization (13/16 exact matches on a tiny fixed set) but the held-out generalization gap was confirmed real, not a bug artifact (CER still 1.23 vs 0.028). **Then directly tested "does more budget close the gap"**, not assumed: re-ran scaled up 4×/6× (12,000 examples, 2400 steps). Result: training perplexity 60.5→**7.40**; held-out CER improved to **0.62** (WER 0.89, 1/200 exact) — roughly halved, failure mode shifted from degenerate repetition to recognizable-but-flawed text — but still ~22× worse than the no-op baseline, not fixed. **Then tried a targeted structural fix**, per real user-proposed guidance: added a `--noop-fraction` (15% uncorrected examples, teaching "leave this alone" not just "fix this") and made the no-op baseline an explicit PASS/FAIL gate in `scripts/evaluate_grammar.py` rather than just reported context. Retrained; perplexity improved further to **6.23**. Real result: CER **0.53** (WER 0.79, 1/200 exact) — a further modest narrowing, same direction, still **FAIL** — ~26× worse than the no-op baseline. Three real attempts (bug fix, scale-up, no-op subset), three real improvements in degree, zero in kind — this task has not been solved by this recipe. Full writeup in `README.md`.
4. ⚠️ **Structured JSON generation — done: three dataset versions, a retracted conclusion, a real (partial) capability.** v1: shared vocab, 100% official / 0% novel-entity. v2: disjoint 32-40-item vocab pools per field, verified zero overlap — still 0.00% exact-value-match on 200 properly held-out examples, confirmed by an independent check; concluded (at the time) that this was a likely-architectural copy-mechanism ceiling, not a data problem. **v3 retracted that conclusion**: `daralm/data/structured_facts.py` now generates a fresh, phonotactically-plausible-but-invented random string per field per example (real syllable-structure rules, zero fixed vocabulary of any size) — no lookup table exists at all. Retrained; perplexity converged to 1.17 (not v1/v2's ~1.00 — the task got genuinely harder, not gameable the same way). Real result: **85.5% valid-JSON, 85.5% schema-match, 42.00% exact-value-match** — a real, qualitative shift: failures are now copying mistakes on long strings (a few wrong characters), not v2's wholesale substitutions, direct evidence a copy mechanism *is* learnable here. **But a second check (same script, repurposed to test real recognizable English words instead of the now-inapplicable vocab-overlap check) found 0/30 exact matches on real words** — the learned copying is narrowly tuned to the synthetic training distribution's character statistics, not a general-purpose operation. Full writeup in `README.md`.
5. Translation (only after sourcing and license-vetting a real parallel corpus)

**Stage 3 — gate result: partial pass, with real evidence and a specific, actionable caveat.** Item 4's finding evolved twice, both times by testing rather than assuming: a finite vocabulary (however large and disjoint) gets gamed as classification, not extraction, dropping to 0.00% exact-value-match — but replacing it with genuinely unbounded random training values produced a real 42.00% exact-value-match with a qualitatively different, copying-not-substituting failure mode, disproving the earlier "likely-architectural ceiling" conclusion. **The remaining caveat, also found by testing**: that learned copy behavior didn't transfer to real recognizable words (0/30 on a distribution-shift check) — it's tuned to the synthetic training distribution's specific character statistics, not general-purpose. **Before any Stage 3 item (NER, tool calling, NL-to-SQL)**: train on random values that vary in *surface style* too (real-word-like, syllable-soup, alphanumeric, mixed-script), not just randomized *content* within one style — the v3 dataset only varied content, and that's exactly what left the narrow-transfer gap. Skipping this and reusing v3-style single-style randomization for a Stage 3 dataset should be expected to hit the same wall:
6. NER / information extraction
7. Tool calling
8. NL-to-SQL (with the described read-only safety net regardless of model quality)

**Stage 4 — genuinely new infrastructure, independent of the above**
9. Embeddings (new objective, new eval infra)
10. Reranking
11. Document QA/RAG (retrieval half can start anytime; generation half waits on demonstrated answer reliability)

**Not scheduled**: OCR post-correction (blocked on an OCR system this project doesn't have), summarization (blocked on data this project doesn't have), synthetic-data-generation-as-a-tool (useful opportunistically once other capabilities exist to generate examples for, not a standalone milestone).

---

## Engineering rules carried forward (already this project's practice, reaffirmed for this expansion)

Every rule in the original request's "Important Engineering Rules" section already matches how Phases 0-10 were actually built: config-driven (never hard-coded paths/architecture), training separated from inference, seeds recorded per checkpoint, every dataset's source/license tracked in a manifest, tests added alongside every new capability (tiny-fixture pattern, no real-checkpoint dependency), MPS/CPU compatibility measured not assumed. Nothing new to establish here — the work is continuing an existing discipline, not adopting one.

**One rule to make explicit given this document's own contents**: *do not claim a capability works until it has measurable evaluation results* (the original request's own final line) — this is why the feasibility table above says "Low" and "Not currently realistic" for several items instead of padding the table with optimistic "High"s. An honest "this probably won't work well yet, here's why" is more useful than a table that reads well but doesn't survive contact with a real fine-tuning attempt.

---

## Recommendation: what to build first

~~**Khmer text normalization**, followed immediately by **classification/intent/language detection**.~~ Both done — see Stage 1 above, both real and measured (98.37% val accuracy on language detection; 28.5% of `។` occurrences fixed by normalization).

~~**Next up: Stage 2, item 3 — grammar/spelling correction.**~~ Done (see Stage 2 above): a bug found+fixed, then a real, quantified partial win from scaling up (gap narrowed, not closed).

~~**Item 4 — structured JSON generation.**~~ Also done — see Stage 2/3 above: a perfect official score that a novel-entity stress test showed was closed-set classification in disguise, not real extraction.

~~**Next up: item 5, translation — or, alternatively, fixing item 4's dataset first.**~~ Superseded — item 4's dataset was fixed twice (v2 disjoint pools, v3 unbounded random values; see item 4 above), landing on a real, partial capability (42% exact-match, narrow surface-distribution transfer).

**A user-directed detour, run as a real experiment, not a plan**: three parallel tracks aimed at "strengthen the 50M baseline before attempting 150M v2" — JSON's v3 fix (above), grammar correction's no-op-subset retrain, and a 3×-scaled DaraLM-50M-Instruct retrain (784→2,313 examples). Real results, all in `README.md`: JSON improved genuinely (0%→42%). Grammar improved in degree, not in kind — still FAILs its own explicit no-op baseline gate (CER 0.53 vs 0.02, ~26× worse than doing nothing) after three separate real attempts (bug fix, scale-up, no-op subset). Instruct's perplexity improved (110.7→79.98) but real generation quality showed no detectable gain — same fluent-sounding, factually-empty hallucination pattern, differently worded. **Conclusion: the 50M baseline is not "fixed."** One of three tracks (JSON) made real progress; two (grammar, instruct) did not, despite genuinely improving perplexity each time — the same perplexity-vs-real-capability disconnect this project keeps finding, now confirmed a third time. **150M v2 should not proceed on the assumption these tracks succeeded** — scaling parameters on top of an unresolved generation-quality problem repeats this project's own earlier 150M lesson (more params without proportionally more of everything else made results worse, not better) rather than fixing what's actually broken.

**Next up, for real**: either (a) item 5, translation, sourcing a real parallel corpus for the first time this roadmap has needed one — independent of the grammar/instruct ceiling just found; or (b) a genuine investigation into *why* grammar/instruct-style generation tasks plateau on this recipe while classification and (now, partially) extraction don't — e.g. whether a copy/pointer mechanism, a different training objective, or simply a larger backbone is the actual lever, before spending more budget on data-side fixes that have now been tried three times on grammar alone without closing the gap.
