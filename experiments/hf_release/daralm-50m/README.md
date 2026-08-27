---
license: cc-by-sa-4.0
language:
  - km
  - en
pipeline_tag: text-generation
tags:
  - from-scratch
  - decoder-only
  - transformer
  - khmer
  - english
  - pytorch
---

# DaraLM-50M

DaraLM-50M is a 33.4M-parameter Khmer/English base language model implemented
from scratch in PyTorch. It is an educational artifact, not a production or
factual question-answering model.

This is a raw completion model. For the instruction-tuned checkpoint, see
[`daraa1607/daralm-50m-instruct`](https://huggingface.co/daraa1607/daralm-50m-instruct).

## Architecture

| Field | Value |
|---|---|
| Parameters | 33,366,528 |
| Hidden size | 512 |
| Layers / heads | 8 / 8 |
| Feed-forward size | 2,048 |
| Context | 1,024 tokens |
| Vocabulary | 16,000 SentencePiece Unigram tokens |

The decoder uses RMSNorm, causal attention, RoPE, GELU feed-forward layers, and
tied input/output embeddings.

## Usage

This custom architecture is not directly compatible with
`AutoModelForCausalLM`. Clone the
[`DaraLM repository`](https://github.com/sovandara1607/daralm) for its model
implementation.

```python
import torch
from huggingface_hub import hf_hub_download

from daralm.inference.generator import generate
from daralm.model.config import ModelConfig
from daralm.model.transformer import DaraLMTransformer
from daralm.tokenizer.tokenizer import DaraLMTokenizer

repo_id = "daraa1607/daralm-50m"
weights_path = hf_hub_download(repo_id, "pytorch_model.pt")
config_path = hf_hub_download(repo_id, "config.yaml")
tokenizer_path = hf_hub_download(repo_id, "tokenizer.model")

config = ModelConfig.from_yaml(config_path)
tokenizer = DaraLMTokenizer.from_pretrained(tokenizer_path)
model = DaraLMTransformer(config.architecture, pad_token_id=tokenizer.pad_id)
release = torch.load(weights_path, map_location="cpu", weights_only=True)
model.load_state_dict(release["model_state_dict"])
model.eval()

text = generate(
    model,
    tokenizer,
    prompt="Cambodia is",
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.9,
)
print(text)
```

`pytorch_model.pt` contains inference weights and basic metadata. It does not
contain optimizer, scheduler, or RNG state for resuming training.

## Training

- 2,990 cleaned Wikipedia documents: 2,691 train, 149 validation, 150 test.
- Approximately 8.25M corpus tokens, balanced between Khmer and English by
  document count.
- 1,500 AdamW steps, effective batch 16, bf16 precision.
- 24,576,000 tokens processed on Apple Silicon MPS.
- Final validation loss 5.2041; perplexity 182.0.

This checkpoint predates the repository’s later corpus expansion and Common
Crawl merge.

## Limitations

- Outputs are often repetitive, fabricated, or grammatically incomplete.
- It does not reliably follow instructions or answer questions.
- Khmer output is weaker than English and language drift can occur.
- No safety tuning, preference tuning, filtering, or red-teaming was performed.

Do not use this model for factual, medical, legal, financial, or safety-critical
decisions.

## License

Code is MIT licensed. Wikipedia training data is CC BY-SA 4.0 / GFDL, and the
weights inherit applicable share-alike terms.
