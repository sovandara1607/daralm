---
license: cc-by-nc-4.0
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
  - instruction-tuned
---

# DaraLM-50M-Instruct

DaraLM-50M-Instruct is a supervised fine-tune of
[`daraa1607/daralm-50m`](https://huggingface.co/daraa1607/daralm-50m). It is an
educational artifact and is not reliable enough for production or factual use.

This repository contains the 300-step, 923-example release. A later local
900-step experiment reported lower validation perplexity, but it is not the
artifact published here.

## Architecture

| Field | Value |
|---|---|
| Parameters | 33,366,528 |
| Hidden size | 512 |
| Layers / heads | 8 / 8 |
| Context | 1,024 tokens |
| Vocabulary | 16,000 SentencePiece Unigram tokens |
| Base checkpoint | DaraLM-50M at step 1,500 |

## Chat format

```text
<bos><user>
{instruction}
</user>
<assistant>
{response}
</assistant><eos>
```

Chat markers are ordinary text. The Wikipedia-trained tokenizer has poor
`<`/`>` coverage, so decoded markers can contain `⁇`.

## Usage

This custom architecture is not directly compatible with
`AutoModelForCausalLM`. Clone the
[`DaraLM repository`](https://github.com/sovandara1607/daralm) for its model
implementation.

```python
import torch
from huggingface_hub import hf_hub_download

from daralm.inference.generator import generate_chat
from daralm.model.config import ModelConfig
from daralm.model.transformer import DaraLMTransformer
from daralm.tokenizer.tokenizer import DaraLMTokenizer

repo_id = "daraa1607/daralm-50m-instruct"
weights_path = hf_hub_download(repo_id, "pytorch_model.pt")
config_path = hf_hub_download(repo_id, "config.yaml")
tokenizer_path = hf_hub_download(repo_id, "tokenizer.model")

config = ModelConfig.from_yaml(config_path)
tokenizer = DaraLMTokenizer.from_pretrained(tokenizer_path)
model = DaraLMTransformer(config.architecture, pad_token_id=tokenizer.pad_id)
release = torch.load(weights_path, map_location="cpu", weights_only=True)
model.load_state_dict(release["model_state_dict"])
model.eval()

response = generate_chat(
    model,
    tokenizer,
    instruction="What is the capital of France?",
    max_new_tokens=60,
)
print(response)
```

## Training

The packaged dataset contains 923 instruction/response pairs: 784 train, 138
validation, and 1 test.

| Source | Examples | Language | Terms |
|---|---:|---|---|
| `tatsu-lab/alpaca` | 499 | English | CC BY-NC 4.0 |
| `saillab/alpaca_khmer_taco` | 399 | Khmer | Treated as Alpaca-derived CC BY-NC 4.0 |
| Hand-authored DaraLM examples | 25 | Khmer | MIT |

Only response tokens contributed to loss. SFT ran for 300 AdamW steps with an
effective batch of 16, a 768-token block size, and bf16 precision.

| Step | Validation loss | Perplexity |
|---:|---:|---:|
| 50 | 4.913 | 136.1 |
| 150 | 4.753 | 115.9 |
| 300 | 4.707 | 110.7 |

English generation became longer and more grammatical after stronger base
pretraining, but it remained factually wrong. Khmer generation remained short
and unstable.

## Limitations

- Fluent-looking responses are frequently fabricated.
- Khmer output is weaker than English output.
- Chat markers can contain `⁇` because of tokenizer coverage.
- The small dataset is partly machine translated and its Khmer quality was not
  independently audited.
- No preference tuning, safety tuning, filtering, or red-teaming was performed.

Do not use this model for factual, medical, legal, financial, or safety-critical
decisions.

## License

The weights combine Wikipedia-derived CC BY-SA 4.0 / GFDL terms with
Alpaca-derived CC BY-NC 4.0 terms. This release must not be used commercially.
