# Running Open-Source LLMs with Transformers and vLLM

A hands-on tutorial covering inference with Hugging Face Transformers and vLLM, and a deep dive into how vLLM's Transformers modelling backend works so you can bring your own model to vLLM with minimal effort.

## Lessons

| # | Notebook | Topics |
|---|----------|--------|
| 1 | `notebooks/lesson_1_transformers.ipynb` | Loading models with `from_pretrained`, text generation, chat templates, continuous batching with `generate_batch`, serving via `transformers serve` |
| 2 | `notebooks/lesson_2_vllm.ipynb` | The `LLM` class and `SamplingParams`, PagedAttention and continuous batching, `llm.chat()`, serving via `vllm serve` |
| 3 | `notebooks/lesson_3_transformers_backend.ipynb` | Mixin composition (`Base`, `CausalMixin`, `MoEMixin`), weight mapping and component replacement in `Base.__init__`, the attention bridge via `vllm_flash_attention_forward` and `ALL_ATTENTION_FUNCTIONS`, MoE expert fusion with `FusedMoE` |
| 4 | `notebooks/lesson_4_bring_your_model.ipynb` | Compatibility checklist for the Transformers backend: `_supports_attention_backend`, `ALL_ATTENTION_FUNCTIONS` dispatch, `**kwargs` pass-through, replaceable experts submodule, `base_model_tp_plan` / `base_model_pp_plan` — walked through on OLMoE vs. DiffLlama |

Lessons 1–3 use [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B). Lesson 4 uses [OLMoE-1B-7B-Instruct](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct) (7B total parameters, 1B active per token) since it walks through that model's compatibility with the Transformers backend.

## Requirements

- Python 3.10+
- Memory (RAM for CPU inference, or VRAM on a CUDA GPU):
  - Qwen3-0.6B (lessons 1–3): ~1.2 GB for weights in bf16, plus a little for KV cache — budget ~2 GB
  - OLMoE-1B-7B-Instruct (lesson 4): ~14 GB for weights in bf16, plus a couple of GB for KV cache — budget ~16 GB

## Setup

Clone the repo with submodules (pinned to vLLM v0.19.1 and Transformers v5.5.3):

```bash
git clone --recurse-submodules https://github.com/hmellor/vllm-transformers-modelling-backend-tutorial.git
cd vllm-transformers-modelling-backend-tutorial
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

Create a virtual environment and install dependencies ([uv installation](https://docs.astral.sh/uv/getting-started/installation/)):

```bash
uv venv --python 3.12
source .venv/bin/activate
# CPU
uv pip install -r requirements.txt
# GPU
VLLM_USE_PRECOMPILED=1 uv pip install -r requirements.txt
```

Launch Jupyter and open the first notebook:

```bash
jupyter notebook notebooks/lesson_1_transformers.ipynb
```
