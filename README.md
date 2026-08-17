# On-Device Function Calling via SFT + GRPO on a Single RTX 5090

Train a ~3B on-device model that invokes Android intents / calls functions
(DroidCall, 24 functions, JSON tool calls). Pipeline: SFT warm-start → GRPO
with a verifiable function-calling reward, all on a **single NVIDIA RTX 5090**
(32 GB, Blackwell) with Qwen2.5-3B-Instruct + LoRA.

The design follows **Code-R1: Reproducing R1 for Code with Reliable Rewards**
(Liu & Zhang, 2025): reward quality matters more than reward quantity — a
verifiable reward drives GRPO. Here the verifiable signal is a tolerant
semantic match on the produced function call.

## Result

Graded on DroidCall's 200-sample held-out test set with the same semantic
reward used in RL (function name + arguments, optional args may be omitted):

| Model | exact acc | mean reward |
|---|---|---|
| Qwen2.5-3B-Instruct (base) | 42/200 = 21.0% | 0.358 |
| + SFT (DroidCall, LoRA, 3 epochs) | 50/200 = 25.0% | 0.395 |
| + GRPO (function-calling RL, 400 steps) | 95/200 = 47.5% | 0.649 |
| + GRPO (1000 steps, continued) | **102/200 = 51.0%** | **0.684** |

**Finding.** SFT gives a small lift; **GRPO with a verifiable function-calling
reward more than doubles exact accuracy (25% → 51%, mean reward 0.395 →
0.684)**, and continued RL (1000 steps) adds further gains (47.5% → 51%). The
RL step sharpens function/argument selection and output format. Raw eval JSONs
in [`results/`](results/README.md).

**Download the trained adapters:** [v1.0 Release](https://github.com/xilon-my/ondevice-fc/releases/tag/v1.0)
(`fc_sft_droidcall.tgz`, `fc_fc_rl.tgz`, `fc_fc_rl2.tgz`). Each contains
`adapter_model.safetensors` + `adapter_config.json` + tokenizer files, to be
loaded on Qwen2.5-3B-Instruct via `PeftModel.from_pretrained`.

## Method

1. **Data** — DroidCall (Android intent / function calling). `cr/build_fc_sft.py`
   turns each `(query + tool schema + answer)` triple into chat-format SFT data.
2. **Reward** — `cr/fc_reward.py` parses the model's completion tolerantly
   (JSON array / object / `action` style / Python call) and grades it
   semantically: function name 0.4 + arguments 0.6, optional args may be
   omitted. This is the verifiable reward driving GRPO.
3. **SFT** — `cr/train_fc_sft.py` (LoRA, 3 epochs) warms the base model up on
   the DroidCall distribution.
4. **GRPO** — `cr/train_fc_grpo.py` continues from the SFT adapter (plain
   in-process generation, `n=8`, `beta=0.005`); continued runs resume from a
   prior RL adapter via `START_ADAPTER`.

## Reproduce

### Environment (single node, 1× GPU)

```bash
# Python 3.12, CUDA 12.8, PyTorch 2.7, TRL 1.10, transformers 5.15
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

No vLLM required — rollout uses TRL's plain in-process generation, so the whole
pipeline fits one 32 GB GPU. (Where huggingface.co is unreachable, set
`HF_ENDPOINT=https://hf-mirror.com`.)

### Data

```bash
python cr/build_fc_sft.py   # query + tools + answers -> chat-format jsonl
```

### Train

```bash
# SFT warm-start
TAG=fc_sft_droidcall python cr/train_fc_sft.py

# GRPO 400 steps (from SFT adapter)
TAG=fc_rl SFT_ADAPTER=runs/fc_sft_droidcall python cr/train_fc_grpo.py

# Continued GRPO to 1000 steps (resume from the 400-step adapter)
TAG=fc_rl2 START_ADAPTER=runs/fc_fc_rl MAX_STEPS=1000 python cr/train_fc_grpo.py
```

### Evaluate

```bash
TAG=base                           python cr/eval_fc.py
TAG=sft    ADAPTER=runs/fc_sft_droidcall python cr/eval_fc.py
TAG=rl     ADAPTER=runs/fc_fc_rl         python cr/eval_fc.py
TAG=rl1000 ADAPTER=runs/fc_fc_rl2        python cr/eval_fc.py
```

## Repo layout

```
cr/
  fc_reward.py        function-calling GRPO reward (tolerant parse + semantic score)
  build_fc_sft.py     DroidCall -> chat-format SFT data
  train_fc_sft.py     SFT warm-start (LoRA)
  train_fc_grpo.py    GRPO RL (plain engine, n=8), resumable via START_ADAPTER
  eval_fc.py          DroidCall-test evaluation (base / sft / rl / rl1000)
  test_fc_reward.py   unit tests for the reward (8 hand-written cases)
results/              raw eval JSONs + summary (fc_eval_{base,sft,rl,rl1000}.json)
requirements.txt      pinned runtime deps
```
