# Results — DroidCall function calling (raw eval JSONs)

Each `fc_eval_<tag>.json` is written by `cr/eval_fc.py` (TAG=<tag>) on the
200-sample held-out DroidCall test set, graded with the same tolerant semantic
`evaluate_completion` used for the GRPO reward (exact acc = reward == 1.0).

| tag | model | exact | mean reward | file |
|---|---|---|---|---|
| base | Qwen2.5-3B-Instruct | 42/200 = 21.0% | 0.358 | `fc_eval_base.json` |
| sft | + SFT LoRA (DroidCall, 3 epochs) | 50/200 = 25.0% | 0.395 | `fc_eval_sft.json` |
| rl | + GRPO 400 steps (from SFT) | 95/200 = 47.5% | 0.649 | `fc_eval_rl.json` |
| rl1000 | + GRPO 1000 steps (continued) | 102/200 = 51.0% | 0.684 | `fc_eval_rl1000.json` |

## Fields

- `tag` — eval tag (`base` | `sft` | `rl` | `rl1000`)
- `exact` — number of test samples with reward == 1.0
- `total` — 200 (DroidCall held-out test set)
- `mean_reward` — mean semantic reward over the test set (partial credit)
