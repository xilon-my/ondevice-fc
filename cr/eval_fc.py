"""
Evaluate function-calling accuracy (DroidCall test set).

Loads base / SFT adapter / RL adapter on Qwen2.5-3B-Instruct, generates the
function-call JSON for each test query, and grades it with the same
`evaluate_completion` used for the GRPO reward. Reports:
  - exact acc  (reward == 1.0)
  - mean reward (partial credit)
Usage: TAG=base | TAG=sft ADAPTER=/root/shared-nvme/runs/fc_sft_droidcall | TAG=rl ...
"""
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fc_reward import evaluate_completion

MODEL = "/root/shared-nvme/models/Qwen2.5-3B-Instruct"
DATA = "/root/shared-nvme/cr/data_fc/sft_test.jsonl"
ADAPTER = os.environ.get("ADAPTER", "/root/shared-nvme/runs/fc_sft_droidcall")


def main():
    tag = os.environ.get("TAG", "base")
    max_new = int(os.environ.get("MAX_NEW", "1024"))
    gen_batch = int(os.environ.get("GEN_BATCH", "8"))
    temp = float(os.environ.get("TEMP", "0.0"))

    print(f"loading {tag} model...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
    if tag != "base":
        model = PeftModel.from_pretrained(base, ADAPTER)
    else:
        model = base
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = []
    with open(DATA) as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"test rows: {len(rows)}", flush=True)

    # build templated prompts (messages already in the data)
    templated = [tokenizer.apply_chat_template(r["messages"][:-1], add_generation_prompt=True, tokenize=False)
                 for r in rows]

    exact = 0
    total_r = 0.0
    t0 = time.time()
    for start in range(0, len(rows), gen_batch):
        end = min(start + gen_batch, len(rows))
        enc = tokenizer(templated[start:end], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new, do_sample=(temp > 0), temperature=max(temp, 0.001),
                top_p=0.95, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        for j in range(start, end):
            gen = out[j - start, enc["input_ids"].shape[1]:]
            text = tokenizer.decode(gen, skip_special_tokens=True)
            r = evaluate_completion(text, rows[j]["answers"])
            total_r += r
            exact += int(r == 1.0)
        if (start + gen_batch) % 40 == 0:
            print(f"  {end}/{len(rows)} done, exact so far: {exact}/{end} = {exact/end:.3f} ({time.time()-t0:.0f}s)",
                  flush=True)

    print("=" * 30, flush=True)
    print(f"[{tag}] exact acc = {exact}/{len(rows)} = {exact/len(rows):.3f} | mean reward = {total_r/len(rows):.3f}",
          flush=True)
    with open(os.path.join("/root/shared-nvme/runs", f"fc_eval_{tag}.json"), "w") as f:
        json.dump({"tag": tag, "exact": exact, "total": len(rows), "mean_reward": total_r / len(rows)}, f)


if __name__ == "__main__":
    main()
