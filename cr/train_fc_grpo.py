"""
GRPO for function calling on Qwen2.5-3B (DroidCall). Starts from the SFT adapter.

Env-driven:
    N_SAMPLES, MAX_STEPS, TAG, SAVE, VLLM, CB, NUM_GENERATIONS, GRAD_ACCUM,
    LR, BETA, SFT_ADAPTER (LoRA adapter path from SFT), MAX_COMPLETION
"""
import os
import random
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fc_reward import fc_reward

MODEL = "/root/shared-nvme/models/Qwen2.5-3B-Instruct"
DATA = "/root/shared-nvme/cr/data_fc"
SFT_ADAPTER = os.environ.get("SFT_ADAPTER", "/root/shared-nvme/runs/fc_sft_droidcall")
# optional starting adapter for continued RL (e.g. the step-400 RL adapter)
START_ADAPTER = os.environ.get("START_ADAPTER", SFT_ADAPTER)


def load_train(n_samples, seed):
    ds = load_dataset("json", data_files=os.path.join(DATA, "sft_train.jsonl"))["train"]

    def map_fn(ex):
        # GRPO prompt = the input messages (no assistant answer); keep answers for the reward
        return {"prompt": ex["messages"][:-1], "answers": ex["answers"]}

    ds = ds.map(map_fn)
    if n_samples and n_samples < len(ds):
        rng = random.Random(seed)
        idxs = sorted(rng.sample(range(len(ds)), n_samples))
        ds = ds.select(idxs)
    print(f"train dataset: {len(ds)}", flush=True)
    return ds


def main():
    n_samples = int(os.environ.get("N_SAMPLES", "2000"))
    max_steps = int(os.environ.get("MAX_STEPS", "400"))
    max_completion = int(os.environ.get("MAX_COMPLETION", "512"))
    lora_r = int(os.environ.get("LORA_R", "16"))
    tag = os.environ.get("TAG", "fc_rl")
    seed = int(os.environ.get("SEED", "42"))
    output_dir = os.environ.get("OUTPUT_DIR", os.path.join("/root/shared-nvme/runs", f"fc_{tag}"))
    use_vllm = os.environ.get("VLLM", "0") == "1"
    use_cb = os.environ.get("CB", "0") == "1"
    num_generations = int(os.environ.get("NUM_GENERATIONS", "8"))
    grad_accum = int(os.environ.get("GRAD_ACCUM", "8"))
    lr = float(os.environ.get("LR", "5e-5"))
    beta = float(os.environ.get("BETA", "0.005"))
    save = os.environ.get("SAVE", "0") == "1"
    os.makedirs(output_dir, exist_ok=True)

    train = load_train(n_samples, seed)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=max(16, lora_r * 2),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    common = dict(
        output_dir=output_dir,
        num_generations=num_generations,
        max_completion_length=max_completion,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        learning_rate=lr,
        max_grad_norm=0.5,
        beta=beta,
        bf16=True,
        max_steps=max_steps,
        logging_steps=1,
        log_level="info",
        save_strategy="steps" if save else "no",
        save_steps=50,
        save_total_limit=4,
        report_to=[],
        seed=seed,
        model_init_kwargs={"device_map": "cuda:0", "torch_dtype": torch.bfloat16},
    )
    if use_vllm:
        config = GRPOConfig(**common, use_vllm=True, vllm_gpu_memory_utilization=0.5, vllm_max_model_length=4096)
    elif use_cb:
        config = GRPOConfig(
            **common,
            use_transformers_continuous_batching=True,
            transformers_continuous_batching_config={"use_cuda_graph": False, "max_memory_percent": 0.4},
        )
    else:
        config = GRPOConfig(**common)  # plain in-process generation (stable)

    trainer = GRPOTrainer(
        model=MODEL,
        reward_funcs=fc_reward,
        args=config,
        processing_class=tokenizer,
        train_dataset=train,
        peft_config=lora_config,
    )
    # load the starting LoRA adapter (SFT, or a prior RL adapter for continued training)
    trainer.model.load_adapter(START_ADAPTER, adapter_name="default")
    trainer.model.set_adapter("default")
    print(f"trainer built (adapter {START_ADAPTER} loaded), starting train()...", flush=True)
    t0 = time.time()
    trainer.train()
    print(f"GRPO_FINISHED seconds={time.time() - t0:.0f}", flush=True)
    if save:
        trainer.save_model(output_dir)
        print("GRPO_MODEL_SAVED", output_dir, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
