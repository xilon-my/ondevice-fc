"""
SFT warm-start: Qwen2.5-3B-Instruct on DroidCall function-calling data.

Env-driven:
    N_EPOCHS, LR, LORA_R, TAG, SAVE, SEED, DATA_DIR
Trains a LoRA adapter; saves to /root/shared-nvme/runs/fc_sft_<tag>.
"""
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL = "/root/shared-nvme/models/Qwen2.5-3B-Instruct"
DATA = "/root/shared-nvme/cr/data_fc"


def main():
    tag = os.environ.get("TAG", "droidcall")
    epochs = int(os.environ.get("N_EPOCHS", "3"))
    lr = float(os.environ.get("LR", "1.41e-5"))
    lora_r = int(os.environ.get("LORA_R", "16"))
    save = os.environ.get("SAVE", "0") == "1"
    seed = int(os.environ.get("SEED", "42"))
    out_dir = os.environ.get("OUTPUT_DIR", f"/root/shared-nvme/runs/fc_sft_{tag}")
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train = load_dataset("json", data_files=os.path.join(DATA, "sft_train.jsonl"))["train"]
    val = load_dataset("json", data_files=os.path.join(DATA, "sft_val.jsonl"))["train"]
    print(f"train={len(train)} val={len(val)}", flush=True)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=max(16, lora_r * 2),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.08,
        bias="none",
        task_type="CAUSAL_LM",
    )

    config = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        learning_rate=lr,
        max_grad_norm=0.5,
        bf16=True,
        max_length=1280,
        logging_steps=5,
        save_strategy="steps" if save else "no",
        save_steps=200,
        save_total_limit=2,
        eval_strategy="steps" if save else "no",
        eval_steps=200,
        report_to=[],
        seed=seed,
        dataset_text_field=None,  # use chat messages
        model_init_kwargs={"device_map": "cuda:0", "torch_dtype": torch.bfloat16},
    )

    trainer = SFTTrainer(
        model=MODEL,
        args=config,
        processing_class=tokenizer,
        train_dataset=train,
        eval_dataset=val if save else None,
        peft_config=lora_config,
    )
    print("SFT starting...", flush=True)
    t0 = time.time()
    trainer.train()
    print(f"SFT_FINISHED seconds={time.time() - t0:.0f}", flush=True)
    if save:
        trainer.save_model(out_dir)
        print("SFT_MODEL_SAVED", out_dir, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
