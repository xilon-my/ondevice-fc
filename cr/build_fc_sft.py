"""
Convert DroidCall (query + tools + answers) into a chat-formatted SFT dataset.

Each sample -> messages:
  system:  "You are a helpful assistant with access to the following functions. Use them if required.\n<tools JSON>"
  user:    <query>
  assistant: <answers JSON>   (the expected function call(s))

Output: JSONL of {"messages": [...], "tools": ..., "query": ..., "answers": ...}
The extra fields (tools/query/answers) are kept for the GRPO reward + eval.
"""
import json
import os
import random
import sys

FC_SYSTEM = "You are a helpful assistant with access to the following functions. Use them if required."


def to_chat(example):
    tools = example["tools"]
    tools_json = json.dumps(tools, indent=2, ensure_ascii=False)
    system = f"{FC_SYSTEM}\n\n{tools_json}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": example["query"]},
        {"role": "assistant", "content": json.dumps(example["answers"], indent=2, ensure_ascii=False)},
    ]
    return {"messages": messages, "tools": tools, "query": example["query"], "answers": example["answers"]}


def main():
    train_in = sys.argv[1] if len(sys.argv) > 1 else "/root/shared-nvme/DroidCall/data/fc/DroidCall_train.jsonl"
    test_in = sys.argv[2] if len(sys.argv) > 2 else "/root/shared-nvme/DroidCall/data/fc/DroidCall_test.jsonl"
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "/root/shared-nvme/cr/data_fc"
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    with open(train_in) as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"train rows: {len(rows)}", flush=True)

    # hold out a val split from train (disjoint from test)
    rng = random.Random(7)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    val_n = 500
    val_idx = set(idxs[:val_n])
    train_rows, val_rows = [], []
    for i, r in enumerate(rows):
        c = to_chat(r)
        (val_rows if i in val_idx else train_rows).append(c)

    def dump(rows, path):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {path}: {len(rows)}", flush=True)

    dump(train_rows, os.path.join(out_dir, "sft_train.jsonl"))
    dump(val_rows, os.path.join(out_dir, "sft_val.jsonl"))

    # test set (for final eval) in chat format too
    test_rows = []
    with open(test_in) as f:
        for line in f:
            test_rows.append(to_chat(json.loads(line)))
    dump(test_rows, os.path.join(out_dir, "sft_test.jsonl"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
