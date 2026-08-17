"""
GRPO reward for function calling (DroidCall-style).

The model is given a user request + tool schemas and must emit the expected
sequence of function calls as JSON. Reward (verifiable, no learned RM):

  * valid JSON array            -> base 0.0
  * per expected call:          + (name matches ? 0.4 : 0) + (args match ? 0.6 : 0)
  * score = sum(call scores) / len(expected calls), capped at 1.0
  * invalid/unparseable output  -> 0.0

Arg comparison is a tolerant deep-compare: None/empty fields match, dict keys
compared recursively, lists matched order-independently (like DroidCall).
"""
import json
import re

import numpy as np

_MAX_WORKERS = 16


def _is_none(v):
    return v is None or (isinstance(v, str) and v.strip().lower() == "none")


def deep_compare(gen, exp):
    """Tolerant recursive compare: None fields match; dict keys must match; list order-independent."""
    if _is_none(gen) and _is_none(exp):
        return True
    if type(gen) != type(exp):
        return False
    if isinstance(gen, dict):
        if isinstance(exp, dict):
            for k, ev in exp.items():
                if k not in gen:
                    if _is_none(ev):
                        continue  # optional arg with null value may be omitted
                    return False
                if not deep_compare(gen[k], ev):
                    return False
            return True
        return False
    if isinstance(gen, list):
        if isinstance(exp, list):
            if len(gen) != len(exp):
                return False
            for e in exp:
                if not any(deep_compare(g, e) for g in gen):
                    return False
            for g in gen:
                if not any(deep_compare(g, e) for e in exp):
                    return False
            return True
        return False
    return gen == exp


def extract_calls(text):
    """
    Parse the model output into a list of call dicts, tolerating several formats:
      - JSON array of {"name"/"arguments"}     (DroidCall style)
      - single JSON object {"name"/"arguments"}
      - single JSON object {"action"/"arguments"}
      - single JSON object {"FUNCTION_NAME": {args...}}   (name as key)
      - Python-call text  FUNC(ARG=v, ...)
    Returns a list of {"name", "arguments"} dicts (may be empty).
    """
    if not text:
        return []
    # try JSON array
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return [_normalize_call(c) for c in parsed if isinstance(c, dict)]
        except Exception:
            pass
    # try JSON object
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                if "name" in obj or "action" in obj:  # single-call object
                    return [_normalize_call(obj)]
                # dict of {FUNCTION_NAME: {args}} — possibly multiple
                calls = []
                for k, v in obj.items():
                    if isinstance(v, dict) and ("arguments" in v or "name" in v or "action" in v):
                        calls.append(_normalize_call({**v, "name": v.get("name", v.get("action", k))}))
                    else:
                        calls.append({"name": k, "arguments": v if isinstance(v, dict) else {}})
                return calls
        except Exception:
            pass
    # try Python-call text: FUNC(ARG=v, ...)
    m = re.search(r"(\w+)\s*\(([^)]*)\)", text, re.S)
    if m:
        fname = m.group(1)
        args = {}
        for am in re.finditer(r"(\w+)\s*=\s*(\"[^\"]*\"|\'[^\']*\'|\[[^\]]*\]|\{[^}]*\}|[^,)]+)", m.group(2)):
            k, v = am.group(1), am.group(2).strip()
            try:
                args[k] = json.loads(v)
            except Exception:
                args[k] = v.strip("\"'")
        return [{"name": fname, "arguments": args}]
    return []


def _normalize_call(c):
    """Normalize a parsed call dict to {name, arguments}."""
    name = c.get("name") or c.get("action") or ""
    args = c.get("arguments") or {}
    return {"name": str(name), "arguments": args if isinstance(args, dict) else {}}


def score_call(gen_call, exp_call):
    if not isinstance(gen_call, dict) or not isinstance(exp_call, dict):
        return 0.0
    name_ok = str(gen_call.get("name", "")) == str(exp_call.get("name", ""))
    if not name_ok:
        return 0.0  # wrong function = wrong call, regardless of coincidental args
    gen_args = gen_call.get("arguments") or {}
    exp_args = exp_call.get("arguments") or {}
    args_ok = deep_compare(gen_args, exp_args) if (gen_args or exp_args) else True
    return 0.4 + (0.6 if args_ok else 0.0)


def evaluate_completion(completion, answers):
    """Return reward in [0,1]."""
    parsed = extract_calls(completion)
    if not parsed:
        return 0.0
    if not isinstance(answers, list) or len(answers) == 0:
        return 0.0
    # align: expected calls in order; generated must cover them in order
    gen_iter = iter(parsed)
    total = 0.0
    n = len(answers)
    for exp in answers:
        # find the next generated call matching this expected one (in order)
        best = 0.0
        for g in gen_iter:
            s = score_call(g, exp)
            if s > best:
                best = s
            if s >= 1.0:
                break
        total += best
    return min(1.0, total / n)


def fc_reward(prompts, completions, completion_ids, answers=None, **kwargs):
    """TRL reward func. Returns a float per (prompt, completion) pair."""
    if answers is None:
        answers = kwargs.get("answers")
    if answers is None:
        return [0.0] * len(completions)
    assert len(completions) == len(answers)
    rewards = []
    for comp, ans in zip(completions, answers):
        if isinstance(comp, list):  # conversational messages
            comp = " ".join(m.get("content", "") if isinstance(m, dict) else str(m) for m in comp)
        rewards.append(evaluate_completion(comp, ans))
    mean_r = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"[fc_reward] n={len(rewards)} mean={mean_r:.3f}", flush=True)
    return rewards
