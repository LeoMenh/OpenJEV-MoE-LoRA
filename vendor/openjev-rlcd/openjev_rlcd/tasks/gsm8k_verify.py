"""GSM8K-Verify: a calibrated yes/no DECISION that needs reasoning.

Each item is a GSM8K problem (Cobbe et al. 2021) plus ONE proposed final answer written by a weaker
solver (Qwen3-0.6B, a sampled brief chain of thought). The question -- a Jev-style 'noul' -- is
whether the proposed answer is correct. Without reasoning a model can only use surface cues; with
reasoning it can recompute the answer, so this is epistemic uncertainty that reasoning removes.
Labels come from comparing the proposal with the gold answer, so q is one-hot over [yes, no].

  python -m openjev_rlcd.tasks.gsm8k_verify --build   rebuilds data/gsm8k_verify.json (one GPU, ~20 min; the
                                                  released file is the one used in the paper)
Splits: train/dev from GSM8K train (disjoint problems), test = GSM8K test.
"""
import argparse
import json
import random
import re
from pathlib import Path

import torch

DATA = Path(__file__).resolve().parents[2] / "data" / "gsm8k_verify.json"
LABELS = ["yes", "no"]


def _number(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    if not nums:
        return None
    try:
        v = float(nums[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None
    return int(v) if v == int(v) else round(v, 4)


def _fmt(v):
    return str(v) if isinstance(v, int) else f"{v:g}"


@torch.no_grad()
def build(model_name="Qwen/Qwen3-0.6B", seed=0, batch=64, max_new_tokens=320):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    ds = load_dataset("openai/gsm8k", "main")
    out = []
    for split in ["train", "test"]:
        rows = list(ds[split])
        for s in range(0, len(rows), batch):
            chunk = rows[s:s + batch]
            prompts = [tok.apply_chat_template(
                [{"role": "user", "content": f"{x['question']}\n\nSolve the problem. Think step by step briefly, "
                                             "then finish with exactly one line of the form 'Answer: <number>'."}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False) for x in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left").to("cuda")
            gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                 max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
            for i, (x, row) in enumerate(zip(chunk, gen)):
                text = tok.decode(row[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                cand = _number(text.split("Answer:")[-1]) if "Answer:" in text else _number(text)
                gold = _number(x["answer"].split("####")[-1])
                if cand is None or gold is None:
                    continue
                out.append(dict(uid=f"{split}-{s + i}", split=split, question=x["question"], gold=_fmt(gold),
                                proposed=_fmt(cand), correct=bool(abs(float(cand) - float(gold)) < 1e-6)))
            print(json.dumps(dict(split=split, done=s + len(chunk), kept=len(out),
                                  acc=sum(o["correct"] for o in out) / max(1, len(out)))), flush=True)
    DATA.parent.mkdir(exist_ok=True)
    DATA.write_text(json.dumps(dict(solver=model_name, seed=seed, items=out)))


def load_items(split_seed=0, n_train=4000, n_dev=500):
    rows = json.loads(DATA.read_text())["items"]
    items = [dict(uid=r["uid"], question=r["question"], proposed=r["proposed"],
                  q=[1.0, 0.0] if r["correct"] else [0.0, 1.0], counts=[1.0, 0.0] if r["correct"] else [0.0, 1.0],
                  split=r["split"]) for r in rows]
    pool = [it for it in items if it["split"] == "train"]
    random.Random(split_seed).shuffle(pool)
    return dict(train=pool[:n_train], dev=pool[n_train:n_train + n_dev], test=[it for it in items if it["split"] == "test"],
                pool=pool[n_train + n_dev:])


def user_message(item, reasoning=True):
    tail = ("Check it: think step by step, briefly, then finish with exactly one line of the form "
            "'Answer: yes' or 'Answer: no'." if reasoning else
            "Reply with exactly one line of the form 'Answer: yes' or 'Answer: no'.")
    return (f"Problem: {item['question']}\n\nProposed final answer: {item['proposed']}\n\n"
            f"Is the proposed final answer correct? {tail}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
