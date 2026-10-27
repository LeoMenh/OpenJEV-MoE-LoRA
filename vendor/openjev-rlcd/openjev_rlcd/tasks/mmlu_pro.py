"""MMLU-Pro (Wang et al. 2024) as an EPISTEMIC calibrated-choice task: one correct option out of
up to ten, and reasoning genuinely changes what the model can know (unlike ChaosNLI, where the
uncertainty is annotator disagreement that no amount of reasoning removes).

Items carry the same fields the ChaosNLI code uses: q = one-hot ground truth, counts = q, so the
Brier objective, the metrics (JSD to one-hot, accuracy, ECE, selective risk = 0/1 error) and the
label draws all work unchanged. Options beyond an item's own count are masked out of u.

  python -m openjev_rlcd.tasks.mmlu_pro --probe N   zero-shot base model: direct answer vs one brief rationale
"""
import argparse
import json
import random

import torch

from ..metrics import selective_metrics

LETTERS = "ABCDEFGHIJ"


STEM = ("math", "physics", "chemistry", "engineering", "computer science")  # computation-heavy subjects


def load_items(split_seed=0, n_train=4000, n_dev=500, n_test=1500, categories=None):
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro")["test"]
    items = []
    for x in ds:
        if categories and x["category"] not in categories:
            continue
        k = len(x["options"])
        y = int(x["answer_index"])
        q = [1.0 if i == y else 0.0 for i in range(k)]
        items.append(dict(uid=str(x["question_id"]), question=x["question"], options=list(x["options"]),
                          category=x["category"], q=q, counts=q))
    random.Random(split_seed).shuffle(items)
    return dict(train=items[:n_train], dev=items[n_train:n_train + n_dev],
                test=items[n_train + n_dev:n_train + n_dev + n_test],
                pool=items[n_train + n_dev + n_test:])  # unused questions: an unlabeled pool for distillation


def user_message(item, reasoning=True):
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(item["options"]))
    tail = ("Think step by step, briefly (a few sentences), then finish with exactly one line of the form "
            "'Answer: X', where X is the letter of the correct option." if reasoning else
            "Reply with exactly one line of the form 'Answer: X', where X is the letter of the correct option.")
    return f"Question: {item['question']}\n\nOptions:\n{opts}\n\n{tail}"


def letter_ids(tok):
    ids = [tok.encode(" " + c, add_special_tokens=False) for c in LETTERS]
    assert all(len(i) == 1 for i in ids) and len({i[0] for i in ids}) == len(LETTERS)
    return [i[0] for i in ids]


@torch.no_grad()
def probe(n, max_new_tokens=256):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    name = "Qwen/Qwen3-1.7B"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    ids = letter_ids(tok)
    items = load_items()["dev"][:n]
    marker = tok.encode("\nAnswer:", add_special_tokens=False)

    def read(ctxs, ks):  # left-padded so only the last position's logits are materialized
        width = max(map(len, ctxs))
        x = torch.full((len(ctxs), width), tok.pad_token_id)
        m = torch.zeros_like(x)
        for i, c in enumerate(ctxs):
            x[i, width - len(c):], m[i, width - len(c):] = torch.tensor(c), 1
        z = model(input_ids=x.cuda(), attention_mask=m.cuda(), logits_to_keep=1).logits[:, -1, ids].float()
        for i, k in enumerate(ks):
            z[i, k:] = -float("inf")
        return z.softmax(-1).cpu()

    out = {}
    for mode in ["direct", "reasoning"]:
        us, lens, trunc = [], [], 0
        for s in range(0, n, 16):
            chunk = items[s:s + 16]
            prompts = [tok.apply_chat_template([{"role": "user", "content": user_message(it, mode == "reasoning")}],
                                               tokenize=False, add_generation_prompt=True, enable_thinking=False)
                       for it in chunk]
            if mode == "direct":
                ctxs = [tok(p)["input_ids"] + tok.encode("Answer:", add_special_tokens=False) for p in prompts]
            else:
                enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left").to("cuda")
                gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                     max_new_tokens=max_new_tokens, stop_strings=["Answer:"], tokenizer=tok,
                                     pad_token_id=tok.pad_token_id)
                ctxs = []
                for row, prow in zip(gen, enc["input_ids"]):
                    comp = row[enc["input_ids"].shape[1]:].tolist()
                    comp = [t for t in comp if t != tok.pad_token_id]
                    text = tok.decode(comp)
                    if not text.rstrip().endswith("Answer:"):
                        trunc += 1
                        comp = [t for t in comp if t not in (tok.eos_token_id,)] + marker
                    lens.append(len(comp))
                    ctxs.append(prow[prow != tok.pad_token_id].tolist() + comp)
            us.append(read(ctxs, [len(it["options"]) for it in chunk]))
        u = torch.cat(us)
        probs = [p[:len(it["options"])].tolist() for p, it in zip(u, items)]
        acc = sum(p.index(max(p)) == it["q"].index(1.0) for p, it in zip(probs, items)) / n
        sel = selective_metrics(items, probs)
        out[mode] = dict(acc=acc, mean_max_p=sum(max(p) for p in probs) / n, ece=sel["ece"], aurc=sel["aurc"],
                         mean_tokens=(sum(lens) / len(lens)) if lens else 0, truncated=trunc / n)
        print(mode, json.dumps({k: round(v, 3) for k, v in out[mode].items()}), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    a = ap.parse_args()
    if a.probe:
        probe(a.probe, a.max_new_tokens)
