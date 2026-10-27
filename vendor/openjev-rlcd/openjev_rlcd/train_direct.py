#!/usr/bin/env python3
"""Baseline: SFT + temperature scaling (direct answer, no rationale).

The SAME instruct model as the RLCD runs (Qwen3-1.7B), the same prompt and the same
label tokens, but the assistant answers immediately ("Answer:" with an empty rationale) and is
fine-tuned with cross-entropy on the label token -- i.e. the RLCD model with an empty rationale,
trained by SFT instead of RL. Afterwards a temperature is fitted on dev with ONE annotator label per dev
item (fit_temperature, same protocol as every other arm).
  --labels single : ONE fixed label per train item (same draw as train.py --fixed-label)
  --labels fresh  : a fresh label per visit, the exact (item, label) stream the RL runs see
                    for the same seed and --items-per-step

  python -m openjev_rlcd.train_direct --task mmlupro --labels fresh --steps 400 --output results/x.json
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch

from .metrics import fit_temperature, readout_metrics
from .tasks import TASKS, prompt
from .train import DEVICE, MARKER, Rationales, draw


def logits_of(u, ks):
    """log u with the options beyond each row's count at -inf (so temperature scaling cannot revive them)."""
    z = u.clamp_min(1e-12).log()
    return z.masked_fill(torch.arange(u.shape[1])[None] >= torch.tensor(ks)[:, None], float("-inf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), default="chaosnli")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--labels", choices=["single", "fresh"], required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--items-per-step", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--dev-cap", type=int, default=0, help="0 = whole split")
    ap.add_argument("--test-cap", type=int, default=0, help="0 = whole split")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    rat = Rationales(tok, args.task)
    answer = tok.encode(MARKER, add_special_tokens=False)
    splits = TASKS[args.task]["load"]()
    for name, cap in [("dev", args.dev_cap), ("test", args.test_cap)]:
        splits[name] = splits[name][:cap or None]
    fixed = {it["uid"]: draw(random.Random(f"{args.seed}-{it['uid']}"), it) for it in splits["train"]}
    cache = {}

    def group(items):
        for it in items:
            if it["uid"] not in cache:
                cache[it["uid"]] = tok(prompt(tok, it, args.task))["input_ids"]
        return [(cache[it["uid"]], answer, False) for it in items]

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa").to(DEVICE)
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    @torch.no_grad()
    def predict(items):
        """u [N, K] (zero beyond each item's option count) and the per-item counts."""
        model.eval()
        ks = [rat.k(it) for it in items]
        u = torch.cat([rat.forward(model, group(items[s:s + 16]), grad=False, k=ks[s:s + 16])[0]
                       for s in range(0, len(items), 16)]).cpu()
        return u, ks

    def lists(u, ks):
        return [row[:k] for row, k in zip(u.tolist(), ks)]

    log, t0 = dict(config=vars(args), curve=[]), time.perf_counter()

    def record(step):
        m = readout_metrics(splits["dev"], lists(*predict(splits["dev"])))
        m.pop("per_item_jsd")
        log["curve"].append(dict(step=step, dev=m, minutes=(time.perf_counter() - t0) / 60))
        print(json.dumps(log["curve"][-1]), flush=True)

    record(0)
    for step in range(1, args.steps + 1):
        items = [rng.choice(splits["train"]) for _ in range(args.items_per_step)]
        ys = [draw(rng, it) for it in items]  # the RL runs' stream, same rng order
        if args.labels == "single":
            ys = [fixed[it["uid"]] for it in items]
        model.train()
        u, _ = rat.forward(model, group(items), grad=True, k=[rat.k(it) for it in items])
        loss = -u.clamp_min(1e-12).log()[torch.arange(len(items), device=DEVICE), torch.tensor(ys, device=DEVICE)].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0:
            record(step)

    (test_u, test_k), (dev_u, dev_k) = predict(splits["test"]), predict(splits["dev"])
    t_fit = fit_temperature(list(logits_of(dev_u, dev_k)), splits["dev"], args.seed)
    log["temperature"] = t_fit
    log["test"] = readout_metrics(splits["test"], lists(test_u, test_k))
    log["test_ts"] = readout_metrics(splits["test"], lists((logits_of(test_u, test_k) / t_fit).softmax(-1), test_k))
    log["test_dists"], log["dev_dists"] = lists(test_u, test_k), lists(dev_u, dev_k)
    print(json.dumps({k: {kk: vv for kk, vv in log[k].items() if kk != "per_item_jsd"} for k in ["test", "test_ts"]}
                     | dict(temperature=t_fit)), flush=True)
    Path(args.output).write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
