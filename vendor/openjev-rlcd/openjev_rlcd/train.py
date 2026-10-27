#!/usr/bin/env python3
"""RLCD for a reasoning model, and the RL baselines, in one training loop (see estimators.py).

Only the rationale is sampled. Generation stops at "Answer:"; the model's explicit next-token
distribution over the answer-option tokens is then READ, not sampled: u_i = softmax(logits[option ids])
(options beyond an item's own count are masked). If a rationale ends without the marker (EOS or length
limit), "\\nAnswer:" is appended before reading u, so there are no invalid answers by construction.

Per question: M rationales and one outcome Y (ChaosNLI: a fresh annotator; MMLU-Pro / GSM8K-Verify: the
answer key). Objectives (--objective):
  rlcd : estimators.lambda_surrogate -- exact (pathwise) gradient through u plus REINFORCE for the
         rationales with leave-one-out baselines. --diversity-weight lambda: 1 = Brier of the mixture
         p = E_r u (pays a bonus Var_r(u) for disagreeing rationales), 0 = per-rationale score
         (optimum u(r) = q for every rationale). The paper uses lambda = 0 and
           stage 1 (calibrate):  --no-score-function                  (pathwise only)
           stage 2 (reinforce):  --init-checkpoint <stage 1> --sf-coef 0.3 --kl-beta 0.04
  grpo : RLVR -- answer A_i ~ u_i sampled, reward 1[A_i = Y], group-normalized advantage.
  rft  : STaR / rejection-sampling fine-tuning on the trajectories whose sampled answer is correct.
Readouts at evaluation:
  mixture : mean of u_i over M rationales
  vote    : frequency of argmax u_i
  single  : u_1 from ONE rationale (a single query)

  python -m openjev_rlcd.train --task gsm8kv ... --output results/x.json   (see scripts/ for every run)
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from .estimators import lambda_surrogate
from .metrics import readout_metrics
from .tasks import TASKS, prompt

MARKER = "Answer:"
DEVICE = os.environ.get("OPENJEV_DEVICE", "cuda")  # "cpu" only for smoke tests


def _token_logp(lm_head, h, tgt):
    return lm_head(h).float().log_softmax(-1).gather(-1, tgt[:, None]).squeeze(-1)


class Rationales:
    def __init__(self, tok, task="chaosnli"):
        self.tok, self.task = tok, task
        self.label_ids = [tok.encode(" " + lab, add_special_tokens=False)[0] for lab in TASKS[task]["labels"]]
        assert len(set(self.label_ids)) == len(self.label_ids)
        self.eos = {tok.eos_token_id, tok.pad_token_id}
        self.suffix = tok.encode("\n" + MARKER, add_special_tokens=False)

    def k(self, item):
        return TASKS[self.task]["k"](item)

    @torch.no_grad()
    def sample(self, gen_model, items, m, max_new_tokens, micro=64):
        """Per item, m tuples (prompt_ids, rationale_ids, forced: bool)."""
        gen_model.eval()
        reqs = [(i, prompt(self.tok, it, self.task)) for i, it in enumerate(items) for _ in range(m)]
        out = [[] for _ in items]
        for s in range(0, len(reqs), micro):
            chunk = reqs[s:s + micro]
            enc = self.tok([p for _, p in chunk], return_tensors="pt", padding=True, padding_side="left").to(DEVICE)
            gen = gen_model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                     max_new_tokens=max_new_tokens, stop_strings=[MARKER], tokenizer=self.tok,
                                     pad_token_id=self.tok.pad_token_id)
            width = enc["input_ids"].shape[1]
            for (i, _), row, prow in zip(chunk, gen, enc["input_ids"]):
                comp = row[width:].tolist()
                cut = next((k for k, t in enumerate(comp) if t in self.eos), None)
                head = comp if cut is None else comp[:cut]
                if self.tok.decode(head).rstrip().endswith(MARKER):
                    rat, forced = head, False           # stopped at the marker
                else:
                    rat, forced = (comp if cut is None else comp[:cut + 1]), True  # keep the sampled EOS
                p_ids = prow[prow != self.tok.pad_token_id].tolist()
                out[i].append((p_ids, rat, forced))
        return out

    def forward(self, model, group, grad, need_logp=None, k=None, chunk=512):
        """group: list of (prompt_ids, rationale_ids, forced). Returns u [M, K] (zero beyond each
        row's option count k: int or per-row list) and logp_r [M] (None unless grad or need_logp).
        Vocabulary logits are computed only at the answer position and, in checkpointed chunks,
        at the rationale positions -- never over the prompt."""
        need_logp = grad if need_logp is None else need_logp
        seqs, spans = [], []
        for p, r, forced in group:
            ctx = p + r + (self.suffix if forced else [])
            seqs.append(ctx)
            spans.append((len(p), len(p) + len(r)))
        width = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), width), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for j, x in enumerate(seqs):
            ids[j, :len(x)] = torch.tensor(x)
            mask[j, :len(x)] = 1
        ids, mask = ids.to(DEVICE), mask.to(DEVICE)
        rows = torch.arange(len(seqs), device=DEVICE)
        last = torch.tensor([len(x) - 1 for x in seqs], device=DEVICE)
        with torch.set_grad_enabled(grad), torch.autocast(DEVICE, dtype=torch.bfloat16):
            h = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            z = model.lm_head(h[rows, last])[:, self.label_ids].float()
            if k is not None:
                ks = torch.tensor([k] * len(seqs) if isinstance(k, int) else k, device=DEVICE)
                z = z.masked_fill(torch.arange(z.shape[1], device=DEVICE)[None] >= ks[:, None], float("-inf"))
            u = z.softmax(-1)
            if not need_logp:
                return u, None
            pos = torch.arange(width - 1, device=DEVICE)[None]
            start = torch.tensor([a for a, _ in spans], device=DEVICE)[:, None]
            end = torch.tensor([b for _, b in spans], device=DEVICE)[:, None]
            b, t = ((pos >= start - 1) & (pos < end - 1)).nonzero(as_tuple=True)  # logits at t predict token t+1
            tgt = ids[b, t + 1]
            parts = []
            for s in range(0, len(b), chunk):
                hs = h[b[s:s + chunk], t[s:s + chunk]]
                if grad:
                    parts.append(checkpoint(_token_logp, model.lm_head, hs, tgt[s:s + chunk], use_reentrant=False))
                else:
                    parts.append(_token_logp(model.lm_head, hs, tgt[s:s + chunk]))
            lp = torch.cat(parts) if parts else torch.zeros(0, device=DEVICE)
        return u, torch.zeros(len(seqs), device=DEVICE).index_add(0, b, lp)


def repetition(text):
    words = text.split()
    return 1 - len(set(words)) / max(1, len(words))


@torch.no_grad()
def decision_dists(model, rat, items, m, max_new_tokens, stats=None, examples=None):
    groups = rat.sample(model, items, m, max_new_tokens)
    mix, vote, single, forced = [], [], [], 0
    if stats is not None:
        texts = [rat.tok.decode(r) for g in groups for _, r, _ in g]
        stats.update(mean_rationale_tokens=sum(len(r) for g in groups for _, r, _ in g) / len(texts),
                     length_limit_rate=sum(len(r) >= max_new_tokens for g in groups for _, r, _ in g) / len(texts),
                     repetition_ratio=sum(map(repetition, texts)) / len(texts))
    for it, g in zip(items, groups):
        k = rat.k(it)
        u = torch.cat([rat.forward(model, g[s:s + 16], grad=False, k=k)[0] for s in range(0, len(g), 16)]).cpu()[:, :k]
        mix.append(u.mean(0).tolist())
        vote.append((torch.bincount(u.argmax(1), minlength=k).float() / len(u)).tolist())
        single.append(u[0].tolist())
        forced += sum(f for _, _, f in g)
        if examples is not None and len(examples) < 6:
            examples += [dict(uid=it["uid"], q=it["q"], rationale=rat.tok.decode(r),
                              u=[round(x, 3) for x in uu.tolist()]) for (_, r, _), uu in zip(g[:2], u[:2])]
    return mix, vote, single, forced / (m * len(items))


def evaluate(model, rat, items, m, max_new_tokens, keep_dists=False):
    stats, examples = {}, []
    mix, vote, single, forced = decision_dists(model, rat, items, m, max_new_tokens, stats, examples)
    out = dict(mixture=readout_metrics(items, mix), vote=readout_metrics(items, vote),
               single=readout_metrics(items, single), forced_marker_rate=forced, rationale=stats, examples=examples)
    if keep_dists:
        out["dists"] = dict(mixture=mix, single=single)
    return out


def strip(m):
    return {k: {kk: vv for kk, vv in v.items() if kk != "per_item_jsd"} if isinstance(v, dict) else v for k, v in m.items()}


def draw(rng, item):
    """One outcome: a random annotator (ChaosNLI) or the answer key (MMLU-Pro: counts are one-hot)."""
    return rng.choices(range(len(item["counts"])), weights=item["counts"])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), default="chaosnli")
    ap.add_argument("--objective", choices=["rlcd", "grpo", "rft"], default="rlcd",
                    help="rft = STaR / rejection-sampling fine-tuning: behaviour cloning (rationale + answer token) of "
                         "this step's sampled trajectories whose sampled answer is correct; no baseline, no KL")
    ap.add_argument("--readout-score", choices=["brier", "log"], default="brier",
                    help="proper score for the readout-only objective (with --no-score-function): 'log' = CE on the "
                         "answer distribution read after the model's own rationale")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--items-per-step", type=int, default=8)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--m-eval", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--dev-cap", type=int, default=199)
    ap.add_argument("--test-cap", type=int, default=400)
    ap.add_argument("--kl-beta", type=float, default=0.0, help="KL(pi || pi_ref) on rationales, via reward shaping")
    ap.add_argument("--length-penalty", type=float, default=0.0, help="penalty for rationales that never write the marker")
    ap.add_argument("--diversity-weight", type=float, default=1.0,
                    help="lambda in J_single + lambda Var_r(u): 1 = mixture objective, 0 = per-rationale objective")
    ap.add_argument("--fixed-label", action="store_true",
                    help="ONE fixed annotator label per train item instead of a fresh draw per visit (ChaosNLI)")
    ap.add_argument("--no-score-function", action="store_true",
                    help="ablation: drop the REINFORCE term for rationales (answer-head/pathwise gradient only)")
    ap.add_argument("--sf-coef", type=float, default=1.0,
                    help="scale of the REINFORCE (score-function) term for the rationales, KL shaping included; "
                         "1 = unbiased, 0 = --no-score-function. Under Adam a noisy term dilutes the pathwise signal")
    ap.add_argument("--sf-reader", choices=["self", "ref", "none"], default="self",
                    help="whose answer distribution scores a rationale in the REINFORCE term: the policy's own readout "
                         "(unbiased gradient of E_r S(u_r, Y)) or the frozen reference model's (a calibrated external "
                         "reader after --pathwise-warmup: rewards informative reasoning, not self-assertive reasoning); "
                         "the policy's readout is always trained on its own u by the pathwise term; 'none' = control: "
                         "the REINFORCE term carries only the shaping (KL anchor), no outcome signal")
    ap.add_argument("--grad-diag", type=int, default=0,
                    help="every N steps log the pathwise vs score-function gradient norms (first item, 3 MLP weights)")
    ap.add_argument("--accuracy-bonus", type=float, default=0.0,
                    help="alpha: per-rationale reward J(u_i) + alpha 1[argmax u_i = Y]. The optimum is still u(r) = q "
                         "(argmax q maximizes the bonus), but the reasoning gets a correctness signal (RLCR-style)")
    ap.add_argument("--pathwise-warmup", type=int, default=0,
                    help="first N steps without the REINFORCE term: calibrate the readout after the model's own "
                         "reasoning before optimizing the reasoning itself. With --kl-beta the reference is reset to "
                         "the model at step N, so the KL anchors the RL phase at the calibrated readout (the warm-up "
                         "drifts far from the initial model because the anchor only acts through REINFORCE)")
    ap.add_argument("--save-checkpoint", default="")
    ap.add_argument("--init-checkpoint", default="",
                    help="start training from this saved (bf16) teacher; the KL reference is set to it as well, so "
                         "several continuations fork from an identical point (fresh optimizer state for all)")
    ap.add_argument("--checkpoint", default="", help="weights for --eval-only")
    ap.add_argument("--eval-only", action="store_true",
                    help="re-evaluate --checkpoint (or the base model) on dev + test with the current metrics")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    rat = Rationales(tok, args.task)
    load = TASKS[args.task]["load"]

    if args.eval_only:  # same bf16 weights the training loop samples from and saves
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEVICE)
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
        splits = load()
        d = evaluate(model, rat, splits["dev"][:args.dev_cap], args.m_eval, args.max_new_tokens, keep_dists=True)
        t = evaluate(model, rat, splits["test"][:args.test_cap], args.m_eval, args.max_new_tokens, keep_dists=True)
        Path(args.output).write_text(json.dumps(dict(config=vars(args), dev_dists=d.pop("dists"), dev=d, test=t), indent=2))
        print(json.dumps({k: v for k, v in strip(t).items() if k not in ("dists", "examples")}), flush=True)
        return

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa").to(DEVICE)
    model.gradient_checkpointing_enable()
    gen_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEVICE)
    gp, tp = list(gen_model.parameters()), list(model.parameters())
    assert [p.shape for p in gp] == [p.shape for p in tp]

    ref_model = None
    if args.kl_beta > 0:  # frozen initial policy for the KL regularizer
        ref_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEVICE)
        ref_model.eval().requires_grad_(False)
    if args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location=DEVICE)
        for net in [model, gen_model] + ([ref_model] if ref_model is not None else []):
            net.load_state_dict(state)
        del state

    def sync():
        with torch.no_grad():
            for g, p in zip(gp, tp):
                g.copy_(p)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    splits = load()
    dev, test = splits["dev"][:args.dev_cap], splits["test"][:args.test_cap]
    log, t0 = dict(config=vars(args), curve=[]), time.perf_counter()

    def record(step):
        m = strip(evaluate(gen_model, rat, dev, args.m_eval, args.max_new_tokens, keep_dists=step == args.steps))
        if "dists" in m:  # final dev distributions: lets every arm be temperature-scaled on dev afterwards
            log["dev_dists"] = m.pop("dists")
        for ex in m["examples"][:2]:
            print(json.dumps(ex)[:600], flush=True)
        log["curve"].append(dict(step=step, dev=m, minutes=(time.perf_counter() - t0) / 60))
        print(json.dumps(log["curve"][-1]), flush=True)
        Path(args.output).write_text(json.dumps(log, indent=2))

    fixed = {it["uid"]: draw(random.Random(f"{args.seed}-{it['uid']}"), it) for it in splits["train"]}
    record(0)
    for step in range(1, args.steps + 1):
        if ref_model is not None and args.pathwise_warmup and step == args.pathwise_warmup + 1:
            ref_model.load_state_dict(gen_model.state_dict())
            print(json.dumps(dict(step=step, event="reference reset to the warmed-up model")), flush=True)
        items = [rng.choice(splits["train"]) for _ in range(args.items_per_step)]
        groups = rat.sample(gen_model, items, args.m, args.max_new_tokens)
        model.train()
        opt.zero_grad(set_to_none=True)
        hit, forced, kl_sum, umax, var = 0.0, 0, 0.0, 0.0, 0.0
        for it, g in zip(items, groups):
            y = draw(rng, it)  # one fresh outcome (always drawn: same item order across arms)
            if args.fixed_label:
                y = fixed[it["uid"]]
            k = rat.k(it)
            u, logp_r = rat.forward(model, g, grad=True, k=k)
            u = u[:, :k]
            if args.no_score_function or step <= args.pathwise_warmup:
                logp_r = logp_r.detach()
            m = len(g)
            extra = torch.zeros(m, device=DEVICE)  # per-rationale shaping terms (KL anchor, length penalty)
            u_ref = None
            if ref_model is not None:
                with torch.no_grad():
                    u_ref, ref_lp = rat.forward(ref_model, g, grad=False, need_logp=True, k=k)
                u_ref = u_ref[:, :k]
                logratio = logp_r.detach() - ref_lp
                extra -= (args.kl_beta / m) * logratio
                kl_sum += logratio.mean().item()
            if args.length_penalty:
                extra -= (args.length_penalty / m) * torch.tensor([float(f) for _, _, f in g], device=DEVICE)
            if args.accuracy_bonus:  # depends on r_i only through argmax u_i: score-function part only
                extra += (args.accuracy_bonus / m) * (u.detach().argmax(1) == y).float()
            logp_sf = logp_r * args.sf_coef if args.sf_coef != 1.0 else logp_r
            if args.objective == "rft":
                a = torch.multinomial(u.detach(), 1).squeeze(1)
                keep = (a == y).float()  # clone only the trajectories whose sampled answer is right
                loss = -(keep * (logp_r + u[:, y].clamp_min(1e-12).log())).sum() / m
                if ref_model is not None:  # optional KL anchor (--kl-beta), same shaping as the other arms
                    reg = extra - (extra.sum() - extra) / (m - 1)
                    loss = loss - (reg * logp_r).sum()
            elif args.readout_score == "log":  # readout-only with the log score (pathwise; no REINFORCE)
                assert args.no_score_function, "--readout-score log is only defined for the readout-only objective"
                loss = -u[:, y].clamp_min(1e-12).log().mean()
            elif args.objective == "grpo":
                a = torch.multinomial(u.detach(), 1).squeeze(1)
                r = (a == y).float()
                adv = (r - r.mean()) / (r.std(unbiased=False) + 1e-4)
                reg = extra - (extra.sum() - extra) / (m - 1)  # same shaping, leave-one-out baseline (unbiased)
                loss = (-(adv * (logp_sf + u[torch.arange(m, device=DEVICE), a].clamp_min(1e-12).log())).mean()
                        - (reg * logp_sf).sum())
            elif args.sf_reader == "none":  # anchor-only control: equal rewards -> only the shaping terms remain
                loss = (lambda_surrogate(u, logp_sf.detach(), y, args.diversity_weight)
                        + lambda_surrogate(torch.full_like(u, 1.0 / k), logp_sf, y, args.diversity_weight, extra_local=extra))
            elif args.sf_reader == "ref" and u_ref is not None:  # pathwise on own u; rationales scored by the reader
                loss = (lambda_surrogate(u, logp_sf.detach(), y, args.diversity_weight)
                        + lambda_surrogate(u_ref, logp_sf, y, args.diversity_weight, extra_local=extra))
            else:
                loss = lambda_surrogate(u, logp_sf, y, args.diversity_weight, extra_local=extra)
                if args.grad_diag and step % args.grad_diag == 0 and it is items[0] and logp_sf.requires_grad:
                    layers = model.model.layers
                    sub = [layers[i].mlp.down_proj.weight for i in (0, len(layers) // 2, len(layers) - 1)]
                    d_path = torch.autograd.grad(lambda_surrogate(u, logp_sf.detach(), y, args.diversity_weight, extra),
                                                 sub, retain_graph=True, allow_unused=True)
                    d_sf = torch.autograd.grad(lambda_surrogate(u.detach(), logp_sf, y, args.diversity_weight, extra),
                                               sub, retain_graph=True, allow_unused=True)
                    vp = torch.cat([(d if d is not None else torch.zeros_like(w)).flatten() for d, w in zip(d_path, sub)])
                    vs = torch.cat([(d if d is not None else torch.zeros_like(w)).flatten() for d, w in zip(d_sf, sub)])
                    print(json.dumps(dict(step=step, diag_path_norm=vp.norm().item(), diag_sf_norm=vs.norm().item(),
                                          diag_ratio=(vs.norm() / vp.norm().clamp_min(1e-12)).item(),
                                          diag_cos=torch.nn.functional.cosine_similarity(vp, vs, dim=0).item())),
                          flush=True)
            (loss / len(items)).backward()
            hit += u[:, y].mean().item()
            ud = u.detach()
            umax += ud.max(1).values.mean().item()
            var += ((ud - ud.mean(0)) ** 2).sum(1).mean().item()
            forced += sum(f for _, _, f in g)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sync()
        if step % 10 == 0:
            print(json.dumps(dict(step=step, mean_p_of_outcome=hit / len(items),
                                  forced_marker_rate=forced / (args.m * len(items)), kl_est=kl_sum / len(items),
                                  mean_u_max=umax / len(items), var_u=var / len(items),
                                  minutes=(time.perf_counter() - t0) / 60)), flush=True)
        if step % args.eval_every == 0:
            record(step)
    log["test"] = evaluate(gen_model, rat, test, args.m_eval, args.max_new_tokens, keep_dists=True)
    print(json.dumps({k: v for k, v in strip(log["test"]).items() if k not in ("dists", "examples")}), flush=True)
    Path(args.output).write_text(json.dumps(log, indent=2))
    if args.save_checkpoint:
        torch.save(gen_model.state_dict(), args.save_checkpoint)


if __name__ == "__main__":
    main()
