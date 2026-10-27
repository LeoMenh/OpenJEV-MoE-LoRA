#!/usr/bin/env python3
"""Final tables for the epistemic tasks (GSM8K-Verify, MMLU-Pro): every arm, all seeds, after temperature
scaling on dev, with paired tests on the test items (per-item values averaged over seeds first).

  python -m openjev_rlcd.final_tables --task gsm8kv|mmlupro [--results results]
Arms (file prefix gv_ / mp_ in the results directory):
  base              zero-shot reasoning model                                   <px>_base_eval*.json
  SFT+TS            direct answer, CE, 400 (and 800) steps                      <px>_direct[800]_lr2e-6_seed*
  GRPO+TS           RLVR from the base model, KL 0.04, 400 steps                <px>_grpokl_seed*
  stage 1           RLCD readout training (pathwise proper score on own rationales), 400 steps  <px>_warm400_seed*
  stage 1 + RLCD-RL two-stage RLCD: + 400 steps REINFORCE on reasoning (sf 0.3, KL at stage 1)  <px>_fork_rl
  stage 1 + readout readout-only continuation (400 more)                        <px>_fork_ro
  stage 1 + anchor  REINFORCE noise + KL anchor, no outcome reward              <px>_fork_anchor
  stage 1 + GRPO    GRPO continuation from the same checkpoint                  <px>_fork_grpo
  stage 1 (log)     stage 1 with the log score instead of Brier                 <px>_ce_seed*
  RFT/STaR+TS       clone correct self-generated trajectories, 400 steps        <px>_rft_seed*  (+KL: _rftkl)
"""
import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

from . import evaluation as S
from .tasks import TASKS

R = Path(__file__).resolve().parents[1] / "results"


def arms(px):
    base = "gv_base_eval_full.json" if px == "gv" else "mp_base_eval.json"
    return [("base", [base], "teacher"),
            ("SFT+TS 400", [f"{px}_direct_lr2e-6_seed{s}.json" for s in (17, 23, 29)], "direct"),
            ("SFT+TS 800", [f"{px}_direct800_lr2e-6_seed{s}.json" for s in (17, 23, 29)], "direct"),
            ("GRPO+TS", [f"{px}_grpokl_seed{s}.json" for s in (17, 23, 29)], "teacher"),
            ("stage 1 (readout)", [f"{px}_warm400_seed{s}.json" for s in (17, 23, 29)], "teacher"),
            ("stage1 + RLCD-RL", [f"{px}_fork_rl_seed{s}.json" for s in (17, 23, 29)], "teacher"),
            ("stage1 + readout", [f"{px}_fork_ro_seed{s}.json" for s in (17, 23, 29)], "teacher"),
            ("stage1 + anchor", [f"{px}_fork_anchor_seed{s}.json" for s in (17, 23, 29)], "teacher"),
            ("stage1 + GRPO", [f"{px}_fork_grpo_seed{s}.json" for s in (17, 23, 29)], "teacher"),
            ("stage 1 (log score)", [f"{px}_ce_seed{s}.json" for s in (17, 23, 29)], "teacher"),
            ("RFT/STaR+TS", [f"{px}_rft_seed{s}.json" for s in (17, 23, 29)], "teacher"),
            ("RFT+KL+TS", [f"{px}_rftkl_seed{s}.json" for s in (17, 23, 29)], "teacher")]


def set_results(path):
    global R
    R = Path(path)


def load_runs(files, kind, readout):
    out = []
    for f in files:
        p = R / f
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        if kind == "direct":
            out.append(S.scored(d["test_dists"], d["dev_dists"], d["config"]["seed"]))
        elif "dev_dists" in d and "dists" in d.get("test", {}):
            out.append(S.scored(d["test"]["dists"][readout], d["dev_dists"][readout], d["config"]["seed"]))
    return out


def pack(runs):
    ts = [r[1] for r in runs]
    return dict(brier=np.mean([t["per_item_brier"] for t in ts], 0), correct=np.mean([t["per_item_correct"] for t in ts], 0),
                seeds=[(np.array(t["per_item_conf"]), 1 - np.array(t["per_item_correct"], float)) for t in ts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["gsm8kv", "mmlupro"], required=True)
    ap.add_argument("--results", default=str(R), help="directory with the run files")
    args = ap.parse_args()
    task = args.task
    set_results(args.results)
    px = {"gsm8kv": "gv", "mmlupro": "mp"}[task]
    S.SPLITS = TASKS[task]["load"]()
    table = {}
    print(f"=== {task} test, after TS (mean±sd over seeds; raw T = fitted temperature) ===")
    for name, files, kind in arms(px):
        for ro in (["single"] if kind == "direct" else ["single", "mixture"]):
            runs = load_runs(files, kind, ro)
            if not runs:
                continue
            key = f"{name} [{'one' if ro == 'single' else 'mix'}]"
            table[key] = pack(runs)
            cell = lambda k: f"{st.mean(r[1][k] for r in runs):.3f}" + (f"±{st.pstdev([r[1][k] for r in runs]):.3f}" if len(runs) > 1 else "")
            print(f"{key:26s} n={len(runs)}  acc={cell('acc')}  Brier={cell('brier')}  NLL={cell('nll')}  ECE={cell('ece')}  "
                  f"AURC={cell('aurc')}  oc@.05={cell('oracle_cov@0.05')}  oc@.1={cell('oracle_cov@0.1')}  "
                  f"T={st.mean(r[2] for r in runs):.2f}")
    print("\nPaired on test items (per-item means over seeds): Δ = first − second; Brier/AURC negative = first better")
    pairs = [("stage1 + RLCD-RL", "SFT+TS 400"), ("stage1 + RLCD-RL", "GRPO+TS"), ("stage1 + RLCD-RL", "base"),
             ("stage1 + RLCD-RL", "stage 1 (readout)"), ("stage1 + RLCD-RL", "stage1 + readout"),
             ("stage1 + RLCD-RL", "stage1 + anchor"), ("stage1 + RLCD-RL", "stage1 + GRPO"),
             ("stage 1 (readout)", "SFT+TS 400"), ("stage 1 (readout)", "GRPO+TS"),
             ("stage 1 (readout)", "stage 1 (log score)"), ("stage 1 (readout)", "RFT/STaR+TS"),
             ("stage1 + RLCD-RL", "RFT/STaR+TS"), ("stage 1 (readout)", "RFT+KL+TS"), ("stage1 + RLCD-RL", "RFT+KL+TS")]
    for a, b in pairs:
        for ro in ["one", "mix"]:
            ka, kb = f"{a} [{ro}]", (f"{b} [one]" if b.startswith("SFT") else f"{b} [{ro}]")
            if ka not in table or kb not in table or len(table[ka]["brier"]) != len(table[kb]["brier"]):
                continue
            x, y = table[ka], table[kb]
            db, da, du = S.paired(x["brier"], y["brier"]), S.paired(x["correct"], y["correct"]), S.paired_aurc(x, y)
            print(f"  {ka:26s} vs {kb:24s}: dBrier {db[0]:+.4f} [{db[1]:+.4f},{db[2]:+.4f}] p={db[3]:.1e}  "
                  f"dAcc {da[0]:+.3f} [{da[1]:+.3f},{da[2]:+.3f}] p={da[3]:.1e}  dAURC {du[0]:+.3f} p~{du[3]:.1e}")


if __name__ == "__main__":
    main()
