# openjev-rlcd

**OpenJev-RLCD: A Working RLCD Implementation** ([arXiv:2609.38850](https://arxiv.org/abs/2609.38850))

Code, run scripts and per-item results for the paper: reinforcement learning for calibrated decisions
(RLCD) for reasoning language models.

The model samples a rationale, and we **read** the answer distribution `u(r)` it commits to after
`Answer:`. Each rationale is scored by a strictly proper scoring rule of that distribution. The
recipe is **calibrate, then reinforce**:

1. **Stage 1 (calibrate the decision).** Train the post-rationale answer distribution with a proper
   score on the model's own on-policy rationales. This stage uses only the pathwise gradient.
2. **Stage 2 (reinforce the reasoning).** Starting from the stage-1 model, apply REINFORCE to the
   rationales, using the same proper score of the model's own readout as the reward. The
   score-function weight is 0.3, with leave-one-out baselines and a KL anchor (β = 0.04) at the
   stage-1 model.

Results with Qwen3-1.7B (3 seeds, single query, temperature-scaled on dev):

| | GSM8K-Verify acc / Brier / AURC / Cov@5% | MMLU-Pro acc / Brier / AURC / Cov@20% |
|---|---|---|
| SFT + TS | 70.4 / 0.382 / 0.174 / 6% | 42.3 / 0.711 / 0.386 / 16% |
| RFT/STaR + KL + TS | 91.4 / 0.160 / 0.075 / 12% | 40.9 / 0.761 / 0.493 / 0% |
| GRPO + KL + TS | 91.8 / 0.150 / 0.061 / 19% | 42.4 / 0.753 / 0.481 / 0% |
| **RLCD two-stage** | **92.2 / 0.135 / 0.034 / 81%** | **47.7 / 0.658 / 0.304 / 28%** |

The paper also covers the negative results: the mixture objective rewards disagreeing
rationales, the per-rationale objective switches reasoning off without a KL anchor,
policy-gradient dilution, and aleatoric tasks where RL cannot beat cross-entropy.

## Layout

```
openjev_rlcd/
  estimators.py      mixture (Rao-Blackwellized) and per-rationale estimators, lambda-interpolation
  train.py           RLCD stage 1 / stage 2, GRPO, RFT/STaR -- one training loop (python -m openjev_rlcd.train)
  train_direct.py    SFT baseline: direct answer, cross-entropy
  metrics.py         temperature scaling, ECE, AURC, coverage at a risk budget
  evaluation.py      per-item scoring after TS, paired bootstrap / sign-flip tests
  final_tables.py    all arms of a task, pooled paired tests (python -m openjev_rlcd.final_tables)
  tasks/             chaosnli, mmlu_pro, gsm8k_verify
tests/               exact-enumeration unbiasedness checks of every estimator
scripts/             one script per group of runs in the paper, a smoke test, the table builder
results/             the 82 result files behind every table and figure (per-item dev/test distributions)
data/gsm8k_verify.json   the GSM8K-Verify dataset (GSM8K + Qwen3-0.6B proposed answers)
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # exact versions used for the paper
# or: pip install -e ".[analysis,test]"
python -m pytest tests                 # estimator unbiasedness, CPU, a few seconds
```

The experiments use one 48 GB GPU per run (the paper used RTX 6000 Ada). MMLU-Pro, ChaosNLI, GSM8K and Qwen3 are downloaded from
the Hugging Face Hub on first use.

## Reproduce the paper

Rebuild every table from the released results. This needs no GPU.

```bash
bash scripts/make_tables.sh            # all arms of both tasks, 3 seeds, after TS, with pooled paired tests
```

Rerun the experiments. Each run writes `runs/<name>.json` and `runs/<name>.log`, and finished runs
are skipped:

```bash
bash scripts/smoke_test.sh                                     # every training mode, 2 steps (~5 min)
CUDA_VISIBLE_DEVICES=0 bash scripts/two_stage.sh gsm8kv        # stage 1 + four forked stage-2 arms, 3 seeds
CUDA_VISIBLE_DEVICES=1 bash scripts/two_stage.sh mmlupro
CUDA_VISIBLE_DEVICES=0 bash scripts/baselines.sh gsm8kv        # base, SFT, GRPO, RFT(+KL), log-score stage 1
CUDA_VISIBLE_DEVICES=1 bash scripts/baselines.sh mmlupro
CUDA_VISIBLE_DEVICES=0 bash scripts/ablations.sh               # Figure 1, Table 3 (ChaosNLI), Table 4
RESULTS=runs bash scripts/make_tables.sh
```

On an RTX 6000 Ada, a 400-step run takes about 1–1.5 h plus the test evaluation. Sampling on GPUs is not bitwise
deterministic, so reruns match the paper up to seed-level noise. The paper's tests pool three
seeds for this reason.

Single runs:

```bash
# stage 1: calibrate the readout (pathwise proper score only)
python -m openjev_rlcd.train --task mmlupro --m 4 --m-eval 4 --max-new-tokens 256 --steps 400 \
  --dev-cap 300 --test-cap 1000 --diversity-weight 0 --kl-beta 0.04 --no-score-function \
  --seed 17 --save-checkpoint runs/stage1.pt --output runs/stage1.json
# stage 2: reinforce the reasoning with the same proper score, anchored at stage 1
python -m openjev_rlcd.train --task mmlupro --m 4 --m-eval 4 --max-new-tokens 256 --steps 400 \
  --dev-cap 300 --test-cap 1000 --diversity-weight 0 --kl-beta 0.04 --sf-coef 0.3 \
  --init-checkpoint runs/stage1.pt --seed 1017 --output runs/stage2.json
```

`python -m openjev_rlcd.train --help` lists every option, including the ablations (`--sf-reader`,
`--grad-diag`, `--readout-score log`, `--diversity-weight`, `--accuracy-bonus`).

## Result files

Each training run writes a JSON with `config` (all arguments), `curve` (dev metrics during training),
`dev_dists` / `test.dists` (per-item answer distributions for the `mixture` and `single` readouts),
and `test` (raw metrics and rationale statistics). All reported metrics are recomputed from the
per-item distributions after temperature scaling on dev (`openjev_rlcd/evaluation.py`). File name
prefixes: `gv_` = GSM8K-Verify, `mp_` = MMLU-Pro, `teacher*` / `direct_fresh*` = ChaosNLI.

## Citation

```bibtex
@misc{gao2026openjevrlcd,
  title         = {{OpenJev-RLCD}: A Working {RLCD} Implementation},
  author        = {Gao, Zhimin and Wang, Pichao},
  year          = {2026},
  eprint        = {2609.38850},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2609.38850},
}
```

## License

Code: [MIT](LICENSE). Datasets keep their own licenses: GSM8K (MIT), MMLU-Pro (MIT), ChaosNLI (CC BY-NC 4.0); `data/gsm8k_verify.json` is derived from GSM8K.

Jev is a product of TypeSafe AI. This repository is an independent implementation, and makes no
claim about Jev's internal training algorithm.
