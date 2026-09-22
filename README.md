# signal-av

MSc AI thesis: cooperative signalling in multi-agent highway driving via MARL.

---

## Setup

```bash
conda env create -f environment.yml
conda activate signal-av
pip install -e .
```
---

## Experiment Overview

Four conditions x 20 seeds each (`1 13 33 42 59 100 200 270 300 400 500 515 600 700 800 900 999 1000 1100 1200`).

| Condition | CLI flags | Iters | Role |
|-----------|-----------|-------|------|
| `bimodal` | `--condition bimodal` | 1600 | Principal condition (RQ1) |
| `same_reward` | `--condition same_reward` | 400 | Rules out apparent signal without reward heterogeneity (RQ2) |
| `random` | `--condition bimodal --random-signal` | 400 | Noise floor: token-type binding severed during training |
| `no_signal` | `--condition bimodal --no-signal` | 1600 | Channel removed entirely; kinematics-only task baseline |

Checkpoints save to `checkpoints/<label>_seed<N>/`. All seed/iteration/hyperparameter
values are defined once in the `Makefile` (`SEEDS`, `ITERS_BIMODAL`, `ITERS_CONTROL`, ...) --
edit them there rather than in individual commands.

---

## Training

Run all seeds for a condition via `make`:

```bash
make train-bimodal
make train-same-reward
make train-random
make train-no-signal
```

Or everything at once: `make train-all`. Single-seed override:

```bash
make train-bimodal-seed SEED=42
```

Add `--device cuda` behaviour via `make train-bimodal DEVICE=cuda` for a GPU speedup.
Extra hyperparameters can be forwarded, e.g.:

```bash
make train-bimodal EXTRA_TRAIN_ARGS="--signal-entropy-end 0.003 --aux-coef 0.5"
```

Training output (printed every 10 iters):

```
iter 0010 | reward=+0.312 | rho=0.041 (pooling) | loss: pi=-0.023 v=0.412 H=1.098 aux=0.693 | dt=18.3s
```

---

## Evaluation

For each checkpoint, run all interventions over 200 episodes:

```bash
make eval-bimodal
make eval-same-reward
make eval-random
make eval-no-signal   # no_signal has no token, so only `baseline` is run
```

Or `make eval-all`. Single-seed override: `make eval-bimodal-seed SEED=42`.

Interventions (`INTERVENTIONS` in the Makefile):

| Name | What changes | What it tests |
|------|-------------|---------------|
| `baseline` | Nothing | Baseline performance |
| `randomise` | Emitted signals replaced with uniform random tokens | Whether receivers condition on signal content |
| `hide` | Signal channel zeroed out in neighbour observations | Whether the signal has any functional role |
| `permute` | Token-to-profile mapping scrambled | Whether signal-profile pairing is load-bearing |
| `hide_kin` | Kinematic features zeroed, signal intact | Whether receivers use kinematics vs. signal |
| `hide_all` | Both kinematics and signal zeroed | Combined removal control |
| `desync` | Token stays correct in neighbour observations, but the sender's actual driving profile is swapped for a random one | Breaks the token<->behaviour binding without changing what receivers see |


Each `results/<name>.json` holds one record per intervention with fields:
`condition`, `rho`, `regime`, `mi_bits`, `mean_return`, `collision_rate`,
`per_agent_collision_rate`, `n_episodes`, `n_steps`, `sc`,
`p_sig_given_type`, `p_type_given_sig`.

---

## Analysis

```bash
make aggregate         # per-condition mean +/- CI table across seeds, printed to terminal
make figures            # scripts/analyze.py -> table + 5 figures in dissertation/figures/
make training-curves    # + rho-over-training panels from intermediate checkpoints.csv (slow)
python scripts/paired_stats.py   # Wilcoxon signed-rank + Holm-Bonferroni across interventions
python scripts/compute_cic.py    # Causal Influence of Communication (Lowe et al., 2019) -> results/cic_<condition>.json
python scripts/compute_cde.py    # Counterfactual Decoding Effect -> results/cde.json
```

---

## Source Layout

```
src/
    config.py     -- experiment constants (profiles, type weights, hyperparams)
    env.py        -- SignallingHighwayEnv: types, profiles, windowed neighbour buffer
    network.py    -- TypeEncoder (GRU), DeepSets, SignalHead, ResponseHead, CentralisedCritic
    rollout.py    -- collect_rollout, RolloutBuffer, compute_gae
    train.py      -- ppo_update, training loop, checkpoint save/load, CLI
    evaluate.py   -- intervention runner, episode loop, CLI
    metrics.py    -- rho, mutual information, CIC, CDE, joint distribution, full_rho_report
scripts/
    analyze.py         -- aggregate results into tables + figures (dissertation/figures/)
    compute_cic.py      -- Causal Influence of Communication
    compute_cde.py      -- Counterfactual Decoding Effect
    paired_stats.py      -- Wilcoxon signed-rank + Holm-Bonferroni across interventions
    plot_curves.py       -- training-curve plots from metrics.csv
```
