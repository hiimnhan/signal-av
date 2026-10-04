
# Use the project venv when `make install-gpu` has created it.
VENV         := .venv
PYTHON       := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python)
DEVICE       := auto

# Parallel seeds: make train-silent JOBS=16 (one process per seed; per-seed logs in LOG_DIR).
JOBS         := 16
LOG_DIR      := logs
# FORCE=1 retrains / re-evaluates even when final.pt or the result JSON exists.
FORCE        :=
FORCE_FLAG   := $(if $(FORCE),--force,)
# One BLAS/OpenMP thread per process, so parallel seeds don't oversubscribe the CPU.
export OMP_NUM_THREADS ?= 1
export MKL_NUM_THREADS ?= 1

# Seeds used in all conditions (must match across train + eval for comparability)
# 1 13 33 42 59 270 515 999
SEEDS := 1 13 33 42 59 270 515 999 100 200 300 400 500 600 700 800 900 1000 1100 1200

# Iteration counts per condition (proposal sec.3.4: longer for bimodal because
# credit-assignment difficulty is expected to be the principal challenge)
ITERS_BIMODAL := 1600
ITERS_CONTROL := 400   # same_reward and random-signal; expected regimes resolve quickly

# Extra args forwarded to src.train for hyperparameter sweeps.
# Example: make train-bimodal EXTRA_TRAIN_ARGS="--signal-entropy-end 0.003 --aux-coef 0.5"
EXTRA_TRAIN_ARGS :=

# Single-seed override: make train-bimodal-seed SEED=42
SEED := 1

# Checkpoint and result directories
CKPT_DIR     := checkpoints
RESULT_DIR   := results

# Evaluation settings (proposal sec.4.2)
EVAL_EPISODES := 200
EVAL_SEED     := 9999
INTERVENTIONS := baseline randomise hide permute hide_kin hide_all desync

SIGNAL_ENTROPY_END   := 0.003
ENTROPY_ANNEAL_ITERS := 400
CRASH_SCALE          := 1.0

N_BG_VEHICLES    := 0
CURRICULUM_ITERS := 0





.PHONY: all install-gpu \
        train-all train-bimodal train-same-reward train-random train-no-signal train-silent \
        train-bimodal-seed train-same-reward-seed train-random-seed train-no-signal-seed train-silent-seed \
        eval-all eval-bimodal eval-same-reward eval-random eval-no-signal eval-silent \
        eval-bimodal-seed eval-same-reward-seed eval-random-seed eval-no-signal-seed eval-silent-seed \
        aggregate aggregate-bimodal aggregate-same-reward aggregate-random aggregate-no-signal aggregate-silent \
        figures analyze training-curves \
        plot-curves-seed plot-curves-bimodal \
        smoke clean-checkpoints clean-results


all: train-all eval-all aggregate

install-gpu:
	@command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: install the NVIDIA driver first"; exit 1; }
	@nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
	@command -v uv >/dev/null || [ -x $$HOME/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh
	@UV=$$(command -v uv || echo $$HOME/.local/bin/uv); \
	$$UV venv --python 3.11 $(VENV) && \
	$$UV pip install --python $(VENV)/bin/python -e .
	$(VENV)/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible to torch: driver too old for this torch build? Update the NVIDIA driver'; print('torch', torch.__version__, '| cuda', torch.version.cuda, '|', torch.cuda.get_device_name(0))"
	@echo "Installed. Next: make smoke DEVICE=cuda"

bimodal-all: train-bimodal eval-bimodal aggregate-bimodal

train-all: train-bimodal train-same-reward train-random train-no-signal train-silent

eval-all: eval-bimodal eval-same-reward eval-random eval-no-signal eval-silent


# =============================================================================
# TRAINING
# =============================================================================

train-bimodal:
	@mkdir -p $(CKPT_DIR)
	@mkdir -p $(LOG_DIR)
	@printf '%s\n' $(SEEDS) | xargs -n 1 -P $(JOBS) sh -c 'seed=$$1; log=$(LOG_DIR)/$@_seed$$seed.log; ( \
		echo ""; \
		echo "=== bimodal seed=$$seed ==="; \
		$(PYTHON) -m src.train $(FORCE_FLAG) \
			--condition bimodal \
			--seed $$seed \
			--n-bg-vehicles $(N_BG_VEHICLES) \
			--curriculum-iters $(CURRICULUM_ITERS) \
			--device $(DEVICE) \
			--save-dir $(CKPT_DIR) \
			$(EXTRA_TRAIN_ARGS); \
	) > $$log 2>&1 && echo "done $@ seed=$$seed" || { echo "FAILED $@ seed=$$seed (see $$log)"; exit 1; }' _

train-same-reward:
	@mkdir -p $(CKPT_DIR)
	@mkdir -p $(LOG_DIR)
	@printf '%s\n' $(SEEDS) | xargs -n 1 -P $(JOBS) sh -c 'seed=$$1; log=$(LOG_DIR)/$@_seed$$seed.log; ( \
		echo ""; \
		echo "=== same_reward seed=$$seed ==="; \
		$(PYTHON) -m src.train $(FORCE_FLAG) \
			--condition same_reward \
			--seed $$seed \
			--iters $(ITERS_CONTROL) \
			--n-bg-vehicles $(N_BG_VEHICLES) \
			--curriculum-iters $(CURRICULUM_ITERS) \
			--signal-entropy-end $(SIGNAL_ENTROPY_END) \
			--entropy-anneal-iters $(ENTROPY_ANNEAL_ITERS) \
			--crash-scale $(CRASH_SCALE) \
			--device $(DEVICE) \
			--save-dir $(CKPT_DIR) \
			$(EXTRA_TRAIN_ARGS); \
	) > $$log 2>&1 && echo "done $@ seed=$$seed" || { echo "FAILED $@ seed=$$seed (see $$log)"; exit 1; }' _

train-random:
	@mkdir -p $(CKPT_DIR)
	@mkdir -p $(LOG_DIR)
	@printf '%s\n' $(SEEDS) | xargs -n 1 -P $(JOBS) sh -c 'seed=$$1; log=$(LOG_DIR)/$@_seed$$seed.log; ( \
		echo ""; \
		echo "=== random-signal seed=$$seed ==="; \
		$(PYTHON) -m src.train $(FORCE_FLAG) \
			--condition bimodal \
			--random-signal \
			--seed $$seed \
			--iters $(ITERS_CONTROL) \
			--n-bg-vehicles $(N_BG_VEHICLES) \
			--curriculum-iters $(CURRICULUM_ITERS) \
			--signal-entropy-end $(SIGNAL_ENTROPY_END) \
			--entropy-anneal-iters $(ENTROPY_ANNEAL_ITERS) \
			--crash-scale $(CRASH_SCALE) \
			--device $(DEVICE) \
			--save-dir $(CKPT_DIR) \
			$(EXTRA_TRAIN_ARGS); \
	) > $$log 2>&1 && echo "done $@ seed=$$seed" || { echo "FAILED $@ seed=$$seed (see $$log)"; exit 1; }' _

train-no-signal:
	@mkdir -p $(CKPT_DIR)
	@mkdir -p $(LOG_DIR)
	@printf '%s\n' $(SEEDS) | xargs -n 1 -P $(JOBS) sh -c 'seed=$$1; log=$(LOG_DIR)/$@_seed$$seed.log; ( \
		echo ""; \
		echo "=== no-signal baseline seed=$$seed ==="; \
		$(PYTHON) -m src.train $(FORCE_FLAG) \
			--condition bimodal \
			--no-signal \
			--seed $$seed \
			--iters $(ITERS_BIMODAL) \
			--n-bg-vehicles $(N_BG_VEHICLES) \
			--curriculum-iters $(CURRICULUM_ITERS) \
			--crash-scale $(CRASH_SCALE) \
			--device $(DEVICE) \
			--save-dir $(CKPT_DIR) \
			$(EXTRA_TRAIN_ARGS); \
	) > $$log 2>&1 && echo "done $@ seed=$$seed" || { echo "FAILED $@ seed=$$seed (see $$log)"; exit 1; }' _


train-silent:
	@mkdir -p $(CKPT_DIR)
	@mkdir -p $(LOG_DIR)
	@printf '%s\n' $(SEEDS) | xargs -n 1 -P $(JOBS) sh -c 'seed=$$1; log=$(LOG_DIR)/$@_seed$$seed.log; ( \
		echo ""; \
		echo "=== silent seed=$$seed ==="; \
		$(PYTHON) -m src.train $(FORCE_FLAG) \
			--condition bimodal \
			--silent \
			--seed $$seed \
			--iters $(ITERS_BIMODAL) \
			--n-bg-vehicles $(N_BG_VEHICLES) \
			--curriculum-iters $(CURRICULUM_ITERS) \
			--device $(DEVICE) \
			--save-dir $(CKPT_DIR) \
			$(EXTRA_TRAIN_ARGS); \
	) > $$log 2>&1 && echo "done $@ seed=$$seed" || { echo "FAILED $@ seed=$$seed (see $$log)"; exit 1; }' _


# =============================================================================
# SINGLE-SEED TRAINING  (make train-bimodal-seed SEED=42)
# =============================================================================

train-bimodal-seed:
	@mkdir -p $(CKPT_DIR)
	$(PYTHON) -m src.train $(FORCE_FLAG) \
		--condition bimodal \
		--seed $(SEED) \
		--device $(DEVICE) \
		--save-dir $(CKPT_DIR) \
		$(EXTRA_TRAIN_ARGS)

train-same-reward-seed:
	@mkdir -p $(CKPT_DIR)
	$(PYTHON) -m src.train $(FORCE_FLAG) \
		--condition same_reward \
		--seed $(SEED) \
		--iters $(ITERS_CONTROL) \
		--signal-entropy-end $(SIGNAL_ENTROPY_END) \
		--entropy-anneal-iters $(ENTROPY_ANNEAL_ITERS) \
		--crash-scale $(CRASH_SCALE) \
		--device $(DEVICE) \
		--save-dir $(CKPT_DIR) \
		$(EXTRA_TRAIN_ARGS)

train-random-seed:
	@mkdir -p $(CKPT_DIR)
	$(PYTHON) -m src.train $(FORCE_FLAG) \
		--condition bimodal \
		--random-signal \
		--seed $(SEED) \
		--iters $(ITERS_CONTROL) \
		--signal-entropy-end $(SIGNAL_ENTROPY_END) \
		--entropy-anneal-iters $(ENTROPY_ANNEAL_ITERS) \
		--crash-scale $(CRASH_SCALE) \
		--device $(DEVICE) \
		--save-dir $(CKPT_DIR) \
		$(EXTRA_TRAIN_ARGS)

train-no-signal-seed:
	@mkdir -p $(CKPT_DIR)
	$(PYTHON) -m src.train $(FORCE_FLAG) \
		--condition bimodal \
		--no-signal \
		--seed $(SEED) \
		--iters $(ITERS_BIMODAL) \
		--crash-scale $(CRASH_SCALE) \
		--device $(DEVICE) \
		--save-dir $(CKPT_DIR) \
		$(EXTRA_TRAIN_ARGS)
train-silent-seed:
	@mkdir -p $(CKPT_DIR)
	$(PYTHON) -m src.train $(FORCE_FLAG) \
		--condition bimodal \
		--silent \
		--seed $(SEED) \
		--iters $(ITERS_BIMODAL) \
		--device $(DEVICE) \
		--save-dir $(CKPT_DIR) \
		$(EXTRA_TRAIN_ARGS)

# =============================================================================
# EVALUATION
# =============================================================================

eval-bimodal:
	@mkdir -p $(RESULT_DIR)
	@mkdir -p $(LOG_DIR)
	@printf '%s\n' $(SEEDS) | xargs -n 1 -P $(JOBS) sh -c 'seed=$$1; log=$(LOG_DIR)/$@_seed$$seed.log; ( \
		ckpt=$(CKPT_DIR)/bimodal_seed$$seed/final.pt; \
		out=$(RESULT_DIR)/bimodal_seed$$seed.json; \
		echo ""; \
		echo "=== eval bimodal seed=$$seed -> $$out ==="; \
		$(PYTHON) -m src.evaluate $(FORCE_FLAG) \
			--ckpt $$ckpt \
			--interventions $(INTERVENTIONS) \
			--episodes $(EVAL_EPISODES) \
			--seed $(EVAL_SEED) \
			--device $(DEVICE) \
			--out $$out; \
	) > $$log 2>&1 && echo "done $@ seed=$$seed" || { echo "FAILED $@ seed=$$seed (see $$log)"; exit 1; }' _

eval-same-reward:
	@mkdir -p $(RESULT_DIR)
	@mkdir -p $(LOG_DIR)
	@printf '%s\n' $(SEEDS) | xargs -n 1 -P $(JOBS) sh -c 'seed=$$1; log=$(LOG_DIR)/$@_seed$$seed.log; ( \
		ckpt=$(CKPT_DIR)/same_reward_seed$$seed/final.pt; \
		out=$(RESULT_DIR)/same_reward_seed$$seed.json; \
		echo ""; \
		echo "=== eval same_reward seed=$$seed -> $$out ==="; \
		$(PYTHON) -m src.evaluate $(FORCE_FLAG) \
			--ckpt $$ckpt \
			--interventions $(INTERVENTIONS) \
			--episodes $(EVAL_EPISODES) \
			--seed $(EVAL_SEED) \
			--device $(DEVICE) \
			--out $$out; \
	) > $$log 2>&1 && echo "done $@ seed=$$seed" || { echo "FAILED $@ seed=$$seed (see $$log)"; exit 1; }' _

eval-random:
	@mkdir -p $(RESULT_DIR)
	@mkdir -p $(LOG_DIR)
	@printf '%s\n' $(SEEDS) | xargs -n 1 -P $(JOBS) sh -c 'seed=$$1; log=$(LOG_DIR)/$@_seed$$seed.log; ( \
		ckpt=$(CKPT_DIR)/random_seed$$seed/final.pt; \
		out=$(RESULT_DIR)/random_seed$$seed.json; \
		echo ""; \
		echo "=== eval random seed=$$seed -> $$out ==="; \
		$(PYTHON) -m src.evaluate $(FORCE_FLAG) \
			--ckpt $$ckpt \
			--interventions $(INTERVENTIONS) \
			--episodes $(EVAL_EPISODES) \
			--seed $(EVAL_SEED) \
			--device $(DEVICE) \
			--out $$out; \
	) > $$log 2>&1 && echo "done $@ seed=$$seed" || { echo "FAILED $@ seed=$$seed (see $$log)"; exit 1; }' _

eval-no-signal:
	@mkdir -p $(RESULT_DIR)
	@mkdir -p $(LOG_DIR)
	@printf '%s\n' $(SEEDS) | xargs -n 1 -P $(JOBS) sh -c 'seed=$$1; log=$(LOG_DIR)/$@_seed$$seed.log; ( \
		ckpt=$(CKPT_DIR)/no_signal_seed$$seed/final.pt; \
		out=$(RESULT_DIR)/no_signal_seed$$seed.json; \
		echo ""; \
		echo "=== eval no-signal seed=$$seed -> $$out ==="; \
		$(PYTHON) -m src.evaluate $(FORCE_FLAG) \
			--ckpt $$ckpt \
			--interventions baseline \
			--episodes $(EVAL_EPISODES) \
			--seed $(EVAL_SEED) \
			--device $(DEVICE) \
			--out $$out; \
	) > $$log 2>&1 && echo "done $@ seed=$$seed" || { echo "FAILED $@ seed=$$seed (see $$log)"; exit 1; }' _

eval-silent:
	@mkdir -p $(RESULT_DIR)
	@mkdir -p $(LOG_DIR)
	@printf '%s\n' $(SEEDS) | xargs -n 1 -P $(JOBS) sh -c 'seed=$$1; log=$(LOG_DIR)/$@_seed$$seed.log; ( \
		ckpt=$(CKPT_DIR)/silent_seed$$seed/final.pt; \
		out=$(RESULT_DIR)/silent_seed$$seed.json; \
		echo ""; \
		echo "=== eval silent seed=$$seed -> $$out ==="; \
		$(PYTHON) -m src.evaluate $(FORCE_FLAG) \
			--ckpt $$ckpt \
			--interventions $(INTERVENTIONS) \
			--episodes $(EVAL_EPISODES) \
			--seed $(EVAL_SEED) \
			--device $(DEVICE) \
			--out $$out; \
	) > $$log 2>&1 && echo "done $@ seed=$$seed" || { echo "FAILED $@ seed=$$seed (see $$log)"; exit 1; }' _

eval-silent-seed:
	@mkdir -p $(RESULT_DIR)
	$(PYTHON) -m src.evaluate $(FORCE_FLAG) \
		--ckpt $(CKPT_DIR)/silent_seed$(SEED)/final.pt \
		--interventions $(INTERVENTIONS) \
		--episodes $(EVAL_EPISODES) \
		--seed $(EVAL_SEED) \
		--device $(DEVICE) \
		--out $(RESULT_DIR)/silent_seed$(SEED).json

eval-no-signal-seed:
	@mkdir -p $(RESULT_DIR)
	$(PYTHON) -m src.evaluate $(FORCE_FLAG) \
		--ckpt $(CKPT_DIR)/no_signal_seed$(SEED)/final.pt \
		--interventions baseline \
		--episodes $(EVAL_EPISODES) \
		--seed $(EVAL_SEED) \
		--device $(DEVICE) \
		--out $(RESULT_DIR)/no_signal_seed$(SEED).json

eval-bimodal-seed:
	@mkdir -p $(RESULT_DIR)
	$(PYTHON) -m src.evaluate $(FORCE_FLAG) \
		--ckpt $(CKPT_DIR)/bimodal_seed$(SEED)/final.pt \
		--interventions $(INTERVENTIONS) \
		--episodes $(EVAL_EPISODES) \
		--seed $(EVAL_SEED) \
		--device $(DEVICE) \
		--out $(RESULT_DIR)/bimodal_seed$(SEED).json

eval-same-reward-seed:
	@mkdir -p $(RESULT_DIR)
	$(PYTHON) -m src.evaluate $(FORCE_FLAG) \
		--ckpt $(CKPT_DIR)/same_reward_seed$(SEED)/final.pt \
		--interventions $(INTERVENTIONS) \
		--episodes $(EVAL_EPISODES) \
		--seed $(EVAL_SEED) \
		--device $(DEVICE) \
		--out $(RESULT_DIR)/same_reward_seed$(SEED).json

eval-random-seed:
	@mkdir -p $(RESULT_DIR)
	$(PYTHON) -m src.evaluate $(FORCE_FLAG) \
		--ckpt $(CKPT_DIR)/random_seed$(SEED)/final.pt \
		--interventions $(INTERVENTIONS) \
		--episodes $(EVAL_EPISODES) \
		--seed $(EVAL_SEED) \
		--device $(DEVICE) \
		--out $(RESULT_DIR)/random_seed$(SEED).json



# =============================================================================
# AGGREGATION
# =============================================================================

aggregate: aggregate-bimodal aggregate-same-reward aggregate-random aggregate-no-signal aggregate-silent

aggregate-bimodal:
	@echo ""
	@echo "=== bimodal ==="
	@$(PYTHON) -c "\
from pathlib import Path; \
from src.metrics import aggregate_seeds, print_aggregate_table; \
paths = sorted(Path('$(RESULT_DIR)').glob('bimodal_seed*.json')); \
print(f'{len(paths)} seeds found'); \
print_aggregate_table(aggregate_seeds(paths)) if paths else print('no results')"

aggregate-same-reward:
	@echo ""
	@echo "=== same_reward ==="
	@$(PYTHON) -c "\
from pathlib import Path; \
from src.metrics import aggregate_seeds, print_aggregate_table; \
paths = sorted(Path('$(RESULT_DIR)').glob('same_reward_seed*.json')); \
print(f'{len(paths)} seeds found'); \
print_aggregate_table(aggregate_seeds(paths)) if paths else print('no results')"

aggregate-random:
	@echo ""
	@echo "=== random-signal ==="
	@$(PYTHON) -c "\
from pathlib import Path; \
from src.metrics import aggregate_seeds, print_aggregate_table; \
paths = sorted(Path('$(RESULT_DIR)').glob('random_seed*.json')); \
print(f'{len(paths)} seeds found'); \
print_aggregate_table(aggregate_seeds(paths)) if paths else print('no results')"

aggregate-no-signal:
	@echo ""
	@echo "=== no-signal baseline ==="
	@$(PYTHON) -c "\
from pathlib import Path; \
from src.metrics import aggregate_seeds, print_aggregate_table; \
paths = sorted(Path('$(RESULT_DIR)').glob('no_signal_seed*.json')); \
print(f'{len(paths)} seeds found'); \
print_aggregate_table(aggregate_seeds(paths)) if paths else print('no results')"


aggregate-silent:
	@echo ""
	@echo "=== silent (profile choice, no token shown) ==="
	@$(PYTHON) -c "\
from pathlib import Path; \
from src.metrics import aggregate_seeds, print_aggregate_table; \
paths = sorted(Path('$(RESULT_DIR)').glob('silent_seed*.json')); \
print(f'{len(paths)} seeds found'); \
print_aggregate_table(aggregate_seeds(paths)) if paths else print('no results')"


# =============================================================================
# FIGURES
#   make figures / make analyze   run scripts/analyze.py (table + 5 figures)
#   make training-curves          + rho-over-training from intermediate checkpoints (slow)
# =============================================================================

FIGURES_DIR := figures
PLOTS_DIR   := plots
CONDITION   := bimodal

figures: analyze

analyze:
	@mkdir -p $(FIGURES_DIR)
	$(PYTHON) scripts/analyze.py

cic:
	$(PYTHON) scripts/compute_cic.py

training-curves:
	@mkdir -p $(FIGURES_DIR)
	$(PYTHON) scripts/analyze.py --training

# Plot training curves for one seed from its metrics.csv.
# Usage: make plot-curves-seed SEED=1 CONDITION=bimodal
plot-curves-seed:
	@mkdir -p $(PLOTS_DIR)
	$(PYTHON) scripts/plot_curves.py \
		$(CKPT_DIR)/$(CONDITION)_seed$(SEED)/metrics.csv \
		--out $(PLOTS_DIR)/curves_$(CONDITION)_seed$(SEED).png \
		--window 20

# Plot training curves for all bimodal seeds that have a metrics.csv.
plot-curves-bimodal:
	@mkdir -p $(PLOTS_DIR)
	@for seed in $(SEEDS); do \
		f=$(CKPT_DIR)/bimodal_seed$$seed/metrics.csv; \
		[ -f $$f ] || continue; \
		echo "plotting bimodal seed=$$seed"; \
		$(PYTHON) scripts/plot_curves.py $$f \
			--out $(PLOTS_DIR)/curves_bimodal_seed$$seed.png \
			--window 20; \
	done


# =============================================================================
# SMOKE TEST  (fast sanity check -- no GPU required)
# =============================================================================

smoke:
	@echo "=== smoke: 3-iter bimodal seed=0 ==="
	$(PYTHON) -m src.train $(FORCE_FLAG) \
		--condition bimodal \
		--seed 0 \
		--iters 3 \
		--device $(DEVICE) \
		--force \
		--save-dir /tmp/signal_av_smoke
	@echo ""
	@echo "=== smoke: eval (10 episodes, none only) ==="
	$(PYTHON) -m src.evaluate $(FORCE_FLAG) \
		--ckpt /tmp/signal_av_smoke/bimodal_seed0/final.pt \
		--interventions baseline \
		--episodes 10 \
		--seed 9999 \
		--device $(DEVICE) \
		--force
	@echo ""
	@echo "=== smoke: PASS ==="


# =============================================================================
# CLEANUP
# =============================================================================

clean-checkpoints:
	rm -rf $(CKPT_DIR)

clean-results:
	rm -rf $(RESULT_DIR)
