"""
Compute Causal Influence of Communication (Lowe et al., 2019) for all bimodal seeds.

Loads each final.pt checkpoint, runs N_EPISODES rollout episodes to collect
(self_obs, nbr_obs, nbr_sig, nbr_mask) tensors, then calls
causal_influence_of_communication() from src.metrics.

Output: results/cic_bimodal.json
  [{"seed": "bimodal_seed1", "cic": 0.123, "n_steps": 4800}, ...]

Run: python scripts/compute_cic.py
"""

import json
import sys
from pathlib import Path

import numpy as np  # used for stats in main()
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import resolve_device
from src.env import SignallingHighwayEnv
from src.metrics import causal_influence_of_communication
from src.rollout import nbr_tensors_from_info, self_obs_from_obs
from src.train import load_checkpoint

CHECKPOINTS = Path("checkpoints")
RESULTS = Path("results")
CONDITIONS = ["bimodal", "same_reward", "random", "no_signal"]
N_EPISODES = 10
DEVICE = resolve_device("auto")


def collect_rollout_tensors(checkpoint_path: Path, device: torch.device):
    """Run N_EPISODES and return stacked rollout tensors for CIC computation."""
    policy, meta = load_checkpoint(checkpoint_path, device)
    policy.eval()

    env = SignallingHighwayEnv(config=dict(meta["env_config"]))

    all_self_obs, all_nbr_obs, all_nbr_sig, all_nbr_mask = [], [], [], []

    for ep in range(N_EPISODES):
        obs, info = env.reset(seed=8000 + ep)
        done = False
        while not done:
            so = self_obs_from_obs(obs).to(device)
            nbr_obs, nbr_sig, nbr_mask = nbr_tensors_from_info(info)
            nbr_obs = nbr_obs.to(device)
            nbr_sig = nbr_sig.to(device)
            nbr_mask = nbr_mask.to(device)
            own_type = torch.from_numpy(info["types"]).long().to(device)

            with torch.no_grad():
                out = policy(so, own_type, nbr_obs, nbr_sig, nbr_mask)
                action = out["action"].cpu().numpy()
                signal = out["signal"].cpu().numpy()

            all_self_obs.append(so)
            all_nbr_obs.append(nbr_obs)
            all_nbr_sig.append(nbr_sig)
            all_nbr_mask.append(nbr_mask)

            N = len(env.controlled_vehicles)
            step_input = [(np.clip(action[i], -1.0, 1.0), int(signal[i])) for i in range(N)]
            obs, _, terminated, truncated, info = env.step(step_input)
            done = bool(terminated or truncated)

    return (
        policy,
        torch.cat(all_self_obs, dim=0),
        torch.cat(all_nbr_obs, dim=0),
        torch.cat(all_nbr_sig, dim=0),
        torch.cat(all_nbr_mask, dim=0),
    )


def compute_condition(condition: str):
    """Compute per-seed CIC for one condition and write results/cic_<condition>.json."""
    seed_dirs = sorted(CHECKPOINTS.glob(f"{condition}_seed*"))
    if not seed_dirs:
        print(f"  {condition}: no checkpoints, skipping")
        return

    records = []
    for seed_dir in seed_dirs:
        ckpt = seed_dir / "final.pt"
        if not ckpt.exists():
            print(f"  {seed_dir.name}: final.pt missing, skipping")
            continue

        print(f"  {seed_dir.name} ...", end=" ", flush=True)
        try:
            policy, so, no, ns, nm = collect_rollout_tensors(ckpt, DEVICE)
            cic = causal_influence_of_communication(policy, so, no, ns, nm)
            print(f"CIC={cic:.4f}  n_steps={so.shape[0]}")
            records.append({"seed": seed_dir.name, "cic": cic, "n_steps": int(so.shape[0])})
        except Exception as e:
            print(f"ERROR: {e}")

    out_path = RESULTS / f"cic_{condition}.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  Saved {len(records)} seeds to {out_path}")

    if records:
        cics = [r["cic"] for r in records]
        print(f"  {condition}: CIC mean={np.mean(cics):.4f}  std={np.std(cics):.4f}  "
              f"min={np.min(cics):.4f}  max={np.max(cics):.4f}")


def main():
    """Compute CIC for every condition in CONDITIONS.

    Pass condition names as args to restrict (e.g. `python scripts/compute_cic.py
    same_reward random`); otherwise all of CONDITIONS are computed. Skips any
    condition whose cic_<condition>.json already exists unless --force is passed.
    """
    force = "--force" in sys.argv
    requested = [a for a in sys.argv[1:] if not a.startswith("-")] or CONDITIONS

    for condition in requested:
        out_path = RESULTS / f"cic_{condition}.json"
        if out_path.exists() and not force:
            print(f"\n=== {condition}: cached ({out_path.name}), skip (--force to recompute) ===")
            continue
        print(f"\n=== {condition} ===")
        compute_condition(condition)


if __name__ == "__main__":
    main()
