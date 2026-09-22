import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import N_SIGNALS, N_TYPES, resolve_device
from src.env import SignallingHighwayEnv
from src.metrics import counterfactual_decoding_effect, full_rho_report
from src.rollout import nbr_tensors_from_info, self_obs_from_obs
from src.train import load_checkpoint

CHECKPOINTS = Path("checkpoints")
RESULTS = Path("results")
CONDITIONS = ["bimodal", "no_signal", "random", "same_reward"]
N_EPISODES = 10
DEVICE = resolve_device("auto")


def collect(checkpoint_path: Path, device: torch.device):
    policy, meta = load_checkpoint(checkpoint_path, device)
    policy.eval()
    env = SignallingHighwayEnv(config=dict(meta["env_config"]))

    self_obs, nbr_obs, nbr_sig, nbr_mask, own_sig = [], [], [], [], []
    em_types, em_sig = [], []

    for ep in range(N_EPISODES):
        obs, info = env.reset(seed=8000 + ep)
        done = False
        while not done:
            so = self_obs_from_obs(obs).to(device)
            no, ns, nm = nbr_tensors_from_info(info)
            no, ns, nm = no.to(device), ns.to(device), nm.to(device)
            own_type = torch.from_numpy(info["types"]).long().to(device)

            with torch.no_grad():
                out = policy(so, own_type, no, ns, nm)
                signal = out["signal"].cpu()

            self_obs.append(so)
            nbr_obs.append(no)
            nbr_sig.append(ns)
            nbr_mask.append(nm)
            own_sig.append(signal.to(device))
            em_types.append(info["types"].copy())
            em_sig.append(signal.numpy())

            N = len(env.controlled_vehicles)
            action = out["action"].cpu().numpy()
            step_input = [
                (np.clip(action[i], -1.0, 1.0), int(signal[i])) for i in range(N)
            ]
            obs, _, terminated, truncated, info = env.step(step_input)
            done = bool(terminated or truncated)

    return (
        policy,
        torch.cat(self_obs, 0),
        torch.cat(nbr_obs, 0),
        torch.cat(nbr_sig, 0),
        torch.cat(nbr_mask, 0),
        torch.cat(own_sig, 0),
        np.concatenate(em_types),
        np.concatenate(em_sig),
    )


def build_swap_remap(types: np.ndarray, signals: np.ndarray) -> torch.Tensor:
    rep = full_rho_report(types, signals, N_TYPES, N_SIGNALS)
    modal = rep["p_sig_given_type"].argmax(axis=1)  # (n_types,) modal token per type
    remap = np.arange(N_SIGNALS)
    for t in range(N_TYPES):
        remap[modal[t]] = modal[(t + 1) % N_TYPES]
    return torch.from_numpy(remap).long()


def main():
    RESULTS.mkdir(exist_ok=True)
    records = []

    for cond in CONDITIONS:
        seed_dirs = sorted(CHECKPOINTS.glob(f"{cond}_seed*"))
        if not seed_dirs:
            continue
        print(f"\n=== {cond} ({len(seed_dirs)} seeds) ===")
        for sd in seed_dirs:
            ckpt = sd / "final.pt"
            if not ckpt.exists():
                print(f"  {sd.name}: final.pt missing, skipping")
                continue
            print(f"  {sd.name} ...", end=" ", flush=True)
            try:
                policy, so, no, ns, nm, osig, ty, si = collect(ckpt, DEVICE)
                remap = build_swap_remap(ty, si)
                swap = counterfactual_decoding_effect(
                    policy, so, no, ns, nm, osig, mode="swap", token_remap=remap
                )
                rand = counterfactual_decoding_effect(
                    policy, so, no, ns, nm, osig, mode="randomise", n_signals=N_SIGNALS
                )
                records.append(
                    {
                        "seed": sd.name,
                        "swap": swap,
                        "randomise": rand,
                        "n_steps": int(so.shape[0]),
                    }
                )
                print(f"swap|Δa|={swap['cde_l2']:.4f}  rand|Δa|={rand['cde_l2']:.4f}")
            except Exception as e:
                print(f"ERROR: {e}")

    out_path = RESULTS / "cde.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} seeds to {out_path}")

    # Per-condition summary. randomise is the primary metric (always defined);
    # swap only moves in a separating regime (coincident modal tokens -> no-op).
    print(
        f"\n{'condition':>12s} | {'randomise CDE (L2)':>22s} | {'swap CDE (L2)':>16s}"
    )
    for cond in CONDITIONS:
        rnd = [r["randomise"]["cde_l2"] for r in records if r["seed"].startswith(cond)]
        swp = [r["swap"]["cde_l2"] for r in records if r["seed"].startswith(cond)]
        if rnd:
            print(
                f"  {cond:>10s} | {np.mean(rnd):>8.4f} +/- {np.std(rnd):.4f} | "
                f"{np.mean(swp):>7.4f} +/- {np.std(swp):.4f}  (n={len(rnd)})"
            )


if __name__ == "__main__":
    main()
