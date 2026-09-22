import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from src.config import N_SIGNALS, N_TYPES, resolve_device
from src.env import SignallingHighwayEnv
from src.metrics import full_rho_report, speaker_consistency
from src.network import SignallingPolicy
from src.rollout import nbr_tensors_from_info, self_obs_from_obs
from src.train import load_checkpoint


@dataclass
class EvalResult:
    """Metrics for one (checkpoint, intervention) combination."""

    condition: str
    rho: float
    regime: str
    mi_bits: float
    mean_return: float
    collision_rate: float
    per_agent_collision_rate: float
    n_episodes: int
    n_steps: int
    sc: float = 0.0                           # speaker consistency
    p_sig_given_type: Optional[List] = None   # (n_types, n_signals) as nested list
    p_type_given_sig: Optional[List] = None   # (n_signals, n_types) as nested list


def apply_signal_randomisation(
    signals: np.ndarray, n_signals: int, rng: np.random.Generator
) -> np.ndarray:
    """Replace every signal token with a uniform random draw from {0, ..., n_signals-1}."""
    return rng.integers(0, n_signals, size=signals.shape).astype(signals.dtype)


def apply_signal_hiding(nbr_sig: np.ndarray) -> np.ndarray:
    """Zero out all signal tokens in the neighbour observation tensors."""
    return np.zeros_like(nbr_sig)


def apply_kinematic_hiding(nbr_obs: np.ndarray) -> np.ndarray:
    """Zero kinematic features (Dx, Dy, Dvx, Dvy) in nbr_obs, keeping the presence column intact."""
    out = nbr_obs.copy()
    out[..., 1:] = 0.0
    return out


def make_profile_permutation(n_signals: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a uniformly random non-identity permutation of {0, ..., n_signals-1}."""
    while True:
        perm = rng.permutation(n_signals)
        if not np.all(perm == np.arange(n_signals)):
            return perm


def run_episode(
    env: SignallingHighwayEnv,
    policy: SignallingPolicy,
    intervention: str,
    rng: np.random.Generator,
    device: torch.device,
    profile_perm: Optional[np.ndarray],
    ep_seed: int,
) -> dict:
    """Roll a single episode under the specified intervention."""
    obs, info = env.reset(seed=ep_seed)
    N = len(env.controlled_vehicles)

    types_log: List[np.ndarray] = []
    signals_log: List[np.ndarray] = []

    cum_reward = np.zeros(N, dtype=np.float64)
    n_steps = 0
    done = False

    while not done:
        self_obs = self_obs_from_obs(obs).to(device)
        nbr_obs_np, nbr_sig_np, nbr_mask_np = (
            info["nbr_window_obs"],
            info["nbr_window_sig"],
            info["nbr_window_mask"],
        )

        # Receiver-side interventions modify what receivers see before inference.
        if intervention == "hide":
            nbr_sig_np = apply_signal_hiding(nbr_sig_np)
        elif intervention == "hide_kin":
            nbr_obs_np = apply_kinematic_hiding(nbr_obs_np)
        elif intervention == "hide_all":
            nbr_obs_np = apply_kinematic_hiding(nbr_obs_np)
            nbr_sig_np = apply_signal_hiding(nbr_sig_np)

        nbr_obs = torch.from_numpy(nbr_obs_np.astype(np.float32)).to(device)
        nbr_sig = torch.from_numpy(nbr_sig_np.astype(np.int64)).to(device)
        nbr_mask = torch.from_numpy(nbr_mask_np.astype(bool)).to(device)
        own_type = torch.from_numpy(info["types"]).long().to(device)

        with torch.no_grad():
            out = policy(self_obs, own_type, nbr_obs, nbr_sig, nbr_mask)
            action = out["action"].cpu().numpy()
            signal = out["signal"].cpu().numpy()

        # Log the policy's actual token before any sender-side intervention,
        # so rho always reflects what the sender chose, not what the env acted on.
        types_log.append(info["types"].copy())
        signals_log.append(signal.copy())

        if intervention == "randomise":
            signal_for_env = apply_signal_randomisation(signal, N_SIGNALS, rng)
            signals_log[-1] = signal_for_env.copy()
        elif intervention == "permute":
            signal_for_env = profile_perm[signal]
        else:
            signal_for_env = signal

        # desync: policy token goes to env (nbr_sig history stays correct) but
        # a random profile is applied kinematically via override_profiles.
        # Breaks the token<->kinematic-image binding without changing what receivers see.
        override_profiles = None
        if intervention == "desync":
            override_profiles = rng.integers(0, N_SIGNALS, size=N).tolist()

        action_clip = np.clip(action, -1.0, 1.0)
        step_input = [(action_clip[i], int(signal_for_env[i])) for i in range(N)]
        obs, rewards, terminated, truncated, info = env.step(
            step_input, override_profiles=override_profiles
        )

        cum_reward += np.asarray(rewards, dtype=np.float64)
        n_steps += 1
        done = bool(terminated or truncated)

    n_agent_crashes = sum(1 for v in env.controlled_vehicles if v.crashed)

    return {
        "types": np.concatenate(types_log),
        "signals": np.concatenate(signals_log),
        "reward": float(cum_reward.mean()),
        "crashed": bool(n_agent_crashes > 0),
        "n_agent_crashes": n_agent_crashes,
        "n_steps": n_steps,
    }


def evaluate(
    ckpt_path: Path,
    intervention: str,
    n_episodes: int,
    seed: int,
    device_str: str,
) -> EvalResult:
    """Evaluate a checkpoint under one intervention across n_episodes episodes."""
    device = resolve_device(device_str)

    policy, ckpt_meta = load_checkpoint(ckpt_path, device)
    policy.eval()

    env_config = dict(ckpt_meta["env_config"])
    env_config["seed"] = seed
    env = SignallingHighwayEnv(config=env_config)

    rng = np.random.default_rng(seed)

    profile_perm = (
        make_profile_permutation(N_SIGNALS, rng) if intervention == "permute" else None
    )

    all_types: List[np.ndarray] = []
    all_signals: List[np.ndarray] = []
    all_rewards: List[float] = []
    all_crashed: List[bool] = []
    all_agent_crashes: List[int] = []
    all_steps: List[int] = []

    for ep in range(n_episodes):
        result = run_episode(
            env,
            policy,
            intervention,
            rng,
            device,
            profile_perm,
            ep_seed=seed + ep,
        )
        all_types.append(result["types"])
        all_signals.append(result["signals"])
        all_rewards.append(result["reward"])
        all_crashed.append(result["crashed"])
        all_agent_crashes.append(result["n_agent_crashes"])
        all_steps.append(result["n_steps"])

        if (ep + 1) % 50 == 0:
            print(f"  {intervention}: episode {ep + 1}/{n_episodes}")

    types_concat = np.concatenate(all_types)
    signals_concat = np.concatenate(all_signals)
    rho_data = full_rho_report(types_concat, signals_concat, N_TYPES, N_SIGNALS)

    sc = speaker_consistency(rho_data["p_sig_given_type"])

    return EvalResult(
        condition=intervention,
        rho=float(rho_data["rho"]),
        regime=rho_data["regime"],
        mi_bits=float(rho_data["mi_bits"]),
        mean_return=float(np.mean(all_rewards)),
        collision_rate=float(np.mean(all_crashed)),
        per_agent_collision_rate=float(np.sum(all_agent_crashes)) / (n_episodes * ckpt_meta["n_agents"]),
        n_episodes=n_episodes,
        n_steps=int(np.sum(all_steps)),
        sc=sc,
        p_sig_given_type=rho_data["p_sig_given_type"].tolist(),
        p_type_given_sig=rho_data["p_type_given_sig"].tolist(),
    )


def main() -> None:
    """Evaluate a checkpoint under all specified interventions and print a summary table."""
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint under one or more interventions.")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument(
        "--interventions",
        nargs="+",
        default=["baseline", "randomise", "hide", "permute", "hide_kin", "hide_all", "desync"],
        choices=["baseline", "randomise", "hide", "permute", "hide_kin", "hide_all", "desync"],
    )
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=9999)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--force", action="store_true", help="re-evaluate even if --out already exists")
    args = p.parse_args()

    if args.out and Path(args.out).exists() and not args.force:
        print(f"skip: {args.out} exists (use --force to re-evaluate)")
        return

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"\nEvaluating: {ckpt_path}")
    print(f"Interventions: {args.interventions}")
    print(f"Episodes per condition: {args.episodes}\n")

    print(
        f"{'Intervention':>12s} | {'rho':>6s} | {'SC':>5s} | {'Regime':>15s} | "
        f"{'Return':>8s} | {'Crash%':>7s} | {'PerAgent%':>9s} | {'Steps':>8s}"
    )
    print("-" * 90)

    results = []
    for iv in args.interventions:
        result = evaluate(ckpt_path, iv, args.episodes, args.seed, args.device)
        results.append(result)

        print(
            f"{result.condition:>12s} | "
            f"{result.rho:>6.3f} | "
            f"{result.sc:>5.3f} | "
            f"{result.regime:>15s} | "
            f"{result.mean_return:>+8.3f} | "
            f"{result.collision_rate:>6.1%} | "
            f"{result.per_agent_collision_rate:>8.1%} | "
            f"{result.n_steps:>8d}"
        )

    print("\nNote: randomise=random tokens, hide=zero signal, hide_kin=zero kinematics, hide_all=both, desync=random profiles.")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(
                [asdict(r) for r in results],
                f,
                indent=2,
            )
        print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
