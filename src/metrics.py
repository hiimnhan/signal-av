import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from src.network import SignallingPolicy


def joint_distribution(
    types: np.ndarray,
    signals: np.ndarray,
    n_types: int,
    n_signals: int,
) -> np.ndarray:
    types = np.asarray(types).ravel().astype(int)
    signals = np.asarray(signals).ravel().astype(int)

    assert len(types) == len(signals), (
        f"types and signals must be the same length: "
        f"got {len(types)} and {len(signals)}"
    )

    _, type_idx = np.unique(types, return_inverse=True)
    _, sig_idx = np.unique(signals, return_inverse=True)

    table = np.zeros((n_types, n_signals), dtype=np.float64)
    np.add.at(table, (type_idx, sig_idx), 1)

    total = table.sum()
    if total == 0:
        return table

    return table / total


def mutual_information(joint: np.ndarray) -> float:
    p_type = joint.sum(axis=1, keepdims=True)
    p_signal = joint.sum(axis=0, keepdims=True)
    p_independent = p_type * p_signal

    valid = (joint > 0) & (p_independent > 0)
    mi = float((joint[valid] * np.log2(joint[valid] / p_independent[valid])).sum())
    return mi


def separation_coefficient(
    types: np.ndarray,
    signals: np.ndarray,
    n_types: int,
    n_signals: int,
) -> float:
    if n_types <= 1:
        return 0.0

    joint = joint_distribution(types, signals, n_types, n_signals)
    mi = mutual_information(joint)
    return mi / np.log2(n_types)


def speaker_consistency(p_sig_given_type: np.ndarray) -> float:
    return float(p_sig_given_type.max(axis=1).mean())


def regime_label(rho: float) -> str:
    if rho < 0.1:
        return "pooling"
    elif rho > 0.9:
        return "separating"
    else:
        return "partial-pooling"


def p_type_given_signal(
    types: np.ndarray,
    signals: np.ndarray,
    n_types: int,
    n_signals: int,
) -> np.ndarray:
    joint = joint_distribution(types, signals, n_types, n_signals)
    p_signal = joint.sum(axis=0)

    conditional = np.zeros((n_signals, n_types), dtype=np.float64)
    for m in range(n_signals):
        if p_signal[m] > 0:
            conditional[m] = joint[:, m] / p_signal[m]
        else:
            conditional[m] = 1.0 / n_types

    return conditional


def full_rho_report(
    types: np.ndarray,
    signals: np.ndarray,
    n_types: int,
    n_signals: int,
) -> dict:
    """Compute all separation diagnostics at once. Returns a dict with rho, regime,
    mi_bits, joint, p_type_given_sig, p_sig_given_type."""
    joint = joint_distribution(types, signals, n_types, n_signals)
    mi = mutual_information(joint)
    rho = mi / np.log2(n_types) if n_types > 1 else 0.0

    p_type_given_sig = p_type_given_signal(types, signals, n_types, n_signals)

    p_type = joint.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        p_sig_given_type = np.where(p_type > 0, joint / p_type, 1.0 / n_signals)

    return {
        "rho": rho,
        "regime": regime_label(rho),
        "mi_bits": mi,
        "joint": joint,
        "p_type_given_sig": p_type_given_sig,
        "p_sig_given_type": p_sig_given_type,
    }


def causal_influence_of_communication(
    policy: SignallingPolicy,
    self_obs: torch.Tensor,  # (B, obs_features)
    nbr_obs: torch.Tensor,  # (B, K, W, obs_features)
    nbr_sig: torch.Tensor,  # (B, K, W) int64
    nbr_mask: torch.Tensor,  # (B, K) bool
    own_signals: torch.Tensor = None,  # (B,) int64; neutral (1) if not provided
    n_msg_samples: int = 64,
    n_action_samples: int = 16,
) -> float:
    """
    Causal Influence of Communication (Lowe et al., 2019).
    https://doi.org/10.48550/arXiv.1903.05168

    """
    policy.eval()
    with torch.no_grad():
        B = nbr_sig.shape[0]

        if own_signals is None:
            own_signals = torch.ones(B, dtype=torch.long, device=self_obs.device)

        # Action distribution with actual received signals.
        _, agg_actual = policy.encode_neighbours(nbr_obs, nbr_sig, mask=nbr_mask)
        dist_actual = policy.response_head(self_obs, agg_actual, own_signals)
        mu_actual = dist_actual.mean  # (B, D)
        std = dist_actual.stddev  # (B, D)

        # Build random-message action means by shuffling signals across batch.
        mu_random = []
        for _ in range(n_msg_samples):
            perm = torch.randperm(B, device=nbr_sig.device)
            _, agg_r = policy.encode_neighbours(nbr_obs, nbr_sig[perm], mask=nbr_mask)
            mu_random.append(policy.response_head(self_obs, agg_r, own_signals).mean)
        mu_random = torch.stack(mu_random)  # (S, B, D)

        # Sample actions from actual distribution.
        noise = torch.randn(
            n_action_samples, B, mu_actual.shape[-1], device=mu_actual.device
        )
        actions = mu_actual.unsqueeze(0) + std.unsqueeze(0) * noise  # (T, B, D)

        # Log-prob under actual dist (constant terms cancel in KL).
        log_p_actual = -0.5 * (noise**2).sum(dim=-1)  # (T, B)

        # Log-prob under each random-message dist, then log-average over S.
        diff = actions.unsqueeze(0) - mu_random.unsqueeze(1)  # (S, T, B, D)
        log_p_each = -0.5 * ((diff / std) ** 2).sum(dim=-1)  # (S, T, B)
        log_p_mix = torch.logsumexp(log_p_each, dim=0) - np.log(n_msg_samples)  # (T, B)

        # KL = E_action[log p_actual - log p_mix], mean over actions and batch.
        kl = (log_p_actual - log_p_mix).mean(dim=0).clamp(min=0.0)  # (B,)

    return float(kl.mean().item())


def counterfactual_decoding_effect(
    policy: SignallingPolicy,
    self_obs: torch.Tensor,  # (B, obs_features)
    nbr_obs: torch.Tensor,  # (B, K, W, obs_features)
    nbr_sig: torch.Tensor,  # (B, K, W) int64  -- received tokens
    nbr_mask: torch.Tensor,  # (B, K) bool
    own_signals: torch.Tensor,  # (B,) int64  -- receiver's own committed token
    mode: str = "swap",  # "swap" | "randomise" | "shuffle"
    token_remap: torch.Tensor = None,  # (n_signals,) long; required for mode="swap"
    n_signals: int = None,  # required for mode="randomise"
    rng_seed: int = 0,
) -> dict:
    policy.eval()
    with torch.no_grad():
        device = self_obs.device

        # Factual action mean (real received tokens).
        _, agg_f = policy.encode_neighbours(nbr_obs, nbr_sig, mask=nbr_mask)
        mu_f = policy.response_head(self_obs, agg_f, own_signals).mean  # (B, A)

        # Build the counterfactual received tokens.
        if mode == "swap":
            if token_remap is None:
                raise ValueError("mode='swap' requires token_remap")
            nbr_sig_cf = token_remap.to(device)[nbr_sig.long()]
        elif mode == "randomise":
            if n_signals is None:
                raise ValueError("mode='randomise' requires n_signals")
            g = torch.Generator().manual_seed(rng_seed)
            nbr_sig_cf = torch.randint(0, n_signals, nbr_sig.shape, generator=g).to(
                device
            )
        elif mode == "shuffle":
            g = torch.Generator().manual_seed(rng_seed)
            perm = torch.randperm(nbr_sig.shape[0], generator=g).to(device)
            nbr_sig_cf = nbr_sig[perm]
        else:
            raise ValueError(f"unknown mode: {mode}")

        # Only alter real received tokens: keep absent neighbours and padding
        # history frames (sig < 0) identical, else the encoder would read padding
        # as a genuine token (and remap[-1] wraps to the last token).
        keep = ~nbr_mask.unsqueeze(-1).expand_as(nbr_sig) | (nbr_sig < 0)
        nbr_sig_cf = torch.where(keep, nbr_sig, nbr_sig_cf)

        _, agg_cf = policy.encode_neighbours(nbr_obs, nbr_sig_cf, mask=nbr_mask)
        mu_cf = policy.response_head(self_obs, agg_cf, own_signals).mean  # (B, A)

        delta = mu_cf - mu_f  # (B, A)
        changed = (nbr_sig_cf != nbr_sig).flatten(1).any(dim=1)  # (B,)

        if int(changed.sum()) == 0:
            return {
                "cde_l2": 0.0,
                "d_accel": 0.0,
                "d_steer": 0.0,
                "signed_accel": 0.0,
                "signed_steer": 0.0,
                "n_valid": 0,
            }

        d = delta[changed]
        return {
            "cde_l2": float(d.norm(dim=-1).mean()),
            "d_accel": float(d[:, 0].abs().mean()),
            "d_steer": float(d[:, 1].abs().mean()),
            "signed_accel": float(d[:, 0].mean()),
            "signed_steer": float(d[:, 1].mean()),
            "n_valid": int(changed.sum()),
        }


def aggregate_seeds(result_paths: List[Path]) -> Dict[str, dict]:
    all_seed_data: List[List[dict]] = []
    for path in result_paths:
        with open(path) as f:
            all_seed_data.append(json.load(f))

    per_intervention: Dict[str, Dict[str, list]] = {}

    for seed_results in all_seed_data:
        for record in seed_results:
            iv = record["condition"]
            if iv not in per_intervention:
                per_intervention[iv] = {
                    "rho": [],
                    "mean_return": [],
                    "collision_rate": [],
                    "regime": [],
                }
            per_intervention[iv]["rho"].append(record["rho"])
            per_intervention[iv]["mean_return"].append(record["mean_return"])
            per_intervention[iv]["collision_rate"].append(record["collision_rate"])
            per_intervention[iv]["regime"].append(record["regime"])

    aggregated: Dict[str, dict] = {}
    for iv, values in per_intervention.items():
        rhos = np.array(values["rho"])
        returns = np.array(values["mean_return"])
        crashes = np.array(values["collision_rate"])

        median_rho = float(np.median(rhos))
        closest_idx = int(np.argmin(np.abs(rhos - median_rho)))

        aggregated[iv] = {
            "rho_mean": float(rhos.mean()),
            "rho_std": float(rhos.std()),
            "return_mean": float(returns.mean()),
            "return_std": float(returns.std()),
            "crash_mean": float(crashes.mean()),
            "crash_std": float(crashes.std()),
            "regime": values["regime"][closest_idx],
            "n_seeds": len(result_paths),
        }

    # Per-seed deltas vs the "baseline" (intact) cell. delta = intervention - baseline per seed.
    if "baseline" in per_intervention:
        none_rhos = np.array(per_intervention["baseline"]["rho"])
        none_returns = np.array(per_intervention["baseline"]["mean_return"])
        none_crashes = np.array(per_intervention["baseline"]["collision_rate"])
        for iv, values in per_intervention.items():
            if iv == "baseline":
                continue
            d_rho = np.array(values["rho"]) - none_rhos
            d_ret = np.array(values["mean_return"]) - none_returns
            d_crash = np.array(values["collision_rate"]) - none_crashes
            aggregated[iv]["delta_rho_mean"] = float(d_rho.mean())
            aggregated[iv]["delta_rho_std"] = float(d_rho.std())
            aggregated[iv]["delta_return_mean"] = float(d_ret.mean())
            aggregated[iv]["delta_return_std"] = float(d_ret.std())
            aggregated[iv]["delta_crash_mean"] = float(d_crash.mean())
            aggregated[iv]["delta_crash_std"] = float(d_crash.std())

    return aggregated


def print_aggregate_table(stats: Dict[str, dict]) -> None:
    header = (
        f"{'Intervention':>12s} | "
        f"{'rho':>14s} | "
        f"{'Regime':>15s} | "
        f"{'Return':>16s} | "
        f"{'Crash%':>13s}"
    )
    print(header)
    print("-" * len(header))

    for iv, s in stats.items():
        print(
            f"{iv:>12s} | "
            f"{s['rho_mean']:>6.3f}+/-{s['rho_std']:.3f} | "
            f"{s['regime']:>15s} | "
            f"{s['return_mean']:>+7.3f}+/-{s['return_std']:.3f} | "
            f"{s['crash_mean']:>5.1%}+/-{s['crash_std']:.1%}"
        )

    has_deltas = any("delta_rho_mean" in v for v in stats.values())
    if has_deltas:
        print()
        d_header = (
            f"{'Intervention':>12s} | "
            f"{'d_rho':>14s} | "
            f"{'d_return':>16s} | "
            f"{'d_crash%':>14s}"
        )
        print(d_header)
        print("-" * len(d_header))
        for iv, s in stats.items():
            if "delta_rho_mean" not in s:
                print(f"{iv:>12s} | {'---':>14s} | {'---':>16s} | {'---':>14s}")
                continue
            print(
                f"{iv:>12s} | "
                f"{s['delta_rho_mean']:>+6.3f}+/-{s['delta_rho_std']:.3f} | "
                f"{s['delta_return_mean']:>+7.3f}+/-{s['delta_return_std']:.3f} | "
                f"{s['delta_crash_mean']:>+5.1%}+/-{s['delta_crash_std']:.1%}"
            )
