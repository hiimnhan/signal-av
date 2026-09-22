import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from src.config import (
    N_SIGNALS,
    N_TYPES,
    EnvConfig,
    NetworkConfig,
    TrainConfig,
    resolve_device,
)
from src.env import SignallingHighwayEnv
from src.metrics import full_rho_report, speaker_consistency
from src.network import CentralisedCritic, SignallingPolicy
from src.rollout import (
    RolloutBuffer,
    collect_rollout,
    episode_crash_rate,
)


def ppo_update(
    policy: SignallingPolicy,
    critic: CentralisedCritic,
    policy_optimiser: optim.Optimizer,
    critic_optimiser: optim.Optimizer,
    buffer: RolloutBuffer,
    cfg: TrainConfig,
    device: torch.device,
    entropy_coef_signal: float,
    no_signal: bool = False,
) -> dict:
    T, N = buffer.rewards.shape

    adv = buffer.advantages
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    n_total = T * N

    def flat(x):
        """Flatten (T, N, ...) -> (T*N, ...)."""
        return x.reshape(n_total, *x.shape[2:])

    flat_self_obs = flat(buffer.self_obs)
    flat_nbr_obs = flat(buffer.nbr_obs)
    flat_nbr_sig = flat(buffer.nbr_sig)
    flat_nbr_mask = flat(buffer.nbr_mask)
    flat_types = flat(buffer.types)
    flat_actions = flat(buffer.actions)
    flat_signals = flat(buffer.signals)
    flat_action_logps = flat(buffer.action_logps)
    flat_signal_logps = flat(buffer.signal_logps)
    flat_nbr_types = flat(buffer.nbr_types)
    flat_adv = flat(adv)
    flat_returns = flat(buffer.returns)

    flat_values_buf = flat(buffer.values)
    residuals = flat_returns - flat_values_buf
    explained_var = (1.0 - residuals.var() / (flat_returns.var() + 1e-8)).item()

    total_losses = {
        "policy": 0.0,
        "value": 0.0,
        "entropy": 0.0,
        "entropy_action": 0.0,
        "entropy_signal": 0.0,
        "aux": 0.0,
    }
    n_clipped = 0
    n_samples = 0
    n_steps = 0

    for _ in range(cfg.ppo_epochs):
        perm = torch.randperm(n_total)

        for start in range(0, n_total, cfg.minibatch_size):
            idx = perm[start : start + cfg.minibatch_size]

            # Timestep index lets us look up the joint state for the critic.
            timesteps = idx // N

            mb_self_obs = flat_self_obs[idx].to(device)
            mb_nbr_obs = flat_nbr_obs[idx].to(device)
            mb_nbr_sig = flat_nbr_sig[idx].to(device)
            mb_nbr_mask = flat_nbr_mask[idx].to(device)
            mb_types = flat_types[idx].to(device)
            mb_actions = flat_actions[idx].to(device)
            mb_signals = flat_signals[idx].to(device)
            mb_old_alp = flat_action_logps[idx].to(device)
            mb_old_slp = flat_signal_logps[idx].to(device)
            mb_adv = flat_adv[idx].to(device)
            mb_returns = flat_returns[idx].to(device)

            embeds, agg = policy.encode_neighbours(
                mb_nbr_obs, mb_nbr_sig, mask=mb_nbr_mask
            )

            action_dist = policy.response_head(
                mb_self_obs, agg, mb_signals.long(), no_signal=no_signal
            )
            new_action_lp = action_dist.log_prob(mb_actions).sum(dim=-1)

            if no_signal:
                new_signal_lp = torch.zeros_like(new_action_lp)
                entropy_signal = torch.tensor(0.0, device=device)
            else:
                signal_dist = policy.signal_head(mb_self_obs, mb_types, agg)
                new_signal_lp = signal_dist.log_prob(mb_signals.long())
                entropy_signal = signal_dist.entropy().mean()

            # Combined log-prob over both heads (factorised)
            new_logp = new_action_lp + new_signal_lp
            old_logp = mb_old_alp + mb_old_slp

            ratio = torch.exp(new_logp - old_logp)
            clipped_ratio = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
            policy_loss = -torch.min(
                ratio * mb_adv,
                clipped_ratio * mb_adv,
            ).mean()

            entropy_action = action_dist.entropy().sum(dim=-1).mean()
            entropy = entropy_action + entropy_signal

            # Supervise neighbour embeddings using the true neighbour type (privileged).
            # -1 = background vehicle, excluded via the valid mask.
            flat_embeds = embeds.reshape(-1, embeds.shape[-1])
            mb_nbr_types = flat_nbr_types[idx].to(device)
            aux_targets = mb_nbr_types.reshape(-1)

            valid = aux_targets >= 0
            if valid.any():
                aux_logits = policy.aux_head(flat_embeds[valid])
                aux_loss = F.cross_entropy(aux_logits, aux_targets[valid])
            else:
                aux_loss = flat_embeds.sum() * 0.0

            # Critic uses the joint state at each sample's timestep.
            joint_self_obs = buffer.self_obs[timesteps].to(device)
            joint_types = buffer.types[timesteps].to(device)
            joint_agg = buffer.agg_all[timesteps].to(device)

            # PPO-style value clipping: prevents large critic jumps between iters.
            v_old = buffer.values[timesteps].mean(dim=-1).to(device)
            predicted_values = critic(joint_self_obs, joint_types, joint_agg)
            v_clipped = v_old + torch.clamp(
                predicted_values - v_old, -cfg.value_clip_eps, cfg.value_clip_eps
            )
            ts_returns = buffer.returns[timesteps].mean(dim=-1).to(device)
            value_loss = torch.max(
                (ts_returns - predicted_values) ** 2,
                (ts_returns - v_clipped) ** 2,
            ).mean()

            pi_loss = (
                policy_loss
                - cfg.entropy_coef_action * entropy_action
                - entropy_coef_signal * entropy_signal
                + cfg.aux_coef * aux_loss
            )
            policy_optimiser.zero_grad()
            pi_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            policy_optimiser.step()

            v_loss_scaled = cfg.value_coef * value_loss
            critic_optimiser.zero_grad()
            v_loss_scaled.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), cfg.grad_clip)
            critic_optimiser.step()

            n_clipped += (
                ((ratio < 1.0 - cfg.clip_eps) | (ratio > 1.0 + cfg.clip_eps))
                .sum()
                .item()
            )
            n_samples += ratio.numel()

            total_losses["policy"] += policy_loss.item()
            total_losses["value"] += value_loss.item()
            total_losses["entropy"] += entropy.item()
            total_losses["entropy_action"] += entropy_action.item()
            total_losses["entropy_signal"] += entropy_signal.item()
            total_losses["aux"] += aux_loss.item()
            n_steps += 1

    result = {k: v / max(n_steps, 1) for k, v in total_losses.items()}
    result["clip_frac"] = n_clipped / max(n_samples, 1)
    result["explained_var"] = explained_var
    return result


def save_checkpoint(
    path: Path,
    policy: SignallingPolicy,
    critic: CentralisedCritic,
    cfg: TrainConfig,
    env_config: dict,
    obs_features: int,
    n_agents: int,
) -> None:
    """Save policy + critic weights and all metadata needed to reload them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "obs_features": obs_features,
            "n_agents": n_agents,
            "n_signals": N_SIGNALS,
            "n_types": N_TYPES,
            "embed_dim": policy.encoder.embed_dim,
            "agg_dim": critic.agg_dim,
            "env_config": env_config,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple:
    """Load a saved checkpoint and reconstruct the policy. Returns (policy, metadata_dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)

    policy = SignallingPolicy(
        obs_features=ckpt["obs_features"],
        n_signals=ckpt["n_signals"],
        n_types=ckpt["n_types"],
        embed_dim=ckpt["embed_dim"],
        agg_dim=ckpt["agg_dim"],
    ).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])

    return policy, ckpt


def train(args: argparse.Namespace) -> None:
    """Full training run for one (condition, seed) combination."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    device = resolve_device(args.device)
    print(f"device: {device}")

    assert not (args.no_signal and args.ungrounded), (
        "use only one of --no-signal / --ungrounded"
    )

    run_label = (
        args.run_name
        if args.run_name
        else "no_signal"
        if args.no_signal
        else "ungrounded"
        if args.ungrounded
        else "random"
        if args.random_signal
        else args.condition
    )
    final_path = Path(args.save_dir) / f"{run_label}_seed{args.seed}" / "final.pt"
    if final_path.exists() and not args.force:
        print(
            f"skip {run_label}_seed{args.seed}: {final_path} exists (use --force to retrain)"
        )
        return

    env_config = {
        "controlled_vehicles": args.n_agents,
        "vehicles_count": args.n_bg_vehicles,
        "n_neighbours": args.n_neighbours,
        "type_condition": args.condition,
        "divergence_scale": args.divergence_scale,
        "crash_scale": args.crash_scale,
        "seed": args.seed,
        "no_signal": args.no_signal,
        "ungrounded": args.ungrounded,
    }
    env = SignallingHighwayEnv(config=env_config)
    obs, info = env.reset(seed=args.seed)

    n_agents = len(env.controlled_vehicles)
    obs_features = obs[0].shape[-1]

    net_cfg = NetworkConfig()
    policy = SignallingPolicy(
        obs_features=obs_features,
        n_signals=N_SIGNALS,
        n_types=N_TYPES,
        action_dim=net_cfg.action_dim,
        embed_dim=net_cfg.embed_dim,
        agg_dim=net_cfg.agg_dim,
        hidden_dim=net_cfg.hidden_dim,
    ).to(device)

    critic = CentralisedCritic(
        obs_features=obs_features,
        agg_dim=net_cfg.agg_dim,
        n_types=N_TYPES,
        n_agents=n_agents,
        hidden_dim=args.critic_hidden_dim,
    ).to(device)

    policy_optimiser = optim.Adam(policy.parameters(), lr=args.lr)
    critic_optimiser = optim.Adam(critic.parameters(), lr=args.critic_lr)

    train_cfg = TrainConfig(
        total_iters=args.iters,
        rollout_steps=args.rollout_steps,
        seed=args.seed,
        device=args.device,
        entropy_coef_action=args.entropy_coef_action,
        entropy_coef_signal_start=args.signal_entropy_start,
        entropy_coef_signal_end=args.signal_entropy_end,
        entropy_anneal_iters=args.entropy_anneal_iters,
        aux_coef=args.aux_coef,
    )

    n_policy = sum(p.numel() for p in policy.parameters())
    n_critic = sum(p.numel() for p in critic.parameters())
    print(
        f"Training: condition={args.condition}, seed={args.seed}, iters={args.iters}, random_signal={args.random_signal}, no_signal={args.no_signal}"
    )
    print(
        f"  env: n_agents={args.n_agents}, n_bg={args.n_bg_vehicles}, n_neighbours={args.n_neighbours}"
    )
    print(
        f"  ppo: lr={args.lr}, rollout={args.rollout_steps}, clip={train_cfg.clip_eps}, gamma={train_cfg.gamma}"
    )
    print(f"  policy={n_policy:,} params, critic={n_critic:,} params")

    run_dir = final_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = open(run_dir / "metrics.csv", "w", newline="")
    metrics_writer = csv.DictWriter(
        metrics_file,
        fieldnames=[
            "iter",
            "reward",
            "crash",
            "rho",
            "sc",
            "t0_top",
            "t1_top",
            "ep_len",
            "pi_loss",
            "v_loss",
            "entropy",
            "entropy_action",
            "entropy_signal",
            "aux",
            "clip_frac",
            "explained_var",
            "sig_ent",
        ],
    )
    metrics_writer.writeheader()

    try:
        for it in range(train_cfg.total_iters):
            t_start = time.time()

            # Curriculum: linearly ramp background vehicles from 0 to n_bg_vehicles.
            # Takes effect at the next episode reset inside collect_rollout.
            if args.curriculum_iters > 0 and args.n_bg_vehicles > 0:
                curriculum_frac = min(float(it) / args.curriculum_iters, 1.0)
                n_bg_current = round(curriculum_frac * args.n_bg_vehicles)
                env.configure({"vehicles_count": n_bg_current})

            buffer, obs, info = collect_rollout(
                env=env,
                policy=policy,
                critic=critic,
                current_obs=obs,
                current_info=info,
                rollout_steps=train_cfg.rollout_steps,
                device=device,
                random_signal=args.random_signal,
                no_signal=args.no_signal,
                gamma=train_cfg.gamma,
                lam=train_cfg.gae_lambda,
            )

            # Linearly anneal signal entropy bonus from start to end over the first
            # entropy_anneal_iters iterations, then hold. Keeps signal exploration high
            # early so receivers encounter all tokens before a convention locks in.
            anneal_frac = min(float(it) / max(train_cfg.entropy_anneal_iters, 1), 1.0)
            sig_ent_coef = (
                train_cfg.entropy_coef_signal_start * (1.0 - anneal_frac)
                + train_cfg.entropy_coef_signal_end * anneal_frac
            )
            loss_stats = ppo_update(
                policy,
                critic,
                policy_optimiser,
                critic_optimiser,
                buffer,
                train_cfg,
                device,
                sig_ent_coef,
                no_signal=args.no_signal,
            )

            if (it + 1) % train_cfg.log_every == 0:
                types_flat = buffer.types.numpy().ravel()
                signals_flat = buffer.signals.numpy().ravel()
                rho_report = full_rho_report(
                    types_flat, signals_flat, N_TYPES, N_SIGNALS
                )

                mean_reward = buffer.returns.mean().item()
                crash_rate = episode_crash_rate(buffer.crashes, buffer.dones)
                sc = speaker_consistency(rho_report["p_sig_given_type"])
                p = rho_report["p_sig_given_type"]
                t0_top = int(p[0].argmax())
                t1_top = int(p[1].argmax())
                ep_len = train_cfg.rollout_steps / max(buffer.dones.sum().item(), 1)
                dt = time.time() - t_start
                type_sig_str = " | ".join(
                    f"t{t}:["
                    + " ".join(f"{p[t, m]:.2f}" for m in range(N_SIGNALS))
                    + "]"
                    for t in range(N_TYPES)
                )
                print(
                    f"iter {it + 1:04d} | "
                    f"reward={mean_reward:+.3f} | "
                    f"crash={crash_rate:.1%} | "
                    f"rho={rho_report['rho']:.3f} ({rho_report['regime']}) | "
                    f"P(sig|type): {type_sig_str} | "
                    f"loss: pi={loss_stats['policy']:+.3f} "
                    f"v={loss_stats['value']:.3f} "
                    f"H={loss_stats['entropy']:.3f} "
                    f"aux={loss_stats['aux']:.3f} | "
                    f"sig_ent={sig_ent_coef:.4f} | "
                    f"dt={dt:.1f}s"
                )
                metrics_writer.writerow(
                    {
                        "iter": it + 1,
                        "reward": round(mean_reward, 4),
                        "crash": round(crash_rate, 4),
                        "rho": round(rho_report["rho"], 4),
                        "sc": round(sc, 4),
                        "t0_top": t0_top,
                        "t1_top": t1_top,
                        "ep_len": round(ep_len, 1),
                        "pi_loss": round(loss_stats["policy"], 4),
                        "v_loss": round(loss_stats["value"], 4),
                        "entropy": round(loss_stats["entropy"], 4),
                        "entropy_action": round(loss_stats["entropy_action"], 4),
                        "entropy_signal": round(loss_stats["entropy_signal"], 4),
                        "aux": round(loss_stats["aux"], 4),
                        "clip_frac": round(loss_stats["clip_frac"], 4),
                        "explained_var": round(loss_stats["explained_var"], 4),
                        "sig_ent": round(sig_ent_coef, 6),
                    }
                )
                metrics_file.flush()

            if (it + 1) % train_cfg.checkpoint_every == 0:
                label = (
                    args.run_name
                    if args.run_name
                    else "no_signal"
                    if args.no_signal
                    else "ungrounded"
                    if args.ungrounded
                    else "random"
                    if args.random_signal
                    else args.condition
                )
                ckpt_path = (
                    Path(args.save_dir)
                    / f"{label}_seed{args.seed}"
                    / f"iter_{it + 1:04d}.pt"
                )
                save_checkpoint(
                    ckpt_path,
                    policy,
                    critic,
                    train_cfg,
                    env_config,
                    obs_features,
                    n_agents,
                )
                print(f"  -> checkpoint saved: {ckpt_path}")

        label = (
            args.run_name
            if args.run_name
            else "no_signal"
            if args.no_signal
            else "ungrounded"
            if args.ungrounded
            else "random"
            if args.random_signal
            else args.condition
        )
        final_path = Path(args.save_dir) / f"{label}_seed{args.seed}" / "final.pt"
        save_checkpoint(
            final_path, policy, critic, train_cfg, env_config, obs_features, n_agents
        )
    finally:
        metrics_file.close()
    print(f"Training complete. Final checkpoint: {final_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the signalling highway agent.")
    p.add_argument(
        "--condition", type=str, default="bimodal", choices=["bimodal", "same_reward"]
    )
    p.add_argument("--divergence-scale", type=float, default=1.0)
    p.add_argument("--random-signal", action="store_true")
    p.add_argument("--no-signal", action="store_true")
    p.add_argument("--ungrounded", action="store_true")
    p.add_argument("--n-agents", type=int, default=6)
    p.add_argument("--n-bg-vehicles", type=int, default=0)
    p.add_argument("--n-neighbours", type=int, default=5)
    p.add_argument("--critic-hidden-dim", type=int, default=128)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=1600)
    p.add_argument("--rollout-steps", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument(
        "--force", action="store_true", help="retrain even if final.pt already exists"
    )
    p.add_argument("--aux-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef-action", type=float, default=0.001)
    p.add_argument("--signal-entropy-start", type=float, default=0.08)
    p.add_argument("--signal-entropy-end", type=float, default=0.005)
    p.add_argument("--entropy-anneal-iters", type=int, default=1200)
    p.add_argument("--curriculum-iters", type=int, default=0)
    p.add_argument("--crash-scale", type=float, default=1.0)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
