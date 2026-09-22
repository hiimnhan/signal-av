from dataclasses import dataclass, field

import numpy as np
import torch

from src.env import SignallingHighwayEnv
from src.network import CentralisedCritic, SignallingPolicy


@dataclass
class RolloutBuffer:
    """
    T = rollout_steps, N = n_agents, K = n_neighbours, W = window_steps,
    F = obs_features, A = action_dim (2).
    """

    self_obs: torch.Tensor = field(default=None)  # (T, N, F)
    nbr_obs: torch.Tensor = field(default=None)  # (T, N, K, W, F)
    nbr_sig: torch.Tensor = field(default=None)  # (T, N, K, W)
    nbr_mask: torch.Tensor = field(default=None)  # (T, N, K) bool
    types: torch.Tensor = field(default=None)  # (T, N)
    nbr_types: torch.Tensor = field(default=None)  # (T, N, K)  -1 = background
    actions: torch.Tensor = field(default=None)  # (T, N, A)
    signals: torch.Tensor = field(default=None)  # (T, N)
    action_logps: torch.Tensor = field(default=None)  # (T, N)
    signal_logps: torch.Tensor = field(default=None)  # (T, N)
    rewards: torch.Tensor = field(default=None)  # (T, N)
    dones: torch.Tensor = field(default=None)  # (T,)
    values: torch.Tensor = field(default=None)  # (T, N)
    advantages: torch.Tensor = field(default=None)  # (T, N)
    returns: torch.Tensor = field(default=None)  # (T, N)
    agg_all: torch.Tensor = field(default=None)  # (T, N, agg_dim)
    crashes: torch.Tensor = field(default=None)  # (T, N) 1=crashed 0=ok


def self_obs_from_obs(obs_tuple: tuple) -> torch.Tensor:
    stacked = np.stack(obs_tuple).astype(np.float32)
    return torch.from_numpy(stacked[:, 0, :])


def nbr_tensors_from_info(
    info: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nbr_obs = torch.from_numpy(info["nbr_window_obs"].astype(np.float32))
    nbr_sig = torch.from_numpy(info["nbr_window_sig"].astype(np.int64))
    nbr_mask = torch.from_numpy(info["nbr_window_mask"].astype(bool))
    return nbr_obs, nbr_sig, nbr_mask


def collect_rollout(
    env: SignallingHighwayEnv,
    policy: SignallingPolicy,
    critic: CentralisedCritic,
    current_obs: tuple,
    current_info: dict,
    rollout_steps: int,
    device: torch.device,
    random_signal: bool = False,
    no_signal: bool = False,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[RolloutBuffer, tuple, dict]:
    N = len(env.controlled_vehicles)
    K = env.config["n_neighbours"]
    W = env.config["window_steps"]
    obs_f = current_obs[0].shape[-1]
    T = rollout_steps

    buf_self_obs = torch.zeros(T, N, obs_f)
    buf_nbr_obs = torch.zeros(T, N, K, W, obs_f)
    buf_nbr_sig = torch.zeros(T, N, K, W, dtype=torch.long)
    buf_nbr_mask = torch.zeros(T, N, K, dtype=torch.bool)
    buf_nbr_types = torch.full((T, N, K), -1, dtype=torch.long)
    buf_types = torch.zeros(T, N, dtype=torch.long)
    buf_actions = torch.zeros(T, N, 2)
    buf_signals = torch.zeros(T, N, dtype=torch.long)
    buf_action_logps = torch.zeros(T, N)
    buf_signal_logps = torch.zeros(T, N)
    buf_rewards = torch.zeros(T, N)
    buf_dones = torch.zeros(T)
    buf_values = torch.zeros(T, N)
    buf_agg_all = torch.zeros(T, N, critic.agg_dim)
    buf_crashes = torch.zeros(T, N)

    obs = current_obs
    info = current_info

    with torch.no_grad():
        for t in range(T):
            self_obs = self_obs_from_obs(obs)
            nbr_obs, nbr_sig, nbr_mask = nbr_tensors_from_info(info)
            own_type = torch.from_numpy(info["types"]).long()

            self_obs_d = self_obs.to(device)
            nbr_obs_d = nbr_obs.to(device)
            nbr_sig_d = nbr_sig.to(device)
            nbr_mask_d = nbr_mask.to(device)
            own_type_d = own_type.to(device)

            out = policy(
                self_obs_d,
                own_type_d,
                nbr_obs_d,
                nbr_sig_d,
                nbr_mask_d,
                no_signal=no_signal,
            )

            action = out["action"].cpu()
            signal = out["signal"].cpu()
            act_logp = out["action_logp"].cpu()
            sig_logp = out["signal_logp"].cpu()
            agg = out["agg"].cpu()

            if random_signal:
                signal = torch.randint(0, env.config.get("n_signals", 3), (N,))
                n_sig = env.config.get("n_signals", 3)
                sig_logp = torch.full((N,), -np.log(n_sig))
            elif no_signal:
                signal = torch.ones(N, dtype=torch.long)
                sig_logp = torch.zeros(N)

            value = critic(
                self_obs_d.unsqueeze(0),
                own_type_d.unsqueeze(0),
                agg.to(device).unsqueeze(0),
            )
            value_expanded = value.cpu().expand(N)

            action_np = action.clamp(-1.0, 1.0).numpy()
            signal_np = signal.numpy()
            step_input = [(action_np[i], int(signal_np[i])) for i in range(N)]
            next_obs, rewards, terminated, truncated, next_info = env.step(step_input)

            buf_self_obs[t] = self_obs
            buf_nbr_obs[t] = nbr_obs
            buf_nbr_sig[t] = nbr_sig
            buf_nbr_mask[t] = nbr_mask
            buf_nbr_types[t] = torch.from_numpy(info["nbr_window_types"])
            buf_types[t] = own_type
            buf_actions[t] = action
            buf_signals[t] = signal
            buf_action_logps[t] = act_logp
            buf_signal_logps[t] = sig_logp
            buf_rewards[t] = torch.from_numpy(np.asarray(rewards, dtype=np.float32))
            buf_dones[t] = float(terminated or truncated)
            buf_values[t] = value_expanded
            buf_agg_all[t] = agg
            buf_crashes[t] = torch.tensor(
                [float(v.crashed) for v in env.controlled_vehicles]
            )

            if terminated or truncated:
                obs, info = env.reset()
            else:
                obs, info = next_obs, next_info

    # Bootstrap V(s_T) for GAE.
    with torch.no_grad():
        self_obs_last = self_obs_from_obs(obs).to(device)
        nbr_obs_l, nbr_sig_l, nbr_mask_l = nbr_tensors_from_info(info)
        own_type_last = torch.from_numpy(info["types"]).long().to(device)
        _, agg_last = policy.encode_neighbours(
            nbr_obs_l.to(device), nbr_sig_l.to(device), mask=nbr_mask_l.to(device)
        )
        last_value = (
            critic(
                self_obs_last.unsqueeze(0),
                own_type_last.unsqueeze(0),
                agg_last.unsqueeze(0),
            )
            .cpu()
            .expand(N)
        )

    buffer = RolloutBuffer(
        self_obs=buf_self_obs,
        nbr_obs=buf_nbr_obs,
        nbr_sig=buf_nbr_sig,
        nbr_mask=buf_nbr_mask,
        nbr_types=buf_nbr_types,
        types=buf_types,
        actions=buf_actions,
        signals=buf_signals,
        action_logps=buf_action_logps,
        signal_logps=buf_signal_logps,
        rewards=buf_rewards,
        dones=buf_dones,
        values=buf_values,
        agg_all=buf_agg_all,
        crashes=buf_crashes,
    )
    buffer.advantages, buffer.returns = compute_gae(
        buffer.rewards, buffer.values, buffer.dones, last_value, gamma=gamma, lam=lam
    )

    return buffer, obs, info


def episode_crash_rate(crashes: torch.Tensor, dones: torch.Tensor) -> float:
    """Fraction of completed episodes where at least one agent crashed."""
    n_episodes = 0
    n_crashed = 0
    ep_crashed = False
    for t in range(dones.shape[0]):
        if crashes[t].any():
            ep_crashed = True
        if dones[t]:
            n_episodes += 1
            if ep_crashed:
                n_crashed += 1
            ep_crashed = False
    return n_crashed / max(n_episodes, 1)


def compute_gae(
    rewards: torch.Tensor,  # (T, N)
    values: torch.Tensor,  # (T, N)
    dones: torch.Tensor,  # (T,)
    last_value: torch.Tensor,  # (N,) bootstrap value after the rollout
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(N)

    for t in reversed(range(T)):
        not_done = 1.0 - dones[t].float()

        next_value = last_value if t == T - 1 else values[t + 1]

        delta = rewards[t] + gamma * next_value * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns
