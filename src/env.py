from collections import deque
from typing import List, Optional, Sequence, Tuple

import numpy as np
from highway_env.envs.highway_env import HighwayEnv

from src.config import (
    PROFILES,
    N_SIGNALS,
    N_TYPES,
    TYPE_WEIGHTS_BIMODAL,
    TYPE_WEIGHTS_SAME_REWARD,
    TypeWeights,
    make_bimodal_weights,
)

# Placeholder signal for background (non-controlled) cars.
NO_SIGNAL = -1

# Per-vehicle, per-frame feature vector: (presence, Dx, Dy, Dvx, Dvy).
OBS_FEATURES = 5


class SignallingHighwayEnv(HighwayEnv):
    """
    Highway env with hidden driver types and a discrete signal channel.

    step() takes a list of (action, signal) tuples, one per controlled agent:
        action: np.ndarray shape (2,), [steering in [-1,1], acceleration in [-1,1]]
        signal: int in {0, 1, 2}

    Extra info dict fields beyond base HighwayEnv:
        "types"           - (n_agents,) latent type labels
        "signals"         - (n_agents,) last signal tokens
        "nbr_window_obs"  - (n_agents, K, W, 5) neighbour kinematics history
        "nbr_window_sig"  - (n_agents, K, W) neighbour signal history
        "nbr_window_mask" - (n_agents, K) bool, True = real neighbour (not padding)
    """

    @classmethod
    def default_config(cls) -> dict:
        """Base HighwayEnv config plus our signalling fields. Override via config=."""
        cfg = super().default_config()
        cfg.update(
            {
                "controlled_vehicles": 6,
                "vehicles_count": 20,
                "lanes_count": 3,
                "duration": 40,
                "policy_frequency": 5,
                "observation": {
                    "type": "MultiAgentObservation",
                    "observation_config": {
                        "type": "Kinematics",
                        "vehicles_count": 6,
                        "absolute": False,
                        "normalize": True,
                    },
                },
                "action": {
                    "type": "MultiAgentAction",
                    "action_config": {
                        "type": "ContinuousAction",
                        "longitudinal": True,
                        "lateral": True,
                    },
                },
                "type_condition": "bimodal",
                "type_distribution": [0.5, 0.5],
                "window_steps": 5,
                "n_neighbours": 5,
                "reward_speed_range": [20, 30],
                "crash_scale": 1.0,
                "no_signal": False,
                "ungrounded": False,
            }
        )
        return cfg

    def __init__(
        self,
        config: Optional[dict] = None,
        render_mode: Optional[str] = None,
    ):
        seed = (config or {}).get("seed", None)
        self._rng = np.random.default_rng(seed)

        self._types = np.zeros(0, dtype=np.int64)
        self._signals = np.zeros(0, dtype=np.int64)

        # Rolling history per vehicle: {vehicle_id -> deque of frames}.
        # Each frame is (x, y, vx, vy, signal_token).
        self._history: dict = {}

        self._prev_accel: List[float] = []

        super().__init__(config=config, render_mode=render_mode)

        # Make the Kinematics observation match n_neighbours.
        # Must be done after super().__init__ so self.config is merged.
        n_neighbours = self.config.get("n_neighbours", 5)
        self.config["observation"]["observation_config"]["vehicles_count"] = (
            n_neighbours + 1
        )
        self.define_spaces()

        self._weight_table = self._resolve_weight_table()

    @property
    def types(self) -> np.ndarray:
        return self._types.copy()

    @property
    def signals(self) -> np.ndarray:
        return self._signals.copy()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple:
        """Reset the scene and resample every agent's hidden type."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        obs, info = super().reset(seed=seed, options=options)

        n = len(self.controlled_vehicles)

        p = np.array(self.config["type_distribution"], dtype=np.float64)
        p /= p.sum()
        self._types = self._rng.choice(N_TYPES, size=n, p=p).astype(np.int64)

        self._signals = np.zeros(n, dtype=np.int64)
        self._prev_accel = [0.0] * n

        self._history.clear()
        self._update_history()

        info["types"] = self.types
        info["signals"] = self.signals
        info.update(self._build_window_tensors())

        return obs, info

    def step(
        self,
        action_signal_pairs: Sequence[Tuple[np.ndarray, int]],
        override_profiles: Optional[List[int]] = None,
    ) -> Tuple:
        """
        Run one environment step.
        """
        raw_actions, signals = self._unpack_actions(action_signal_pairs)

        # Store the token the policy emitted so nbr_sig history matches what the sender chose.
        self._signals = np.array(signals, dtype=np.int64)

        if self.config.get("no_signal", False):
            profiles_for_kinematics = [1] * len(raw_actions)
        elif override_profiles is not None:
            profiles_for_kinematics = override_profiles
        else:
            profiles_for_kinematics = signals
        shaped_actions = self._apply_profiles(raw_actions, profiles_for_kinematics)

        for i, a in enumerate(shaped_actions):
            self._prev_accel[i] = float(a[1])

        obs, _base_reward, terminated, truncated, info = super().step(
            tuple(shaped_actions)
        )

        rewards = self._compute_rewards()

        self._update_history()
        window_data = self._build_window_tensors()

        info["types"] = self.types
        info["signals"] = self.signals
        info["agent_rewards"] = rewards
        info.update(window_data)

        return obs, rewards, terminated, truncated, info

    def _unpack_actions(
        self,
        pairs: Sequence[Tuple[np.ndarray, int]],
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Split the (action, signal) list into two parallel lists."""
        actions, signals = [], []
        for action, signal in pairs:
            actions.append(np.asarray(action, dtype=np.float32))
            signals.append(int(signal) % N_SIGNALS)
        return actions, signals

    def _apply_profiles(
        self,
        actions: List[np.ndarray],
        signals: List[int],
    ) -> List[np.ndarray]:
        """Reshape each agent's action according to its chosen profile."""
        dt = 1.0 / max(self.config.get("policy_frequency", 5), 1)

        shaped = []
        for i, (action, signal) in enumerate(zip(actions, signals)):
            profile = PROFILES[signal]
            vehicle = self.controlled_vehicles[i]

            steering = float(action[0])
            acceleration = float(action[1])

            # Jerk cap: limit how much acceleration can change per step.
            max_delta = profile.jerk_cap * dt
            prev = self._prev_accel[i]
            acceleration = float(
                np.clip(acceleration, prev - max_delta, prev + max_delta)
            )

            # Headway: cut positive acceleration when the gap is smaller than the profile wants.
            front, _ = vehicle.road.neighbour_vehicles(vehicle, vehicle.lane_index)
            if front is not None:
                pos_diff = front.position - vehicle.position
                long_gap = float(np.dot(pos_diff, vehicle.direction))
                gap = max(0.0, long_gap - vehicle.LENGTH)

                desired_gap = profile.gap_accept * profile.headway_mult

                if gap < desired_gap and acceleration > 0:
                    scale = gap / desired_gap
                    acceleration *= scale

            # Lateral drift: amplify steering above a small threshold so bold
            # lane changes look visually different from cautious ones.
            LANE_CHANGE_THRESHOLD = 0.1
            if abs(steering) > LANE_CHANGE_THRESHOLD:
                drift_factor = 1.0 + profile.lateral_drift * 0.4
                steering = float(np.clip(steering * drift_factor, -1.0, 1.0))

            shaped.append(np.array([steering, acceleration], dtype=np.float32))

        return shaped

    def _resolve_weight_table(self) -> List[TypeWeights]:
        """Choose the reward weight table based on the experimental condition."""
        condition = self.config.get("type_condition", "bimodal")
        if condition == "bimodal":
            k = float(self.config.get("divergence_scale", 1.0))
            if k == 1.0:
                return TYPE_WEIGHTS_BIMODAL
            return make_bimodal_weights(k)
        elif condition == "same_reward":
            return TYPE_WEIGHTS_SAME_REWARD
        else:
            raise ValueError(
                f"Unknown type_condition: '{condition}'. "
                "Use 'bimodal' or 'same_reward'."
            )

    def _compute_rewards(self) -> np.ndarray:
        """Per-agent reward vector, each agent weighted by its own type's table."""
        rewards = np.zeros(len(self.controlled_vehicles), dtype=np.float32)
        # multiplicative scale preserves the between-type ratio.
        crash_scale = float(self.config.get("crash_scale", 1.0))

        for i, vehicle in enumerate(self.controlled_vehicles):
            components = self._reward_components_for(vehicle)
            w = self._weight_table[self._types[i]]
            rewards[i] = (
                w.collision_penalty * crash_scale * components["collision"]
                + w.speed_reward * components["high_speed"]
                + w.right_lane_reward * components["right_lane"]
                + w.on_road_reward * components["on_road"]
            )

        return rewards

    def _reward_components_for(self, vehicle) -> dict:
        """Raw reward components for a single vehicle."""
        collision = float(vehicle.crashed)

        speed_min, speed_max = self.config.get("reward_speed_range", [20, 30])
        speed_fraction = np.clip(
            (vehicle.speed - speed_min) / (speed_max - speed_min), 0.0, 1.0
        )

        lane_idx = vehicle.lane_index[2]
        all_lanes = self.road.network.all_side_lanes(vehicle.lane_index)
        n_lanes = max(len(all_lanes), 1)
        right_lane_fraction = lane_idx / (n_lanes - 1) if n_lanes > 1 else 0.0

        on_road = float(vehicle.on_road)

        return {
            "collision": collision,
            "high_speed": float(speed_fraction),
            "right_lane": float(right_lane_fraction),
            "on_road": on_road,
        }

    def _update_history(self) -> None:
        """
        Append every vehicle's current state to its history buffer.

        Stores absolute (x, y, vx, vy); ego-relative conversion happens in
        _build_window_tensors so the maths runs once per build, not every step.
        """
        W = self.config["window_steps"]

        controlled_id_to_agent = {
            id(v): i for i, v in enumerate(self.controlled_vehicles)
        }

        for v in self.road.vehicles:
            vid = id(v)

            if vid not in self._history:
                self._history[vid] = deque(maxlen=W)

            x, y = float(v.position[0]), float(v.position[1])
            vx, vy = float(v.velocity[0]), float(v.velocity[1])

            if vid in controlled_id_to_agent and not self.config.get(
                "no_signal", False
            ):
                agent_idx = controlled_id_to_agent[vid]
                sig = int(self._signals[agent_idx])
            else:
                sig = NO_SIGNAL

            self._history[vid].append((x, y, vx, vy, sig))

    def _build_window_tensors(self) -> dict:
        """
        Build the (n_agents, K, W, 5) neighbour history tensors from the buffer.

        For each agent: find K nearest neighbours, pull their last W frames,
        make kinematics ego-relative at the same historical frame, and left-pad
        with zeros when fewer than W frames are available.
        """
        N = len(self.controlled_vehicles)
        K = self.config["n_neighbours"]
        W = self.config["window_steps"]

        nbr_obs = np.zeros((N, K, W, OBS_FEATURES), dtype=np.float32)
        nbr_sig = np.full((N, K, W), NO_SIGNAL, dtype=np.int64)
        nbr_mask = np.zeros((N, K), dtype=bool)
        # -1 = background car or empty slot (no type label).
        nbr_types = np.full((N, K), -1, dtype=np.int64)

        controlled_id_to_type = {
            id(v): int(self._types[j]) for j, v in enumerate(self.controlled_vehicles)
        }

        for i, ego in enumerate(self.controlled_vehicles):
            ego_id = id(ego)

            others = [v for v in self.road.vehicles if id(v) != ego_id]
            if not others:
                continue

            ego_pos = ego.position
            dists = [np.linalg.norm(v.position - ego_pos) for v in others]
            sort_idx = np.argsort(dists)
            neighbours = [others[j] for j in sort_idx[:K]]

            ego_history = list(self._history.get(ego_id, []))

            for k, nbr in enumerate(neighbours):
                nbr_id = id(nbr)
                nbr_history = list(self._history.get(nbr_id, []))

                if not nbr_history:
                    continue

                nbr_mask[i, k] = True
                nbr_types[i, k] = controlled_id_to_type.get(nbr_id, -1)

                n_real = len(nbr_history)
                n_ego_real = len(ego_history)
                pad_count = W - n_real

                for w in range(W):
                    if w < pad_count:
                        nbr_obs[i, k, w, 0] = 0.0
                        nbr_sig[i, k, w] = NO_SIGNAL
                        continue

                    real_w = w - pad_count
                    nbr_x, nbr_y, nbr_vx, nbr_vy, nbr_s = nbr_history[real_w]

                    # Align ego history to the same historical step as the neighbour.
                    ego_frame_idx = n_ego_real - n_real + real_w
                    if 0 <= ego_frame_idx < n_ego_real:
                        ego_x, ego_y, ego_vx, ego_vy, _ = ego_history[ego_frame_idx]
                    else:
                        # Fallback: histories don't align, use ego's current state.
                        ego_x, ego_y = ego_pos
                        ego_vx, ego_vy = float(ego.velocity[0]), float(ego.velocity[1])

                    nbr_obs[i, k, w, 0] = 1.0
                    nbr_obs[i, k, w, 1] = nbr_x - ego_x
                    nbr_obs[i, k, w, 2] = nbr_y - ego_y
                    nbr_obs[i, k, w, 3] = nbr_vx - ego_vx
                    nbr_obs[i, k, w, 4] = nbr_vy - ego_vy
                    nbr_sig[i, k, w] = nbr_s

        return {
            "nbr_window_obs": nbr_obs,
            "nbr_window_sig": nbr_sig,
            "nbr_window_mask": nbr_mask,
            "nbr_window_types": nbr_types,
        }
