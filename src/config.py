from dataclasses import dataclass, field
from typing import List

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Pick the device to run on. 'auto' tries CUDA, then MPS, then CPU."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Execution profiles = the signal vocabulary M (3 tokens).
# Each token maps to a driving style with distinct kinematic limits.
# Values come from Table 1 of the proposal.


@dataclass(frozen=True)
class Profile:
    """One driving style (its kinematic limits)."""

    name: str
    headway_mult: float  # multiplier on the safe following gap (>1 = wants more space)
    jerk_cap: float  # max change in acceleration per second (m/s^3)
    gap_accept: float  # min gap in metres before the agent decelerates
    lateral_drift: float  # steering amplification when starting a lane change


# These line up with signal tokens 0, 1, 2.
PROFILES: List[Profile] = [
    Profile(
        name="conservative",
        headway_mult=1.5,
        jerk_cap=1.0,
        gap_accept=15.0,
        lateral_drift=0.3,
    ),
    Profile(
        name="neutral",
        headway_mult=1.0,
        jerk_cap=2.0,
        gap_accept=10.0,
        lateral_drift=0.6,
    ),
    Profile(
        name="assertive",
        headway_mult=0.7,
        jerk_cap=4.0,
        gap_accept=6.0,
        lateral_drift=1.0,
    ),
]

N_SIGNALS = len(PROFILES)  # |M| = 3


# Driver types = the private type space Theta.
# Sampled once per episode; never revealed to other agents.
# In the bimodal condition the two reward tables differ meaningfully,
# giving each type a reason to signal at all.


@dataclass(frozen=True)
class TypeWeights:
    """Reward weights for one driver type."""

    collision_penalty: float
    speed_reward: float
    right_lane_reward: float
    on_road_reward: float


# type 0 = cautious, type 1 = assertive
CAUTIOUS = TypeWeights(
    collision_penalty=-2.0,
    speed_reward=0.2,
    right_lane_reward=0.2,
    on_road_reward=1.0,
)
ASSERTIVE = TypeWeights(
    collision_penalty=-1.5,
    speed_reward=0.8,
    right_lane_reward=0.05,
    on_road_reward=1.0,
)

# same_reward: RQ2 control — both labels get the same weights, so the type
# token has no behavioural meaning.
TYPE_WEIGHTS_BIMODAL = [CAUTIOUS, ASSERTIVE]
TYPE_WEIGHTS_SAME_REWARD = [CAUTIOUS, CAUTIOUS]


def make_bimodal_weights(k: float) -> list:
    """Interpolate TypeWeights at divergence scale k. k=1.0 = bimodal, k=0 = midpoint (NOT equal to same-reward baseline)."""
    cp = (CAUTIOUS.collision_penalty + ASSERTIVE.collision_penalty) / 2
    sp = (CAUTIOUS.speed_reward + ASSERTIVE.speed_reward) / 2
    rl = (CAUTIOUS.right_lane_reward + ASSERTIVE.right_lane_reward) / 2
    on = (CAUTIOUS.on_road_reward + ASSERTIVE.on_road_reward) / 2

    dcp = (CAUTIOUS.collision_penalty - ASSERTIVE.collision_penalty) / 2
    dsp = (CAUTIOUS.speed_reward - ASSERTIVE.speed_reward) / 2
    drl = (CAUTIOUS.right_lane_reward - ASSERTIVE.right_lane_reward) / 2
    don = (CAUTIOUS.on_road_reward - ASSERTIVE.on_road_reward) / 2

    new_c = TypeWeights(
        collision_penalty=cp + k * dcp,
        speed_reward=sp + k * dsp,
        right_lane_reward=rl + k * drl,
        on_road_reward=on + k * don,
    )
    new_a = TypeWeights(
        collision_penalty=cp - k * dcp,
        speed_reward=sp - k * dsp,
        right_lane_reward=rl - k * drl,
        on_road_reward=on - k * don,
    )
    return [new_c, new_a]


N_TYPES = 2  # |Theta|


@dataclass
class EnvConfig:
    """Settings for the highway simulation."""

    n_agents: int = 6
    n_bg_vehicles: int = 0
    n_lanes: int = 4
    episode_duration: int = 40
    policy_frequency: int = 5

    window_steps: int = 5  # W: history frames fed to the GRU
    n_neighbours: int = 5  # K: max neighbours each agent can see

    # features per vehicle per frame: (presence, Dx, Dy, Dvx, Dvy), relative to ego
    obs_features: int = 5

    # "bimodal" = main condition (RQ1), "same_reward" = control (RQ2)
    type_condition: str = "bimodal"


@dataclass
class NetworkConfig:
    """Network sizes, shared by all neural net components."""

    embed_dim: int = 64  # GRU hidden size = one neighbour embedding size
    agg_dim: int = 64  # DeepSets output size
    hidden_dim: int = 64  # MLP hidden width
    action_dim: int = 2  # continuous action: [steering, acceleration]


@dataclass
class TrainConfig:
    """MAPPO training settings."""

    total_iters: int = 1600
    rollout_steps: int = 512

    ppo_epochs: int = 4
    minibatch_size: int = 128
    clip_eps: float = 0.2
    value_clip_eps: float = 5.0  # clip for critic
    grad_clip: float = 0.5

    gamma: float = 0.99
    gae_lambda: float = 0.95

    lr: float = 3e-4
    value_coef: float = 0.5

    entropy_coef_action: float = 0.001
    entropy_coef_signal_start: float = 0.08  # signal entropy bonus at iter 0
    entropy_coef_signal_end: float = 0.005  # after annealing
    entropy_anneal_iters: int = 1200

    aux_coef: float = 0.5

    seed: int = 0
    device: str = "cpu"
    log_every: int = 10
    checkpoint_every: int = 50
    save_dir: str = "checkpoints"


@dataclass
class EvalConfig:
    """Settings for the evaluation script."""

    n_episodes: int = 200  # episodes per condition (proposal sec.4.2)
    seed: int = 9999  # separate from the training seeds
    device: str = "cpu"
