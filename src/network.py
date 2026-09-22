from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class TypeEncoder(nn.Module):
    def __init__(self, obs_features: int, n_signals: int, embed_dim: int = 64):
        super().__init__()
        self.obs_features = obs_features
        self.n_signals = n_signals
        self.embed_dim = embed_dim

        self.gru = nn.GRU(
            input_size=obs_features + n_signals,
            hidden_size=embed_dim,
            batch_first=True,
        )

    def forward(
        self,
        obs_seq: torch.Tensor,  # (B, W, obs_features)
        signal_seq: torch.Tensor,  # (B, W) integer signal tokens
    ) -> torch.Tensor:
        # NO_SIGNAL slots (== -1) are clamped to 0 then zeroed out so padding
        # frames contribute no signal information.
        valid = (signal_seq >= 0).unsqueeze(-1)
        sig_onehot = (
            F.one_hot(
                signal_seq.long().clamp(min=0), num_classes=self.n_signals
            ).float()
            * valid
        )

        gru_input = torch.cat([obs_seq, sig_onehot], dim=-1)
        _, h = self.gru(gru_input)
        return h.squeeze(0)  # (1, B, embed_dim) -> (B, embed_dim)


class DeepSets(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 64):
        super().__init__()

        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        embeddings: torch.Tensor,  # (B, K, in_dim)
        mask: Optional[torch.Tensor] = None,  # (B, K) bool, True = valid
    ) -> torch.Tensor:
        h = self.phi(embeddings)

        if mask is not None:
            h = h * mask.unsqueeze(-1).float()

        s = h.sum(dim=1)
        return self.rho(s)


class SignalHead(nn.Module):
    def __init__(
        self,
        obs_features: int,
        agg_dim: int,
        n_types: int,
        n_signals: int,
        hidden_dim: int = 64,
        type_embed_dim: int = 16,
    ):
        super().__init__()

        self.type_embedding = nn.Embedding(n_types, type_embed_dim)

        in_dim = obs_features + type_embed_dim + agg_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_signals),
        )

    def forward(
        self,
        self_obs: torch.Tensor,  # (B, obs_features)
        own_type: torch.Tensor,  # (B,) integer type labels
        agg: torch.Tensor,  # (B, agg_dim)
    ) -> Categorical:
        """Return a Categorical over {0, ..., n_signals-1}."""
        type_emb = self.type_embedding(own_type.long())
        x = torch.cat([self_obs, type_emb, agg], dim=-1)
        return Categorical(logits=self.net(x))


class ResponseHead(nn.Module):
    """Receiver: (self_obs, neighbourhood, own_signal) -> factorised Gaussian action distribution.

    Signal token embedded and concatenated so the action head learns profile-optimal
    targets within the committed kinematic envelope (option-conditioned policy).
    """

    def __init__(
        self,
        obs_features: int,
        agg_dim: int,
        n_signals: int,
        action_dim: int = 2,
        hidden_dim: int = 64,
        signal_embed_dim: int = 8,
    ):
        super().__init__()

        self.signal_embedding = nn.Embedding(n_signals, signal_embed_dim)

        in_dim = obs_features + agg_dim + signal_embed_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)

        # Init -1.0 -> std=0.37; clamped to [-3, 0.5] -> std in [0.05, 1.65].
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

    def forward(
        self,
        self_obs: torch.Tensor,  # (B, obs_features)
        agg: torch.Tensor,  # (B, agg_dim)
        signal: torch.Tensor,  # (B,) committed hard signal token
        no_signal: bool = False,
    ) -> Normal:
        """Return a Normal with tanh-squashed mean in [-1, 1]."""
        if no_signal:
            sig_emb = torch.zeros(
                signal.shape[0],
                self.signal_embedding.embedding_dim,
                device=signal.device,
            )
        else:
            sig_emb = self.signal_embedding(signal.long())
        x = torch.cat([self_obs, agg, sig_emb], dim=-1)
        h = self.trunk(x)
        mean = torch.tanh(self.mean_layer(h))
        std = self.log_std.clamp(-3.0, 0.5).exp().expand_as(mean)
        return Normal(mean, std)


class AuxHead(nn.Module):
    """Auxiliary classifier: neighbour embedding -> type logits. Supervised during training only."""

    def __init__(self, embed_dim: int, n_types: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_types),
        )

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        """embed: (B, embed_dim) -> (B, n_types) type-classification logits."""
        return self.net(embed)


class SignallingPolicy(nn.Module):
    """Full per-agent policy (shared weights): TypeEncoder + DeepSets + signal/action/aux heads."""

    def __init__(
        self,
        obs_features: int,
        n_signals: int,
        n_types: int,
        action_dim: int = 2,
        embed_dim: int = 64,
        agg_dim: int = 64,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.encoder = TypeEncoder(obs_features, n_signals, embed_dim)
        self.aggregator = DeepSets(embed_dim, hidden_dim, agg_dim)
        self.signal_head = SignalHead(
            obs_features, agg_dim, n_types, n_signals, hidden_dim
        )
        self.response_head = ResponseHead(
            obs_features, agg_dim, n_signals, action_dim, hidden_dim
        )
        self.aux_head = AuxHead(embed_dim, n_types, hidden_dim)

    def encode_neighbours(
        self,
        nbr_obs: torch.Tensor,  # (B, K, W, obs_features)
        nbr_sig: torch.Tensor,  # (B, K, W)
        mask: Optional[torch.Tensor] = None,  # (B, K) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the GRU on each of K neighbour windows, then aggregate.

        Flattens (B, K) into one batch dim so the GRU processes all windows
        at once, then restores the (B, K) structure.

        Returns (embeds: (B, K, embed_dim), agg: (B, agg_dim)).
        """
        B, K, W, F = nbr_obs.shape

        flat_obs = nbr_obs.reshape(B * K, W, F)
        flat_sig = nbr_sig.reshape(B * K, W)

        flat_embed = self.encoder(flat_obs, flat_sig)
        embeds = flat_embed.reshape(B, K, -1)
        agg = self.aggregator(embeds, mask=mask)

        return embeds, agg

    def forward(
        self,
        self_obs: torch.Tensor,  # (B, obs_features)
        own_type: torch.Tensor,  # (B,) integer type tokens
        nbr_obs: torch.Tensor,  # (B, K, W, obs_features)
        nbr_sig: torch.Tensor,  # (B, K, W)
        nbr_mask: Optional[torch.Tensor] = None,  # (B, K)
        no_signal: bool = False,
    ) -> dict:
        """
        One forward pass: encode neighbours, sample an action and a signal.

        Returns a dict with action, action_logp, signal, signal_logp,
        embeds, agg, signal_dist, action_dist.
        """
        embeds, agg = self.encode_neighbours(nbr_obs, nbr_sig, mask=nbr_mask)

        if no_signal:
            B = self_obs.shape[0]
            signal = torch.ones(B, dtype=torch.long, device=self_obs.device)
            signal_logp = torch.zeros(B, device=self_obs.device)
            signal_dist = None
        else:
            signal_dist = self.signal_head(self_obs, own_type, agg)
            signal = signal_dist.sample()
            signal_logp = signal_dist.log_prob(signal)

        # own_type NOT passed; signal IS passed (hard sample, already committed).
        action_dist = self.response_head(self_obs, agg, signal, no_signal=no_signal)
        action = action_dist.rsample()
        action_logp = action_dist.log_prob(action).sum(dim=-1)

        return {
            "action": action,
            "action_logp": action_logp,
            "signal": signal,
            "signal_logp": signal_logp,
            "embeds": embeds,
            "agg": agg,
            "signal_dist": signal_dist,
            "action_dist": action_dist,
        }


class CentralisedCritic(nn.Module):
    def __init__(
        self,
        obs_features: int,
        agg_dim: int,
        n_types: int,
        n_agents: int,
        type_embed_dim: int = 16,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.agg_dim = agg_dim

        self.type_embedding = nn.Embedding(n_types, type_embed_dim)

        in_dim = n_agents * (obs_features + type_embed_dim + agg_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        self_obs_all: torch.Tensor,  # (B, n_agents, obs_features)
        types_all: torch.Tensor,  # (B, n_agents)
        agg_all: torch.Tensor,  # (B, n_agents, agg_dim)
    ) -> torch.Tensor:
        """Return (B,) state values V(s)."""
        type_embs = self.type_embedding(types_all.long())
        x = torch.cat([self_obs_all, type_embs, agg_all], dim=-1)
        x = x.flatten(start_dim=1)
        return self.net(x).squeeze(-1)
