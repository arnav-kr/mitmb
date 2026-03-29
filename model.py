"""
PedestrianTrajectoryPredictor

Architecture overview:
  AgentEncoder     — MLP encodes (T, 5) past states → single token (hidden_dim)
  NeighbourEncoder — shared MLP encodes each neighbour (T, 4) → token
  SocialAttention  — cross-attention: ego queries over neighbours → social context
  MapEncoder       — small CNN: (C, H, W) raster → map feature vector
  GoalPredictor    — from [social + map] context → K goal points at t+3s
  TrajectoryDecoder— GRU conditioned on goal → 6-step trajectory per mode
  Classifier       — predicts probability for each mode

All internal dims follow cfg.model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


# ---------------------------------------------------------------------------
# Agent encoder
# ---------------------------------------------------------------------------

class AgentEncoder(nn.Module):
    """
    Encodes the ego-agent's past trajectory.
    Input:  (B, T, agent_input_size)
    Output: (B, hidden_dim)
    """

    def __init__(self):
        super().__init__()
        c = cfg.model
        T = cfg.data.past_steps
        in_dim = c.agent_input_size * T   # flatten time

        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, c.hidden_dim),
            nn.LayerNorm(c.hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return self.net(x.reshape(B, -1))   # (B, hidden_dim)


# ---------------------------------------------------------------------------
# Neighbour encoder (shared weights across neighbours)
# ---------------------------------------------------------------------------

class NeighbourEncoder(nn.Module):
    """
    Shared MLP that maps each neighbour's past trajectory to a token.
    Input:  (B, N, T, neighbor_input_size)
    Output: (B, N, hidden_dim)
    """

    def __init__(self):
        super().__init__()
        c = cfg.model
        T = cfg.data.past_steps
        in_dim = c.neighbor_input_size * T

        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, c.hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, T, F = x.shape
        flat = x.reshape(B * N, T * F)
        out = self.net(flat)
        return out.reshape(B, N, -1)   # (B, N, hidden_dim)


# ---------------------------------------------------------------------------
# Social attention
# ---------------------------------------------------------------------------

class SocialAttention(nn.Module):
    """
    Multi-head cross-attention: ego token (query) attends over neighbour tokens (keys/values).
    Output: ego social-context vector (B, hidden_dim)
    """

    def __init__(self):
        super().__init__()
        c = cfg.model
        self.attn = nn.MultiheadAttention(
            embed_dim=c.hidden_dim,
            num_heads=c.num_heads,
            dropout=c.dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(c.hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(c.hidden_dim, c.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(c.hidden_dim * 2, c.hidden_dim),
        )

    def forward(
        self,
        agent_token: torch.Tensor,    # (B, hidden_dim)
        neighbor_tokens: torch.Tensor, # (B, N, hidden_dim)
        neighbor_mask: torch.Tensor,   # (B, N) bool — True = valid
    ) -> torch.Tensor:
        """Returns social-context vector (B, hidden_dim)."""
        B = agent_token.shape[0]
        query = agent_token.unsqueeze(1)   # (B, 1, hidden_dim)

        # MultiheadAttention key_padding_mask: True = IGNORE
        # neighbor_mask: True = VALID → invert
        key_padding_mask = ~neighbor_mask   # (B, N)  True = padding

        # If all neighbours are masked for a sample, temporarily un-mask slot 0
        # to avoid NaN from attending over nothing
        all_masked = key_padding_mask.all(dim=1, keepdim=True)   # (B, 1)
        key_padding_mask = key_padding_mask & ~all_masked         # safe

        attn_out, _ = self.attn(
            query=query,
            key=neighbor_tokens,
            value=neighbor_tokens,
            key_padding_mask=key_padding_mask,
        )
        attn_out = attn_out.squeeze(1)   # (B, hidden_dim)

        # Residual + feedforward
        out = self.norm(agent_token + attn_out)
        out = self.norm(out + self.ff(out))
        return out   # (B, hidden_dim)


# ---------------------------------------------------------------------------
# Map encoder (lightweight CNN)
# ---------------------------------------------------------------------------

class MapEncoder(nn.Module):
    """
    Small CNN that compresses a (C, H, W) binary raster into a feature vector.
    C = len(map_layers) = 3, H = W = 100
    """

    def __init__(self):
        super().__init__()
        C = len(cfg.data.map_layers)
        d = cfg.model.map_feat_dim

        self.cnn = nn.Sequential(
            # 3×100×100 → 16×50×50
            nn.Conv2d(C, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            # 16×50×50 → 32×25×25
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            # 32×25×25 → 64×12×12
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # 64×12×12 → 64×6×6
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(64, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)  →  (B, map_feat_dim)"""
        feat = self.cnn(x)          # (B, 64, 6, 6)
        feat = self.pool(feat)      # (B, 64, 1, 1)
        feat = feat.flatten(1)      # (B, 64)
        return self.proj(feat)      # (B, map_feat_dim)


# ---------------------------------------------------------------------------
# Context fusion
# ---------------------------------------------------------------------------

class ContextFusion(nn.Module):
    """Fuses social vector and map feature into a single context vector."""

    def __init__(self):
        super().__init__()
        c = cfg.model
        in_dim = c.hidden_dim + c.map_feat_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, c.hidden_dim),
            nn.LayerNorm(c.hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        social: torch.Tensor,  # (B, hidden_dim)
        map_feat: torch.Tensor, # (B, map_feat_dim)
    ) -> torch.Tensor:
        return self.net(torch.cat([social, map_feat], dim=-1))  # (B, hidden_dim)


# ---------------------------------------------------------------------------
# Goal predictor (multi-modal endpoints)
# ---------------------------------------------------------------------------

class GoalPredictor(nn.Module):
    """
    Predicts K endpoint candidates (x, y) at t+future_seconds.
    Goal-conditioned prediction: predict destinations first, then decode paths.
    """

    def __init__(self):
        super().__init__()
        c = cfg.model
        K = c.num_modes
        self.max_goal_radius = c.max_goal_radius

        self.net = nn.Sequential(
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.GELU(),
            nn.Linear(c.hidden_dim, K * 2),   # K × (x, y)
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """context: (B, hidden_dim) → goals: (B, K, 2)"""
        B = context.shape[0]
        K = cfg.model.num_modes
        raw = self.net(context).reshape(B, K, 2)
        return torch.tanh(raw) * self.max_goal_radius


# ---------------------------------------------------------------------------
# Trajectory decoder
# ---------------------------------------------------------------------------

class TrajectoryDecoder(nn.Module):
    """
    GRU decoder conditioned on a goal point.
    For each mode k, uses [context || goal_k] as GRU initial hidden state
    and decodes future_steps waypoints autoregressively.
    """

    def __init__(self):
        super().__init__()
        c = cfg.model
        T_fut = cfg.data.future_steps

        # Project [context + goal (2d)] → GRU hidden init
        self.hidden_init = nn.Linear(c.hidden_dim + 2, c.hidden_dim)

        # GRU input at each step: previous waypoint (2d)
        self.gru = nn.GRU(
            input_size=2,
            hidden_size=c.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=c.dropout,
        )

        # Project GRU output → (x, y) delta
        self.out_proj = nn.Linear(c.hidden_dim, 2)

        self.T_fut = T_fut

    def forward(
        self,
        context: torch.Tensor,  # (B, hidden_dim)
        goals: torch.Tensor,    # (B, K, 2)
    ) -> torch.Tensor:
        """Returns trajectories (B, K, T_fut, 2) in agent frame."""
        B, K, _ = goals.shape
        trajs = []

        for k in range(K):
            goal_k = goals[:, k, :]   # (B, 2)
            h_init = torch.tanh(
                self.hidden_init(torch.cat([context, goal_k], dim=-1))
            )  # (B, hidden_dim)

            # Stack for num_layers=2 GRU
            h = h_init.unsqueeze(0).expand(2, -1, -1).contiguous()  # (2, B, hidden)

            # Start token: zeros (agent is at origin in agent frame)
            step_in = torch.zeros(B, 1, 2, device=context.device)

            steps = []
            pos = torch.zeros(B, 1, 2, device=context.device)
            for _ in range(self.T_fut):
                gru_out, h = self.gru(step_in, h)  # (B, 1, hidden)
                delta = self.out_proj(gru_out)      # (B, 1, 2)
                pos = pos + delta
                steps.append(pos)
                step_in = delta   # feed previous output as next input

            traj_k = torch.cat(steps, dim=1)   # (B, T_fut, 2)

            # Encourage geometric consistency: endpoint should align with goal_k.
            # Apply a linear correction ramp so early points move less than late points.
            end_err = (goal_k - traj_k[:, -1, :]).unsqueeze(1)  # (B, 1, 2)
            ramp = torch.linspace(
                1.0 / self.T_fut,
                1.0,
                self.T_fut,
                device=context.device,
                dtype=traj_k.dtype,
            ).view(1, self.T_fut, 1)
            traj_k = traj_k + ramp * end_err

            trajs.append(traj_k)

        return torch.stack(trajs, dim=1)   # (B, K, T_fut, 2)


# ---------------------------------------------------------------------------
# Mode classifier
# ---------------------------------------------------------------------------

class ModeClassifier(nn.Module):
    """Predicts log-softmax probability over K modes."""

    def __init__(self):
        super().__init__()
        c = cfg.model
        self.net = nn.Linear(c.hidden_dim, c.num_modes)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Returns log-probabilities (B, K)."""
        return F.log_softmax(self.net(context), dim=-1)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class PedestrianTrajectoryPredictor(nn.Module):
    """
    End-to-end multi-modal goal-conditioned trajectory predictor.

    Forward inputs (all as batch tensors):
        agent_state     (B, T, 5)
        neighbor_states (B, N, T, 4)
        neighbor_mask   (B, N) bool
        map_raster      (B, C, H, W)

    Forward outputs:
        trajectories    (B, K, T_fut, 2)  — K predicted paths
        goals           (B, K, 2)         — predicted endpoints
        log_probs       (B, K)            — log-softmax mode scores
    """

    def __init__(self):
        super().__init__()
        self.agent_enc    = AgentEncoder()
        self.neighbor_enc = NeighbourEncoder()
        self.social_attn  = SocialAttention()
        self.map_enc      = MapEncoder()
        self.fusion       = ContextFusion()
        self.goal_pred    = GoalPredictor()
        self.traj_dec     = TrajectoryDecoder()
        self.mode_clf     = ModeClassifier()

    def forward(
        self,
        agent_state: torch.Tensor,      # (B, T, 5)
        neighbor_states: torch.Tensor,  # (B, N, T, 4)
        neighbor_mask: torch.Tensor,    # (B, N) bool
        map_raster: torch.Tensor,       # (B, C, H, W)
    ):
        # 1. Encode ego agent
        agent_token = self.agent_enc(agent_state)           # (B, hidden)

        # 2. Encode neighbours (shared weights)
        nb_tokens = self.neighbor_enc(neighbor_states)      # (B, N, hidden)

        # 3. Social cross-attention
        social = self.social_attn(agent_token, nb_tokens, neighbor_mask)  # (B, hidden)

        # 4. Encode map
        map_feat = self.map_enc(map_raster)                 # (B, map_feat_dim)

        # 5. Fuse social + map
        context = self.fusion(social, map_feat)             # (B, hidden)

        # 6. Predict K goal endpoints
        goals = self.goal_pred(context)                     # (B, K, 2)

        # 7. Decode K trajectories, each conditioned on its goal
        trajectories = self.traj_dec(context, goals)        # (B, K, T_fut, 2)

        # 8. Mode classification
        log_probs = self.mode_clf(context)                  # (B, K)

        return trajectories, goals, log_probs

    def predict(
        self,
        agent_state: torch.Tensor,
        neighbor_states: torch.Tensor,
        neighbor_mask: torch.Tensor,
        map_raster: torch.Tensor,
    ):
        """Inference: returns top-K trajectories sorted by probability (descending)."""
        with torch.no_grad():
            trajs, goals, log_probs = self.forward(
                agent_state, neighbor_states, neighbor_mask, map_raster
            )
        probs = log_probs.exp()   # (B, K)
        # Sort modes by probability
        order = probs.argsort(dim=-1, descending=True)   # (B, K)
        B, K = order.shape
        trajs_sorted = trajs[
            torch.arange(B).unsqueeze(1).expand(B, K),
            order
        ]
        probs_sorted = probs[
            torch.arange(B).unsqueeze(1).expand(B, K),
            order
        ]
        return trajs_sorted, probs_sorted


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
