"""
Loss functions for multi-modal trajectory prediction.

Key design:
  • Winner-takes-all (WTA): only backprop through the best-of-K mode per sample.
    Best mode = the one with minimum FDE to ground truth.
  • Goal loss: additional supervision on predicted endpoints.
  • Mode classification: cross-entropy with the winner as label (soft RL-style).
  • Visibility weighting: down-weight samples with low ground-truth visibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


def ade(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Average Displacement Error per sample.
    pred: (B, T, 2)
    gt:   (B, T, 2)
    returns: (B,)
    """
    return (pred - gt).norm(dim=-1).mean(dim=-1)


def fde(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Final Displacement Error per sample.
    pred: (B, T, 2)
    gt:   (B, T, 2)
    returns: (B,)
    """
    return (pred[:, -1, :] - gt[:, -1, :]).norm(dim=-1)


def min_ade_k(trajs: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    minADE_K: minimum ADE across K modes per sample.
    trajs: (B, K, T, 2)
    gt:    (B, T, 2)
    returns: (B,)
    """
    K = trajs.shape[1]
    gt_exp = gt.unsqueeze(1).expand_as(trajs)   # (B, K, T, 2)
    ade_k = (trajs - gt_exp).norm(dim=-1).mean(dim=-1)   # (B, K)
    return ade_k.min(dim=-1).values   # (B,)


def min_fde_k(trajs: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    minFDE_K: minimum FDE across K modes per sample.
    trajs: (B, K, T, 2)
    gt:    (B, T, 2)
    returns: (B,)
    """
    K = trajs.shape[1]
    gt_exp = gt.unsqueeze(1).expand_as(trajs)
    fde_k = (trajs[:, :, -1, :] - gt_exp[:, :, -1, :]).norm(dim=-1)   # (B, K)
    return fde_k.min(dim=-1).values   # (B,)


class TrajectoryLoss(nn.Module):
    """
    Combined loss:
      L = w_ade * L_ade_wta
        + w_fde * L_fde_wta
        + w_goal * L_goal
        + w_mode * L_mode_cls
    all weighted by per-sample visibility.
    """

    def __init__(self):
        super().__init__()
        t = cfg.train

        self.w_ade  = t.w_ade
        self.w_fde  = t.w_fde
        self.w_goal = t.w_goal
        self.w_mode = t.w_mode

    def forward(
        self,
        trajs: torch.Tensor,        # (B, K, T_fut, 2)
        goals: torch.Tensor,        # (B, K, 2)
        log_probs: torch.Tensor,    # (B, K)
        gt_future: torch.Tensor,    # (B, T_fut, 2)
        vis_weight: torch.Tensor,   # (B,)
    ):
        B, K, T, _ = trajs.shape
        gt_exp = gt_future.unsqueeze(1).expand_as(trajs)  # (B, K, T, 2)

        # ------------------------------------------------------------------
        # Find winner: mode with minimum FDE to ground truth
        # ------------------------------------------------------------------
        with torch.no_grad():
            fde_k = (trajs[:, :, -1, :] - gt_exp[:, :, -1, :]).norm(dim=-1)   # (B, K)
            winner = fde_k.argmin(dim=-1)   # (B,)

        # Gather winning trajectory
        idx = winner.view(B, 1, 1, 1).expand(B, 1, T, 2)
        best_traj = trajs.gather(1, idx).squeeze(1)   # (B, T, 2)

        # ------------------------------------------------------------------
        # ADE loss — winner only
        # ------------------------------------------------------------------
        l_ade = (best_traj - gt_future).norm(dim=-1).mean(dim=-1)   # (B,)

        # ------------------------------------------------------------------
        # FDE loss — winner only
        # ------------------------------------------------------------------
        l_fde = (best_traj[:, -1, :] - gt_future[:, -1, :]).norm(dim=-1)   # (B,)

        # ------------------------------------------------------------------
        # Goal loss — L2 from predicted goal endpoints to GT final position
        # Winner goal conditioned
        # ------------------------------------------------------------------
        gt_goal = gt_future[:, -1, :]   # (B, 2)
        goal_idx = winner.view(B, 1, 1).expand(B, 1, 2)
        best_goal = goals.gather(1, goal_idx).squeeze(1)   # (B, 2)
        l_goal = (best_goal - gt_goal).norm(dim=-1)        # (B,)

        # ------------------------------------------------------------------
        # Mode classification: cross-entropy, winner = label
        # ------------------------------------------------------------------
        l_mode = F.nll_loss(log_probs, winner, reduction="none")   # (B,)

        # ------------------------------------------------------------------
        # Combine with visibility weighting
        # ------------------------------------------------------------------
        total = (
            self.w_ade  * l_ade
          + self.w_fde  * l_fde
          + self.w_goal * l_goal
          + self.w_mode * l_mode
        )
        weighted = (total * vis_weight).mean()

        # Metrics for logging (no grad)
        with torch.no_grad():
            mean_ade  = min_ade_k(trajs, gt_future).mean().item()
            mean_fde  = min_fde_k(trajs, gt_future).mean().item()

        return weighted, {
            "loss":     weighted.item(),
            "ade":      mean_ade,
            "fde":      mean_fde,
            "l_ade":    l_ade.mean().item(),
            "l_fde":    l_fde.mean().item(),
            "l_goal":   l_goal.mean().item(),
            "l_mode":   l_mode.mean().item(),
        }
