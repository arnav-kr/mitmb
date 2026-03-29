"""
Visualise predicted trajectories overlaid on the map raster.

Produces one figure per agent showing:
  • Grey raster: drivable area / walkway / ped crossing channels
  • Blue line: 2s past history
  • Green line: ground-truth future
  • Red / Orange / Purple dashed lines: K predicted modes (sized by probability)
  • ADE / FDE annotations

Usage:
    python visualize.py
    python visualize.py --checkpoint ./checkpoints/best.pt --n_samples 20 --save_dir outputs/viz
"""

import argparse
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torch.utils.data import DataLoader

from config import cfg
from dataset import NuScenesDataset, collate_fn
from model import PedestrianTrajectoryPredictor


# Colour scheme
PAST_COLOR  = "#4A90D9"     # blue
GT_COLOR    = "#27AE60"     # green
PRED_COLORS = ["#E74C3C", "#E67E22", "#8E44AD"]  # red, orange, purple
TEXT_COLOR = "#E8E6DF"      # soft whitish text
LEGEND_BG_COLOR = "#202434" # dark legend panel
MAP_COLORS  = {
    "walkway":          "#D5E8D4",  # light green
    "ped_crossing":     "#DAE8FC",  # light blue
    "drivable_surface": "#F5F5F5",  # near white
}


def render_map_background(ax, map_raster: np.ndarray, patch_size: float):
    """
    map_raster: (C, H, W)
    Renders a clean dark background first, then each map layer as a colour fill.
    Handles cases where the raster may be all-zero (map not found).
    """
    H, W = map_raster.shape[1], map_raster.shape[2]
    half = patch_size / 2
    extent = [-half, half, -half, half]

    # Clamp to [0,1] in case of any float drift
    raster = np.clip(map_raster, 0.0, 1.0)

    # Dark base
    ax.set_facecolor("#000000")

    # Draw drivable surface first (channel 2), then walkway (0), then crossing (1)
    draw_order = [2, 0, 1]
    layer_names = cfg.data.map_layers

    for i in draw_order:
        if i >= len(layer_names):
            continue
        name = layer_names[i]
        mask = raster[i]   # (H, W) in [0,1]

        if mask.max() < 0.01:
            continue   # layer empty — skip rather than draw noise

        color = MAP_COLORS.get(name, "#EEEEEE")
        r = int(color[1:3], 16) / 255
        g = int(color[3:5], 16) / 255
        b = int(color[5:7], 16) / 255

        rgba = np.zeros((H, W, 4), dtype=np.float32)
        rgba[..., 0] = r
        rgba[..., 1] = g
        rgba[..., 2] = b
        rgba[..., 3] = mask * 0.75
        ax.imshow(rgba, extent=extent, origin="lower", zorder=i + 1)


def metres_to_pixel(xy: np.ndarray, patch_size: float, canvas_size: int) -> np.ndarray:
    """Convert agent-frame metres → pixel coordinates for imshow."""
    scale = canvas_size / patch_size
    px = canvas_size // 2 + xy[:, 0] * scale
    py = canvas_size // 2 + xy[:, 1] * scale   # y not flipped — imshow origin=lower
    return np.stack([px, py], axis=-1)


def visualize_sample(
    ax,
    agent_state: np.ndarray,   # (T, 5)
    gt_future:   np.ndarray,   # (T_fut, 2)
    trajs:       np.ndarray,   # (K, T_fut, 2)
    probs:       np.ndarray,   # (K,)
    map_raster:  np.ndarray,   # (C, H, W)
    title: str = "",
):
    ax.set_aspect("equal")
    ax.set_facecolor("#000000")

    patch = cfg.data.map_patch_size

    # Draw map background
    render_map_background(ax, map_raster, patch)

    # Past trajectory (columns 0,1 are x,y)
    past_xy = agent_state[:, :2]   # (T, 2)
    ax.plot(past_xy[:, 0], past_xy[:, 1],
            color=PAST_COLOR, linewidth=3, zorder=10, label="Past")
    ax.scatter(past_xy[-1, 0], past_xy[-1, 1],
               color=PAST_COLOR, s=60, zorder=11)

    origin = np.array([[0.0, 0.0]])
    gt_connected = np.concatenate([origin, gt_future], axis=0)   # (T_fut+1, 2)

    # Small star at the exact prediction origin
    ax.scatter([0], [0], color="white", s=80, marker="*", zorder=12)

    # Ground truth future — connected from origin
    ax.plot(gt_connected[:, 0], gt_connected[:, 1],
            color=GT_COLOR, linewidth=3, zorder=10, label="Ground truth")
    ax.scatter(gt_connected[-1, 0], gt_connected[-1, 1],
               color=GT_COLOR, s=80, marker="*", zorder=11)

    # Predicted trajectories — each connected from origin
    for k in range(len(trajs)):
        color = PRED_COLORS[k % len(PRED_COLORS)]
        lw = max(1.5, 3.0 * float(probs[k]))
        traj_connected = np.concatenate([origin, trajs[k]], axis=0)   # (T_fut+1, 2)
        ax.plot(traj_connected[:, 0], traj_connected[:, 1],
                color=color, linewidth=lw, linestyle="--",
                zorder=9, label=f"M{k+1} p={probs[k]:.2f}")
        ax.scatter(traj_connected[-1, 0], traj_connected[-1, 1],
                   color=color, s=50, zorder=10)

    # Compute ADE / FDE vs ground truth (best mode)
    ade_k = np.mean(np.linalg.norm(trajs - gt_future[None], axis=-1), axis=-1)
    fde_k = np.linalg.norm(trajs[:, -1, :] - gt_future[-1], axis=-1)
    min_ade = ade_k.min()
    min_fde = fde_k.min()

    # Dynamic zoom around the actual trajectories.
    # This keeps local motion legible while avoiding extreme crop jitter.
    pts = [past_xy, gt_future, trajs.reshape(-1, 2)]
    all_pts = np.concatenate(pts, axis=0)
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)

    pad = 0.8
    span_x = max(4.0, (x_max - x_min) + 2 * pad)
    span_y = max(4.0, (y_max - y_min) + 2 * pad)
    span = min(12.0, max(span_x, span_y))

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    half = 0.5 * span

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_xlabel("x (m)", fontsize=8, color=TEXT_COLOR)
    ax.set_ylabel("y (m)", fontsize=8, color=TEXT_COLOR)
    ax.set_title(
        f"{title}\nminADE={min_ade:.2f}m  minFDE={min_fde:.2f}m",
        fontsize=8,
        color=TEXT_COLOR,
    )
    ax.tick_params(axis="both", colors=TEXT_COLOR, labelsize=7)
    ax.grid(True, color="#A9B1D6", alpha=0.22, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)
        spine.set_alpha(0.5)

    legend = ax.legend(
        loc="upper right",
        fontsize=6,
        framealpha=0.92,
        facecolor=LEGEND_BG_COLOR,
        edgecolor=TEXT_COLOR,
    )
    for txt in legend.get_texts():
        txt.set_color(TEXT_COLOR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    if args.dataroot:
        cfg.data.dataroot = args.dataroot

    ckpt_path = args.checkpoint or os.path.join(cfg.train.checkpoint_dir, "best.pt")
    save_dir  = args.save_dir
    n_samples = args.n_samples
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = PedestrianTrajectoryPredictor().to(device)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[viz] Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")
    else:
        print("[viz] No checkpoint found — using random weights (for pipeline testing)")
    model.eval()

    # Data
    val_ds = NuScenesDataset(split="val")
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
    )

    n_cols = 2
    n_rows = math.ceil(n_samples / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten() if n_samples > 1 else [axes]
    fig.patch.set_facecolor("#1E1E1E")

    count = 0
    for batch in val_loader:
        if batch is None or count >= n_samples:
            break

        if batch["future"].shape[1] < cfg.data.future_steps:
            continue

        gt_future = batch["future"][0].numpy()
        if np.allclose(gt_future, 0, atol=1e-3) or np.max(np.linalg.norm(np.diff(gt_future, axis=0), axis=1)) < 1e-2:
            continue

        with torch.no_grad():
            trajs, probs = model.predict(
                batch["agent_state"].to(device),
                batch["neighbor_states"].to(device),
                batch["neighbor_mask"].to(device),
                batch["map_raster"].to(device),
            )

        pred_flat = trajs[0].cpu().numpy().reshape(-1, 2)
        if np.allclose(pred_flat, 0, atol=1e-3) or np.max(np.linalg.norm(np.diff(pred_flat, axis=0), axis=1)) < 1e-2:
            continue

        visualize_sample(
            axes[count],
            agent_state=batch["agent_state"][0].numpy(),
            gt_future=  gt_future,
            trajs=       trajs[0].cpu().numpy(),
            probs=       probs[0].cpu().numpy(),
            map_raster=  batch["map_raster"][0].numpy(),
            title=f"Agent {count+1}",
        )
        count += 1

    # Hide unused axes
    for i in range(count, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "trajectory_predictions.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[viz] Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot",   type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n_samples",  type=int, default=9)
    parser.add_argument("--save_dir",   type=str, default="outputs/viz")
    args = parser.parse_args()
    main(args)