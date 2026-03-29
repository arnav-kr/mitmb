"""
Evaluation script — runs on the nuScenes val split and reports:
  • minADE_K
  • minFDE_K
  • MissRate_2_K  (miss if max pointwise L2 > 2m)
  • OffRoadRate   (fraction of trajectories with any point off drivable area)

Usage:
    python evaluate.py
    python evaluate.py --checkpoint ./checkpoints/best.pt --dataroot /path/to/nuscenes
"""

import argparse
import json
import os
import math
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import cfg
from dataset import NuScenesDataset, collate_fn
from model import PedestrianTrajectoryPredictor


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_min_ade(trajs: np.ndarray, gt: np.ndarray) -> float:
    """
    trajs: (K, T, 2)
    gt:    (T, 2)
    Returns minADE_K (scalar).
    """
    ade_per_mode = np.mean(np.linalg.norm(trajs - gt[None], axis=-1), axis=-1)  # (K,)
    return float(ade_per_mode.min())


def compute_min_fde(trajs: np.ndarray, gt: np.ndarray) -> float:
    """
    trajs: (K, T, 2)
    gt:    (T, 2)
    Returns minFDE_K (scalar).
    """
    fde_per_mode = np.linalg.norm(trajs[:, -1, :] - gt[-1], axis=-1)   # (K,)
    return float(fde_per_mode.min())


def compute_miss_rate(trajs: np.ndarray, gt: np.ndarray, threshold: float = 2.0) -> float:
    """
    A sample is a miss if ALL K modes have max pointwise L2 > threshold.
    Returns 1.0 if miss, 0.0 if at least one mode is within threshold.
    """
    for k in range(len(trajs)):
        max_l2 = np.linalg.norm(trajs[k] - gt, axis=-1).max()
        if max_l2 <= threshold:
            return 0.0   # at least one mode within threshold
    return 1.0            # all modes missed


def is_off_road(traj: np.ndarray, map_raster: np.ndarray, patch_size: float) -> bool:
    """
    Check if any point in traj (T, 2) is off the drivable-area channel of map_raster.
    map_raster: (C, H, W) in agent frame, channel 2 = drivable_surface.
    patch_size: metres covered by the full raster (cfg.data.map_patch_size).
    Returns True if any point lands on a non-drivable pixel.
    """
    C, H, W = map_raster.shape
    drivable = map_raster[2]   # channel index 2 = drivable_surface

    for pt in traj:
        # Convert agent-frame metres → pixel index
        # Centre of map = (H//2, W//2); scale = H / patch_size px/m
        scale = H / patch_size
        px = int(W // 2 + pt[0] * scale)
        py = int(H // 2 - pt[1] * scale)   # y-axis is flipped in image coords

        if px < 0 or px >= W or py < 0 or py >= H:
            return True   # out of map bounds = off road
        if drivable[py, px] < 0.5:
            return True   # non-drivable pixel
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    if args.dataroot:
        cfg.data.dataroot = args.dataroot
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = os.path.join(cfg.train.checkpoint_dir, "best.pt")

    os.makedirs("outputs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device}")
    print(f"[eval] Checkpoint: {ckpt_path}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model = PedestrianTrajectoryPredictor().to(device)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[eval] Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    else:
        print(f"[eval] WARNING: checkpoint not found at {ckpt_path}. Using random weights.")

    model.eval()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    val_ds = NuScenesDataset(split="val")
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=cfg.data.pin_memory,
    )

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    all_min_ade  = []
    all_min_fde  = []
    all_miss     = []
    all_off_road = []
    all_predictions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch is None:
                continue

            agent_state     = batch["agent_state"].to(device)
            neighbor_states = batch["neighbor_states"].to(device)
            neighbor_mask   = batch["neighbor_mask"].to(device)
            map_raster      = batch["map_raster"].to(device)
            gt_future       = batch["future"]          # keep on CPU for numpy ops

            trajs, probs = model.predict(
                agent_state, neighbor_states, neighbor_mask, map_raster
            )
            # trajs: (B, K, T, 2),  probs: (B, K)  — sorted desc by prob
            trajs_np  = trajs.cpu().numpy()
            probs_np  = probs.cpu().numpy()
            gt_np     = gt_future.numpy()
            map_np    = batch["map_raster"].numpy()   # (B, C, H, W)

            B = trajs_np.shape[0]
            for i in range(B):
                t  = trajs_np[i]   # (K, T, 2)
                g  = gt_np[i]      # (T, 2)
                pr = probs_np[i]   # (K,)

                ade_i  = compute_min_ade(t, g)
                fde_i  = compute_min_fde(t, g)
                miss_i = compute_miss_rate(t, g, cfg.eval.miss_threshold)

                # Off-road: check the most-likely predicted trajectory
                off_i = is_off_road(t[0], map_np[i], cfg.data.map_patch_size)

                all_min_ade.append(ade_i)
                all_min_fde.append(fde_i)
                all_miss.append(miss_i)
                all_off_road.append(float(off_i))

                # Store prediction in submission-ready format
                all_predictions.append({
                    "instance_token": batch["instance_tokens"][i],
                    "sample_token":   batch["sample_tokens"][i],
                    "prediction":     t.tolist(),         # K×T×2
                    "probabilities":  pr.tolist(),         # K
                })

            if (batch_idx + 1) % 20 == 0:
                print(
                    f"  batch {batch_idx+1} | "
                    f"ADE so far: {np.mean(all_min_ade):.3f} | "
                    f"FDE: {np.mean(all_min_fde):.3f}"
                )

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    results = {
        f"minADE_{cfg.eval.top_k}":    round(float(np.mean(all_min_ade)),  4),
        f"minFDE_{cfg.eval.top_k}":    round(float(np.mean(all_min_fde)),  4),
        f"MissRate_2_{cfg.eval.top_k}":round(float(np.mean(all_miss)),     4),
        "OffRoadRate":                  round(float(np.mean(all_off_road)), 4),
        "num_samples":                  len(all_min_ade),
    }

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("=" * 50)

    # Save
    with open(cfg.eval.results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] Results saved to {cfg.eval.results_path}")

    with open(cfg.eval.predictions_path, "w") as f:
        json.dump(all_predictions, f)
    print(f"[eval] Predictions saved to {cfg.eval.predictions_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot",   type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    main(args)
