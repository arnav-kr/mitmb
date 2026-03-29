"""
Training script for PedestrianTrajectoryPredictor.

Usage:
    python train.py
    python train.py --dataroot /path/to/nuscenes --epochs 80
"""

import argparse
import json
import os
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from config import cfg
from dataset import NuScenesDataset, collate_fn
from model import PedestrianTrajectoryPredictor, count_parameters
from losses import TrajectoryLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def move_batch(batch, device):
    """Move all tensor fields to device."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, scaler, train: bool):
    model.train(train)
    total_loss = total_ade = total_fde = 0.0
    n_batches = 0

    for batch in loader:
        if batch is None:
            continue

        batch = move_batch(batch, device)

        with torch.set_grad_enabled(train):
            amp_enabled = (device.type == "cuda")
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                trajs, goals, log_probs = model(
                    batch["agent_state"],
                    batch["neighbor_states"],
                    batch["neighbor_mask"],
                    batch["map_raster"],
                )

                loss, metrics = criterion(
                    trajs, goals, log_probs,
                    batch["future"],
                    batch["visibility_weight"],
                )

        if train:
            optimizer.zero_grad()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.train.grad_clip
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.train.grad_clip
                )
                optimizer.step()

        total_loss += metrics["loss"]
        total_ade  += metrics["ade"]
        total_fde  += metrics["fde"]
        n_batches  += 1

    if n_batches == 0:
        return {"loss": 0.0, "ade": 0.0, "fde": 0.0}

    return {
        "loss": total_loss / n_batches,
        "ade":  total_ade  / n_batches,
        "fde":  total_fde  / n_batches,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # Apply CLI overrides
    if args.dataroot:
        cfg.data.dataroot = args.dataroot
    if args.epochs:
        cfg.train.num_epochs = args.epochs
    if args.batch_size:
        cfg.train.batch_size = args.batch_size

    set_seed(cfg.train.seed)

    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_ds = NuScenesDataset(split="train")
    val_ds   = NuScenesDataset(split="train_val")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=cfg.data.pin_memory,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = PedestrianTrajectoryPredictor().to(device)
    n_params = count_parameters(model)
    print(f"[train] Parameters: {n_params:,}")

    criterion = TrajectoryLoss()
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    # LR schedule: linear warmup → cosine decay
    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=cfg.train.warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cfg.train.num_epochs - cfg.train.warmup_epochs,
        eta_min=cfg.train.min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[cfg.train.warmup_epochs],
    )

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    start_epoch = 0
    history = {"train": [], "val": []}
    ckpt_path = os.path.join(cfg.train.checkpoint_dir, "latest.pt")

    if os.path.exists(ckpt_path) and not args.fresh:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", history)
        print(f"[train] Resumed from epoch {start_epoch}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_fde = float("inf")

    for epoch in range(start_epoch, cfg.train.num_epochs):
        t0 = time.time()

        train_m = run_epoch(model, train_loader, criterion, optimizer, device, scaler, train=True)
        val_m   = run_epoch(model, val_loader,   criterion, optimizer, device, scaler, train=False)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch+1:03d}/{cfg.train.num_epochs} | "
            f"lr={lr:.2e} | "
            f"train loss={train_m['loss']:.4f} ade={train_m['ade']:.3f} fde={train_m['fde']:.3f} | "
            f"val loss={val_m['loss']:.4f} ade={val_m['ade']:.3f} fde={val_m['fde']:.3f} | "
            f"{elapsed:.1f}s"
        )

        history["train"].append({**train_m, "epoch": epoch})
        history["val"].append({**val_m,   "epoch": epoch})

        # Save checkpoint
        ckpt = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history":   history,
        }
        torch.save(ckpt, ckpt_path)

        if (epoch + 1) % cfg.train.save_every == 0:
            ep_path = os.path.join(cfg.train.checkpoint_dir, f"epoch_{epoch+1:03d}.pt")
            torch.save(ckpt, ep_path)

        # Save best model by val FDE
        if val_m["fde"] < best_val_fde:
            best_val_fde = val_m["fde"]
            torch.save(ckpt, os.path.join(cfg.train.checkpoint_dir, "best.pt"))
            print(f"  ✓ New best val FDE: {best_val_fde:.4f}")

    # Save history
    with open("outputs/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[train] Done. Best val FDE: {best_val_fde:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot",   type=str,  default=None)
    parser.add_argument("--epochs",     type=int,  default=None)
    parser.add_argument("--batch_size", type=int,  default=None)
    parser.add_argument("--fresh",      action="store_true",
                        help="Ignore existing checkpoint and train from scratch")
    args = parser.parse_args()
    main(args)
