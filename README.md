# Pedestrian Trajectory Prediction
![Prediction](<prediction.png>)
Goal-conditioned, map-aware, multi-modal trajectory predictor for pedestrians and cyclists evaluated on the nuScenes dataset. 

## Performance (nuScenes v1.0-mini val split)

| Metric | Value |
|---|---|
| minADE_3 | TBD |
| minFDE_3 | TBD |
| MissRate_2_3 | TBD |
| OffRoadRate | TBD |

## Architecture

The model predicts $K=3$ future trajectories based on 2 seconds of history (4 timesteps at 2Hz) to predict 3 seconds of future motion (6 timesteps).

```text
Past trajectory (4 steps × 5 features)
    └─► AgentEncoder (MLP)
                └─► agent token (256d)
                        │
Neighbours (≤10 × 4 steps × 4 features)
    └─► NeighbourEncoder (shared MLP)
                └─► N tokens (256d each)
                        │
                ┌───────▼────────┐
                │ SocialAttention│  cross-attention: ego queries neighbours
                └───────┬────────┘
                        │ social context (256d)
Map raster (3×100×100)  │
    └─► MapEncoder (CNN)│
                └──────►│
                ┌───────▼────────┐
                │  ContextFusion │  concat + MLP
                └───────┬────────┘
                        │ context (256d)
                ┌───────▼────────────────────┐
                │  GoalPredictor             │  → K goal endpoints (K=3)
                └───────┬────────────────────┘
                        │ goals (K×2)
                ┌───────▼────────────────────┐
                │  TrajectoryDecoder (GRU)   │  goal-conditioned, per-mode
                └───────┬────────────────────┘
                        │ K trajectories (K×6×2)
                ┌───────▼────────────────────┐
                │  ModeClassifier            │  → K log-probabilities
                └────────────────────────────┘
```

### Technical Characteristics

* **Coordinate Frame:** Agent-frame normalized. Origin is $(0,0)$ and heading aligns with the $+x$ axis. Ensures learning of relative motion patterns.
* **Social Context:** Cross-attention mechanism over a maximum of 10 neighbors within a 30m radius.
* **Map Integration:** 3-channel binary map rasterization (walkway, pedestrian crossing, drivable surface) processed via CNN.
* **Loss Formulation:** Winner-takes-all (WTA) backpropagation. Loss is computed exclusively through the mode with the minimum FDE to the ground truth to prevent mode collapse. Gradients are scaled by ground truth visibility levels to down-weight occluded, noisy samples.
* **Decoding:** Decodes step deltas autoregressively via GRU. An endpoint linear correction is applied to enforce consistency with the predicted goal mode.
* **Metrics Check:** MissRate evaluates strict confidence failures (distance > 2m). OffRoadRate checks trajectory intersections with non-drivable map pixels.

## Requirements and Setup

Install prerequisites. Requires `torch`, `numpy`, and `nuscenes-devkit`.

```bash
pip install -r requirements.txt
```

Extract the nuScenes v1.0-mini dataset. The expected directory structure is:

```text
data/nuscenes/
├── v1.0-mini/
├── maps/
└── samples/
```

*Note: The codebase defaults to `v1.0-mini`. Modify `config.py` to run on the full `trainval` split.*

## Usage

All hyperparameters, including learning rate schedules, network dimensions, and data filtering constraints, are defined in `config.py`. 

### Training

Initiates training with a linear warmup and cosine decay. Applies 90-degree map and coordinate rotation augmentation dynamically. Best checkpoints are saved based on validation FDE.

```bash
python train.py --dataroot ./data/nuscenes --epochs 80 --batch_size 64
```

To ignore previous checkpoints and start fresh:
```bash
python train.py --fresh
```

Training history is logged to `outputs/training_history.json`.

### Evaluation

Evaluates the validation split and outputs results. Generates `outputs/results.json` containing aggregate metrics and `outputs/predictions.json` adhering to the nuScenes submission format.

```bash
python evaluate.py --dataroot ./data/nuscenes --checkpoint ./checkpoints/best.pt
```

### Visualization

Generates map-overlaid trajectory plots for visual debugging and presentation. Saves a grid format image to `outputs/viz/trajectory_predictions.png`.

```bash
python visualize.py --dataroot ./data/nuscenes --n_samples 16
```

## Repository Structure

```text
.
├── config.py          Centralized hyperparameters and settings
├── dataset.py         nuScenes loading, agent-frame transforms, map rasterization
├── evaluate.py        Validation evaluation (ADE, FDE, MissRate, OffRoadRate)
├── losses.py          WTA, ADE, FDE, goal, and mode classification losses
├── model.py           Full multi-modal network architecture
├── train.py           Main training loop, AMP, and checkpointing
├── visualize.py       Matplotlib overlay generation
└── requirements.txt
```

## References

* nuScenes Prediction Challenge: [https://www.nuscenes.org/prediction](https://www.nuscenes.org/prediction)
* Trajectron++: Ivanovic & Pavone, ECCV 2020
* AgentFormer: Yuan et al., ICCV 2021