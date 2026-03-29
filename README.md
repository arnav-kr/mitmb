# Pedestrian Trajectory Prediction — MAHE Mobility Challenge (PS1)

Goal-conditioned, map-aware, multi-modal trajectory predictor for pedestrians and cyclists on nuScenes.

---

## Results (val split)

| Metric | Value |
|---|---|
| minADE\_3 | TBD after training |
| minFDE\_3 | TBD after training |
| MissRate\_2\_3 | TBD |
| OffRoadRate | TBD |

---

## Architecture

```
Past trajectory (4 steps × 5 features)
    └─► AgentEncoder (MLP)
                └─► agent token (256d)
                        │
Neighbours (≤10 × 4 steps × 4 features)
    └─► NeighbourEncoder (shared MLP)
                └─► N tokens (256d each)
                        │
                ┌───────▼────────┐
                │ SocialAttention │  cross-attention: ego queries neighbours
                └───────┬────────┘
                        │ social context (256d)
Map raster (3×100×100)  │
    └─► MapEncoder (CNN) │
                └──────►│
                ┌───────▼────────┐
                │  ContextFusion  │  concat + MLP
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

### Why this beats vanilla LSTM

| Aspect | Vanilla LSTM | This model |
|---|---|---|
| Input features | (x,y) only | (x,y, vx, vy, heading\_rate) |
| Social context | None | Cross-attention over neighbours |
| Map usage | None | CNN over 3-channel raster |
| Output | 1 path | 3 goal-conditioned paths |
| Coordinate frame | Global (broken) | Agent-frame normalised |
| Visibility | Ignored | Weighted in loss |
| Extra metrics | ADE/FDE | + MissRate + OffRoadRate |

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download nuScenes v1.0-trainval from https://www.nuscenes.org/nuscenes
#    and extract to ./data/nuscenes

# 3. Download nuScenes map expansion v1.2 into the same folder

# 4. Verify your folder structure:
#    data/nuscenes/
#    ├── v1.0-trainval/
#    ├── maps/
#    └── samples/
```

---

## Training

```bash
python train.py --dataroot ./data/nuscenes --epochs 80
```

Training automatically:
- Applies rotation augmentation (triples effective dataset size)
- Uses linear warmup (5 epochs) + cosine LR decay
- Saves best checkpoint by val FDE
- Logs history to `outputs/training_history.json`

To resume from a checkpoint (automatic):
```bash
python train.py   # picks up from checkpoints/latest.pt
```

To restart fresh:
```bash
python train.py --fresh
```

---

## Evaluation

```bash
python evaluate.py --dataroot ./data/nuscenes
```

Outputs `outputs/results.json` with all metrics and `outputs/predictions.json` in nuScenes submission format.

---

## Visualisation

```bash
python visualize.py --dataroot ./data/nuscenes --n_samples 16
```

Saves `outputs/viz/trajectory_predictions.png` — a grid of predicted paths overlaid on map rasters. Use this in your PPT.

---

## Key implementation details

### Agent-frame normalisation
Every sample is transformed so the agent's current position is the origin `(0,0)` and its heading points along the `+x` axis. This means the model learns _relative motion patterns_, not city-specific coordinates — essential for generalisation.

### Winner-takes-all training
During training, we only backpropagate through the mode with the **minimum FDE** to the ground truth. This prevents mode collapse (all K paths converging to the same prediction) and forces genuine diversity between modes.

### Map-constrained goal prediction
The GoalPredictor outputs K endpoint candidates. Using the map raster, trajectories that land on non-walkable pixels have higher reconstruction error, implicitly pushing goals toward walkable space.

### Visibility weighting
Annotations with low visibility (occluded agents) have noisy ground truth. We scale each sample's loss by `visibility_token / 4` so high-occlusion samples contribute less gradient.

### MissRate metric
A prediction is a _miss_ if **all K modes** have a maximum pointwise L2 distance > 2 m from ground truth. This is stricter than ADE/FDE and detects cases where the model is confidently wrong. We report this alongside ADE/FDE.

### OffRoadRate
We check the most-probable predicted trajectory against the drivable\_surface channel of the map raster. If any waypoint falls on a non-drivable pixel, that sample counts as off-road. This is a hidden metric in the nuScenes devkit that most teams won't report.

---

## Repository structure

```
.
├── config.py          All hyperparameters in one place
├── dataset.py         nuScenes loading, normalisation, map rasterisation
├── model.py           Full model architecture
├── losses.py          WTA loss, ADE, FDE, goal loss
├── train.py           Training loop with LR schedule + checkpointing
├── evaluate.py        Evaluation: ADE, FDE, MissRate, OffRoadRate
├── visualize.py       Trajectory visualisation overlaid on map
├── requirements.txt
├── checkpoints/       Saved model weights
└── outputs/           Predictions JSON, results JSON, visualisations
```

---

## References

- nuScenes prediction challenge: https://www.nuscenes.org/prediction
- Trajectron++: Ivanovic & Pavone, ECCV 2020
- AgentFormer: Yuan et al., ICCV 2021
- DGCN\_ST\_LANE: nuScenes leaderboard leader (MinADE\_10 = 1.092)
