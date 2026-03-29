# Pedestrian Trajectory Prediction
![prediction](<prediction.png>)

## Project overview
Goal-conditioned, map-aware, multi-modal trajectory predictor for pedestrians and cyclists, developed for the MAHE Mobility Challenge (PS1). Standard deterministic models predict a single average path, which fails in highly stochastic urban environments. This architecture resolves this by forecasting K=3 distinct, probable future trajectories. It fuses agent coordinate history, social interactions via cross-attention, and physical map boundaries to ensure predictions are both diverse and strictly compliant with drivable/walkable surfaces.

## Example outputs / results
Performance evaluated on the nuScenes v1.0-mini validation split.

| Metric | Value | Description |
|---|---|---|
| minADE_3 | 0.308 | Minimum Average Displacement Error across 3 modes. |
| minFDE_3 | 0.518 | Minimum Final Displacement Error across 3 modes. |
| MissRate_2_3 | 0.046 | Fraction of samples where all 3 modes have max L2 > 2m. |
| OffRoadRate | 1.00 | Fraction of primary predictions intersecting non-drivable pixels. |

Visual outputs are generated via the included visualization pipeline, mapping past history (blue), ground truth (green), and predicted modes (dashed) over semantic map rasters.

## Model architecture
The system predicts 3 seconds of future motion (6 timesteps) based on 2 seconds of history (4 timesteps at 2Hz).

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

### Technical Characteristics
* **Coordinate Frame:** Agent-frame normalized. Origin is (0,0) and heading aligns with the +x axis, enforcing the learning of relative motion patterns.
* **Social Context:** Cross-attention mechanism computing interactions over a maximum of 10 neighbors within a 30m radius.
* **Map Integration:** 3-channel binary map rasterization (walkway, pedestrian crossing, drivable surface) processed via a lightweight custom CNN.
* **Loss Formulation:** Winner-takes-all (WTA) backpropagation. Loss routes exclusively through the mode with the minimum FDE to the ground truth, preventing mode collapse. Gradients are scaled by visibility tokens to down-weight noisy, occluded samples.
* **Decoding:** Autoregressive GRU decoding of step deltas with a linear endpoint correction to enforce goal-mode consistency.

## Dataset used
Trained and evaluated on the **nuScenes** dataset. 
* Targets vulnerable road users: pedestrians (adult, child, construction worker, mobility device, police, stroller, wheelchair) and bicycles.
* Data is preprocessed into agent-centric local frames. Map data is rasterized locally.
* Dynamic 90-degree scene and map rotation augmentations are applied during training to ensure rotational invariance and expand the effective dataset.

## Setup & installation instructions
Requires `torch`, `numpy`, and `nuscenes-devkit`.

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
*Note: Default configuration targets `v1.0-mini`. Adjust `config.py` for full `trainval` splits.*

## Usage
### How to run the code
All network dimensions, learning rate schedules, and data filtering constraints are centralized in `config.py`.

**Training**
Initiates training with a 5-epoch linear warmup and cosine decay. Automatically applies rotation augmentation and saves best checkpoints based on validation FDE.

```bash
python train.py --dataroot ./data/nuscenes --epochs 80 --batch_size 64
```
To force a fresh training cycle, ignoring existing checkpoints:
```bash
python train.py --fresh
```

**Evaluation**
Evaluates the validation split. Generates `outputs/results.json` with aggregate metrics and `outputs/predictions.json` formatted to the nuScenes submission standard.

```bash
python evaluate.py --dataroot ./data/nuscenes --checkpoint ./checkpoints/best.pt
```

**Visualization**
Generates map-overlaid trajectory plots for debugging and presentation. Saves to `outputs/viz/trajectory_predictions.png`.

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
* nuScenes Prediction Challenge: https://www.nuscenes.org/prediction
* Trajectron++: Ivanovic & Pavone, ECCV 2020
* AgentFormer: Yuan et al., ICCV 2021