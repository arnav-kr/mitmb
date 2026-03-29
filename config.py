from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    # nuScenes dataset root
    dataroot: str = "./data/nuscenes"
    version: str = "v1.0-mini"

    # Prediction horizon
    past_seconds: float = 2.0       # 2s history  → 4 timesteps at 2Hz
    future_seconds: float = 3.0     # 3s future   → 6 timesteps at 2Hz
    freq_hz: float = 2.0
    past_steps: int = 4             # past_seconds * freq_hz
    future_steps: int = 6           # future_seconds * freq_hz
    dt: float = 0.5                 # seconds per step

    # Agent filtering
    agent_categories: List[str] = field(default_factory=lambda: [
        "human.pedestrian.adult",
        "human.pedestrian.child",
        "human.pedestrian.construction_worker",
        "human.pedestrian.personal_mobility",
        "human.pedestrian.police_officer",
        "human.pedestrian.stroller",
        "human.pedestrian.wheelchair",
        "vehicle.bicycle",
    ])
    min_visibility: int = 1         # 1-4 scale; filter visibility=0 (occluded)

    # Neighbour context
    neighbor_radius: float = 30.0   # metres — only agents within this range
    max_neighbors: int = 10

    # Map rasterisation
    map_layers: List[str] = field(default_factory=lambda: [
        "walkway",
        "ped_crossing",
        "drivable_surface",
    ])
    map_patch_size: float = 50.0    # side length in metres, centred on agent
    map_canvas_size: int = 100      # pixels (100×100 → 0.4 m/px resolution)

    # Dataloader/runtime
    num_workers: int = 2
    pin_memory: bool = True


@dataclass
class ModelConfig:
    # Feature dimensions
    agent_feat_dim: int = 32        # encoded single-agent state
    neighbor_feat_dim: int = 32     # per-neighbour encoding
    map_feat_dim: int = 64          # CNN output
    hidden_dim: int = 256           # shared context dimension

    # Social attention
    num_heads: int = 4
    dropout: float = 0.1

    # Multi-modal output
    num_modes: int = 3              # K trajectories predicted simultaneously
    max_goal_radius: float = 15.0   # metres, goal endpoint clamp range

    # Input feature size per timestep: (x, y, vx, vy, heading_rate) = 5
    agent_input_size: int = 5
    # Neighbour input: (x, y, vx, vy) relative = 4 per timestep
    neighbor_input_size: int = 4


@dataclass
class TrainConfig:
    batch_size: int = 64
    num_epochs: int = 80
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # LR schedule
    warmup_epochs: int = 5
    min_lr: float = 1e-5

    # Loss weights
    w_ade: float = 1.0
    w_fde: float = 2.0              # FDE weighted more (judges look at it)
    w_goal: float = 0.5             # auxiliary goal-endpoint loss
    w_mode: float = 0.1             # classification loss

    # Augmentation
    use_rotation_aug: bool = True

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 10            # save checkpoint every N epochs

    # Reproducibility
    seed: int = 42


@dataclass
class EvalConfig:
    miss_threshold: float = 2.0     # metres — MissRate_2_K
    top_k: int = 3                  # evaluate min over K modes
    predictions_path: str = "./outputs/predictions.json"
    results_path: str = "./outputs/results.json"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


# Singleton — import this everywhere
cfg = Config()
