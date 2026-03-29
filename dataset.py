"""
NuScenesDataset — loads the nuScenes prediction challenge splits and returns
per-agent tensors ready for the model.

Key design decisions:
  • Agent-frame normalisation: origin=(0,0), heading=+x  (mandatory for generalisation)
  • Visibility filtering: skip agents with token '0' (fully occluded)
  • Neighbour extraction: up to cfg.data.max_neighbors within radius, agent-frame
  • Map rasterisation: 3-channel binary mask (walkway / ped_crossing / drivable)
  • Per-sample visibility weight for loss down-weighting of noisy labels
"""

import math
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Tuple, List

from config import cfg

# nuScenes imports — install with: pip install nuscenes-devkit
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.prediction import PredictHelper
    from nuscenes.map_expansion.map_api import NuScenesMap
    from nuscenes.eval.prediction.splits import get_prediction_challenge_split
    from nuscenes.utils.splits import create_splits_scenes
    from pyquaternion import Quaternion
    NUSCENES_AVAILABLE = True
except ImportError:
    NUSCENES_AVAILABLE = False
    print("[dataset] nuScenes devkit not found — running in MOCK mode for testing.")


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def rotation_matrix_2d(yaw: float) -> np.ndarray:
    """2×2 CCW rotation matrix for angle `yaw` (radians)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def to_agent_frame(
    points: np.ndarray,   # (N, 2) global x,y
    origin: np.ndarray,   # (2,)   agent global position
    yaw: float,           # agent heading in radians
) -> np.ndarray:
    """Translate then rotate points into the agent's local frame."""
    R = rotation_matrix_2d(-yaw)          # rotate world by -yaw to align agent → +x
    return (points - origin) @ R.T        # (N, 2)


def quaternion_to_yaw(q_list: List[float]) -> float:
    """Extract yaw from a [w, x, y, z] quaternion list."""
    q = Quaternion(q_list)
    return q.yaw_pitch_roll[0]


def ensure_xy_array(arr) -> np.ndarray:
    """
    Convert various devkit return formats to float32 array shaped (N, 2).
    Handles empty lists, nested lists, dict rows with x/y, and malformed outputs.
    """
    if arr is None:
        return np.zeros((0, 2), dtype=np.float32)

    if isinstance(arr, list) and len(arr) > 0 and isinstance(arr[0], dict):
        out = []
        for row in arr:
            x = row.get("x", 0.0)
            y = row.get("y", 0.0)
            out.append([x, y])
        return np.asarray(out, dtype=np.float32).reshape(-1, 2)

    out = np.asarray(arr, dtype=np.float32)

    if out.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    if out.ndim == 1:
        if out.shape[0] >= 2:
            return out[-2:].reshape(1, 2)
        # Not interpretable as x,y pairs
        return np.zeros((0, 2), dtype=np.float32)

    if out.shape[-1] >= 2:
        return out[..., -2:].reshape(-1, 2)

    return np.zeros((0, 2), dtype=np.float32)


# ---------------------------------------------------------------------------
# Main dataset
# ---------------------------------------------------------------------------

class NuScenesDataset(Dataset):
    _split_fallback_logged = False

    def __init__(self, split: str = "train"):
        """
        Args:
            split: one of 'train', 'train_val', 'val'
        """
        assert split in ("train", "train_val", "val"), f"Unknown split: {split}"
        self.split = split
        self.is_train = split in ("train", "train_val")
        self.cfg_d = cfg.data
        self.cfg_m = cfg.model
        self.cfg_t = cfg.train

        if not NUSCENES_AVAILABLE:
            self._mock_mode = True
            self._mock_len = 200
            return
        self._mock_mode = False

        # Load nuScenes
        self.nusc = NuScenes(
            version=self.cfg_d.version,
            dataroot=self.cfg_d.dataroot,
            verbose=False,
        )
        self.helper = PredictHelper(self.nusc)

        # Load split tokens: each token is "instance_token_sample_token"
        split_name = self._resolve_split_name(split)
        all_tokens = self._load_split_tokens(split_name)

        # Filter to our agent categories and minimum visibility
        self.tokens = self._filter_tokens(all_tokens)
        print(f"[dataset] {split}: {len(self.tokens)} samples after filtering")

        # Cache maps by location name (Singapore-OneNorth, boston-seaport, etc.)
        self._map_cache: Dict[str, NuScenesMap] = {}

    def _resolve_split_name(self, split: str) -> str:
        """
        nuScenes-mini does not expose the same train/train_val/val setup as trainval.
        Map canonical names to mini-compatible names when needed.
        """
        if "mini" in self.cfg_d.version:
            mapping = {
                "train": "mini_train",
                "train_val": "mini_val",
                "val": "mini_val",
            }
            return mapping[split]
        return split

    def _load_split_tokens(self, split_name: str) -> List[str]:
        """Try the official split API first, then build from scene split if metadata is absent."""
        try:
            return get_prediction_challenge_split(split_name, dataroot=self.cfg_d.dataroot)
        except Exception as e:
            if isinstance(e, FileNotFoundError):
                if not NuScenesDataset._split_fallback_logged:
                    print(
                        "[dataset] prediction_scenes.json not found in maps/prediction; "
                        "building tokens from scene splits (mini-compatible fallback)."
                    )
                    NuScenesDataset._split_fallback_logged = True
                return self._build_tokens_from_scene_split(split_name)
            raise RuntimeError(f"Unable to load prediction split '{split_name}': {e}")

    def _build_tokens_from_scene_split(self, split_name: str) -> List[str]:
        """
        Fallback for environments where maps/prediction/prediction_scenes.json is absent.
        Builds prediction tokens from all annotations in scenes that belong to split_name.
        """
        scene_splits = create_splits_scenes()
        scene_names = set(scene_splits.get(split_name, []))
        if not scene_names:
            raise RuntimeError(f"Unknown scene split '{split_name}'")

        tokens: List[str] = []
        seen = set()

        for scene in self.nusc.scene:
            if scene["name"] not in scene_names:
                continue

            sample_token = scene["first_sample_token"]
            while sample_token:
                sample = self.nusc.get("sample", sample_token)
                for ann_tok in sample["anns"]:
                    ann = self.nusc.get("sample_annotation", ann_tok)
                    key = f"{ann['instance_token']}_{sample_token}"
                    if key in seen:
                        continue
                    seen.add(key)
                    tokens.append(key)
                sample_token = sample["next"]

        return tokens

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _filter_tokens(self, tokens: List[str]) -> List[str]:
        kept = []
        for token in tokens:
            inst_tok, samp_tok = token.split("_")
            ann = self.helper.get_sample_annotation(inst_tok, samp_tok)
            cat = ann["category_name"]
            vis = int(ann.get("visibility_token", "0"))
            if any(cat.startswith(c) for c in self.cfg_d.agent_categories):
                if vis >= self.cfg_d.min_visibility:
                    kept.append(token)
        return kept

    # ------------------------------------------------------------------
    # Map access (cached)
    # ------------------------------------------------------------------

    def _get_map(self, location: str) -> "NuScenesMap":
        if location not in self._map_cache:
            self._map_cache[location] = NuScenesMap(
                dataroot=self.cfg_d.dataroot,
                map_name=location,
            )
        return self._map_cache[location]

    def _get_location(self, sample_token: str) -> str:
        sample = self.nusc.get("sample", sample_token)
        scene = self.nusc.get("scene", sample["scene_token"])
        log = self.nusc.get("log", scene["log_token"])
        return log["location"]

    # ------------------------------------------------------------------
    # Map rasterisation
    # ------------------------------------------------------------------

    def _rasterize_map(
        self,
        location: str,
        origin: np.ndarray,  # (2,) global x,y of agent
        yaw: float,          # agent heading
    ) -> np.ndarray:
        """Returns (C, H, W) binary float32 map centred on agent, agent-frame aligned."""
        canvas_size = (self.cfg_d.map_canvas_size, self.cfg_d.map_canvas_size)
        try:
            nmap = self._get_map(location)
        except Exception:
            # Keep pipeline alive when map expansion files are unavailable.
            return np.zeros((len(self.cfg_d.map_layers), *canvas_size), dtype=np.float32)

        half = self.cfg_d.map_patch_size / 2.0
        patch_box = (origin[0], origin[1], self.cfg_d.map_patch_size, self.cfg_d.map_patch_size)
        patch_angle = math.degrees(yaw)   # NuScenesMap expects degrees

        masks = []
        for layer in self.cfg_d.map_layers:
            try:
                mask = nmap.get_map_mask(patch_box, patch_angle, [layer], canvas_size)
                masks.append(mask[0].astype(np.float32))   # (H, W)
            except Exception:
                masks.append(np.zeros(canvas_size, dtype=np.float32))

        return np.stack(masks, axis=0)   # (C, H, W)

    # ------------------------------------------------------------------
    # Per-agent state extraction
    # ------------------------------------------------------------------

    def _get_agent_state(
        self,
        inst_tok: str,
        samp_tok: str,
        origin: np.ndarray,
        yaw: float,
    ) -> np.ndarray:
        """
        Returns (past_steps, agent_input_size) float32 array.
        Features per step: [x, y, vx, vy, heading_rate]
        All in agent frame.
        """
        T = self.cfg_d.past_steps
        feat_dim = self.cfg_m.agent_input_size   # 5

        try:
            past_xy_global = self.helper.get_past_for_agent(
                inst_tok, samp_tok,
                seconds=self.cfg_d.past_seconds,
                in_agent_frame=True,
                just_xy=True,
            )  # (T, 2), most-recent-first
            past_xy_global = ensure_xy_array(past_xy_global)[::-1].copy()   # oldest-first
        except Exception:
            past_xy_global = np.zeros((0, 2), dtype=np.float32)

        # Pad / truncate to exactly T steps
        if len(past_xy_global) < T:
            pad = np.tile(past_xy_global[:1] if len(past_xy_global) > 0 else np.zeros((1, 2)),
                          (T - len(past_xy_global), 1))
            past_xy_global = np.concatenate([pad, past_xy_global], axis=0)
        past_xy_global = past_xy_global[-T:]   # (T, 2)

        # Include current position as the T+1-th point
        all_xy_local = np.vstack([past_xy_global, np.zeros((1, 2), dtype=np.float32)])
        past_xy_local = all_xy_local[:T]                             # (T, 2)

        # Velocities: finite differences in agent frame (m/s)
        dxy = np.diff(all_xy_local, axis=0)   # (T, 2)  — includes step to current pos
        vel = dxy / self.cfg_d.dt              # (T, 2) as vx, vy

        # Heading rate from global annotation kinematics (scalar per step)
        try:
            hr = self.helper.get_heading_change_rate_for_agent(inst_tok, samp_tok)
        except Exception:
            hr = 0.0
        if not np.isfinite(hr):
            hr = 0.0
        heading_rate = np.full((T, 1), hr, dtype=np.float32)

        state = np.concatenate([past_xy_local, vel, heading_rate], axis=1)  # (T, 5)
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        return state.astype(np.float32)

    # ------------------------------------------------------------------
    # Neighbour extraction
    # ------------------------------------------------------------------

    def _get_neighbors(
        self,
        inst_tok: str,
        samp_tok: str,
        origin: np.ndarray,
        yaw: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            neighbor_states: (max_neighbors, past_steps, 4) float32
            neighbor_mask:   (max_neighbors,) bool  — True = valid slot
        """
        T = self.cfg_d.past_steps
        N = self.cfg_d.max_neighbors
        radius = self.cfg_d.neighbor_radius

        neighbor_states = np.zeros((N, T, 4), dtype=np.float32)
        neighbor_mask = np.zeros(N, dtype=bool)

        sample = self.nusc.get("sample", samp_tok)
        ann_tokens = sample["anns"]

        slot = 0
        for ann_tok in ann_tokens:
            if slot >= N:
                break
            ann = self.nusc.get("sample_annotation", ann_tok)
            if ann["instance_token"] == inst_tok:
                continue   # skip the ego agent
            cat = ann["category_name"]
            if not any(cat.startswith(c) for c in self.cfg_d.agent_categories):
                continue

            neighbor_global = np.array(ann["translation"][:2], dtype=np.float32)
            dist = np.linalg.norm(neighbor_global - origin)
            if dist > radius:
                continue

            # Get past trajectory for this neighbour
            try:
                nb_past_global = self.helper.get_past_for_agent(
                    ann["instance_token"], samp_tok,
                    seconds=self.cfg_d.past_seconds,
                    in_agent_frame=False,
                    just_xy=True,
                )
                nb_past_global = ensure_xy_array(nb_past_global)[::-1].copy()
            except Exception:
                nb_past_global = np.zeros((0, 2), dtype=np.float32)

            # Pad / truncate
            if len(nb_past_global) < T:
                pad = np.tile(
                    nb_past_global[:1] if len(nb_past_global) > 0 else neighbor_global.reshape(1, 2),
                    (T - len(nb_past_global), 1)
                )
                nb_past_global = np.concatenate([pad, nb_past_global], axis=0)
            nb_past_global = nb_past_global[-T:]   # (T, 2)

            # Transform to ego agent frame
            nb_past_local = to_agent_frame(nb_past_global, origin, yaw)   # (T, 2)

            # Velocities
            all_nb = np.vstack([nb_past_global, neighbor_global.reshape(1, 2)])
            all_nb_local = to_agent_frame(all_nb, origin, yaw)
            nb_vel = np.diff(all_nb_local, axis=0) / self.cfg_d.dt   # (T, 2)

            neighbor_states[slot] = np.concatenate([nb_past_local, nb_vel], axis=1)  # (T, 4)
            neighbor_mask[slot] = True
            slot += 1

        neighbor_states = np.nan_to_num(neighbor_states, nan=0.0, posinf=0.0, neginf=0.0)
        return neighbor_states, neighbor_mask

    # ------------------------------------------------------------------
    # Ground truth future
    # ------------------------------------------------------------------

    def _get_future(
        self,
        inst_tok: str,
        samp_tok: str,
        origin: np.ndarray,
        yaw: float,
    ) -> np.ndarray:
        """Returns (future_steps, 2) in agent frame."""
        T = self.cfg_d.future_steps
        try:
            fut_global = self.helper.get_future_for_agent(
                inst_tok, samp_tok,
                seconds=self.cfg_d.future_seconds,
                in_agent_frame=True,
                just_xy=True,
            )  # (T, 2), oldest-first
            fut_global = ensure_xy_array(fut_global)
        except Exception:
            fut_global = np.zeros((T, 2), dtype=np.float32)

        if len(fut_global) < T:
            pad_val = fut_global[-1:] if len(fut_global) > 0 else np.zeros((1, 2))
            fut_global = np.vstack([fut_global, np.tile(pad_val, (T - len(fut_global), 1))])
        fut_global = fut_global[:T]   # (T, 2)
        fut = fut_global.astype(np.float32)
        fut = np.nan_to_num(fut, nan=0.0, posinf=0.0, neginf=0.0)
        return fut

    # ------------------------------------------------------------------
    # Rotation augmentation
    # ------------------------------------------------------------------

    def _augment(
        self,
        agent_state: np.ndarray,    # (T, 5)
        neighbor_states: np.ndarray, # (N, T, 4)
        future: np.ndarray,          # (T_fut, 2)
        map_raster: np.ndarray,      # (C, H, W)
    ):
        """
        Random scene rotation — all coordinates and map raster rotate together.
        Uses 90-degree rotations so map_raster can be rotated exactly via np.rot90.
        """
        k = int(np.random.randint(0, 4))
        angle = k * (math.pi / 2.0)
        R = rotation_matrix_2d(angle).T   # (2, 2) for row-vector multiplication

        def rot_xy(arr):
            return arr @ R   # works for (*, 2) arrays

        # Agent: rotate (x,y) and (vx,vy) columns
        agent_out = agent_state.copy()
        agent_out[:, :2] = rot_xy(agent_state[:, :2])
        agent_out[:, 2:4] = rot_xy(agent_state[:, 2:4])
        # heading_rate (col 4) is a scalar — rotation doesn't change it

        # Neighbours
        nb_out = neighbor_states.copy()
        nb_out[:, :, :2] = (neighbor_states[:, :, :2].reshape(-1, 2) @ R).reshape(
            neighbor_states.shape[0], neighbor_states.shape[1], 2
        )
        nb_out[:, :, 2:4] = (neighbor_states[:, :, 2:4].reshape(-1, 2) @ R).reshape(
            neighbor_states.shape[0], neighbor_states.shape[1], 2
        )

        fut_out = rot_xy(future)

        map_out = np.rot90(map_raster, k=k, axes=(1, 2)).copy()

        return agent_out, nb_out, fut_out, map_out

    # ------------------------------------------------------------------
    # Mock mode (no nuScenes installed)
    # ------------------------------------------------------------------

    def _mock_item(self) -> Dict[str, torch.Tensor]:
        T_p = self.cfg_d.past_steps
        T_f = self.cfg_d.future_steps
        N = self.cfg_d.max_neighbors
        C = len(self.cfg_d.map_layers)
        H = W = self.cfg_d.map_canvas_size

        return {
            "agent_state":     torch.randn(T_p, self.cfg_m.agent_input_size),
            "future":          torch.randn(T_f, 2),
            "neighbor_states": torch.randn(N, T_p, self.cfg_m.neighbor_input_size),
            "neighbor_mask":   torch.ones(N, dtype=torch.bool),
            "map_raster":      torch.rand(C, H, W),
            "visibility_weight": torch.tensor(1.0),
            "instance_token":  "mock_inst",
            "sample_token":    "mock_samp",
        }

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self._mock_mode:
            return self._mock_len
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        if self._mock_mode:
            return self._mock_item()

        token = self.tokens[idx]
        inst_tok, samp_tok = token.split("_")

        try:
            ann = self.helper.get_sample_annotation(inst_tok, samp_tok)
            origin = np.array(ann["translation"][:2], dtype=np.float32)
            yaw = quaternion_to_yaw(ann["rotation"])
            visibility = int(ann.get("visibility_token", "1"))
            vis_weight = visibility / 4.0   # normalise to [0.25, 1.0]

            location = self._get_location(samp_tok)

            agent_state = self._get_agent_state(inst_tok, samp_tok, origin, yaw)
            future = self._get_future(inst_tok, samp_tok, origin, yaw)
            neighbor_states, neighbor_mask = self._get_neighbors(inst_tok, samp_tok, origin, yaw)
            map_raster = self._rasterize_map(location, origin, yaw)

            # Augmentation (training only)
            if self.is_train and self.cfg_t.use_rotation_aug:
                agent_state, neighbor_states, future, map_raster = self._augment(
                    agent_state, neighbor_states, future, map_raster
                )

            return {
                "agent_state":       torch.from_numpy(agent_state),        # (T, 5)
                "future":            torch.from_numpy(future),             # (T_fut, 2)
                "neighbor_states":   torch.from_numpy(neighbor_states),   # (N, T, 4)
                "neighbor_mask":     torch.from_numpy(neighbor_mask),     # (N,) bool
                "map_raster":        torch.from_numpy(map_raster),        # (C, H, W)
                "visibility_weight": torch.tensor(vis_weight, dtype=torch.float32),
                "instance_token":    inst_tok,
                "sample_token":      samp_tok,
            }

        except Exception as e:
            # Return None; collate_fn will skip this sample
            print(f"[dataset] WARNING: skipping {token}: {e}")
            return None


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """Filter None samples and stack tensors."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    keys = [k for k in batch[0].keys() if k not in ("instance_token", "sample_token")]
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    out["instance_tokens"] = [b["instance_token"] for b in batch]
    out["sample_tokens"]   = [b["sample_token"]   for b in batch]
    return out
