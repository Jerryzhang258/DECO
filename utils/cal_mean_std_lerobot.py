import os
import sys
import argparse
import numpy as np
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lerobot_compat import LeRobotDataset, LeRobotDatasetMetadata
from lerobot_dataset import episode_split

"""
Compute observation/action mean, std, min, max directly from a LeRobot dataset's TRAIN
split (never the val split -- stats leaking from val into normalization would let val
loss look better than it should). Output is a `data:` yaml block you paste into
config/deco_vitac2026_*.yaml.

Loads the dataset UNFILTERED and restricts to train rows by position (same approach as
lerobot_dataset.py's ManiskillVitacDataset), rather than passing `episodes=` to
LeRobotDataset. Passing a large `episodes=` list is dramatically slower in lerobot 0.4.4 --
confirmed on KaiyueChen/black_smash_03 (500 episodes, ~50GB): filtering to the ~450 train
episodes this way didn't finish in 20+ minutes, whereas the unfiltered load + column access
below runs in well under a minute.
"""


def cal_mean_std(repo_id, root=None, val_ratio=0.1, split_seed=42, obs_dim=20, action_dim=20, batch_size=4096):
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    train_episodes, _ = episode_split(meta.total_episodes, val_ratio, split_seed)
    print(f"train episodes: {len(train_episodes)} / {meta.total_episodes}")

    dataset = LeRobotDataset(repo_id, root=root)
    episode_col = dataset.hf_dataset.data.column("episode_index").to_numpy()
    train_positions = np.where(np.isin(episode_col, np.array(train_episodes)))[0]
    n = len(train_positions)
    print(f"train frames: {n} / {len(episode_col)}")

    obs_col = dataset.hf_dataset.data.column("observation.state")
    act_col = dataset.hf_dataset.data.column("actions")

    obs_sum = torch.zeros(obs_dim, dtype=torch.float64)
    obs_sq_sum = torch.zeros(obs_dim, dtype=torch.float64)
    obs_min = torch.full((obs_dim,), float("inf"))
    obs_max = torch.full((obs_dim,), float("-inf"))

    act_sum = torch.zeros(action_dim, dtype=torch.float64)
    act_sq_sum = torch.zeros(action_dim, dtype=torch.float64)
    act_min = torch.full((action_dim,), float("inf"))
    act_max = torch.full((action_dim,), float("-inf"))

    for start in tqdm(range(0, n, batch_size)):
        idx = train_positions[start : start + batch_size]
        obs = torch.from_numpy(np.stack(obs_col.take(idx).to_numpy(zero_copy_only=False))).double()
        act = torch.from_numpy(np.stack(act_col.take(idx).to_numpy(zero_copy_only=False))).double()

        obs_sum += obs.sum(dim=0)
        obs_sq_sum += (obs**2).sum(dim=0)
        obs_min = torch.minimum(obs_min, obs.min(dim=0).values.float())
        obs_max = torch.maximum(obs_max, obs.max(dim=0).values.float())

        act_sum += act.sum(dim=0)
        act_sq_sum += (act**2).sum(dim=0)
        act_min = torch.minimum(act_min, act.min(dim=0).values.float())
        act_max = torch.maximum(act_max, act.max(dim=0).values.float())

    obs_mean = obs_sum / n
    obs_std = torch.sqrt((obs_sq_sum / n - obs_mean**2).clamp_min(0))
    act_mean = act_sum / n
    act_std = torch.sqrt((act_sq_sum / n - act_mean**2).clamp_min(0))

    return {
        "observation_mean": obs_mean.float().tolist(),
        "observation_std": obs_std.float().tolist(),
        "observation_min": obs_min.tolist(),
        "observation_max": obs_max.tolist(),
        "action_mean": act_mean.float().tolist(),
        "action_std": act_std.float().tolist(),
        "action_min": act_min.tolist(),
        "action_max": act_max.tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--root", type=str, default=None, help="Local dataset root (v3.0 layout). Omit to fetch from the HF Hub.")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--obs-dim", type=int, default=20)
    parser.add_argument("--action-dim", type=int, default=20)
    parser.add_argument("--save-path", type=str, default="./data_statistics_lerobot.yaml")
    args = parser.parse_args()

    stats = cal_mean_std(
        args.repo_id,
        root=args.root,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
    )
    with open(args.save_path, "w") as f:
        yaml.dump({"data": stats}, f, default_flow_style=None)
    print("Saved stats to", args.save_path)
