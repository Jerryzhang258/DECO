import os
import sys
import argparse
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

Uses `LeRobotDataset.select_columns(...)`, which returns raw per-frame values without
decoding any of the 6 image streams, so this stays fast even on the full ~500-episode /
~500k-frame dataset.
"""


def cal_mean_std(repo_id, root=None, val_ratio=0.1, split_seed=42, obs_dim=20, action_dim=20, batch_size=4096):
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    train_episodes, _ = episode_split(meta.total_episodes, val_ratio, split_seed)
    print(f"train episodes: {len(train_episodes)} / {meta.total_episodes}")

    dataset = LeRobotDataset(repo_id, root=root, episodes=train_episodes)
    cols = dataset.select_columns(["observation.state", "actions"])
    n = len(cols)

    obs_sum = torch.zeros(obs_dim, dtype=torch.float64)
    obs_sq_sum = torch.zeros(obs_dim, dtype=torch.float64)
    obs_min = torch.full((obs_dim,), float("inf"))
    obs_max = torch.full((obs_dim,), float("-inf"))

    act_sum = torch.zeros(action_dim, dtype=torch.float64)
    act_sq_sum = torch.zeros(action_dim, dtype=torch.float64)
    act_min = torch.full((action_dim,), float("inf"))
    act_max = torch.full((action_dim,), float("-inf"))

    for start in tqdm(range(0, n, batch_size)):
        batch = cols[start : start + batch_size]
        obs = torch.stack(batch["observation.state"]).double()
        act = torch.stack(batch["actions"]).double()

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
