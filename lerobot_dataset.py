import json
import random
import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms

from lerobot_compat import LeRobotDataset, LeRobotDatasetMetadata

# order matches models/deco_vitac/tactile_img_encoder.py's expected sensor ordering:
# (left-arm/left-pad, left-arm/right-pad, right-arm/left-pad, right-arm/right-pad)
TACTILE_KEYS = (
    "observation.images.tactile_left_0",
    "observation.images.tactile_right_0",
    "observation.images.tactile_left_1",
    "observation.images.tactile_right_1",
)


def episode_split(num_episodes: int, val_ratio: float = 0.1, seed: int = 42):
    """Split whole episodes (not frames) into train/val so validation never leaks
    near-duplicate frames from an episode that's also in train."""
    episodes = list(range(num_episodes))
    rng = random.Random(seed)
    rng.shuffle(episodes)
    n_val = max(1, round(num_episodes * val_ratio)) if num_episodes > 1 else 0
    val_episodes = sorted(episodes[:n_val])
    train_episodes = sorted(episodes[n_val:])
    return train_episodes, val_episodes


class ManiskillVitacDataset(Dataset):
    """Adapts the official ManiSkill-ViTac 2026 LeRobot dataset (v3.0, already converted
    from the officially-released v2.1 shards) to the (img1, img2, tactile_imgs, obs, action,
    mask, lang_embed) 7-tuple expected by models/deco_vitac/train_one_epoch.py.

    Wraps lerobot's own LeRobotDataset rather than reading parquet by hand -- it already
    knows how to resolve a local root or download+cache a repo_id from the Hub, chunk
    actions via delta_timestamps, and report per-chunk padding via `actions_is_pad`.
    """

    def __init__(
        self,
        repo_id,
        root=None,
        train=True,
        chunk_size=32,
        use_tactile=False,
        tactile_t_hist=1,
        img_size=(224, 224),
        img_mean=None,
        img_std=None,
        observation_mean=None,
        observation_std=None,
        action_mean=None,
        action_std=None,
        lang_embed_cache=None,
        lang_embed_dim=384,
        val_ratio=0.1,
        split_seed=42,
        train_augment=True,
        **_ignored,
    ):
        self.use_tactile = use_tactile
        self.tactile_t_hist = tactile_t_hist
        self.chunk_size = chunk_size
        self.lang_embed_dim = lang_embed_dim

        meta = LeRobotDatasetMetadata(repo_id, root=root)
        train_episodes, val_episodes = episode_split(meta.total_episodes, val_ratio, split_seed)
        episodes = train_episodes if train else val_episodes
        if len(episodes) == 0:
            raise ValueError(
                f"Episode split produced an empty {'train' if train else 'val'} set "
                f"(total_episodes={meta.total_episodes}, val_ratio={val_ratio}); use more episodes or a smaller val_ratio."
            )

        delta_timestamps = {"actions": [t / meta.fps for t in range(chunk_size)]}
        # NOTE: deliberately NOT passing `episodes=episodes` to LeRobotDataset here. Its
        # episode filtering builds an absolute-index map from the raw `index` column, which
        # on some officially-released ManiSkill-ViTac shards is non-contiguous across
        # episodes (an artifact of how the upstream collection assigned global frame
        # indices before splitting into per-task/color shards); that breaks delta_timestamps
        # boundary queries for held-out episode subsets. Instead we load the dataset
        # unfiltered (this path is unaffected, verified against KaiyueChen/black_smash_03)
        # and do the train/val split ourselves at the row-position level below, which is
        # always contiguous regardless of the `index` column's values.
        self.dataset = LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps)
        episode_col = self.dataset.hf_dataset.data.column("episode_index").to_numpy()
        self.valid_positions = np.where(np.isin(episode_col, np.array(episodes)))[0].tolist()
        if len(self.valid_positions) == 0:
            raise ValueError(f"No frames found for episodes {episodes} in {repo_id} (root={root}).")

        if observation_mean is None or action_mean is None:
            raise ValueError(
                "observation_mean/action_mean (and _std) must be provided -- run "
                "utils/cal_mean_std_lerobot.py on the train split first and paste the result "
                "into this config's `data:` section."
            )
        self.obs_mean = torch.tensor(observation_mean, dtype=torch.float32)
        self.obs_std = torch.tensor(observation_std, dtype=torch.float32).clamp_min(1e-8)
        self.action_mean = torch.tensor(action_mean, dtype=torch.float32)
        self.action_std = torch.tensor(action_std, dtype=torch.float32).clamp_min(1e-8)

        img_mean = img_mean or [0.485, 0.456, 0.406]
        img_std = img_std or [0.229, 0.224, 0.225]
        img_size = list(img_size)

        rgb_aug = []
        if train and train_augment:
            # PDF 6.4: light photometric jitter only, no flips/crops that would swap
            # left/right semantics or distort the (already learned) action frame.
            rgb_aug.append(
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=(0.7, 1.3), contrast=(0.8, 1.2), saturation=(0.8, 1.2))], p=0.5
                )
            )
        self.rgb_transform = transforms.Compose(
            [transforms.Resize(img_size), *rgb_aug, transforms.Normalize(mean=img_mean, std=img_std)]
        )
        # Tactile images are contact/deformation patterns, not natural photos: no photometric
        # jitter (PDF 7.3) -- robustness instead comes from whole-stream dropout, applied at
        # train time in models/deco_vitac/train_one_epoch.py, not here.
        self.tactile_transform = transforms.Compose([transforms.Resize(img_size), transforms.Normalize(mean=img_mean, std=img_std)])

        self.lang_cache = {}
        if lang_embed_cache is not None and pathlib.Path(lang_embed_cache).exists():
            with open(lang_embed_cache) as f:
                raw = json.load(f)
            self.lang_cache = {k: torch.tensor(v, dtype=torch.float32) for k, v in raw.items()}

    def __len__(self):
        return len(self.valid_positions)

    def _lang_embed(self, task: str) -> torch.Tensor:
        if task in self.lang_cache:
            return self.lang_cache[task]
        # Every task string in the split should have been embedded by
        # utils/cal_text_embeddings.py; zero-vector fallback only guards unseen prompts.
        print(f"[ManiskillVitacDataset] WARNING: no cached language embedding for task: {task!r}")
        return torch.zeros(self.lang_embed_dim, dtype=torch.float32)

    def __getitem__(self, idx):
        item = self.dataset[self.valid_positions[idx]]

        img1 = self.rgb_transform(item["observation.images.camera0"])
        img2 = self.rgb_transform(item["observation.images.camera1"])

        if self.use_tactile:
            frames = [self.tactile_transform(item[k]) for k in TACTILE_KEYS]
            tactile_imgs = torch.stack(frames, dim=0)  # (n_sensors, 3, H, W)
        else:
            h, w = img1.shape[-2:]
            tactile_imgs = torch.zeros(4 * self.tactile_t_hist, 3, h, w)

        obs = (item["observation.state"] - self.obs_mean) / self.obs_std
        action = (item["actions"] - self.action_mean) / self.action_std
        mask = ~item["actions_is_pad"]

        lang_embed = self._lang_embed(item["task"])

        return img1, img2, tactile_imgs, obs, action, mask, lang_embed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--use-tactile", action="store_true")
    args = parser.parse_args()

    dummy_stats = dict(
        observation_mean=[0.0] * 20, observation_std=[1.0] * 20,
        action_mean=[0.0] * 20, action_std=[1.0] * 20,
    )
    ds = ManiskillVitacDataset(args.repo_id, root=args.root, train=True, chunk_size=8, use_tactile=args.use_tactile, **dummy_stats)
    print("len(train):", len(ds))
    img1, img2, tactile_imgs, obs, action, mask, lang_embed = ds[0]
    print("img1", img1.shape, img1.dtype)
    print("img2", img2.shape, img2.dtype)
    print("tactile_imgs", tactile_imgs.shape, tactile_imgs.dtype)
    print("obs", obs.shape)
    print("action", action.shape)
    print("mask", mask.shape, mask.dtype, mask)
    print("lang_embed", lang_embed.shape)
