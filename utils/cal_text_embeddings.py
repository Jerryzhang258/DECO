import os
import sys
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lerobot_compat import LeRobotDatasetMetadata
from lang_encoder import FrozenTextEncoder, DEFAULT_LANG_MODEL

"""
Precompute a {task string: frozen-text-encoder embedding} cache for every unique task
prompt in a LeRobot dataset (PDF's "语言 embedding 离线缓存" recommendation). This avoids
running the text encoder on every training sample -- lerobot_dataset.py just does a dict
lookup by the raw task string.
"""


def build_cache(repo_id, root=None, model_name=DEFAULT_LANG_MODEL, save_path="./lang_embeddings.json", device="cpu"):
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    tasks = list(meta.tasks.index)
    print(f"found {len(tasks)} unique task string(s) in {repo_id}:")
    for t in tasks:
        print("  -", t)

    encoder = FrozenTextEncoder(model_name=model_name, device=device)
    embeddings = encoder.embed(tasks)

    cache = {task: embeddings[i].tolist() for i, task in enumerate(tasks)}
    with open(save_path, "w") as f:
        json.dump(cache, f)
    print(f"saved {len(cache)} embeddings (dim={embeddings.shape[-1]}) to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--root", type=str, default=None, help="Local dataset root (v3.0 layout). Omit to fetch from the HF Hub.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_LANG_MODEL)
    parser.add_argument("--save-path", type=str, default="./lang_embeddings.json")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    build_cache(args.repo_id, root=args.root, model_name=args.model_name, save_path=args.save_path, device=args.device)
