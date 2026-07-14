# DECO x ManiSkill-ViTac 2026: end-to-end training pipeline

*[中文版](MANISKILL_VITAC2026_PIPELINE.zh-CN.md)*

This is a runbook for the fork's adaptation of DECO to the official ManiSkill-ViTac 2026
LeRobot (v3.0) data contract. It covers the data pipeline, the two-stage training recipe,
every bug that had to be fixed to get a real training run going, and how to reproduce the
whole thing from a fresh dataset.

The high-level code changes are summarized in the README's "ManiSkill-ViTac 2026
adaptation" section; this document goes one level deeper -- the actual sequence of steps,
and the failure modes hit along the way, so the next person (or a future session) doesn't
have to rediscover them.

## 1. Architecture recap

- **Model**: `models/deco_vitac/DECOVitac` -- same joint-attention diffusion/flow-matching
  transformer as the original DECO, with `obs_dim` decoupled from `act_dim`, a vision-based
  tactile RGB encoder (`tactile_img_encoder.py`) instead of the original scalar tactile
  regions, and language conditioning via a frozen text encoder (`lang_encoder.py`) instead
  of a one-hot task index.
- **Two-stage recipe** (DECO's own convention, unchanged for this adaptation):
  1. **Stage 1** (`config/deco_vitac2026_vis.yaml`): vision-only. `use_tactile: False`,
     `plugin: False`. Trains the ResNet34 backbone, joint-attention blocks, and action head
     end to end.
  2. **Stage 2** (`config/deco_vitac2026_tactile.yaml`): `use_tactile: True`,
     `plugin: True`. Loads stage 1's checkpoint into `pretrain_model_path`, freezes every
     parameter that matches stage 1 by name+shape, and trains only the new tactile encoder,
     tactile cross-attention, and per-layer LoRA-style `PI_Adapter` (rank 32) modules.
- **Data pipeline**: `lerobot_dataset.py`'s `ManiskillVitacDataset` wraps lerobot's own
  `LeRobotDataset` and adapts it to the `(img1, img2, tactile_imgs, obs, action, mask,
  lang_embed)` 7-tuple both stages' `train_one_epoch.py` expect.

## 2. Why the dataset must be lerobot v3.0

The installed `lerobot` release (0.4.4 at the time of writing) hard-refuses to load a v2.1
dataset through `LeRobotDataset`:

```
BackwardCompatibilityError: The dataset you requested is in 2.1 format.
We introduced a new format since v3.0 which is not backward compatible with v2.1.
```

This isn't a choice made in this repo's code -- the delta_timestamps/chunking/`actions_is_pad`
machinery this repo's data pipeline relies on only exists for v3.0-format datasets in this
lerobot version. So any officially-released v2.1 ManiSkill-ViTac 2026 dataset has to be
converted once before it's usable here.

## 3. Environment setup

On the GPU training machine (a single-GPU box is enough; adjust `--device_id`/`--batch-size`
for more GPUs):

```bash
python -m venv deco_venv && source deco_venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins `lerobot[dataset]` and `wandb` unversioned (`lerobot` in particular
moves fast enough that pinning a specific tag is worth doing per-project). It also lists
`matplotlib`, which `train.py`'s loss-curve plotting needs but which is easy to miss if you
only `pip install` a subset of the file by hand.

Download the ImageNet-pretrained ResNet34 checkpoint that `img_pretrain` in both configs
points at:

```bash
mkdir -p ~/.cache/torch/hub/checkpoints
curl -L -o ~/.cache/torch/hub/checkpoints/resnet34-b627a593.pth \
  https://download.pytorch.org/models/resnet34-b627a593.pth
```

## 4. Converting the dataset (v2.1 -> v3.0)

```bash
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
  --repo-id=<HF_DATASET_REPO_ID> \
  --push-to-hub=false
```

Two things to know about this command:

- **`--push-to-hub=false` is required unless you own the target HF repo.** The script's
  default behavior is to push the converted v3.0 result back to the Hub (so other users can
  download the v3.0 version directly) and it will crash with a `401 Unauthorized` /
  `RepositoryNotFoundError` at the very last step if you don't have write access -- even
  though the actual local conversion (data files, episode metadata) already completed
  successfully by that point. Don't pass a custom `--root`; the default lets it reuse
  whatever v2.1 copy is already cached locally, keyed by `repo_id` under
  `~/.cache/huggingface/lerobot/<repo_id>`, and writes the v3.0 result there too. Passing a
  fresh `--root` instead makes it try to re-download the v2.1 source keyed to a `v2.1` Hub
  revision tag, which most datasets don't have.

- **Known bug: the converted `index` column doesn't match its own episode metadata.**
  `LeRobotDataset._get_query_indices` uses each row's `index` value directly as an absolute
  row position when no `episodes=` subset is requested (see `_absolute_to_relative_idx`,
  only built when `episodes` is passed). The v2.1->v3.0 converter carries the `index` column
  over verbatim from the v2.1 source instead of renumbering it to match the newly-computed
  `dataset_from_index`/`dataset_to_index` boundaries in `meta/episodes/*.parquet`. The
  result: for every episode but the first, every delta_timestamps chunk query -- including
  the delta=0 query for the *current* frame -- falls outside that episode's expected index
  range, so `actions_is_pad` comes back `True` for the whole chunk. A training batch drawn
  entirely from such rows has `mask.sum() == 0`, and `_masked_flow_loss`'s division turns
  into `0/0 = NaN`. Confirmed on a 500-episode dataset: 499/500 episodes affected, each
  episode's `index` column internally contiguous but offset from where the meta expects it
  to start.

  Fix (mechanical, one-time, per converted dataset):
  ```bash
  python utils/fix_v30_dataset_index.py --root ~/.cache/huggingface/lerobot/<repo_id>
  ```
  This shifts each episode's `index` column by a constant so it lines up with its
  `dataset_from_index`. Verify it worked by running a few training steps and confirming
  `mask.float().mean()` isn't suspiciously low/zero across batches -- see
  `utils/fix_v30_dataset_index.py`'s docstring for the full mechanism.

## 5. Computing dataset statistics and the language embedding cache

```bash
python utils/cal_mean_std_lerobot.py \
  --repo-id <HF_DATASET_REPO_ID> \
  --root ~/.cache/huggingface/lerobot/<repo_id> \
  --val-ratio 0.1 --split-seed 42 --obs-dim 20 --action-dim 20 \
  --save-path assets/stats/<name>.yaml

python utils/cal_text_embeddings.py \
  --repo-id <HF_DATASET_REPO_ID> \
  --root ~/.cache/huggingface/lerobot/<repo_id> \
  --save-path ./assets/lang_embeddings/<name>.json --device cpu
```

**Performance note**: `cal_mean_std_lerobot.py` used to pass `episodes=train_episodes` to
`LeRobotDataset(...)` to restrict to the train split. On a 500-episode / ~50GB dataset this
did not finish in 20+ minutes (high CPU usage, RSS slowly climbing, zero further disk I/O --
consistent with an expensive in-memory concatenation/filter path in lerobot 0.4.4 when the
`episodes=` list is large). The fix was to load the dataset **unfiltered** and restrict to
train rows by position afterward (same pattern `lerobot_dataset.py`'s `ManiskillVitacDataset`
already uses, and for the same underlying reason -- see that file's own comment about
`episodes=` filtering being unreliable/slow for this use case). That version finishes in
under 15 seconds on the same dataset. If you ever see `cal_mean_std_lerobot.py` hang, this
is almost certainly why -- check `git log` on that file to make sure you have the
position-based version, not one that reintroduced `episodes=`.

Paste the resulting `observation_*`/`action_*` stats into **both**
`config/deco_vitac2026_vis.yaml` and `config/deco_vitac2026_tactile.yaml`'s `data:` block
(kept identical between the two so stage 2 sees the same normalization stage 1 was trained
on). Point both configs' `dataset.root` at the same local v3.0 directory and
`dataset.lang_embed_cache` at the generated JSON.

## 6. Launching training

### Single-GPU gotcha: `--distributed` is a broken argparse flag

`train.py` declares `--distributed` with `type=bool`. Argparse's `type=bool` just calls
`bool(the_string_you_passed)`, and `bool("False")` is `True` in Python (any non-empty
string is truthy) -- so `--distributed False` silently does **not** disable distributed
mode. Two ways to actually run single-GPU:

```bash
# Option A: pass an empty string, which bool("") correctly evaluates to False
python train.py --config config/deco_vitac2026_vis.yaml --distributed '' --amp True \
  --device_id '0' --batch-size 256 --num-workers 32 \
  --lr 1e-4 --lr_f 5e-6 --warm_up_epoch 1 --epochs 20 --val_per_epoch 2 --save_period 5 \
  --logs ./logs/log_deco_vitac_vis --wandb

# Option B: keep --distributed True but launch through torchrun with a single process
# (avoids the argparse footgun entirely; matches this repo's multi-GPU convention)
torchrun --nproc_per_node=1 train.py --config config/deco_vitac2026_vis.yaml \
  --distributed True --amp True --device_id '0' ...
```
(Note: if any of your own extra CLI flags share a prefix with one of `torchrun`'s own
options -- e.g. a flag starting with `--logs` colliding with `torchrun`'s `--logs-specs` --
`torchrun`'s own arg parser can swallow it as an ambiguous prefix match. Option A sidesteps
this entirely.)

### Known bug: `dist.barrier()`/`dist.all_reduce()` called unconditionally

`models/deco_vitac/train_one_epoch.py`'s `train()` and `val()` (and, inherited from the same
original-DECO template, `models/deco/`, `models/dp/`, and `models/act/`'s equivalents) call
`dist.barrier()` after every training epoch and `dist.all_reduce()` per validation batch
**unconditionally** -- they were written assuming DDP is always initialized. Running single-GPU
without `torch.distributed.init_process_group()` (i.e. `--distributed ''` above) crashes at
the end of the very first epoch with:

```
ValueError: Default process group has not been initialized, please make sure to call init_process_group.
```

Fixed in `models/deco_vitac/train_one_epoch.py` by guarding both calls with
`if dist.is_initialized():`. The other three model families (`deco`, `dp`, `act`) have the
same bug and are not yet patched -- fix them the same way before trying single-GPU training
on those.

### Stage 1 -> stage 2 handoff

Stage 1 needs to actually finish (or at least produce a `best.pth`, which happens after the
first validation epoch) before stage 2 can start -- `config/deco_vitac2026_tactile.yaml`'s
`pretrain_model_path` needs to point at it. To automate the handoff instead of babysitting
it: a small watcher script polls for the stage-1 process to exit, confirms
`logs/<stage1_run>/best.pth` exists, patches the tactile config's `pretrain_model_path` in
place via a targeted text substitution (not a YAML round-trip -- `yaml.safe_load` +
`yaml.dump` would strip all the doc comments in the config file), and launches stage 2 with
the same single-GPU settings. Run it detached (`nohup ... & disown`) on the training
machine itself so it survives independently of whatever session launched it.

## 7. Monitoring

```bash
wandb login <your API key from https://wandb.ai/authorize>
```
then add `--wandb` (and optionally `--wandb_project`/`--wandb_entity`/`--wandb_run_name`) to
the `train.py` invocation. Logged per-step: `train/step_loss`, `lr`, `epoch`; per-validation-
epoch: `val/epoch_loss`, `val/mae_mean`, `val/mae_per_dim` (normalized-space L1 error, not
denormalized physical units).

**Sanity check for "is this loss number reasonable?"**: at initialization the action
prediction head is zero-initialized (`initialize_weights()` in `deco_vitac.py`, only run
when `plugin=False`, i.e. stage 1), so the network's predicted velocity starts at ~0. The
flow-matching target is `noise - action`; since both `noise` (standard normal) and
`action` (z-scored during normalization, so unit variance by construction) are independent
zero-mean unit-variance, `E[(noise-action)^2] = 2`. So an untrained model's loss should sit
right around **2.0** -- which is exactly what's observed in practice -- and should trend
down noticeably below that within the first epoch or so once warmup finishes and the
learning rate reaches its target value. If it's still pinned near 2.0 after several epochs,
something's wrong (check the `index`-column fix above first -- a NaN-producing batch that
gets silently `nan_to_num`'d or masked out elsewhere in a custom edit is a common way to
quietly break learning without an outright crash).

## 8. Data throughput note

On an 80GB-class single GPU, expect the model itself to be fast enough that GPU utilization
is bursty (brief spikes to ~100%, long idle stretches) rather than pinned high -- the
bottleneck is CPU-side image decoding (2 RGB + 4 tactile streams per sample, embedded as
bytes inside the parquet files) feeding the DataLoader, not GPU compute. Increasing
`--num-workers` only helps up to the point where decode throughput, not worker count, is the
limit; profile before assuming more workers = faster.
