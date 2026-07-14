# DECO x ManiSkill-ViTac 2026：端到端训练流程（中文版）

*[English version](MANISKILL_VITAC2026_PIPELINE.md)*

这是本 fork 把 DECO 适配到官方 ManiSkill-ViTac 2026 LeRobot（v3.0）数据规范的操作手册。内容包括数据管线、两阶段训练方案、为了跑通真实训练而修复的每一个 bug，以及如何从一份原始数据集完整复现整个流程。

代码层面的改动总览见 README 里 "ManiSkill-ViTac 2026 adaptation" 那一节；这份文档更深入一层——记录实际的操作步骤，以及过程中踩到的各种坑，这样下一个人（或者未来的某次会话）不用再重新踩一遍。

## 1. 架构回顾

- **模型**：`models/deco_vitac/DECOVitac` —— 和原始 DECO 一样的联合注意力 diffusion/flow-matching transformer，区别是把 `obs_dim` 和 `act_dim` 解耦、用基于视觉的触觉 RGB 编码器（`tactile_img_encoder.py`）替换了原来的标量触觉区域数据、用一个冻结的文本编码器（`lang_encoder.py`）做语言条件（而不是 one-hot 任务索引）。
- **两阶段训练方案**（沿用 DECO 自己的约定，这次适配没有改）：
  1. **Stage 1**（`config/deco_vitac2026_vis.yaml`）：纯视觉。`use_tactile: False`、`plugin: False`。端到端训练 ResNet34 backbone、联合注意力 blocks 和动作预测头。
  2. **Stage 2**（`config/deco_vitac2026_tactile.yaml`）：`use_tactile: True`、`plugin: True`。把 stage 1 的 checkpoint 加载进 `pretrain_model_path`，冻结所有跟 stage 1 名字+形状对得上的参数，只训练新加的触觉编码器、触觉 cross-attention，以及每层 attention 里新插入的 LoRA 风格 `PI_Adapter`（rank 32）。
- **数据管线**：`lerobot_dataset.py` 里的 `ManiskillVitacDataset` 包了一层 lerobot 自己的 `LeRobotDataset`，把它适配成两个阶段的 `train_one_epoch.py` 都需要的 `(img1, img2, tactile_imgs, obs, action, mask, lang_embed)` 七元组。

## 2. 为什么数据集必须是 lerobot v3.0

当前装的 `lerobot` 版本（写这份文档时是 0.4.4）会直接拒绝通过 `LeRobotDataset` 加载 v2.1 格式的数据集：

```
BackwardCompatibilityError: The dataset you requested is in 2.1 format.
We introduced a new format since v3.0 which is not backward compatible with v2.1.
```

这不是本仓库代码里的选择——本仓库数据管线依赖的 delta_timestamps 分块、`actions_is_pad` 这些机制，在这个 lerobot 版本里只对 v3.0 格式的数据集存在。所以任何官方发布的 v2.1 版 ManiSkill-ViTac 2026 数据集，在能用之前都必须先转换一次。

## 3. 环境搭建

在 GPU 训练机器上（单卡就够用；多卡的话相应调整 `--device_id`/`--batch-size`）：

```bash
python -m venv deco_venv && source deco_venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` 里 `lerobot[dataset]` 和 `wandb` 是不锁版本号的（`lerobot` 这种迭代很快的库，建议每个项目单独锁一个具体 tag）。里面也列了 `matplotlib`——`train.py` 画 loss 曲线要用，但如果只挑着装依赖很容易漏掉。

下载两个 config 里 `img_pretrain` 指向的 ImageNet 预训练 ResNet34 权重：

```bash
mkdir -p ~/.cache/torch/hub/checkpoints
curl -L -o ~/.cache/torch/hub/checkpoints/resnet34-b627a593.pth \
  https://download.pytorch.org/models/resnet34-b627a593.pth
```

## 4. 转换数据集（v2.1 -> v3.0）

```bash
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
  --repo-id=<HF_DATASET_REPO_ID> \
  --push-to-hub=false
```

关于这条命令有两点要注意：

- **除非你拥有目标 HF 仓库的写权限，否则必须加 `--push-to-hub=false`。** 这个脚本默认行为是把转换好的 v3.0 结果推回 Hub（这样其他人以后能直接下载 v3.0 版本），如果你没有写权限，它会在最后一步崩溃，报 `401 Unauthorized` / `RepositoryNotFoundError`——尽管这时候实际的本地转换（数据文件、episode 元数据）早就已经成功完成了。也不要传自定义的 `--root`；默认行为会复用本地已经缓存好的 v2.1 数据（按 `repo_id` 缓存在 `~/.cache/huggingface/lerobot/<repo_id>` 下），转换出的 v3.0 结果也写在同一个目录。如果传一个新的 `--root`，它会尝试按一个 `v2.1` 的 Hub revision tag 重新下载 v2.1 源数据，而大多数数据集根本没有这个 tag。

- **已知 bug：转换后的 `index` 列跟它自己的 episode 元数据对不上。** `LeRobotDataset._get_query_indices` 在没有指定 `episodes=` 子集的情况下，会直接把每一行的 `index` 值当作绝对行位置来用（参见只有传了 `episodes` 才会构建的 `_absolute_to_relative_idx`）。v2.1→v3.0 的转换工具把 `index` 列原封不动地从 v2.1 源数据搬了过来，而不是按新算出来的 `meta/episodes/*.parquet` 里的 `dataset_from_index`/`dataset_to_index` 边界重新编号。结果就是：除了第一个 episode，每个 episode 的每一次 delta_timestamps 分块查询——包括当前帧本身对应的 delta=0 查询——都会落在这个 episode预期的 index 范围之外，导致 `actions_is_pad` 对整个 chunk 都返回 `True`。如果一个训练 batch 里的样本全部来自这种行，`mask.sum() == 0`，`_masked_flow_loss` 里的除法就会变成 `0/0 = NaN`。在一个 500-episode 的数据集上验证过：499/500 个 episode 受影响，每个 episode 自己内部的 `index` 是连续的，只是整体偏移了，跟元数据期望的起点对不上。

  修复方法（机械式的一次性操作，每个转换出来的数据集都要跑一次）：
  ```bash
  python utils/fix_v30_dataset_index.py --root ~/.cache/huggingface/lerobot/<repo_id>
  ```
  这个脚本把每个 episode 的 `index` 列整体平移一个常数，让它跟自己的 `dataset_from_index` 对齐。验证是否修复成功：跑几步训练，确认各个 batch 的 `mask.float().mean()` 不会出现异常偏低/等于零的情况——具体机制见 `utils/fix_v30_dataset_index.py` 的 docstring。

## 5. 计算数据集统计量和语言 embedding 缓存

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

**性能提示**：`cal_mean_std_lerobot.py` 以前是给 `LeRobotDataset(...)` 传 `episodes=train_episodes` 来限定训练集划分的。在一个 500-episode / ~50GB 的数据集上，这样跑 20 多分钟都跑不完（CPU 占用很高，RSS 缓慢增长，但磁盘 I/O 完全没有新增——和 lerobot 0.4.4 在 `episodes=` 列表很大时走了一条很慢的内存拼接/过滤路径的特征吻合）。修复方法是改成**不做过滤**地加载整个数据集，之后按行位置筛选出训练集（跟 `lerobot_dataset.py` 里 `ManiskillVitacDataset` 自己的做法一致，原因也一样——具体可以看那个文件里关于 `episodes=` 过滤不可靠/慢的注释）。改完之后同一个数据集不到 15 秒就能跑完。如果你发现 `cal_mean_std_lerobot.py` 卡住不动，十有八九就是这个问题——检查一下 `git log`，确认你用的是按行位置筛选的版本，而不是又改回 `episodes=` 的版本。

把算出来的 `observation_*`/`action_*` 统计量同时粘贴进 `config/deco_vitac2026_vis.yaml` **和** `config/deco_vitac2026_tactile.yaml` 的 `data:` 字段（两边要保持一致，这样 stage 2 用的归一化方式才跟 stage 1 训练时一样）。两个 config 的 `dataset.root` 都指向同一个本地 v3.0 目录，`dataset.lang_embed_cache` 指向生成的 JSON 文件。

## 6. 启动训练

### 单卡训练的坑：`--distributed` 是个坏掉的 argparse 参数

`train.py` 里 `--distributed` 声明的是 `type=bool`。argparse 的 `type=bool` 其实就是对你传的字符串调用 `bool(...)`，而 `bool("False")` 在 Python 里是 `True`（任何非空字符串都是真值）——所以 `--distributed False` 并不会真的把它关掉。真正跑单卡有两种办法：

```bash
# 方法 A：传一个空字符串，bool("") 才会正确地算出 False
python train.py --config config/deco_vitac2026_vis.yaml --distributed '' --amp True \
  --device_id '0' --batch-size 256 --num-workers 32 \
  --lr 1e-4 --lr_f 5e-6 --warm_up_epoch 1 --epochs 20 --val_per_epoch 2 --save_period 5 \
  --logs ./logs/log_deco_vitac_vis --wandb

# 方法 B：--distributed 还是传 True，但用 torchrun 只起一个进程来跑
#（彻底绕开 argparse 那个坑；跟仓库里多卡训练的写法保持一致）
torchrun --nproc_per_node=1 train.py --config config/deco_vitac2026_vis.yaml \
  --distributed True --amp True --device_id '0' ...
```
（注意：如果你自己加的某个 CLI 参数刚好和 `torchrun` 自带的某个参数前缀重合——比如某个以 `--logs` 开头的参数跟 `torchrun` 自己的 `--logs-specs` 撞了——`torchrun` 自己的参数解析器可能会把它当成有歧义的前缀匹配吞掉。方法 A 完全不会有这个问题。）

### 已知 bug：`dist.barrier()`/`dist.all_reduce()` 被无条件调用

`models/deco_vitac/train_one_epoch.py` 的 `train()` 和 `val()`（以及继承自同一套原始 DECO 模板的 `models/deco/`、`models/dp/`、`models/act/` 里的对应函数）在每个训练 epoch 结束后调用 `dist.barrier()`，在每个验证 batch 调用 `dist.all_reduce()`，而且都是**无条件调用**——写的时候默认 DDP 一定是初始化好的。单卡跑、不走 `torch.distributed.init_process_group()`（也就是上面的 `--distributed ''`）的话，会在第一个 epoch 刚跑完的时候崩溃：

```
ValueError: Default process group has not been initialized, please make sure to call init_process_group.
```

已经在 `models/deco_vitac/train_one_epoch.py` 里修好了，给这两处调用都加上了 `if dist.is_initialized():` 的判断。另外三个模型（`deco`、`dp`、`act`）也有同样的 bug，还没修——如果要在这几个模型上跑单卡训练，得先用同样的方式修一下。

### Stage 1 到 Stage 2 的交接

Stage 1 得真正跑完（或者至少产生一个 `best.pth`，第一次验证 epoch 之后就会有）才能开始 Stage 2——`config/deco_vitac2026_tactile.yaml` 的 `pretrain_model_path` 需要指向它。与其人工盯着，不如自动化这个交接：写一个小 watcher 脚本，轮询 stage 1 的进程是否退出，确认 `logs/<stage1_run>/best.pth` 确实存在，用一次针对性的文本替换（不是 YAML 反序列化再序列化——那样会把 config 文件里所有的说明注释都删掉）把 `pretrain_model_path` 原地改好，然后用同样的单卡配置启动 stage 2。用 `nohup ... & disown` 让它脱离终端在训练机器上独立跑，不依赖启动它的那个会话。

## 7. 监控

```bash
wandb login <从 https://wandb.ai/authorize 拿到的 API key>
```
然后在 `train.py` 命令后面加 `--wandb`（可以再加 `--wandb_project`/`--wandb_entity`/`--wandb_run_name`）。每个 step 会记录 `train/step_loss`、`lr`、`epoch`；每个验证 epoch 会记录 `val/epoch_loss`、`val/mae_mean`、`val/mae_per_dim`（这是归一化空间里的 L1 误差，不是反归一化之后的真实物理单位）。

**"这个 loss 数值正常吗"的判断依据**：初始化时动作预测头是 zero-init 的（`deco_vitac.py` 里的 `initialize_weights()`，只有 `plugin=False` 也就是 stage 1 时才会调用），所以网络一开始预测的 velocity 约等于 0。flow-matching 的目标是 `noise - action`；因为 `noise`（标准正态）和 `action`（归一化之后按定义就是单位方差）都是独立的零均值单位方差变量，所以 `E[(noise-action)^2] = 2`。也就是说，一个还没训练的模型 loss 应该刚好落在 **2.0** 附近——实际观察到的也确实是这样——而且warmup 结束、学习率到达目标值之后，第一个 epoch 左右就应该能看到明显往下降。如果好几个 epoch 之后 loss 还是钉在 2.0 附近不动，那就说明有问题了（先检查上面那个 `index` 列的修复有没有生效——一个本该产生 NaN 的 batch，如果在别的地方被悄悄 `nan_to_num` 掉或者被 mask 掉了，是一种很隐蔽的、看起来没崩溃但实际上学不到东西的方式）。

## 8. 数据吞吐量提示

在单张 80GB 级别的 GPU 上，模型本身算得足够快，GPU 利用率会是突发式的（短暂冲到接近 100%，然后长时间空闲），而不是持续跑满——瓶颈在 CPU 端的图像解码（每个样本 2 路 RGB + 4 路触觉图像，都是以字节形式内嵌在 parquet 文件里的），而不是 GPU 算力。增大 `--num-workers` 只在解码吞吐量还没到瓶颈之前才有用；先做 profiling 再假设"worker 越多越快"。
