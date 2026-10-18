# 独立动作条件未来预测器

本模块只训练未来预测器，不更新策略，也不在 Memory Slot 中存储动作。
输入是当前冻结 DINO Patch、本体状态和干净专家动作窗口，输出是执行窗口结束后的 DINO Patch。
历史检索和 Slot Gate 贡献评估尚未接入本训练入口。

## 与 DynaGuide 对齐的基础配置

`imitation_learning/configs/train_future_predictor.yaml` 的默认值为：

| 配置 | 当前值 | DynaGuide |
| --- | --- | --- |
| Transformer 宽度 / 层数 / 头数 | 当前实验设置 512 / 6 / 8 | 384 / 6 / 8 |
| FFN 宽度 | 2048 | PyTorch 默认 2048 |
| 激活、归一化 | ReLU、Post-LN，无额外 encoder final norm | 相同 |
| Transformer dropout | 0.1 | PyTorch 默认 0.1；代码中的 embedding dropout 0 是另一个设置 |
| 优化器 / 学习率 | Adam / 1e-4，固定学习率 | 相同 |
| weight decay / betas / epsilon | 0 / [0.9, 0.999] / 1e-8 | 相同 |
| 梯度裁剪 | 不裁剪，仍检查非有限梯度 | DynaGuide 不裁剪 |
| batch size / workers | 16 / 4 | 相同 |
| pin memory / persistent workers | false / false | 相同 |
| 特征预测损失 | MSE | 相同 |

以下部分沿用本项目的设计：

- 视觉特征使用已有的冻结 DINOv2 base 缓存，每个 Patch 768 维；DynaGuide 使用 small，每个 Patch 384 维。输入在预测器内投影为 `model.hidden_dim`（当前 512），预测结果映射回 768 维，不需要重算缓存。
- 动作窗口取真正执行的 10 步；DynaGuide 示例使用 16 步。这里逐步编码动作，并使用可学习的 Future Query；DynaGuide 将动作压成一个 Token，再读取更新后的图像 Token。
- 标签是对应的 `t+10` 帧；DynaGuide 原始数据加载器取示范终帧，并可将尾部动作补零。
- 不引入 RGB 重建器和动作噪声增强。DynaGuide 的 RGB 重建支路不反向更新未来特征预测器，动作加噪由 `--noised` 选择。
- `training.epochs` 以当前 YAML 为准。每个 epoch 抽取 `len(train_dataset)` 个样本，而 DynaGuide 的 `num_epochs` 每轮只执行 100 个 batch；README 的 6000 对应约 60 万次更新，不能直接当作这里的 6000 轮。

共享 `FutureFeaturePredictor` 的旧默认行为保持不变，新增 Transformer 选项由独立训练 YAML 显式启用，已有策略和第一版预测器 checkpoint 仍可加载。

## 输入、输出与数据加载

以单个 head_camera、`L=10`、动作和本体状态均为 14 维为例，DataLoader 每个 batch 返回：

| 字段 | Shape | 含义 |
| --- | --- | --- |
| `current_features` | `[B, 256, 768]` | `t` 时刻当前图像的冻结 DINO Patch |
| `proprio` | `[B, 14]` | `agent_pos[t]`，物理值 |
| `actions` | `[B, 10, 14]` | 干净专家动作 `action[t:t+10]`，物理值 |
| `target_features` | `[B, 256, 768]` | 专家图像在 `t+10` 的冻结 Patch，训练标签 |

动作/本体维度从实际 Zarr 读取，不固定为 14；多相机时两个特征字段的 Token 数为 `camera_num * 256`。
模型 `forward(current_features, proprio, actions)` 返回 `[B, camera_num * 256, 768]`，不返回 RGB。
动作与本体状态在模型内使用训练集统计标准化；输入可传物理值，但不能传策略自己的标准化值。

内部数据流为：

```text
256 当前图像 Patch + 1 本体 Token → 投影为 H 维 ─┐
10 个动作 Token，H 维 ─────────────────────────┼→ 6 层 Transformer
256 个可学习 Future Query，H 维 ───────────────┘       ↓
                                              256 个 Future 输出
                                                      ↓
                                             映射回 768 维 → MSE
```

`H=model.hidden_dim`，当前为 512。单相机时合计 523 个 Token。当前观测仅含当前帧，预测器没有历史图像或历史动作输入。

数据集直接只读打开转换后的 Zarr：

```text
task.zarr/
  episode_0/
    head_camera_patch_feature  # [T, 257, 768]，第一个是 CLS
    agent_pos                  # [T, Dp]
    action                     # [T, Da] 或 [T-1, Da]
  episode_1/...
```

加载时先校验缓存编码器、预处理、相机、shape 和完成标记，再按 episode 划分训练/验证。
为每个 episode 建立所有完整窗口的索引；读取 `t` 和 `t+L` 的缓存时去掉 CLS，动作读取半开区间 `[t, t+L)`。
例如 404 帧、窗口 10，对应 394 个有效锚点 `t=0..393`，尾部不补齐，不跨 episode。
数据按需从磁盘读取，启动时只扫描元数据、索引及拟合标准化所需的低维数据，不将全部图像特征载入内存。
校验时保留 Zarr Array 句柄，训练样本复用这些句柄，避免每帧重复读取 `.zarray` 等元数据。
使用 spawn worker 时不序列化打开的句柄，worker 首次访问后独立复用。

训练采用有放回的加权采样，每轮抽取训练集长度个样本：各任务总概率相等，任务内的完整窗口等概率；单任务就是均匀采样。
验证按固定顺序遍历，统计完整验证集；只有限制 `max_val_batches` 的短检查会提前停止。
训练阶段不再次运行 DINO、不接入策略模型，优化器只更新未来预测器。

## 默认多任务训练脚本

激活 `RMBench` 或 `RMBench_90` 环境后运行，脚本使用当前环境中的 Python：

```bash
conda activate RMBench          # 或 conda activate RMBench_90
bash train_pre.sh               # 自动选择设备，按 YAML 设置抽取各任务轨迹
bash train_pre.sh 4             # 指定 cuda:4
bash train_pre.sh --dry-run     # 校验数据、显示训练计划，不启动训练
```

默认扫描 `RMBench/data/imitation_learning_policies/*.zarr`，将所有转换后的任务数据集共同训练。
当前已有 observe_and_pickup、put_back_block、rearrange_blocks、swap_T、swap_blocks 五个任务，默认全部纳入。
每个任务先按 `data.episodes_per_task` 无放回随机抽取完整示范轨迹，再划分训练/验证集；不足指定数量时使用全部可用轨迹。
抽样由 `data.seed`（默认跟随 `seed=42`）固定，每个 epoch 使用同一批轨迹。
启动摘要和 `--dry-run` 显示每个任务选中的轨迹数、训练/验证轨迹数和窗口数。
`--data-dir PATH` 或 `RMBENCH_PROCESSED_DATA_DIR` 可以切换数据目录；Python 可通过 `POLICY_PYTHON` 或 `PYTHON_BIN` 指定。
脚本默认声明共同接口 `rmbench_joint_target_v1`，数据加载器继续校验维度、时序和缓存兼容性；外部数据应通过 `data.control_interface=...` 指定实际接口。
显式传入 `--dataset` 或 YAML `data.sources` 时优先使用指定选集。续训使用 checkpoint 中已经保存的选集，新增加的数据集不会改变旧训练。

`output_dir: null` 时，每次新训练自动创建独立目录 `outputs/future_predictor/shared_YYYYMMDD_HHMMSS_微秒`。
启动摘要会显示实际路径，`--dry-run` 只显示计划、不创建目录。
使用 `--output-dir PATH`、`output_dir=PATH` 或 YAML 显式指定目录时按该路径保存，非空目录仍拒绝覆盖。
`--resume PATH/last.pt` 使用原目录和已保存配置，不生成新目录。参数可直接透传：

```bash
bash train_pre.sh 4 --epochs 100 --output-dir outputs/future_predictor/shared_run2
bash train_pre.sh --resume outputs/future_predictor/shared/last.pt
bash train_pre.sh data.episodes_per_task=30 --output-dir outputs/future_predictor/shared_30
bash train_pre.sh data.episodes_per_task=null --output-dir outputs/future_predictor/shared_all
```

`data.episodes_per_task=null` 使用全部轨迹。旧 checkpoint 续训沿用保存的配置与划分，
不会自动应用当前 YAML 的抽样数量；要切换选集，应开始一次新训练。

### 多卡训练

`train_pre.sh` 支持逗号分隔的 GPU 编号，通过当前环境的 `python -m torch.distributed.run`
启动单机 DDP；端口自动分配。例如以总 batch 16、新实验 110 个 epoch 运行：

```bash
bash train_pre.sh 0,1 loader.batch_size=8 training.epochs=110
bash train_pre.sh 0,1,2,3 loader.batch_size=4 training.epochs=110
bash train_pre.sh 0,1 --resume outputs/future_predictor/你的实验目录/last.pt
```

`loader.batch_size` 表示 **每卡/每进程的 batch**，总 batch = 配置值 × 卡数。
例如配置 16 时，两卡总 batch 32，四卡总 batch 64；不要求配置值能被卡数整除。
要保持总 batch 16，两卡配置 8，四卡配置 4，如上例所示；学习率不自动乘以卡数。
`global_step` 表示同步梯度后的一次 optimizer 更新，多卡不会重复计数。
固定 epoch 数时，每卡 batch 不变、增加卡数会减少每个 epoch 的更新次数；
样本使用总量基本不变。若要对齐 DynaGuide 的 60 万次更新，应按总 batch 计算训练量。
任务均衡采样先生成一份共同的随机序列，再按卡切分；每轮训练抽取总量保持为训练窗口数。
当窗口总数不能被卡数整除时，最多补齐 `卡数 - 1` 个样本，使所有卡的最后一个 batch 一致。
`loader.num_workers` 是每个进程的 worker 数量，多卡使用 spawn 避免继承通信线程。

数据索引和归一化统计由主进程计算一次后分发。SwanLab、TensorBoard、进度条和 checkpoint
只由主进程写入，step loss 是所有卡的平均值。验证在主卡使用配置的单卡 batch 完成，
同任务换动作对照在该验证 batch 内进行；其余卡等待验证完成。
checkpoint 保存未包装的模型权重，旧加载接口可直接使用；同样卡数续训会恢复每卡的随机状态。
单卡和多卡之间可以切换，但需保持原有 global batch、数据及训练设置；更换卡数会改变
Dropout 的随机过程，不能保证与原运行逐位相同。
切换卡数续训时，显式调整 `loader.batch_size` 以保持 checkpoint 的总 batch；总 batch
发生变化会拒绝续训。早期以总 batch 记录配置的旧 checkpoint 会自动换算成当前卡数的
每卡 batch；无法整除时需选择兼容的卡数。新 checkpoint 同时保存每卡和总 batch。

总 batch 较小或 JuiceFS 读取受限时，通信、数据加载和单卡验证会限制加速。
先比较两卡与四卡的稳定每步耗时，再决定使用几张卡。

2026-10-10 在 240 的 RTX 5090、`RMBench_90` 上做过计算部分短测：当前 512 宽/6 层模型，
总 batch 16，FP32 合成输入，预热 5 步后统计 15 步；单卡约 35.45 ms/更新，
双卡约 26.15 ms，四卡约 25.95 ms。双卡约快 1.36 倍，四卡额外收益很小，建议先用双卡。
这些数字不包含 JuiceFS 读取、启动、验证或 checkpoint 保存，不能当成完整训练的加速比。

## 指定单个任务

在 `policy/imitation-learning-policies` 下、使用已有策略环境运行：

```bash
python scripts/train_future_predictor.py \
  --dataset ../../data/imitation_learning_policies/rearrange_blocks-demo_clean-50.zarr \
  --output-dir outputs/future_predictor/rearrange_blocks \
  --device cuda:0 \
  seed=42
```

默认使用 head_camera 的完整 Patch，执行窗口为 10；特征维度、动作维度和本体维度从缓存、数据读取。
需要先有完整的 `*_patch_feature` 缓存。缺失时使用现有脚本：

```bash
python scripts/extract_rmbench_dinov2_features.py \
  ../../data/imitation_learning_policies/rearrange_blocks-demo_clean-50.zarr \
  --cameras head_camera --feature-kind patches --device cuda:0
```

数据项严格为 `(v[t], p[t], A[t:t+L], v[t+L])`。不跨 episode，不重复末帧，不使用示范终帧替代对应未来帧。
转换器的 `action[t]=vector[t+1]` 表示从当前状态开始执行的下一控制目标，因此 L 步对应 `obs[t+L]`。
其他数据来源必须核实这一时序约定。

默认先随机选取 20 条 episode，再按 98% 比例划分：19 条训练、1 条验证。
标准化仅使用选中的训练 episode；未选中的轨迹不参与训练、验证或标准化。
完整训练/验证 episode ID 保存在 `data_metadata.json` 和 checkpoint 中，便于复现与核对。
关闭轨迹抽样后，选择算法与现有策略数据集一致；还需相同 episode 集合、seed 和比例。
若策略另有选集，应显式指定下面的划分，避免验证泄漏。

## 显式划分与多任务

创建独立 YAML，例如：

```yaml
data:
  # 按实际机器人、关节顺序、动作/本体含义和单位填写共同接口标识。
  control_interface: my_shared_robot_control_v1
  sources:
    - path: /absolute/path/task_a.zarr
      name: task_a
      train_episodes: [0, 1, 2]
      validation_episodes: [3]
    - path: /absolute/path/task_b.zarr
      name: task_b
      train_episodes: [0, 1, 2]
      validation_episodes: [3]
```

```bash
python scripts/train_future_predictor.py --config /path/manifest.yaml \
  --output-dir outputs/future_predictor/shared --device cuda:0
```

两组 episode 必须非空且互不重叠，可以留出额外 episode 不参与训练和验证。
显式划分优先于 `episodes_per_task`，不再对指定的轨迹进行二次抽样。
也可重复传入 `--dataset`，并通过 `data.control_interface=...` 声明兼容接口。
多任务要求动作/本体维度、控制接口、时序约定、编码器和预处理一致；相同维度不代表控制含义一致。
训练按任务平衡采样，避免长任务占据全部批次。验证同时记录总体误差和每个任务的误差。
动作和本体状态的共享范围仅在训练 episode 上拟合，不加载各任务自己的 policy normalizer。

多相机通过 `data.camera_names=[head_camera,left_camera]` 配置，所有相机均需完整 Patch 缓存；输入输出按该列表顺序拼接，去掉每个缓存的 CLS。

## 输出和续训

- `config.yaml`：解析后的训练配置。
- `data_metadata.json`：缓存契约、规范化前的数据接口、完整 episode 划分。
- `metrics.jsonl`：训练、验证 MSE，以及当前帧复制、零动作、时间反转动作、同任务交换动作的对照误差。
- `swanlog/`：SwanLab 本地实验记录，默认同步到云端。
- `swanlab_run.json`：当前实验 ID、项目、workspace、模式；实验标识也保存在 checkpoint 中。
- `tensorboard/`：显式启用可选 TensorBoard 后才写入事件文件。
- `last.pt`：最近完成 epoch 的模型、normalizer、元数据、优化器和随机状态。
- `best.pt`：各任务验证 MSE 均值最低的 checkpoint。
- `epoch_0001.pt`、`epoch_0002.pt` 等：按实际完成的 epoch 编号保存，包含模型、标准化、元数据、优化器和随机状态，可以独立加载用于评估。

当前 YAML 的 `training.save_every_epochs: 2` 表示每 2 个 epoch 保存；改为 1 可每轮保存，最后一个 epoch 始终保存。
`best.pt` 和 `last.pt` 保留，续训仍使用原输出目录的 `last.pt`。

同任务交换动作只统计 batch 中有至少两条同任务样本的部分，样本数记为 `shuffled_samples`。
验证对照是诊断，不会自动证明动作评分准确；尤其应检查背景主导、预测器忽略动作的情况。

续训使用原输出目录的 `last.pt`。入口自动恢复 checkpoint 中的配置，保留相同训练总 epoch 数和其它训练设置，避免新版 YAML 改变旧训练：

```bash
python scripts/train_future_predictor.py --config /path/manifest.yaml \
  --output-dir outputs/future_predictor/shared \
  --resume outputs/future_predictor/shared/last.pt --device cuda:0
```

新训练不会覆盖非空输出目录。`training.max_train_batches` 和 `training.max_val_batches` 可用于短检查，正式训练均应为 `null`。

## SwanLab 训练曲线

`train_pre.sh` 默认使用与 policy 相同的 SwanLab SDK。两个指定环境已安装 `swanlab==0.9.8`。
默认在线记录到 `rmbench_future_predictor` 项目，实验名默认取输出目录名，例如 `shared_20`；
初始化时 SDK 会显示实验链接。若尚未登录，在训练环境中先运行 `swanlab login`。
训练入口禁用交互登录提示，避免在 tmux 中等待输入。

所有曲线的横轴都是累计优化器更新步数；`progress/epoch` 标明对应完成的轮数。

| Tag | 记录内容 |
| --- | --- |
| `train/mse`、`train/learning_rate` | 第一步及每 `swanlab.log_every_steps` 步的 batch MSE、学习率，默认间隔 100 |
| `epoch/train_mse` | 每轮按样本加权的训练 MSE |
| `epoch/validation_mse`、`epoch/validation_macro_mse` | 每轮总体验证 MSE、各任务等权平均验证 MSE |
| `validation/task_mse/<任务名>` | 每个任务的验证 MSE |
| `validation/*_actions_mse`、`validation/copy_current_mse` | 零动作、反转动作、交换动作和复制当前帧的对照误差 |
| `validation/samples`、`validation/shuffled_samples` | 验证样本数和有效交换动作样本数 |

```yaml
swanlab:
  enabled: true
  project: rmbench_future_predictor
  workspace: null
  experiment_name: null  # 默认取输出目录名
  mode: online
  log_every_steps: 100
```

可透传 `swanlab.log_every_steps=10` 调整记录频率、`swanlab.mode=offline` 只写本地，
或 `swanlab.enabled=false` 关闭。日志选项与优化设置独立，可在续训时调整。
正常结束会完成实验，异常/中断会标记对应状态并关闭日志。

在线续训在项目/workspace/模式相同时，用 checkpoint 保存的实验 ID 继续原实验，累计 step 来自 checkpoint。
旧 TensorBoard checkpoint 通过 `--resume` 启动时，只迁移日志默认值到 SwanLab，不改变原数据、模型或优化设置。
离线模式续训新建本地记录分段，沿用累计 step；SwanLab 云端续训不会删除中断时已经上传的尾部 batch 指标，
最近完成的 epoch 应以 checkpoint 和 `metrics.jsonl` 为准。旧 TensorBoard/JSON 历史曲线不自动导入 SwanLab。

TensorBoard 保留为可选后端，默认 `tensorboard.enabled=false`，需要时可显式开启；
其原有累计 step 和中断事件 purge 行为保留。

## 启动和读取耗时

启动脚本显示实际 Python 路径，控制台不输出 `[startup]` 阶段计时提示；`--dry-run` 的 stdout 为 JSON。
每个 epoch 使用一条训练进度条，显示完成 batch 数、百分比、速度、
预计剩余时间和当前 `loss`；`data_s` 为等待下一批数据的时间，`step_s` 为传输与训练用时。
验证阶段也有进度条，显示 batch 进度和当前累积 `mse`。
每轮指标写入 SwanLab 和本地 `metrics.jsonl`，控制台不再打印 `{"epoch": ...}` 整行 JSON。
`max_train_batches`、`max_val_batches` 限制短检查时，进度条的总数按实际执行的 batch 数显示。

```yaml
progress:
  enabled: true
  mininterval: 0.1   # 与 policy 默认值相同，按时间限制刷新，不按步数打印
```

`progress.enabled=false` 可关闭进度条；显示选项可在续训时调整，不影响 SwanLab、优化器或采样。

在 240，`/home` 位于 JuiceFS 共享盘，环境依赖和图像特征的首次读取可能较慢。
现有特征分块为 `[10, 257, 768]`，一次随机取帧仍会读取并解压整块；一个样本的当前/未来两帧一般涉及两块。
句柄复用减少元数据 I/O，不改变分块大小，也不会将全部图像载入内存。
若 `data_wait` 持续占主导，应进一步使用本机磁盘上的特征副本，或离线准备单帧分块缓存。
修改代码和显示配置只影响下一次启动，已运行的训练继续使用启动时加载的代码和配置。

## 加载及动作梯度

```python
import torch

from imitation_learning.models.future_prediction.state_action_future_predictor import (
    load_future_predictor_checkpoint,
)

model, checkpoint = load_future_predictor_checkpoint("outputs/future_predictor/shared/best.pt", device="cuda:0")
# v: [B, cameras * patches, D]；proprio: [B, Dp]，为物理值。
# policy_actions: [B, L, Da]，来自保留计算图的动作采样。
# scale/offset 取自该策略的 action normalizer，使用 Tensor 运算保留梯度。
physical_actions = policy_actions * policy_action_scale + policy_action_offset
prediction = model(v.detach(), proprio.detach(), physical_actions)
loss = (prediction - expert_future.detach()).square().mean()
gradient = torch.autograd.grad(loss, policy_actions)[0]
```

默认加载为 eval、冻结参数，但未禁止输入梯度。调用方不能再套 `torch.no_grad()` 或 `inference_mode()`。
模型内部执行自己的标准化，不要传入策略标准化空间中的动作。
现有 `FixedNormalizer.unnormalize()` 带 `torch.no_grad()`，审计时不能通过它转换动作，应使用上面的 scale/offset 公式。
加载接口支持 `expected_feature_contract` 与 `expected_control_interface` 校验；调用方也应核对执行长度和动作/本体语义。
第一版不使用噪声动作、不训练 RGB 重建器，也不复用现有策略内部未来预测器的权重。
