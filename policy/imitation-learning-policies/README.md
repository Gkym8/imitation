# Environment Setup

## Prerequisites

- A **Linux** machine with NVIDIA GPU(s) — Windows and macOS will run into package compatibility issues
- A compatible NVIDIA driver installed on the host (the CUDA toolkit itself is managed by conda)
- [Miniforge3](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install) or any conda distribution

## Installation

```bash
conda env create -f env.yaml
conda activate imitation
```

> **Note:** For the latest GPU (e.g. RTX 5090), you may need to install a nightly build of PyTorch.
```bash
pip install --pre torch torchvision torchaudio torchcodec --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
```

## Download Checkpoints and Datasets

Install the Hugging Face CLI:
```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

If you have installed `huggingface_hub` in your current python environment and there are errors running the following download commands, please either deactivate the current environment or update the package through `pip install -U huggingface_hub`. Please do not update the package in the `imitation` conda environment, as it is incompatible with other packages (e.g. `transformers`).

```bash
conda deactivate
mkdir -p data/checkpoints

# Download all checkpoints
hf download yihuai-gao/gated-memory-policy --type model --local-dir ./data/checkpoints # 48.8GB

# Or download checkpoints for specific environments
hf download yihuai-gao/gated-memory-policy --type model --include "memmimic/**" --local-dir ./data/checkpoints # 19.1GB
hf download yihuai-gao/gated-memory-policy --type model --include "robomimic/**" --local-dir ./data/checkpoints # 26.5GB
hf download yihuai-gao/gated-memory-policy --type model --include "mikasa/**" --local-dir ./data/checkpoints # 
hf download yihuai-gao/gated-memory-policy --type model --include "real/**" --local-dir ./data/checkpoints # 3.2GB

# Or download specific models / tasks 
hf download yihuai-gao/gated-memory-policy --type model --include "memmimic/*gated*" --local-dir ./data/checkpoints # 4.4 GB
hf download yihuai-gao/gated-memory-policy --type model --include "memmimic/*push_cube*" --local-dir ./data/checkpoints
hf download yihuai-gao/gated-memory-policy --type model --include "memmimic/*push_cube*gated*" --local-dir ./data/checkpoints
```

```bash
mkdir -p data/datasets

# Download all datasets
hf download yihuai-gao/gated-memory-policy --type dataset --local-dir ./data/datasets # 325GB 

# Or download specific datasets
hf download yihuai-gao/gated-memory-policy --type dataset --include "memmimic/**" --local-dir ./data/datasets # 141GB
hf download yihuai-gao/gated-memory-policy --type dataset --include "robomimic/**" --local-dir ./data/datasets # 184GB
hf download yihuai-gao/gated-memory-policy --type dataset --include "iphumi/**" --local-dir ./data/datasets # 44.8GB
hf download yihuai-gao/gated-memory-policy --type dataset --include "real_world/**" --local-dir ./data/datasets # 12.6GB
```

## Serve a Checkpoint

```bash
shell_scripts/serve_policy_ckpt.sh <path/to/checkpoint.ckpt>
```

For example:

```bash
shell_scripts/serve_policy_ckpt.sh data/checkpoints/memmimic/push_cube_diffusion_gated.ckpt # Use GPU 0 (default)
shell_scripts/serve_policy_ckpt.sh data/checkpoints/memmimic/push_cube_diffusion_gated.ckpt 1 # Use GPU 1
```

## Rollout all Simulation Checkpoints

Please ensure all the simulation environments are working in [mujoco-env](../mujoco-env/README.md).

The following script will use all the GPUs to rollout all the RoboMimic / MemMimic simulation checkpoints. It will spawn 1 process to send commands to send checkpoint paths and gather results, N (number of GPUs) processes to run policy servers, and N processes to host simulation environments. The entire evaluation process (100 episodes per checkpoint) takes about 2hrs on a 8-GPU server. You can specify the filter string to rollout only specific checkpoints.

```bash
shell_scripts/rollout_policies.sh <checkpoints_dir> <filter_str>
```

## Train a Policy

### Config System

This repository uses the [Hydra](https://hydra.cc/) config system. We set multiple levels of configurations for different components, avoiding repeated configuration across different policies and tasks. Each checkpoint will keep the entire config file in the checkpoint directory, so you can easily reproduce the training process by simply loading the checkpoint and instantiating the config.

For regular training, the config overriding order is (from lowest to highest):

- `base_config.yaml`: defaults to `base_workspace.yaml`
- `workspace/policy/{policy_name}.yaml`: will be specified based on the policy name passed from the command line.
- `workspace/dataset/{dataset_name}.yaml`: dataset name is determined by both policy (using memory or not: single-traj or multi-traj) and task (which specific dataset)
- `task/{task_name}.yaml`: will apply task-specific overrides, for example, different keys
- `train_policy.yaml` and **shell script additional arguments**: please use `+workspace.xxx=xxx` to append an additional override to `train_policy.yaml`. This config will be applied last to ensure the highest priority.

### Simulation

RMBench training with `train.sh` first looks for
`imitation_learning/configs/task/<TASK_NAME>.yaml`. Settings in that file override
`task/rmbench.yaml`; every setting omitted by the task file keeps the RMBench
default. If the task file does not exist, all settings come from `rmbench.yaml`.
The task file can contain only the differences. Like the existing Mikasa task
files, it can declare inheritance with `defaults`; the RMBench launcher also
supplies these defaults when that block is omitted. For example,
`task/rearrange_blocks.yaml`:

```yaml
defaults:
  - rmbench
  - _self_

visual_encoder:
  spatial_merge_size: [4, 8]

workspace:
  model:
    denoising_network_partial:
      shadow_gate_scale: 0.1
  trainer:
    num_epochs: 401
```

Launch commands stay the same. The second `train.sh` argument (for example,
`demo_clean`) still selects the demonstration dataset configuration. Command-line
overrides retain the highest priority. Training prints the selected task YAML
and saves the merged settings and YAML name in the checkpoint. Missing files
fall back; syntax errors or broken defaults in an existing file are reported.
Explicit `+task_yaml_name=...` in direct Python launches still selects that file.

For RMBench, select future prediction in
`imitation_learning/configs/task/rmbench.yaml`:

```yaml
workspace:
  model:
    future_prediction_mode: pooled  # pooled (default) or tokens
```

`pooled` mean-pools the retrieved decision memory `h` to one vector and
predicts the next anchor's CLS token (DINOv2) or MAP token (SigLIP), one per
camera. `tokens` passes the complete `h` token sequence to the predictor and
predicts the next anchor's spatial patch features, without pooling `h`.
The target encoder and predictor output token count switch together. `tokens`
uses MSE on raw predicted/target features, without L2 normalization, averaged
over valid target tokens and feature dimensions. `pooled` and legacy configs
without a mode retain cosine-distance loss. Both modes use the same action
conditioning and memory update path. Invalid next-anchor transitions and
padding are excluded; no valid target returns a differentiable zero.

Patch targets follow `visual_encoder.spatial_merge_size`: with DINOv2 at
224 pixels, `8` gives 4 target tokens per camera; `1`
keeps all 256 spatial patches. Cached training uses the existing CLS+patch
cache for either mode; raw-image training and online inference use the same
encoder settings. The mode is saved in checkpoints. Existing checkpoints
without this setting retain their original pooling and target shapes.

`spatial_merge_size` accepts an integer (square regions, as before) or
`[height, width]` in DINO patch units. For a `16 x 16` patch grid,
`[4, 8]` produces a `4 x 2` output grid and `[8, 4]` produces a `2 x 4`
output grid: both give 8 tokens per view. Both dimensions must divide 16.
This applies to both local mean pooling and local learned-query pooling,
including cached features and online images. Future patch targets use the
same rectangular average regions; pooled CLS targets stay unchanged.

```yaml
visual_encoder:
  spatial_merge_size: [4, 8]
  learned_query_merge: true
```

`learned_query_merge` is strictly boolean. With `true`, every spatial region
has one learned query that reads only its own patches. For a 16 x 16 patch
grid, size 8 produces a 2 x 2 output grid (4 tokens); [4, 8] produces a 4 x 2
output grid (8 tokens). Each spatial position has its own query parameters,
shared across cameras and time; cameras never attend to each other.
After the image LayerNorm/linear adapter, each region uses SigLIP-style MAP:
`h = MultiheadAttention(Q_region, X_region, X_region)` followed by
`h + MLP(LayerNorm(h))`. There is no query residual or robot-state fusion
inside this module. The MLP expands the width fourfold and uses tanh GELU.
Current and history images share this pooling head. Streaming training
retains detached patches and re-pools them at chunk boundaries; online banks
store the resulting image tokens.

`visual_encoder.cls_token: true` appends each image's DINO CLS token after
its local tokens, for both mean pooling and learned-query MAP. The default is
false. At merge size 4, each view has 16 local tokens plus one CLS token.
CLS is excluded from local pooling/attention and uses the image-width adapter;
current/history images and cached/raw paths share the same per-view ordering
`[local tokens, CLS]`. Future prediction targets retain their existing shapes.

With `false`, each region uses uniform mean pooling in the image encoder.
`spatial_merge_size` controls the same regional output grid in both modes.
The removed `patch_merge` and `state_query_merge` options are no longer used.
Integer query-count configurations and global/state-fused MAP checkpoints
must be retrained with the new boolean configuration and regional token grid.
Future target encoders remain independent of the learned pooling head.

Current images, history images, and future supervision share one frozen DINO
backbone in both cached and online modes. Online DINO forwards follow the
caller's autocast: BF16 mixed precision during training and FP32 during the
default RMBench evaluation. It stays frozen and in eval mode during policy
training. Frozen parameters are copied exactly into the EMA model rather
than averaged. The patch merger, DiT, and future predictor retain the
trainer's mixed precision and their gradients. Existing
FP32 feature caches remain usable. Legacy checkpoints containing a second
`future_target_dinov2` backbone are folded into the shared model when their
weights match; differing legacy target weights are not silently discarded.

RMBench also enables an observation-conditioned, multi-head slot gate in
`workspace.model.denoising_network_partial`:

```yaml
use_conditioned_slot_gate: true
slot_gate_hidden_dim: 128
slot_gate_bias_init: 0.0
```

The gate pools each projected historical slot and the current visual/state
tokens separately, normalizes both summaries, and feeds their concatenation
through a shared `Linear -> GELU -> Linear` MLP. For width 512 and 8 attention
heads this is `1024 -> 128 -> 8`. The attention logits receive `logsigmoid`
of the MLP outputs, broadcast over each slot's patches. The complete visual
tokens remain available to attention and to `future_prediction_mode: tokens`.
One gate evaluation is reused across retrieval layers and Diffusion steps;
each new observation recomputes scores against the retained slots.

For `perception_patches`, the final `TokenGateFusion` blends current visual/state
tokens `C` with the history-aware output `R` of the last retrieval block:
`g = sigmoid(Linear(concat(C, R)))`, `h = g*C + (1-g)*R`.
The gate has one value per token and feature dimension: with width 512,
the linear layer maps 1024 features to 512. Its weights and bias use the
original normal initialization with standard deviation 1e-3. Query residuals
remain inside the retrieval blocks; with no valid history, `R=C`, so the
blend reduces to the current tokens. This fusion gate is distinct from the
per-slot, per-head gate inside history attention, and both receive gradients
from the action and future-prediction losses.

RMBench task defaults use
`workspace.model.denoising_network_partial.state_as_dit_token: false`.
The DiT input is `[diffusion step, actions]`: with 16 action tokens and width
512 it has shape `[B, 17, 512]` in every layer. The action head still reads
only the final 16 tokens. Numeric learned-query MAP already reads each image's
own state before history retrieval, so this route does not add the older
state-to-image attention/FFN again. State encoding and the slot-gate MLP's
separate current-state summary remain active (`3*512 -> 128 -> heads`).
The state-position entry is retained in the position table for this unchanged
Gate input, then removed from the main sequence before any DiT block.
Current/history image token counts, their state reads, fusion and Shadow
behavior are unchanged by removing the independent DiT state token.

The existing `state_as_dit_token: true` setting retains the 18-token
`[diffusion step, state, actions]` route, including for saved numeric MAP
checkpoints. Non-numeric pooling modes with this setting absent/false retain
their original state-to-image route and parameter layout. Saved checkpoint
configuration determines which route is used; editing the task defaults does
not alter the token count of a checkpoint loaded with its saved configuration.

Fusion parameters are saved and loaded as `perception_gate.proj.weight` and
`perception_gate.proj.bias` under the memory retriever. Checkpoint loading is
strict: checkpoints trained with the fusion layer removed do not contain these
parameters and cannot be directly evaluated with the restored architecture.
Train a matching model to evaluate the restored fusion gate.

Training and inference share one chronological memory bank with `active` and
`shadow` slot states. After reading history, a full Active bank demotes the
eligible Active slot with the lowest mean raw gate over heads, then inserts
the current observation as Active. Ties demote the oldest eligible slot.
The original first slot and the newest two existing Active slots are protected
by default. `keep_recent` counts only history already retained before insertion;
the incoming slot always enters afterwards and is not included in that count.
For a full bank containing slots 0 through 7 and an incoming slot 8,
`keep_recent: 2` protects 6 and 7 (plus slot 0 when `keep_first: true`). Slot 8
is written only after the current read and eviction decision, in both training
and inference. On the following insertion, the newest existing Active slots
are 7 and 8. Shadow slots do not count toward recent Active protection.
Protection must leave at least one existing eviction candidate:
`int(keep_first) + keep_recent < max_history_len`.

```yaml
workspace:
  model:
    denoising_network_partial:
      max_history_len: 8       # Active capacity
      shadow_horizon: 4       # Future valid anchors; 0 deletes immediately
      shadow_gate_scale: 0.5  # Per-read decay rate; effective gate = raw gate * rate**k
    memory_eviction:
      keep_first: true
      keep_recent: 2
    gate_selectivity_min_std: 0.1
  trainer:
    gate_selectivity_loss_weight: 0.01
```

Shadow slots share the same MLP and Attention normalization with Active slots.
Their effective log-gates are `logsigmoid(MLP) + k * log(shadow_gate_scale)`,
where `k=1..shadow_horizon` counts valid reads since that slot was demoted.
Raw gates are recomputed at every read, then multiplied by the rate to the
power `k`; previously decayed gates are not reused. With rate `0.5` and horizon
4 the multipliers are `0.5, 0.25, 0.125, 0.0625`; rate `0.3` gives
`0.3, 0.09, 0.027, 0.0081`. A slot demoted after anchor `t` is
read as Shadow at `t+1` through `t+shadow_horizon`, then physically deleted
after the final read. It never re-enters the eviction candidates or consumes
an Active place. Total feature storage and history position embeddings are
bounded by `max_history_len + shadow_horizon`.
Counters advance only on valid supervised/online anchors, excluding padding
and lookahead. State, expiry, and decay progress survive chunk boundaries
along with features. Training and inference use the same per-slot clock.

The selectivity loss computes population standard deviation over valid Active
slots' raw gates for each head, then averages `relu(0.1 - std)**2` over heads.
Shadow gates are excluded. Only full Active banks
receive this penalty; unfilled banks contribute zero to the average over valid
training anchors. The default total loss is
`action + 0.05 * future_prediction + 0.01 * gate_selectivity`.
Both task losses and this regularizer train the gate. Chunk boundaries detach
stored features; the next chunk reprojects them using current parameters.
Future lookahead anchors provide targets without entering the memory bank.
Online inference only reads gates and predicts actions, without running the
future predictor or future target encoder. The obsolete prediction-error queue,
manually updated slot weights, and scalar attention-bias settings are removed.

For inspection, `policy.denoising_network.memory_retriever.latest_slot_gates`
contains detached raw `[batch, Active_capacity + shadow_horizon, head]` scores
for the latest read, with zero for padded slots. `latest_effective_slot_gates`
also includes the Shadow scaling. Adding `history_slot_gates` and/or
`history_effective_slot_gates` to `record_data_entries` records these values.
RMBench evaluation writes one row per historical slot read to
`memory_slot_gates.csv`, with only `episode_index`, `query_anchor_index`,
`slot_id`, `slot_state`, `gate_mean`, `gate_head_0` through `gate_head_{H-1}`,
and `attention_mean`. Gates are raw MLP sigmoid outputs; `gate_mean` averages
the heads, matching the eviction score. Attention is the actual post-softmax
slot probability (summed over its patches), averaged over heads and current
query tokens in the final retrieval layer. Active and Shadow slots both appear;
padding does not. An empty bank produces one row with blank slot values.
Episodes append to the same file without repeating its header.

Checkpoints save the Shadow configuration and matching positional embedding
shape. Configs without `shadow_horizon` retain immediate eviction and their
original shape. Enabling a nonzero horizon changes that shape and requires
training a matching model. Old checkpoint configs containing removed memory
settings must be updated before loading this policy version.

Please go through all comments in `shell_scripts/train_sim.sh` for detailed instructions.

By default it runs on all GPUs. To use a different single GPU or multiple GPUs, pass the GPU IDs as a comma-separated argument:

```bash
shell_scripts/train_sim.sh 1      # single GPU (GPU 1)
shell_scripts/train_sim.sh 0,1    # multi-GPU
```

### Real-world

Please go through all comments in `shell_scripts/train_real.sh` for detailed instructions.

By default it runs on all GPUs. To use a different single GPU or multiple GPUs, pass the GPU IDs as a comma-separated argument:

```bash
shell_scripts/train_real.sh 1      # single GPU (GPU 1)
shell_scripts/train_real.sh 0,1    # multi-GPU
```

## Misc.

- The rotation 6d is following the UMI convention, which uses the first 2 **rows** of the rotation matrix rather than the first 2 columns proposed in [this original paper](https://arxiv.org/abs/1812.07035).


## Citation

This repository is developed during the following [research project](https://github.com/real-stanford/gated-memory-policy). If you find it useful, please cite our corresponding paper:
```latex
@misc{gao2026gatedmemorypolicy,
  title         = {Gated Memory Policy: In-Context Memorization and Adaptation},
  author        = {Yihuai Gao and Jeff Jinyun Liu and Shuang Li and Shuran Song},
  year          = {2026},
  eprint        = {2604.18933},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2604.18933},
}
```
