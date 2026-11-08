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
This applies to spatial averaging, MLP merging and state-query merging,
including cached features and online images. In future `tokens` mode, targets
use the same rectangular average regions; pooled CLS targets stay unchanged.
For example, to use 8 state-query tokens per view:

```yaml
visual_encoder:
  spatial_merge_size: [4, 8]
  patch_merge: false
  state_query_merge: true
```

Existing integer configurations and checkpoint layouts are preserved. Changing
the token grid changes token-dependent parameters, so train a model with the
selected grid instead of editing the config of an existing checkpoint.

Select learnable current/history patch merging in the same task YAML:

```yaml
visual_encoder:
  spatial_merge_size: 8
  patch_merge: true  # false (default): retain spatial averaging
```

With `patch_merge: true`, frozen DINO features keep all 256 patches until
the shared current/history projector. It normalizes each patch, concatenates
each local `8 x 8` group, and applies `49152 -> 512 -> GELU -> 512`, producing
4 tokens per camera at the default width. Cameras are grouped independently.
Cached training reads the existing full CLS+patch caches. Training banks keep
detached, unmerged DINO patches and rebuild their merged latents at each chunk
boundary using the latest parameters. Online inference stores only merged
latents. The frozen future CLS/patch targets retain their existing behavior.
The new merger requires training a new model; disabling the flag, or loading
an old checkpoint without it, retains the original weights and computation.

Alternatively, use state-modulated learned queries to pool each local region:

```yaml
visual_encoder:
  spatial_merge_size: 4
  patch_merge: false
  state_query_merge: true
```

The two learned merging modes are mutually exclusive; both default to false.
With `state_query_merge`, current and history encoders keep all 256 DINO patch
tokens per view until one shared trainable pooling module. Each local spatial
region has one learned query, modulated by the frame's proprioception:
`q_t,g = (1 + gamma(state_t)) * Q_g + beta(state_t)`. The final modulation
layer starts at zero, so initial queries equal `Q_g`. Cross-attention uses only
that region's image patches as keys/values, without adding the query back to
the output. Views are handled independently; `m=4/8` produces `16/4` tokens
per view at the DiT hidden width.

These pooled current-image tokens query history and are also the current-image
conditioning tokens read by every DiT layer. Historical slots use their own
frame's state. Training banks retain detached full patches and normalized raw
proprioception, and re-encode the retained states and re-pool the images with
the latest parameters at every streaming chunk boundary, including Shadows.
Inference stores only the resulting image tokens. Frozen DINO and existing
full-patch caches remain usable; future CLS/patch targets retain their previous
definition. The option is saved in checkpoints and requires training a matching
new model. Missing/false flags preserve the original module and checkpoint
layout.

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

New training uses `workspace.model.denoising_network_partial.state_as_dit_token: true`.
The DiT input keeps `[diffusion step, state, actions]`: with one state token,
16 action tokens and width 512 it has shape `[B, 18, 512]`. Every DiT layer
updates the full sequence; the action head reads only the final 16 tokens.
Current image tokens directly query the historical image slots, without first
reading state. The slot-gate MLP separately receives normalized summaries of
the historical slot, current image and current state (`3*512 -> 128 -> heads`
at the default sizes). State therefore affects the history gates and the DiT
actions without modifying the image query through state attention. The future
predictor consumes the retrieved/fused image tokens, including the gated history.
Current/history fusion, patch merging and Shadow decay retain their behavior.
The state-to-image attention and feed-forward modules are absent in this mode.
Older checkpoints without this setting retain the original state-to-image
route and parameter layout; loading their saved configs still uses that route.

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
