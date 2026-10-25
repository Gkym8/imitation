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
conditioning and memory update path. Future prediction runs only when the
current decision already has history in its memory bank and a valid next-anchor
target. Each stream's first valid observation trains only action prediction,
including streams starting at a nonzero frame. Chunk boundaries retain this
eligibility whenever the carried bank has history. Invalid next-anchor
transitions and padding are excluded; no eligible target skips the predictor
and returns a differentiable zero. Action supervision still uses every valid
anchor, and the first observation is stored for subsequent memory queries.

New training compresses the future predictor's action condition into one token
with `workspace.model.future_feature_predictor.action_condition_mode: chunk`.
Following DynaGuide, a 1x1 Conv1d projects each action, then the ordered chunk is
flattened and linearly projected to the predictor width. At width 512 and chunk
length 10: `[B,10,A] -> [B,51,10] -> [B,1,510] -> [B,1,512]`.
The chunk length follows `execution_chunk_length`, so the condition still uses
only actions before the next visual target. Masked/padded steps contribute zero;
shorter inputs are padded in their original temporal positions. Feature targets
and cosine/MSE losses retain their existing definitions. `tokens` selects the
original per-action projection and position embeddings. Checkpoints without the
new settings retain that original parameter layout and inference behavior.

Patch targets follow `visual_encoder.spatial_merge_size`: with DINOv2 at
224 pixels, the current value `4` gives 16 target tokens per camera; `1`
keeps all 256 spatial patches. Cached training uses the existing CLS+patch
cache for either mode; raw-image training and online inference use the same
encoder settings. The mode is saved in checkpoints. Existing checkpoints
without this setting retain their original pooling and target shapes.

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
An optional `workspace.model.denoising_network_partial.use_learned_history_query`
switch (default `false`) changes only the initial history query. When enabled,
`Q_init` is a learned `[1, query_tokens, hidden_dim]` parameter. Set
`workspace.model.denoising_network_partial.query_tokens` to a positive integer
to choose its token count; the default `null` derives the count from the configured
image encoder, including all cameras. Each query token has `hidden_dim` features
(512 in the RMBench config). Every observation starts from this same parameter and forms
`Q_t = Q_init + CrossAttn(Q_init, C_t, C_t)`, where `C_t` concatenates the
projected current image and state tokens along the token dimension. The resulting
`Q_t` enters the existing history-retrieval blocks; there is no additional
self-attention or FFN in query initialization. The existing fusion gate still
blends the original current visual/state representation with the final retrieval
output. If an explicit query count differs from the image token count, the final
retrieval output is aligned back with
`R = current + CrossAttn(current, retrieved, retrieved)` before the existing gate.
Only this differing-count configuration creates the extra alignment attention;
equal counts and `null` keep the existing path and parameter layout. The final
decision memory keeps the current image token count regardless of `query_tokens`,
including empty-history rows and `fusion_gate: false`. This setting takes effect
only with `use_learned_history_query: true`. Slot-gate scoring, slot storage,
eviction, Shadow decay, and losses keep
their existing inputs and rules. Empty-history rows still use the current
visual/state representation. Initialization and retrieval run once per observation,
with the resulting decision memory reused across Diffusion steps. New query
parameters are created only when enabled; checkpoints without the flag retain
the original path and load strictly. Enabling this option on an existing disabled
checkpoint introduces new trainable parameters and requires training them.

For RMBench, set `workspace.model.dit_three_view: true` to make each DiT
block's direct current-image attention read `head_camera`, `left_camera`,
and `right_camera`, in that order. The default is `false`. Only this DiT
input expands: `workspace.train_dataset.image_keys` continues to select
the observations used by history queries, slot gates, memory writes, and
future-prediction inputs/targets. For example, with `image_keys: [head_camera]`
and four merged patches per view, DiT reads 12 current-image tokens while
memory slots and the final decision memory retain four tokens. The dataset
loads the extra cameras automatically, for both raw images and cached features,
and checkpoint-backed inference prepares all three inputs. Cache `auto`
selection still follows the original `image_keys`; adding DiT views does not
switch a patch cache to CLS. The extra views must exist in the selected cache.
The image backbone and feature projector are shared; DiT has a separate
position embedding for its three-view sequence. Disabled configurations create
no additional encoder parameters and retain the previous checkpoint layout.

History retrieval retains the query residual in every attention block, then
blends the final retrieved representation `R` with the current visual/state
representation `C`:

```text
g = sigmoid(Linear(concat(C, R)))
h = g * C + (1 - g) * R
```

This is the same feature-wise gate formula and initialization used by
MemoryVLA. Its retrieval blocks also retain query residuals before fusion.
Our attention uses multiple heads, RMSNorm/QK normalization, per-slot gates,
and optional learned queries, so the full retrieval implementation is not
identical to MemoryVLA's single-head, LayerNorm blocks. Empty history returns
the current representation rather than zeros. DiT receives `h` as its second
condition; future prediction uses this same `h` (mean-pooled for `pooled`,
kept as tokens for `tokens`) plus the action condition. It does not receive a
second, separate copy of the current observation. The selected history cameras
still determine `h` and future targets; extra direct DiT views remain separate.

The retired pure-history switch and its empty-memory attention masking have
been removed. Saved metadata carrying that switch is ignored for weight-loading
compatibility; checkpoints trained with pure-history retrieval do not retain
their old behavior under the restored fusion path.

Set `workspace.model.denoising_network_partial.fusion_gate: false` to use
`h=R` instead: `R` already contains the query plus history retrieval residuals.
By default that query is the current visual/state representation; with learned
history queries it is `Q_t`, conditioned on current image and state tokens.
The default is `true`, including checkpoints without the new setting.
The gate has one value per token and feature dimension: with width 512,
the linear layer maps 1024 features to 512. Its weights and bias use the
original normal initialization with standard deviation 1e-3. Query residuals
remain inside the retrieval blocks; with no valid history, `R=C`, so the
blend reduces to the current tokens. This fusion gate is distinct from the
per-slot, per-head gate inside history attention. When enabled, both receive
gradients from the action and future-prediction losses.

Fusion parameters are saved and loaded as `perception_gate.proj.weight` and
`perception_gate.proj.bias` under the memory retriever. Checkpoint loading is
strict: checkpoints trained with the fusion layer removed do not contain these
parameters and cannot be directly evaluated with the restored architecture.
Disabling the new flag keeps these parameter names for checkpoint compatibility
but freezes and bypasses the fusion layer. The flag is saved in checkpoint config.

RMBench `train.sh` first looks for `imitation_learning/configs/task/<TASK_NAME>.yaml`
using the task name from its first argument, and falls back to `task/rmbench.yaml`
when that file is absent. Selection adds no startup message. A task-specific
file can inherit the shared RMBench defaults and override only a few settings:

```yaml
defaults:
  - rmbench
  - _self_

workspace:
  model:
    denoising_network_partial:
      max_history_len: 4
      shadow_window: false
```

Streaming training can select the first frame of each stream through
`workspace.train_dataset.sample_range`, an inclusive `[start, end]` range of
episode-frame indices. `[30, 40]` selects 11 streams starting at frames 30..40;
each bank starts empty at its selected frame, without reading frames 0..29.
`[0, 5]` selects six starts, `[0, 0]` selects only frame 0, and `null` (the
default) keeps all original phases, 0..9 with interval 10 and valid start 0.
Explicit ranges can extend beyond the sampling interval. Starts are intersected
with the valid episode range; episodes with no overlap contribute no streams,
and an empty dataset raises an error. Each selected stream still runs to the
end at the unchanged interval; padding, bank updates, and
future-target alignment keep their original rules. Repetition duplicates the
selected streams, and shuffling changes only their training order.

For example, add this under `workspace` in a task YAML:

```yaml
workspace:
  train_dataset:
    sample_range: [30, 40]
```

This differs from `episode_starting_idx_max`, which also limits the end of a
stream in the existing implementation. `sample_range` only controls stream starts
and is supported by streaming sampling. It does not affect evaluation or model
parameter shapes; older configurations without either setting keep all phases.
For saved training-config compatibility, the retired `sample_nums: N` maps to
`sample_range: [0, N]`; specifying both non-null settings is rejected.

Training and inference use the same chronological memory bank. With
`shadow_window: true` (the default), slots have `active` and `shadow` states.

`workspace.model.denoising_network_partial.history_time_encoding` selects how
history keys receive time positions. `relative` (default) keeps the existing
trainable, sinusoidally initialized table, its Bank-rank ordering, and left-padding
alignment unchanged. `absolute` computes sinusoidal encodings directly
from each retained slot's episode frame index. For retained frames `[0, 30, 80]`,
it encodes `[0, 30, 80]` regardless of eviction or the slot's position in the Bank.
It uses frame indices directly, without dividing by the anchor interval or
subtracting the stream's starting frame. Training uses `traj_idx`; online inference
starts at frame 0 and advances by `execution_chunk_length` per valid observation.
Active and Shadow slots use the same frame convention across chunk boundaries.
Only history keys receive the encoding; stored slot values and slot-gate inputs
remain unchanged. Absolute encodings are generated on demand, so frame numbers
can exceed Bank capacity. The unused rank table stays in absolute-mode checkpoint
state for shape compatibility but is frozen. Older configs without the setting
keep `relative`; the selected mode is saved with the checkpoint and restored
during inference. Direct non-streaming network calls in absolute mode must supply
`history_frame_indices` for their history slots/source anchors.
With `shadow_window: false`, both training and inference hard-delete evictions
and retain only Active capacity. After reading history, a full Active bank selects the
eligible Active slot with the lowest mean raw gate over heads, then inserts
the current observation as Active. Ties demote the oldest eligible slot.
The original first slot and the newest two Active slots after insertion are
protected by default. `keep_recent` includes the incoming slot, so before
insertion only the newest `max(keep_recent - 1, 0)` existing Active slots are
protected. For a full bank containing slots 0 through 7 and an incoming slot 8,
`keep_recent: 2` protects 7 and 8; slot 6 is an eviction candidate. The incoming
slot is written only after the current read, in both training and inference.
Protection must leave at least one existing eviction candidate.

```yaml
workspace:
  model:
    denoising_network_partial:
      max_history_len: 8       # Active capacity
      shadow_window: true     # Enable Shadow in both training and inference
      shadow_horizon: 3       # Number of future valid Shadow reads
      shadow_gate_scale: 0.01  # First Shadow read, then divide by 10 per read
      fusion_gate: true       # False uses current + retrieved history residuals
    memory_eviction:
      keep_first: true
      keep_recent: 2
    gate_selectivity_min_std: 0.1
  trainer:
    gate_selectivity_loss_weight: 0.01
```

Shadow slots share the same MLP and Attention normalization with Active slots.
Their effective log-gates are
`logsigmoid(MLP) + log(shadow_gate_scale) - age * log(10)`, with age starting at 0;
raw gates are recomputed at every read. A slot demoted after anchor `t` is
read as Shadow at `t+1` through `t+shadow_horizon`, then physically deleted
after the final read. It never re-enters the eviction candidates or consumes
an Active place. Total feature storage and history position embeddings are
bounded by `max_history_len + shadow_horizon`. With horizon 3 and scale 0.01,
the next three reads multiply the newly computed raw gate by 0.01, 0.001,
and 0.0001. The scale is not accumulated on the previous read's gate.
When disabled, banks and packed history contain only `max_history_len` slots
in both training and inference. The configured `shadow_horizon` is retained for
checkpoint positional-table compatibility, and reads use the final
`max_history_len` positions. When enabled, both use Active plus Shadow capacity.
Counters advance only on valid supervised/online anchors, excluding padding
and lookahead. State and expiry survive chunk boundaries along with features.

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
contains detached raw `[batch, storage_capacity, head]` scores
for the latest read, with zero for padded slots. `latest_effective_slot_gates`
also includes the Shadow scaling. Adding `history_slot_gates` and/or
`history_effective_slot_gates` to `record_data_entries` records these values.
RMBench evaluation writes one row per historical slot read to
`memory_slot_gates.csv`, with only `episode_index`, `query_anchor_index`,
`slot_id`, `slot_state`, `gate_mean`, `gate_head_0` through `gate_head_{H-1}`,
and `attention_mean`. Gates are raw MLP sigmoid outputs; `gate_mean` averages
the heads, matching the eviction score. Attention is the actual post-softmax
slot probability (summed over its patches), averaged over heads and current
query tokens in the final retrieval layer. Online CSV rows include Shadow slots
when the window is enabled; otherwise they contain only Active slots;
padding does not. An empty bank produces one row with blank slot values.
Episodes append to the same file without repeating its header.

Checkpoints save the Shadow configuration and matching positional embedding
shape. `shadow_window` is saved in checkpoint config and controls both training
and inference. Checkpoints without this flag default to enabled; a zero horizon
still means immediate eviction. Toggling the flag does not change parameter shapes.
Configs without `shadow_horizon` retain immediate eviction and their
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


### Mikasa streaming-memory training

The same policy code supports RMBench and Mikasa. RMBench keeps using `train.sh`.
For Mikasa, activate the policy environment and run from this directory:

```bash
bash train_mikasa.sh mikasa_remember_color_3 0 my_run
# Multiple GPUs:
bash train_mikasa.sh mikasa_remember_color_3 0,1 my_run
# Explicit dataset root (contains RememberColor3-v0/episode_data.zarr):
bash train_mikasa.sh mikasa_remember_color_3 0 my_run --data-root /path/to/mikasa-zarr
```

Data preparation can also be run explicitly before training, in this order.
Use the already converted `--max-episodes 250 --abs-joint-pos` Mikasa data:

```bash
python fit_mikasa_normalizer.py \
  /path/to/mikasa-zarr-250/ShellGameTouch-v0 \
  mikasa_shell_game_touch

python scripts/extract_mikasa_dinov2_features.py \
  /path/to/mikasa-zarr-250/ShellGameTouch-v0/episode_data.zarr \
  --batch-size 128 --feature-chunk-length 10 --device cuda:0

bash train_mikasa.sh mikasa_shell_game_touch 0 my_run
```

The normalizer command accepts either the task directory or `episode_data.zarr`;
it does not require an image-feature cache. It reuses existing action statistics
and backs up an old file before adding missing state statistics. The feature
command always selects `third_person_camera` and stores CLS+patch features.
Training reuses the completed normalizer and feature cache.

`MIKASA_DATA_DIR` can also supply the data root. When neither is set, the launcher
looks in this policy's `data/mikasa`, RMBench's `data/mikasa`, then the sibling
`gated-memory-policy/data/mikasa-zarr-250` directory. It reads existing converted
Mikasa Zarr stores; NPZ conversion is still handled by the Mikasa project.
`POLICY_PYTHON` can select the policy Python executable.

The first run prepares missing frozen DINOv2 CLS+patch features and normalizer
fields, then starts `scripts/train_policy.py`. Existing action statistics are
reused exactly. Old action-only normalizers are backed up before adding the
missing `robot0_8d` state statistics; complete normalizers are left untouched.
The active file is `<task>_normalizer.json`; `.backup.<timestamp>` files preserve
the previous contents and are not loaded for training. Legacy image augmentation
metadata may remain in the JSON, but the loader reads only normalization values
from it. Transforms use the current YAML, which specifies only deterministic
Resize; neither cached-feature training nor raw-image training uses the old
random crop or color jitter settings.
Preparation is done once before DDP starts, and complete compatible caches are
reused. Only the fixed
third-person camera (`third_person_camera`) is selected; the wrist view is not
loaded, encoded, or cached. Input images are 128x128, resized to 224x224 by the same
deterministic transform in offline extraction and online inference. The default
spatial merge produces four patch tokens per slot, each 768-dimensional before
projection. `robot0_8d` provides the state and `action0_8d` the eight action channels.
Data converted with `--abs-joint-pos` already contains absolute joint targets.
The loader only normalizes those values; it does not convert them again. Use
`abs_joint_pos: true` in the Mikasa evaluator (the current default).

`configs/task/mikasa.yaml` (under `imitation_learning/`) inherits the current
`rmbench.yaml` method. Mikasa deliberately keeps a 10-step prediction horizon,
51 epochs, checkpointing every 5 epochs and validation every 2 epochs; camera,
state/action field names and their eight-channel shapes follow the Mikasa data.
Other method settings, including history capacity, gates, shadow memory, visual
encoder and loss weights, use the same shared defaults as RMBench. The old
45-slot overrides in ShellGame and Rotate task YAMLs have been removed.
`mikasa_remember_color_3.yaml` inherits the common configuration and other Mikasa
task YAMLs inherit that task; explicit task overrides still apply. For example, put the following override in the desired task YAML:

```yaml
workspace:
  train_dataset:
    sample_range: [0, 0] # Only streams beginning at original frame 0.
  model:
    future_prediction_mode: pooled # Or tokens; same losses as RMBench.
    denoising_network_partial:
      max_history_len: 4
      shadow_window: false
      fusion_gate: true
```

`sample_range: null` keeps all original phases. Streaming continues to the episode
end, carries the same gate memory bank between chunks, and truncates gradients
between chunks. The default sampling interval and `execution_chunk_length` are
both 10. If changing the execution interval, make the Mikasa evaluation server
execute the same number of actions per policy call. Pooled mode predicts one
future CLS token for the selected camera, while DiT retains all four context
tokens. Future prediction is a training objective and is not executed online.

Runtime overrides are also available, for example
`--override 'workspace.train_dataset.sample_range=[0,0]'` or
`--override workspace.model.denoising_network_partial.shadow_window=false`.
`--dry-run` prints the launch command without changing data or starting training;
`--skip-prepare` requires a compatible cache and normalizer to exist already.
Raw-image training is available with `use_cached_image_features: false`.
When `visual_encoder.frozen: true`, current images, historical images and future
targets share one DINOv2 backbone in both cached-feature and raw-image training.
Only `visual_encoder.frozen: false` creates a separate frozen
`future_target_dinov2` for future supervision. Frozen raw-image runs therefore
avoid saving and loading an extra approximately 330 MiB of backbone weights.
Existing checkpoints retain their saved encoder registration settings and
continue to load strictly, including older checkpoints with two frozen backbones;
their files are not converted automatically.
The older `shell_scripts/train_mikasa.sh` path forwards to the same entry point.
