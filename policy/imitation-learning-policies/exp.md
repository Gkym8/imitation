  ### 1. 创建两个环境

  在仓库根目录执行：

  conda env create -f imitation-learning-policies/env.yaml
  conda env create -f mikasa-robo-env/env.yml

  conda activate mikasa
  cd mikasa-robo-env
  pip install -e . --no-deps
  cd ..

  imitation 用于训练，mikasa 用于数据转换和仿真评测。

  ### 2. NPZ 转成训练需要的 Zarr

  训练代码不能直接读取当前 NPZ。执行：

  mkdir -p data/mikasa-zarr

  conda run -n mikasa python \
    mikasa-robo-env/tools/convert_npz_to_zarr.py \
    --input-dir data/mikasa \
    --output-dir data/mikasa-zarr \
    --tasks RememberColor3-v0 \
    --abs-joint-pos

  这里必须使用 --abs-joint-pos，因为仓库说明默认模型使用绝对关节目标训练，评测时也需要匹配 --abs-joint-pos。

  生成结果应为：

  data/mikasa-zarr/
  └── RememberColor3-v0/
      └── episode_data.zarr/

  原始图像保持 128×128，训练数据增强会自动放大到 256×256，不需要提前 resize。

  转换脚本见 mikasa-robo-env/tools/convert_npz_to_zarr.py。

  ### 3. 拟合动作 normalizer

  conda activate imitation
  cd imitation-learning-policies
  export PYTHONPATH="$PWD:${PYTHONPATH:-}"

  python scripts/fit_dataset_normalizer.py \
    ../data/mikasa-zarr/RememberColor3-v0 \
    mikasa_remember_color_3 \
    --quantile 1.0

  预期生成：

  data/mikasa-zarr/RememberColor3-v0_normalizer.json

  没有这个文件，训练入口会直接报 Normalizer is not found。



  cd /home/gongming/Code/gated-memory-policy
  mkdir -p data/mikasa-zarr-250

  conda run -n mikasa python \
    mikasa-robo-env/tools/convert_npz_to_zarr.py \
    --input-dir data/mikasa \
    --output-dir data/mikasa-zarr-250 \
    --tasks ShellGameTouch-v0 \
    --max-episodes 250 \
    --abs-joint-pos

  然后只在这个数据集上拟合 normalizer：

  conda activate imitation
  cd imitation-learning-policies
  export PYTHONPATH="$PWD:${PYTHONPATH:-}"

  python scripts/fit_dataset_normalizer.py \
    ../data/mikasa-zarr-250/RememberColor9-v0 \
    mikasa_remember_color_9 \
    --quantile 1.0

  最终得到：

  data/mikasa-zarr-250/RememberColor3-v0_normalizer.json

  训练时将三个路径都指向这个目录：

  +workspace.train_dataset.root_dir=../data/mikasa-zarr-250 \
  +workspace.train_dataset.compressed_dir=../data/mikasa-zarr-250 \
  +workspace.train_dataset.normalizer_dir=../data/mikasa-zarr-250


  export PYTHONPATH="$PWD:${PYTHONPATH:-}"
  export CUDA_VISIBLE_DEVICES=0
  python scripts/train_policy.py \
    +policy_name=diffusion_memory_transformer \
    +task_name=mikasa_remember_color_3 \
    +project_name=mikasa \
    +logger_project_name=mikasa_shell_game_touch \
    +run_name=remember_color3_gmp \
    +train_server_name=localhost \
    +workspace.train_dataset.root_dir=../data/mikasa-zarr-250 \
    +workspace.train_dataset.compressed_dir=../data/mikasa-zarr-250 \
    +workspace.train_dataset.normalizer_dir=../data/mikasa-zarr-250 \
    +workspace.train_dataset.starting_percentile_max=1.0

accelerate launch \
    --gpu_ids 0,1,2,3 \
    --num_processes 4 \
    --main_process_port 29503 \
    scripts/train_policy.py \
    +policy_name=diffusion_memory_transformer \
    +task_name=mikasa_remember_color_9 \
    +project_name=mikasa \
    +logger_project_name=mikasa \
    +run_name=remember_color_9 \
    +train_server_name=localhost \
    +workspace.train_dataset.root_dir=../data/mikasa-zarr-250 \
    +workspace.train_dataset.compressed_dir=../data/mikasa-zarr-250 \
    +workspace.train_dataset.normalizer_dir=../data/mikasa-zarr-250 \
    +workspace.model.denoising_network_partial.record_history_attention_epoch_stats=True

accelerate launch \
    --gpu_ids 4,5 \
    --num_processes 2 \
    --main_process_port 29501 \
    scripts/train_diffusion_policy_resume.py \
    --ckpt_path data/mikasa_remember_color_3/2026-08-30/22-31-19_remember_color_3/checkpoints/workspace.ckpt \
    --record_history_attention \
    --train_batch_size 1
    
cd mikasa-robo-env

  python eval/mikasa_eval.py \
      --env-id RememberColor3-v0 \
      --checkpoint /home/gongming/Code/RMBench/policy/imitation-learning-policies/data/mikasa_remember_color_3/2026-10-22/15-08-37_6h/checkpoints/epoch_50_train_mean_loss_0_004.ckpt
      --num-envs 10 \
      --num-eval-episodes 100 \
      --abs-joint-pos \
      --seed 42 \
      --output-dir /mnt/conda/gongming/gate-memory-policy/eval_results/remember_color9_epoch50_seed42
     

修改了base_workspace.yaml，train_policy.yaml输出位置



RMBench：
cd /home/gongming/Code/RMBench/policy/imitation-learning-policies
python process_data.py swap_blocks demo_clean 50
python fit_rmbench_normalizer.py /home/gongming/Code/RMBench/data/imitation_learning_policies/swap_blocks-demo_clean-50.zarr 14

python scripts/extract_rmbench_dinov2_features.py \
../../data/imitation_learning_policies/swap_blocks-demo_clean-50.zarr \
--cameras head_camera \
--feature-kind patches \
--batch-size 128 \
--feature-chunk-length 10 \
--device cuda:4

238加上：
cuda_lib="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
LD_PRELOAD="$cuda_lib/nccl/lib/libnccl.so.2:$cuda_lib/cudnn/lib/libcudnn.so.9" \
LD_LIBRARY_PATH="$cuda_lib/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
bash train.sh observe_and_pickup demo_clean 50 0 14 0,1,2,3 8h3obs

deploy_policy.yml评测输出位置

export CUDA_HOME=/usr/local/cuda-12.6
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11
export TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2
export TORCH_EXTENSIONS_DIR=/tmp/rmbench237-curobo-torch241-cu126

cuda_lib="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
LD_PRELOAD="$cuda_lib/nccl/lib/libnccl.so.2:$cuda_lib/cudnn/lib/libcudnn.so.9" \
LD_LIBRARY_PATH="$cuda_lib/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
bash eval.sh \
    observe_and_pickup \
    demo_clean \
    epoch300_8h3obs \
    50 \
    0 \
    4 \
    data/observe_and_pickup/2026-10-23/11-55-50_observe_and_pickup-demo_clean-50-seed0_8h3obs/checkpoints/epoch_300_train_mean_loss_0_000.ckpt

Mikasa：

export MIKASA_DATA_DIR=/home/gongming/Code/gated-memory-policy/data/mikasa-zarr-250

python fit_mikasa_normalizer.py \
  "$MIKASA_DATA_DIR/RememberColor9-v0" \
  mikasa_remember_color_9
  
python scripts/extract_mikasa_dinov2_features.py \
  "$MIKASA_DATA_DIR/ShellGameTouch-v0/episode_data.zarr" \
  --batch-size 128 \
  --feature-chunk-length 10 \
  --device cuda:4
  
bash train_mikasa.sh mikasa_remember_color_3 0,1,2,3 remember_color_3_5h_wofirstpre_2obs

  python eval/mikasa_eval.py \
  --env-id RememberColor3-v0 \
  --checkpoint /home/gongming/Code/RMBench/policy/imitation-learning-policies/data/mikasa_remember_color_3/2026-10-22/21-33-07_remember_color_3_5h_wofirstpre_2obs/checkpoints/epoch_200_train_mean_loss_0_000.ckpt \
  --policy-root /home/gongming/Code/RMBench/policy/imitation-learning-policies \
  --num-envs 10 \
  --num-eval-episodes 100 \
  --abs-joint-pos \
  --seed 42 \
  --output-dir /home/gongming/Code/RMBench/eval_results/remember_color3_epoch100_5h_wofirstpre_2obs