# 仿真短流程：数采、回放、五步训练、推理

[仿真流程](README.md) · [仿真实现](../../architecture/simulation.md) · [仿真排错](../../troubleshooting/simulation.md)

先完成 [仿真流程的环境与资源准备](README.md#安装与首次检查)，从仓库根目录逐条执行。
新终端需先按[安装与首次检查](README.md#安装与首次检查)清理继承的环境变量，再执行下列渲染设置与命令。
以下使用 `outputs/piper_sim/smoke_run1`；每次检查请换一个尚未使用的输出目录。
默认使用 CPU 验证链路，已有可用 GPU 时可将两处 `cpu` 改为 `cuda`。

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

uv run --project examples/piper_sim --no-sync piper-sim doctor --width 160 --height 120

uv run --project examples/piper_sim --no-sync piper-sim record \
  --controller expert --episodes 2 --success-only --width 160 --height 120 \
  --root outputs/piper_sim/smoke_run1/dataset --repo-id local/piper_sim_smoke

uv run --project examples/piper_sim --no-sync piper-sim replay \
  --root outputs/piper_sim/smoke_run1/dataset --repo-id local/piper_sim_smoke

uv run --project examples/piper_sim --no-sync lerobot-train \
  --dataset.repo_id=local/piper_sim_smoke \
  --dataset.root=outputs/piper_sim/smoke_run1/dataset \
  --dataset.video_backend=pyav \
  --policy.type=act --policy.device=cpu --policy.push_to_hub=false \
  --policy.pretrained_backbone_weights=null --policy.dim_model=128 --policy.n_heads=4 \
  --policy.dim_feedforward=512 --policy.n_encoder_layers=2 --policy.n_decoder_layers=1 \
  --policy.n_vae_encoder_layers=2 --policy.chunk_size=8 --policy.n_action_steps=8 \
  --batch_size=4 --steps=5 --save_freq=5 --log_freq=1 --num_workers=0 \
  --env_eval_freq=0 --eval_steps=0 --wandb.enable=false \
  --output_dir=outputs/piper_sim/smoke_run1/train

uv run --project examples/piper_sim --no-sync piper-sim evaluate \
  --policy outputs/piper_sim/smoke_run1/train/checkpoints/last/pretrained_model \
  --device cpu --width 160 --height 120 --episodes 2 --duration 2 --seed 10000 \
  --report outputs/piper_sim/smoke_run1/evaluation.json \
  --video-dir outputs/piper_sim/smoke_run1/videos
```

该流程只训练五步，用于检查依赖、数据、checkpoint 与推理能否衔接；任务学习需要更多
演示与训练步数。所有动作由程序专家或策略发送给虚拟从臂，不连接真实主臂。
