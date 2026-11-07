#!/usr/bin/env bash
# A short end-to-end check. Five optimizer steps validate plumbing, not task learning.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
run=(uv run --project "$repo_root/examples/piper_sim" --no-sync)
export MUJOCO_GL="${MUJOCO_GL:-egl}"
if [[ "$MUJOCO_GL" == "egl" ]]; then
    export PYOPENGL_PLATFORM=egl
fi
result_root="${1:-$repo_root/outputs/piper_sim/smoke}"
device="${PIPER_SIM_DEVICE:-cpu}"
if [[ -e "$result_root" ]]; then
    echo "Output already exists: $result_root. Pass a fresh output directory." >&2
    exit 1
fi
mkdir -p "$result_root"
result_root="$(cd -- "$result_root" && pwd)"
"${run[@]}" piper-sim doctor --width 160 --height 120
"${run[@]}" piper-sim record --controller expert --episodes 2 --success-only --width 160 --height 120 \
    --root "$result_root/dataset" --repo-id local/piper_sim_smoke
"${run[@]}" piper-sim replay --root "$result_root/dataset" --repo-id local/piper_sim_smoke
"${run[@]}" lerobot-train \
    --dataset.repo_id=local/piper_sim_smoke --dataset.root="$result_root/dataset" \
    --dataset.video_backend=pyav \
    --policy.type=act --policy.device="$device" --policy.push_to_hub=false \
    --policy.pretrained_backbone_weights=null --policy.dim_model=128 --policy.n_heads=4 \
    --policy.dim_feedforward=512 --policy.n_encoder_layers=2 --policy.n_decoder_layers=1 \
    --policy.n_vae_encoder_layers=2 --policy.chunk_size=8 --policy.n_action_steps=8 \
    --batch_size=4 --steps=5 --save_freq=5 --log_freq=1 --num_workers=0 \
    --env_eval_freq=0 --eval_steps=0 --wandb.enable=false --output_dir="$result_root/train"
"${run[@]}" piper-sim evaluate --policy "$result_root/train/checkpoints/last/pretrained_model" \
    --device "$device" --width 160 --height 120 --episodes 2 --duration 2 --seed 10000 \
    --report "$result_root/evaluation.json" --video-dir "$result_root/videos"
echo "End-to-end check completed: $result_root"
