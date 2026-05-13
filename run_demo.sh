#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

SEQ_PATH="${1:-examples/taylor.mp4}"
OUTPUT_DIR="${2:-tmp/filt3r_demo}"
GPU="${GPU:-0}"
MODEL_PATH="${MODEL_PATH:-src/cut3r_512_dpt_4_64.pth}"
MODEL_UPDATE_TYPE="${MODEL_UPDATE_TYPE:-filt3r}"
PORT="${PORT:-8080}"

CUDA_VISIBLE_DEVICES="${GPU}" python demo.py \
  --model_path "${MODEL_PATH}" \
  --size 512 \
  --seq_path "${SEQ_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --port "${PORT}" \
  --model_update_type "${MODEL_UPDATE_TYPE}" \
  --frame_interval 1 \
  --downsample_factor 100 \
  --vis_threshold 6.0
