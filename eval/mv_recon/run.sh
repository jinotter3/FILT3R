#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# shellcheck source=eval/public_common.sh
source "${ROOT_DIR}/eval/public_common.sh"

public_load_model_list MODEL_LIST
public_load_extra_hparams EXTRA_HPARAMS

MODEL_WEIGHTS="${MODEL_WEIGHTS:-${ROOT_DIR}/src/cut3r_512_dpt_4_64.pth}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29502}"
EVAL_SIZE="$(public_normalize_int "${EVAL_SIZE:-${SIZE:-512}}" 512 "eval size")"
RUN_TAG="${RUN_TAG:-}"
OVERWRITE="${OVERWRITE:-false}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/eval_results/mv_recon}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data/7scenes}"
SCENE_ID="${SCENE_ID:-}"
KF_EVERY="${KF_EVERY:-}"
VOXEL_SIZE="${VOXEL_SIZE:-0.0}"
CONF_THRESH="${CONF_THRESH:-0.0}"
EVAL_CENTER_CROP="${EVAL_CENTER_CROP:-224}"
DIST_TIMEOUT_MIN="${DIST_TIMEOUT_MIN:-180}"
INFERENCE_IMPL="${INFERENCE_IMPL:-recurrent_lighter}"

if [[ -n "${MAX_FRAMES:-}" ]]; then
  FRAME_BUDGET_LIST=("${MAX_FRAMES}")
elif [[ -n "${SEVENSCENES_MAX_FRAMES_LIST:-}" ]]; then
  read -r -a FRAME_BUDGET_LIST <<< "${SEVENSCENES_MAX_FRAMES_LIST}"
elif [[ -n "${FRAME_BUDGETS:-}" ]]; then
  read -r -a FRAME_BUDGET_LIST <<< "${FRAME_BUDGETS}"
else
  FRAME_BUDGET_LIST=("300" "400" "500")
fi

for max_frames in "${FRAME_BUDGET_LIST[@]}"; do
  for model_name in "${MODEL_LIST[@]}"; do
    HPARAM_ARGS=()
    public_build_hparam_args "${model_name}" EXTRA_HPARAMS HPARAM_ARGS

    suffix=""
    if [[ -n "${RUN_TAG}" ]]; then
      suffix="-${RUN_TAG}"
    fi
    output_dir="${OUT_ROOT}/7scenes/frames_${max_frames}/${model_name}${suffix}"

    if [[ -d "${output_dir}" ]] && ! public_is_true "${OVERWRITE}"; then
      echo "[mv_recon] skipping existing output ${output_dir}"
      continue
    fi

    cmd=(
      accelerate launch
      --num_processes "${NUM_PROCESSES}"
      --main_process_port "${MAIN_PROCESS_PORT}"
      eval/mv_recon/launch.py
      --weights "${MODEL_WEIGHTS}"
      --output_dir "${output_dir}"
      --eval_dataset "7scenes"
      --size "${EVAL_SIZE}"
      --max_frames "${max_frames}"
      --model_update_type "${model_name}"
      --conf_thresh "${CONF_THRESH}"
      --voxel_size "${VOXEL_SIZE}"
      --eval_center_crop "${EVAL_CENTER_CROP}"
      --dist_timeout_min "${DIST_TIMEOUT_MIN}"
      --inference_impl "${INFERENCE_IMPL}"
      --data_root "${DATA_ROOT}"
    )

    if [[ -n "${SCENE_ID}" ]]; then
      cmd+=(--scene_id "${SCENE_ID}")
    fi
    if [[ -n "${KF_EVERY}" ]]; then
      cmd+=(--kf_every "${KF_EVERY}")
    fi
    if [[ ${#HPARAM_ARGS[@]} -gt 0 ]]; then
      cmd+=("${HPARAM_ARGS[@]}")
    fi

    echo "[mv_recon] dataset=7scenes frames=${max_frames} model=${model_name} output=${output_dir}"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_TIMEOUT=360000 "${cmd[@]}"
  done
done
