#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# shellcheck source=eval/public_common.sh
source "${ROOT_DIR}/eval/public_common.sh"

public_load_model_list MODEL_LIST
public_load_seq_list SEQ_LIST_ARGS
public_load_extra_hparams EXTRA_HPARAMS
public_load_space_separated_env \
  SCANNET_DATASETS \
  DATASET_LIST \
  "scannet_s3_50" \
  "scannet_s3_100" \
  "scannet_s3_150" \
  "scannet_s3_200" \
  "scannet_s3_300" \
  "scannet_s3_400" \
  "scannet_s3_500" \
  "scannet_s3_600" \
  "scannet_s3_700" \
  "scannet_s3_800" \
  "scannet_s3_900" \
  "scannet_s3_1000"

MODEL_WEIGHTS="${MODEL_WEIGHTS:-${ROOT_DIR}/src/cut3r_512_dpt_4_64.pth}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29550}"
EVAL_SIZE="$(public_normalize_int "${EVAL_SIZE:-${SIZE:-512}}" 512 "eval size")"
POSE_EVAL_STRIDE="${POSE_EVAL_STRIDE:-1}"
MAX_FRAMES="${MAX_FRAMES:-0}"
REVISIT="${REVISIT:-1}"
DIST_TIMEOUT_SEC="${DIST_TIMEOUT_SEC:-3600}"
FULL_SEQ="${FULL_SEQ:-false}"
RUN_TAG="${RUN_TAG:-}"
OVERWRITE="${OVERWRITE:-false}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/eval_results/relpose}"

RESUME_FLAG=()
public_load_resume_flag RESUME_FLAG

for eval_dataset in "${DATASET_LIST[@]}"; do
  for model_name in "${MODEL_LIST[@]}"; do
    HPARAM_ARGS=()
    public_build_hparam_args "${model_name}" EXTRA_HPARAMS HPARAM_ARGS

    suffix=""
    if [[ -n "${RUN_TAG}" ]]; then
      suffix="-${RUN_TAG}"
    fi
    output_dir="${OUT_ROOT}/${eval_dataset}/${model_name}${suffix}"

    cmd=(
      accelerate launch
      --num_processes "${NUM_PROCESSES}"
      --main_process_port "${MAIN_PROCESS_PORT}"
      eval/relpose/launch.py
      --weights "${MODEL_WEIGHTS}"
      --output_dir "${output_dir}"
      --eval_dataset "${eval_dataset}"
      --size "${EVAL_SIZE}"
      --model_update_type "${model_name}"
      --pose_eval_stride "${POSE_EVAL_STRIDE}"
      --max_frames "${MAX_FRAMES}"
      --revisit "${REVISIT}"
      --dist_timeout_sec "${DIST_TIMEOUT_SEC}"
    )

    if public_is_true "${FULL_SEQ}"; then
      cmd+=(--full_seq)
    fi
    if [[ ${#SEQ_LIST_ARGS[@]} -gt 0 ]]; then
      cmd+=(--seq_list "${SEQ_LIST_ARGS[@]}")
    fi
    if [[ ${#HPARAM_ARGS[@]} -gt 0 ]]; then
      cmd+=("${HPARAM_ARGS[@]}")
    fi
    cmd+=("${RESUME_FLAG[@]}")

    echo "[relpose] dataset=${eval_dataset} model=${model_name} output=${output_dir}"
    "${cmd[@]}"
  done
done
