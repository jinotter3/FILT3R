#!/usr/bin/env bash

PUBLIC_FILT3R_DEFAULT_HPARAMS=(
  "kalman_p_init=1.5"
  "kalman_gamma_p=1.0"
  "kalman_q_min=0.02"
  "kalman_q_max=0.5"
  "kalman_alpha_q=20.0"
  "kalman_tau_q=3.0"
  "kalman_ema_beta_delta=0.05"
  "kalman_ema_delta_floor=1e-2"
  "kalman_fixed_r=1.0"
)

public_is_true() {
  local value="${1:-}"
  value="${value,,}"
  [[ "${value}" == "1" || "${value}" == "true" || "${value}" == "yes" ]]
}

public_normalize_int() {
  local value="$1"
  local fallback="$2"
  local label="$3"

  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "Invalid ${label} '${value}'. Falling back to ${fallback}." >&2
    echo "${fallback}"
    return
  fi

  echo "${value}"
}

public_load_model_list() {
  local -n out_ref="$1"
  if [[ -n "${MODEL_NAMES:-}" ]]; then
    read -r -a out_ref <<< "${MODEL_NAMES}"
  else
    out_ref=("cut3r" "ttt3r" "filt3r")
  fi
}

public_load_space_separated_env() {
  local env_name="$1"
  local -n out_ref="$2"
  shift 2

  if [[ -n "${!env_name:-}" ]]; then
    read -r -a out_ref <<< "${!env_name}"
  else
    out_ref=("$@")
  fi
}

public_load_seq_list() {
  local -n out_ref="$1"
  if [[ -n "${SEQ_LIST:-}" ]]; then
    read -r -a out_ref <<< "${SEQ_LIST}"
  else
    out_ref=()
  fi
}

public_load_extra_hparams() {
  local -n out_ref="$1"
  if [[ -n "${EXTRA_MODEL_HPARAMS:-}" ]]; then
    read -r -a out_ref <<< "${EXTRA_MODEL_HPARAMS}"
  else
    out_ref=()
  fi
}

public_load_resume_flag() {
  local -n out_ref="$1"
  out_ref=(--resume)
  if public_is_true "${OVERWRITE:-false}"; then
    out_ref=(--no_resume)
  fi
}

public_build_hparam_args() {
  local model_name="$1"
  local -n extra_ref="$2"
  local -n out_ref="$3"

  out_ref=()
  if [[ "${model_name}" == "filt3r" ]]; then
    local hparam
    for hparam in "${PUBLIC_FILT3R_DEFAULT_HPARAMS[@]}"; do
      out_ref+=(--model_hparam "${hparam}")
    done
  fi

  local extra_hparam
  for extra_hparam in "${extra_ref[@]}"; do
    out_ref+=(--model_hparam "${extra_hparam}")
  done
}
