#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source "${HOME}/myenv/bin/activate"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

TOKENIZED_MANIFEST="${TOKENIZED_MANIFEST:-${ROOT_DIR}/manifests/tokenized_keep_manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/checkpoints/formal_training}"
INITIAL_RESUME_FROM="${INITIAL_RESUME_FROM:-}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-formal}"
START_SEQ_LEN="${START_SEQ_LEN:-2048}"
SCHEDULE_STATE_FILE="${SCHEDULE_STATE_FILE:-${OUTPUT_DIR}/schedule_state.json}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
SAVE_EVERY="${SAVE_EVERY:-3000}"
EVAL_EVERY="${EVAL_EVERY:-3000}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-1337}"

LEARNING_RATE="${LEARNING_RATE:-3e-4}"
MIN_LEARNING_RATE="${MIN_LEARNING_RATE:-3e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

D_MODEL="${D_MODEL:-512}"
N_LAYERS="${N_LAYERS:-18}"
N_HEADS="${N_HEADS:-8}"
HEAD_DIM="${HEAD_DIM:-64}"
FFN_HIDDEN_DIM="${FFN_HIDDEN_DIM:-2048}"
KV_LATENT_DIM="${KV_LATENT_DIM:-256}"
DROPOUT="${DROPOUT:-0.0}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
STEP_SLEEP_SEC="${STEP_SLEEP_SEC:-0.2}"

WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-decoder-pretrain}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_NAME="${WANDB_NAME:-${RUN_NAME_PREFIX}_training}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_RESUME="${WANDB_RESUME:-allow}"
WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}}"
WANDB_ID="${WANDB_ID:-}"
WANDB_RUN_ID_FILE="${WANDB_RUN_ID_FILE:-${OUTPUT_DIR}/wandb_run_id.txt}"

EPOCHS_2048="${EPOCHS_2048:-3}"
EPOCHS_4096="${EPOCHS_4096:-2}"
EPOCHS_8192="${EPOCHS_8192:-2}"

BATCH_2048="${BATCH_2048:-16}"
BATCH_4096="${BATCH_4096:-8}"
BATCH_8192="${BATCH_8192:-4}"

GRAD_ACCUM_2048="${GRAD_ACCUM_2048:-1}"
GRAD_ACCUM_4096="${GRAD_ACCUM_4096:-1}"
GRAD_ACCUM_8192="${GRAD_ACCUM_8192:-1}"

if [[ ! -f "${TOKENIZED_MANIFEST}" ]]; then
  echo "Missing tokenized manifest: ${TOKENIZED_MANIFEST}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

validate_seq_len() {
  case "$1" in
    2048|4096|8192) ;;
    *) echo "Unsupported seq_len $1" >&2; exit 1 ;;
  esac
}

validate_seq_len "${START_SEQ_LEN}"

if [[ "${WANDB_ENABLED}" != "0" ]]; then
  mkdir -p "${WANDB_DIR}"
  if [[ -z "${WANDB_ID}" && -f "${WANDB_RUN_ID_FILE}" ]]; then
    WANDB_ID="$(<"${WANDB_RUN_ID_FILE}")"
  fi
  if [[ -z "${WANDB_ID}" ]]; then
    WANDB_ID="$(
      python - <<'PY'
import uuid
print(uuid.uuid4().hex)
PY
    )"
  fi
  printf '%s\n' "${WANDB_ID}" > "${WANDB_RUN_ID_FILE}"
fi

manifest_equivalent_steps() {
  local seq_len="$1"
  python - "$TOKENIZED_MANIFEST" "$seq_len" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
seq_len = int(sys.argv[2])
entries = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
for entry in entries:
    if entry.get("split") == "train" and int(entry.get("seq_len", -1)) == seq_len:
        print(int(round(float(entry["equivalent_steps"]))))
        break
else:
    raise SystemExit(f"Missing train entry for seq_len={seq_len} in {manifest_path}")
PY
}

checkpoint_step() {
  local checkpoint_path="$1"
  if [[ -z "${checkpoint_path}" || ! -f "${checkpoint_path}" ]]; then
    echo 0
    return
  fi
  python - "$checkpoint_path" <<'PY'
import sys
import torch

checkpoint_path = sys.argv[1]
try:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
print(int(checkpoint.get("step", 0)))
PY
}

read_schedule_state_field() {
  local field_name="$1"
  if [[ ! -f "${SCHEDULE_STATE_FILE}" ]]; then
    return 1
  fi
  python - "$SCHEDULE_STATE_FILE" "$field_name" <<'PY'
import json
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
field_name = sys.argv[2]
state = json.loads(state_path.read_text())
value = state[field_name]
if isinstance(value, (dict, list)):
    print(json.dumps(value))
else:
    print(value)
PY
}

write_schedule_state() {
  local base_step="$1"
  python - "$SCHEDULE_STATE_FILE" "$base_step" "$START_SEQ_LEN" "$INITIAL_RESUME_FROM" "$EPOCHS_2048" "$EPOCHS_4096" "$EPOCHS_8192" <<'PY'
import json
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
state = {
    "base_step": int(sys.argv[2]),
    "start_seq_len": int(sys.argv[3]),
    "initial_resume_from": sys.argv[4],
    "epochs": {
        "2048": int(sys.argv[5]),
        "4096": int(sys.argv[6]),
        "8192": int(sys.argv[7]),
    },
}
state_path.write_text(json.dumps(state, indent=2) + "\n")
PY
}

seq_epochs() {
  case "$1" in
    2048) echo "${EPOCHS_2048}" ;;
    4096) echo "${EPOCHS_4096}" ;;
    8192) echo "${EPOCHS_8192}" ;;
    *) echo "Unsupported seq_len $1" >&2; exit 1 ;;
  esac
}

seq_batch() {
  case "$1" in
    2048) echo "${BATCH_2048}" ;;
    4096) echo "${BATCH_4096}" ;;
    8192) echo "${BATCH_8192}" ;;
    *) echo "Unsupported seq_len $1" >&2; exit 1 ;;
  esac
}

seq_grad_accum() {
  case "$1" in
    2048) echo "${GRAD_ACCUM_2048}" ;;
    4096) echo "${GRAD_ACCUM_4096}" ;;
    8192) echo "${GRAD_ACCUM_8192}" ;;
    *) echo "Unsupported seq_len $1" >&2; exit 1 ;;
  esac
}

print_schedule() {
  printf '%s\n' 'Formal training schedule:'
  for seq_len in 2048 4096 8192; do
    local epochs
    local steps_per_epoch
    epochs="$(seq_epochs "${seq_len}")"
    steps_per_epoch="$(manifest_equivalent_steps "${seq_len}")"
    printf '  len=%s epochs=%s steps_per_epoch=%s total_segment_steps=%s\n' \
      "${seq_len}" "${epochs}" "${steps_per_epoch}" "$(( epochs * steps_per_epoch ))"
  done
}

print_schedule

RESUME_FROM=""
if [[ -f "${OUTPUT_DIR}/latest_checkpoint.pt" ]]; then
  RESUME_FROM="${OUTPUT_DIR}/latest_checkpoint.pt"
elif [[ -n "${INITIAL_RESUME_FROM}" ]]; then
  RESUME_FROM="${INITIAL_RESUME_FROM}"
fi

BASE_STEP=0
if [[ -f "${SCHEDULE_STATE_FILE}" ]]; then
  BASE_STEP="$(read_schedule_state_field base_step)"
  START_SEQ_LEN="$(read_schedule_state_field start_seq_len)"
else
  if [[ -n "${INITIAL_RESUME_FROM}" && ! -f "${OUTPUT_DIR}/latest_checkpoint.pt" ]]; then
    BASE_STEP="$(checkpoint_step "${INITIAL_RESUME_FROM}")"
  fi
  write_schedule_state "${BASE_STEP}"
fi

validate_seq_len "${START_SEQ_LEN}"

CURRENT_STEP="$(checkpoint_step "${RESUME_FROM}")"
CURRENT_STEP="${CURRENT_STEP:-0}"
CUMULATIVE_TARGET="${BASE_STEP}"
RAN_ANY_SEGMENT=0

echo "start_seq_len=${START_SEQ_LEN}"
echo "base_step=${BASE_STEP}"
echo "resume_checkpoint=${RESUME_FROM:-<none>}"

for SEQ_LEN in 2048 4096 8192; do
  if (( SEQ_LEN < START_SEQ_LEN )); then
    echo "Skipping len=${SEQ_LEN}: before start_seq_len=${START_SEQ_LEN}"
    continue
  fi

  EPOCHS="$(seq_epochs "${SEQ_LEN}")"
  if (( EPOCHS <= 0 )); then
    echo "Skipping len=${SEQ_LEN}: epochs=${EPOCHS}"
    continue
  fi

  STEPS_PER_EPOCH="$(manifest_equivalent_steps "${SEQ_LEN}")"
  SEGMENT_STEPS=$(( STEPS_PER_EPOCH * EPOCHS ))
  CUMULATIVE_TARGET=$(( CUMULATIVE_TARGET + SEGMENT_STEPS ))

  if (( CURRENT_STEP >= CUMULATIVE_TARGET )); then
    echo "Skipping len=${SEQ_LEN}: checkpoint already at step ${CURRENT_STEP} >= target ${CUMULATIVE_TARGET}"
    continue
  fi

  MICRO_BATCH="$(seq_batch "${SEQ_LEN}")"
  GRAD_ACCUM="$(seq_grad_accum "${SEQ_LEN}")"
  RUN_NAME="${RUN_NAME_PREFIX}_len${SEQ_LEN}"

  echo
  echo "=== Running ${RUN_NAME} ==="
  echo "seq_len=${SEQ_LEN} epochs=${EPOCHS} steps_per_epoch=${STEPS_PER_EPOCH} target_step=${CUMULATIVE_TARGET}"
  echo "micro_batch=${MICRO_BATCH} grad_accum=${GRAD_ACCUM} resume_from=${RESUME_FROM:-<none>}"
  echo "step_sleep_sec=${STEP_SLEEP_SEC}"
  if [[ "${WANDB_ENABLED}" != "0" ]]; then
    echo "wandb_project=${WANDB_PROJECT} wandb_name=${WANDB_NAME} wandb_id=${WANDB_ID} wandb_mode=${WANDB_MODE}"
  fi

  CMD=(
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m decoder_distill.train
    --run-name "${RUN_NAME}"
    --seq-len "${SEQ_LEN}"
    --per-device-batch-size "${MICRO_BATCH}"
    --grad-accum-steps "${GRAD_ACCUM}"
    --tokenized-train-manifest "${TOKENIZED_MANIFEST}"
    --tokenized-eval-manifest "${TOKENIZED_MANIFEST}"
    --train-split train
    --eval-split valid
    --output-dir "${OUTPUT_DIR}"
    --num-steps "${CUMULATIVE_TARGET}"
    --save-every "${SAVE_EVERY}"
    --eval-every "${EVAL_EVERY}"
    --eval-max-batches "${EVAL_MAX_BATCHES}"
    --num-workers "${NUM_WORKERS}"
    --seed "${SEED}"
    --learning-rate "${LEARNING_RATE}"
    --min-learning-rate "${MIN_LEARNING_RATE}"
    --warmup-steps "${WARMUP_STEPS}"
    --weight-decay "${WEIGHT_DECAY}"
    --grad-clip "${GRAD_CLIP}"
    --d-model "${D_MODEL}"
    --n-layers "${N_LAYERS}"
    --n-heads "${N_HEADS}"
    --head-dim "${HEAD_DIM}"
    --ffn-hidden-dim "${FFN_HIDDEN_DIM}"
    --kv-latent-dim "${KV_LATENT_DIM}"
    --dropout "${DROPOUT}"
    --max-seq-len "${MAX_SEQ_LEN}"
    --step-sleep-sec "${STEP_SLEEP_SEC}"
  )

  if [[ "${WANDB_ENABLED}" != "0" ]]; then
    CMD+=(
      --wandb-enabled
      --wandb-project "${WANDB_PROJECT}"
      --wandb-name "${WANDB_NAME}"
      --wandb-id "${WANDB_ID}"
      --wandb-resume "${WANDB_RESUME}"
      --wandb-mode "${WANDB_MODE}"
      --wandb-dir "${WANDB_DIR}"
    )
    if [[ -n "${WANDB_ENTITY}" ]]; then
      CMD+=(--wandb-entity "${WANDB_ENTITY}")
    fi
    if [[ -n "${WANDB_GROUP}" ]]; then
      CMD+=(--wandb-group "${WANDB_GROUP}")
    fi
  fi

  if [[ -n "${RESUME_FROM}" && -f "${RESUME_FROM}" ]]; then
    CMD+=(--resume-from "${RESUME_FROM}")
  fi

  "${CMD[@]}"

  RAN_ANY_SEGMENT=1
  RESUME_FROM="${OUTPUT_DIR}/latest_checkpoint.pt"
  CURRENT_STEP="$(checkpoint_step "${RESUME_FROM}")"
done

if (( RAN_ANY_SEGMENT == 0 )); then
  echo "Nothing to run: all requested segments are already complete."
fi
