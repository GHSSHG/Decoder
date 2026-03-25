#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source "${HOME}/myenv/bin/activate"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

TOKENIZED_MANIFEST="${TOKENIZED_MANIFEST:-${ROOT_DIR}/manifests/tokenized_keep_manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/checkpoints/phase2}"
DEFAULT_RESUME="${ROOT_DIR}/checkpoints/phase1/latest_checkpoint.pt"
RESUME_FROM="${RESUME_FROM:-${DEFAULT_RESUME}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_STEPS="${NUM_STEPS:-0}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
EVAL_EVERY="${EVAL_EVERY:-2000}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-8}"

if [[ "${NUM_STEPS}" -le 0 ]]; then
  echo "Set NUM_STEPS to the formal Phase 2 step count." >&2
  exit 1
fi

if [[ ! -f "${TOKENIZED_MANIFEST}" ]]; then
  echo "Missing tokenized manifest: ${TOKENIZED_MANIFEST}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

CMD=(
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m decoder_distill.train
  --phase phase2
  --bucket-spec "${BUCKET_2048:-2048,0.6,16,1}"
  --bucket-spec "${BUCKET_4096:-4096,0.3,8,1}"
  --bucket-spec "${BUCKET_8192:-8192,0.1,4,1}"
  --tokenized-train-manifest "${TOKENIZED_MANIFEST}"
  --tokenized-eval-manifest "${TOKENIZED_MANIFEST}"
  --train-split train
  --eval-split valid
  --output-dir "${OUTPUT_DIR}"
  --num-steps "${NUM_STEPS}"
  --save-every "${SAVE_EVERY}"
  --eval-every "${EVAL_EVERY}"
  --eval-max-batches "${EVAL_MAX_BATCHES}"
  --learning-rate "${LEARNING_RATE:-3e-4}"
  --min-learning-rate "${MIN_LEARNING_RATE:-3e-5}"
  --warmup-steps "${WARMUP_STEPS:-100}"
  --weight-decay "${WEIGHT_DECAY:-0.1}"
  --grad-clip "${GRAD_CLIP:-1.0}"
  --d-model "${D_MODEL:-512}"
  --n-layers "${N_LAYERS:-18}"
  --n-heads "${N_HEADS:-8}"
  --head-dim "${HEAD_DIM:-64}"
  --ffn-hidden-dim "${FFN_HIDDEN_DIM:-2048}"
  --kv-latent-dim "${KV_LATENT_DIM:-256}"
  --dropout "${DROPOUT:-0.0}"
)

if [[ -f "${RESUME_FROM}" ]]; then
  CMD+=(--resume-from "${RESUME_FROM}")
fi

"${CMD[@]}"
