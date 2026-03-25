#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOKENIZED_MANIFEST="${TOKENIZED_MANIFEST:-${ROOT_DIR}/manifests/tokenized_keep_manifest.jsonl}"

if [[ ! -f "${TOKENIZED_MANIFEST}" ]]; then
  echo "Missing tokenized manifest: ${TOKENIZED_MANIFEST}" >&2
  exit 1
fi

TOKENIZED_MANIFEST="${TOKENIZED_MANIFEST}" "${ROOT_DIR}/scripts/run_phase1.sh"
TOKENIZED_MANIFEST="${TOKENIZED_MANIFEST}" "${ROOT_DIR}/scripts/run_phase2.sh"
