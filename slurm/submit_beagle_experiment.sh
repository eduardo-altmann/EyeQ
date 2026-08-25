#!/bin/bash
# submit_single_experiment.sh
#
# Descobre automaticamente o próximo índice de execução (01, 02, 03, ...)
# dentro de single/, cria a estrutura de pastas na home e submete
# train_job_single.sbatch. Os logs do SLURM são escritos diretamente
# na home (sem etapa extra de rsync).
#
# Uso:
#   ./submit_single_experiment.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_RESULTS=/home/users/eadbem/eyeq/results/single

mkdir -p "$BASE_RESULTS"

LAST_ID=$(find "$BASE_RESULTS" -maxdepth 1 -mindepth 1 -type d -name '[0-9][0-9]' -printf '%f\n' \
            | sort -n | tail -1)

if [ -z "$LAST_ID" ]; then
    NEXT_ID=1
else
    NEXT_ID=$((10#$LAST_ID + 1))
fi

RUN_ID=$(printf "%02d" "$NEXT_ID")

echo "Nova execução single: RUN_ID=${RUN_ID}"

mkdir -p "${BASE_RESULTS}/${RUN_ID}/preprocess"
mkdir -p "${BASE_RESULTS}/${RUN_ID}/training"
mkdir -p "${BASE_RESULTS}/${RUN_ID}/logs"

sbatch \
    --job-name="hcpa_eyeq_single_${RUN_ID}" \
    --output="${BASE_RESULTS}/${RUN_ID}/logs/%x_%j.out" \
    --error="${BASE_RESULTS}/${RUN_ID}/logs/%x_%j.err" \
    --export=ALL,RUN_ID="${RUN_ID}" \
    "${SCRIPT_DIR}/train_job_beagle.sbatch"

echo "Job submetido. Resultados irão para: ${BASE_RESULTS}/${RUN_ID}/"