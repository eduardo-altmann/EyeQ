#!/bin/bash
# submit_experiment.sh
#
# Descobre automaticamente o próximo índice de execução (01, 02, 03, ...),
# cria a estrutura de pastas correspondente e submete train_job.sbatch
# passando esse índice como variável de ambiente (RUN_ID).
#
# Uso:
#   ./submit_experiment.sh
#
# Não precisa editar nada aqui a cada execução — os dois scripts ficam fixos.

set -euo pipefail

# Descobre o diretório onde este script está, independente de onde ele for chamado
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_RESULTS=/home/users/eadbem/eyeq/results/optuna
BASE_LOGS=/ssd2/eadbem/thiago/EyeQ/optuna

mkdir -p "$BASE_RESULTS"

# Procura a maior pasta numérica já existente (01, 02, ...) e soma 1
LAST_ID=$(find "$BASE_RESULTS" -maxdepth 1 -mindepth 1 -type d -name '[0-9][0-9]' -printf '%f\n' \
            | sort -n | tail -1)

if [ -z "$LAST_ID" ]; then
    NEXT_ID=1
else
    # 10#$LAST_ID força interpretação em base 10 (evita erro com zero à esquerda, ex: "08")
    NEXT_ID=$((10#$LAST_ID + 1))
fi

RUN_ID=$(printf "%02d" "$NEXT_ID")

echo "Nova execução: RUN_ID=${RUN_ID}"

mkdir -p "${BASE_RESULTS}/${RUN_ID}/preprocess"
mkdir -p "${BASE_RESULTS}/${RUN_ID}/training"
mkdir -p "${BASE_RESULTS}/${RUN_ID}/logs"
mkdir -p "${BASE_LOGS}/${RUN_ID}"

sbatch \
    --job-name="hcpa_eyeq_optuna_${RUN_ID}" \
    --output="${BASE_LOGS}/${RUN_ID}/%x_%j.out" \
    --error="${BASE_LOGS}/${RUN_ID}/%x_%j.err" \
    --export=ALL,RUN_ID="${RUN_ID}" \
    "${SCRIPT_DIR}/train_job.sbatch"

echo "Job submetido. Resultados irão para: ${BASE_RESULTS}/${RUN_ID}/"
echo "Logs (stdout/stderr) irão para:      ${BASE_LOGS}/${RUN_ID}/"