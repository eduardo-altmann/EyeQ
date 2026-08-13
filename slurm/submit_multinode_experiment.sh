#!/bin/bash
# submit_multinode_experiment.sh
#
# Descobre automaticamente o próximo índice de execução (01, 02, 03, ...),
# cria a estrutura de pastas correspondente em /home (compartilhado) e submete
# train_job_multinode.sbatch passando esse índice como variável de ambiente (RUN_ID).
#
# Uso:
#   ./submit_multinode_experiment.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_RESULTS=/home/users/eadbem/eyeq/results/multinode

mkdir -p "$BASE_RESULTS"

LAST_ID=$(find "$BASE_RESULTS" -maxdepth 1 -mindepth 1 -type d -name '[0-9][0-9]' -printf '%f\n' \
            | sort -n | tail -1)

if [ -z "$LAST_ID" ]; then
    NEXT_ID=1
else
    NEXT_ID=$((10#$LAST_ID + 1))
fi

RUN_ID=$(printf "%02d" "$NEXT_ID")

echo "Nova execução: RUN_ID=${RUN_ID}"

# training/ e logs/ ficam sob a mesma arvore em /home (compartilhado entre nos) -
# nao ha mais um passo separado de "copiar logs depois", o SLURM ja escreve
# --output/--error direto aqui, entao funciona independente de qual no
# (draco1 ou nao) o SLURM escolher como "batch host" do job.
mkdir -p "${BASE_RESULTS}/${RUN_ID}/training"
mkdir -p "${BASE_RESULTS}/${RUN_ID}/logs"

sbatch \
    --job-name="hcpa_eyeq_multinode_${RUN_ID}" \
    --output="${BASE_RESULTS}/${RUN_ID}/logs/%x_%j.out" \
    --error="${BASE_RESULTS}/${RUN_ID}/logs/%x_%j.err" \
    --export=ALL,RUN_ID="${RUN_ID}" \
    "${SCRIPT_DIR}/train_job_multinode.sbatch"

echo "Job submetido. Resultados irão para: ${BASE_RESULTS}/${RUN_ID}/"
