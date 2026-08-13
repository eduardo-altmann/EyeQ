#!/bin/bash
#SBATCH --job-name=hcpa_eyeq_optuna
#SBATCH --partition=blaise
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=/ssd2/eadbem/thiago/EyeQ/optuna/%x_%j.out
#SBATCH --error=/ssd2/eadbem/thiago/EyeQ/optuna/%x_%j.err

set -uo pipefail

RESULTS_DIR=/home/users/eadbem/eyeq/results/optuna
LOGS_DIR=/ssd2/eadbem/thiago/EyeQ/optuna

mkdir -p "${RESULTS_DIR}/preprocess" "${RESULTS_DIR}/training" "${RESULTS_DIR}/logs"

# ---------- Atualização do repositório ----------
echo "Atualizando repositório..."
cd /ssd2/eadbem/thiago/EyeQ
git pull

# ---------- Pré-processamento ----------
#echo "Iniciando o pré-processamento..."
#cd /ssd2/eadbem/thiago/EyeQ/EyeQ_preprocess
#uv run EyeQ_process_main.py
#rsync -avz metrics/ "${RESULTS_DIR}/preprocess/"

# ---------- Treinamento ----------
echo "Iniciando o treinamento..."
cd /ssd2/eadbem/thiago/EyeQ/MCF_Net

uv run tensorboard --logdir runs --host 143.54.85.24 --port 6006 &
TB_PID=$!

uv run optuna-dashboard sqlite:///optuna_study.db --host 143.54.85.24 --port 8080 &
DASH_PID=$!

uv run optuna_driver.py --n_gpus 4 --n_trials 12 --epochs 25

# Encerra os dashboards (senão ficam pendurados após o job acabar)
kill "$TB_PID" "$DASH_PID" 2>/dev/null

rsync -avz result/ "${RESULTS_DIR}/training/"
rsync -avz optuna_study.db "${RESULTS_DIR}/training/"
rsync -avz runs/ "${RESULTS_DIR}/training/runs/"

echo "=== Execução finalizada ==="

# ---------- Logs do SLURM ----------
sleep 15
rsync -avz "${LOGS_DIR}/" "${RESULTS_DIR}/logs/"
