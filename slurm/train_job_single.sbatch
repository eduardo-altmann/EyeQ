#!/bin/bash
#SBATCH --partition=poti
#SBATCH --nodelist=poti5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
# job-name, output e error são definidos dinamicamente pelo submit_single_experiment.sh

set -uo pipefail

if [ -z "${RUN_ID:-}" ]; then
    echo "ERRO: RUN_ID não definido. Não rode este script diretamente com sbatch;" >&2
    echo "use ./submit_single_experiment.sh, que define o índice e submete o job." >&2
    exit 1
fi

RESULTS_DIR=/home/users/eadbem/eyeq/results/single/${RUN_ID}
echo "=== Execução single RUN_ID=${RUN_ID} ==="
echo "Resultados serão salvos em: ${RESULTS_DIR}"

# ---------- Atualização do repositório ----------
echo "Atualizando repositório..."
cd /scratch/eadbem/EyeQ
git pull

# ---------- Pré-processamento ----------
#echo "Iniciando o pré-processamento..."
#cd /scratch/eadbem/EyeQ/EyeQ_preprocess
#uv run EyeQ_process_main.py
#rsync -avz metrics/ "${RESULTS_DIR}/preprocess/"

# ---------- Treinamento ----------
echo "Iniciando o treinamento..."
cd /scratch/eadbem/EyeQ/MCF_Net

uv run tensorboard --logdir runs --host 192.168.30.51 --port 8080 &
TB_PID=$!

uv run torchrun \
    Main_EyeQuality_tuned_parallel.py \
    --lr 0.00837248326772002 \
    --batch-size 8 \
    --weight_decay 1.3866006245078523e-05 \
    --lr_scheduler cosine \
    --momentum 0.8007658696453809 \
    --warmup_epochs 5 \
    --epochs 60

# Encerra o TensorBoard (senão fica pendurado após o job acabar)
kill "$TB_PID" 2>/dev/null

rsync -avz result/ "${RESULTS_DIR}/training/"
rsync -avz runs/ "${RESULTS_DIR}/training/runs/"

echo "=== Execução single RUN_ID=${RUN_ID} finalizada ==="
