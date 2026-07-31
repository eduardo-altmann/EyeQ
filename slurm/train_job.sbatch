#!/bin/bash
#SBATCH --partition=blaise
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
# job-name, output e error são definidos dinamicamente pelo submit_experiment.sh

# sem "-e": queremos garantir que o rsync final rode mesmo se uma etapa falhar
set -uo pipefail

if [ -z "${RUN_ID:-}" ]; then
    echo "ERRO: RUN_ID não definido. Não rode este script diretamente com sbatch;" >&2
    echo "use ./submit_experiment.sh, que define o índice e submete o job." >&2
    exit 1
fi

RESULTS_DIR=/home/users/eadbem/eyeq/results/optuna/${RUN_ID}
LOGS_DIR=/ssd2/eadbem/thiago/EyeQ/optuna/${RUN_ID}
echo "=== Execução RUN_ID=${RUN_ID} ==="
echo "Resultados serão salvos em: ${RESULTS_DIR}"

# ---------- Atualização do repositório ----------
echo "Atualizando repositório..."
cd /ssd2/eadbem/thiago/EyeQ
git pull

# ---------- Pré-processamento ----------
echo "Iniciando o pré-processamento..."
cd /ssd2/eadbem/thiago/EyeQ/EyeQ_preprocess
uv run EyeQ_process_main.py
rsync -avz metrics/ "${RESULTS_DIR}/preprocess/"

# ---------- Treinamento ----------
echo "Iniciando o treinamento..."
cd /ssd2/eadbem/thiago/EyeQ/MCF_Net

uv run tensorboard --logdir runs --host 143.54.85.24 --port 6006 &
TB_PID=$!

uv run optuna-dashboard sqlite:///optuna_study.db --host 143.54.85.24 --port 8080 &
DASH_PID=$!

uv run optuna_driver.py --n_gpus 4 --n_trials 6 --epochs 25

# Encerra os dashboards (senão ficam pendurados após o job acabar)
kill "$TB_PID" "$DASH_PID" 2>/dev/null

rsync -avz result/ "${RESULTS_DIR}/training/"
rsync -avz optuna_study.db "${RESULTS_DIR}/training/"
rsync -avz optuna_results.csv "${RESULTS_DIR}/training/"
rsync -avz runs/ "${RESULTS_DIR}/training/runs/"

echo "=== Execução RUN_ID=${RUN_ID} finalizada ==="

# ---------- Logs do SLURM ----------
# Pequeno sleep para dar tempo do SO liberar/gravar o restante do buffer
# do .out/.err antes de lermos o arquivo. Mesmo assim, como o job ainda
# está rodando, as últimas linhas (incluindo esta) podem ficar de fora
# da cópia levada para a home — rode o rsync manualmente depois se precisar
# do log 100% completo.
sleep 15
rsync -avz "${LOGS_DIR}/" "${RESULTS_DIR}/logs/"
sleep 15
