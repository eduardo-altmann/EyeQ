#!/bin/bash
#SBATCH --partition=draco
#SBATCH --nodelist=draco1          # garante que o no com o repo/dataset esta sempre na alocacao
#SBATCH --nodes=2                  # draco1 + demais nos (script assume 2 no total, ver nota abaixo)
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
# job-name, output e error sao definidos dinamicamente pelo submit_multinode_experiment.sh

set -uo pipefail

if [ -z "${RUN_ID:-}" ]; then
    echo "ERRO: RUN_ID nao definido. Nao rode este script diretamente com sbatch;" >&2
    echo "use ./submit_multinode_experiment.sh, que define o indice e submete o job." >&2
    exit 1
fi

# --- no de origem: unico que tem o repositorio + dataset completos hoje ---
SOURCE_NODE=draco1
REPO_DIR=/scratch/eadbem/EyeQ
export SOURCE_NODE REPO_DIR

RESULTS_DIR=/home/users/eadbem/eyeq/results/multinode/${RUN_ID}
echo "=== Execucao RUN_ID=${RUN_ID} ==="
echo "Resultados serao salvos em: ${RESULTS_DIR}"

# ---------- Atualizacao do repositorio (sempre no SOURCE_NODE) ----------
echo "Atualizando repositorio em ${SOURCE_NODE}..."
srun --nodes=1 --ntasks=1 -w "$SOURCE_NODE" bash -c "cd '$REPO_DIR' && git pull"

# ---------- Propaga repo+dataset do SOURCE_NODE para os demais nos ----------
# rsync so transfere o que mudou -- a primeira execucao e cara (copia tudo),
# execucoes seguintes sao rapidas (so o diff). Requer SSH sem senha entre os
# nos de compute para o usuario eadbem (confirme se ja esta configurado).
echo "Sincronizando repositorio/dataset para os demais nos..."
srun --ntasks-per-node=1 --exclude="$SOURCE_NODE" bash -c '
    mkdir -p "$REPO_DIR"
    rsync -avz --delete \
        --exclude="MCF_Net/result/" --exclude="MCF_Net/runs/" \
        "${SOURCE_NODE}:${REPO_DIR}/" "${REPO_DIR}/"
'

# ---------- Rendezvous: o master e sempre o SOURCE_NODE ----------
export MASTER_ADDR=$SOURCE_NODE
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=eno1     # confirme a interface correta com `ip addr`
export NCCL_DEBUG=INFO

echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT  NNODES=$SLURM_NNODES"

# ---------- Treinamento ----------
# NODE_RANK e forcado via comparacao de hostname (nao usa $SLURM_NODEID) para
# garantir que rank 0 caia sempre no SOURCE_NODE -- assim sabemos de antemao
# onde os resultados/checkpoints vao ser escritos. So funciona para 2 nos
# (SOURCE_NODE=0, o outro=1); para >2 nos essa logica precisa ser refeita.
echo "Iniciando o treinamento multinode..."

srun --ntasks-per-node=1 bash -c '
    NODE_RANK=0
    [ "$(hostname -s)" = "$SOURCE_NODE" ] || NODE_RANK=1

    cd "$REPO_DIR/MCF_Net"

    # TensorBoard so precisa rodar uma vez, no no que escreve runs/ (rank 0 / SOURCE_NODE)
    if [ "$NODE_RANK" -eq 0 ]; then
        uv run tensorboard --logdir runs --host 143.54.49.79 --port 6006 &
        TB_PID=$!
    fi

    uv run torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=1 \
        --node_rank=$NODE_RANK \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        Main_EyeQuality_tuned_parallel.py \
        --lr 0.00837248326772002 \
        --batch-size 8 \
        --weight_decay 1.3866006245078523e-05 \
        --lr_scheduler cosine \
        --momentum 0.8007658696453809 \
        --warmup_epochs 5 \
        --epochs 60

    if [ "$NODE_RANK" -eq 0 ]; then
        kill "$TB_PID" 2>/dev/null
    fi
'

# ---------- Coleta de resultados (sempre do SOURCE_NODE, que e sempre rank 0) ----------
echo "Copiando resultados de ${SOURCE_NODE} para ${RESULTS_DIR}..."
srun --nodes=1 --ntasks=1 -w "$SOURCE_NODE" bash -c "
    rsync -avz '$REPO_DIR/MCF_Net/result/' '$RESULTS_DIR/training/'
    rsync -avz '$REPO_DIR/MCF_Net/runs/' '$RESULTS_DIR/training/runs/'
"

echo "=== Execucao RUN_ID=${RUN_ID} finalizada ==="