#!/bin/bash
# =============================================================================
# job_g5k.sh — job OAR do EyeQ no Grid'5000 (cluster hydra, 1 x NVIDIA GH200, aarch64)
# =============================================================================
#
# Não rode direto: use hydra/submit_g5k.sh, que reserva o nó e passa as variáveis.
#
# Fluxo:
#   1. copia o repositório da home para /tmp do nó (sem as imagens)
#   2. copia só as imagens originais que aparecem nos CSVs de rótulos do EyeQ
#   3. checa CUDA/GPU (convolução, matmul e all_reduce via NCCL)
#   4. pré-processamento, que também gera o split patient_stratified
#   5. treino com torchrun, cortado com folga antes do fim do walltime
#   6. copia resultados para a home e apaga /tmp
#
# Ambiente Python: se a imagem Apptainer existir (SIF), usa ela. Senão, monta
# um venv com uv a partir de hydra/uv.lock em /tmp. Os dois usam o mesmo lock.
#
# Variáveis (todas opcionais, o submit_g5k.sh preenche):
#   RUN_ID          índice da execução (obrigatório)
#   REPO_SRC        repositório na home (padrão: ~/eyeq/EyeQ)
#   IMG_SRC         imagens originais (padrão: $REPO_SRC/EyeQ_preprocess/original_img)
#   RESULTS_ROOT    raiz dos resultados (padrão: ~/eyeq/results/g5k)
#   EXPERIMENT      subpasta dos resultados (padrão: stratified)
#   DATA_PROTOCOL   official | patient_stratified (padrão: patient_stratified)
#   GPUS            processos do torchrun (padrão: todas as GPUs do nó)
#   EPOCHS, BATCH_SIZE, CROP_SIZE   hiperparâmetros (BATCH_SIZE é por GPU; 1 GPU x 32 = batch global 32)
#   IMAGE_LIMIT     copia só as N primeiras imagens de cada CSV (teste rápido)
#   SIF             imagem Apptainer (padrão: ~/eyeq/containers/eyeq-cuda12.8-aarch64.sif)
#   ENV_MODE        auto | container | uv (padrão: auto)
#   MARGIN_MIN      minutos reservados no fim para copiar resultados (padrão: 20)
#   KEEP_WORKDIR    1 = não apaga /tmp no fim
# =============================================================================

set -uo pipefail

JOB_START=$(date +%s)

if [ -z "${RUN_ID:-}" ]; then
    echo "ERRO: RUN_ID não definido. Submeta com hydra/submit_g5k.sh." >&2
    exit 1
fi

REPO_SRC="${REPO_SRC:-$HOME/eyeq/EyeQ}"
IMG_SRC="${IMG_SRC:-${REPO_SRC}/EyeQ_preprocess/original_img}"
RESULTS_ROOT="${RESULTS_ROOT:-$HOME/eyeq/results/g5k}"
EXPERIMENT="${EXPERIMENT:-stratified}"
DATA_PROTOCOL="${DATA_PROTOCOL:-patient_stratified}"
EPOCHS="${EPOCHS:-45}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CROP_SIZE="${CROP_SIZE:-384}"
IMAGE_LIMIT="${IMAGE_LIMIT:-}"
SIF="${SIF:-$HOME/eyeq/containers/eyeq-cuda12.8-aarch64.sif}"
ENV_MODE="${ENV_MODE:-auto}"
MARGIN_MIN="${MARGIN_MIN:-20}"
KEEP_WORKDIR="${KEEP_WORKDIR:-0}"

RESULTS_DIR="${RESULTS_ROOT}/${EXPERIMENT}/${RUN_ID}"
WORK_ROOT="/tmp/eyeq_${RUN_ID}_${OAR_JOB_ID:-manual$$}"
BASE_DIR="${WORK_ROOT}/EyeQ"

mkdir -p "${RESULTS_DIR}"/{logs,preprocess,training}

# ---------- Prazo: o treino é cortado antes do fim do walltime ----------
# OAR mata o job no fim do walltime sem dar tempo de copiar nada. Calculamos o
# fim do job e deixamos MARGIN_MIN minutos para a cópia dos resultados.
WALLTIME_S="${OAR_JOB_WALLTIME_SECONDS:-}"
if [ -z "$WALLTIME_S" ] && [ -n "${OAR_JOB_WALLTIME:-}" ]; then
    IFS=: read -r wh wm ws <<< "$OAR_JOB_WALLTIME"
    WALLTIME_S=$((10#$wh * 3600 + 10#$wm * 60 + 10#${ws:-0}))
fi
if [ -n "$WALLTIME_S" ]; then
    JOB_END=$((JOB_START + WALLTIME_S))
    echo "Fim do job (walltime): $(date -d "@${JOB_END}" '+%Y-%m-%d %H:%M:%S %Z')"
else
    JOB_END=""
    echo "AVISO: walltime desconhecido; o treino não terá prazo." >&2
fi

seconds_left() {
    # segundos até o fim do job menos a margem; vazio se não há prazo
    [ -z "$JOB_END" ] && return 0
    echo $((JOB_END - MARGIN_MIN * 60 - $(date +%s)))
}

# ---------- Limpeza (sucesso, erro, oardel ou fim do prazo) ----------
TB_PID=""
copy_results() {
    [ -d "${BASE_DIR}/EyeQ_preprocess/metrics" ] && \
        rsync -a "${BASE_DIR}/EyeQ_preprocess/metrics/" "${RESULTS_DIR}/preprocess/"
    # Manifests gerados no pré-processamento: registram o split exato usado.
    if compgen -G "${BASE_DIR}/data/Label_EyeQ_*" >/dev/null; then
        mkdir -p "${RESULTS_DIR}/preprocess/manifests"
        rsync -a "${BASE_DIR}"/data/Label_EyeQ_* "${RESULTS_DIR}/preprocess/manifests/"
    fi
    [ -d "${BASE_DIR}/MCF_Net/result" ] && rsync -a "${BASE_DIR}/MCF_Net/result/" "${RESULTS_DIR}/training/"
    [ -d "${BASE_DIR}/MCF_Net/runs" ]   && rsync -a "${BASE_DIR}/MCF_Net/runs/"   "${RESULTS_DIR}/training/runs/"
}
cleanup() {
    local status=$?
    trap - EXIT
    [ -n "$TB_PID" ] && kill "$TB_PID" 2>/dev/null
    echo "Copiando resultados para ${RESULTS_DIR}..."
    copy_results
    if [ "$KEEP_WORKDIR" = "1" ]; then
        echo "KEEP_WORKDIR=1: cópia temporária mantida em ${WORK_ROOT}"
    else
        case "$WORK_ROOT" in
            /tmp/eyeq_*) rm -rf "$WORK_ROOT" ;;
        esac
    fi
    echo "Fim: $(date '+%Y-%m-%d %H:%M:%S %Z') (status ${status})"
    exit "$status"
}
trap cleanup EXIT
trap 'echo "Sinal recebido, encerrando..." >&2; exit 143' TERM INT

# ---------- Informações da execução ----------
# GPUs NVIDIA vistas pelo driver.
N_GPUS_NODE=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')
GPUS="${GPUS:-${N_GPUS_NODE}}"

echo "=== EyeQ hydra (Grid'5000) RUN_ID=${RUN_ID} ==="
echo "Nó: $(hostname) | OAR job: ${OAR_JOB_ID:-n/a} | GPUs no nó: ${N_GPUS_NODE} | GPUS=${GPUS}"
echo "Protocolo: ${DATA_PROTOCOL} | épocas: ${EPOCHS} | batch/GPU: ${BATCH_SIZE} | crop: ${CROP_SIZE}"
echo "Repositório: ${REPO_SRC} | imagens: ${IMG_SRC}"
echo "Trabalho: ${WORK_ROOT} | resultados: ${RESULTS_DIR}"
[ -n "$IMAGE_LIMIT" ] && echo "MODO TESTE: só ${IMAGE_LIMIT} imagens por CSV"

for path in "$REPO_SRC/hydra/pyproject.toml" "$REPO_SRC/data/Label_EyeQ_train.csv" \
            "$REPO_SRC/data/Label_EyeQ_test.csv" "$IMG_SRC/train" "$IMG_SRC/test"; do
    if [ ! -e "$path" ]; then
        echo "ERRO: não encontrado: ${path}" >&2
        exit 1
    fi
done

if git -C "$REPO_SRC" rev-parse --git-dir >/dev/null 2>&1; then
    {
        echo "data:   $(date '+%Y-%m-%d %H:%M:%S %Z')"
        echo "nó:     $(hostname) | OAR job ${OAR_JOB_ID:-n/a}"
        echo "origem: ${REPO_SRC}"
        echo "branch: $(git -C "$REPO_SRC" rev-parse --abbrev-ref HEAD)"
        echo "commit: $(git -C "$REPO_SRC" rev-parse HEAD)"
        echo
        echo "Alterações não commitadas (git status --short):"
        git -C "$REPO_SRC" status --short
    } > "${RESULTS_DIR}/logs/repo_info.txt"
    git -C "$REPO_SRC" diff HEAD > "${RESULTS_DIR}/logs/repo_diff.patch"
fi

# ---------- 1. Cópia do repositório (sem imagens) ----------
T0=$(date +%s)
mkdir -p "$BASE_DIR"
if ! rsync -a \
        --exclude='.git/' --exclude='.venv/' --exclude='__pycache__/' \
        --exclude='/EyeQ_preprocess/original_img/' \
        --exclude='/EyeQ_preprocess/train/' --exclude='/EyeQ_preprocess/test/' \
        --exclude='/EyeQ_preprocess/metrics/' \
        --exclude='/MCF_Net/result/' --exclude='/MCF_Net/runs/' \
        "${REPO_SRC}/" "${BASE_DIR}/"; then
    echo "ERRO: falha ao copiar o repositório." >&2
    exit 1
fi

# ---------- 2. Cópia das imagens usadas pelo EyeQ ----------
# original_img tem o Kaggle DR inteiro (~88 mil imagens, 90 GB), mas o EyeQ só
# rotula ~28,8 mil. As outras seriam pré-processadas e descartadas pelo
# filter_csv, então copiamos só as que estão nos CSVs. O resultado do treino é
# o mesmo; só as contagens em preprocess/*_metrics.txt ficam menores.
for split in train test; do
    list="${WORK_ROOT}/files_${split}.txt"
    python3 -c '
import csv, sys
limit = int(sys.argv[2]) if sys.argv[2] else None
with open(sys.argv[1]) as f:
    names = [row["image"] for row in csv.DictReader(f)]
print("\n".join(names[:limit]))
' "${BASE_DIR}/data/Label_EyeQ_${split}.csv" "$IMAGE_LIMIT" > "$list"
    mkdir -p "${BASE_DIR}/EyeQ_preprocess/original_img/${split}"
    echo "Copiando $(wc -l < "$list") imagens de ${split}..."
    if ! rsync -a --files-from="$list" "${IMG_SRC}/${split}/" \
            "${BASE_DIR}/EyeQ_preprocess/original_img/${split}/"; then
        echo "AVISO: rsync de ${split} terminou com erro (imagens faltando?)." >&2
    fi
done
echo "Cópia concluída em $(( $(date +%s) - T0 ))s ($(du -sh "$WORK_ROOT" | cut -f1))."

# ---------- Ambiente Python ----------
unset PYTHONPATH LD_LIBRARY_PATH

if [ "$ENV_MODE" = "auto" ]; then
    if [ -f "$SIF" ] && command -v apptainer >/dev/null 2>&1; then
        ENV_MODE=container
    else
        ENV_MODE=uv
    fi
fi

case "$ENV_MODE" in
    container)
        # Copia a imagem para o disco local: 8 processos lendo do NFS é lento.
        cp "$SIF" "${WORK_ROOT}/env.sif"
        # --nv injeta o driver NVIDIA do nó (libcuda, nvidia-smi). /tmp e a
        # home já são montados por padrão.
        RUN=(apptainer exec --nv "${WORK_ROOT}/env.sif")
        echo "Ambiente: Apptainer ${SIF} (sha256 $(sha256sum "$SIF" | cut -c1-16))"
        ;;
    uv)
        # Na home (NFS) há um uv por arquitetura: ~/.local/bin-aarch64 neste nó.
        [ -x "$HOME/.local/bin-$(uname -m)/uv" ] && export PATH="$HOME/.local/bin-$(uname -m):$PATH"
        if ! command -v uv >/dev/null 2>&1; then
            export PATH="$HOME/.local/bin:$PATH"
        fi
        if ! command -v uv >/dev/null 2>&1; then
            echo "uv não encontrado; instalando em ~/.local/bin..."
            curl -LsSf https://astral.sh/uv/0.12.0/install.sh | env UV_NO_MODIFY_PATH=1 sh || exit 1
        fi
        export UV_PROJECT_ENVIRONMENT="${WORK_ROOT}/venv"
        export UV_LINK_MODE=copy
        RUN=(uv run --frozen --project "${BASE_DIR}/hydra")
        echo "Ambiente: uv ($(uv --version)) em ${UV_PROJECT_ENVIRONMENT}"
        T0=$(date +%s)
        if ! "${RUN[@]}" python -c "pass"; then
            echo "ERRO: não foi possível montar o venv a partir de hydra/uv.lock." >&2
            exit 1
        fi
        echo "venv pronto em $(( $(date +%s) - T0 ))s."
        ;;
    *)
        echo "ERRO: ENV_MODE inválido: ${ENV_MODE} (auto, container ou uv)." >&2
        exit 1 ;;
esac

# ---------- 3. Checagem do CUDA ----------
echo "Checando CUDA/GPU..."
if ! "${RUN[@]}" python "${BASE_DIR}/hydra/check_cuda.py" --gpus "$GPUS"; then
    echo "ERRO: checagem do CUDA falhou." >&2
    exit 1
fi
if ! timeout 300 "${RUN[@]}" torchrun --standalone --nproc_per_node="$GPUS" \
        "${BASE_DIR}/hydra/check_cuda.py" --ddp; then
    echo "ERRO: teste de all_reduce (NCCL) falhou ou travou." >&2
        exit 1
fi

# ---------- 4. Pré-processamento ----------
echo "Iniciando o pré-processamento..."
T0=$(date +%s)
cd "${BASE_DIR}/EyeQ_preprocess"
if ! "${RUN[@]}" python EyeQ_process_main.py > "${RESULTS_DIR}/logs/preprocess.log" 2>&1; then
    echo "ERRO: o pré-processamento falhou (ver logs/preprocess.log)." >&2
    tail -20 "${RESULTS_DIR}/logs/preprocess.log" >&2
    exit 1
fi
echo "Pré-processamento concluído em $(( $(date +%s) - T0 ))s."
cat metrics/*_metrics.txt 2>/dev/null

# ---------- 5. Treinamento ----------
cd "${BASE_DIR}/MCF_Net"
"${RUN[@]}" tensorboard --logdir runs --host 0.0.0.0 --port 6006 \
    > "${RESULTS_DIR}/logs/tensorboard.log" 2>&1 &
TB_PID=$!
echo "TensorBoard: ssh -L 6006:$(hostname):6006 lyon.g5k  ->  http://localhost:6006"

TIMEOUT_ARGS=()
LEFT=$(seconds_left)
if [ -n "$LEFT" ]; then
    if [ "$LEFT" -le 600 ]; then
        echo "ERRO: sobram só ${LEFT}s antes da margem final; treino não iniciado." >&2
        exit 1
    fi
    TIMEOUT_ARGS=(timeout --signal=TERM --kill-after=120 "$LEFT")
    echo "Treino terá no máximo $((LEFT / 60)) min (corte às $(date -d "@$(( $(date +%s) + LEFT ))" '+%H:%M'))."
fi

echo "Iniciando o treinamento..."
T0=$(date +%s)
"${TIMEOUT_ARGS[@]}" "${RUN[@]}" torchrun \
    --standalone \
    --nproc_per_node="${GPUS}" \
    Main_EyeQuality_tuned_parallel.py \
    --data_protocol "$DATA_PROTOCOL" \
    --lr 0.00837248326772002 \
    --batch-size "$BATCH_SIZE" \
    --weight_decay 1.3866006245078523e-05 \
    --lr_scheduler cosine \
    --momentum 0.8007658696453809 \
    --warmup_epochs 5 \
    --epochs "$EPOCHS" \
    --crop_size "$CROP_SIZE"
TRAIN_STATUS=$?
echo "Treino terminou em $(( ($(date +%s) - T0) / 60 )) min com código ${TRAIN_STATUS}."

if [ "$TRAIN_STATUS" -eq 124 ] || [ "$TRAIN_STATUS" -eq 137 ]; then
    echo "AVISO: treino cortado pelo prazo. O melhor checkpoint até aqui está em training/," >&2
    echo "mas a avaliação final no test não rodou." >&2
fi
exit "$TRAIN_STATUS"
