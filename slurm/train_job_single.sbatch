#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
# job-name, output, error, partition, gres e nodelist são definidos
# dinamicamente pelo submit_single_experiment.sh (via linha de comando do sbatch).
#
# Fluxo do job:
#   1. copia o repositório da home para /<BASEDIR>/eadbem/jobs/<job>/EyeQ
#   2. pré-processamento (sempre, do zero)
#   3. treinamento + TensorBoard
#   4. copia resultados para a home
#   5. apaga a cópia temporária (exceto com KEEP_WORKDIR=1)

set -uo pipefail
export DATA_PROTOCOL=${DATA_PROTOCOL:-official}

if [ -z "${RUN_ID:-}" ]; then
    echo "ERRO: RUN_ID não definido. Não rode este script diretamente com sbatch;" >&2
    echo "use ./submit_single_experiment.sh, que define o índice e submete o job." >&2
    exit 1
fi

# ---------- Detecção de arquitetura (ARM/aarch64 vs x86_64) ----------
# grace/grace1 são nós ARM; o resto do cluster (ex: poti) é x86_64. Em vez de
# fixar isso ao nome da máquina, detectamos em tempo de execução com `uname -m`,
# assim o mesmo script funciona em qualquer nó sem precisar de flag extra.
ARCH="$(uname -m)"
UV_RUN_ARGS=()

if [ "$ARCH" = "aarch64" ]; then
    echo "Arquitetura detectada: aarch64 (ARM) — configurando ambiente ARM"

    export PYENV_ROOT="$HOME/.pyenv-aarch64"
    export PATH="$HOME/.pyenv-aarch64/bin:$HOME/.local/bin/aarch64:$PATH"

    if command -v pyenv >/dev/null 2>&1; then
        eval "$(pyenv init -)"
    else
        echo "AVISO: pyenv não encontrado em \$PYENV_ROOT/bin após ajustar o PATH." >&2
        echo "Confira se ~/.pyenv-aarch64 existe neste nó (${HOSTNAME:-desconhecido})." >&2
    fi

    # .python-version do projeto ainda aponta pra 3.7.17, mas o ambiente ARM
    # usa Python 3.10. Forçamos 3.10 aqui pra não deixar o uv tentar resolver
    # de novo com 3.7 dentro do job.
    UV_RUN_ARGS=(--python 3.10)

    echo "uv em uso: $(command -v uv || echo 'não encontrado no PATH')"
else
    echo "Arquitetura detectada: ${ARCH} — usando ambiente padrão (x86_64)"
fi

# ---------- Configuração de caminhos ----------
BASEDIR="${BASEDIR:-scratch}"
case "$BASEDIR" in
    scratch|ssd|ssd2) ;;
    *)
        echo "ERRO: BASEDIR inválido: '${BASEDIR}'. Use scratch, ssd ou ssd2." >&2
        exit 1 ;;
esac

if [ ! -d "/${BASEDIR}" ]; then
    echo "ERRO: /${BASEDIR} não existe no nó ${HOSTNAME:-desconhecido}." >&2
    echo "Escolha outro --basedir ou outro nó." >&2
    exit 1
fi

REPO_SRC="${REPO_SRC:-$HOME/eyeq/EyeQ}"
KEEP_WORKDIR="${KEEP_WORKDIR:-0}"
RESULTS_DIR=/home/users/eadbem/eyeq/results/single/stratified/${RUN_ID}

# Pasta exclusiva por job: jobs simultâneos no mesmo nó não se atropelam.
JOB_TAG="single_${RUN_ID}_${SLURM_JOB_ID:-manual$$}"
WORK_ROOT="/${BASEDIR}/eadbem/jobs/${JOB_TAG}"
BASE_DIR="${WORK_ROOT}/EyeQ"

# Ambientes virtuais persistentes, um por arquitetura e por subprojeto, fora da
# cópia temporária. Assim o .venv não é copiado da home a cada job (o da home
# pode nem ser da arquitetura certa) e não é recriado do zero toda vez: o uv
# só sincroniza o que mudou. O primeiro job em cada disco/arquitetura demora
# mais porque monta o ambiente.
VENV_ROOT="/${BASEDIR}/eadbem/venvs"
export UV_LINK_MODE=copy   # cache do uv na home e venv em outro disco: sem hardlink

echo "=== Execução single RUN_ID=${RUN_ID} ==="
echo "Nó: ${HOSTNAME:-desconhecido} | Job: ${SLURM_JOB_ID:-n/a} | GPUs: ${GPUS:-1}"
echo "Repositório de origem: ${REPO_SRC}"
echo "Cópia de trabalho: ${BASE_DIR}"
echo "Resultados serão salvos em: ${RESULTS_DIR}"

# ---------- Limpeza (roda sempre: sucesso, erro, scancel ou fim do tempo) ----------
TB_PID=""
cleanup() {
    local status=$?
    if [ -n "$TB_PID" ]; then
        kill "$TB_PID" 2>/dev/null
    fi
    if [ "$KEEP_WORKDIR" = "1" ]; then
        echo "KEEP_WORKDIR=1: cópia temporária mantida em ${WORK_ROOT}"
    else
        case "$WORK_ROOT" in
            /*/eadbem/jobs/single_*)
                echo "Removendo cópia temporária ${WORK_ROOT}"
                rm -rf "$WORK_ROOT" ;;
        esac
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'echo "Sinal recebido, encerrando..." >&2; exit 143' TERM INT

# ---------- Cópia do repositório ----------
if [ ! -d "$REPO_SRC" ]; then
    echo "ERRO: repositório de origem não encontrado: ${REPO_SRC}" >&2
    exit 1
fi

mkdir -p "${RESULTS_DIR}/logs" "${RESULTS_DIR}/preprocess" "${RESULTS_DIR}/training"

# Registra exatamente qual versão do código foi usada.
if git -C "$REPO_SRC" rev-parse --git-dir >/dev/null 2>&1; then
    {
        echo "data:   $(date '+%Y-%m-%d %H:%M:%S')"
        echo "origem: ${REPO_SRC}"
        echo "branch: $(git -C "$REPO_SRC" rev-parse --abbrev-ref HEAD)"
        echo "commit: $(git -C "$REPO_SRC" rev-parse HEAD)"
        echo
        echo "Alterações não commitadas (git status --short):"
        git -C "$REPO_SRC" status --short
    } > "${RESULTS_DIR}/logs/repo_info.txt"
    git -C "$REPO_SRC" diff HEAD > "${RESULTS_DIR}/logs/repo_diff.patch"
fi

# Itens que NÃO são copiados. Os caminhos que começam com / são relativos à
# raiz do repositório. Saídas de execuções antigas ficam de fora para não
# contaminar os resultados deste job. Adicione aqui o que mais não precisar
# ir para o nó (ex.: pastas de saída do pré-processamento rodado na home).
RSYNC_EXCLUDES=(
    --exclude='.git/'
    --exclude='.venv/'
    --exclude='__pycache__/'
    --exclude='/EyeQ_preprocess/metrics/'
    --exclude='/MCF_Net/result/'
    --exclude='/MCF_Net/runs/'
)

echo "Copiando repositório para ${BASE_DIR}..."
mkdir -p "$BASE_DIR"
if ! rsync -a "${RSYNC_EXCLUDES[@]}" "${REPO_SRC}/" "${BASE_DIR}/"; then
    echo "ERRO: falha ao copiar o repositório para ${BASE_DIR}." >&2
    exit 1
fi
echo "Cópia concluída ($(du -sh "$BASE_DIR" | cut -f1))."

# ---------- Pré-processamento (sempre executado) ----------
echo "Iniciando o pré-processamento..."
cd "${BASE_DIR}/EyeQ_preprocess"
export UV_PROJECT_ENVIRONMENT="${VENV_ROOT}/eyeq-preprocess-${ARCH}"

if ! uv run "${UV_RUN_ARGS[@]}" EyeQ_process_main.py; then
    echo "ERRO: o pré-processamento falhou. Treinamento não será executado." >&2
    [ -d metrics ] && rsync -a metrics/ "${RESULTS_DIR}/preprocess/"
    exit 1
fi
rsync -a metrics/ "${RESULTS_DIR}/preprocess/"

# ---------- Treinamento ----------
echo "Iniciando o treinamento..."
cd "${BASE_DIR}/MCF_Net"
export UV_PROJECT_ENVIRONMENT="${VENV_ROOT}/eyeq-mcfnet-${ARCH}"

# Sincroniza o ambiente antes de subir TensorBoard e torchrun em paralelo,
# para os dois não tentarem instalar pacotes ao mesmo tempo.
if ! uv run "${UV_RUN_ARGS[@]}" python -c "pass"; then
    echo "ERRO: não foi possível preparar o ambiente do MCF_Net." >&2
    exit 1
fi

TB_HOST="${TB_HOST:-192.168.30.51}"
uv run "${UV_RUN_ARGS[@]}" tensorboard --logdir runs --host "$TB_HOST" --port 8080 &
TB_PID=$!

uv run "${UV_RUN_ARGS[@]}" torchrun \
    --standalone \
    --nproc_per_node="${GPUS:-1}" \
    Main_EyeQuality_tuned_parallel.py \
    --lr 0.00837248326772002 \
    --batch-size 8 \
    --weight_decay 1.3866006245078523e-05 \
    --lr_scheduler cosine \
    --momentum 0.8007658696453809 \
    --warmup_epochs 5 \
    --data_protocol patient_stratified \
    --epochs 45
TRAIN_STATUS=$?

# Encerra o TensorBoard (senão fica pendurado após o job acabar)
kill "$TB_PID" 2>/dev/null
TB_PID=""

# Copia o que houver, mesmo se o treino falhou (resultados parciais ajudam a depurar).
[ -d result ] && rsync -a result/ "${RESULTS_DIR}/training/"
[ -d runs ]   && rsync -a runs/   "${RESULTS_DIR}/training/runs/"

if [ "$TRAIN_STATUS" -ne 0 ]; then
    echo "ERRO: o treinamento terminou com código ${TRAIN_STATUS}." >&2
    exit "$TRAIN_STATUS"
fi

echo "=== Execução single RUN_ID=${RUN_ID} finalizada ==="
