#!/bin/bash
# =============================================================================
# submit_single_experiment.sh
# =============================================================================
#
# O QUE ESTE SCRIPT FAZ
#   1. Descobre o próximo índice de execução (01, 02, 03, ...) em
#      ~/eyeq/results/single/stratified/
#   2. Cria na home a pasta da execução, com preprocess/, training/ e logs/.
#   3. Submete o train_job_single.sbatch, que no nó de computação:
#        a. copia o repositório da home (~/eyeq/EyeQ) para uma pasta exclusiva
#           do job em /<basedir>/eadbem/jobs/single_<ID>_<jobid>/EyeQ
#        b. roda SEMPRE o pré-processamento, do zero, sobre essa cópia;
#        c. roda o treinamento (torchrun) com o TensorBoard em paralelo;
#        d. copia métricas e resultados de volta para a home;
#        e. apaga a cópia temporária (a não ser que você use --keep).
#   Os logs do SLURM (.out/.err) são escritos direto na home.
#
# ATENÇÃO: a cópia acontece quando o job COMEÇA a rodar, não no momento da
#   submissão. Se o job ficar na fila e você editar o código na home nesse
#   meio tempo, o job vai usar a versão editada. O commit e as alterações não
#   commitadas que foram de fato usados ficam registrados em
#   <execução>/logs/repo_info.txt e <execução>/logs/repo_diff.patch.
#
# -----------------------------------------------------------------------------
# COMO MONTAR O COMANDO
# -----------------------------------------------------------------------------
#   ./submit_single_experiment.sh -m <partição> [opções]
#
#   Obrigatório:
#     -m, --machine <partição>  Partição do SLURM onde o job vai rodar.
#                               Para ver as partições: sinfo
#
#   Opcionais:
#     -n, --nodelist <nó>       Força um nó específico da partição.
#                               Sem isso, o SLURM escolhe qualquer nó livre.
#                               Para ver os nós e o estado: sinfo -N -p <partição>
#     -d, --basedir <dir>       Onde fica a cópia temporária do repositório:
#                               scratch, ssd ou ssd2 (padrão: scratch).
#                               O diretório /<basedir> precisa existir no nó
#                               escolhido; o job falha logo no início se não
#                               existir.
#     -g, --gpus <N>            Número de GPUs reservadas (padrão: 1). Também
#                               define o --nproc_per_node do torchrun.
#     -t, --tb-host <ip>        IP/host onde o TensorBoard escuta
#                               (padrão: 192.168.30.51). Porta: 8080.
#     -r, --repo <caminho>      Repositório de origem a ser copiado
#                               (padrão: ~/eyeq/EyeQ).
#     -e, --experiment <nome>   Nome da pasta dentro de results/single/ onde a
#                               execução é salva (padrão: stratified). Ex.:
#                               -e resolution -> results/single/resolution/<ID>
#     -k, --keep                Não apaga a cópia temporária no fim do job
#                               (útil para depurar). Lembre de apagar depois.
#     -f, --folds <spec>        Cross-validation: submete um job array com um
#                               treino por fold (0 a 8). Ex.: 0-8 (todos), 3,
#                               0,4,7 ou 0-2,5. Força DATA_PROTOCOL=patient_stratified.
#                               Cada fold é uma alocação separada.
#     -p, --max-parallel <N>    Com --folds: no máximo N folds rodando ao
#                               mesmo tempo (padrão: sem limite).
#     -i, --run-id <ID>         Com --folds: reaproveita uma execução que já
#                               existe (ex.: 07) em vez de criar um índice novo.
#                               Útil para refazer um fold que falhou.
#     -h, --help                Mostra a ajuda resumida.
#
#   Todas as opções aceitam as formas "-m poti", "--machine poti" e
#   "--machine=poti", e podem vir em qualquer ordem.
#
#   Roteiro rápido para montar o comando:
#     1. Escolha a partição (-m). É a única coisa obrigatória.
#     2. Precisa de um nó específico (ex.: o único com ssd2, ou um nó ARM)?
#        Adicione -n <nó>.
#     3. Quer a cópia em outro disco que não o scratch? Adicione -d ssd|ssd2.
#     4. Vai usar mais de uma GPU? Adicione -g <N>.
#
# EXEMPLOS
#   # O mais simples: qualquer nó livre da poti, cópia no scratch, 1 GPU
#   ./submit_single_experiment.sh -m poti
#
#   # Nó poti5, cópia no ssd2
#   ./submit_single_experiment.sh -m poti -n poti5 -d ssd2
#
#   # 4 GPUs no mesmo nó (torchrun com 4 processos)
#   ./submit_single_experiment.sh -m poti -g 4
#
#   # Nó ARM (a arquitetura é detectada sozinha pelo job)
#   ./submit_single_experiment.sh -m grace -n grace1 -d ssd
#
#   # Salva em results/single/resolution/ em vez de stratified/
#   ./submit_single_experiment.sh -m poti -e resolution
#
#   # Mantém a cópia temporária para inspecionar depois
#   ./submit_single_experiment.sh -m poti -n poti5 --keep
#
#   # Variáveis de ambiente são repassadas ao job (--export=ALL), ex.:
#   DATA_PROTOCOL=official ./submit_single_experiment.sh -m poti
#   EPOCHS=1 ./submit_single_experiment.sh -m poti     # teste rápido
#
#   # Cross-validation 9-fold, no máximo 3 folds rodando ao mesmo tempo
#   ./submit_single_experiment.sh -m poti -e kfold --folds 0-8 -p 3
#
#   # O fold 3 da execução 02 falhou: refaz só ele, na mesma pasta
#   ./submit_single_experiment.sh -m poti -e kfold --folds 3 --run-id 02
#
# CROSS-VALIDATION
#   O test (10%) é fixo e igual para todos os folds. Os outros 90% são
#   divididos em 9 grupos por paciente; o fold k usa o grupo k como validation
#   e o resto como train (80/10/10). O fold 0 é idêntico ao split
#   patient_stratified sem fold. Uma execução (<ID>) é um CV inteiro:
#     <ID>/fold_<k>/{preprocess,training,logs}
#   Para resumir os folds (média ± desvio no test):
#     python MCF_Net/aggregate_folds.py ~/eyeq/results/single/<experimento>/<ID>
#
# DEPOIS DE SUBMETER
#   squeue -u $USER                       # ver se está na fila / rodando
#   tail -f ~/eyeq/results/single/<experimento>/<ID>/logs/*.out   # acompanhar
#   http://<tb-host>:8080                 # TensorBoard (enquanto treina)
#   scancel <jobid>                       # cancelar (a limpeza roda mesmo assim)
#
#   Resultados em ~/eyeq/results/single/<experimento>/<ID>/ (com --folds,
#   em <ID>/fold_<k>/):
#     preprocess/       métricas do pré-processamento
#     training/         saída de MCF_Net/result
#     training/runs/    logs do TensorBoard
#     logs/             .out/.err do SLURM + repo_info.txt + repo_diff.patch
#
# Hiperparâmetros do treino: edite o bloco do torchrun no
# train_job_single.sbatch.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat >&2 <<USAGE
Uso: $0 -m <partição> [-n <nó>] [-d scratch|ssd|ssd2] [-g <N>] [-t <ip>] [-r <repo>] [-e <experimento>] [-k]
          [-f <folds> [-p <N>] [-i <ID>]]

  -m, --machine   Partição do SLURM (obrigatório). Ex: poti
  -n, --nodelist  Nó específico dentro da partição (opcional).
                  Se omitido, o SLURM escolhe qualquer nó livre da partição.
  -d, --basedir   Onde fica a cópia temporária: scratch, ssd ou ssd2
                  (padrão: scratch)
  -g, --gpus      Número de GPUs a reservar no nó (padrão: 1). Também define
                  o --nproc_per_node do torchrun — os dois sempre andam juntos.
  -t, --tb-host   IP/host onde o TensorBoard vai escutar (padrão: 192.168.30.51)
  -r, --repo      Repositório de origem (padrão: ~/eyeq/EyeQ)
  -e, --experiment Pasta dentro de results/single/ (padrão: stratified)
  -k, --keep      Não apaga a cópia temporária ao fim do job
  -f, --folds     Cross-validation: folds a rodar, ex. 0-8, 3 ou 0,4 (job array)
  -p, --max-parallel  Com --folds: máximo de folds rodando ao mesmo tempo
  -i, --run-id    Com --folds: reaproveita a execução <ID> existente

Veja o cabeçalho deste script para o tutorial completo e exemplos.
USAGE
    exit 1
}

MACHINE=""
NODELIST=""
BASEDIR="scratch"
TB_HOST="192.168.30.51"
GPUS=1
REPO_SRC="${HOME}/eyeq/EyeQ"
KEEP_WORKDIR=0
EXPERIMENT="stratified"
FOLDS=""
MAX_PARALLEL=""
REUSE_RUN_ID=""
N_CV_FOLDS=9   # tem que bater com N_CV_FOLDS em create_patient_stratified_splits.py

while [ $# -gt 0 ]; do
    case "$1" in
        -m|--machine)   MACHINE="$2"; shift 2 ;;
        --machine=*)    MACHINE="${1#*=}"; shift ;;
        -n|--nodelist)  NODELIST="$2"; shift 2 ;;
        --nodelist=*)   NODELIST="${1#*=}"; shift ;;
        -d|--basedir)   BASEDIR="$2"; shift 2 ;;
        --basedir=*)    BASEDIR="${1#*=}"; shift ;;
        -t|--tb-host)   TB_HOST="$2"; shift 2 ;;
        --tb-host=*)    TB_HOST="${1#*=}"; shift ;;
        -g|--gpus)      GPUS="$2"; shift 2 ;;
        --gpus=*)       GPUS="${1#*=}"; shift ;;
        -r|--repo)      REPO_SRC="$2"; shift 2 ;;
        --repo=*)       REPO_SRC="${1#*=}"; shift ;;
        -e|--experiment) EXPERIMENT="$2"; shift 2 ;;
        --experiment=*) EXPERIMENT="${1#*=}"; shift ;;
        -k|--keep)      KEEP_WORKDIR=1; shift ;;
        -f|--folds)     FOLDS="$2"; shift 2 ;;
        --folds=*)      FOLDS="${1#*=}"; shift ;;
        -p|--max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
        --max-parallel=*)  MAX_PARALLEL="${1#*=}"; shift ;;
        -i|--run-id)    REUSE_RUN_ID="$2"; shift 2 ;;
        --run-id=*)     REUSE_RUN_ID="${1#*=}"; shift ;;
        -h|--help)      usage ;;
        *)
            echo "Opção desconhecida: $1" >&2
            usage ;;
    esac
done

if [ -z "$MACHINE" ]; then
    echo "ERRO: máquina não especificada. Use -m/--machine." >&2
    usage
fi

case "$BASEDIR" in
    scratch|ssd|ssd2) ;;
    *)
        echo "ERRO: basedir inválido: '${BASEDIR}'. Use scratch, ssd ou ssd2." >&2
        exit 1 ;;
esac

if ! [[ "$EXPERIMENT" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "ERRO: experimento inválido: '${EXPERIMENT}'. Use letras, números, '.', '_' ou '-'." >&2
    exit 1
fi

if ! [[ "$GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERRO: gpus inválido: '${GPUS}'. Use um inteiro positivo." >&2
    exit 1
fi

# Expande a lista de folds ("0-2,5" -> "0 1 2 5") para validar e criar as pastas.
FOLD_LIST=()
if [ -n "$FOLDS" ]; then
    IFS=',' read -ra FOLD_PARTS <<< "$FOLDS"
    for part in "${FOLD_PARTS[@]}"; do
        if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            for ((k = 10#${BASH_REMATCH[1]}; k <= 10#${BASH_REMATCH[2]}; k++)); do FOLD_LIST+=("$k"); done
        elif [[ "$part" =~ ^[0-9]+$ ]]; then
            FOLD_LIST+=("$((10#$part))")
        else
            echo "ERRO: folds inválido: '${FOLDS}'. Use ex.: 0-8, 3 ou 0,4,7." >&2
            exit 1
        fi
    done
    for k in "${FOLD_LIST[@]}"; do
        if [ "$k" -ge "$N_CV_FOLDS" ]; then
            echo "ERRO: fold ${k} não existe (folds vão de 0 a $((N_CV_FOLDS - 1)))." >&2
            exit 1
        fi
    done
    if [ "${#FOLD_LIST[@]}" -eq 0 ]; then
        echo "ERRO: folds inválido: '${FOLDS}' não contém nenhum fold." >&2
        exit 1
    fi

    if [ -n "${DATA_PROTOCOL:-}" ] && [ "$DATA_PROTOCOL" != "patient_stratified" ]; then
        echo "ERRO: --folds exige DATA_PROTOCOL=patient_stratified (recebido: ${DATA_PROTOCOL})." >&2
        exit 1
    fi
    DATA_PROTOCOL=patient_stratified
fi

if [ -n "$MAX_PARALLEL" ]; then
    if [ -z "$FOLDS" ]; then
        echo "ERRO: --max-parallel só faz sentido com --folds." >&2
        exit 1
    fi
    if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERRO: max-parallel inválido: '${MAX_PARALLEL}'. Use um inteiro positivo." >&2
        exit 1
    fi
fi

if [ -n "$REUSE_RUN_ID" ]; then
    if [ -z "$FOLDS" ]; then
        echo "ERRO: --run-id só faz sentido com --folds." >&2
        exit 1
    fi
    if ! [[ "$REUSE_RUN_ID" =~ ^[0-9]+$ ]]; then
        echo "ERRO: run-id inválido: '${REUSE_RUN_ID}'. Use o número da execução, ex.: 07." >&2
        exit 1
    fi
fi

# Caminho absoluto do repositório (o job roda em outro diretório).
if [ ! -d "$REPO_SRC" ]; then
    echo "ERRO: repositório não encontrado: '${REPO_SRC}'." >&2
    exit 1
fi
REPO_SRC="$(cd "$REPO_SRC" && pwd)"

for sub in EyeQ_preprocess MCF_Net; do
    if [ ! -d "${REPO_SRC}/${sub}" ]; then
        echo "ERRO: '${REPO_SRC}' não parece ser o repositório EyeQ (falta ${sub}/)." >&2
        exit 1
    fi
done

BASE_RESULTS=/home/users/eadbem/eyeq/results/single/${EXPERIMENT}
mkdir -p "$BASE_RESULTS"

if [ -n "$REUSE_RUN_ID" ]; then
    RUN_ID=$(printf "%02d" "$((10#$REUSE_RUN_ID))")
    if [ ! -d "${BASE_RESULTS}/${RUN_ID}" ]; then
        echo "ERRO: execução ${BASE_RESULTS}/${RUN_ID} não existe." >&2
        exit 1
    fi
else
    LAST_ID=$(find "$BASE_RESULTS" -maxdepth 1 -mindepth 1 -type d -name '[0-9][0-9]' -printf '%f\n' \
                | sort -n | tail -1)

    if [ -z "$LAST_ID" ]; then
        NEXT_ID=1
    else
        NEXT_ID=$((10#$LAST_ID + 1))
    fi

    # mkdir sem -p é atômico: se duas submissões rodarem ao mesmo tempo, só uma
    # consegue criar a pasta; a outra tenta o próximo número.
    while ! mkdir "${BASE_RESULTS}/$(printf "%02d" "$NEXT_ID")" 2>/dev/null; do
        NEXT_ID=$((NEXT_ID + 1))
    done
    RUN_ID=$(printf "%02d" "$NEXT_ID")
fi

echo "Execução single: RUN_ID=${RUN_ID} (experimento=${EXPERIMENT}, machine=${MACHINE}, basedir=${BASEDIR}, gpus=${GPUS})"
echo "Repositório de origem: ${REPO_SRC}"

EXPORTS="ALL,RUN_ID=${RUN_ID},BASEDIR=${BASEDIR},TB_HOST=${TB_HOST},GPUS=${GPUS},REPO_SRC=${REPO_SRC},KEEP_WORKDIR=${KEEP_WORKDIR},EXPERIMENT=${EXPERIMENT}"

if [ -n "$FOLDS" ]; then
    # O SLURM não cria a pasta dos logs, então cada fold_<k>/logs já tem que existir.
    for k in "${FOLD_LIST[@]}"; do
        mkdir -p "${BASE_RESULTS}/${RUN_ID}/fold_${k}"/{preprocess,training,logs}
    done
    ARRAY_SPEC="$(IFS=,; echo "${FOLD_LIST[*]}")${MAX_PARALLEL:+%${MAX_PARALLEL}}"
    echo "Cross-validation: folds ${FOLD_LIST[*]} (array=${ARRAY_SPEC})"

    SBATCH_ARGS=(
        --job-name="hcpa_eyeq_single_${RUN_ID}_cv"
        --array="${ARRAY_SPEC}"
        --output="${BASE_RESULTS}/${RUN_ID}/fold_%a/logs/%x_%A_%a.out"
        --error="${BASE_RESULTS}/${RUN_ID}/fold_%a/logs/%x_%A_%a.err"
        --export="${EXPORTS},DATA_PROTOCOL=${DATA_PROTOCOL}"
    )
else
    mkdir -p "${BASE_RESULTS}/${RUN_ID}"/{preprocess,training,logs}

    SBATCH_ARGS=(
        --job-name="hcpa_eyeq_single_${RUN_ID}"
        --output="${BASE_RESULTS}/${RUN_ID}/logs/%x_%j.out"
        --error="${BASE_RESULTS}/${RUN_ID}/logs/%x_%j.err"
        --export="${EXPORTS}"
    )
fi

SBATCH_ARGS+=(
    --partition="${MACHINE}"
    --gres="gpu:${GPUS}"
)

if [ -n "$NODELIST" ]; then
    SBATCH_ARGS+=(--nodelist="${NODELIST}")
fi

sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/train_job_single.sbatch"

echo "Job submetido. Resultados irão para: ${BASE_RESULTS}/${RUN_ID}/"
