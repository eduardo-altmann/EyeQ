#!/bin/bash
# =============================================================================
# submit_g5k.sh — submete o pipeline EyeQ no Grid'5000 (Lyon, cluster hydra)
# =============================================================================
#
# Rode no frontend de Lyon (ssh lyon.g5k), a partir de qualquer pasta:
#   ~/eyeq/EyeQ/hydra/submit_g5k.sh                 # job noturno (19:00 -> 09:00)
#   ~/eyeq/EyeQ/hydra/submit_g5k.sh --start "2026-11-06 19:00:00"   # reserva com hora fixa
#   ~/eyeq/EyeQ/hydra/submit_g5k.sh --test          # teste sem night, 59 min, assim que houver nó
#   ~/eyeq/EyeQ/hydra/submit_g5k.sh --test --start "2026-11-11 18:15:00" --end-time 18:55   # teste com fim garantido
#
# Padrão: job com os tipos night e exotic (o hydra é exótico) num nó hydra
# inteiro (1 x GH200, aarch64). No Grid'5000 o tipo night faz o OAR só iniciar o job a
# partir das 19:00 e exigir que termine até as 09:00 da manhã seguinte. O OAR
# escolhe a hora de início dentro dessa janela, conforme os nós liberarem. Se
# nenhum nó couber na janela, o job não roda: confira com oarstat -u.
#
# Com --start a reserva é antecipada (advance reservation) na hora dada, com
# fim às --end-time (padrão 08:50). O OAR responde na hora se o nó está livre.
#
# Em todos os modos o job corta o treino MARGIN_MIN minutos (padrão 20) antes
# do fim do walltime, para dar tempo de copiar os resultados para a home.
#
# Opções:
#   --walltime H:MM     walltime do job noturno (padrão 10:00; máximo 14:00)
#   --start "AAAA-MM-DD HH:MM:SS"   reserva com início fixo (horário de Paris)
#   --end-time HH:MM    fim da reserva com --start (padrão 08:50)
#   --test              sem night; EPOCHS=2, IMAGE_LIMIT=1500, walltime 0:59 (ou --start/--end-time)
#   --host <nó>         força um nó, ex.: hydra-3
#   -e <experimento>    pasta em ~/eyeq/results/g5k/ (padrão: stratified)
#
# Variáveis repassadas ao job (ver hydra/job_g5k.sh): EPOCHS, BATCH_SIZE,
# CROP_SIZE, GPUS, DATA_PROTOCOL, IMAGE_LIMIT, ENV_MODE, SIF, MARGIN_MIN,
# KEEP_WORKDIR, REPO_SRC, IMG_SRC. Ex.: EPOCHS=60 ./submit_g5k.sh
#
# Acompanhar:
#   oarstat -u                                       # estado do job
#   tail -f ~/eyeq/results/g5k/<exp>/<ID>/logs/OAR.*.stdout
#   oardel <job_id>                                  # cancelar
# =============================================================================

set -euo pipefail
export TZ=Europe/Paris

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_SRC="${REPO_SRC:-$(dirname "$SCRIPT_DIR")}"
RESULTS_ROOT="${RESULTS_ROOT:-$HOME/eyeq/results/g5k}"
EXPERIMENT="stratified"
TEST=0
TEST_EXPERIMENT="test"
WALLTIME_SET=0
START=""
END_TIME="08:50"
WALLTIME_NIGHT="10:00"
HOST=""

while [ $# -gt 0 ]; do
    case "$1" in
        --test)       TEST=1; shift ;;
        --start)      START="$2"; shift 2 ;;
        --end-time)   END_TIME="$2"; shift 2 ;;
        --walltime)   WALLTIME_NIGHT="$2"; WALLTIME_SET=1; shift 2 ;;
        --host)       HOST="$2"; shift 2 ;;
        -e|--experiment) EXPERIMENT="$2"; TEST_EXPERIMENT="$2"; shift 2 ;;
        -h|--help)    sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "Opção desconhecida: $1" >&2; exit 1 ;;
    esac
done

if ! [[ "$EXPERIMENT" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "ERRO: experimento inválido: '${EXPERIMENT}'." >&2
    exit 1
fi

# ---------- Janela de tempo ----------
# Teste (--test): sem o tipo night, para poder rodar de dia. Sem --start usa
# --walltime (padrão 0:59) e o OAR começa assim que houver nó livre. Com --start
# vira reserva com hora fixa e o fim é --end-time: o job NUNCA passa dele.
# Noturno (sem --test): tipo night, início >= 19:00 e fim <= 09:00.
if [ "$TEST" = "1" ]; then
    EXPERIMENT="${TEST_EXPERIMENT}"
    TYPES=(-t exotic)
    export EPOCHS="${EPOCHS:-2}"
    export IMAGE_LIMIT="${IMAGE_LIMIT:-1500}"
    export MARGIN_MIN="${MARGIN_MIN:-5}"
    [ "$WALLTIME_SET" = "1" ] || WALLTIME_NIGHT="0:59"
else
    TYPES=(-t night -t exotic)
fi

if [ -z "$START" ]; then
    if ! [[ "$WALLTIME_NIGHT" =~ ^([0-9]+):([0-5][0-9])$ ]] || \
       [ $((10#${BASH_REMATCH[1]} * 60 + 10#${BASH_REMATCH[2]})) -gt 840 ]; then
        echo "ERRO: --walltime inválido: '${WALLTIME_NIGHT}'. Use H:MM, no máximo 14:00." >&2
        exit 1
    fi
    WALLTIME="${WALLTIME_NIGHT}:00"
    OAR_TIME_ARGS=()
    if [ "$TEST" = "1" ]; then
        echo "Teste: começa quando houver nó livre, walltime ${WALLTIME}"
    else
        echo "Job noturno: início a partir das 19:00, fim até as 09:00 (walltime ${WALLTIME})"
    fi
else
    START_S=$(date -d "$START" +%s)
    END_S=$(date -d "$(date -d "@${START_S}" +%F) ${END_TIME}" +%s)
    [ "$END_S" -le "$START_S" ] && END_S=$(date -d "$(date -d "@${START_S}" +%F) ${END_TIME} tomorrow" +%s)
    DUR=$((END_S - START_S))
    MIN_DUR=3600
    [ "$TEST" = "1" ] && MIN_DUR=1200
    if [ "$DUR" -lt "$MIN_DUR" ]; then
        echo "ERRO: janela de só $((DUR / 60)) min até ${END_TIME}." >&2
        exit 1
    fi
    WALLTIME=$(printf '%d:%02d:00' $((DUR / 3600)) $((DUR % 3600 / 60)))
    START_STR=$(date -d "@${START_S}" '+%Y-%m-%d %H:%M:%S')
    OAR_TIME_ARGS=(-r "$START_STR")
    echo "Reserva: ${START_STR} -> $(date -d "@${END_S}" '+%Y-%m-%d %H:%M') (walltime ${WALLTIME})"
fi

# ---------- Índice da execução ----------
BASE_RESULTS="${RESULTS_ROOT}/${EXPERIMENT}"
mkdir -p "$BASE_RESULTS"
LAST_ID=$(find "$BASE_RESULTS" -maxdepth 1 -mindepth 1 -type d -name '[0-9][0-9]' -printf '%f\n' | sort -n | tail -1)
NEXT_ID=$(( ${LAST_ID:+10#$LAST_ID} + 1 ))
while ! mkdir "${BASE_RESULTS}/$(printf '%02d' "$NEXT_ID")" 2>/dev/null; do
    NEXT_ID=$((NEXT_ID + 1))
done
RUN_ID=$(printf '%02d' "$NEXT_ID")
RUN_DIR="${BASE_RESULTS}/${RUN_ID}"
mkdir -p "${RUN_DIR}/logs"

# ---------- Comando do job ----------
# OAR não repassa o ambiente da submissão: as variáveis vão no comando via env.
ENV_VARS=(RUN_ID="$RUN_ID" EXPERIMENT="$EXPERIMENT" REPO_SRC="$REPO_SRC" RESULTS_ROOT="$RESULTS_ROOT")
for v in EPOCHS BATCH_SIZE CROP_SIZE GPUS DATA_PROTOCOL IMAGE_LIMIT ENV_MODE SIF MARGIN_MIN KEEP_WORKDIR IMG_SRC; do
    [ -n "${!v:-}" ] && ENV_VARS+=("${v}=${!v}")
done
CMD="env $(printf '%q ' "${ENV_VARS[@]}") bash $(printf '%q' "${SCRIPT_DIR}/job_g5k.sh")"

PROP="cluster='hydra'"
[ -n "$HOST" ] && PROP="${PROP} and host like '${HOST}.%'"

echo "RUN_ID=${RUN_ID} -> ${RUN_DIR}"
oarsub \
    -n "eyeq_${EXPERIMENT}_${RUN_ID}" \
    "${TYPES[@]}" \
    -p "$PROP" \
    -l "host=1,walltime=${WALLTIME}" \
    "${OAR_TIME_ARGS[@]}" \
    -O "${RUN_DIR}/logs/OAR.%jobid%.stdout" \
    -E "${RUN_DIR}/logs/OAR.%jobid%.stderr" \
    "$CMD"
