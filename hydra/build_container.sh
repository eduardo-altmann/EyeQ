#!/bin/bash
# Constrói a imagem Apptainer do ambiente hydra (hydra/eyeq-cuda.def).
#
# No Grid'5000 o build roda dentro de um job em um nó (o frontend não tem
# apptainer). A imagem tem ~6 GB e fica em ~/eyeq/containers/, onde o
# job_g5k.sh a procura. Só precisa ser refeita quando hydra/uv.lock mudar.
#
#   ssh lyon.g5k
#   oarsub -t exotic -p "cluster='hydra'" -l host=1,walltime=0:50 \
#       "~/eyeq/EyeQ/hydra/build_container.sh"
#
# Variáveis: OUT (caminho do .sif), APPTAINER (binário, padrão: apptainer).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
OUT="${OUT:-$HOME/eyeq/containers/eyeq-cuda12.8-aarch64.sif}"
APPTAINER="${APPTAINER:-apptainer}"

if ! command -v "$APPTAINER" >/dev/null 2>&1; then
    echo "ERRO: ${APPTAINER} não encontrado neste nó." >&2
    exit 1
fi

# Build em disco local: mais rápido e não estoura a quota da home com temporários.
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp/apptainer_build_$$}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/tmp/apptainer_cache_$$}"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$(dirname "$OUT")"
trap 'rm -rf "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"' EXIT

LOCAL_SIF="${APPTAINER_TMPDIR}/eyeq.sif"
cd "$REPO"
"$APPTAINER" build --fakeroot "$LOCAL_SIF" hydra/eyeq-cuda.def

cp "$LOCAL_SIF" "$OUT.partial" && mv "$OUT.partial" "$OUT"
sha256sum "$OUT" | tee "${OUT}.sha256"
"$APPTAINER" exec "$OUT" cat /opt/eyeq/versions.txt | head -3
echo "Imagem pronta: ${OUT}"
