#! /bin/bash

set -e -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# shellcheck disable=SC1091
. "${SCRIPT_DIR}/.env"

docker run --rm -it -v "${MODEL_VOLUME}:/models" \
    ghcr.io/ggerganov/llama.cpp:light \
    -m /models/${MODEL_FILE} \
    -n 512 \
    -p "[INST] $* [/INST]"
