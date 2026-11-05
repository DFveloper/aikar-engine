#!/usr/bin/env bash
set -euo pipefail
umask 077
source /opt/aikar-engine/scripts/logit-distillation/runpod/jupyter-env.sh
touch /run/aikar-jupyter/jupyter.log
chmod 600 /run/aikar-jupyter/jupyter.log
ln -sfn /run/aikar-jupyter/jupyter.log /jupyter.log
exec /runpod-start.sh "$@"
