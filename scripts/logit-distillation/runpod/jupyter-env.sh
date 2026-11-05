#!/usr/bin/env bash
export TMPDIR=${TMPDIR:-/workspace/tmp}
export JUPYTER_CONFIG_DIR=${JUPYTER_CONFIG_DIR:-/run/aikar-jupyter/config}
export JUPYTER_RUNTIME_DIR=${JUPYTER_RUNTIME_DIR:-/run/aikar-jupyter/runtime}
export JUPYTER_DATA_DIR=${JUPYTER_DATA_DIR:-/run/aikar-jupyter/data}
mkdir -p /run/aikar-jupyter "$TMPDIR" "$JUPYTER_CONFIG_DIR" "$JUPYTER_RUNTIME_DIR" "$JUPYTER_DATA_DIR"
chmod 700 /run/aikar-jupyter
if [[ -z "${JUPYTER_PASSWORD:-}" && "${JUPYTER_DISABLE_AUTH:-}" != true ]]; then
    JUPYTER_PASSWORD=$(python - <<'PY'
import os
from pathlib import Path
import secrets

path = Path('/run/aikar-jupyter/token')
path.parent.mkdir(parents=True, exist_ok=True)
try:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
except FileExistsError:
    token = path.read_text().strip()
    if not token:
        raise RuntimeError('Jupyter token file is empty; set JUPYTER_PASSWORD or remove the empty file.')
else:
    token = secrets.token_urlsafe(32)
    with os.fdopen(fd, 'w') as stream:
        stream.write(token + '\n')
os.chmod(path, 0o600)
print(token)
PY
    )
    export JUPYTER_PASSWORD
    echo 'JupyterLab token is stored in /run/aikar-jupyter/token (owner access only).'
fi
