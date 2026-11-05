import json
import os
from pathlib import Path
import secrets
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

name = 'aikar-jupyter-auto-' + secrets.token_hex(4)
directory = tempfile.mkdtemp(prefix='aikar-jupyter-workspace-', dir='/tmp')
image = os.environ.get('TEST_IMAGE', 'aikar-distill:recovery-jupyter')
Path(directory, 'logs').mkdir()
Path(directory, 'logs/jupyter.log').write_text('old workspace log\n')
try:
    subprocess.run(['docker', 'run', '-d', '--name', name, '-p', '127.0.0.1::8888',
                    '-e', 'JUPYTER_PASSWORD=', '-e', 'JUPYTER_DISABLE_AUTH=', '-e', 'TMPDIR=/tmp',
                    '-v', directory + ':/workspace:ro',
                    '--entrypoint', 'bash', image, '-c',
                    'ln -s /workspace/logs/jupyter.log /jupyter.log; exec /start.sh'],
                   check=True, stdout=subprocess.DEVNULL)
    token = None
    for restart in [False, True]:
        if restart:
            subprocess.run(['docker', 'restart', name], check=True, stdout=subprocess.DEVNULL)
        address = subprocess.check_output(['docker', 'port', name, '8888/tcp'], text=True).strip()
        for attempt in range(40):
            current = subprocess.run(['docker', 'exec', name, 'cat', '/run/aikar-jupyter/token'],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            if current.returncode == 0:
                value = current.stdout.strip()
                if token is None:
                    token = value
                assert token == value
                request = urllib.request.Request('http://' + address + '/api/kernelspecs',
                    headers={'Authorization': 'token ' + token})
                try:
                    with urllib.request.urlopen(request, timeout=2) as response:
                        assert 'python3' in json.load(response)['kernelspecs']
                    break
                except (urllib.error.URLError, ConnectionError, TimeoutError):
                    pass
            time.sleep(1)
        else:
            raise RuntimeError('Auto-start Jupyter did not become ready')
        try:
            urllib.request.urlopen('http://' + address + '/api/kernelspecs', timeout=2)
        except urllib.error.HTTPError as error:
            assert error.code in (401, 403)
        else:
            raise AssertionError('Unauthenticated API was accepted')
        mode = subprocess.check_output(['docker', 'exec', name, 'stat', '-c', '%a', '/run/aikar-jupyter/token'], text=True).strip()
        assert mode == '600'
        logs = subprocess.check_output(['docker', 'logs', name], stderr=subprocess.STDOUT, text=True)
        assert token not in logs
        print('Read-only workspace, stale log symlink, authenticated API, private token and restart: OK')
finally:
    subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import shutil
    shutil.rmtree(directory)
