import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class StageScriptTest(unittest.TestCase):
    def test_stage_commands_run_in_flare_tmux_layout(self):
        source = Path(__file__).resolve().parent.parent / 'pulsar-s'
        if not source.is_dir():
            source = Path(__file__).resolve().parent.parent
            if source.name == 'Lumen-3.2-Flare':
                self.skipTest('Pulsar launch scripts are deployed in the Pulsar directory')
        for stage in range(1, 5):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory(prefix='pulsar $draft `path` \" script ') as directory:
                root = Path(directory)
                script = root / f'stage{stage}.sh'
                shutil.copy2(source / script.name, script)
                (root / 'model.gguf').touch()
                (root / 'data.jsonl').touch()
                (root / 'Q4_0_XL.txt').touch()
                commands = root / 'commands.jsonl'
                args = root / 'args.json'
                (root / 'pgrep').write_text('#!/bin/sh\nexit 1\n')
                (root / 'tmux').write_text('''#!/usr/bin/env python3
import json, os, subprocess, sys
with open(os.environ['COMMAND_TRACE'], 'a') as f:
    f.write(json.dumps(sys.argv[1:])+'\\n')
if sys.argv[1] == 'new-window':
    sys.exit(subprocess.call(['bash', '-c', sys.argv[-1]]))
''')
                trainer = root / 'trainer'
                trainer.write_text('''#!/usr/bin/env python3
import json, os, sys
with open(os.environ['ARGV_TRACE'], 'w') as f:
    json.dump(sys.argv[1:], f)
print('trainer started')
''')
                for path in [root / 'tmux', root / 'pgrep', trainer]:
                    path.chmod(0o755)
                env = dict(os.environ, PATH=f'{root}:'+os.environ['PATH'],
                           MODEL_DIR=str(root), MODEL_PATH=str(root/'model.gguf'), TRAIN_FILE=str(root/'data.jsonl'),
                           OUTPUT_PATH=str(root/'adapter.gguf'), QLION_BIN=str(trainer),
                           COMMAND_TRACE=str(commands), ARGV_TRACE=str(args))
                result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                calls = [json.loads(line) for line in commands.read_text().splitlines()]
                self.assertEqual(calls[0][:6], ['new-session', '-d', '-s', 'train', '-n', 'webui'])
                self.assertEqual(calls[2][:5], ['new-window', '-t', 'train', '-n', 'finetune'])
                argv = json.loads(args.read_text())
                def value(flag):
                    return argv[argv.index(flag)+1]
                self.assertEqual(value('-ub'), '64')
                self.assertEqual(value('-ctk'), 'f16')
                self.assertEqual(value('--model'), str(root/'model.gguf'))
                self.assertEqual(value('--epochs'), '1' if stage < 3 else '3')
                self.assertNotIn('attn_qkv', value('--lora-targets'))
                if stage == 4:
                    self.assertEqual(argv.count('--warmup-steps'), 1)
                    self.assertEqual(value('--warmup-steps'), '8')


if __name__ == '__main__':
    unittest.main()
