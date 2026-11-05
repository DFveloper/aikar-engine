import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

from production import run_training

spec = importlib.util.spec_from_file_location('hf_upload', Path(__file__).with_name('hf_upload.py'))
hf_upload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hf_upload)


class FakeApi:
    def __init__(self, token):
        self.token = token
        self.commits = []
        self.private = False

    def whoami(self):
        return {'name': 'token-owner'}

    def create_repo(self, **kwargs):
        assert kwargs['private'] is False

    def repo_info(self, **kwargs):
        return SimpleNamespace(private=self.private)

    def create_commit(self, **kwargs):
        self.commits.append(kwargs)
        return SimpleNamespace(oid='commit-id', commit_url='https://huggingface.co/test/commit/commit-id')


class UploadTest(unittest.TestCase):
    def test_error_preserves_cause_and_redacts_tokens(self):
        with patch.dict(os.environ, {'HF_TOKEN': 'runtime-secret'}):
            message = hf_upload.upload_error(RuntimeError('403 model write denied runtime-secret hf_OtherSecret123'))
        self.assertIn('403 model write denied', message)
        self.assertNotIn('runtime-secret', message)
        self.assertNotIn('hf_OtherSecret123', message)

    def publisher(self, **env):
        with patch('huggingface_hub.HfApi', FakeApi):
            return hf_upload.Publisher({'HF_TOKEN': 'test-secret', **env})

    def test_environment_identity_and_public_repository(self):
        with self.assertRaisesRegex(RuntimeError, 'HF_TOKEN'):
            hf_upload.Publisher({})
        publisher = self.publisher()
        self.assertEqual(publisher.repos['model'], 'token-owner/lumen-recovery-checkpoints')
        self.assertEqual(self.publisher(HF_MODEL_REPO='team/checkpoints').repos['model'], 'team/checkpoints')
        publisher.api.private = True
        with self.assertRaisesRegex(RuntimeError, 'private'):
            publisher.prepare('dataset')

    def test_dataset_checksums_and_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'mixed.jsonl'
            data.write_text('{"messages": []}\n')
            cache = root / 'teacher-top512-c512.dat'
            cache.write_bytes(b'completed-cache')
            manifest = {'sha256': hf_upload.sha256(cache), 'inputs': {
                'data': {'path': str(data)}, 'data_sha256': hf_upload.sha256(data),
                'top_k': 512, 'context': 512}}
            (root / 'cache.json').write_text(json.dumps(manifest))
            publisher = self.publisher()
            publisher.dataset(root)
            operations = publisher.api.commits[0]['operations']
            self.assertEqual(len(operations), 4)
            self.assertTrue(any(op.path_in_repo.endswith('/train.jsonl') for op in operations))
            self.assertEqual(publisher.api.commits[0]['repo_type'], 'dataset')
            publisher.dataset(root)
            self.assertEqual(len(publisher.api.commits), 1)
            self.assertNotIn('test-secret', (root / 'hf-dataset-upload.json').read_text())
            cache.write_bytes(b'torn-cache')
            with self.assertRaisesRegex(RuntimeError, 'checksum'):
                publisher.dataset(root)

    def test_only_completed_pairs_and_one_model_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'recovery-01'
            run.mkdir()
            (run / 'run.json').write_text('{}')
            (run / 'train-a.log').write_text('qat_epoch_callback: checkpoint set saved at step 1000 window 1000\n')
            step = run / 'final.gguf.step1000.gguf'
            step.write_bytes(b'model')
            (run / 'final.adapter.gguf.step1000.gguf').write_bytes(b'adapter')
            final = run / 'final.gguf'
            final.write_bytes(b'final')
            (run / 'final.gguf.step2000.gguf').write_bytes(b'incomplete')
            self.assertEqual(hf_upload.completed_checkpoints(run), [step])
            publisher = self.publisher()
            with self.assertRaisesRegex(RuntimeError, 'both'):
                publisher.checkpoint(run, step)
            Path(str(step) + '.qat-state.gguf').write_bytes(b'optimizer')
            publisher.checkpoint(run, step)
            publisher.checkpoint(run, step)
            self.assertEqual(len(publisher.api.commits), 1)
            Path(str(final) + '.qat-state.gguf').write_bytes(b'final-optimizer')
            (run / 'final.adapter.gguf').write_bytes(b'final-adapter')
            with (run / 'train-a.log').open('a') as stream:
                stream.write('qat_adapter_checkpoint: saved final path=final.adapter.gguf\n')
            self.assertEqual(hf_upload.completed_checkpoints(run, run / 'train-a.log'), [step, final])
            (run / 'status.json').write_text('{"phase":"complete"}')
            self.assertEqual(hf_upload.completed_checkpoints(run), [step, final])
            current_log = run / 'train-new.log'
            current_log.write_text('training resumed\n')
            self.assertEqual(hf_upload.completed_checkpoints(run, current_log), [])
            with patch.object(publisher.api, 'create_commit', side_effect=RuntimeError('network unavailable')):
                with self.assertRaises(RuntimeError):
                    publisher.checkpoint(run, final)
            self.assertFalse((run / 'final.gguf.hf-upload.json').exists())
            publisher.checkpoint(run, final)
            self.assertEqual({c['repo_id'] for c in publisher.api.commits}, {'token-owner/lumen-recovery-checkpoints'})
            for commit in publisher.api.commits:
                paths = [op.path_in_repo for op in commit['operations']]
                self.assertEqual(len(paths), 4)
                self.assertTrue(any(p.endswith('/adapter.gguf') for p in paths))
                self.assertFalse(any(p.endswith('/model.gguf') for p in paths))
                self.assertTrue(any(p.endswith('/model.gguf.qat-state.gguf') for p in paths))
            self.assertNotIn('test-secret', (run / (step.name + '.hf-upload.json')).read_text())

    def test_training_process_publishes_only_successful_save_events(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'recovery-02'
            run.mkdir()
            (run / 'run.json').write_text('{}')
            (run / 'status.json').write_text('{"phase":"running"}')
            publisher = self.publisher()
            signal = run / 'uploaded.fifo'
            os.mkfifo(signal)
            script = (
                'from pathlib import Path; import sys; '
                'p=Path(sys.argv[1]); p.write_bytes(b"model"); '
                'Path(str(p)+".qat-state.gguf").write_bytes(b"state"); '
                '(p.parent/"final.adapter.gguf.step1000.gguf").write_bytes(b"adapter"); '
                'print("checkpoint set saved at step 1000 window 1000", flush=True); '
                'open(sys.argv[2],"rb").read(1)'
            )
            model = run / 'final.gguf.step1000.gguf'
            original = publisher.api.create_commit
            def acknowledge_upload(**kwargs):
                result = original(**kwargs)
                with signal.open('wb') as stream:
                    stream.write(b'1')
                return result
            with patch.object(publisher.api, 'create_commit', side_effect=acknowledge_upload):
                run_training([sys.executable, '-c', script, str(model), str(signal)], run / 'train-new.log', run, publisher)
            self.assertEqual(len(publisher.api.commits), 1)
            partial = run / 'final.gguf.step2000.gguf'
            script = 'from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(b"partial"); sys.exit(1)'
            with self.assertRaisesRegex(RuntimeError, 'status 1'):
                run_training([sys.executable, '-c', script, str(partial)], run / 'train-failed.log', run, publisher)
            self.assertEqual(len(publisher.api.commits), 1)


if __name__ == '__main__':
    unittest.main()
