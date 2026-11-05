import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('fetch_models', Path(__file__).with_name('fetch_models.py'))
fetch_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_models)


class FetchModelsTest(unittest.TestCase):
    def test_resume_verified_chunk_and_finalize(self):
        import json
        first, second = b'GGUFfirst', b'second'
        digest = lambda data: 'sha256:' + hashlib.sha256(data).hexdigest()
        config = json.dumps({'files': [{'name': 'student.gguf', 'size': len(first + second),
                             'sha256': hashlib.sha256(first + second).hexdigest(),
                             'parts': [{'digest': digest(data), 'size': len(data)} for data in (first, second)]}]}).encode()

        class Response:
            def __init__(self, data):
                self.content = data
            def json(self):
                return json.loads(self.content)
            def raise_for_status(self):
                pass
            def iter_content(self, size):
                yield self.content
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        class Session:
            headers = {}
            def get(self, url, **kwargs):
                if url.endswith('/token'):
                    return Response(b'{"token":"test"}')
                if '/manifests/' in url:
                    return Response(json.dumps({'config': {'digest': digest(config)}}).encode())
                if url.endswith(digest(config)):
                    return Response(config)
                if url.endswith(digest(second)):
                    return Response(second)
                raise AssertionError('Verified first chunk should not be downloaded')

        with tempfile.TemporaryDirectory() as directory, patch.object(fetch_models.requests, 'Session', Session):
            path = Path(directory)
            (path / 'student.gguf.partial').write_bytes(first + b'bad unfinished chunk')
            fetch_models.fetch('owner/package', 'tag', path)
            self.assertEqual((path / 'student.gguf').read_bytes(), first + second)
            self.assertFalse((path / 'student.gguf.partial').exists())
            fetch_models.fetch('owner/package', 'tag', path)
            (path / 'student.gguf').write_bytes(b'bad')
            with self.assertRaises(ValueError):
                fetch_models.fetch('owner/package', 'tag', path)
