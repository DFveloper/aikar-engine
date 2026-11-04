import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('mix_data', Path(__file__).with_name('mix_data.py'))
mix_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mix_data)


class MixtureTest(unittest.TestCase):
    def test_first_turn_and_deterministic_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ko, other = root / 'ko.jsonl', root / 'other.jsonl'
            messages = [{'role': 'system', 'content': 'system'},
                        {'role': 'user', 'content': 'first'},
                        {'role': 'assistant', 'content': 'answer', 'reasoning': 'reason'},
                        {'role': 'user', 'content': 'broken'},
                        {'role': 'assistant', 'content': ''}]
            ko.write_text(json.dumps({'messages': messages}) + '\n')
            other.write_text(''.join(json.dumps({'id': index, 'messages': messages[1:]}) + '\n' for index in range(20)))
            original = ko.read_bytes()
            first, second = root / 'first.jsonl', root / 'second.jsonl'
            mix_data.mix(ko, other, first, 42)
            mix_data.mix(ko, other, second, 42)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            records = [json.loads(line) for line in first.read_text().splitlines()]
            self.assertEqual(len(records), 2)
            korean = next(record for record in records if 'id' not in record)
            english = next(record for record in records if 'id' in record)
            self.assertEqual(korean['messages'], messages[:3])
            self.assertEqual(english['messages'], messages[1:])
            self.assertEqual(ko.read_bytes(), original)
            mix_data.mix(ko, other, first, 42)
            with self.assertRaises(ValueError):
                mix_data.mix(ko, other, first, 43)


if __name__ == '__main__':
    unittest.main()
