import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


prepare = module('prepare_ko')
budget = module('budget')


class KoreanBudgetTest(unittest.TestCase):
    def test_measured_log_rates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'benchmark.log'
            for marker in ('distill_throughput', 'qat_throughput'):
                path.write_text('initialization done\n' + marker + ': input_tokens=8192 supervised_tokens=6000 elapsed_s=2.000 input_tok_s=4096.000 supervised_tok_s=3000.000\nfinished\n')
                self.assertEqual(budget.rate(path, marker), 4096.0)
                path.write_text(marker + ': input_tokens=8192\nunrelated input_tok_s=4096.000\n')
                with self.assertRaises(ValueError):
                    budget.rate(path, marker)
                path.write_text(marker + ': input_tok_s=0.000\n')
                with self.assertRaises(ValueError):
                    budget.rate(path, marker)

    def test_reasoning_and_korean_final_answer(self):
        row = {'instruction': '물이 끓으면 수증기가 발생하는 이유를 단계별로 설명해주세요.',
               'response': '<think>Consider molecular kinetic energy. 분자의 운동 에너지와 상태 변화의 관계를 고려해 원인을 설명합니다.</think>\n'
                           '물을 가열하면 분자의 운동 에너지가 커지고 액체에서 기체로 상태가 변하기 때문에 수증기가 발생합니다.'}
        result = prepare.normalize(row, 'reasoning', 'source')
        self.assertIn('kinetic', result['messages'][1]['reasoning_content'])
        self.assertNotIn('<think>', result['messages'][1]['content'])
        self.assertEqual(len(result['messages']), 2)
        english = dict(row, response='<think>Consider kinetic energy and phase transitions.</think>분자의 운동 에너지가 증가하기 때문에 수증기가 발생합니다.')
        self.assertIsNone(prepare.normalize(english, 'reasoning', 'source'))
        row['response'] = '<think>incomplete'
        self.assertIsNone(prepare.normalize(row, 'reasoning', 'source'))

    def test_knowledge_filters(self):
        row = {'instruction': '광합성의 원리와 식물에서 담당하는 역할을 설명해주세요.',
               'input': '', 'output': '식물은 빛 에너지를 이용하여 물과 이산화탄소로부터 유기물을 합성하며 산소를 방출합니다.'}
        self.assertIsNotNone(prepare.normalize(row, 'knowledge', 'source'))
        row['instruction'] = '주어진 문서를 읽고 내용을 간단히 요약해주세요.'
        self.assertIsNone(prepare.normalize(row, 'knowledge', 'source'))
        row['instruction'] = 'Explain photosynthesis and the role of chlorophyll.'
        self.assertIsNone(prepare.normalize(row, 'knowledge', 'source'))

    def test_json_messages_reasoning_and_first_turn(self):
        messages = [{'role': 'system', 'content': '한국어로 답변하세요.'},
                    {'role': 'user', 'content': '버스에 타야 하는 사람이 총 260명이고 정원이 41명이면 몇 대가 필요합니까?'},
                    {'role': 'assistant', 'content': '260을 41로 나누고 올림하면 되므로 버스는 총 7대가 필요합니다.',
                     'reasoning': '먼저 전체 인원을 버스 한 대의 정원으로 나누고 올림하여 필요한 버스 대수를 계산합니다.\n부분 버스는 빌릴 수 없습니다.'},
                    {'role': 'user', 'content': 'later'}, {'role': 'assistant', 'content': ''}]
        result = prepare.normalize({'messages': json.dumps(messages)}, 'reasoning', 'source')
        self.assertEqual(len(result['messages']), 3)
        self.assertIn('\n', result['messages'][2]['reasoning_content'])
        self.assertEqual(result['messages'][0], messages[0])

    def test_time_cache_and_checkpoint_limits(self):
        settings = {'CTX': '1024', 'TOP_K': '512', 'DATA': '/workspace/data.jsonl',
                    'MODEL_DIR': '/workspace/models', 'TRAIN_UBATCH': '1024'}
        values, report = budget.estimate(settings, 500, 100, 10, 2, 3, 64)
        self.assertLessEqual(report['estimated_compute_hours'], 8)
        self.assertLessEqual(report['worst_case_cache_gib'], 64)
        self.assertLessEqual(report['checkpoint_sets_cap'], 9)
        slower, _ = budget.estimate(settings, 100, 20, 10, 2, 3, 64)
        self.assertLess(int(slower['PROD_MAX_WINDOWS']), int(values['PROD_MAX_WINDOWS']))
        self.assertEqual(values['TRAIN_UBATCH'], '1024')
        with self.assertRaises(ValueError):
            budget.estimate(settings, 500, 100, 1, 2, 3, 64)


if __name__ == '__main__':
    unittest.main()
