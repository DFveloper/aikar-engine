import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import server


def train(step, total=8, elapsed=2):
    return f'train: data={step:07d}/{total:07d} loss_ema100=2.0 acc_ema100=60.0% t=00:00:{elapsed:02d} ETA=00:00:10\n'


def val(step, epoch):
    return f'validation: step={step} epoch={epoch} loss=2.5 acc=55.0%\n'


HEADER = 'packed windows: train=4 val=1\ndataset: 5 windows x 2 ubatches = 8 steps per epoch\nlr scheduler total_steps=12 start_step=0\n'
CRITICAL = 'critical_sft: active_tokens=10 critical_tokens=2 critical_fraction=0.2 mean_active_weight=1.5 max_active_weight=2.0 unweighted_nll=2.0 weighted_loss=3.0\n'


class MonitorTest(unittest.TestCase):
    def snapshot(self, text, ema_n=100):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'train.log'
            log.write_text(text)
            with patch.object(server, 'training_process', return_value=123):
                return server.Monitor(log, 'train', ema_n).snapshot()

    def test_chart_keeps_microbatch_history(self):
        data = self.snapshot(HEADER + train(1) + train(2) + train(3) + train(4))
        self.assertEqual([p['step'] for p in data['history']], [0.5, 1.0, 1.5, 2.0])
        self.assertLess(data['history'][0]['step'], 1.0)

    def test_chart_domain_ends_at_current_partial_window(self):
        data = self.snapshot(HEADER + train(2) + train(3))
        self.assertEqual(data['current'], 1)
        self.assertEqual(data['chart_start'], 0)
        self.assertEqual(data['chart_current'], 1.5)
        self.assertEqual([p['step'] for p in data['history']], [1.0, 1.5])

    def test_validation_progress_is_not_training(self):
        data = self.snapshot(HEADER + train(4) + 'val: data=1/2 loss_ema100=99 acc_ema100=1% t=00:00:01 ETA=00:00:00\n')
        self.assertEqual(data['current'], 2)
        self.assertEqual(data['loss'], 2.0)
        self.assertEqual(len(data['history']), 1)

    def test_epoch_two_axes_and_elapsed(self):
        text = HEADER + train(8, elapsed=8) + CRITICAL + val(4, 1)
        text += 'epoch 1/3: train_loss=2.0\n' + train(2) + 'epoch 1 lr=0.1\n'
        text += CRITICAL + val(1, 2) + train(4, elapsed=4) + 'epoch 1 lr=0.1\n'
        data = self.snapshot(text)
        self.assertEqual(data['current'], 6)
        self.assertEqual([p['step'] for p in data['validation_history']], [4, 5])
        self.assertEqual([p['step'] for p in data['nll_history']], [4, 5])
        self.assertEqual(data['elapsed'], '00:00:12')
        self.assertEqual(data['rate'], 0.5)
        self.assertEqual(data['eta'], '00:00:12')

    def test_resume_keeps_real_microbatch_factor(self):
        text = HEADER.replace('start_step=0', 'start_step=6') + train(2, total=4) + val(3, 2)
        data = self.snapshot(text)
        self.assertEqual(data['current'], 7)
        self.assertEqual(data['validation_history'][0]['step'], 7)

    def test_resumed_initial_validation_is_already_global(self):
        text = HEADER.replace('start_step=0', 'start_step=6') + val(6, 2) + train(2, total=4)
        data = self.snapshot(text)
        self.assertEqual(data['validation_history'][0]['step'], 6)

    def test_native_qat_validation_is_already_global(self):
        text = HEADER + train(8, elapsed=8) + 'qat_epoch: epoch=1/3\n' + train(2) + val(5, 2)
        data = self.snapshot(text)
        self.assertEqual(data['validation_history'][0]['step'], 5)

    def test_checkpoints_distinguish_epochs(self):
        text = HEADER + train(2)
        text += 'save_adapter: adapter saved to /tmp/a.epoch1.ckpt2.gguf\n'
        text += 'save_adapter: adapter saved to /tmp/a.epoch2.ckpt2.gguf\n'
        data = self.snapshot(text)
        self.assertEqual(data['checkpoints']['successful'], 2)
        self.assertEqual(data['checkpoints']['last_saved_step'], 6)

    def test_large_log_retains_epoch_and_schedule_metadata(self):
        text = HEADER + train(8, elapsed=8) + val(4, 1) + 'epoch 1/3: train_loss=2.0\n'
        text += 'unrelated log line\n' * 100 + train(4, elapsed=4)
        original = server.read_tail
        with patch.object(server, 'read_tail', side_effect=lambda path: original(path, 220)):
            data = self.snapshot(text)
        self.assertEqual(data['total'], 12)
        self.assertEqual(data['current'], 6)
        self.assertEqual(data['elapsed'], '00:00:12')

    def test_window_without_critical_weights_has_train_nll(self):
        data = self.snapshot(HEADER + train(1) + train(2))
        self.assertEqual(data['nll_history'], [{'step': 0.5, 'nll': 2.0}, {'step': 1.0, 'nll': 2.0}])


if __name__ == '__main__':
    unittest.main()
