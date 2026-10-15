"""Boundaries, formal budgets, masks, statistics and disposable CPU training."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from gift import data, engine
from gift.metrics import metrics

ROOT = Path(__file__).resolve().parents[1]


class PipelineContracts(unittest.TestCase):
    def test_four_formal_update_budgets_and_test_boundary(self):
        manifest = json.loads((ROOT / 'configs/datasets.json').read_text())
        for path in ROOT.joinpath('configs').glob('*.json'):
            if path.name == 'datasets.json': continue
            cfg = json.loads(path.read_text())
            h, horizon = cfg['model']['history_length'], cfg['horizon']
            source, target = manifest['cities'][cfg['source']], manifest['cities'][cfg['target']]
            source_end = int(source['expected_shape'][0] * .7)
            support_end = 2 * target['steps_per_day']
            source_starts = data.window_starts_inside(data.Range(0, source_end), h, horizon)
            support_starts = data.window_starts_inside(data.Range(0, support_end), h, horizon)
            self.assertEqual(100 * (len(source_starts) // 16), cfg['source_updates'])
            self.assertEqual(50 * (len(support_starts) // 16), cfg['target_updates'])
            test = data.forecast_starts_after(support_end, target['expected_shape'][0], h, horizon)
            self.assertEqual(int(test[0]) + h, support_end)
            self.assertEqual(int(test[-1]) + h + horizon, target['expected_shape'][0])
            self.assertLessEqual(int(support_starts[-1]) + h + horizon, support_end)

    def test_windows_mask_and_float32_roundtrip(self):
        rng = np.random.default_rng(21)
        speed = rng.normal(35, 10, (128, 5)).astype(np.float32)
        speed[7, 2] = 0
        city = data.CityData('synthetic', speed, 16, 10, {})
        mean, std = data.fit_speed_stats(city, data.Range(0, 32))
        ds = data.TrafficWindows(city, np.arange(0, 32 - 12 - 6 + 1), 12, 6, (mean, std))
        x, y, mask = ds[0]
        self.assertEqual(x.shape, (12, 5, 3))
        self.assertEqual(y.shape, (5, 6))
        self.assertFalse(mask[2, 7])
        restored = (y.numpy().T * std + mean).astype(np.float32)
        expected = (((speed[12:18] - mean) / std) * std + mean).astype(np.float32)
        np.testing.assert_array_equal(restored, expected)
        self.assertEqual(mean, float(speed[:32].mean()))
        self.assertEqual(std, float(speed[:32].std()))
        self.assertEqual(x[..., 0].stride(), (15, 3))

    def test_masked_metrics_and_empty_mask(self):
        truth = np.array([[[2., 4., 0.], [4., 0., 10.]]], dtype=np.float32)
        pred = np.array([[[3., 2., 100.], [2., 100., 13.]]], dtype=np.float32)
        result = metrics(pred, truth)
        self.assertAlmostEqual(result['mae'], 2.)
        self.assertAlmostEqual(result['mape'], 45.)
        self.assertAlmostEqual(result['rmse'], np.sqrt(4.5))
        self.assertEqual(result['horizon_valid_count'], [2, 1, 1])
        empty = metrics(pred, np.zeros_like(truth))
        self.assertIsNone(empty['mae'])
        self.assertEqual(empty['horizon_mae'], [None, None, None])

    def test_expected_statistics_come_from_five_seeds(self):
        with (ROOT / 'results/expected_main.csv').open() as f: means = list(csv.DictReader(f))
        with (ROOT / 'results/expected_seeds.csv').open() as f: seeds = list(csv.DictReader(f))
        self.assertEqual(len(means), 36)
        self.assertEqual(len(seeds), 180)
        for row in means:
            matches = [r for r in seeds if all(r[k] == row[k] for k in ['direction', 'metric', 'report_step'])]
            self.assertEqual(sorted(int(r['seed']) for r in matches), list(range(5)))
            values = [float(r['value']) for r in matches]
            self.assertAlmostEqual(float(np.mean(values)), float(row['mean']))
            self.assertAlmostEqual(float(np.std(values, ddof=1)), float(row['sample_sd']))

    def test_train_adapt_evaluate_small_disposable_dataset(self):
        cfg = json.loads((ROOT / 'configs/cd_sz.json').read_text())
        cfg['source_training']['epochs'] = 1
        cfg['adaptation']['epochs'] = 1
        rng = np.random.default_rng(9)
        speed = rng.normal(40, 8, (192, 5)).astype(np.float32)
        speed[8, 1] = 0
        city = data.CityData('synthetic', speed, 32, 10, {})
        src_stats = data.fit_speed_stats(city, data.Range(0, 128))
        dst_stats = data.fit_speed_stats(city, data.Range(0, 64))
        source = data.TrafficWindows(city, data.window_starts_inside(data.Range(0, 128), 12, 6), 12, 6, src_stats)
        support = data.TrafficWindows(city, data.window_starts_inside(data.Range(0, 64), 12, 6), 12, 6, dst_stats)
        test = data.TrafficWindows(city, data.forecast_starts_after(64, 192, 12, 6), 12, 6, dst_stats)
        def synthetic(config, root, stage, source_stats=None, verify=True):
            return (source if stage == 'pretrain' else support), test, test, {'synthetic': True}
        with tempfile.TemporaryDirectory() as directory, patch('gift.engine.datasets', side_effect=synthetic), contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            source_result = engine.train(cfg, 0, 'pretrain', root, root / 'run', 'cpu', verify=False)
            self.assertEqual(source_result['completed_updates'], len(source) // 16)
            adapted = engine.train(cfg, 0, 'adapt', root, root / 'run', 'cpu', root / 'run/source.pt', verify=False)
            self.assertEqual(adapted['completed_updates'], len(support) // 16)
            self.assertEqual(adapted['metrics']['reported_minutes'], [10, 30, 60])
            source_state = torch.load(root / 'run/source.pt', weights_only=False)
            target_state = torch.load(root / 'run/adapted.pt', weights_only=False)
            m = engine.make_model(cfg, 0, 'cpu', root, 'adapt', False)
            m.freeze_communication()
            for name, parameter in m.named_parameters():
                if not parameter.requires_grad:
                    self.assertTrue(torch.equal(source_state['model'][name], target_state['model'][name]), name)
            evaluated = engine.evaluate(cfg, 0, root / 'run/adapted.pt', root, 'cpu', verify=False)
            self.assertEqual(evaluated, adapted['metrics'])
            with self.assertRaises(FileExistsError):
                engine.train(cfg, 0, 'pretrain', root, root / 'run', 'cpu', verify=False)


if __name__ == '__main__': unittest.main()
