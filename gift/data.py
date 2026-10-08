"""Traffic windows and normalization used by the formal cross-city runs."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class Range:
    start: int
    end: int

    def __post_init__(self):
        if self.start < 0 or self.end <= self.start:
            raise ValueError('Expected a nonempty half-open interval')


@dataclass
class CityData:
    name: str
    speed: np.ndarray
    steps_per_day: int
    interval_minutes: int
    spec: dict


def load_city(data_root, manifest, name, verify=True):
    spec = manifest['cities'][name]
    path = Path(data_root) / spec['series_file']
    raw = np.load(path, mmap_mode='r', allow_pickle=False)
    if tuple(raw.shape) != tuple(spec['expected_shape']):
        raise ValueError(f'{name}: shape {raw.shape} differs from {spec["expected_shape"]}')
    if verify and file_sha256(path) != spec['series_sha256']:
        raise ValueError(f'{name}: data SHA differs from the reported benchmark; use --skip-data-hash only for a different dataset version')
    speed = np.asarray(raw[:, :, spec.get('speed_channel', 0)], dtype=np.float32)
    if not np.isfinite(speed).all():
        raise ValueError('Input speed must be finite. Zero observations are retained as missing-value markers.')
    return CityData(name, speed, spec['steps_per_day'], spec['interval_minutes'], spec)


def window_starts_inside(bounds, history, horizon):
    return np.arange(bounds.start, max(bounds.start, bounds.end - history - horizon + 1), dtype=np.int64)


def forecast_starts_after(boundary, end, history, horizon):
    first, stop = boundary - history, end - history - horizon + 1
    if first < 0 or stop <= first:
        raise ValueError('No legal evaluation windows')
    return np.arange(first, stop, dtype=np.int64)


def fit_speed_stats(city, bounds):
    speed = city.speed[bounds.start:bounds.end]
    return float(speed.mean()), max(float(speed.std()), 1e-6)


class TrafficWindows(Dataset):
    def __init__(self, city, starts, history, horizon, stats):
        self.city, self.starts = city, np.asarray(starts, dtype=np.int64)
        self.history, self.horizon = int(history), int(horizon)
        self.mean, self.std = map(float, stats)
        if self.std <= 0:
            raise ValueError('Normalization standard deviation must be positive')

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = int(self.starts[index]); middle = start + self.history
        observed = self.city.speed[start:middle].copy()
        future = self.city.speed[middle:middle + self.horizon].copy()
        valid_history = (np.isfinite(observed) & (observed != 0)).T.copy()
        # Preserve the formal three-channel tensor layout. The two placeholder
        # channels are ignored by every released GIFT variant.
        x = np.zeros((*observed.shape, 3), dtype=np.float32)
        x[..., 0] = (observed - self.mean) / self.std
        y = ((future - self.mean) / self.std).T.copy()
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(valid_history)


def datasets(config, data_root, stage, source_stats=None, verify=True):
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'configs/datasets.json').read_text())
    source = load_city(data_root, manifest, config['source'], verify)
    target = load_city(data_root, manifest, config['target'], verify)
    history, horizon = config['model']['history_length'], config['horizon']
    source_bounds = Range(0, int(len(source.speed) * config['source_training']['fraction']))
    a, b = [int(round(d * target.steps_per_day)) for d in config['support_days']]
    if b > len(target.speed) or a < 0 or b < a:
        raise ValueError('Support interval lies outside the target data')
    support = Range(a, b) if b > a else None
    source_normalization = fit_speed_stats(source, source_bounds)
    target_normalization = fit_speed_stats(target, support) if support else tuple(source_stats or source_normalization)
    if stage == 'pretrain':
        starts = window_starts_inside(source_bounds, history, horizon)
        training = TrafficWindows(source, starts, history, horizon, source_normalization)
    elif support:
        starts = window_starts_inside(support, history, horizon)
        training = TrafficWindows(target, starts, history, horizon, target_normalization)
    else:
        training = None
    boundary = int(round(config['test_after_day'] * target.steps_per_day))
    if support is not None and support.end > boundary:
        raise ValueError('Support observations must end before the test label interval begins')
    starts = forecast_starts_after(boundary, len(target.speed), history, horizon)
    inputs = TrafficWindows(target, starts, history, horizon, target_normalization)
    # All formal target comparisons share labels produced in first-two-day coordinates.
    canonical_stats = fit_speed_stats(target, Range(0, 2 * target.steps_per_day))
    canonical = TrafficWindows(target, starts, history, horizon, canonical_stats)
    identity = dict(source_interval=[0, source_bounds.end], support_interval=[a, b],
                    test_label_interval=[boundary, len(target.speed)], history=history, horizon=horizon,
                    evaluation_starts_sha256=hashlib.sha256(starts.tobytes()).hexdigest(),
                    input_stats=list(target_normalization), canonical_stats=list(canonical_stats),
                    source_series_sha256=source.spec['series_sha256'], target_series_sha256=target.spec['series_sha256'],
                    verified_dataset_hashes=verify)
    return training, inputs, canonical, identity


def graph_for(config, data_root, stage, verify=True):
    from .models import graph_from_dense
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'configs/datasets.json').read_text())
    city = config['source'] if stage == 'pretrain' else config['target']
    spec = manifest['cities'][city]
    path = Path(data_root) / spec['graph_file']
    if verify and file_sha256(path) != spec['graph_sha256']:
        raise ValueError('Road graph SHA differs from the formal benchmark')
    return graph_from_dense(np.load(path, allow_pickle=False), {'city': city, 'orientation': 'receiver_row_sender_column'})
