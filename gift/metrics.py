"""Original-unit float64 accumulation; finite, nonzero canonical-label masks."""
import numpy as np


class MetricAccumulator:
    def __init__(self, horizon):
        self.absolute = np.zeros(horizon, dtype=np.float64)
        self.percentage = self.absolute.copy()
        self.squared = self.absolute.copy()
        self.count = np.zeros(horizon, dtype=np.int64)

    def update(self, prediction, target, valid=None):
        prediction, target = np.asarray(prediction), np.asarray(target)
        if prediction.shape != target.shape or target.ndim != 3:
            raise ValueError('Expected matching [batch,nodes,horizon] predictions and targets')
        valid = np.isfinite(target) & (target != 0) if valid is None else np.asarray(valid, dtype=bool)
        if valid.shape != target.shape or not np.isfinite(prediction).all():
            raise ValueError('Invalid mask shape or nonfinite prediction')
        d = prediction.astype(np.float64) - target.astype(np.float64)
        absolute = np.abs(d)
        percentage = absolute / np.maximum(np.abs(target.astype(np.float64)), 1e-5) * 100
        self.absolute += np.where(valid, absolute, 0).sum((0, 1))
        self.percentage += np.where(valid, percentage, 0).sum((0, 1))
        self.squared += np.where(valid, d * d, 0).sum((0, 1))
        self.count += valid.sum((0, 1))

    def result(self, steps=(1, 3, 6)):
        def divide(a, root=False):
            v = np.divide(a, self.count, out=np.full_like(a, np.nan), where=self.count > 0)
            if root: v = np.sqrt(v)
            return [float(x) if np.isfinite(x) else None for x in v]
        mae, mape, rmse = divide(self.absolute), divide(self.percentage), divide(self.squared, True)
        total = int(self.count.sum())
        steps = [s for s in steps if s <= len(self.count)]
        return dict(mae=float(self.absolute.sum() / total) if total else None,
                    mape=float(self.percentage.sum() / total) if total else None,
                    rmse=float(np.sqrt(self.squared.sum() / total)) if total else None,
                    horizon_mae=mae, horizon_mape=mape, horizon_rmse=rmse,
                    horizon_valid_count=self.count.tolist(), reported_steps=steps,
                    reported_mae=[mae[s - 1] for s in steps], reported_mape=[mape[s - 1] for s in steps],
                    reported_rmse=[rmse[s - 1] for s in steps])


def metrics(prediction, target, valid=None):
    acc = MetricAccumulator(np.asarray(target).shape[-1])
    acc.update(prediction, target, valid)
    return acc.result()
