"""Convert a speed NPY or DCRNN HDF5 matrix to the benchmark's [T,N,2] layout."""
import argparse
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--interval-minutes', type=int, choices=[5, 10], required=True)
    args = p.parse_args()
    source = Path(args.input)
    if source.suffix.lower() in ('.h5', '.hdf5'):
        try: import pandas as pd
        except ImportError as e: raise ImportError('HDF5 conversion needs requirements-data.txt') from e
        frame = pd.read_hdf(source)
        speed = frame.to_numpy(dtype=np.float32)
        timestamps = frame.index
        if not hasattr(timestamps, 'hour'): raise ValueError('HDF5 index must contain timestamps')
        tod = (timestamps.hour * 60 + timestamps.minute).to_numpy(dtype=np.float32) / (24 * 60)
    else:
        speed = np.load(source, allow_pickle=False)
        if speed.ndim == 3: speed = speed[..., 0]
        if speed.ndim != 2: raise ValueError('Expected speed [time,nodes]')
        speed = speed.astype(np.float32)
        day_steps = 1440 // args.interval_minutes
        tod = (np.arange(len(speed)) % day_steps).astype(np.float32) / day_steps
    if not np.isfinite(speed).all(): raise ValueError('Nonfinite observations require an explicitly documented preprocessing choice')
    output = np.stack((speed, np.broadcast_to(tod[:, None], speed.shape)), axis=-1)
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists(): raise FileExistsError(destination)
    np.save(destination, output)
    print(f'Saved {output.shape} float32. Check the benchmark manifest SHA before claiming the reported protocol.')


if __name__ == '__main__': main()
