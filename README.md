# GIFT

**Graph-Free Interaction for Few-Shot Traffic Forecasting Across Cities**

GIFT learns a shared way to exchange information between traffic sensors without a road graph or cross-city sensor alignment. TIDE encodes local history and recent deviations; SIDE uses traffic-state features to match sensors and aggregates their information with a compact positive kernel. This repository contains the **main model only**, with the four source-to-target configurations used in the paper.

## Repository

```text
gift/
  model.py            # TIDE, SIDE, and the GIFT forecaster
  data.py             # windows and normalization
  engine.py           # source training, target adaptation, evaluation
  metrics.py          # MAE, MAPE, RMSE
configs/              # four transfer directions and dataset fingerprints
results/              # recorded five-seed GIFT results
prepare_data.py       # optional data-layout converter
train.py              # train or adapt one seed
evaluate.py           # evaluate a checkpoint
run_experiments.py    # list or run the 4 × 5 experiment queue
```

## Quick start

Use Python 3.10. The paper's GPU runs used PyTorch 2.1.2 with CUDA 12.1; the tests also run on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Get the datasets as described below, then run one source-to-target transfer:

```bash
python train.py --stage pretrain --config configs/la_bay.json --data-root data --seed 0
python train.py --stage adapt --config configs/la_bay.json --data-root data --seed 0
python evaluate.py --checkpoint outputs/LA_BAY/seed0/adapted.pt --data-root data
```

The commands save checkpoints and JSON results under `outputs/LA_BAY/seed0/`. To preview all four directions and five seeds, run `python run_experiments.py`; add `--execute` to start the full queue. A smaller selection is `python run_experiments.py --directions CD_SZ --seeds 0 --execute`. Existing checkpoints are never overwritten.

## Data

Download the data from their original providers:

- **METR-LA and PEMS-BAY:** the [DCRNN dataset instructions](https://github.com/liyaguang/DCRNN#data-preparation), which link to the traffic speed files.
- **Chengdu and Shenzhen:** DiDi's [GAIA Traffic Index data page](https://github.com/didi/TrafficIndex). Follow its access instructions and data-use terms.

Place the processed arrays at:

```text
data/
  metr-la/dataset.npy
  pems-bay/dataset.npy
  chengdu/dataset.npy
  shenzhen/dataset.npy
```

Each file is a float32 NumPy array of shape `[time, sensors, 2]`, with speed in channel 0 and normalized time of day in channel 1. GIFT reads the speed channel. The optional `prepare_data.py` converts a speed-only `.npy` or DCRNN `.h5` file to this layout; HDF5 conversion needs `pip install -r requirements-data.txt`.

```bash
python prepare_data.py --input metr-la.h5 --output data/metr-la/dataset.npy --interval-minutes 5
```

`configs/datasets.json` records the exact shapes and SHA-256 hashes of the processed arrays used for the reported results. Official raw downloads may require processing to match them byte-for-byte. The loader checks hashes by default; `--skip-data-hash` permits a same-shaped alternative and records that it was not hash-verified. No dataset, graph, or pretrained checkpoint is distributed here.

## Reproduction protocol

| Config | Transfer | Forecast steps | Source updates | Target updates |
| --- | --- | ---: | ---: | ---: |
| `la_bay.json` | METR-LA → PEMS-BAY | 12 × 5 min | 149,700 | 1,700 |
| `bay_la.json` | PEMS-BAY → METR-LA | 12 × 5 min | 227,800 | 1,700 |
| `cd_sz.json` | Chengdu → Shenzhen | 6 × 10 min | 75,400 | 800 |
| `sz_cd.json` | Shenzhen → Chengdu | 6 × 10 min | 75,400 | 800 |

Each run uses 12 history steps, width 128, kernel rank 16, three blocks, batch size 16, and seeds 0–4. The source model trains for 100 epochs on the first 70% of source observations. The target model adapts for 50 epochs on the first two days of target observations; its communication projections and scalar gates remain frozen. AdamW uses learning rates `1e-3` for source training and `1e-4` for adaptation. The forecast objective is L1 plus `0.1 ×` the observed-history deviation loss.

Evaluation uses the same post-day-two test labels for all support settings. MAE, MAPE, and RMSE are measured after restoring physical units; finite nonzero target labels are retained, with a `1e-5` MAPE denominator floor. The reported horizons are steps 1, 3, and 6. The five-seed results are in [`results/expected_main.csv`](results/expected_main.csv), with individual runs in [`results/expected_seeds.csv`](results/expected_seeds.csv). Full budgets require a GPU and can take substantial time. For a disposable pipeline check, pass `--max-updates 2` at both training stages with a separate `--output`; these checkpoints are marked non-formal.

Code is available under the [MIT license](LICENSE). See [third-party and data notices](THIRD_PARTY_NOTICES.md). Citation metadata is in [`CITATION.cff`](CITATION.cff).
