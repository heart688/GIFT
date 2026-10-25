# GIFT

**Graph-Free Interaction for Few-Shot Traffic Forecasting Across Cities**

Anonymous code release for double-anonymous peer review. Author information is withheld during review.

This repository provides the **main GIFT model**, with source training, target adaptation, and evaluation. TIDE encodes local history and recent deviations; SIDE uses traffic-state features to exchange information between sensors through a compact positive kernel. No road graph or cross-city sensor alignment is required.

## Repository

```text
gift/
  model.py            # TIDE, SIDE, and the GIFT forecaster
  data.py             # windows and normalization
  engine.py           # source training, target adaptation, evaluation
  metrics.py          # MAE, MAPE, RMSE
configs/              # four transfer directions and dataset layouts
prepare_data.py       # optional data-layout converter
train.py              # train or adapt one seed
evaluate.py           # evaluate a checkpoint
run_experiments.py    # run selected directions and seeds
```

## Quick start

Use Python 3.10 and PyTorch 2.1.2. The tests also run on CPU.

Download the review ZIP from the anonymous code page and run the commands below from the extracted repository directory.

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
python evaluate.py --checkpoint outputs/LA_BAY/seed0/adapted.pt --data-root data --output outputs/LA_BAY/seed0/evaluation.json
```

The commands save checkpoints and evaluation results under `outputs/LA_BAY/seed0/`. Add `--device cpu` to run on CPU. Existing checkpoints are never overwritten.

To preview commands for all four directions and seeds 0–4, run `python run_experiments.py`. Add `--execute` to train, or select a smaller queue with `--directions CD_SZ --seeds 0 --execute`.

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

Each file has shape `[time, sensors, 2]`, with speed in channel 0 and normalized time of day in channel 1. The loader accepts float32 or float64 arrays and converts speed to float32. GIFT uses only the speed channel. Dataset shapes and sampling intervals are defined in `configs/datasets.json`.

The optional `prepare_data.py` converts a speed-only `.npy` or DCRNN `.h5` file to this layout. HDF5 conversion needs `pip install -r requirements-data.txt`.

```bash
python prepare_data.py --input metr-la.h5 --output data/metr-la/dataset.npy --interval-minutes 5
```

The loader checks array dimensions and finite speed values. Raw zeros are retained. Obtain and prepare the data separately; datasets and pretrained checkpoints are not included.

## Model and training

The four configurations cover METR-LA ↔ PEMS-BAY and Chengdu ↔ Shenzhen. All use 12 history steps. The US configurations predict 12 five-minute steps; the Chinese configurations predict six ten-minute steps.

GIFT uses width 128, kernel rank 16, three blocks, and batch size 16. Source training runs for 100 epochs on the first 70% of source observations. Target adaptation runs for 50 epochs on the first two target days, with communication projections and scalar gates frozen. AdamW uses learning rates `1e-3` and `1e-4`, respectively. The objective is forecast L1 plus `0.1 ×` the observed-history prediction loss.

Evaluation scores labels after the first two target days, restoring predictions and labels to the original speed scale. MAE and RMSE use speed units; MAPE is reported as a percentage with a `1e-5` denominator floor. Metrics retain finite, nonzero target labels. Results include forecast steps 1, 3, and 6.

For a short pipeline check, pass `--max-updates 2` at both training stages with a separate `--output` directory.

Code is available under the [MIT license](LICENSE). See [third-party and data notices](THIRD_PARTY_NOTICES.md). Citation metadata is in [`CITATION.cff`](CITATION.cff).
