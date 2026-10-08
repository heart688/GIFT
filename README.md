# GIFT

Reproducibility code for **GIFT: Graph-Free Interaction for Few-Shot Traffic
Forecasting Across Cities**. GIFT learns communication from traffic histories
and reuses the model across cities with different sensor sets.

This lightweight release contains the model, its ablations and communication
controls, data processing, training/adaptation/evaluation entry points, four
direction configurations, metrics, and recorded expected results. Download
datasets separately. Checkpoints are generated locally by training.

## Environment

The formal GPU experiments used Python **3.10.8**, PyTorch **2.1.2+cu121**,
CUDA **12.1**, cuDNN **8.9.2**, and an NVIDIA RTX 4090 D. Training used FP32,
disabled TF32, and deterministic PyTorch operations. The release pins NumPy
**1.26.4**; the frozen GPU logs do not establish the original NumPy version.

Create a Python 3.10 environment and install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The CUDA installation command follows the [official PyTorch 2.1.2 instructions](https://pytorch.org/get-started/previous-versions/#v212).
For CPU-only checks, install `requirements.txt` directly and use `--device cpu`.
CPU smoke tests exercise the release; full training budgets are intended for a GPU.
The release was also tested on macOS arm64 with Python **3.9.6**, PyTorch
**2.1.2** and NumPy **1.26.4**. `VERIFICATION.json` records the local checks:
32 model equivalence combinations, 216 parameter-policy audits, data-pipeline
equivalence, and disposable pretrain/adapt/evaluation command-line runs.

## Data

Obtain METR-LA, PEMS-BAY, Chengdu and Shenzhen through the data links in the
[TPB authors' repository](https://github.com/zhyliu00/TPB), including their
[processed-data download](https://drive.google.com/drive/folders/1UrKTgR27YmP9PjJ-FWv4SCDH3zUxtc5R?usp=share_link).
[DCRNN](https://github.com/liyaguang/DCRNN) also provides access to raw METR-LA
and PEMS-BAY. Follow provider access terms.

Arrange the processed data as:

```text
data/
  metr-la/dataset.npy
  pems-bay/dataset.npy
  chengdu/dataset.npy
  shenzhen/dataset.npy
```

Each array has shape `[time, nodes, 2]`: traffic speed followed by time of day.
The model consumes speed. Zero speeds remain in the data as missing-observation
markers; do not impute, reorder sensors, subsample, or change units when
reproducing the recorded results. `configs/datasets.json` specifies shape,
sampling interval and exact file SHA-256 for the benchmark used in our runs.
File hashes are verified by default. Public downloads may contain different
versions; the manifest, rather than a download link alone, identifies the
reported data. `--skip-data-hash` allows a different version with the same shape
and marks the data as unverified in output metadata.

The graph-free models need no `matrix.npy`. For `GRAPH_CONSTRAINED` alone, put
the matching road-affinity matrix in each city's folder as `matrix.npy`.
Its hash is also recorded in the manifest. Positive edges plus self determine
allowed senders; the correct source/target graph is used at each stage.

If starting from a speed-only `.npy` or a DCRNN `.h5`, the optional converter
creates the required two-channel layout. Conversion does not guarantee a
byte-identical benchmark file; compare the generated SHA with the manifest.

```bash
python -m pip install -r requirements-data.txt  # only needed for .h5
python prepare_data.py --input metr-la.h5 --output data/metr-la/dataset.npy --interval-minutes 5
```

## Run one direction

```bash
python train.py --stage pretrain --config configs/la_bay.json --data-root data --seed 0
python train.py --stage adapt --config configs/la_bay.json --data-root data --seed 0
python evaluate.py --checkpoint outputs/LA_BAY/GIFT_CE/seed0/adapted.pt --data-root data
```

The default output directory contains `source.pt`, `adapted.pt`, parameter
audits, training logs, and source/adapted results. Source weights are loaded
strictly before adaptation. The final checkpoint is evaluated without target
validation selection or early stopping. Existing checkpoints are retained;
direct training refuses to overwrite them.

To inspect or run all four directions and five seeds:

```bash
python run_experiments.py
python run_experiments.py --execute
# A single requested run:
python run_experiments.py --directions CD_SZ --seeds 0 --execute
```

`--execute` starts the listed full-budget jobs sequentially. Each source job
is trained once and reused for its target adaptation. Short diagnostics can
use `--max-updates 2` at both training stages and a separate `--output`; their
outputs are marked non-formal and should not be compared to full-budget results.

## Fixed protocol

| Config | Source → target | Sampling | Forecast length | Source updates | Adaptation updates |
|---|---|---|---|---:|---:|
| `la_bay.json` | METR-LA → PEMS-BAY | 5 min | 12 steps | 149,700 | 1,700 |
| `bay_la.json` | PEMS-BAY → METR-LA | 5 min | 12 steps | 227,800 | 1,700 |
| `cd_sz.json` | Chengdu → Shenzhen | 10 min | 6 steps | 75,400 | 800 |
| `sz_cd.json` | Shenzhen → Chengdu | 10 min | 6 steps | 75,400 | 800 |

All runs use history length 12, width 128, kernel rank 16, three blocks, FFN
width 256, dropout 0.1, and seeds 0–4. Source training uses the first 70% of
source observations for 100 epochs. Target adaptation uses the first two days
for 50 epochs. Both use batch size 16, shuffled complete batches, AdamW,
weight decay `1e-4`, and gradient clipping at 5. Learning rates are `1e-3` for
source training and `1e-4` for adaptation. The configuration update counts are
checked against actual window counts.

Training windows keep histories and labels inside the training/support
interval. Target test labels begin at day two and extend to the data end;
the first test window can use its preceding support history. A single speed
mean/population standard deviation is fitted over source training for
pretraining, and over target support for adaptation. Zero observations are
included in these statistics. The model additionally normalizes each node's
input history within its own window and restores that scale for prediction.

Forecast training uses **unmasked L1** in dataset-normalized coordinates.
E-enabled variants add `0.1 ×` masked observed-history prediction L1, requiring
all four involved observations to be valid. This auxiliary target comes from
the input history. It does not use target test labels.

Default adaptation freezes every block's Q/K/V/output projection, including
biases, its scalar gate, and the E output projection. History/state/deviation
encoders, the history predictor, layer normalizations, FFNs, and forecast head
remain trainable. Audits report each parameter's status.

## Variants and adaptation policies

Pass the same `--variant` to source training and adaptation:

| Variant | Included paths / communication |
|---|---|
| `GIFT_BASE` | History encoder and GIK |
| `C_ONLY` | State statistics added to first-block Q/K |
| `E_ONLY` | Historical prediction deviations added to first-block V |
| `GIFT_CE` | Both C and E; main release configuration |
| `LOCAL_INFO_MATCHED` | Local computation with C/E information; no node communication |
| `GRAPH_CONSTRAINED` | C/E communication limited to road neighbours |
| `DENSE_1H` | C/E with single-head dense attention |
| `UNIFORM_E` | Uniform node pooling with E content |

`--policy FULL` adapts all parameters; `HEAD_ONLY` updates the final linear
forecast head with final LayerNorm frozen; `INTERACTION_ONLY` updates
Q/K/V/output projections, scalar gates, and the E output projection;
`RESET_QK` reinitializes Q/K and then applies default freezing. Other trainable
parameters keep the source initialization. The default is `DEFAULT_CE`.
Use a distinct output directory for each policy while reusing its matching
source checkpoint. Changing the communication variant requires matching
source training.

`evaluate.py --zero-day` evaluates a source checkpoint with source normalization
statistics and per-window normalization, using no target support fitting.
It uses the same canonical target labels and mask as the two-day comparison.

## Metrics and recorded results

`gift/metrics.py` accumulates MAE, MAPE (%) and RMSE in float64 on restored
physical-unit predictions. Labels use the formal float32 normalization and
restoration round trip in first-two-day target coordinates. The shared mask
keeps finite, nonzero restored labels. MAPE divides by `max(abs(label), 1e-5)`.
This mask is applied to evaluation metrics; training's forecast L1 is unmasked.

Reported steps are 1, 3 and 6: **5/15/30 min** in METR-LA and PEMS-BAY,
**10/30/60 min** in Chengdu and Shenzhen. The evaluator also reports every
forecast step and pooled metrics. Predictions are streamed through metrics;
large prediction files are only generated when `--save-predictions` is used.

`results/expected_main.csv` records the four-direction GIFT_CE means and sample
standard deviations (`ddof=1`) from five actual seeds. `results/expected_seeds.csv`
contains the corresponding individual seed values. These are measured formal
GPU results, not synthetic test outputs. Hardware, library versions and
floating-point kernels can affect exact numerical reproduction.

## Contents and citation

- `gift/models.py`, `gift/_compat.py`: complete model and construction compatibility.
- `gift/data.py`, `prepare_data.py`: data loading, windows, normalization and layout conversion.
- `gift/engine.py`, `train.py`, `evaluate.py`: source training, adaptation and evaluation.
- `configs/`: directions, seeds, budgets, modules and data fingerprints.
- `tests/`: small CPU checks; `results/`: expected metrics.

Code is distributed under MIT; see `LICENSE` and `THIRD_PARTY_NOTICES.md`.
GitHub can read `CITATION.cff` for the software citation. The associated
manuscript title is given above; author, venue, DOI and paper URL should be
added when confirmed. This release does not assert an accepted venue.
