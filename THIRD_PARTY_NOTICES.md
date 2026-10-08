# Provenance and third-party materials

The model implementation was extracted from the project's frozen GIFT experiment
snapshot, `gift_ce_www_full_20261006`. Its original `models.py` SHA-256 is
`16629612a981d53daeda3ab7be4d61ddd21f08b635b085b2e52a953ba4b03f5c`.
The small construction layer in `gift/_compat.py` extracts the local
`reference/vendor/old_models.py`, `reference/models.py`,
`legacy/screening_candidates.py`, and `legacy/candidates.py`. It preserves
initialization order, including draws from modules subsequently removed.
Those removed modules do not participate in prediction.

The training/data/metric wrappers expose the same formal protocol through
portable command-line interfaces. Server scheduling, remote backup code,
credentials, absolute workstation paths, checkpoints and experiment datasets
are excluded. No TPB, DCRNN, FEPCross or other external competitor model is
vendored in this release.

The local history encoder is conceptually related to the variate-token approach
in [iTransformer](https://github.com/thuml/iTransformer). Conceptual attribution
is distinct from distribution of an upstream implementation.

Dependencies retain their own licenses:

- [PyTorch](https://github.com/pytorch/pytorch/blob/v2.1.2/LICENSE): BSD-style license.
- [NumPy](https://github.com/numpy/numpy/blob/v1.26.4/LICENSE.txt): BSD-style license.
- Optional [pandas](https://github.com/pandas-dev/pandas/blob/v2.1.4/LICENSE) and
  [PyTables](https://github.com/PyTables/PyTables/blob/v3.9.2/LICENSE.txt): their
  upstream licenses apply.

Datasets are downloaded separately from their providers. The MIT license for
this code does not grant redistribution rights to traffic data. Preserve the
data provider's terms and attribution when obtaining or sharing datasets.
