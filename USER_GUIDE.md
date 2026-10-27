# Tablegen Core

Generate synthetic tables locally from 24 selected priors (8 default + 16 robust). Use Python arrays in training code, save datasets from the command line, or explore tables in a Korean/English desktop interface. No API key, pretrained model or external dataset is required.

## Install

Python 3.10+ is required. CPU generation uses NumPy, PyTorch and SciPy. Tk is needed only for the GUI. Download this repository as a ZIP and extract it, then run these commands **inside the extracted repository**. Create the environment outside a cloud-synced directory.

```bash
python3 -m venv ~/.venvs/tablegen-core
source ~/.venvs/tablegen-core/bin/activate
python -m pip install .
python -m pip check
```

On Windows, create a venv at a suitable local path and activate `Scripts\activate`; Windows has not been execution-tested. The first dependency installation needs Internet access. This is a Python package, not a bundled Python application.

## Generate arrays for training

```python
from tablegen_core import generate

result = generate(mode="direct", priors=["lin"], rows=128,
                  features=4, targets=1, seed=42)
X, y = result.data[:, :4], result.data[:, 4]
# Pass X and y to your own training code.
# Save files only if needed (destination must not already exist):
result.write("output/example")
```

The input columns precede target columns. Data generation alone does not train a model. This runnable example splits a generated table, fits preprocessing on training rows only, performs PyTorch updates and reports held-out error:

```bash
python examples/training_quickstart.py
python examples/training_quickstart.py --rows 512 --features 8 --seed 73 --steps 200 --output output/training-demo
```

`parameters_changed: true` confirms a real model update. The error is in standardized target units; this is a usage smoke test, not a performance benchmark. See training guide (Korean) for leakage, seeds, split semantics and batching.

## Command line and GUI

```bash
tablegen-core generate --mode direct --prior lin --rows 128 --features 4 --seed 42 --output output/demo
tablegen-core-gui
```

CSV stores the full table; metadata records the configuration and provenance; `report.html` provides offline visualization including 3D. Reusing an output directory is rejected. The GUI contains a User guide tab and a Korean/English selector. The selected GUI defaults may differ from API defaults: always specify mode, prior and graph settings when reproducing a GUI result.

On macOS, if launching from Downloads produces `Operation not permitted`, the operating system has denied file access. Check the Terminal privacy permission through System Settings; inspect the underlying error before changing permissions.

## Defaults and training tips

Defaults in 0.2.0 reflect how we use the generator for model pre-training:

| Setting | Default | What it does |
|---|---|---|
| `restrict_cheap_functions` | auto | On in DAG mode when the prior pool contains a cheap function (`gp`, `tree`, `disc`, `lin`, `quad`); off in direct mode. Turning it off lowered downstream quality in our runs. |
| `distinct_fill` | `True` | Repairs duplicated columns. Keep it on; turning it off is faster but changes the data. |
| `outlier_clip` | `4.0` | Robust two-pass clamp of every numeric column: values beyond mean ± 4·std are marked, mean/std are recomputed from the rest, and values are clamped to those bounds. Categorical columns are untouched. `None` disables it. |
| `max_table_attempts` | `32` | A table that is nonfinite, rejected by the graph sampler, or has only constant input columns is redrawn with a derived seed. Rejections are listed in `metadata["table_attempts"]`. `1` disables redraws. |

Tips:

* **Generate a fresh table at every training step** instead of pre-generating a large pool. With a few `DataLoader` workers, generation is usually not the bottleneck.
* **Custom priors and multiprocessing**: register your own prior in every worker process (e.g. in `worker_init_fn`), not only in the main process.

### Caveats

* `outlier_clip` uses statistics from **all rows of the table**, including rows you may later hold out for evaluation. If you need train-only statistics, generate with `outlier_clip=None` and clip after splitting.
* Rare-event columns (almost always one value) can still blow up if you standardize with train-row statistics only; floor the standard deviation or clip after standardizing.
* A redrawn table uses a different seed from the one you requested; `metadata["used_seed"]` records it.
* 0.2.0 output differs from 0.1.x for the same seed. To reproduce 0.1.x exactly, pass `outlier_clip=None, max_table_attempts=1, restrict_cheap_functions=False`.
* For bit-identical reruns, also fix `PYTHONHASHSEED` (e.g. `PYTHONHASHSEED=0`) and pin package versions.

## Reproducibility and limits

Use the same package version, configuration, seed and dependency versions. Pin `PYTHONHASHSEED=0` when reproducing historical runs. The tested current graph group ordering is deterministic, but this is not a guarantee across different software or hardware. Constants, duplicate information and imbalanced classes may be valid outputs; a generated table is not automatically suitable for a task. Preserve metadata and test your task assumptions. CPU generation is supported; large requested tables can be expensive.

## Checks

```bash
python -m pip install '.[dev]'
python -m unittest discover -s tests -p 'test_training_example.py'
python tools/check_release.py . --edition core
```

The repository includes a CI workflow for these checks.

## License

Original code in this repository is released under the BSD 3-Clause License, Copyright (c) 2026 Gauss Labs Inc.

Third-party components keep their own licenses:

| Component | Origin | License |
|---|---|---|
| Graph engine and built-in functions (`core/`) | [TabICL](https://github.com/soda-inria/tabicl), Soda team @ Inria | BSD 3-Clause (text) |
| Nori-derived priors (files with an `SPDX-License-Identifier: Apache-2.0` header) | [synthefy-nori](https://github.com/Synthefy/synthefy-nori) @ c0a8ac2, Synthefy | Apache 2.0 (text, NOTICE) |

See THIRD_PARTY_NOTICES.md for details. Priors that implement published formulas cite their sources in the module docstring.
