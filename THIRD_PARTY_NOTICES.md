# Third-party notices

## TabICL — BSD 3-Clause

Source: https://github.com/soda-inria/tabicl — Copyright (c) 2025, Soda team @ Inria.

* `src/tablegen_core/core/` — graph engine and built-in functions, adapted (namespace rewrite and extensions).
* `src/tablegen_core/postprocess.py` — robust outlier clamp adapted from TabICL's `_reg2cls` reference.

Files carry an `SPDX-License-Identifier: BSD-3-Clause` header. Full text: [licenses/TABICL_LICENSE](licenses/TABICL_LICENSE). TabICL's Apache-2.0 `forecast` component is **not** included.

## Nori — Apache License 2.0

Source: https://github.com/Synthefy/synthefy-nori @ c0a8ac2 — Copyright 2026 Synthefy.

* 1 prior module(s) under `src/tablegen_core/priors/nodes/` (each with an `SPDX-License-Identifier: Apache-2.0` header and a source line naming the upstream file and lines). Modified by Gauss Labs Inc.: Torch vectorisation, multi-column output, numerical guards.

Full text: [licenses/NORI_LICENSE](licenses/NORI_LICENSE). Attribution: [NOTICE](NOTICE).

## Published formulas

Other priors are original implementations of formulas from the scientific literature. Each module docstring lists its references (papers, standards, DOIs). Citing a source does not mean code was copied from it.
