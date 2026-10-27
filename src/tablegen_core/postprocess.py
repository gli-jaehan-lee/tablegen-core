# SPDX-License-Identifier: BSD-3-Clause
# Robust outlier clamp derived from TabICL (Copyright (c) 2025, Soda team @ Inria); modified by Gauss Labs Inc.
"""Post-generation helpers: robust per-column outlier clamp and degenerate-table check."""
import warnings
import numpy as np


def outlier_clip(data, threshold=4.0, columns=None):
    """Two-pass robust clamp of each column to mean +- threshold * std.

    Pass 1 marks values outside mean +- threshold*std; pass 2 recomputes mean/std from the
    remaining values and clamps to those bounds (std floored at 1e-6). Statistics use all
    rows of the table. Returns (clipped copy, number of changed cells).
    """
    x = np.asarray(data, dtype=np.float64)
    cols = list(range(x.shape[1])) if columns is None else list(columns)
    if not cols or x.shape[0] < 2:
        return np.asarray(data).copy(), 0
    sub = x[:, cols]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean, std = sub.mean(0), np.maximum(sub.std(0, ddof=1), 1e-6)
        inlier = np.abs(sub - mean) <= threshold * std
        masked = np.where(inlier, sub, np.nan)
        m2 = np.nanmean(masked, 0)
        s2 = np.maximum(np.nanstd(masked, 0, ddof=1), 1e-6)
    m2 = np.where(np.isnan(m2), mean, m2)
    s2 = np.where(np.isnan(s2), 0.0, s2)
    clipped = np.clip(sub, m2 - threshold * s2, m2 + threshold * s2)
    out = np.asarray(data).copy()
    out[:, cols] = clipped.astype(out.dtype)
    return out, int((out[:, cols] != np.asarray(data)[:, cols]).sum())


def all_features_constant(data, features):
    """True when every input column holds a single value (an unusable training table)."""
    x = np.asarray(data)[:, :features]
    return bool(x.shape[1]) and bool((np.ptp(x, axis=0) == 0).all())
