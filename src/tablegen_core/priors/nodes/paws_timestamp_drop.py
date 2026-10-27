# -*- coding: utf-8 -*-
"""paws_timestamp_drop — PAWS_Drop = 1[SEG.TSval < TS.Recent_{t-1}], 창 안·수락 시에만 TS.Recent 갱신 [prior-920].

만드는 모양: 행 순서를 도착 순서로 보고 TSval=drift*t/n+z, 창 조건 SEQ<=RCV.NXT를 문턱으로 둔다. 수락된 갱신은 항상 비감소이므로 TS.Recent는 창 안 TSval의 누적 최대로 정확히 계산된다. 0/1 출력.

References:
- https://www.rfc-editor.org/rfc/rfc7323
"""
import numpy as np
import torch

from tablegen_core.priors._distinct_fill import derived_generator

from tablegen_core.core._function import RandomFunction


def _clean(x):
    return torch.nan_to_num(x.to(torch.float64), nan=0.0, posinf=1e6, neginf=-1e6)


class _ColumnLawMixin:
    """열마다 입력 결합과 법칙 매개를 따로 뽑아 같은 법칙식을 다시 계산하는 틀."""
    N_PROJ = 1

    def _setup(self, x):
        n_in, n_out = int(self.in_features), int(self.out_features)
        self.rng_ = derived_generator(self.sampler, NODE_NAME, n_out)
        w = self.rng_.standard_normal((n_in, self.N_PROJ * n_out))
        w /= np.sqrt((w ** 2).sum(axis=0, keepdims=True)) + 1e-12
        self.w_ = torch.as_tensor(w, dtype=torch.float64, device=x.device)
        z = _clean(x) @ self.w_
        if z.shape[0] >= 2:
            self.mu_ = z.median(dim=0).values
            self.sd_ = (z - self.mu_).abs().median(dim=0).values * 1.4826 + 1e-6
        else:
            self.mu_ = torch.zeros(z.shape[1], dtype=torch.float64, device=x.device)
            self.sd_ = torch.ones_like(self.mu_)

    def _col(self, name, low, high, use_log=False, use_int=False):
        """첫 열은 sampler 이름 HP, 나머지 열은 파생 RNG로 같은 범위에서 뽑는다."""
        n_out = int(self.out_features)
        if use_int:
            first = int(self.sampler.randint(name, low, high + 1))
            rest = self.rng_.integers(low, high + 1, max(n_out - 1, 0))
        else:
            first = float(self.sampler.numerical(name, low, high, use_log=use_log))
            u = self.rng_.random(max(n_out - 1, 0))
            rest = np.exp(np.log(low) + u * (np.log(high) - np.log(low))) if use_log else low + u * (high - low)
        vals = np.concatenate([[first], rest])[:n_out]
        return torch.as_tensor(vals, dtype=torch.float64)

    def _proj(self, x):
        z = (_clean(x) @ self.w_.to(x.device) - self.mu_.to(x.device)) / self.sd_.to(x.device)
        z = z.clamp(-8.0, 8.0)
        return z.reshape(x.shape[0], self.N_PROJ, int(self.out_features))

    def _finish(self, x, y):
        y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
        return y.to(x.dtype)


NODE_NAME = "paws_timestamp_drop"


class RandomPawsTimestampDropFunction(_ColumnLawMixin, RandomFunction):
    N_PROJ = 2

    def _fit(self, x):
        self._setup(x)
        self.drift_ = self._col("g920_drift", 0.0, 8.0)
        self.thr_ = self._col("g920_seq_thr", -1.0, 1.0)

    def _transform(self, x):
        n, n_out = x.shape[0], int(self.out_features)
        if n == 0:
            return x.new_zeros((0, n_out))
        dev = x.device
        z = self._proj(x)
        t = torch.arange(n, device=dev, dtype=torch.float64)[:, None] / max(n, 1)
        ts = self.drift_.to(dev) * t + z[:, 0]                              # SEG.TSval
        inwin = z[:, 1] <= self.thr_.to(dev)                                # SEG.SEQ <= RCV.NXT
        cand = torch.where(inwin, ts, torch.full_like(ts, -1e30))
        recent = torch.cummax(cand, dim=0).values                           # TS.Recent_t
        prev = torch.cat([torch.full_like(recent[:1], -1e30), recent[:-1]], 0)
        return self._finish(x, (ts < prev).to(torch.float64))


NODE_CLASS = RandomPawsTimestampDropFunction
