# -*- coding: utf-8 -*-
"""bch_error_locator — 이진 오류의 GF(256) 신드롬과 역수 근 locator를 계산한다 [prior-818].

만드는 모양: 행=오류가 있는 이진 BCH 블록의 잠재 오류 패턴. 실수 입력을
열별 투영·양자화하여 서로 겹치지 않는 v개 위치 i_l로 바꾼다(e_i=1).
GF(256), primitive polynomial 0x11d, alpha=2, 블록 길이 255에서
S_j=XOR_l alpha^(j*i_l), Lambda(z)=prod_l(1+alpha^i_l*z)를 계산한다.
출력은 열별 j의 신드롬 또는 열별 질의 z에서의 locator 값(0..255)이다.
Lambda(alpha^(-i_l))=0이며 완전한 BCH 인코더/복호기는 아닌 오류 대수 핵이다.

References:
- https://downloads.bbc.co.uk/rd/pubs/whp/whp-pdf-files/WHP031.pdf
- https://doi.org/10.1002/ecja.4410640204
"""
import numpy as np
import torch

from tablegen_core.priors._distinct_fill import derived_generator
from tablegen_core.core._function import RandomFunction

NODE_NAME = "bch_error_locator"


def _field(device):
    # 255개 체 원소만 순회한다. 관측 행 루프는 없다.
    powers = np.empty(255, dtype=np.int64)
    value = 1
    for k in range(255):
        powers[k] = value
        value <<= 1
        if value & 256:
            value ^= 0x11d
    logs = np.zeros(256, dtype=np.int64)
    logs[powers] = np.arange(255)
    return (torch.as_tensor(powers, device=device), torch.as_tensor(logs, device=device))


def _mul(a, b, powers, logs):
    product = powers[(logs[a] + logs[b]) % 255]
    return torch.where((a == 0) | (b == 0), 0, product)


class RandomBCHErrorLocatorFunction(RandomFunction):
    def _fit(self, x: torch.Tensor):
        self.nout_ = int(self.out_features)
        self.v_ = int(self.sampler.randint("g818_errors", 2, 5, mode="local"))
        self.mode_ = self.sampler.choice("g818_output", ["syndrome", "locator"], mode="local")
        self.gain_ = float(self.sampler.numerical("g818_gain", 0.5, 2., mode="local"))
        rng = derived_generator(self.sampler, NODE_NAME, self.nout_)
        shape = (int(self.in_features), self.nout_, self.v_ + 1)
        w = rng.standard_normal(shape)
        w /= np.maximum(np.linalg.norm(w, axis=0, keepdims=True), 1e-12)
        self.w_ = torch.as_tensor(w.reshape(shape[0], -1), dtype=torch.float64, device=x.device)
        self.offset_ = torch.as_tensor(rng.uniform(-1., 1., shape[1:]), dtype=torch.float64, device=x.device)
        self.j_ = torch.as_tensor(rng.integers(1, 2 * self.v_ + 1, self.nout_), device=x.device)
        self.powers_, self.logs_ = _field(x.device)

    def _pattern(self, x):
        clean = torch.nan_to_num(x.to(torch.float64), nan=0., posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
        z = (clean @ self.w_.to(x.device)).reshape(x.shape[0], self.nout_, self.v_ + 1)
        unit = (torch.tanh(self.gain_ * z + self.offset_.to(x.device)) + 1) / 2
        # 서로 겹치지 않는 위치 대역으로 중복 위치를 피한다. 위치는 255 미만.
        band = 255 // self.v_
        positions = (unit[..., :self.v_] * band).long().clamp(0, band - 1)
        positions += torch.arange(self.v_, device=x.device) * band
        query = (unit[..., -1] * 256).long().clamp(0, 255)
        return positions, query

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[0] or not self.nout_:
            return x.new_zeros((x.shape[0], self.nout_))
        positions, query = self._pattern(x)
        powers, logs = self.powers_.to(x.device), self.logs_.to(x.device)
        if self.mode_ == "syndrome":
            result = torch.zeros_like(query)
            for k in range(self.v_):
                result ^= powers[(positions[..., k] * self.j_.to(x.device)) % 255]
        else:
            result = torch.ones_like(query)
            for k in range(self.v_):
                factor = 1 ^ _mul(powers[positions[..., k]], query, powers, logs)
                result = _mul(result, factor, powers, logs)
        return torch.nan_to_num(result.to(x.dtype), nan=0., posinf=255., neginf=0.).clamp(0, 255)


NODE_CLASS = RandomBCHErrorLocatorFunction
