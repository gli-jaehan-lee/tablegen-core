# -*- coding: utf-8 -*-
"""tetragonal_interplanar_spacing — 정방정 (hkl) 면간거리 [prior-572].

만드는 모양: 행별 h,k,l 에서 d^{-2}=(h²+k²)/a²+l²/c² 를 계산한 양수 표.
"""
import torch
from tablegen_core.core._function import RandomFunction  # noqa: F401

NODE_NAME = "tetragonal_interplanar_spacing"


class RandomTetragonalInterplanarSpacingFunction(RandomFunction):
    def _fit(self, x: torch.Tensor):
        o = self.out_features
        self.a_ = torch.as_tensor(self.sampler.numerical("g572_a", 0.7, 2.5, size=(1, o), mode="local"), device=x.device, dtype=x.dtype)
        self.c_ = torch.as_tensor(self.sampler.numerical("g572_c", 0.7, 2.5, size=(1, o), mode="local"), device=x.device, dtype=x.dtype)
        self.shift_ = torch.as_tensor(self.sampler.numerical("g572_hkl_shift", 0.1, 1.0, size=(1, 3), mode="local"), device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return x.new_empty((0, self.out_features))
        z = torch.nan_to_num(x, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)
        h = z[:, 0:1].abs() + self.shift_[:, 0:1]
        k = z[:, 1:2].abs() + self.shift_[:, 1:2] if self.in_features > 1 else (0.7 * z[:, 0:1] + self.shift_[:, 1:2]).abs()
        l = z[:, 2:3].abs() + self.shift_[:, 2:3] if self.in_features > 2 else (1.3 * z[:, 0:1] - self.shift_[:, 2:3]).abs()
        inverse_square = (h.square() + k.square()) / self.a_.square() + l.square() / self.c_.square()
        out = torch.rsqrt(inverse_square.clamp_min(1e-8))
        return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
