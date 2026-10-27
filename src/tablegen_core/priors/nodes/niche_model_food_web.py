# -*- coding: utf-8 -*-
"""niche_model_food_web — Williams–Martinez 니치 모형의 섭식구간 연결 [prior-416].

만드는 모양: 각 행 = 한 종. 입력에서 니치값 n_i∈(0,1), 섭식폭 r_i = x_i n_i
(x_i ~ Beta(1,b), b=(1-2C)/2C), 섭식중심 c_i ~ U(r_i/2, n_i) 를 만든다. 출력 열 k 는
표의 니치 분포 분위 q_k 에 해당하는 실제 기준종 k 와의 연결 A∈{0,1}:
포식 시점 1[|n_k - c_i| ≤ r_i/2] 또는 피식 시점 1[|n_i - c_k| ≤ r_k/2]. 서명
`mat-thr-pair-bnd-bin`.
"""
import math

import torch

from tablegen_core.core._function import RandomFunction  # noqa: F401

NODE_NAME = "niche_model_food_web"

_MECHS = ["williams_martinez_niche"]


class RandomNicheModelFoodWebFunction(RandomFunction):
    def _draw(self, name, low, high, shape, x):
        a = self.sampler.numerical(name, low, high, size=shape, mode="local")
        return torch.as_tensor(a, device=x.device, dtype=x.dtype)

    def _hash_u(self, xs):
        z = xs.double() @ self.HA_.double() + self.HB_.double()
        h = torch.sin(z) * 43758.5453
        return (h - torch.floor(h)).clamp(1e-6, 1.0 - 1e-6).to(xs.dtype)

    def _fit(self, x: torch.Tensor):
        d = max(1, int(self.in_features))
        o = max(1, int(self.out_features))
        self.mech_ = self.sampler.choice("g416_mech", _MECHS)
        C = float(self.sampler.numerical("g416_conn", 0.05, 0.3))
        self.b_ = (1.0 - 2.0 * C) / (2.0 * C)
        w = self._draw("g416_Wn", -1.0, 1.0, (d, 1), x)
        self.Wn_ = w / w.norm().clamp_min(1e-6)
        self.HA_ = self._draw("g416_hashA", -40.0, 40.0, (d, 2), x)
        self.HB_ = self._draw("g416_hashB", 0.0, 2.0 * math.pi, (2,), x)
        self.q_ = self._draw("g416_proto_q", 0.03, 0.97, (o,), x)
        self.role_ = self._draw("g416_role", 0.0, 1.0, (o,), x) < 0.5   # True=포식 시점
        self.o_ = o

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        if n == 0:
            return x.new_empty((0, self.out_features))
        xs = torch.nan_to_num(x, nan=0.0, posinf=6.0, neginf=-6.0).clamp(-6.0, 6.0)
        s = (xs @ self.Wn_).squeeze(1)
        s = (s - s.mean()) / s.std(unbiased=False).clamp_min(1e-6) if n > 1 else s
        ni = (0.5 * (1.0 + torch.erf(s / math.sqrt(2.0)))).clamp(1e-4, 1.0 - 1e-4)
        u = self._hash_u(xs)
        xi = 1.0 - (1.0 - u[:, 0]) ** (1.0 / self.b_)
        ri = xi * ni
        ci = ri / 2.0 + u[:, 1] * (ni - ri / 2.0)

        # 기준종: 니치 순위가 해당 분위에 놓인 실제 표 행
        order = torch.argsort(ni)
        target = order[(self.q_ * (n - 1)).long()]
        nk, rk, ck = ni[target], ri[target], ci[target]

        # 포식 시점: (n_k - c_i)^2 - (r_i/2)^2  (행쌍 행렬곱)
        Li = torch.stack([ci ** 2 - ri ** 2 / 4.0, -2.0 * ci, torch.ones_like(ci)], dim=1)
        Rk = torch.stack([torch.ones_like(nk), nk, nk ** 2], dim=0)
        Dpred = Li @ Rk
        # 피식 시점: (n_i - c_k)^2 - (r_k/2)^2
        Li2 = torch.stack([torch.ones_like(ni), ni, ni ** 2], dim=1)
        Rk2 = torch.stack([ck ** 2 - rk ** 2 / 4.0, -2.0 * ck, torch.ones_like(ck)], dim=0)
        Dprey = Li2 @ Rk2
        D = torch.where(self.role_.unsqueeze(0), Dpred, Dprey)

        frac = (D <= 0).to(D.dtype).mean(0, keepdim=True)
        degen = torch.logical_or(frac == 0, frac == 1)
        thr = torch.where(degen, torch.quantile(D, 0.5, dim=0, keepdim=True), torch.zeros_like(frac))
        out = (D <= thr).to(x.dtype)
        out = torch.nan_to_num(out[:, : self.out_features], nan=0.0, posinf=1e6, neginf=-1e6)
        return torch.nan_to_num(torch.clamp(out, -1e6, 1e6), nan=0.0)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


NODE_CLASS = RandomNicheModelFoodWebFunction
