# -*- coding: utf-8 -*-
"""modulated_satellite_structure_factor — 정현 변조 위성 구조인자 F_m∝Σ f_j T_j J_m(2πH·u_j)e^{iφ_j} [prior-611].

만드는 모양: 각 행이 반사 하나(주 반사 h 의 연속 좌표)인 표. 출력 열마다 위성
차수 `m ∈ {0..m_max}` 가 고정되고, 산란벡터 `H=h+m·q`(q 는 비정합 변조 파수벡터)에서
`F_m = Σ_j f_j T_j(H) J_m(2π H·u_j) exp(i(2π h·r_j − mφ_j))` 를 계산해
`|F|²`(세기)·`|F|`·`log(1+|F|²)` 중 하나를 양수로 낸다(서명 many2one-osc-none-pos-cont).
m=0 열은 주 반사(J₀ 로 약화), m≥1 열은 위성 — 위성 세기가
**베셀 함수의 진동·영점**을 따라 H·u 에 대해 들쭉날쭉 꺼지는 모양이다.
⚠️ 근사: cont 서명에 맞춰 정수 반사 사이의 h도 연속 평가한다. 위성 차수 m은 정수로 유지한다.
"""
import math

import torch

from tablegen_core.core._function import RandomFunction  # noqa: F401

NODE_NAME = "modulated_satellite_structure_factor"

_OUTS = ["intensity", "amplitude", "log_intensity"]
_NQ = 64


class RandomModulatedSatelliteStructureFactorFunction(RandomFunction):
    def _fit(self, x: torch.Tensor):
        d = max(1, int(self.in_features))
        o = max(1, int(self.out_features))
        self.outm_ = self.sampler.choice("g611_out", _OUTS)
        self.natom_ = int(self.sampler.randint("g611_natom", 2, 7))
        self.mmax_ = int(self.sampler.randint("g611_mmax", 1, 4))
        self.gain_ = float(self.sampler.numerical("g611_hkl_gain", 1.5, 4.0))
        self.uamp_ = float(self.sampler.numerical("g611_u_amp", 0.03, 0.25, use_log=True))
        self.bdw_ = float(self.sampler.numerical("g611_debye_waller", 0.05, 1.5, use_log=True))
        seed = int(self.sampler.randint("g611_seed", 0, 2_147_483_647))
        self._gen_ = torch.Generator(device=x.device)
        self._gen_.manual_seed(seed)
        g, dev, dt = self._gen_, x.device, x.dtype
        J = self.natom_

        self.W_ = torch.randn(d, 3 * o, generator=g, device=dev, dtype=dt) / (d ** 0.5)
        self.c_ = torch.randn(1, 3 * o, generator=g, device=dev, dtype=dt) * 1.5
        self.m_ = torch.randint(0, self.mmax_ + 1, (o,), generator=g, device=dev).to(dt)
        self.q_ = torch.rand(o, 3, generator=g, device=dev, dtype=dt) * 0.6 - 0.3   # 비정합 q
        self.f_ = 0.5 + torch.rand(J, generator=g, device=dev, dtype=dt) * 2.0        # 산란인자
        self.r_ = torch.rand(J, 3, generator=g, device=dev, dtype=dt)                 # 평균 위치
        self.u_ = torch.randn(J, 3, generator=g, device=dev, dtype=dt) * self.uamp_   # 변위 진폭
        self.phi_ = torch.rand(J, generator=g, device=dev, dtype=dt) * (2.0 * math.pi)
        # 베셀 적분 절점 (사다리꼴, 끝점 절반 가중)
        tau = torch.linspace(0.0, math.pi, _NQ + 1, device=dev, dtype=dt)
        w = torch.full((_NQ + 1,), 1.0, device=dev, dtype=dt)
        w[0] = 0.5
        w[-1] = 0.5
        self.tau_ = tau
        self.wq_ = w / _NQ
        self.hscale_ = 3.0 * (self.gain_ * 1.5 + 1.5) ** 2
        self.o_ = o

    def _bessel(self, m: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # J_m(z) = (1/π)∫₀^π cos(mτ − z sinτ) dτ ; m:(o,), z:(n,o,J)
        arg = m.view(1, -1, 1, 1) * self.tau_ - z.unsqueeze(-1) * torch.sin(self.tau_)
        return (torch.cos(arg) * self.wq_).sum(dim=-1)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        o = self.o_
        h = self.gain_ * (x @ self.W_) + self.c_
        h = torch.clamp(h, -40.0, 40.0).view(n, o, 3)                        # 연속 주 반사 좌표
        H = h + self.m_.view(1, o, 1) * self.q_.unsqueeze(0)                 # H = h + m q
        Hu = torch.einsum("noc,jc->noj", H, self.u_)                         # H·u_j
        z = torch.clamp(2.0 * math.pi * Hu, -40.0, 40.0)
        Jm = self._bessel(self.m_, z)                                        # (n,o,J)
        H2 = (H * H).sum(dim=2, keepdim=True)                                # |H|²
        T = torch.exp(-torch.clamp(self.bdw_ * H2 / self.hscale_, max=30.0))  # (n,o,1)
        ph = 2.0 * math.pi * torch.einsum("noc,jc->noj", h, self.r_) \
            - self.m_.view(1, o, 1) * self.phi_.view(1, 1, -1)
        amp = self.f_.view(1, 1, -1) * T * Jm
        Fr = (amp * torch.cos(ph)).sum(dim=2)
        Fi = (amp * torch.sin(ph)).sum(dim=2)
        I = Fr * Fr + Fi * Fi

        if self.outm_ == "intensity":
            out = I
        elif self.outm_ == "amplitude":
            out = torch.sqrt(I + 1e-12)
        elif self.outm_ == "log_intensity":
            out = torch.log1p(I)
        else:
            raise ValueError(f"unknown g611_out: {self.outm_}")

        out = torch.nan_to_num(out[:, : self.out_features], nan=0.0, posinf=1e6, neginf=-1e6)
        return torch.clamp(out, 1e-12, 1e6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._transform(x)


NODE_CLASS = RandomModulatedSatelliteStructureFactorFunction
