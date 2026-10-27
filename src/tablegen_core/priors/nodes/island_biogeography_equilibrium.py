# -*- coding: utf-8 -*-
"""island_biogeography_equilibrium — MacArthur–Wilson 섬 생물지리 평형 종수 [prior-488].

만드는 모양: 각 행 = 한 섬. 입력에서 본토 거리 D, 면적 A, 종 풀 S_p 를 만들고
이민률 I_0 = I_max e^{-D/λ}, 절멸률 E_0 = E_max A^{-z} 로
I_0(S_p - S) = E_0 S 의 평형 S* = I_0 S_p/(I_0+E_0) 를 낸다. 첫 열 S*, 이어
점유율 S*/S_p ∈(0,1), 평형 교체율 T = I_0E_0S_p/(I_0+E_0), 빈 섬에서 시각 t 의 종수
S(t) = S*(1-e^{-(I_0+E_0)t}), 이완시간 1/(I_0+E_0), 그 뒤는 다른 t 의 S(t). 모두 양수.
서명 `many2one-sat-none-pos-cont`.
"""
import torch

from tablegen_core.core._function import RandomFunction  # noqa: F401

NODE_NAME = "island_biogeography_equilibrium"

_MECHS = ["macarthur_wilson_equilibrium"]


class RandomIslandBiogeographyEquilibriumFunction(RandomFunction):
    def _draw(self, name, low, high, shape, x):
        a = self.sampler.numerical(name, low, high, size=shape, mode="local")
        return torch.as_tensor(a, device=x.device, dtype=x.dtype)

    def _fit(self, x: torch.Tensor):
        d = max(1, int(self.in_features))
        o = max(1, int(self.out_features))
        self.mech_ = self.sampler.choice("g488_mech", _MECHS)
        self.logImax_ = float(self.sampler.numerical("g488_logImax", -2.3, 1.6))
        self.logEmax_ = float(self.sampler.numerical("g488_logEmax", -3.0, 0.7))
        self.z_ = float(self.sampler.numerical("g488_z", 0.15, 0.35))
        self.lam_ = float(self.sampler.numerical("g488_lambda", 0.5, 3.0))
        self.logSp_ = float(self.sampler.numerical("g488_logSp", 3.0, 6.2))
        self.gd_ = float(self.sampler.numerical("g488_gdist", 0.3, 0.9))
        self.ga_ = float(self.sampler.numerical("g488_garea", 0.6, 1.8))
        self.gs_ = float(self.sampler.numerical("g488_gpool", 0.1, 0.5))
        W = self._draw("g488_W", -1.0, 1.0, (d, 3), x)
        self.W_ = W / W.norm(dim=0, keepdim=True).clamp_min(1e-6)
        self.t_ = torch.exp(self._draw("g488_logt", -1.5, 2.0, (max(2, o - 3),), x))
        self.o_ = o

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        if n == 0:
            return x.new_empty((0, self.out_features))
        xs = torch.nan_to_num(x, nan=0.0, posinf=6.0, neginf=-6.0).clamp(-6.0, 6.0)
        lin = xs @ self.W_
        dist = torch.exp((self.gd_ * lin[:, 0]).clamp(-5.0, 5.0))
        logA = (self.ga_ * lin[:, 1]).clamp(-10.0, 10.0)
        logSp = self.logSp_ + (self.gs_ * lin[:, 2]).clamp(-3.0, 3.0)
        logI = (self.logImax_ - dist / self.lam_).clamp(-30.0, 5.0)
        logE = (self.logEmax_ - self.z_ * logA).clamp(-30.0, 5.0)
        frac = torch.sigmoid(logI - logE)                         # I_0/(I_0+E_0)
        Sp = torch.exp(logSp)
        Sstar = frac * Sp
        rate = (torch.exp(logI) + torch.exp(logE)).clamp_min(1e-12)
        turnover = torch.exp(logE) * Sstar                        # = I_0 E_0 S_p/(I_0+E_0)
        tau = (1.0 / rate).clamp(max=1e6)
        St = Sstar.unsqueeze(1) * (-torch.expm1(-(rate.unsqueeze(1) * self.t_.unsqueeze(0)).clamp(max=50.0)))
        cols = [Sstar.unsqueeze(1), frac.unsqueeze(1), turnover.unsqueeze(1), St[:, :1], tau.unsqueeze(1), St[:, 1:]]
        out = torch.cat(cols, dim=1)
        out = torch.nan_to_num(out[:, : self.out_features], nan=0.0, posinf=1e6, neginf=-1e6)
        return torch.nan_to_num(torch.clamp(out, -1e6, 1e6), nan=0.0)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


NODE_CLASS = RandomIslandBiogeographyEquilibriumFunction
