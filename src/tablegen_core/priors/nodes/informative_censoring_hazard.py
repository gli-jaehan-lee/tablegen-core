# -*- coding: utf-8 -*-
"""informative_censoring_hazard — 공유 잠재로 위험과 절단을 상관시킨 정보성 절단 [prior-63].

만드는 모양: 생존 레코드 (min(T,C), 1[T<=C], log 노출) 는 `hazard_survival_emitter`
와 같지만, **절단 자체가 결과와 정보를 공유한다** — 관측되지 않은 공통 요인
Z=softplus(h(x)) 가 사건 위험률과 절단 위험률에 **동시에** 실려 있다. 병이 심한
환자가 더 일찍 추적탈락하고, 곧 고장날 장비가 먼저 교체돼 관측에서 빠지는
상황이다. 절단 표시를 무시하고 회귀하면 편향이 생기는, 실데이터의 함정.
기존 58종이 못 만드는 이유: `prior-52 hazard_survival_emitter` 는 C 를 x 와
무관한 상수 위험률에서 뽑아 **무정보 절단**의 경계선을 만든다. 이 노드는 정확히
그 가정을 깬다 — 같은 Z 가 두 시각에 들어가므로 "관측 여부"와 "관측되지 않은 값"이
상관된다. `mixture_of_experts` 류의 게이트도, `location_scale_noise` 의 이분산도
관측 규약 자체를 결과와 얽지는 않는다.
"""
import torch
import torch.nn.functional as F

from tablegen_core.core._function import RandomFunction, CheapRandomFunction

NODE_NAME = "informative_censoring_hazard"

_N_EMIT = 3


class RandomInformativeCensoringHazardFunction(RandomFunction):
    def _fit(self, x: torch.Tensor):
        self.n_blocks_ = max(1, -(-self.out_features // _N_EMIT))
        b = self.n_blocks_

        self.h_ = CheapRandomFunction(self.context, self.in_features, b)   # 공유 잠재 Z
        self.g_ = CheapRandomFunction(self.context, self.in_features, b)   # 사건 위험
        self.gc_ = CheapRandomFunction(self.context, self.in_features, b)  # 절단 위험
        # 공유 잠재의 적재량 — 부호를 반대로 걸 수도 있게 음수 허용
        self.kappa_ = self.sampler.numerical("infcens_kappa", 0.1, 3.0, use_log=True)
        self.kappa_c_ = self.sampler.numerical("infcens_kappa_c", 0.1, 3.0, use_log=True)
        self.same_sign_ = self.sampler.boolean("infcens_same_sign")
        self.shape_ = self.sampler.numerical("infcens_shape", 0.5, 3.0)
        self.censor_bias_ = self.sampler.numerical("infcens_censor_bias", -1.0, 1.0)

        seed = int(self.sampler.randint("infcens_seed", 0, 2_147_483_647))
        self._gen_ = torch.Generator(device=x.device)
        self._gen_.manual_seed(seed)

    def _weibull(self, lam: torch.Tensor) -> torch.Tensor:
        u = torch.rand(lam.shape, generator=self._gen_, device=lam.device, dtype=lam.dtype)
        u = torch.clamp(u, 1e-6, 1.0 - 1e-6)
        k = max(0.2, float(self.shape_))
        t = torch.pow(torch.clamp(-torch.log(u), min=1e-6) / lam, 1.0 / k)
        return torch.nan_to_num(torch.clamp(t, 1e-6, 1e6), nan=0.0)
    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.clamp(F.softplus(self.h_(x)), 0.0, 1e3)                     # (n, b) 공유 잠재
        sign_c = 1.0 if self.same_sign_ else -1.0
        lam = torch.clamp(F.softplus(self.g_(x) + self.kappa_ * z), 1e-3, 1e3)
        lam_c = torch.clamp(
            F.softplus(self.gc_(x) + sign_c * self.kappa_c_ * z + self.censor_bias_), 1e-3, 1e3
        )

        t_event = self._weibull(lam)
        t_cens = self._weibull(lam_c)
        obs = torch.minimum(t_event, t_cens)
        evt = (t_event <= t_cens).to(obs.dtype)
        logexp = torch.log(torch.clamp(obs, min=1e-6))

        cols = torch.stack([obs, evt, logexp], dim=2).reshape(x.shape[0], _N_EMIT * self.n_blocks_)
        return torch.nan_to_num(torch.clamp(cols[:, : self.out_features], -1e6, 1e6), nan=0.0)
NODE_CLASS = RandomInformativeCensoringHazardFunction
