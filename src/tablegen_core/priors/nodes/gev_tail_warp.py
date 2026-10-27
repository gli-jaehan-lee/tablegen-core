# -*- coding: utf-8 -*-
"""gev_tail_warp — GEV(Fréchet) 분위변환으로 중꼬리를 만드는 랭크 워프.

만드는 모양: 베이스 함수 출력의 "형태"(어느 행이 크고 작은지의 순서)는 그대로
유지하면서, 그 값의 "분포 모양"만 일반화 극단값분포(GEV, ξ>0 → Fréchet)로
바꿔치기한 중꼬리 관계 — 강수량·금융손실·최댓값 통계처럼 극단값이 정규분포보다
훨씬 두꺼운 꼬리를 갖는 실데이터의 전형적 패턴.
기존 8노드가 못 만드는 이유: 기존 노드의 출력은 GP/트리/MLP
합성이 만드는 완만한 꼬리 안에 머무른다(가우시안 유사 유한 분산). "순위는
베이스함수를 따르되 값의 분포는 정규분포보다 훨씬 두꺼운 극단값 분포"를 만들려면
경험적 순위를 균등분포로 편 뒤 GEV 분위함수를 통과시키는 별도 변환 단계가 필요하다.
"""
import torch

from tablegen_core.core._function import RandomFunction, CheapRandomFunction

NODE_NAME = "gev_tail_warp"


class RandomGEVTailWarpFunction(RandomFunction):
    def _fit(self, x: torch.Tensor):
        self.base_ = CheapRandomFunction(self.context, self.in_features, self.out_features)
        # 형상모수 ξ>0 (Fréchet만 다룸; 크면 꼬리가 더 두꺼워짐)
        self.xi_ = self.sampler.numerical("gev_xi", 0.05, 0.8, use_log=True)
        self.mu_ = self.sampler.numerical("gev_mu", -1.0, 1.0)
        self.sigma_ = self.sampler.numerical("gev_sigma", 0.1, 2.0, use_log=True)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_(x)                                       # (n, out)
        n = base.shape[0]

        # 랭크(오름차순 0..n-1): argsort 두 번 = "순서의 순서" = 순위
        order = torch.argsort(base, dim=0)
        rank = torch.argsort(order, dim=0).to(base.dtype)           # (n, out)
        u = (rank + 0.5) / n                                          # 랭크 -> (0,1) 균등
        u = torch.clamp(u, 1e-6, 1.0 - 1e-6)

        neg_log_u = torch.clamp(-torch.log(u), min=1e-6)              # ξ>0 발산 억제(U->1 극한)
        warped = self.mu_ + (self.sigma_ / self.xi_) * (neg_log_u.pow(-self.xi_) - 1.0)
        return torch.nan_to_num(torch.clamp(warped, -1e6, 1e6), nan=0.0)
NODE_CLASS = RandomGEVTailWarpFunction
