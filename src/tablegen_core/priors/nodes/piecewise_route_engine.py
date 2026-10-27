# -*- coding: utf-8 -*-
"""piecewise_route_engine — 구간·경로 전환 합성기 [prior-84].

만드는 모양: 한 게이트 좌표의 램프-홀드, 순차 공정, 이중 변환, 건너뛰기 확률,
확률 경로, 끝점 원자 또는 bin 가치를 만든다.
기존 78종이 못 만드는 이유: 기존 piecewise 노드는 단일 분할값을 내지만 이 기계는
정확한 0 홀드, 확률 경로 혼합, 양 끝 원자와 simplex bin 출력을 같은 확정 골격에 둔다.
"""
import torch
from tablegen_core.priors._distinct_fill import ENABLED, derived_generator
import torch.nn.functional as F

from tablegen_core.core._function import RandomFunction, CheapRandomFunction

NODE_NAME = "piecewise_route_engine"


class RandomPiecewiseRouteEngineFunction(RandomFunction):
    def _fit(self, x: torch.Tensor):
        self.mech_ = self.sampler.choice("pre_mech", [
            "ramp_hold", "ordered_process_path", "dual_transform", "skip_probability",
            "route_mixture", "critical_size", "bin_value",
        ])
        self.knot_ = float(self.sampler.numerical("pre_knot", -1.5, 1.5))
        self.slope_ = float(self.sampler.numerical("pre_slope", 0.2, 5.0, use_log=True))
        # critical_size 전용 임계 크기: softplus(g) 중앙값 0.69 가 [c, 2c] 를 가로지르게
        # [0.2, 1.2] 에서 뽑는다. 공유 `pre_slope` (<=5) 는 softplus 최대(~3.5)를 넘어
        # 전열 0 원자로 붕괴시킨다.
        self.crit_ = float(self.sampler.numerical("pre_crit", 0.2, 1.2, use_log=True))
        seed = int(self.sampler.randint("pre_seed", 0, 2_147_483_647))
        self._gen_ = torch.Generator(device=x.device).manual_seed(seed)
        self.a_ = torch.randn(3, self.out_features, generator=self._gen_, device=x.device, dtype=x.dtype)
        self.b_ = torch.randn(3, self.out_features, generator=self._gen_, device=x.device, dtype=x.dtype)

        self.extra_hp_ = None
        if ENABLED and self.out_features > 1 and self.mech_ in {"ramp_hold", "skip_probability", "critical_size"}:
            rng = derived_generator(self.sampler, NODE_NAME, self.out_features)
            count = self.out_features - 1
            weights = rng.normal(size=(self.in_features, count)) / max(self.in_features, 1) ** 0.5
            weights = torch.as_tensor(weights, device=x.device, dtype=x.dtype)
            weights = weights / weights.square().sum(dim=0).sqrt().clamp_min(1e-6)
            gate = x @ weights
            # 확장 열의 문턱도 그 열의 게이트에서 뽑아 전열 홀드 포화를 피한다.
            quantiles = torch.as_tensor(rng.uniform(0.25, 0.65, count), device=x.device, dtype=x.dtype)
            knots = torch.quantile(gate, quantiles, dim=0).diagonal()
            crits = torch.quantile(F.softplus(gate), quantiles, dim=0).diagonal() / 1.5
            slopes = torch.as_tensor(rng.uniform(0.2, 5.0, count), device=x.device, dtype=x.dtype)
            self.extra_hp_ = (weights, knots, slopes, crits)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        g = x[:, :1]
        branches = g.unsqueeze(2) * self.a_.to(x).unsqueeze(0) + self.b_.to(x).unsqueeze(0)
        if self.mech_ == "ramp_hold":
            out = F.relu(self.slope_ * (g - self.knot_)).expand(-1, self.out_features)
        elif self.mech_ == "ordered_process_path":
            q1 = torch.sigmoid(self.slope_ * (g - self.knot_))
            q2 = torch.sigmoid(self.slope_ * (g - self.knot_ - 1.0))
            out = F.softplus(branches[:, 0]) * (1 - q1) + F.softplus(branches[:, 1]) * (q1 - q2) + F.softplus(branches[:, 2]) * q2
        elif self.mech_ == "dual_transform":
            out = torch.where(g > self.knot_, branches[:, 1], branches[:, 0])
        elif self.mech_ == "skip_probability":
            state = (g > self.knot_).to(x.dtype)
            p_skip = torch.sigmoid(self.slope_ * (g - self.knot_))
            out = torch.where(state > 0, p_skip, torch.ones_like(p_skip)).expand(-1, self.out_features)
        elif self.mech_ == "route_mixture":
            probs = torch.softmax(self.b_.to(x).mean(dim=1), dim=0)
            u = torch.rand((x.shape[0], 1), generator=self._gen_, device=x.device, dtype=x.dtype)
            idx = (u > probs.cumsum(0).view(1, -1)).sum(dim=1).clamp(max=2)
            out = branches[torch.arange(x.shape[0], device=x.device), idx]
        elif self.mech_ == "critical_size":
            size = F.softplus(g)
            crit = self.crit_
            middle = 1.0 - crit / (size + 1e-6)
            out = torch.where(size <= crit, torch.zeros_like(size), torch.where(size >= 2 * crit, torch.ones_like(size), middle)).expand(-1, self.out_features)
        elif self.out_features >= 2:
            logits = g * self.a_.to(x)[0:1] + self.b_.to(x)[0:1]
            out = torch.softmax(logits, dim=1)
        else:
            # bin_value·out_features==1: 열 축 softmax 는 (n,1) 에서 항등적으로 1 이라
            # 구간 정보가 사라진다. 열이 하나여도 「구간마다 다른 값」이 남게, 내부에
            # bin 3개(중심 knot-1·knot·knot+1)를 세워 게이트를 soft-binning 하고
            # 기대 bin 가치 Σ p_k·v_k 를 1열로 낸다. bin 가치는 _fit 의 시드 고정
            # `b_` 첫 열을 쓴다(새 난수 소비 없음 → 다른 갈래 재현성 불변).
            v = self.b_.to(x)[:, 0:1].view(1, 3)                     # (1,3) bin 가치
            centers = torch.tensor([-1.0, 0.0, 1.0], device=x.device, dtype=x.dtype)
            centers = (self.knot_ + centers).view(1, 3)              # (1,3) bin 중심
            bl = -self.slope_ * (g - centers) ** 2                   # (n,3) bin 로짓
            p = torch.softmax(torch.clamp(bl, -20.0, 20.0), dim=1)
            out = (p * v).sum(dim=1, keepdim=True)                   # (n,1) 구간 값
        if self.extra_hp_ is not None:
            weights, knots, slopes, crits = (v.to(x) for v in self.extra_hp_)
            gate = x @ weights
            if self.mech_ == "ramp_hold":
                extra = F.relu(slopes * (gate - knots))
            elif self.mech_ == "skip_probability":
                extra = torch.where(gate > knots, torch.sigmoid(slopes * (gate - knots)), torch.ones_like(gate))
            else:
                size = F.softplus(gate)
                middle = 1.0 - crits / (size + 1e-6)
                extra = torch.where(size <= crits, torch.zeros_like(size), torch.where(size >= 2 * crits, torch.ones_like(size), middle))
            out = torch.cat((out[:, :1], extra), dim=1)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
        return torch.nan_to_num(torch.clamp(out, -1e6, 1e6), nan=0.0)
NODE_CLASS = RandomPiecewiseRouteEngineFunction
