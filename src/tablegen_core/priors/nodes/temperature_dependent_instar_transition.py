# -*- coding: utf-8 -*-
"""temperature_dependent_instar_transition — 온도율에 따른 영기 이동 [prior-395].

만드는 모양: 행 순서를 시간으로 삼아 온도별 전이행렬을 지난 양의 영기 밀도 표.
"""
import torch
from tablegen_core.priors._distinct_fill import ENABLED, derived_generator
import torch.nn.functional as F

from tablegen_core.core._function import RandomFunction  # noqa: F401

NODE_NAME = "temperature_dependent_instar_transition"


class RandomTemperatureDependentInstarTransitionFunction(RandomFunction):
    def _fit(self, x: torch.Tensor):
        self.mech_ = self.sampler.choice("g395_mech", ["thermal_instar_matrix"])
        self.stages_ = max(2, min(self.out_features, 8))
        self.steps_ = int(self.sampler.randint("g395_steps", 2, 7))
        self.rho_ = float(self.sampler.numerical("g395_rho", 0.18, 0.55))
        self.activation_ = float(self.sampler.numerical("g395_activation", 1200.0, 3600.0))
        self.low_energy_ = float(self.sampler.numerical("g395_low_energy", 900.0, 2400.0))
        self.high_energy_ = float(self.sampler.numerical("g395_high_energy", 900.0, 2400.0))
        self.low_temp_ = float(self.sampler.numerical("g395_low_temp", 278.0, 289.0))
        self.high_temp_ = float(self.sampler.numerical("g395_high_temp", 305.0, 318.0))
        self.mortality_ = torch.as_tensor(
            self.sampler.numerical("g395_mortality", 0.01, 0.09,
                                   size=(self.stages_,), mode="local"),
            device=x.device, dtype=x.dtype,
        )
        self.stage_weight_ = torch.as_tensor(
            self.sampler.numerical("g395_stage_weight", -1.0, 1.0,
                                   size=(self.in_features, self.stages_), mode="local"),
            device=x.device, dtype=x.dtype,
        ) / max(1, self.in_features) ** 0.5
        self.stage_rate_ = torch.as_tensor(
            self.sampler.numerical("g395_stage_rate", 0.6, 1.3,
                                   size=(self.stages_,), mode="local"),
            device=x.device, dtype=x.dtype,
        )

        self.extra_ = None
        count = self.out_features - self.stages_
        if ENABLED and count > 0:
            rng = derived_generator(self.sampler, "g395_extra", self.out_features)
            def draw(lo, hi, shape):
                return torch.as_tensor(rng.uniform(lo, hi, shape), device=x.device, dtype=x.dtype)
            self.extra_ = {
                "steps": torch.as_tensor(rng.integers(2, 7, count), device=x.device),
                "rho": draw(0.18, 0.55, (count,)),
                "activation": draw(1200.0, 3600.0, (count,)),
                "low_energy": draw(900.0, 2400.0, (count,)),
                "high_energy": draw(900.0, 2400.0, (count,)),
                "low_temp": draw(278.0, 289.0, (count,)),
                "high_temp": draw(305.0, 318.0, (count,)),
                "mortality": draw(0.01, 0.09, (count, self.stages_)),
                "stage_rate": draw(0.6, 1.3, (count, self.stages_)),
                "weight": draw(-1.0, 1.0, (self.in_features, count * self.stages_))
                          / max(1, self.in_features) ** 0.5,
            }

    def _extra_columns(self, clean, temp, row):
        hp = {key: value.to(clean) for key, value in self.extra_.items()}
        t = temp[:, None]
        activation = (hp["activation"] * (1.0 / 298.0 - 1.0 / t)).clamp(-20, 20)
        low = (hp["low_energy"] * (1.0 / t - 1.0 / hp["low_temp"])).clamp(-20, 20)
        high = (hp["high_energy"] * (1.0 / hp["high_temp"] - 1.0 / t)).clamp(-20, 20)
        rate = hp["rho"] * (t / 298.0) * torch.exp(activation)
        rate = (rate / (1.0 + torch.exp(low) + torch.exp(high))).clamp(0.005, 0.55)
        transition = (rate[:, :, None] * hp["stage_rate"]).clamp(max=0.75)
        survival = (1.0 - transition - hp["mortality"]).clamp_min(0.0)
        count = hp["rho"].numel()
        initial = F.softplus((clean @ hp["weight"]).clamp(-20, 20)) + 0.05
        starts = (row[:, None] - hp["steps"].long()).clamp_min(0)
        banks = torch.arange(count, device=clean.device)
        state = initial.reshape(clean.shape[0], count, self.stages_)[starts, banks]
        # Same upper-bidiagonal transition, batched over independent parameter sets.
        for lag in range(6, 0, -1):
            source = (row - lag).clamp_min(0)
            moved = state[:, :, :-1] * transition[source, :, :-1]
            next_state = state * survival[source]
            next_state[:, :, 1:] = next_state[:, :, 1:] + moved
            state = torch.where((hp["steps"] >= lag)[None, :, None], next_state, state)
        stage = torch.arange(count, device=clean.device) % self.stages_
        return state[:, torch.arange(count, device=clean.device), stage]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        if n == 0:
            return x.new_empty((0, self.out_features))
        clean = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)
        temp = 298.0 + 22.0 * torch.tanh(clean[:, 0])
        activation = (self.activation_ * (1.0 / 298.0 - 1.0 / temp)).clamp(-20, 20)
        low = (self.low_energy_ * (1.0 / temp - 1.0 / self.low_temp_)).clamp(-20, 20)
        high = (self.high_energy_ * (1.0 / self.high_temp_ - 1.0 / temp)).clamp(-20, 20)
        rate = self.rho_ * (temp / 298.0) * torch.exp(activation)
        rate = (rate / (1.0 + torch.exp(low) + torch.exp(high))).clamp(0.005, 0.55)
        transition = (rate[:, None] * self.stage_rate_.to(x)[None, :]).clamp(max=0.75)
        survival = (1.0 - transition - self.mortality_.to(x)[None, :]).clamp_min(0.0)
        matrix = torch.diag_embed(survival) + torch.diag_embed(transition[:, :-1], offset=1)
        initial = F.softplus((clean @ self.stage_weight_.to(x)).clamp(-20, 20)) + 0.05
        row = torch.arange(n, device=x.device)
        state = initial[(row - self.steps_).clamp_min(0)].unsqueeze(1)
        for lag in range(self.steps_, 0, -1):
            state = torch.bmm(state, matrix[(row - lag).clamp_min(0)])
        state = state.squeeze(1)
        idx = torch.arange(self.out_features, device=x.device) % self.stages_
        out = state[:, idx]
        if self.extra_ is not None:
            out[:, self.stages_:] = self._extra_columns(clean, temp, row)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
        return torch.clamp(out, -1e6, 1e6)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
