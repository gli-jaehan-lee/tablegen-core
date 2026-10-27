# -*- coding: utf-8 -*-
"""ewens_sampling_formula — 중립 대립유전자 계급 확률 [prior-521].

만드는 모양: 표집한 표본 크기 4~8의 유효 분할에 대한 Ewens 확률 표.
"""
import torch
from tablegen_core.core._function import RandomFunction  # noqa: F401

NODE_NAME = "ewens_sampling_formula"


class RandomEwensSamplingFormulaFunction(RandomFunction):
    def _fit(self, x: torch.Tensor):
        self.theta_ = float(self.sampler.numerical("g521_theta", 0.5, 4.0))
        self.gain_ = float(self.sampler.numerical("g521_population_gain", 0.2, 0.8))
        self.sample_n_ = int(self.sampler.randint("g521_sample_n", 4, 9))
        # 모든 조성의 절단 비트를 벡터화해 유효 정수 분할을 만든다.
        ns = self.sample_n_
        bits = torch.arange(2 ** (ns - 1), device=x.device)[:, None]
        cuts = (bits >> torch.arange(ns - 1, device=x.device)) & 1
        groups = torch.cat((torch.zeros_like(bits), cuts.cumsum(dim=1)), dim=1)
        lengths = torch.zeros((bits.shape[0], ns), device=x.device, dtype=torch.long)
        lengths.scatter_add_(1, groups, torch.ones_like(groups))
        counts = x.new_zeros((bits.shape[0], ns + 1))
        counts.scatter_add_(1, lengths, torch.ones_like(lengths, dtype=x.dtype))
        self.a_ = torch.unique(counts[:, 1:], dim=0)
        j = torch.arange(self.out_features, device=x.device, dtype=x.dtype)[None, :]
        self.theta_scale_ = 1 + 0.2 * j
        self.order_ = torch.arange(1, self.sample_n_ + 1, device=x.device, dtype=x.dtype)[None, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return x.new_empty((0, self.out_features))
        z = torch.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20, 20)
        score = z.mean(dim=1, keepdim=True)
        boundaries = torch.quantile(score[:, 0], torch.arange(1, self.a_.shape[0], device=x.device, dtype=x.dtype) / self.a_.shape[0])
        index = torch.bucketize(score[:, 0].contiguous(), boundaries.contiguous()).clamp_max(self.a_.shape[0] - 1)
        if score.amax() == score.amin() and x.shape[0] > 1:
            ranks = torch.argsort(torch.argsort(score[:, 0], stable=True), stable=True)
            index = ranks * self.a_.shape[0] // x.shape[0]
        a = self.a_[index]
        theta = (self.theta_ * self.theta_scale_ *
                 (1 + self.gain_ * torch.tanh(score.mean()) + 0.1 * torch.tanh(score))).clamp_min(1e-4)
        log_numerator = (torch.lgamma(torch.tensor(float(self.sample_n_ + 1), device=x.device, dtype=x.dtype))
                         + (a[:, :, None] * (torch.log(theta)[:, None, :] - torch.log(self.order_)[:, :, None])).sum(dim=1)
                         - torch.lgamma(a + 1).sum(dim=1, keepdim=True))
        log_denominator = torch.lgamma(theta + self.sample_n_) - torch.lgamma(theta)
        out = torch.exp((log_numerator - log_denominator).clamp(-80, 0))
        return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).clamp(-1e6, 1e6)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
