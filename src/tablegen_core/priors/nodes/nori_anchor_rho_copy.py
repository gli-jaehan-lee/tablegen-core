# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Derived from Synthefy/synthefy-nori @ c0a8ac2 (Copyright 2026 Synthefy); modified by Gauss Labs Inc.
"""nori_anchor_rho_copy — ρ 혼합 앵커를 원래 평균·std로 복원 [prior-755].

출처: Synthefy/synthefy-nori @ c0a8ac2 · Apache-2.0 · prior_port.py:3296–3307, 4128–4134, 4286–4293, 4413–4416
만드는 모양: 본체는 z=(a-mean(a))/std(a), std≤1e-8이면 중심화만 한 뒤
(ρz+sqrt(max(1-ρ²,1e-6)) ε)(std(a)+1e-6)+mean(a)를 방출한다.
GP·이차·블록은 표준화·복원 없이 ρa+sqrt(1-ρ²) ε를 방출한다.
ρ 범위는 본체 .4–.95, GP .3–.8, 이차 .5–.95, 블록 .3–.95다.
"""
import torch
from tablegen_core.priors._distinct_fill import derived_generator
from tablegen_core.core._function import RandomFunction

NODE_NAME = "nori_anchor_rho_copy"


class RandomNoriAnchorRhoCopyFunction(RandomFunction):
    def _fit(self, x):
        self.anchor_ = torch.as_tensor(self.sampler.numerical(
            "g755_anchor", 0, self.in_features, size=self.out_features, mode="local"),
            device=x.device, dtype=torch.long)
        self.profile_ = torch.as_tensor(self.sampler.numerical(
            "g755_profile", 0, 4, size=self.out_features, mode="local"),
            device=x.device, dtype=torch.long)
        u = torch.as_tensor(self.sampler.numerical(
            "g755_rho_uniform", 0., 1., size=self.out_features, mode="local"),
            device=x.device, dtype=x.dtype)
        low = x.new_tensor([.4, .3, .5, .3])[self.profile_]
        high = x.new_tensor([.95, .8, .95, .95])[self.profile_]
        self.rho_ = low+(high-low)*u
        self.noise_rng_ = derived_generator(self.sampler, "g755_epsilon", self.out_features)

    def _transform(self, x):
        if not len(x):
            return x.new_empty((0, self.out_features))
        a = torch.nan_to_num(x, nan=0., posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)[:, self.anchor_.to(x.device)]
        mean, std = a.mean(0), a.std(0, correction=0)
        z = torch.where(std > 1e-8, (a-mean)/std.clamp_min(1e-8), a-mean)
        eps = torch.as_tensor(self.noise_rng_.standard_normal(a.shape), device=x.device, dtype=x.dtype)
        rho = self.rho_.to(x)
        variance = 1-rho.square()
        restored = (rho*z+variance.clamp_min(1e-6).sqrt()*eps)*(std+1e-6)+mean
        simple = rho*a+variance.clamp_min(0).sqrt()*eps
        y = torch.where(self.profile_.to(x.device) == 0, restored, simple)
        return torch.nan_to_num(y, nan=0., posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)


NODE_CLASS = RandomNoriAnchorRhoCopyFunction
