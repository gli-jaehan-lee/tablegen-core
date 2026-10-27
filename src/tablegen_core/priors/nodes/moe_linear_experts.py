# -*- coding: utf-8 -*-
"""moe_linear_experts — mixture_of_experts와 동형이나 expert가 순수 affine.

만드는 모양: K개의 **affine(선형) expert**를 x-종속 소프트맥스 게이트로 가중합 —
소프트 게이팅 조각선형(soft-gated piecewise-linear) 표면. hard_moe_affine(argmax
하드 선택, 경계에서 불연속)과 쌍을 이루는 **연속판**: 경계 근방에서 여러 affine
조각이 매끄럽게 섞이다가 게이트가 뾰족해지면(온도↓) 하드 버전으로 강등된다.
기존 8노드가 못 만드는 이유: mixture_of_experts는 expert가
CheapRandomFunction(트리/GP 등 비선형 포함)이라 "소프트 조각선형"이라는 순수한
형태를 못 낸다. hard_moe_affine은 같은 affine expert 집합을 쓰지만 argmax라 미분
불가능한 경계를 만든다. 둘 다 아닌, expert=affine + 게이트=softmax 조합이 빠져 있었다.
"""
import torch

from tablegen_core.core._function import RandomFunction, RandomLinearFunction

NODE_NAME = "moe_linear_experts"


class RandomMoELinearExpertsFunction(RandomFunction):
    def _fit(self, x: torch.Tensor):
        self.K_ = int(self.sampler.randint("moelin_n_experts", 2, 7))
        # 소프트맥스 온도(로그균등)
        self.temperature_ = self.sampler.numerical("moelin_temperature", 0.1, 3.0, use_log=True)
        self.gate_ = RandomLinearFunction(self.context, self.in_features, self.K_)
        self.experts_ = [
            RandomLinearFunction(self.context, self.in_features, self.out_features)
            for _ in range(self.K_)
        ]

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate_(x) / self.temperature_
        weights = torch.softmax(logits, dim=-1)                  # (n, K)
        expert_out = torch.stack([e(x) for e in self.experts_], dim=1)  # (n, K, out)
        out = (weights[:, :, None] * expert_out).sum(dim=1)
        return torch.nan_to_num(torch.clamp(out, -1e6, 1e6), nan=0.0)
NODE_CLASS = RandomMoELinearExpertsFunction
