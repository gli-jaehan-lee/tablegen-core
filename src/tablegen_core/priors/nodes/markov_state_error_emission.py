# -*- coding: utf-8 -*-
"""markov_state_error_emission — 2상태 정상분포와 상태별 오류율을 결합해 평균 오류율을 계산한다 [prior-806].

만드는 모양: pi_G=b/(a+b), pi_B=a/(a+b), bar_e=pi_G*e_G+pi_B*e_B를 출력한다.
행=독립 채널 설정. 상태·오류열을 표본 생성하지 않는다.

References:
- https://www.rfc-editor.org/rfc/rfc6534.html
"""
import torch
from tablegen_core.priors._distinct_fill import derived_generator
from tablegen_core.core._function import RandomFunction

NODE_NAME = "markov_state_error_emission"


class RandomMarkovStateErrorEmissionFunction(RandomFunction):
    def _fit(self, x: torch.Tensor):
        self.gain_ = float(self.sampler.numerical("g806_gain", 0.6, 1.8, mode="local"))
        self.error_max_ = float(self.sampler.numerical("g806_error_max", 0.15, 0.65, mode="local"))
        d, o = int(self.in_features), int(self.out_features)
        rng = derived_generator(self.sampler, NODE_NAME, o)
        # 열마다 채널 전이·방출의 입력 사상을 독립 추첨한다.
        self.weights_ = torch.as_tensor(rng.uniform(-1.7, 1.7, (d, o * 4)) / max(d, 1)**0.5,
                                        dtype=torch.float64, device=x.device)
        self.bias_ = torch.as_tensor(rng.uniform(-1.5, 1.5, (o, 4)),
                                     dtype=torch.float64, device=x.device)

    def _parameters(self, x):
        clean = torch.nan_to_num(x.to(torch.float64), nan=0.0, posinf=1e6,
                                 neginf=-1e6).clamp(-1e6, 1e6)
        z = 3.0 * torch.tanh(clean / 3.0)
        p = torch.sigmoid((self.gain_ * (z @ self.weights_.to(x.device))).reshape(
            x.shape[0], int(self.out_features), 4) + self.bias_.to(x.device))
        a = 0.005 + 0.49 * p[..., 0]
        b = 0.005 + 0.49 * p[..., 1]
        eg = 0.001 + 0.08 * p[..., 2]
        eb = eg + (self.error_max_ - eg) * p[..., 3]
        return a, b, eg, eb

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[0] or not self.out_features:
            return x.new_zeros((x.shape[0], int(self.out_features)))
        a, b, eg, eb = self._parameters(x)
        out = (b * eg + a * eb) / (a + b)
        return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1).to(x.dtype)


NODE_CLASS = RandomMarkovStateErrorEmissionFunction
