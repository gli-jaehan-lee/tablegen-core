# -*- coding: utf-8 -*-
"""eos_excess_mixing_enthalpy — 동일 온도·압력에서 혼합물과 순수성분 잔차 엔탈피의 차를 계산한다 [prior-1001].

만드는 모양: 행=온도·압력·조성 조건 한 벌, 열=서로 다른 물성계의 카드 관측량.

References:
- https://doi.org/10.1021/ie901830c
"""
import numpy as np
import torch
from tablegen_core.priors._distinct_fill import derived_generator
from tablegen_core.core._function import RandomFunction

NODE_NAME = "eos_excess_mixing_enthalpy"


class RandomEosExcessMixingEnthalpyFunction(RandomFunction):
    def _fit(self, x):
        assert isinstance(self.in_features, (int, np.integer))
        assert isinstance(self.out_features, (int, np.integer))
        self.c_ = int(self.sampler.randint("g1001_components", 2, 5))
        self.scale_ = float(self.sampler.numerical("g1001_input_scale", .6, 2.0))
        d, o, c = int(self.in_features), int(self.out_features), self.c_
        rng = derived_generator(self.sampler, NODE_NAME, o)
        def tensor(v):
            return torch.as_tensor(v, dtype=torch.float64, device=x.device)
        self.w_ = tensor(rng.standard_normal((d, o*(c+2))) / max(d, 1)**.5)
        self.bias_ = tensor(rng.uniform(-1.5, 1.5, (o, c+2)))
        self.a_ = tensor(rng.uniform(.5, 2.0, (o, c)))
        self.b_ = tensor(rng.uniform(.2, .6, (o, c)))
        self.cv_ = tensor(rng.uniform(2.0, 5.0, (o, c)))
        self.mass_ = tensor(rng.uniform(.5, 2.0, (o, c)))
        k = tensor(rng.uniform(-.3, .3, (o, c, c)))
        self.cross_ = 1 - (k + k.transpose(-1, -2))/2
        self.cross_.diagonal(dim1=-2, dim2=-1).fill_(1)

    def _state(self, x):
        n, o, c = x.shape[0], int(self.out_features), self.c_
        u = torch.nan_to_num(x.double(), nan=0., posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6).tanh()
        h = self.scale_*(u @ self.w_.to(x.device)).reshape(n, o, c+2) + self.bias_.to(x.device)
        z = (2*h[..., :c].tanh()).softmax(-1)
        a, b, cv, mass, cross = [v.to(x.device) for v in
            (self.a_, self.b_, self.cv_, self.mass_, self.cross_)]
        weighted = z*a.sqrt()
        am = (weighted*(cross*weighted[..., None, :]).sum(-1)).sum(-1)
        bm = (z*b).sum(-1)
        return h, z, am, bm, (z*cv).sum(-1), (z*mass).sum(-1), a, b

    def _transform(self, x):
        if not x.shape[0] or not self.out_features:
            return x.new_empty((x.shape[0], int(self.out_features)))
        h, z, am, bm, _, _, a, b = self._state(x)
        t = 2 + 2*h[..., -2].sigmoid()
        p = .1 + .7*h[..., -1].sigmoid()
        # V=T/P+B(T), B=b-a/T: Hres=P*(B-T*B_T)=P*(b-2a/T).
        mixture = p*(bm - 2*am/t)
        pure = p[..., None]*(b - 2*a/t[..., None])
        value = mixture - (z*pure).sum(-1)
        limit = torch.finfo(x.dtype).max
        return value.clamp(-limit, limit).to(dtype=x.dtype, device=x.device)


NODE_CLASS = RandomEosExcessMixingEnthalpyFunction
