# -*- coding: utf-8 -*-
"""bourgouin_precipitation_type — 프로파일별 융해·결빙 에너지 강수형 표 [prior-239].

만드는 모양: 첫 열 분위수 profile 묶음 안에서 압력-온도 층의 양/음 에너지 면적을
합쳐, 눈·얼음싸라기·어는비·비의 정수 코드를 모든 원래 층 행에 broadcast한다.
"""
import numpy as np
import torch
from tablegen_core.priors._distinct_fill import ENABLED, distinct_fill, apply_plan
import torch.nn.functional as F

from tablegen_core.core._function import RandomFunction  # noqa: F401

NODE_NAME = "bourgouin_precipitation_type"


class RandomBourgouinPrecipitationTypeFunction(RandomFunction):
    def _fit(self, x: torch.Tensor):
        n = max(1, int(x.shape[0]))
        self.mech_ = self.sampler.choice("g239_mech", ["bourgouin_energy_threshold"])
        requested = int(self.sampler.randint("g239_profiles", 2, 9))
        self.n_profiles_ = max(1, min(requested, n))
        key = torch.nan_to_num(x[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
        cuts_q = torch.arange(1, self.n_profiles_, device=x.device, dtype=x.dtype) / self.n_profiles_
        self.profile_cuts_ = torch.quantile(key, cuts_q) if cuts_q.numel() else key.new_empty(0)

        # 행사용 fit의 선별은 앞 256행으로 한정한다. 운용 출력 전체는 apply_plan이 재검사한다.
        if ENABLED and self.out_features > 1:
            self.distinct_plan_, self.distinct_diagnostics_ = distinct_fill(
                sampler=self.sampler, prefix=NODE_NAME, out_features=self.out_features,
                in_features=self.in_features,
                origin=["first"] + ["replica"] * (self.out_features - 1),
                old_output=self._old_forward, candidate_hp=self._candidate_hp,
                evaluate=self._evaluate_extra, screen="fit_data", fit_data=x[:256],
                row_using_fit=True, column_types=["cat"] * self.out_features,
                candidate_types=["cat"] * max(64, 3 * (self.out_features - 1)),
                frozen_stats=self._extra_stats(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        old = self._old_forward(x)
        if ENABLED and self.out_features > 1:
            old, self.distinct_diagnostics_ = apply_plan(
                self.distinct_plan_, x, old, self._evaluate_extra)
        return old

    def _old_forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
        n = xs.shape[0]
        profile = torch.bucketize(xs[:, 0], self.profile_cuts_.to(x)).clamp(0, self.n_profiles_ - 1)
        p_src = xs[:, 1] if xs.shape[1] > 1 else xs[:, 0]
        t_src = xs[:, 2] if xs.shape[1] > 2 else (xs[:, 1] if xs.shape[1] > 1 else -xs[:, 0])
        pressure = F.softplus(torch.clamp(p_src, -20.0, 20.0)) + 1e-4
        temp_c = 8.0 * torch.clamp(t_src, -20.0, 20.0)
        # Profile order is group then original row index; only same-profile neighbors contribute a layer thickness.
        row = torch.arange(n, device=x.device)
        order = torch.argsort(profile.to(torch.int64) * max(1, n) + row, stable=True)
        inv = torch.argsort(order)
        ps, ts, gs = pressure[order], temp_c[order], profile[order]
        prev_p = torch.cat([ps[:1], ps[:-1]], dim=0)
        same = torch.cat([torch.zeros(1, device=x.device, dtype=torch.bool), gs[1:] == gs[:-1]], dim=0)
        dlog = torch.where(same, (torch.log(ps) - torch.log(prev_p)).abs(), torch.zeros_like(ps))
        dlog = torch.clamp(dlog, 0.0, 20.0)
        pa_layer = F.relu(ts) * dlog
        na_layer = F.relu(-ts) * dlog
        pa_group = torch.zeros(self.n_profiles_, device=x.device, dtype=x.dtype).scatter_add_(0, gs, pa_layer)
        na_group = torch.zeros(self.n_profiles_, device=x.device, dtype=x.dtype).scatter_add_(0, gs, na_layer)
        pa, na = pa_group[profile], na_group[profile]
        melted = pa >= 2.0
        refrozen = na >= 56.0 + 0.66 * pa
        rain = (pa > 13.2) & (na < 46.0 + 0.66 * pa)
        # 0=snow, 1=ice pellets, 2=freezing rain, 3=rain; fixed J/kg decision table.
        code = torch.where(~melted, 0, torch.where(refrozen, 1, torch.where(rain, 3, 2))).to(x.dtype)
        score_group = na_group - (56.0 + 0.66 * pa_group)
        score_cut = torch.quantile(score_group, 0.50)
        fallback_group = torch.where(score_group >= score_cut, 1.0, 2.0).to(x.dtype)
        constant_code = code.amax() == code.amin()
        code = torch.where(constant_code, fallback_group[profile], code)
        out = code.unsqueeze(1).repeat(1, self.out_features)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
        return torch.nan_to_num(torch.clamp(out, -1e6, 1e6), nan=0.0)
    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def _extra_stats(self, x):
        return torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)

    def _candidate_hp(self, rng, count):
        i = np.arange(count)
        # 서로 다른 프로파일 경계와 영도/에너지 문턱을 한 후보 배치에 둔다.
        return {"cuts_q": np.sort(.03 + .94 * rng.random((count, 15)), axis=1),
                "zero_q": .08 + .84 * ((i * 17 % count + rng.random(count) * .3) / count),
                "melt_q": .1 + .7 * rng.random(count),
                "freeze_q": .15 + .7 * rng.random(count),
                "rain_q": .15 + .7 * rng.random(count)}

    def _evaluate_extra(self, x, hp, stats):
        xs = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
        n, c = len(x), len(hp["zero_q"])
        def param(key):
            return torch.as_tensor(hp[key], device=x.device, dtype=x.dtype)
        cuts = torch.quantile(stats[:, 0].to(x), param("cuts_q").flatten()).reshape(c, 15)
        profile = (xs[:, :1, None] > cuts[None, :, :]).sum(2)
        p_src = xs[:, 1] if xs.shape[1] > 1 else xs[:, 0]
        t_src = xs[:, 2] if xs.shape[1] > 2 else (xs[:, 1] if xs.shape[1] > 1 else -xs[:, 0])
        st = stats[:, 2] if stats.shape[1] > 2 else (stats[:, 1] if stats.shape[1] > 1 else -stats[:, 0])
        zero = torch.quantile(st.to(x), param("zero_q"))
        pressure = (F.softplus(p_src.clamp(-20, 20)) + 1e-4)[:, None].expand(n, c)
        temp = 8 * (t_src[:, None] - zero).clamp(-20, 20)
        row = torch.arange(n, device=x.device)[:, None]
        order = torch.argsort(profile * max(n, 1) + row, dim=0, stable=True)
        ps, ts, gs = pressure.gather(0, order), temp.gather(0, order), profile.gather(0, order)
        prev = torch.cat([ps[:1], ps[:-1]], 0)
        same = torch.cat([torch.zeros(1, c, device=x.device, dtype=torch.bool), gs[1:] == gs[:-1]], 0)
        dlog = torch.where(same, (ps.log() - prev.log()).abs(), 0).clamp(0, 20)
        idx = gs + torch.arange(c, device=x.device)[None, :] * 16
        pa = xs.new_zeros(c * 16).scatter_add_(0, idx.flatten(), (F.relu(ts) * dlog).flatten()).reshape(c, 16).T
        na = xs.new_zeros(c * 16).scatter_add_(0, idx.flatten(), (F.relu(-ts) * dlog).flatten()).reshape(c, 16).T
        # 에너지 눈금은 열별 프로파일 통계; 타 열의 분류에는 관여하지 않는다.
        def qcol(values, q):
            ordered = values.sort(dim=0).values
            pos = q * 15
            lo = pos.floor().long(); hi = pos.ceil().long()
            col = torch.arange(c, device=x.device)
            return ordered[lo, col] * (1 - (pos - lo)) + ordered[hi, col] * (pos - lo)
        melted = pa >= qcol(pa, param("melt_q"))
        score = na - .66 * pa
        refrozen = score >= qcol(score, param("freeze_q"))
        rain = (pa >= qcol(pa, param("rain_q"))) & (na < 46.0 + .66 * pa)
        code = torch.where(~melted, 0, torch.where(refrozen, 1, torch.where(rain, 3, 2))).to(x.dtype)
        return code.gather(0, profile)
