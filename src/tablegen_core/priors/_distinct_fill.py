"""열 복제 수선 공용 API v1 (261001 동결).

노드는 원래 모양으로 old_output(x)를 계산한다. prepare_plan()에 각 출력 칸의
first/replica 출처, 층화 순서의 candidate_hp(rng, count), 별도 배치 계산 함수
evaluate(x, hp, frozen_stats)를 전달한다. apply_plan()은 저장 HP로만 재평가한다.
evaluate/old_output은 난수 없는 순수 함수여야 한다. 새 식·배율·잡음 허용 API가
아니며, 노드별 확장 축 교차 검토는 호출자의 책임이다. 열간 통계는 원본에서
계산해 frozen_stats에 전달한다. first 칸은 old_output의 비트를 그대로 복사한다.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import os
from typing import Any, Callable

import numpy as np
import torch

VERSION = "distinct-fill-v1"
ENABLED = os.environ.get("TFM_DISTINCT_FILL", "1") != "0"
CAPACITY_COUNTER = Counter()


def derived_generator(sampler, prefix: str, out_features: int) -> np.random.Generator:
    """부모 상태를 읽기만 한다. 같은 상태의 다른 접두/폭은 다른 확장 스트림이다."""
    parent = sampler.rng if hasattr(sampler, "rng") else sampler
    state = parent.bit_generator.state
    payload = json.dumps([VERSION, prefix, int(out_features), state], sort_keys=True,
                         default=lambda a: np.asarray(a).tolist()).encode()
    seed = int.from_bytes(hashlib.blake2b(payload, digest_size=16).digest(), "little")
    return np.random.default_rng(seed)


def canonical_input(rng, in_features: int, *, dtype=torch.float32, device="cpu"):
    """random만으로 Box–Muller 정규 128행 + 균등 128행. 전역 RNG 미사용."""
    u = rng.random((128, in_features, 2))
    normal = np.sqrt(-2 * np.log(np.maximum(u[..., 0], np.finfo(float).tiny)))
    normal *= np.cos(2 * np.pi * u[..., 1])
    uniform = 2 * rng.random((128, in_features)) - 1
    return torch.as_tensor(np.concatenate([normal, uniform]), dtype=dtype, device=device)


@dataclass
class Diagnostics:
    requested_width: int
    unique_count: int = 0
    collisions: int = 0
    constants: int = 0
    capacity_reason: str | None = None
    missing: int = 0
    success: bool = False
    disabled: bool = False
    gate_reasons: dict = field(default_factory=dict)
    candidate_collisions: int = 0
    candidate_constants: int = 0


@dataclass
class DistinctPlan:
    prefix: str
    origin: tuple
    hp: Any = None
    hp_log: dict = field(default_factory=dict)
    frozen_stats: Any = None
    candidate_indices: tuple = ()
    output_indices: tuple = ()
    screen: str = "canonical"
    column_types: tuple = ()
    candidate_types: tuple = ()
    diagnostics: Diagnostics | None = None
    version: str = VERSION


def _count(y, column_types=None, **kwargs):
    # gate 정본은 중앙 검사기 한 곳뿐이다(도우미 안에 복제하지 않는다).
    from tablegen_core.priors.tools.distinct_col_check import representative_columns
    return representative_columns(y, column_types=column_types, **kwargs)


def _select_hp(hp, indices, count):
    """후보 축은 HP 배열의 첫 축. 공통 값은 스칼라, 공통 텐서는 frozen_stats."""
    if not isinstance(hp, dict):
        raise ValueError("candidate_hp는 수치 HP dict를 반환해야 합니다")
    selected = {}
    for name, value in hp.items():
        arr = np.asarray(value)
        if arr.dtype.kind not in "iufb" or not np.isfinite(arr).all():
            raise ValueError(f"local HP는 유한 수치/정수만: {name}")
        if arr.ndim and arr.shape[0] != count:
            raise ValueError(f"HP 후보 축은 첫 축이어야 합니다: {name}")
        selected[name] = arr[np.asarray(indices, dtype=int)].copy() if arr.ndim else arr.item()
    return selected


def prepare_plan(*, sampler, prefix: str, out_features: int, in_features: int,
                 origin, old_output: Callable, candidate_hp: Callable,
                 evaluate: Callable, screen="canonical", fit_data=None,
                 row_using_fit=False, frozen_stats=None, column_types=None,
                 candidate_types=None, capacity_reason=None):
    """(plan, diagnostics) 반환. 후보는 한 배치·결정론 순서, 추가 재시도 없음.

    candidate_hp는 rng.integers/random만 쓰며 후보별 HP 배열을 dict로 반환한다.
    각 배열의 첫 축=후보 수, 공통 값은 스칼라다. 선택 후 HP를 압축하므로
    transform은 선택된 열만 한 배치로 계산한다(64후보 재계산을 하지 않는다).
    evaluate는 전체 후보를 한 호출로 계산한다. transform에서도 같은 HP 배치를
    별도 호출한다. column_types는 원 출력, candidate_types는 후보 열의 형이다.
    fit_data 선별은 이미 행을 쓰는 fit에만 명시적으로 허용한다.
    """
    origin = tuple(origin)
    if len(origin) != out_features or any(v not in {"first", "replica"} for v in origin):
        raise ValueError("origin은 요청 폭의 first/replica 목록이어야 합니다")
    if screen not in {"canonical", "fit_data"}:
        raise ValueError("screen은 canonical 또는 fit_data")
    if screen == "fit_data" and (not row_using_fit or fit_data is None):
        raise ValueError("fit_data는 행사용 fit과 실데이터 명시가 필요합니다")
    types = tuple(column_types or ["auto"] * out_features)
    if len(types) != out_features:
        raise ValueError("column_types 폭 불일치")
    d = Diagnostics(out_features, disabled=not ENABLED)
    plan = DistinctPlan(prefix, origin, frozen_stats=deepcopy(frozen_stats),
                        screen=screen, column_types=types, diagnostics=d)
    slots = np.flatnonzero(np.array(origin) == "replica")
    if not ENABLED or not len(slots):
        # 확장이 없으면 입력 생성·RNG 파생·원식 재호출조차 하지 않는다.
        return plan, d
    rng = derived_generator(sampler, prefix, out_features)
    x = fit_data if screen == "fit_data" else canonical_input(
        rng, in_features, dtype=fit_data.dtype if fit_data is not None else torch.float32,
        device=fit_data.device if fit_data is not None else "cpu")
    old = old_output(x)
    if old.shape != (x.shape[0], out_features):
        raise ValueError("old_output shape 불일치")
    count = max(8, 3 * len(slots), 64)
    plan.hp = deepcopy(candidate_hp(rng, count))
    _select_hp(plan.hp, range(count), count)  # 계산 전 수치/후보 축 계약 확인
    candidates = evaluate(x, plan.hp, plan.frozen_stats)
    if candidates.shape != (x.shape[0], count):
        raise ValueError("evaluate는 후보 전체를 한 배치로 반환해야 합니다")
    ct = tuple(candidate_types or ["auto"] * count)
    if len(ct) != count:
        raise ValueError("candidate_types 폭 불일치")
    plan.candidate_types = ct
    first = np.flatnonzero(np.array(origin) == "first")
    merged = torch.cat([old[:, first], candidates], dim=1)
    first_count = len(_count(old[:, first], [types[i] for i in first])["representatives"]) if len(first) else 0
    result = _count(merged, [types[i] for i in first] + list(ct),
                    max_representatives=first_count + len(slots))
    chosen = [i - len(first) for i in result["representatives"] if i >= len(first)]
    chosen = chosen[:len(slots)]
    # HP를 선택 열만 남겨 압축한다. 원 후보 번호는 hp_log에 추적한다.
    plan.hp = _select_hp(plan.hp, chosen, count)
    plan.hp_log = {"values": deepcopy(plan.hp), "source_candidate_indices": tuple(chosen)}
    plan.candidate_types = tuple(ct[i] for i in chosen)
    plan.candidate_indices = tuple(range(len(chosen)))
    plan.output_indices = tuple(int(i) for i in slots[:len(chosen)])
    screened = old.clone()
    if chosen:
        screened[:, plan.output_indices] = candidates[:, chosen]
    screened_types = list(types)
    for dst, src in zip(plan.output_indices, chosen):
        screened_types[dst] = ct[src]
    screened_result = _count(screened, screened_types)
    d.unique_count = len(screened_result["representatives"])
    d.candidate_collisions = len(result["rejected"]) - len(result["constants"])
    d.candidate_constants = len(result["constants"])
    d.collisions = len(screened_result["rejected"]) - len(screened_result["constants"])
    d.constants = len(screened_result["constants"])
    d.gate_reasons = screened_result["rejected"]
    d.missing = max(len(slots) - len(chosen), out_features - d.unique_count)
    d.capacity_reason = capacity_reason or ("candidate_capacity" if len(chosen) < len(slots)
                                           else "preserved_collision" if d.missing else None)
    d.success = d.missing == 0
    return plan, d


def apply_plan(plan: DistinctPlan, x: torch.Tensor, old: torch.Tensor,
               evaluate: Callable, *, metadata=None):
    """(출력, diagnostics). 부족 칸은 old 유지, 전역/메타 카운터 누적.

    입력이 선별 시점과 달라 후보가 충돌할 수도 있으므로 운용 출력도 계측한다.
    용량 부족에는 예외를 내지 않는다. 프로그래밍 오류는 숨기지 않는다.
    """
    if old.shape != (x.shape[0], len(plan.origin)):
        raise ValueError("old shape 불일치")
    if not ENABLED or plan.diagnostics.disabled or "replica" not in plan.origin:
        return old, deepcopy(plan.diagnostics)
    out = old.clone()
    if plan.candidate_indices:
        extra = evaluate(x, plan.hp, plan.frozen_stats)
        out[:, plan.output_indices] = extra[:, plan.candidate_indices]
    types = list(plan.column_types)
    for dst, src in zip(plan.output_indices, plan.candidate_indices):
        types[dst] = plan.candidate_types[src]
    r = _count(out, types)
    d = deepcopy(plan.diagnostics)
    d.unique_count = len(r["representatives"])
    d.constants = len(r["constants"])
    d.collisions = len(r["rejected"]) - d.constants
    d.gate_reasons = r["rejected"]
    # 새 입력에서 충돌한 replica도 옛 칸으로 복원한다(first는 불변).
    failed = [i for i in r["rejected"] if plan.origin[i] == "replica"]
    if failed:
        out[:, failed] = old[:, failed]
        r = _count(out, types)
        d.unique_count = len(r["representatives"])
        d.constants = len(r["constants"])
        d.collisions = len(r["rejected"]) - d.constants
        d.gate_reasons = r["rejected"]
    d.missing = max(plan.diagnostics.missing, len(plan.origin) - d.unique_count)
    d.success = d.missing == 0
    if d.missing:
        d.capacity_reason = d.capacity_reason or "runtime_capacity"
        CAPACITY_COUNTER["events"] += 1
        CAPACITY_COUNTER["missing_columns"] += d.missing
        CAPACITY_COUNTER[plan.prefix] += d.missing
        if metadata is not None:
            counters = metadata.setdefault("distinct_fill", {})
            counters["events"] = counters.get("events", 0) + 1
            counters["missing_columns"] = counters.get("missing_columns", 0) + d.missing
            counters["last_reason"] = d.capacity_reason
    return out, d


# 조각들이 쓰는 고정 이름: fit에서 plan 준비, transform에서 apply_plan.
distinct_fill = prepare_plan
