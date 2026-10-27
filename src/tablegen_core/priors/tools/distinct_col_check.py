"""원본 distinct-fill 대표열 판정 수치 함수; 연구용 탐침 도구는 제외한다."""
import math
import os
import numpy as np
import torch

VERSION = "distinct-col-check-v3-tablegen-070"
# v3: 생성기는 DAG 의 한 노드일 뿐이다. 노드 판정은 입력 폭 ≥2 탐침만으로 한다.
#   실패 = 어느 탐침에서 복사 열이 절반 이상(대표 ≤ out/2) · 또는 탐침 전체의 복사 열 수 중앙값 ≥ 2. 폭 1 탐침은 기록만.
#   v1·v2 의 「모든 탐침에서 대표 = out」 은 DISTINCT_STRICT=1 에서만 쓴다.
# v2: 복사·미세 오프셋·한 행만 다름·배율/오프셋(아핀)·범주 불일치 부족만 중복으로 센다.
# 순위 동일(argsort·spearman)과 기존 열의 선형 조합(span)은 경고로만 기록한다. v1 엄격 판정은 DISTINCT_STRICT=1.
STRICT = os.environ.get("DISTINCT_STRICT", "0") == "1"
WARN_ONLY = {"argsort", "spearman", "span"}

def _ranks(a):
    """동률은 평균 순위로 계산한다(Spearman)."""
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return (ends - (counts + 1) / 2)[inv]


def _rms(a):
    return float(np.sqrt(np.mean(a * a))) if len(a) else 0.0


class NumericalCheckError(RuntimeError):
    """대표열 관계를 수치적으로 판정하지 못함. 복제/대표 판정으로 바꾸지 않는다."""


def _relative_residual(a, design, *, normalization_cache=None, target_key=None, design_key=None):
    # 정상 경로의 연산 순서 유지. 예외/비유한 중간값만 안정화 + QR로 재평가.
    def normalized(stable=False):
        aa, dd = a, design
        if stable:
            aa = a / max(float(np.max(np.abs(a))), np.finfo(float).tiny)
            dd = design / np.maximum(np.max(np.abs(design), axis=0), np.finfo(float).tiny)
        # 이 호출의 입력 값/마스크가 같은 열만 재사용한다. 연산식·순서·판정 임계는 보존한다.
        tk = ("target", stable, target_key)
        dk = ("design", stable, design_key)
        cached_target = normalization_cache.get(tk) if normalization_cache is not None and target_key is not None else None
        if cached_target is None:
            centered = aa - aa.mean()
            scale = _rms(centered)
            if scale == 0:
                return None, None, 0.0
            target = centered / scale
        else:
            target = cached_target
        cached_design = normalization_cache.get(dk) if normalization_cache is not None and design_key is not None else None
        if cached_design is None:
            d = dd - dd.mean(axis=0)
            norms = np.sqrt(np.mean(d * d, axis=0))
            d = d[:, norms > 0] / norms[norms > 0]
        else:
            d, norms = cached_design
        if not np.isfinite(target).all() or not np.isfinite(d).all() or not np.isfinite(norms).all():
            raise NumericalCheckError('nonfinite normalization in distinct column check')
        if not d.shape[1]:
            return None, None, 1.0
        if normalization_cache is not None:
            if target_key is not None: normalization_cache.put(tk, target)
            if design_key is not None: normalization_cache.put(dk, (d, norms))
        return d, target, None
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            d, target, result = normalized()
            if result is not None:
                return result
            weights = np.linalg.lstsq(d, target, rcond=1e-12)[0]
            residual = _rms(target - d @ weights)
            if not np.isfinite(weights).all() or not np.isfinite(residual):
                raise NumericalCheckError('nonfinite least squares result')
            return residual
    except (np.linalg.LinAlgError, FloatingPointError, NumericalCheckError) as original:
        from tablegen_core.priors import _distinct_fill
        _distinct_fill.CAPACITY_COUNTER['numerical_qr_fallback'] += 1
        try:
            from scipy.linalg import lstsq
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                d, target, result = normalized(stable=True)
                if result is not None:
                    return result
                weights = lstsq(d, target, cond=1e-12, lapack_driver='gelsy', check_finite=True)[0]
                residual = _rms(target - d @ weights)
                if not np.isfinite(weights).all() or not np.isfinite(residual):
                    raise NumericalCheckError('nonfinite QR result')
                return residual
        except (np.linalg.LinAlgError, FloatingPointError, NumericalCheckError, ValueError) as exc:
            _distinct_fill.CAPACITY_COUNTER['numerical_unresolved'] += 1
            raise NumericalCheckError('distinct column least squares and QR fallback failed') from exc


class _NormalizationCache:
    """대표 열 검사 한 번에만 존재하는 8MiB 제한 캐시. RNG/전역 상태를 쓰지 않는다."""
    def __init__(self):
        self.items = {}
        self.bytes = 0

    def get(self, key):
        return self.items.get(key)

    def put(self, key, value):
        if key in self.items:
            return
        arrays = value if isinstance(value, tuple) else (value,)
        size = sum(a.nbytes for a in arrays) + len(key[2][1])
        if size > 8 * 1024 * 1024:
            return
        if len(self.items) >= 512 or self.bytes + size > 8 * 1024 * 1024:
            self.items.clear()
            self.bytes = 0
        self.items[key] = value
        self.bytes += size


def representative_columns(y, *, column_types=None, sentinel_mask=None,
                           sentinel_values=(-1e6, -1e4, 1e4, 1e6), max_representatives=None):
    """열 0부터 탐욕 등록. 상수 열은 대표 0개, span은 연속 대표만 사용.

    형: auto/cont/bin/cat/ang. ang는 라디안 2π 주기의 원거리, bin/cat는
    불일치 행 수 gate. 센티널/경계 행은 수치·순위·회귀에서 제외하고 그 패턴의
    불일치 수를 별도 기록한다. 명시 sentinel_mask는 자동 경계와 합친다.
    반환: representatives, rejected(열→사유), constants, sentinel_mismatches.
    도우미 후보 선별만 max_representatives로 필요한 첫 대표까지 멈출 수 있다.
    CLI 검사에는 이 제한을 쓰지 않는다.
    """
    if isinstance(y, torch.Tensor):
        source_dtype = y.dtype
        eps = torch.finfo(source_dtype).eps if y.is_floating_point() else np.finfo(float).eps
        data = y.detach().cpu().to(torch.float64).numpy()
    else:
        source = np.asarray(y)
        eps = np.finfo(source.dtype).eps if np.issubdtype(source.dtype, np.floating) else np.finfo(float).eps
        data = source.astype(np.float64)
    if data.ndim != 2 or not len(data) or not np.isfinite(data).all():
        raise ValueError("비어 있거나 비유한 또는 2차원 아닌 출력")
    n, width = data.shape
    if max_representatives is not None and max_representatives < 1:
        raise ValueError("max_representatives는 양수")
    tau = max(1e-5, 32 * eps)
    minimum = max(3, math.ceil(.02 * n))
    kinds = list(column_types or ["auto"] * width)
    if len(kinds) != width or any(t not in {"auto", "cont", "bin", "cat", "ang"} for t in kinds):
        raise ValueError("열 형 목록 불일치")
    boundary = np.isin(data, sentinel_values)
    if sentinel_mask is not None:
        mask = np.asarray(sentinel_mask.detach().cpu() if isinstance(sentinel_mask, torch.Tensor) else sentinel_mask, dtype=bool)
        if mask.shape != data.shape:
            raise ValueError("sentinel_mask shape 불일치")
        boundary |= mask
    for j, kind in enumerate(kinds):
        if kind == "auto":
            unique = np.unique(data[:, j])
            kinds[j] = "bin" if np.isin(unique, [0, 1]).all() else (
                "cat" if len(unique) <= 16 and np.equal(unique, np.floor(unique)).all() else "cont")
    reps, continuous, rejected, constants, sentinel_differences, warnings = [], [], {}, [], {}, {}
    normalization_cache = _NormalizationCache()
    rank_cache = {}
    rank_cache_bytes = 0
    def ranked(column, valid, values):
        nonlocal rank_cache_bytes
        key = (column, valid.tobytes())
        if key in rank_cache:
            return rank_cache[key]
        result = (np.argsort(values, kind="stable"), _ranks(values))
        size = len(key[1]) + sum(v.nbytes for v in result)
        if size <= 8 * 1024 * 1024:
            if len(rank_cache) >= 512 or rank_cache_bytes + size > 8 * 1024 * 1024:
                rank_cache.clear()
                rank_cache_bytes = 0
            rank_cache[key] = result
            rank_cache_bytes += size
        return result
    for j in range(width):
        if max_representatives is not None and len(reps) >= max_representatives:
            break
        a = data[:, j]
        valid_a = ~boundary[:, j]
        values = a[valid_a]
        constant = not len(values) or _rms(values - values.mean()) <= tau * max(
            _rms(values), np.finfo(float).tiny)
        if constant:
            rejected[j] = "constant"
            constants.append(j)
            continue
        reason = None
        for i in reps:
            b = data[:, i]
            # 경계 패턴 차이는 기록하지만 S를 키우거나 새 형상으로 세지 않는다.
            sentinel_rows = boundary[:, i] | boundary[:, j]
            mismatch = int(np.count_nonzero(a[sentinel_rows] != b[sentinel_rows]))
            if sentinel_rows.any():
                sentinel_differences[f"{i}:{j}"] = mismatch
            if np.array_equal(a, b):
                reason = "exact"
                break
            valid = ~sentinel_rows
            av, bv = a[valid], b[valid]
            if len(av) < minimum:
                reason = "insufficient_non_sentinel_rows"
                break
            if kinds[j] in {"bin", "cat"} or kinds[i] in {"bin", "cat"}:
                if kinds[j] == kinds[i] and np.count_nonzero(av != bv) < minimum:
                    reason = "categorical_mismatch"
                    break
                continue
            delta = av - bv
            scale = max(_rms(av), _rms(bv), np.finfo(float).tiny)
            if kinds[j] == kinds[i] == "ang":
                delta = np.arctan2(np.sin(delta), np.cos(delta))
                scale = math.pi
            if _rms(delta) <= tau * scale:
                reason = "relative_distance"
                break
            if np.count_nonzero(np.abs(delta) > tau * scale) < minimum:
                reason = "few_different_rows"
                break
            if kinds[j] != "cont" or kinds[i] != "cont":
                continue
            if _relative_residual(av, bv[:, None], normalization_cache=normalization_cache,
                                  target_key=(j, valid.tobytes()), design_key=(i, valid.tobytes())) <= 1e-6:
                reason = "affine"
                break
            order_a, ra = ranked(j, valid, av)
            order_b, rb = ranked(i, valid, bv)
            if np.array_equal(order_a, order_b):
                reason = "argsort"
                break
            ra, rb = ra - ra.mean(), rb - rb.mean()
            denom = np.linalg.norm(ra) * np.linalg.norm(rb)
            if denom and float(ra @ rb / denom) >= 1 - 1e-9:
                reason = "spearman"
                break
        if reason is None and kinds[j] == "cont" and continuous:
            valid = ~np.any(boundary[:, continuous + [j]], axis=1)
            if np.count_nonzero(valid) < minimum:
                reason = "insufficient_span_rows"
            elif _relative_residual(a[valid], data[valid][:, continuous]) <= 1e-6:
                reason = "span"
        if reason in WARN_ONLY and not STRICT:
            warnings[j] = reason
            reason = None
        if reason:
            rejected[j] = reason
        else:
            reps.append(j)
            if kinds[j] == "cont":
                continuous.append(j)
    return {"representatives": reps, "rejected": rejected, "constants": constants,
            "sentinel_mismatches": sentinel_differences, "warnings": warnings}


def representative_count(y, **kwargs):
    return len(representative_columns(y, **kwargs)["representatives"])

