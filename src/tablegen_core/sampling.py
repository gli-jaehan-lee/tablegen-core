"""기존 GlobalSampler를 보존하면서 최상위 호출 기록과 검증된 고정/범위 설정을 추가한다.

기록 재진입 가드는 원본 viz_prior.RecordingSampler의 패턴을 사용한다.
같은 이름을 여러 번 호출하면 설정을 매번 적용하고 각 호출을 기록한다.
"""
import inspect
import math
import numpy as np
from .core._base import GlobalSampler


def jsonable(value):
    if isinstance(value, np.ndarray):return value.tolist()
    if isinstance(value, np.generic):return value.item()
    if isinstance(value, (str,int,float,bool)) or value is None:return value
    if isinstance(value,(tuple,list)):return [jsonable(v) for v in value]
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    return getattr(value,'__name__',str(value))


def finite_number(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)):return False
    try:return math.isfinite(value)
    except (OverflowError,ValueError):return False


class ConfiguredSampler(GlobalSampler):
    def __init__(self, seed, config, overrides=None):
        super().__init__(seed=seed, config=config)
        self.overrides = overrides or {}
        self.trace = []
        self.used = set()
        self._depth = 0
        # Separate override RNG so range sampling does not corrupt the base sampler state.
        self.override_rng = np.random.default_rng(seed ^ 0x5A17)

    def _call(self, kind, name, *args, **kwargs):
        method = getattr(GlobalSampler,kind)
        top = self._depth == 0
        self._depth += 1
        try:
            if top and name in self.overrides:
                bound=inspect.signature(method).bind(self,name,*args,**kwargs)
                bound.apply_defaults()
                val=self._override(kind,name,bound.arguments)
                self.used.add(name)
            else:
                val=method(self,name,*args,**kwargs)
        finally:
            self._depth -= 1
        if top:
            self.trace.append({'kind':kind,'name':name,'value':jsonable(val),'overridden':name in self.overrides})
        return val

    def _override(self,kind,name,a):
        val=self.overrides[name]
        if isinstance(val,dict):
            if set(val)!={'low','high'} or kind not in ('numerical','randint'):
                raise ValueError(f'{name}: range requires low/high and a numeric sampler')
            lo,hi=val['low'],val['high']
            if any(not finite_number(x) for x in (lo,hi)) or lo>hi:
                raise ValueError(f'{name}: invalid range')
            integer=kind=='randint' or a.get('use_int',False)
            upper=a['high']-1 if kind=='randint' else a['high']
            if lo<a['low'] or hi>upper or (integer and (int(lo)!=lo or int(hi)!=hi)):
                raise ValueError(f'{name}: range outside source bounds [{a["low"]}, {upper}]')
            if integer:val=int(self.override_rng.integers(int(lo),int(hi)+1))
            else:val=float(self.override_rng.uniform(lo,hi))
        if kind in ('numerical','randint'):
            if not finite_number(val):
                raise ValueError(f'{name}: expected finite number')
            integer=kind=='randint' or a.get('use_int',False)
            if integer and int(val)!=val:raise ValueError(f'{name}: expected integer')
            upper=a['high']-1 if kind=='randint' else a['high']
            if not a['low']<=val<=upper:raise ValueError(f'{name}: expected [{a["low"]}, {upper}], got {val}')
            val=np.int64(val) if integer else np.float64(val)
            if a.get('size') is not None:val=np.full(a['size'],val)
        elif kind=='boolean':
            if not isinstance(val,bool):raise ValueError(f'{name}: expected boolean')
        elif kind=='categorical':
            if type(val) is not int or not 0<=val<a['n_categories']:raise ValueError(f'{name}: category index out of range')
        elif kind=='choice':
            choices=a['categories']
            matches=[v for v in choices if type(jsonable(v)) is type(val) and jsonable(v)==val]
            if len(matches)!=1:raise ValueError(f'{name}: choose from {jsonable(choices)}')
            val=matches[0]
        return val

    def finish(self):
        unused=set(self.overrides)-self.used
        if unused:raise ValueError(f'Unknown or unused hyperparameters in this draw: {sorted(unused)}')

    def numerical(self,name,*a,**k):return self._call('numerical',name,*a,**k)
    def randint(self,name,*a,**k):return self._call('randint',name,*a,**k)
    def categorical(self,name,*a,**k):return self._call('categorical',name,*a,**k)
    def choice(self,name,*a,**k):return self._call('choice',name,*a,**k)
    def boolean(self,name,*a,**k):return self._call('boolean',name,*a,**k)
