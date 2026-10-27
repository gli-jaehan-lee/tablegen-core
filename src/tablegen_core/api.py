"""설정 → 생성 → 검증. 전역 RNG를 사용하는 원본 호출은 직렬화하고 상태를 복원한다."""
from dataclasses import replace, dataclass, field, asdict, fields
import hashlib
import json
import math
import random
import threading
import sys
from contextlib import contextmanager
from importlib import import_module
import warnings
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import scipy
from .registry import resolve, resolve_many, functions, INDEX, expand_preset, PRESETS
from ._version import __version__
from .sampling import ConfiguredSampler, jsonable
from .core._base import Context, DatasetProperties
from .core._config import PriorConfig
from .core._dataset import RandomDataset, GraphRejectionLimitError
from .postprocess import outlier_clip, all_features_constant

class NonfiniteOutputError(RuntimeError):
    """Generated output is nonfinite; no sentinel replacement is performed."""


_LOCK=threading.RLock()
SINGLE_OPTIONS = {'multi_fct_types': 'concat', 'allow_kumaraswamy_warping': False, 'allow_act_warping': False}
_UNSUPPORTED={'filter_unpredictable_datasets','remove_trivial_datasets','trivial_dataset_threshold','seq_lens_multiple_of'}
_DIRECT_OPTIONS={'meta_sampling_mode','random_matrix_types','random_weights_types','max_discretization_cardinality','use_corrected_cat_meta_sampling'}
_ENUMS={'meta_sampling_mode':{'meta','global','local'},'random_matrix_types':{'default','gaussian'},'multi_fct_types':{'default','concat','agg'},'random_weights_types':{'simple','uniform'},'cat_modes':{'default','neighbor','softmax','neighbor_id','neighbor_disc','neighbor_func','neighbor_int','softmax_id','softmax_disc','softmax_int'}}

@dataclass
class GenerationConfig:
    mode: str = 'direct'
    rows: int = 128
    features: int = 4
    targets: int = 1
    seed: int = 42
    priors: list[str] = field(default_factory=lambda:['lin'])
    hyperparameters: dict = field(default_factory=dict)
    graph_options: dict = field(default_factory=dict)
    feature_categories: list[int] | None = None
    target_categories: list[int] | None = None
    max_graph_attempts: int = 100
    single_prior: bool = False
    distinct_fill: bool = True
    prior_weights: dict | None = None
    restrict_cheap_functions: bool | None = None  # None = auto: on in DAG when the pool has a cheap function
    outlier_clip: float | None = 4.0
    max_table_attempts: int = 32
    categorical_sampling: str = 'explicit'
    max_category_size: int = 200

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value,dict):raise ValueError('configuration must be a JSON object')
        unknown=set(value)-{f.name for f in fields(cls)}
        if unknown:raise ValueError(f'Unknown configuration fields: {sorted(unknown)}')
        return cls(**value)

    def validated(self, *, max_features=256, max_cells=2_000_000):
        for key, value in (("max_features", max_features), ("max_cells", max_cells)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{key}: positive integer required")
        if type(self.distinct_fill) is not bool:
            raise ValueError("distinct_fill: boolean required")
        if type(self.single_prior) is not bool:
            raise ValueError("single_prior: boolean required")
        if self.outlier_clip is not None and (isinstance(self.outlier_clip,bool) or not isinstance(self.outlier_clip,(int,float)) or not self.outlier_clip>0):raise ValueError('outlier_clip: positive number or null required')
        if type(self.max_table_attempts) is not int or not 1<=self.max_table_attempts<=1000:raise ValueError('max_table_attempts: integer in [1, 1000] required')
        if self.mode not in ('direct','dag'):raise ValueError('mode must be direct or dag')
        for key,lo,hi in [('rows',2,100000),('features',1,max_features),('targets',1,256),('seed',0,2**32-1),('max_graph_attempts',1,10000)]:
            v=getattr(self,key)
            if type(v) is not int or not lo<=v<=hi:raise ValueError(f'{key}: integer in [{lo}, {hi}] required')
        if self.rows*(self.features+self.targets)>max_cells:raise ValueError(f'requested table exceeds {max_cells:,} cells')
        if not isinstance(self.priors,list) or not self.priors:raise ValueError('priors must be a nonempty list')
        requested=[]
        for identifier in self.priors:
            requested.extend(expand_preset(identifier))
        names=resolve_many(requested)
        if len(set(names))!=len(names):raise ValueError('duplicate priors are not mixture weights')
        if self.mode=='direct' and len(names)!=1:raise ValueError('direct mode requires one prior; use dag for mixtures')
        if not isinstance(self.hyperparameters,dict) or any(not isinstance(k,str) for k in self.hyperparameters):raise ValueError('hyperparameters must map names to values')
        if not isinstance(self.graph_options,dict):raise ValueError('graph_options must be an object')
        defaults=asdict(PriorConfig()); supplied=dict(self.graph_options)
        invalid=set(supplied)-(set(defaults)-_UNSUPPORTED-{'fct_types'})
        if invalid:raise ValueError(f'Unknown, training-only, or managed graph options: {sorted(invalid)}')
        if self.mode=='direct' and set(supplied)-_DIRECT_OPTIONS:raise ValueError('DAG-only graph_options supplied in direct mode')
        for key,val in supplied.items():
            if type(val) is not type(defaults[key]) and not (type(defaults[key]) is float and type(val) is int):raise ValueError(f'{key}: wrong value type')
            if type(defaults[key]) is float:
                try:val=float(val)
                except (OverflowError,ValueError):raise ValueError(f'{key}: expected finite value') from None
                if not math.isfinite(val):raise ValueError(f'{key}: expected finite value')
                supplied[key]=val
            if key in _ENUMS and val not in _ENUMS[key]:raise ValueError(f'{key}: choose from {sorted(_ENUMS[key])}')
        if self.single_prior:
            if self.mode != 'dag' or len(names) != 1:
                raise ValueError('single_prior requires dag mode and exactly one prior')
            for key, value in SINGLE_OPTIONS.items():
                if key in supplied and supplied[key] != value:
                    raise ValueError(f'graph_options.{key}: single_prior requires {value!r}')
                supplied[key] = value
        if self.restrict_cheap_functions is not None and type(self.restrict_cheap_functions) is not bool:
            raise ValueError('restrict_cheap_functions: boolean or null required')
        if self.categorical_sampling not in ('explicit', 'tabicl'):
            raise ValueError('categorical_sampling must be explicit or tabicl')
        if type(self.max_category_size) is not int or not 2 <= self.max_category_size <= 1024:
            raise ValueError('max_category_size must be integer 2..1024')
        if self.categorical_sampling == 'tabicl' and (self.mode != 'dag' or self.feature_categories is not None or self.single_prior):
            raise ValueError('tabicl category sampling requires DAG and no explicit feature_categories or single_prior')
        if self.mode == 'direct' and (self.prior_weights is not None or self.restrict_cheap_functions is True):
            raise ValueError('weights and cheap pool restriction require DAG mode')
        if self.single_prior and self.prior_weights is not None:
            raise ValueError('single_prior does not accept mixture weights')
        selection = names
        if self.prior_weights is not None:
            if not isinstance(self.prior_weights, dict) or not self.prior_weights:
                raise ValueError('prior_weights must be a nonempty mapping of positive integer multiplicities')
            weights={n:1 for n in names}; assigned=set()
            for identifier, value in self.prior_weights.items():
                if type(value) is not int or not 1 <= value <= 10000:
                    raise ValueError('prior_weights values must be positive integers <=10000')
                expanded=resolve_many(expand_preset(identifier))
                if any(n not in weights for n in expanded) or assigned.intersection(expanded):
                    raise ValueError('weight keys must identify selected priors without overlapping aliases')
                assigned.update(expanded)
                for n in expanded: weights[n]=value
            if sum(weights.values()) > 100000:
                raise ValueError('total mixture multiplicity must be <=100000')
            selection=[]
            for identifier in self.priors:
                group=resolve_many(expand_preset(identifier))
                weight_key = identifier
                if identifier in ('default', 'default8'):
                    weight_key = next((k for k in ('default', 'default8') if k in self.prior_weights), identifier)
                if identifier in PRESETS and weight_key in self.prior_weights:
                    selection.extend(group * self.prior_weights[weight_key])
                else:
                    selection.extend(n for n in group for _ in range(weights[n]))
        if self.restrict_cheap_functions is None:
            self.restrict_cheap_functions = self.mode == 'dag' and bool(set(names).intersection({'gp','tree','disc','lin','quad'}))
        if self.restrict_cheap_functions and not set(names).intersection({'gp','tree','disc','lin','quad'}):
            raise ValueError('restricted cheap pool must include at least one of gp/tree/disc/lin/quad')
        cfg=PriorConfig(**supplied, fct_types=','.join(selection))
        if not 2<=cfg.min_n_nodes<=cfg.max_n_nodes<=128:raise ValueError('require 2 <= min_n_nodes <= max_n_nodes <= 128')
        if not 2<=cfg.max_discretization_cardinality<=4096:raise ValueError('max_discretization_cardinality must be 2..4096')
        for key,n in [('feature_categories',self.features),('target_categories',self.targets)]:
            cats=getattr(self,key)
            if cats is not None:
                if not isinstance(cats,list) or len(cats)!=n or any(type(c) is not int or c<0 or c==1 or c>1024 for c in cats):raise ValueError(f'{key}: need {n} cardinalities, each 0 or 2..1024')
                if self.single_prior and any(cats):raise ValueError('single_prior is numeric only')
                if self.mode=='direct' and any(cats):raise ValueError('categorical columns require dag mode')
        return cfg,names

@dataclass
class TableResult:
    data: np.ndarray
    columns: list[str]
    metadata: dict

    def write(self, directory):
        from .outputs import write_result
        return write_result(self, directory)



@contextmanager
def _distinct_fill_request(enabled):
    """Temporarily override upstream import-time flags under the generation lock.

    Node imports are eager; patch their copied ENABLED bindings as well as the
    helper and pin the checker to its upstream default non-strict policy.
    Use a fresh Counter object so pre-existing diagnostics are preserved.
    This compatibility switch disables only upstream replica repair.
    """
    helper = import_module("tablegen_core.priors._distinct_fill")
    checker = import_module("tablegen_core.priors.tools.distinct_col_check")
    old_strict = checker.STRICT
    modules = [helper] + [module for name, module in list(sys.modules.items())
                          if name.startswith("tablegen_core.priors.nodes.")
                          and module is not None and hasattr(module, "ENABLED")]
    saved = [(module, module.ENABLED) for module in modules]
    old_counter = helper.CAPACITY_COUNTER
    counter = Counter()
    try:
        for module, _ in saved:
            module.ENABLED = enabled
        helper.CAPACITY_COUNTER = counter
        checker.STRICT = False
        yield {"enabled": enabled,
               "mode": "repaired" if enabled else "legacy",
               "helper_version": helper.VERSION,
               "checker_version": checker.VERSION,
               "strict": False,
               "compatibility_scope": "upstream replica repair only; not a package-version rollback",
               "diagnostics_scope": "helper capacity events only; custom node repairs may not report counters",
               "capacity_counter": counter}
    finally:
        checker.STRICT = old_strict
        helper.CAPACITY_COUNTER = old_counter
        for module, value in saved:
            module.ENABLED = value


def generate(config=None, *, max_features=256, max_cells=2_000_000, **kwargs):
    """Generate one table. Accept a GenerationConfig, JSON-like dict, or keyword fields.

    Defaults (0.2.0): restrict_cheap_functions auto (on in DAG when the pool has a cheap function); numeric columns are clamped with a
    robust two-pass mean +- 4 std rule over all rows (outlier_clip=None disables); a
    nonfinite, graph-rejected or all-constant-input table is redrawn with a derived seed
    up to max_table_attempts times (1 disables). Same seed/config/versions -> same data.
    """
    if config is not None and kwargs:raise ValueError('use config or keywords, not both')
    cfg=replace(config) if isinstance(config,GenerationConfig) else GenerationConfig.from_dict(config if config is not None else kwargs)
    cfg.validated(max_features=max_features, max_cells=max_cells)
    rejected=[]
    for attempt in range(cfg.max_table_attempts):
        seed=cfg.seed if attempt==0 else (cfg.seed+attempt*1_000_003)%(2**32)
        try:
            result=_generate_once(replace(cfg,seed=seed),max_features=max_features,max_cells=max_cells)
        except (NonfiniteOutputError,GraphRejectionLimitError) as e:
            if cfg.max_table_attempts==1:raise
            rejected.append({'seed':seed,'reason':type(e).__name__});continue
        if cfg.max_table_attempts>1 and all_features_constant(result.data,cfg.features):
            rejected.append({'seed':seed,'reason':'all_inputs_constant'});continue
        break
    else:
        raise RuntimeError(f'No usable table after {cfg.max_table_attempts} attempts: {rejected[-3:]}')
    data,changed=result.data,0
    if cfg.outlier_clip is not None:
        cats=list(result.metadata.get('feature_categories') or [])+list(cfg.target_categories or [0]*cfg.targets)
        numeric=[i for i,c in enumerate(cats) if not c] if len(cats)==data.shape[1] else None
        data,changed=outlier_clip(data,float(cfg.outlier_clip),numeric)
    md=result.metadata
    md['config']=asdict(cfg);md['seed']=cfg.seed;md['used_seed']=seed
    md['table_attempts']={'used':len(rejected)+1,'rejected':rejected}
    md['outlier_clip']={'threshold':cfg.outlier_clip,'clipped_cells':changed,'scope':'all rows of the table, numeric columns only'}
    md['data_sha256']=hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()
    md['notes']=[n for n in md['notes'] if 'outer outlier' not in n]+(['Robust outer outlier clamp applied over all rows (transductive).'] if cfg.outlier_clip is not None else [])
    return TableResult(data,result.columns,md)


def _generate_once(config=None, *, max_features=256, max_cells=2_000_000, **kwargs):
    """Generate one table. Accept a GenerationConfig, JSON-like dict, or keyword fields.

    Same seed/config/dependency versions yield identical data. Direct mode samples
    Gaussian X then computes f(X). DAG mode uses the vendored RandomDataset pipeline,
    not the outer model-training pipeline. Exceptions never silently drop rows.
    """
    if config is not None and kwargs:raise ValueError('use config or keywords, not both')
    cfg=config if isinstance(config,GenerationConfig) else GenerationConfig.from_dict(config if config is not None else kwargs)
    prior_cfg,names=cfg.validated(max_features=max_features, max_cells=max_cells)
    with _LOCK, _distinct_fill_request(cfg.distinct_fill) as fill_metadata:
        np_state=np.random.get_state();torch_state=torch.random.get_rng_state();py_state=random.getstate()
        old_threads=torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            np.random.seed(cfg.seed);torch.manual_seed(cfg.seed);random.seed(cfg.seed)
            sampler=ConfiguredSampler(cfg.seed,prior_cfg,cfg.hyperparameters)
            if cfg.single_prior:
                sampler.single_prior_state = {'prior': names[0], 'depth': 0, 'cheap_as_p': 0, 'recursion_guard_fallbacks': 0}
            if cfg.restrict_cheap_functions:
                sampler.cheap_function_names = set(names)
            ctx=Context(sampler=sampler,config=prior_cfg,seed=cfg.seed)
            ctx.max_graph_attempts=cfg.max_graph_attempts
            with torch.no_grad():
                if cfg.mode=='direct':
                    x=torch.randn(cfg.rows,cfg.features)
                    y=functions.FCT_TYPES_DICT[names[0]](ctx,cfg.features,cfg.targets)(x)
                    data=torch.cat([x,y],dim=1)
                    columns=[f'x_{i}' for i in range(cfg.features)]+[f'y_{i}' for i in range(cfg.targets)]
                    graph=[[],[0]];node_features=[columns[:cfg.features],columns[cfg.features:]];attempts=1
                else:
                    feature_categories=cfg.feature_categories or [0]*cfg.features
                    if cfg.categorical_sampling == 'tabicl':
                        from .core._properties import sample_categorical_sizes
                        feature_categories=[int(c) for c in sample_categorical_sizes(cfg.features,ctx,max_cat_size=cfg.max_category_size)]
                    prop=DatasetProperties(n_train=cfg.rows,n_test=0,cat_sizes={'x':feature_categories,'y':cfg.target_categories or [0]*cfg.targets})
                    engine=RandomDataset(ctx)
                    engine.context.max_graph_attempts=cfg.max_graph_attempts
                    ds=engine.sample(prop)
                    columns=list(prop.feature_specs)
                    data=torch.cat([ds.tensors[n].reshape(cfg.rows,-1) for n in columns],dim=1)
                    graph=ds.kwargs['graph'];node_features=ds.kwargs['node_features'];attempts=ds.kwargs['attempts']
            sampler.finish()
            if tuple(data.shape)!=(cfg.rows,cfg.features+cfg.targets):raise RuntimeError(f'Invalid output shape: {tuple(data.shape)}')
            if not torch.isfinite(data).all():raise NonfiniteOutputError('Nonfinite output from prior; no silent replacement applied')
            array=data.detach().cpu().numpy().copy()
            metadata={'schema_version':1,'package_version':__version__,'config':asdict(cfg),'resolved_priors':names,'resolved_graph_options':asdict(prior_cfg),'mode':cfg.mode,'seed':cfg.seed,'shape':list(array.shape),'columns':columns,'graph':graph,'node_features':node_features,'graph_attempts':attempts,'status':'success','hyperparameters_requested':jsonable(cfg.hyperparameters),'hyperparameters_consumed':sorted(sampler.used),'sampling_trace':sampler.trace,'source_manifest_sha256':hashlib.sha256(Path(__file__).with_name('source_manifest.json').read_bytes()).hexdigest(),'data_sha256':hashlib.sha256(array.tobytes()).hexdigest(),'versions':{'numpy':np.__version__,'torch':torch.__version__,'scipy':scipy.__version__},'notes':['CPU float32 generation; row order preserved.','DAG group order sorted and graph rejection bounded.','Direct input is Gaussian; DAG samples root distributions.','No training-time filtering, padding, class balancing, or outer outlier removal.','Source priors may clamp or sanitize internally; their original code is preserved.']}
            metadata['distinct_fill'] = dict(fill_metadata, capacity_counter=dict(fill_metadata['capacity_counter']))
            metadata['feature_categories'] = feature_categories if cfg.mode == 'dag' else [0]*cfg.features
            metadata['generation_limits'] = {'max_features': max_features, 'max_cells': max_cells}
            if cfg.single_prior:
                state = sampler.single_prior_state
                draws = Counter((t['name'], str(t['value'])) for t in sampler.trace
                                if t['name'] in ('random_function_fct_type', 'cheap_random_function_fct_type'))
                metadata['single_prior'] = {
                    'prior': names[0], 'forced_options': dict(SINGLE_OPTIONS),
                    'root_distribution': 'standard_normal', 'numeric_only': True,
                    'cheap_as_p': state['cheap_as_p'],
                    'recursion_guard_fallbacks': state['recursion_guard_fallbacks'],
                    'pure_single': state['recursion_guard_fallbacks'] == 0,
                    'purity_scope': 'function selection only; prior internals and node normalization remain',
                    'trace_scope': 'top-level sampler calls, not a complete nested function audit',
                    'function_draws': {f'{n}:{v}': c for (n, v), c in sorted(draws.items())},
                }
                if state['recursion_guard_fallbacks']:
                    warnings.warn(f"single_prior recursion guard used original cheap functions "
                                  f"{state['recursion_guard_fallbacks']} time(s); pure_single=False", UserWarning, stacklevel=2)
            return TableResult(array,columns,metadata)
        finally:
            np.random.set_state(np_state);torch.random.set_rng_state(torch_state);random.setstate(py_state)
            torch.set_num_threads(old_threads)


def generate_single(prior, rows, features, targets=1, seed=0, *, strict=True, **kwargs):
    """Use one selected function throughout a numeric DAG (see purity_scope metadata).

    strict=False delegates to ordinary DAG generation unchanged. Recursive priors
    can require the legacy cheap-function fallback; this emits a warning and
    sets pure_single=False instead of claiming an exclusively selected function.
    """
    if type(prior) is not str or type(strict) is not bool:
        raise ValueError('prior must be a string and strict must be boolean')
    if {'mode', 'priors', 'single_prior'} & kwargs.keys():
        raise ValueError('mode, priors and single_prior are managed by generate_single')
    return generate(mode='dag', rows=rows, features=features, targets=targets,
                    seed=seed, priors=[prior], single_prior=strict, **kwargs)
