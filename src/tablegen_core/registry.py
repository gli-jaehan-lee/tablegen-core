"""Closed prior registry; user strings never become import paths."""
import json
from pathlib import Path
from . import priors
from .core import _function as functions

_ROOT = Path(__file__).parent
INDEX = json.loads((_ROOT / 'prior_index.json').read_text(encoding='utf-8'))
CATALOG = {p['name']: p for p in json.loads((_ROOT / 'catalog.json').read_text(encoding='utf-8'))}
DEFAULT8 = tuple(functions.PRESETS_DICT['default'].split(','))
ROBUST16 = tuple(priors.NODES)
ALL24 = DEFAULT8 + ROBUST16
PRESETS = {'default': DEFAULT8, 'default8': DEFAULT8, 'robust16': ROBUST16, 'all24': ALL24}


def initialize():
    if set(INDEX['legacy']) != set(DEFAULT8) or set(INDEX['by_name']) != set(ROBUST16):
        raise RuntimeError('Prior index and implementations disagree')
    functions.FCT_TYPES_DICT.update(priors.NODES)


def expand_preset(identifier):
    if not isinstance(identifier, str):
        raise ValueError('prior must be a registered name or prior-N string')
    return list(PRESETS.get(identifier, (identifier,)))


def resolve(identifier):
    initialize()
    if not isinstance(identifier, str):
        raise ValueError('prior must be a registered name or prior-N string')
    if identifier.startswith('prior-'):
        identifier = INDEX['by_code'].get(identifier[6:], identifier)
    if identifier not in ALL24:
        raise ValueError(f'Unknown prior: {identifier}')
    return identifier


def resolve_many(identifiers):
    return [resolve(identifier) for identifier in identifiers]


def list_priors():
    initialize()
    return [dict(CATALOG[name], family='builtin' if name in DEFAULT8 else 'selected',
                 status='available', selectable=True, hold_reason=None)
            for name in INDEX['legacy'] + list(ROBUST16)]


def selectable_priors():
    return [item['name'] for item in list_priors()]


def describe(identifier):
    return next(item for item in list_priors() if item['name'] == resolve(identifier))
