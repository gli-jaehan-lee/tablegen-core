"""독립 합성 테이블 생성기. 설치 진단은 수치 라이브러리 없이도 실행한다."""
from importlib import import_module
from ._version import __version__
__all__ = ['GenerationConfig','TableResult','generate','generate_single','list_priors','describe','selectable_priors']


def __getattr__(name):
    if name in __all__:
        module = '.api' if name in {'GenerationConfig','TableResult','generate','generate_single'} else '.registry'
        value = getattr(import_module(module, __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
