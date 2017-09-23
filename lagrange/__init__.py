# __init__.py

import sys as _sys
import warnings as _warnings

try:
    import matplotlib as _mpl
    import numpy as _np
    import sympy as _sp
    import scipy as _scipy
except ModuleNotFoundError as e:
    apostrophe = "'"
    msg = f'lagrange requires dependency {str(e).split(apostrophe)[1]!r}'
    raise ModuleNotFoundError(msg)
__author__ = 'Mitchell, FHT'
__date__ = (2017, 8, 20)
__version__ = '0.2'
VERBOSE = True

_ver_err = "You are running version {2} of {0}, version {1} or higher is " \
           "required"
_ver_warn = "You are running version {2} of {0}, version {1} or higher is " \
            "expected"


def _ver_assert(module, req_ver, warn=True):
    assert isinstance(req_ver, tuple)
    cur_ver = tuple(int(x) for x in module.__version__.split('.'))
    if cur_ver < req_ver:
        fmt = module.__name__, '.'.join(map(str, req_ver)), module.__version__
        if warn:
            _warnings.warn(_ver_warn.format(*fmt))
        else:
            raise Exception(_ver_err.format(*fmt))


if _sys.version_info < (3, 6, 0):
    _fmt = 'python', '3.6.0', '.'.join(map(str, _sys.version_info[:3]))
    raise Exception(_ver_err.format(*_fmt))


_ver_assert(_np, (1, 11, 3))
_ver_assert(_sp, (1, 0))
_ver_assert(_mpl, (2, 0, 0))
_ver_assert(_scipy, (0, 18, 1))

from . import driver, problem, solver, main, bodies
from .main import all_classes
