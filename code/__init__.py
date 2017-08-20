# __init__.py

import sys
import warnings

import numpy as _np
import sympy as _sp
import matplotlib as _mpl

__author__ = 'Mitchell, FHT'
__date__ = (2017, 8, 20)
__version__ = '0.2'
__verbose__ = True

_ver_err = "You are running version {2} of {0}, version {1} or higher is required"
_ver_warn = "You are running version {2} of {0}, version {1} or higher is expected"

def _ver_assert(module, req_ver, warn=True):
    assert isinstance(req_ver, tuple)
    cur_ver = tuple(int(x) for x in module.__version__.split('.'))
    if cur_ver < req_ver:
        fmt = module.__name__, '.'.join(map(str, req_ver)), module.__version__
        if warn:
            warnings.warn(_ver_warn.format(*fmt))
        else:
            raise Exception(_ver_err.format(*fmt))


if sys.version_info < (3,6,0):
    _fmt = 'python', '3.6.0', '.'.join(map(str, sys.version_info[:3]))
    raise Exception(_ver_err.format(*_fmt))

_ver_assert(_np, (1, 11, 3))
_ver_assert(_sp, (1, 0))
_ver_assert(_mpl, (2, 0, 0))

from . import driver, main, problem, solver, tests