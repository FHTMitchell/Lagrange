# stepper.py

from collections import deque
from typing import *

import numpy as np

from . import method as _method
from .bases import Base

__author__ = 'Mitchell, FHT'
__date__ = (2017, 8, 20)
VERBOSE = True


class Stepper(Base):
    def __init__(self, stepsize_control: Callable, explicit: _method.Method,
                 implicit: _method.Method = None):

        self.stepsize_control = stepsize_control
        self.explicit = explicit
        if implicit is not None:
            self.implicit = implicit
        else:
            self.implicit = _raise_no_implict

    def predict(self, t: float, y: np.ndarray, yd: deque, k: int):

        h = self.stepsize_control(t, y, yd)
        t1 = t + h

        y1 = self.explicit.explicit(yd, h, k)

        return h, t1, y1

    def correct(self, yd: deque, ynd: Union[np.ndarray, list], k: int,
                h: float):

        return self.implicit.implicit(yd, ynd, k, h)


def _raise_no_implict(*args, **kwargs):
    raise ValueError('Implicit method undefined')
