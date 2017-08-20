# dsympy.py - helper functions for converting differential equations in sympy
#             into numeric lambda functions

import sympy as sp
import collections
from sympy.utilities.lambdify import lambdastr as _lambdastr
from sympy.core.function import AppliedUndef
from typing import *

__author__ = 'Mitchell, FHT'
__date__ = (2017, 8, 20)
__verbose__ = True

# needs to be at the top for typing.
class AutoFunc:
    """
    Holds the representation of an auto-generated function
    """

    def __init__(self, func, as_str, args, sym_func=None, name=None):
        self.func = func
        self.as_str = as_str
        self.args = self.params = tuple(str(arg) for arg in args)
        self.sym_func = sym_func
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __str__(self):
        return self.as_str

    def __repr__(self):
        if self.name is None:
            return "<AutoFunc with arg{}: {}>".format(
                's' if len(self.args) != 1 else '',
                ', '.join(self.args) if self.args else ()
            )
        return f'<AutoFunc: {self.name}>'


def dummify_undefined_functions(expr:sp.Expr, ret_map:bool=False) -> sp.expr:
    """
    Used for solving issues with lambdify and differentials. Replaces undefined 
    functions (eg. f(t), df/dt, dg^2/dxdy) with symbols (named f and df_dt and 
    df_dxdy for the previous examples respectively).
    
    By u/PeterE from
    http://stackoverflow.com/questions/29920641/lambdify-a-sp-expression-that-contains-a-derivative-of-undefinedfunction
    
    Issues (u/PeterE):
        
    * no guard against name-collisions
    * perhaps not the best possible name-scheme: df_dxdy for Derivative(f(x,y), x, y)
    * it is assumed that all derivatives are of the form: Derivative(s(t), t, ...) 
      with s(t) being an UndefinedFunction and t a Symbol. I have no idea what will 
      happen if any argument to Derivative is a more complex expression. I kind of 
      think/hope that the (automatic) simplification process will reduce any more 
      complex derivative into an expression consisting of 'basic' derivatives. But 
      I certainly do not guard against it.
    * largely untested (except for my specific use-cases)
    """
    
    mapping = {}    

    # replace all Derivative terms
    for der in expr.atoms(sp.Derivative):
        f_name = der.expr.func.__name__
        var_names = [var.name for var in der.variables]
        name = "d%s_d%s" % (f_name, 'd'.join(var_names))
        mapping[der] = sp.Symbol(name)

    # replace undefined functions
    for f in expr.atoms(AppliedUndef):
        f_name = f.func.__name__
        mapping[f] = sp.Symbol(f_name)
    
    new_expr = expr.subs(mapping)
    return new_expr if not ret_map else (new_expr, mapping)
    
    
def dlambdify(params: tuple,
              expr: sp.Expr,
              *,
              show: bool=False,
              retstr: bool=False,
              **kwargs
              ) -> Callable:
    """
    See sp.lambdify. Used to create lambdas (or strings of lambda expressions
    if `retstr` is True) from sp expressions. Fixes the issues of derivatives
    not working in sp's basic lambdify.
    """
    
    try:
        iter(params)
    except TypeError:
        params = (params,)
    
    
    #dparams = [dummify_undefined_functions(s) for s in params]
    dexpr = dummify_undefined_functions(expr)
    
    if show or retstr:
        s = _lambdastr(params, dexpr, dummify=False, **kwargs)
        if show:
            print(s)
        if retstr:
            return s
            
    return sp.lambdify(params, dexpr, dummify=False, **kwargs)
    
def dlambdastr(params: tuple, expr: sp.Expr, **kwargs) -> str:
    """
    Equivalent to dlambdify(params, expr, retstr=True, **kwargs)
    """
    return dlambdify(params, expr, retstr=True, **kwargs)
    
    
def auto(expr,
         consts: dict = None,
         params: List[str] = None,
         dfuncs: dict = None,
         name: str = None,
         *,
         show: bool = False,
         just_func: bool = False,
         **kwargs
         ) -> AutoFunc:
    """
    Similar to dlambdify, but automatically discovers all parameters in 
    `expr`. 
    
    `consts`, if used, should be dict of {sp.Symbol: float}, and will 
    replace any constants in `expr` with values. Otherwise, they will be included
    in the final lambda. 
    
    If `show` is True, will print the lambda python expression made.
    
    If `just_func` is True, will only return the function, otherwise a 
    AutoFunc instance with attributes: func, args, as_str.
    """
    
    assert hasattr(params, '__iter__'), \
           f'prams must be iterable, currently {type(params).__name__!r}'
    
    if consts is None:
        consts = {}    
    for const, value in consts.items():
        expr = expr.subs(const, value)
    
    dexpr = dummify_undefined_functions(expr)
    
    if dfuncs is None:
        dfuncs = {}
    for dfunc, value in dfuncs.items():
        dexpr = dexpr.subs(dexpr, value)    
    
    if params is None:
        params = sorted(dexpr.atoms(sp.Symbol), 
                        key=lambda s: [len(str(s)), str(s)])
    elif any(isinstance(p, str) for p in params):
        # this actually works if params is just a str, not a tuple
        params = sp.symbols(params)
    
    s = _lambdastr(params, dexpr, dummify=False, **kwargs)
    if show:
        print(s)
       
    f = sp.lambdify(params, dexpr, dummify=False, **kwargs)
    return AutoFunc(f, s, params, sym_func=expr, name=name) if not just_func else f
    
    

def test_func() -> NamedTuple:
    """
    Returns t, x0, x, f, df
    """
    t, x0 = sp.symbols('t x0')
    x = sp.Function('x')(t)
    
    f = 1/sp.sqrt(x - x0)
    df = sp.diff(f, t)  

    return collections.namedtuple('Syms', 't, x0, x, f, df')(t, x0, x, f, df)


def test() -> Tuple[sp.Expr, str]:
    """
    Test case to ensure all is working smoothly
    """
    
    t, x0, x, f, df = test_func()
    return df, dlambdastr([x, sp.diff(x, t)], df)

