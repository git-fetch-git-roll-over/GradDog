# :)
import numpy as np
import graddog.math as math
from graddog.trace import Trace, one_parent
import numbers

'''
Any implementable unary (one_parent) or binary (two_parent) operations can be added here

TODO: update docstrings. t does not have to be a trace

TODO: add domain checks in the math file instead of here
'''


def sin(t):
    '''
    This allows to create sin().
    Parameters:
        t (Trace instance)

    Return Trace that constitues sin() elementary function
    '''
    # try:
    #     hasvalue=t.val
    #     return one_parent(t, math.Ops.sin)
    # except AttributeError:
    #     if isinstance(t, numbers.Number) or isinstance(t, np.ndarray):
    #         return np.sin(t)
    #     else:
    #         raise ValueError("Input must be numerical or Trace instance")
    # else:
    #     raise ValueError("Input must be numerical or Trace instance")
    if isinstance(t, numbers.Number):
        return np.sin(t)
    else:
        return one_parent(t, math.Ops.sin)


def arcsin(t):
    '''
    This allows to creat arcsin(). ValueError is caught if the input Trace
    instance has value not in the domain of (-1, 1)
    
    Parameters:
        t (Trace instance)

    Return Trace that constitues arcsin() elementary function
    '''
    if isinstance(t, numbers.Number):
        return np.arcsin(t)
    else:
        return one_parent(t, math.Ops.arcsin)

def cos(t):
    '''
    This allows to create cos().
    Parameters:
        t (Trace instance)
    Return Trace that constitues cos() elementary function
    '''
    if isinstance(t, numbers.Number):
        return np.cos(t)
    else:
        return one_parent(t, math.Ops.cos)

def arccos(t):
    '''
    This allows to creat arccos(). ValueError is caught if the input Trace
    instance has value not in the domain of (-1, 1)
    
    Parameters:
        t (Trace instance)

    Return Trace that constitues arccos() elementary function
    '''
    if isinstance(t, numbers.Number):
        return np.arccos(t)
    else:
        return one_parent(t, math.Ops.cos)

def tan(t):
    '''
    This allows to create tan().
    Parameters:
        t (Trace instance)
    Return Trace that constitues tan() elementary function
    '''
    if isinstance(t, numbers.Number):
        return np.tan(t)
    else:
        return one_parent(t, math.Ops.tan)

def arctan(t):
    '''
    This allows to creat arctan()
    
    Parameters:
        t: (Trace instance)

    Return Trace that constitues arctan() elementary function
    '''

    if isinstance(t, numbers.Number):
        return np.arctan(t)
    else:
        return one_parent(t, math.Ops.arctan)


def exp(t, base=np.e):
    '''
    This allows to create exp().
    Parameters:
        t (Trace instance)
        base (int, or float)
    Return Trace that constitues exp() elementary function with input base (default=e)
    '''
    if isinstance(t, numbers.Number):
        return np.power(base, t)
    else:
        formula = None
        if base != np.e:
            formula = f'{base}^'
        return one_parent(t, math.Ops.exp, base, formula = formula)


def log(t, base=np.e):
    '''
    This allows to create log().
    Parameters:
        t (Trace instance)
        base (int, or float)
    Return Trace that constitues log() elementary function with input base (default=e)
    '''
    if isinstance(t, numbers.Number):
        return np.log(t)/np.log(base)
    else:
        formula = None
        if base != np.e:
            formula = f'log_{np.round(base,3)}'
        return one_parent(t, math.Ops.log, base, formula = formula)



def sinh(t):
    '''
    This allows to create sinh().
    Parameters:
        t (Trace instance)
        base (int, or float)
    Return Trace that constitues sinh() elementary function
    '''
    if isinstance(t, numbers.Number):
        return np.sinh(t)
    else:
        return one_parent(t, math.Ops.sinh)



def cosh(t):
    '''
    This allows to create cosh().
    Parameters:
        t (Trace instance)
    Return Trace that constitues cosh() elementary function
    '''
    if isinstance(t, numbers.Number):
        return np.cosh(t)
    else:
        return one_parent(t, math.Ops.cosh)

def tanh(t):
    '''
    This allows to create tanh().
    Parameters:
        t (Trace instance)
    Return Trace that constitues tanh() elementary function
    '''
    if isinstance(t, numbers.Number):
        return np.tanh(t)
    else:
        return one_parent(t, math.Ops.tanh)


def sqrt(t):
    '''
    This allows to create sqrt().
    Parameters:
        t (Trace instance)
    Return Trace that constitues sqrt() elementary function
    '''

    if isinstance(t, numbers.Number):
        return t**0.5
    else:
        return one_parent(t, math.Ops.sqrt)


def sigmoid(t):
    '''
    This allows to create sigmoid().
    Parameters:
        t (Trace instance)
    Return Trace that constitues sigmoig() elementary function
    '''
    if isinstance(t, numbers.Number):
        return 1/(1+np.exp(-t))
    else:
        return one_parent(t, math.Ops.sigm)






    
