# :)
import numpy as np
import graddog.math as math
from graddog.trace import Trace, one_parent
import numbers
from collections.abc import Iterable

'''
Any implementable unary (one_parent) or binary (two_parent) operations can be added here

'''

def sin(t):
    '''
    This allows to create sin().
    Parameters:
        t (Trace instance, scalar, or vector/list of Traces/scalars)

    Return Trace that constitues sin() elementary function
    '''
    try:   
        t_val = t.val
        return one_parent(t, math.Ops.sin)
    except AttributeError:
        if isinstance(t, numbers.Number):
            return np.sin(t)
        elif isinstance(t, Iterable) and not isinstance(t, str):
            return np.array([sin(t_) for t_ in t])
        else:
            raise TypeError('Input(s) must be Trace or scalar')

def arcsin(t):
    '''
    This allows to creat arcsin(). ValueError is caught if the input Trace
    instance has value not in the domain of (-1, 1)
    
    Parameters:
        t (Trace instance, scalar, or vector/list of Traces/scalars)

    Return Trace that constitues arcsin() elementary function
    '''
    try:   
        t_val = t.val
        if math.in_domain(t_val, math.Ops.arcsin):
            return one_parent(t, math.Ops.arcsin)
        else:
            raise ValueError('Input out of domain')
    except AttributeError:
        if isinstance(t, numbers.Number):
            if math.in_domain(t, math.Ops.arcsin):
                return np.arcsin(t)
            else:
                raise ValueError('Input out of domain')
        elif isinstance(t, Iterable) and not isinstance(t, str):
            return np.array([arcsin(t_) for t_ in t])
        else:
            raise TypeError('Input(s) must be Trace or scalar')

def cos(t):
    '''
    This allows to create cos().
    Parameters:
        t (Trace instance, scalar, or vector/list of Traces/scalars)
    Return Trace that constitues cos() elementary function
    '''
    try:  
        t_val = t.val
        return one_parent(t, math.Ops.cos)
    except AttributeError:
        if isinstance(t, numbers.Number):
            return np.cos(t)
        elif isinstance(t, Iterable) and not isinstance(t, str):
            return np.array([cos(t_) for t_ in t])
        else:
            raise TypeError('Input(s) must be Trace or scalar')

def arccos(t):
    '''
    This allows to creat arccos(). ValueError is caught if the input Trace
    instance has value not in the domain of (-1, 1)
    
    Parameters:
        t (Trace instance, scalar, or vector/list of Traces/scalars)

    Return Trace that constitues arccos() elementary function
    '''
    try:   
        t_val = t.val
        if math.in_domain(t_val, math.Ops.arccos):
            return one_parent(t, math.Ops.arccos)
        else:
            raise ValueError('Input out of domain')
    except AttributeError:
        if isinstance(t, numbers.Number):
            if math.in_domain(t, math.Ops.arccos):
                return np.arccos(t)
            else:
                raise ValueError('Input out of domain')
        elif isinstance(t, Iterable) and not isinstance(t, str):
            return np.array([arccos(t_) for t_ in t])
        else:
            raise TypeError('Input(s) must be Trace or scalar')

def tan(t):
    '''
    This allows to create tan().
    Parameters:
        t (Trace instance, scalar, or vector/list of Traces/scalars)
    Return Trace that constitues tan() elementary function
    '''
    try:   
        t_val = t.val
        return one_parent(t, math.Ops.tan)
    except AttributeError:
        if isinstance(t, numbers.Number):
            return np.tan(t)
        elif isinstance(t, Iterable) and not isinstance(t, str):
            return np.array([tan(t_) for t_ in t])
        else:
            raise TypeError('Input(s) must be Trace or scalar')

def arctan(t):
    '''
    This allows to creat arctan()
    
    Parameters:
        t: (Trace instance, scalar, or vector/list of Traces/scalars)

    Return Trace that constitues arctan() elementary function
    '''
    try:    
        t_val = t.val
        return one_parent(t, math.Ops.arctan)
    except AttributeError:
        if isinstance(t, numbers.Number):
            return np.arctan(t)
        elif isinstance(t, Iterable) and not isinstance(t, str):
            return np.array([arctan(t_) for t_ in t])
        else:
            raise TypeError('Input(s) must be Trace or scalar')

def exp(t, base=np.e):
    '''
    This allows to create exp().
    Parameters:
        t (Trace instance, scalar, or vector/list of Traces/scalars)
        base (int, or float)
    Return Trace that constitues exp() elementary function with input base (default=e)
    '''
    try:   
        t_val = t.val
        formula = None
        if base != np.e:
            formula = f'{base}^{t._trace_name}'
        return one_parent(t, math.Ops.exp, base, formula)
    except AttributeError:
        if isinstance(t, numbers.Number):
            return np.power(base, t)
        elif isinstance(t, Iterable) and not isinstance(t, str):
            return np.array([exp(t_, base) for t_ in t])
        else:
            raise TypeError('Input(s) must be Trace or scalar')

def log(t, base=np.e):
    '''
    This allows to create log().
    Parameters:
        t (Trace instance, scalar, or vector/list of Traces/scalars)
        base (int, or float)
    Return Trace that constitues log() elementary function with input base (default=e)
    '''
    try:  
        t_val = t.val
        formula = None
        if base != np.e:
            formula = f'log_{base}({t._trace_name})'
        if math.in_domain(t_val, math.Ops.log, base):
            return one_parent(t, math.Ops.log, base, formula)
        else:
            raise ValueError('Input out of domain')
    except AttributeError:
        if isinstance(t, numbers.Number):
            if math.in_domain(t, math.Ops.log, base):
                return np.log(t)/np.log(base)
            else:
                raise ValueError('Input out of domain')
        elif isinstance(t, Iterable) and not isinstance(t, str):
            return np.array([log(t_, base) for t_ in t])
        else:
            raise TypeError('Input(s) must be Trace or scalar')

def sinh(t):
    '''
    This allows to create sinh().
    Parameters:
        t (Trace instance, scalar, or vector/list of Traces/scalars)
        base (int, or float)
    Return Trace that constitues sinh() elementary function
    '''
    try:  
        t_val = t.val
        return one_parent(t, math.Ops.sinh)
    except AttributeError:
        if isinstance(t, numbers.Number):
            return np.sinh(t)
        elif isinstance(t, Iterable) and not isinstance(t, str):
            return np.array([sinh(t_) for t_ in t])
        else:
            raise TypeError('Input(s) must be Trace or scalar')

def cosh(t):
    '''
    This allows to create cosh().
    Parameters:
        t (Trace instance, scalar, or vector/list of Traces/scalars)
    Return Trace that constitues cosh() elementary function
    '''
    try:   
        t_val = t.val
        return one_parent(t, math.Ops.cosh)
    except AttributeError:
        if isinstance(t, numbers.Number):
            return np.cosh(t)
        elif isinstance(t, Iterable) and not isinstance(t, str):
            return np.array([cosh(t_) for t_ in t])
        else:
            raise TypeError('Input(s) must be Trace or scalar')

def tanh(t):
    '''
    This allows to create tanh().
    Parameters:
        t (Trace instance, scalar, or vector/list of Traces/scalars)
    Return Trace that constitues tanh() elementary function
    '''
    try:  
        t_val = t.val
        return one_parent(t, math.Ops.tanh)
    except AttributeError:
        if isinstance(t, numbers.Number):
            return np.tanh(t)
        elif isinstance(t, Iterable) and not isinstance(t, str):
            return np.array([tanh(t_) for t_ in t])
        else:
            raise TypeError('Input(s) must be Trace or scalar')

def sqrt(t):
    '''
    This allows to create sqrt().
    Parameters:
        t (Trace instance, scalar, or vector/list of Traces/scalars)
    Return Trace that constitues sqrt() elementary function
    '''
    try:   
        t_val = t.val
        if math.in_domain(t_val, math.Ops.sqrt):
            return one_parent(t, math.Ops.sqrt)
        else:
            raise ValueError('Input out of domain')
    except AttributeError:
        if isinstance(t, numbers.Number):
            if math.in_domain(t, math.Ops.sqrt):
                return t**0.5
            else:
                raise ValueError('Input out of domain')
        elif isinstance(t, Iterable) and not isinstance(t, str):
            return np.array([sqrt(t_) for t_ in t])
        else:
            raise TypeError('Input(s) must be Trace or scalar')

def sigmoid(t):
    '''
    This allows to create sigmoid().
    Parameters:
        t (Trace instance, scalar, or vector/list of Traces/scalars)
    Return Trace that constitues sigmoig() elementary function
    '''
    try: 
        t_val = t.val
        return one_parent(t, math.Ops.sigm)
    except AttributeError:
        if isinstance(t, numbers.Number):
            return 1/(1 + np.exp(-t))
        elif isinstance(t, Iterable) and not isinstance(t, str):
            return np.array([sigmoid(t_) for t_ in t])
        else:
            raise TypeError('Input(s) must be Trace or scalar')





    
