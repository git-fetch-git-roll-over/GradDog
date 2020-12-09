# :)
import numpy as np
import graddog.math as math
from graddog.trace import Trace, one_parent
import numbers


'''
Any implementable unary (one_parent) or binary (two_parent) operations can be added here
'''

def sin(t : Trace):
    '''
    This allows to create sin().
    Parameters:
        t (Trace instance)

    Return Trace that constitues sin() elementary function
    '''
    try:
        hasvalue=t.val
        return one_parent(t, math.Ops.sin)
    except AttributeError:
        if isinstance(t, numbers.Number) or isinstance(t, np.ndarray):
            return np.sin(t)
        else:
            raise ValueError("Input must be numerical or Trace instance")

def arcsin(t:Trace):
    '''
    This allows to creat arcsin(). ValueError is caught if the input Trace
    instance has value not in the domain of (-1, 1)
    
    Parameters:
        t (Trace instance)

    Return Trace that constitues arcsin() elementary function
    '''
    try:     
        if (t.val <= -1) or (t.val >= 1):
            raise ValueError("The domain of arcsin is (-1, 1)")
        else:
            return one_parent(t, math.Ops.arcsin)
            
    except AttributeError:
        if isinstance(t, numbers.Number) or isinstance(t, np.ndarray):
            if (t <= -1) or (t >= 1):
                raise ValueError("The domain of arcsin is (-1, 1)")
            else:
                return np.arcsin(t)
        else:
            raise ValueError("Input must be numerical or Trace instance")



def cos(t : Trace):
    '''
    This allows to create cos().
    Parameters:
        t (Trace instance)
    Return Trace that constitues cos() elementary function
    '''

    try:
        hasvalue = t.val
        return one_parent(t, math.Ops.cos)
    except AttributeError:
        if isinstance(t, numbers.Number) or isinstance(t, np.ndarray):
            return np.cos(t)
        else:
            raise ValueError("Input must be numerical or Trace instance")



def arccos(t:Trace):
    '''
    This allows to creat arccos(). ValueError is caught if the input Trace
    instance has value not in the domain of (-1, 1)
    
    Parameters:
        t (Trace instance)

    Return Trace that constitues arccos() elementary function
    '''
    try:     
        if (t.val <= -1) or (t.val >= 1):
            raise ValueError("The domain of arccos is (-1, 1)")
        else:
            return one_parent(t, math.Ops.arccos)
    except AttributeError:
        if isinstance(t, numbers.Number) or isinstance(t, np.ndarray):
            if (t <= -1) or (t >= 1):
                raise ValueError("The domain of arccos is (-1, 1)")
            else:
                return np.arccos(t)
        else:
            raise ValueError("Input must be numerical or Trace instance")


def tan(t : Trace):
    '''
    This allows to create tan().
    Parameters:
        t (Trace instance)
    Return Trace that constitues tan() elementary function
    '''
    try:
        hasval = t.val
        return one_parent(t, math.Ops.tan)
    except AttributeError:
        if isinstance(t, numbers.Number) or isinstance(t, np.ndarray):
            return np.tan(t)
        else:
            raise ValueError("Input must be numerical or Trace instance")

def arctan(t:Trace):
    '''
    This allows to creat arctan()
    
    Parameters:
        t: (Trace instance)

    Return Trace that constitues arctan() elementary function
    '''

    try:
        hasval= t.val
        return one_parent(t, math.Ops.arctan)
    
    except AttributeError:
        if isinstance(t, numbers.Number) or isinstance(t, np.ndarray):
            return np.arctan(t)
        else:
            raise ValueError("Input must be numerical or Trace instance")


def exp(t : Trace, base=np.e):
    '''
    This allows to create exp().
    Parameters:
        t (Trace instance)
        base (int, or float)
    Return Trace that constitues exp() elementary function with input base (default=e)
    '''
    try:    
        hasval = t.val
        formula = None
        if base != np.e:
            formula = f'{np.round(base,3)} ^ ({t._trace_name})'
        return one_parent(t, math.Ops.exp, param = base, formula = formula)
    except AttributeError:
        if isinstance(t, numbers.Number) or isinstance(t, np.ndarray):
            return np.power(base, t)
        else:
            raise ValueError("Input must be numerical or Trace instance")


def log(t : Trace, base=np.e):
    '''
    This allows to create log().
    Parameters:
        t (Trace instance)
        base (int, or float)
    Return Trace that constitues log() elementary function with input base (default=e)
    '''
    try:
        hasvalue = t.val
        formula = None
        if base != np.e:
            formula = f'log_{np.round(base,3)}({t._trace_name})'
        return one_parent(t, math.Ops.log, base, formula = formula)
    except AttributeError:
        if isinstance(t, numbers.Number) or isinstance(t, np.ndarray):
            return np.log(t)/np.log(base)
        else:
            raise ValueError("Input must be numerical or Trace instance")

def sinh(t : Trace):
    '''
    This allows to create sinh().
    Parameters:
        t (Trace instance)
        base (int, or float)
    Return Trace that constitues sinh() elementary function
    '''

    try:
        hasvalue = t.val
        return one_parent(t, math.Ops.sinh)
    except AttributeError:
        if isinstance(t, numbers.Number) or isinstance(t, np.ndarray):
            return np.sinh(t)
        else:
            raise ValueError("Input must be numerical or Trace instance")


def cosh(t : Trace):
    '''
    This allows to create cosh().
    Parameters:
        t (Trace instance)
    Return Trace that constitues cosh() elementary function
    '''

    try:
        hasvalue = t.val
        return one_parent(t, math.Ops.cosh)
    except AttributeError:
        if isinstance(t, numbers.Number) or isinstance(t, np.ndarray):
            return np.cosh(t)
        else:
            raise ValueError("Input must be numerical or Trace instance")


def tanh(t : Trace):
    '''
    This allows to create tanh().
    Parameters:
        t (Trace instance)
    Return Trace that constitues tanh() elementary function
    '''

    try:
        hasvalue = t.val
        return one_parent(t, math.Ops.tanh)
    except AttributeError:
        if isinstance(t, numbers.Number) or isinstance(t, np.ndarray):
            return np.tanh(t)
        else:
            raise ValueError("Input must be numerical or Trace instance")


def sqrt(t : Trace):
    '''
    This allows to create sqrt().
    Parameters:
        t (Trace instance)
    Return Trace that constitues sqrt() elementary function
    '''

    try:
        hasvalue = t.val
        return one_parent(t, math.Ops.sqrt)
    except AttributeError:
        if isinstance(t, numbers.Number) or isinstance(t, np.ndarray):
            return np.sqrt(t)
        else:
            raise ValueError("Input must be numerical or Trace instance")



def sigmoid(t : Trace):
    '''
    This allows to create sigmoid().
    Parameters:
        t (Trace instance)
    Return Trace that constitues sigmoig() elementary function
    '''

    try:
        hasvalue = t.val
        return one_parent(t, math.Ops.sigm)
    except AttributeError:
        if isinstance(t, numbers.Number) or isinstance(t, np.ndarray):
            return 1/(1 + np.exp(-t))
        else:
            raise ValueError("Input must be numerical or Trace instance")






    
