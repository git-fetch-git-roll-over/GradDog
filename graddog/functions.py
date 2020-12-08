# :)
import numpy as np
import graddog.math as math
from graddog.trace import Trace, one_parent

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
        return np.sin(t)


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
        return np.arcsin(t)


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
        return np.cos(t)
        
    



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
        return np.arccos(t)

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
        return np.tan(t)


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
        return np.arctan(t)


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
        return np.power(base,t)

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
        return np.log(t)/np.log(base)

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
        return np.sinh(t)

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
        return np.cosh(t)
    
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
        return np.tanh(t)

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
        return np.sqrt(t)

def sigmoid(t : Trace):
    '''
    This allows to create sigmoid().
    Parameters:
        t (Trace instance)
    Return Trace that constitues sigmoid() elementary function
    '''
    try:
        hasvalue = t.val
        return one_parent(t, math.Ops.sigm)
    except AttributeError:
        return 1/(1 + np.exp(-t.val))





    
