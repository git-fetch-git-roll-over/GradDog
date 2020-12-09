# :)
import numpy as np
import graddog.math as math
from graddog.trace import Trace, one_parent
import numbers

'''
Any implementable unary (one_parent) or binary (two_parent) operations can be added here

<<<<<<< HEAD
TODO: update docstrings. t does not have to be a trace

TODO: add domain checks in the math file instead of here
'''


def sin(t):
#=======
#def sin(t : Trace):
#>>>>>>> master
    '''
    This allows to create sin().
    Parameters:
        t (Trace instance)

    Return Trace that constitues sin() elementary function
    '''
#<<<<<<< HEAD
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

# =======
#     try:
#         hasvalue=t.val
#         return one_parent(t, math.Ops.sin)
#     except AttributeError:
#         if isinstance(t, numbers.Number) or isinstance(t, np.ndarray):
#             return np.sin(t)
#         else:
#             raise ValueError("Input must be numerical or Trace instance")
# >>>>>>> master

def arcsin(t):
    '''
    This allows to creat arcsin(). ValueError is caught if the input Trace
    instance has value not in the domain of (-1, 1)
    
    Parameters:
        t (Trace instance)

    Return Trace that constitues arcsin() elementary function
    '''
#<<<<<<< HEAD
    if isinstance(t, numbers.Number):
        return np.arcsin(t)
    else:
        return one_parent(t, math.Ops.arcsin)
# =======
#     try:     
#         if (t.val <= -1) or (t.val >= 1):
#             raise ValueError("The domain of arcsin is (-1, 1)")
#         else:
#             return one_parent(t, math.Ops.arcsin)
            
#     except AttributeError:
#         if isinstance(t, numbers.Number) or isinstance(t, np.ndarray):
#             if (t <= -1) or (t >= 1):
#                 raise ValueError("The domain of arcsin is (-1, 1)")
#             else:
#                 return np.arcsin(t)
#         else:
#             raise ValueError("Input must be numerical or Trace instance")
# >>>>>>> master

def cos(t):
    '''
    This allows to create cos().
    Parameters:
        t (Trace instance)
    Return Trace that constitues cos() elementary function
    '''
#<<<<<<< HEAD
    if isinstance(t, numbers.Number):
        return np.cos(t)
    else:
        return one_parent(t, math.Ops.cos)

def arccos(t):
# =======

#     try:
#         hasvalue = t.val
#         return one_parent(t, math.Ops.cos)
#     except AttributeError:
#         if isinstance(t, numbers.Number) or isinstance(t, np.ndarray):
#             return np.cos(t)
#         else:
#             raise ValueError("Input must be numerical or Trace instance")



#def arccos(t:Trace):
#>>>>>>> master
    '''
    This allows to creat arccos(). ValueError is caught if the input Trace
    instance has value not in the domain of (-1, 1)
    
    Parameters:
        t (Trace instance)

    Return Trace that constitues arccos() elementary function
    '''
#<<<<<<< HEAD
    if isinstance(t, numbers.Number):
        return np.arccos(t)
    else:
        return one_parent(t, math.Ops.arccos)
# =======
#     try:     
#         if (t.val <= -1) or (t.val >= 1):
#             raise ValueError("The domain of arccos is (-1, 1)")
#         else:
#             return one_parent(t, math.Ops.arccos)
#     except AttributeError:
#         if isinstance(t, numbers.Number) or isinstance(t, np.ndarray):
#             if (t <= -1) or (t >= 1):
#                 raise ValueError("The domain of arccos is (-1, 1)")
#             else:
#                 return np.arccos(t)
#         else:
#             raise ValueError("Input must be numerical or Trace instance")
# >>>>>>> master

def tan(t):
    '''
    This allows to create tan().
    Parameters:
        t (Trace instance)
    Return Trace that constitues tan() elementary function
    '''
#<<<<<<< HEAD
    if isinstance(t, numbers.Number):
        return np.tan(t)
    else:
        return one_parent(t, math.Ops.tan)
# =======
#     try:
#         hasval = t.val
#         return one_parent(t, math.Ops.tan)
#     except AttributeError:
#         if isinstance(t, numbers.Number) or isinstance(t, np.ndarray):
#             return np.tan(t)
#         else:
#             raise ValueError("Input must be numerical or Trace instance")
# >>>>>>> master

def arctan(t):
    '''
    This allows to creat arctan()
    
    Parameters:
        t: (Trace instance)

    Return Trace that constitues arctan() elementary function
    '''

#<<<<<<< HEAD
    if isinstance(t, numbers.Number):
        return np.arctan(t)
    else:
        return one_parent(t, math.Ops.arctan)
# =======
#     try:
#         hasval= t.val
#         return one_parent(t, math.Ops.arctan)
    
#     except AttributeError:
#         if isinstance(t, numbers.Number) or isinstance(t, np.ndarray):
#             return np.arctan(t)
#         else:
#             raise ValueError("Input must be numerical or Trace instance")
# >>>>>>> master


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
#<<<<<<< HEAD
            formula = f'{base}^'
        return one_parent(t, math.Ops.exp, base, formula = formula)
# =======
#             formula = f'{np.round(base,3)} ^ ({t._trace_name})'
#         return one_parent(t, math.Ops.exp, param = base, formula = formula)
#     except AttributeError:
#         if isinstance(t, numbers.Number) or isinstance(t, np.ndarray):
#             return np.power(base, t)
#         else:
#             raise ValueError("Input must be numerical or Trace instance")
# >>>>>>> master


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
# <<<<<<< HEAD

# =======
#     except AttributeError:
#         if isinstance(t, numbers.Number) or isinstance(t, np.ndarray):
#             return np.log(t)/np.log(base)
#         else:
#             raise ValueError("Input must be numerical or Trace instance")
# >>>>>>> master


def sinh(t):
    '''
    This allows to create sinh().
    Parameters:
        t (Trace instance)
        base (int, or float)
    Return Trace that constitues sinh() elementary function
    '''
#<<<<<<< HEAD
    if isinstance(t, numbers.Number):
        return np.sinh(t)
    else:
        return one_parent(t, math.Ops.sinh)

# =======

#     try:
#         hasvalue = t.val
#         return one_parent(t, math.Ops.sinh)
#     except AttributeError:
#         if isinstance(t, numbers.Number) or isinstance(t, np.ndarray):
#             return np.sinh(t)
#         else:
#             raise ValueError("Input must be numerical or Trace instance")
# >>>>>>> master


def cosh(t):
    '''
    This allows to create cosh().
    Parameters:
        t (Trace instance)
    Return Trace that constitues cosh() elementary function
    '''
#<<<<<<< HEAD
    if isinstance(t, numbers.Number):
        return np.cosh(t)
    else:
        return one_parent(t, math.Ops.cosh)
# =======

#     try:
#         hasvalue = t.val
#         return one_parent(t, math.Ops.cosh)
#     except AttributeError:
#         if isinstance(t, numbers.Number) or isinstance(t, np.ndarray):
#             return np.cosh(t)
#         else:
#             raise ValueError("Input must be numerical or Trace instance")

# >>>>>>> master

def tanh(t):
    '''
    This allows to create tanh().
    Parameters:
        t (Trace instance)
    Return Trace that constitues tanh() elementary function
    '''
# <<<<<<< HEAD
    if isinstance(t, numbers.Number):
        return np.tanh(t)
    else:
        return one_parent(t, math.Ops.tanh)
# =======

#     try:
#         hasvalue = t.val
#         return one_parent(t, math.Ops.tanh)
#     except AttributeError:
#         if isinstance(t, numbers.Number) or isinstance(t, np.ndarray):
#             return np.tanh(t)
#         else:
#             raise ValueError("Input must be numerical or Trace instance")
# >>>>>>> master


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
# <<<<<<< HEAD
# =======
#     except AttributeError:
#         if isinstance(t, numbers.Number) or isinstance(t, np.ndarray):
#             return np.sqrt(t)
#         else:
#             raise ValueError("Input must be numerical or Trace instance")

# >>>>>>> master


def sigmoid(t):
    '''
    This allows to create sigmoid().
    Parameters:
        t (Trace instance)
    Return Trace that constitues sigmoig() elementary function
    '''
    if isinstance(t, numbers.Number) or isinstance(t, np.ndarray):
        return 1/(1+np.exp(-t))
    else:
        return one_parent(t, math.Ops.sigm)
# <<<<<<< HEAD
# =======
#     except AttributeError:
#         if isinstance(t, numbers.Number) or isinstance(t, np.ndarray):
#             return 1/(1 + np.exp(-t))
#         else:
#             raise ValueError("Input must be numerical or Trace instance")
# >>>>>>> master






    
