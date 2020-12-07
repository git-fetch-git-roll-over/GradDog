import numpy as np
import graddog.calc_rules as calc_rules
Ops = calc_rules.Ops
from graddog.trace import Trace
from graddog.compgraph import CompGraph
from functools import reduce

# TODO: should VectorFunction be in its own file?

# TODO: need to ensure input to log is positive!

class VectorFunction:
    def __init__(self, funcs):
        self.funcs = funcs

        # use the map and reduce functions to combine
        # all of the variables from all the functions in funcs
        # into a single sorted list called total_vars
        # self.total_variables = sorted(reduce(lambda s, t : s.union(t), list(map(lambda f : set(f._der.keys()), funcs))))
        # self._name = 'output'
        # self.calculate_jacobian()
    
    # def calculate_jacobian(self):
    #     M = len(self.total_variables)
    #     N = len(self.funcs)
    #     self.jacobian = np.array([[self.funcs[n].der_wrt(self.total_variables[m]) for m in range(M)] for n in range(N)])

    # @property
    # def name(self):
    #     '''
    #     Returns non-public attribute _name
    #     '''
    #     return self._name

    # @name.setter
    # def name(self, name):
    #     '''
    #     This resets the _name of a Trace instance
    #     '''
    #     self._name = name

    # @property
    # def der(self):
    #     print('Jacobian matrix of', self.name)
    #     return self.jacobian

    # @property
    # def trace_table(self):
    #     print('Trace table of a forward pass')
    #     return repr(CompGraph.instance)

def sin(t : Trace):
    '''
    This allows to create sin().

    Parameters:
        t (Trace instance)

    Return Trace that constitues sin() elementary function
    '''
    op = Ops.sin
    formula = f'{op}({t._trace_name})'
    param = None
    val = Ops.one_parent_rules[op](t, param)#np.sin(t.val)
    der = Ops.one_parent_deriv_rules[op](t,param)#calc_rules.deriv(t, op)
    parents = [t]
    return Trace(formula, val, der, parents, op)

def cos(t : Trace):
    '''
    This allows to create cos().

    Parameters:
        t (Trace instance)

    Return Trace that constitues cos() elementary function
    '''
    op = Ops.cos
    formula = f'{op}({t._trace_name})'
    val = np.cos(t.val)
    der = calc_rules.deriv(t, op)
    parents = [t]
    return Trace(formula, val, der, parents, op)

def tan(t : Trace):
    '''
    This allows to create tan().

    Parameters:
        t (Trace instance)

    Return Trace that constitues tan() elementary function
    '''
    op = Ops.tan
    formula = f'{op}({t._trace_name})'
    val = np.tan(t.val)
    der = calc_rules.deriv(t, op)
    parents = [t]
    return Trace(formula, val, der, parents, op)

def exp(t : Trace, base=np.e):
    '''
    This allows to create exp().

    Parameters:
        t (Trace instance)
        base (int, or float)

    Return Trace that constitues exp() elementary function with input base (default=e)
    '''
    op = Ops.exp
    if base==np.e:
        formula = f'{op}({t._trace_name})'
    else:
        formula = f'{np.round(base,3)} ^ ({t._trace_name})'
    val = np.power(base, t.val)
    der = calc_rules.deriv(t, op, base)
    parents = [t]
    param = base
    return Trace(formula, val, der, parents, op, param)

def log(t : Trace, base=np.e):
    '''
    This allows to create log().

    Parameters:
        t (Trace instance)
        base (int, or float)

    Return Trace that constitues log() elementary function with input base (default=e)
    '''
    op = Ops.log
    if base==np.e:
        formula = f'{op}({t._trace_name})'
    else:
        formula = f'log_{np.round(base,3)}({t._trace_name})'
    val = np.log(t.val)/np.log(base)
    der = calc_rules.deriv(t, op, base)
    parents = [t]
    param = base
    return Trace(formula, val, der, parents, op, param)

def sinh(t : Trace):
    '''
    This allows to create sinh().

    Parameters:
        t (Trace instance)
        base (int, or float)

    Return Trace that constitues sinh() elementary function
    '''
    op = Ops.sinh
    formula = f'{op}({t._trace_name})'
    val = (np.exp(t.val) - np.exp(-t.val))/2
    der = calc_rules.deriv(t, op)
    parents = [t]
    return Trace(formula, val, der, parents, op)

def cosh(t : Trace):
    '''
    This allows to create cosh().

    Parameters:
        t (Trace instance)

    Return Trace that constitues cosh() elementary function
    '''
    op = Ops.cosh
    formula = f'{op}({t._trace_name})'
    val = (np.exp(t.val) + np.exp(-t.val))/2
    der = calc_rules.deriv(t, op)
    parents = [t]
    return Trace(formula, val, der, parents, op)

def tanh(t : Trace):
    '''
    This allows to create tanh().

    Parameters:
        t (Trace instance)

    Return Trace that constitues tanh() elementary function
    '''
    op = Ops.tanh
    formula = f'{op}({t._trace_name})'
    val = (np.exp(t.val) + np.exp(-t.val))/(np.exp(t.val) - np.exp(-t.val))
    der = calc_rules.deriv(t, op)
    parents = [t]
    return Trace(formula, val, der, parents, op)

def sqrt(t : Trace):
    '''
    This allows to create sqrt().

    Parameters:
        t (Trace instance)

    Return Trace that constitues sqrt() elementary function
    '''
    op = Ops.sqrt
    formula = f'{op}({t._trace_name})'
    val = t.val**0.5
    der = calc_rules.deriv(t, op)
    parents = [t]
    return Trace(formula, val, der, parents, op)

def sigmoid(t : Trace):
    '''
    This allows to create sigmoid().

    Parameters:
        t (Trace instance)

    Return Trace that constitues sigmoig() elementary function
    '''
    op = Ops.sigm
    formula = f'{op}({t._trace_name})'
    val = 1/(1 + np.exp(-t.val))
    der = calc_rules.deriv(t, op)
    parents = [t]
    return Trace(formula, val, der, parents, op)


    
