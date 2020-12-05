import numpy as np
import calc_rules as calc_rules
from trace import Trace
from variable import Variable
from compgraph import CompGraph
from functools import reduce

# TODO: should VectorFunction be in its own file?

'''
Changes I made to Max's code:
    1. change calculate_jacobian into _calculate_jacobian
    2. change map(lambda f : set(f.der.keys()), funcs))) into map(lambda f : set(f._der.keys()), funcs)))
    3. Added the inverse trig functions.

Questions to Max's code:
    1. the _name attribute here again is troubling me
'''

class VectorFunction:
    '''
    This is a class for creating vectorFunction class.
    '''
    def __init__(self, funcs):
        '''
        The constructor for VectorFunction class.
        It calculates the jacobian matrix of the vector functions.

        Parameters:
            funcs: vector functions in a list, e.g. [f1, f2, f3]
        
        Attributes:
            funcs: vector functions in a list, e.g. [f1, f2, f3]
            total_variables: an alphabetic sorted list of variables.
            _name: name of the vectorFunction
            jacobian: the numpy form of the associated jacobian matrix
        '''
        self.funcs = funcs
        # use the map and reduce functions to combine
        # all of the variables from all the functions in funcs
        # into a single sorted list called total_vars
        self.total_variables = sorted(reduce(lambda s, t : s.union(t), list(map(lambda f : set(f._der.keys()), funcs))))
        self._name = 'output'
        self._calculate_jacobian()
    
    def _calculate_jacobian(self):
        '''
        This calculates the jacobian matrix of the vector function in numpy 2-D arrays.
        The columns of the jacobian matrix is the alphabetic orderd variable name.
        The rows of the jacobian matrix is the input functions in funcs list.
        '''
        M = len(self.total_variables)
        N = len(self.funcs)
        self.jacobian = np.array([[self.funcs[n].der_wrt(self.total_variables[m]) for m in range(M)] for n in range(N)])

    @property
    def name(self):
        '''
        Returns non-public attribute _name
        '''
        return self._name

    @name.setter
    def name(self, new_name):
        '''
        This resets the _name of a Trace instance
        '''
        self._name = new_name

    @property
    def der(self):
        '''
        This prints the jacobian matrix of inputted function vectors

        Returns the jacobian matrix
        '''
        print('Jacobian matrix of', self.name)
        return self.jacobian

    @property
    def trace_table(self):
        print('Trace table of a forward pass')
        return repr(CompGraph.instance)

def sin(t: Trace):
    '''
    This allows to create sin().

    Parameters:
        t: (Trace instance)

    Return Trace that constitues sin() elementary function
    '''
    new_formula = f'sin({t._formula})'
    new_val = np.sin(t.val)
    new_der = calc_rules.deriv(t, 'sin')
    return Trace(new_formula, new_val, new_der)

def arcsin(t:Trace):
    '''
    This allows to creat arcsin(). ValueError is caught if the input Trace
    instance has value not in the domain of [-1, 1]
    
    Parameters:
        t: (Trace instance)

    Return Trace that constitues arcsin() elementary function
    '''
    if (t.val <= -1) or (t.val >= 1):
        raise ValueError("The domain of arcsin is (-1, 1)")
    else:
        new_formula = f'arcsin({t._formula})'
        new_val = np.arcsin(t.val)
        new_der = calc_rules.deriv(t, 'arcsin')
        return Trace(new_formula, new_val, new_der)

def cos(t: Trace):
    '''
    This allows to create cos().

    Parameters:
        t: (Trace instance)

    Return Trace that constitues cos() elementary function
    '''
    new_formula = f'cos({t._formula})'
    new_val = np.cos(t.val)
    new_der = calc_rules.deriv(t, 'cos')
    return Trace(new_formula, new_val, new_der)

def arccos(t:Trace):
    '''
    This allows to creat arccos(). ValueError is caught if the input Trace
    instance has value not in the domain of [-1, 1]
    
    Parameters:
        t: (Trace instance)

    Return Trace that constitues arccos() elementary function
    '''
    if (t.val <= -1) or (t.val >= 1):
        raise ValueError("The domain of arcsin is (-1, 1)")
    else:
        new_formula = f'arccos({t._formula})'
        new_val = np.arccos(t.val)
        new_der = calc_rules.deriv(t, 'arccos')
        return Trace(new_formula, new_val, new_der)

def tan(t: Trace):
    '''
    This allows to create tan().

    Parameters:
        t: (Trace instance)

    Return Trace that constitues tan() elementary function
    '''
    new_formula = f'tan({t._trace_name})'
    new_val = np.tan(t.val)
    new_der = calc_rules.deriv(t, 'tan')
    return Trace(new_formula, new_val, new_der)

def arctan(t:Trace):
    '''
    This allows to creat arctan()
    
    Parameters:
        t: (Trace instance)

    Return Trace that constitues arctan() elementary function
    '''
    new_formula = f'arctan({t._formula})'
    new_val = np.arctan(t.val)
    new_der = calc_rules.deriv(t, 'arctan')
    return Trace(new_formula, new_val, new_der)

def exp(t: Trace, base=np.e):
    '''
    This allows to create exp().

    Parameters:
        t: (Trace instance)
        base (int, or float)

    Return Trace that constitues exp() elementary function with input base (default=e)
    '''
    if base==np.e:
        new_formula = f'exp({t._formula})'
    else:
        new_formula = f'{np.round(base,3)}^({t._formula})'
    new_val = np.power(base, t.val)
    new_der = calc_rules.deriv(t, 'exp', other=base)
    return Trace(new_formula, new_val, new_der)

def log(t: Trace, base=np.e):
    '''
    This allows to create log().

    Parameters:
        t: (Trace instance)
        base: (int, or float)

    Return Trace that constitues log() elementary function with input base (default=e)
    '''
    if base==np.e:
        new_formula = f'log({t._trace_name})'
    else:
        new_formula = f'log_{np.round(base,3)}({t._trace_name})'
    new_val = np.log(t.val)/np.log(base)
    new_der = calc_rules.deriv(t, 'log', base)
    return Trace(new_formula, new_val, new_der)

def sinh(t: Trace, base=np.e):
    '''
    This allows to create sinh().

    Parameters:
        t: (Trace instance)
        base: (int, or float)

    Return Trace that constitues sinh() elementary function
    '''
    new_formula = f'sinh({t._trace_name})'
    new_val = (np.exp(t.val) - np.exp(-t.val))/2
    new_der = calc_rules.deriv(t, 'sinh')
    return Trace(new_formula, new_val, new_der)

def cosh(t: Trace):
    '''
    This allows to create cosh().

    Parameters:
        t: (Trace instance)

    Return Trace that constitues cosh() elementary function
    '''
    new_formula = f'cosh({t._trace_name})'
    new_val = (np.exp(t.val) + np.exp(-t.val))/2
    new_der = calc_rules.deriv(t, 'cosh')
    return Trace(new_formula, new_val, new_der)

def tanh(t: Trace):
    '''
    This allows to create tanh().

    Parameters:
        t: (Trace instance)

    Return Trace that constitues tanh() elementary function
    '''
    new_formula = f'tanh({t._trace_name})'
    new_val = (np.exp(t.val) + np.exp(-t.val))/(np.exp(t.val) - np.exp(-t.val))
    new_der = calc_rules.deriv(t, 'tanh')
    return Trace(new_formula, new_val, new_der)

def sqrt(t: Trace):
    '''
    This allows to create sqrt().

    Parameters:
        t: (Trace instance)

    Return Trace that constitues sqrt() elementary function
    '''
    new_formula = f'sqrt({t._formula})'
    new_val = t.val**0.5
    new_der = calc_rules.deriv(t, 'sqrt')
    return Trace(new_formula, new_val, new_der)

def sigmoid(t: Trace):
    '''
    This allows to create sigmoid().

    Parameters:
        t: (Trace instance)

    Return Trace that constitues sigmoig() elementary function
    '''
    new_formula = f'sigm({t._trace_name})'
    new_val = 1/(1 + np.exp(-t.val))
    new_der = calc_rules.deriv(t, 'sigm')
    return Trace(new_formula, new_val, new_der)


    


x = Variable('x', 1)
# = Variable('y', 1)
f1 = sin(x)
print(f1)
#y = Variable('y', 1)
#f1 = sin(x)
#f2 = cos(y)
#f3 = exp(x, base=2.4444)
# print(x)
# print(CompGraph.instance)
#vectfun = VectorFunction([f1, f2, f3])
#print(f3)



