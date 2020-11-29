import numpy as np
from trace import Trace


def sin(t : Trace):
    '''
    This allows to create sin().

    Parameters:
        t (Trace instance)

    Return Trace that constitues sin() elementary function
    '''
    new_name = f'sin({t._name})'
    new_val = np.sin(t.val)
    new_der = {x : np.cos(t.val)*t._der[x] for x in t._der}
    return Trace(new_name, new_val, new_der)

def cos(t : Trace):
    '''
    This allows to create cos().

    Parameters:
        t (Trace instance)

    Return Trace that constitues cos() elementary function
    '''
    new_name = f'cos({t._name})'
    new_val = np.cos(t.val)
    new_der = {x : -np.sin(t.val)*t._der[x] for x in t._der}
    return Trace(new_name, new_val, new_der)

def tan(t : Trace):
    '''
    This allows to create tan().

    Parameters:
        t (Trace instance)

    Return Trace that constitues tan() elementary function
    '''
    new_name = f'tan({t._name})'
    new_val = np.tan(t.val)
    new_der = {x : t._der[x]/(np.cos(t.val)**2) for x in t._der}
    return Trace(new_name, new_val, new_der)

def exp(t : Trace, base=np.e):
    '''
    This allows to create exp().

    Parameters:
        t (Trace instance)
        base (int, or float)

    Return Trace that constitues exp() elementary function with input base (default=e)
    '''
    
    if base==np.e:
        new_name = f'exp({t._name})'
        new_val = np.exp(t.val)
        new_der = {x : np.exp(t.val)*t._der[x] for x in t._der}
    else:
        new_name = f'{np.round(base,3)} ^ ({t._name})'
        new_val = np.power(base, t.val)
        new_der = {x : np.power(base, t.val)*np.log(base)*t._der[x] for x in t._der}
    return Trace(new_name, new_val, new_der)

def log(t : Trace, base=np.e):
    '''
    This allows to create log().

    Parameters:
        t (Trace instance)
        base (int, or float)

    Return Trace that constitues log() elementary function with input base (default=e)
    '''

    if base==np.e:
        new_name = f'log({t._name})'
        new_val = np.log(t.val)
        new_der = {x : t._der[x]/t.val for x in t._der}
    else:
        new_name = f'log_{np.round(base,3)}({t._name})'
        new_val = np.log(t.val)/np.log(base)
        new_der = {x : t._der[x]/(t.val*np.log(base)) for x in t._der}
    return Trace(new_name, new_val, new_der)

def sinh(t : Trace, base=np.e):
    '''
    This allows to create sinh().

    Parameters:
        t (Trace instance)
        base (int, or float)

    Return Trace that constitues sinh() elementary function
    '''
    new_name = f'sinh({t._name})'
    new_val = (np.exp(t.val) - np.exp(-t.val))/2

    #derivative of sinh(x) is x'cosh(x)
    new_der = {x : t._der[x]*(np.exp(t.val) + np.exp(-t.val))/2 for x in t._der}

    return Trace(new_name, new_val, new_der)

def cosh(t : Trace):
    '''
    This allows to create cosh().

    Parameters:
        t (Trace instance)

    Return Trace that constitues cosh() elementary function
    '''
    new_name = f'cosh({t._name})'
    new_val = (np.exp(t.val) + np.exp(-t.val))/2

    #derivative of cosh(x) is x'sinh(x)
    new_der = {x : t._der[x]*(np.exp(t.val) - np.exp(-t.val))/2 for x in t._der}
    return Trace(new_name, new_val, new_der)

def tanh(t : Trace):
    '''
    This allows to create tanh().

    Parameters:
        t (Trace instance)

    Return Trace that constitues tanh() elementary function
    '''
    new_name = f'tanh({t._name})'
    new_val = (np.exp(t.val) + np.exp(-t.val))/(np.exp(t.val) - np.exp(-t.val))

    #derivative of tanh(x) is x'(sech(x)**2)
    new_der = {x : t._der[x]*4/((np.exp(t.val) + np.exp(-t.val))**2) for x in t._der}
    return Trace(new_name, new_val, new_der)

def sqrt(t : Trace):
    '''
    This allows to create sqrt().

    Parameters:
        t (Trace instance)

    Return Trace that constitues sqrt() elementary function
    '''
    new_name = f'sqrt({t._name})'
    new_val = t.val**0.5

    #derivative of sqrt(x) is x'/(2sqrt(x))
    new_der = {x : t._der[x]/(2*x**0.5) for x in t._der}
    return Trace(new_name, new_val, new_der)

def sigmoid(t : Trace):
    '''
    This allows to create sigmoid().

    Parameters:
        t (Trace instance)

    Return Trace that constitues sigmoig() elementary function
    '''
    new_name = f'logit({t._name})'
    new_val = 1/(1 + np.exp(-t.val))
    new_der = {x : t._der[x]*np.exp(-t.val)/((1 + np.exp(-t.val))**2) for x in t._der}
    return Trace(new_name, new_val, new_der)


    
