import pytest 
import numpy as np
import graddog as gd
from graddog.trace import Trace, Variable, one_parent, two_parents
from graddog.functions import sin, arcsin, sinh, cos, arccos, cosh, tan, arctan, tanh, exp, log, sigmoid, sqrt

def test_string_input_var():
    with pytest.raises(TypeError):
        x = Variable('x', 'test')
    with pytest.raises(TypeError):   
        x = Variable('x', 3)
        x.val = 2 
        x.val ='test'

def basic_ops():
    x = Trace('x', 3, {'x' : 1.0}, [])    
    y = Trace('x', 3, {'x' : 1.0}, [])
    z = 3
    a = z + x
    b = z * x
    c = z / x
    d = x / y
    e = -x
    f = x**y
    g = z**x
    assert x == y
    assert (x == z) == False 
    assert (x != z) == True
    assert x != z
    assert a.val == 6
    assert b.val == 9
    assert c.val == 1
    assert d.val == 1
    assert e.val == -3
    assert f.val == 27
    assert g.val == 27

def test_basic_reverse():
    # Decorator function maker that can be used to create function variables
    def fm(f):
        def fun(x):
            return f(x) 
        return fun  
      
    value = 0.5
    assert gd.trace(fm(sin), value, mode='reverse') == np.cos(value)
    assert gd.trace(fm(cos), value, mode='reverse') == -np.sin(value)
    assert gd.trace(fm(tan), value, mode='reverse') == 1/(np.cos(value)*np.cos(value))

def test_composite_reverse():
    def f(x):
        return cos(x)*tan(x) + exp(x)
    value = 0.5
    der = gd.trace(f, value, mode='reverse')
    assert der[0] == -1*np.sin(value)*np.tan(value) + 1/np.cos(value) + np.exp(value)

def test_one_parent():
    x = Trace('x', 3, {'x' : 1.0}, [])    
    y = Trace('x', 3, {'x' : 1.0}, [])
    a = one_parent(x, 'cos')
    b = one_parent([x, y], 'cos')
    assert a.val == np.cos(3)
    assert b[1].val == np.cos(3)

    with pytest.raises(ValueError):
        c = one_parent(x, 'cos', param='test')

    with pytest.raises(ValueError):
        x = 'test'
        d = one_parent(x, 'cos')

def test_two_parent():
    x = Trace('x', 3, {'x' : 1.0}, [])    
    y = Trace('x', 3, {'x' : 1.0}, [])
    z = 3
    a = two_parents(x, '^', z)
    b = two_parents(x, '^', [x, y])
    assert a.val == 27
    assert b[1].val == 27

    with pytest.raises(ValueError):
        z = two_parents(x, 'cos', 'test')    

