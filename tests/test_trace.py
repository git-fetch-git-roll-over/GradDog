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
    a = x.__radd__(z)
    b = x.__rmul__(z)
    c = x.__rtruediv__(z)
    d = x.__truediv__(y)
    e = x.__neg__()
    f = x.__pow__y
    g = x.__rpow__(z)
    h = x.__sub__(y)
    i = x.__rsub__(z)
    assert x.__eq__(y)
    assert not x.__eq__(z) 
    assert x.__ne__(z)
    assert not x.__ne__(y)
    assert a.val == 6
    assert b.val == 9
    assert c.val == 1
    assert d.val == 1
    assert e.val == -3
    assert f.val == 27
    assert g.val == 27
    assert h.val == 0
    assert i.val == 0

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

def test_RMtoR():
    def f(v):
        return v[0] + exp(v[1]) + 6*v[2]**2
    x = gd.trace(f, [1,2,4])
    assert x[0][0] == 1.0
    assert x[0][1] == pytest.approx(np.exp(2))
    assert x[0][2] == 48.0

def test_RMtoRN():
    def f(v):
        return [v[0]+v[1], v[1]-v[2], cos(v[2]), exp(v[3])*sin(v[2])]
    x = gd.trace(f, [1,2,3,4])
    assert x[0][0] == 1.0
    assert x[0][1] == 1.0
    assert x[1][1] == 1.0
    assert x[1][2] == -1.0
    assert x[2][2] == pytest.approx(-0.14112001)
    assert x[3][2] == pytest.approx(-54.05175886)
    assert x[3][3] == pytest.approx(7.70489137)

