import pytest 
import numpy as np
import graddog as gd
from graddog.trace import Trace, Variable
from graddog.functions import sin, arcsin, sinh, cos, arccos, cosh, tan, arctan, tanh, exp, log, sigmoid, sqrt

def test_sin():
    value = 0.5
    x = Variable('x', value)
    g = tan(x)
    h = arctan(x)
    i = tanh(x)
    assert g.val == pytest.approx(np.tan(value))
    assert h.val == pytest.approx(np.arctan(value))
    assert i.val == pytest.approx(np.tanh(value))

def test_tan():
    value = 0.5
    x = Variable('x', value)
    g = cos(x)
    h = arccos(x)
    i = cosh(x)
    assert g.val == pytest.approx(np.cos(value))
    assert h.val == pytest.approx(np.arccos(value))
    assert i.val == pytest.approx(np.cosh(value))

def test_cos():
    value = 0.5
    x = Variable('x', value)
    g = sin(x)
    h = arcsin(x)
    i = sinh(x)
    assert g.val == pytest.approx(np.sin(value))
    assert h.val == pytest.approx(np.arcsin(value))
    assert i.val == pytest.approx(np.sinh(value))

def test_sigmoid():
    value = 0.5
    x = Variable('x', value)
    g = sigmoid(x)
    assert g.val == pytest.approx(1/(1 + np.exp(-value)))

def test_sqrt():
    value = 9
    x = Variable('x', value)
    g = sqrt(x)
    assert g.val == pytest.approx(np.sqrt(value))
        
def test_log_base2():
    x = Variable('x', 32)
    base = 2
    f = log(x, base=base)
    assert f._val == pytest.approx(5)
    assert f._der['v1'] == 1/(x._val * np.log(base))
    
def test_exp_base2():
    x = Variable('x', 5)
    base = 2
    f = exp(x, base=base)
    assert f._val == pytest.approx(32)
    assert f._der['v1'] == (base**x._val)*np.log(base)

def test_log():
    x = Variable('x', 4)
    f = log(x)
    print(f._der)
    assert f._val == np.log(4)
    assert f._der['v1'] == pytest.approx(0.25)


def test_exp():
    x = Variable('x', 67)
    f = exp(x)
    assert f._val == pytest.approx(np.exp(67), rel=1e-5)
    assert f._der['v1'] == f._val

def test_composition_val():
    value = np.pi/6
    x = Variable('x', value)
    c = cos(x)
    s = sin(x)
    t = tan(x)
    e = exp(x)
    f = c * t + e
    g = c + s
    assert isinstance(f, Trace)
    assert f._val == np.cos(value)*np.tan(value) + np.exp(value)

def test_basic_der():
    # Decorator function maker that can be used to create function variables
    def fm(f):
        def fun(x):
            return f(x) 
        return fun  
      
    value = 0.5
    assert gd.trace(fm(sin), value) == np.cos(value)
    assert gd.trace(fm(cos), value) == -np.sin(value)
    assert gd.trace(fm(tan), value) == 1/(np.cos(value)*np.cos(value))

def test_composition_der():
    def f(x):
        return cos(x)*tan(x) + exp(x)
    value = 0.5
    der = gd.trace(f, value)
    assert der[0] == -1*np.sin(value)*np.tan(value) + 1/np.cos(value) + np.exp(value)

