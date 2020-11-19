import pytest
import numpy as np
from forward_mode.variable import Variable
from forward_mode.function import Function

def test_sin():
    x1 = Variable('x', 3)
    g = Function.sin(x1)
    assert g.der == Function.cos(x1).val


def test_tan():
    x1 = Variable('x', 2*np.pi)
    t = Function.tan(x1)
    assert t.val == pytest.approx(0, abs=1e-6)
    assert t.der == 1/np.cos(x1.val)**2

def test_cos():
    x1 = Variable('x', 2*np.pi)
    c = Function.cos(x1)
    assert c.val == 1
    assert c.der == pytest.approx(0, abs=1e-6)
    
def test_string_input():
    with pytest.raises(TypeError):
        x_str = Variable('s', 'CS107')
        
def test_log_base2():
    x = Variable('x', 32)
    base = 2
    f = Function.log(x, base=base)
    assert f.val == 5
    assert f.der == 1/(x.val * np.log(base))
    
def test_exp_base2():
    x = Variable('x', 5)
    base = 2
    f = Function.exp(x, base=base)
    assert f.val == 32
    assert f.der == (base**x.val)*np.log(base)

def test_log():
    x = Variable('x', 4)
    f = Function.log(x)
    assert f.val == np.log(4)
    assert f.der == 1/4

def test_exp():
    x = Variable('x', 67)
    f = Function.exp(x)
    assert f.val == np.exp(67)
    assert f.der == f.val

def test_composition():
    value = np.pi/6
    x = Variable('x', value)
    c = Function.cos(x)
    t = Function.tan(x)
    e = Function.exp(x)
    f = c * t + e
    assert isinstance(f, Variable)
    assert f.der == -1*np.sin(value)*np.tan(value) + 1/np.cos(value) + np.exp(value)
