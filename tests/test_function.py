import pytest
import numpy as np
from graddog.variable import Variable
from graddog.trace import Trace
from graddog.functions import sin, cos, tan, exp, log



def test_sin():
    x1 = Variable('x', 3)
    g = sin(x1)
    assert g._der['x'] == pytest.approx(np.cos(3))


def test_tan():
    x1 = Variable('x', 2*np.pi)
    t = tan(x1)
    assert t._val == pytest.approx(0, abs=1e-6)
    assert t._der['x'] == 1/np.cos(x1._val)**2

def test_cos():
    x1 = Variable('x', 2*np.pi)
    c = cos(x1)
    assert c._val == 1
    assert c._der['x'] == pytest.approx(0, abs=1e-6)
    
def test_string_input():
    with pytest.raises(TypeError):
        x_str = Variable('s', 'CS107')
        
def test_log_base2():
    x = Variable('x', 32)
    base = 2
    f = log(x, base=base)
    assert f._val == 5
    assert f._der['x'] == 1/(x._val * np.log(base))
    
def test_exp_base2():
    x = Variable('x', 5)
    base = 2
    f = exp(x, base=base)
    assert f._val == 32
    assert f._der['x'] == (base**x._val)*np.log(base)

def test_log():
    x = Variable('x', 4)
    f = log(x)
    print(f._der)
    assert f._val == np.log(4)
    assert f._der['x'] == pytest.approx(0.25)


def test_exp():
    x = Variable('x', 67)
    f = exp(x)
    assert f._val == np.exp(67)
    assert f._der['x'] == f._val

def test_composition():
    value = np.pi/6
    x = Variable('x', value)
    c = cos(x)
    t = tan(x)
    e = exp(x)
    f = c * t + e
    assert isinstance(f, Trace)
    assert f._der['x'] == -1*np.sin(value)*np.tan(value) + 1/np.cos(value) + np.exp(value)

