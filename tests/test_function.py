import pytest
import numpy as np
from graddog.trace import Trace, get_x, get_xy, get_xyz, function_to_Trace, get_vars
from graddog.functions import sin, cos, tan, exp, log
#from graddog.variable import get_x, get_xy, get_xyz, function_to_Trace, get_vars



def test_sin():
    x1 = get_x([3])
    g = sin(x1)
    assert g._der['x'] == pytest.approx(np.cos(3))


def test_tan():
    x1 = get_x([2*np.pi])
    t = tan(x1)
    assert t._val == pytest.approx(0, abs=1e-6)
    assert t._der['x'] == 1/np.cos(x1._val)**2

def test_cos():
    x1 = get_x([2*np.pi])
    c = cos(x1)
    assert c._val == 1
    assert c._der['x'] == pytest.approx(0, abs=1e-6)
    
def test_string_input():
    with pytest.raises(TypeError):
        x_str = Trace('s', 'CS107', 0)
        
def test_log_base2():
    x = get_x([32])
    base = 2
    f = log(x, base=base)
    assert f._val == pytest.approx(5)
    assert f._der['x'] == 1/(x._val * np.log(base))
    
def test_exp_base2():
    x = get_x([5])
    base = 2
    f = exp(x, base=base)
    assert f._val == pytest.approx(32)
    assert f._der['x'] == (base**x._val)*np.log(base)

def test_log():
    x = get_x([4])
    f = log(x)
    print(f._der)
    assert f._val == np.log(4)
    assert f._der['x'] == pytest.approx(0.25)


def test_exp():
    x = get_x([67])
    f = exp(x)
    assert f._val == pytest.approx(np.exp(67), rel=1e-5)
    assert f._der['x'] == f._val

def test_composition():
    value = np.pi/6
    x = get_x([value])
    c = cos(x)
    t = tan(x)
    e = exp(x)
    f = c * t + e
    assert isinstance(f, Trace)
    assert f._der['x'] == -1*np.sin(value)*np.tan(value) + 1/np.cos(value) + np.exp(value)

