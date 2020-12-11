import pytest 
import numpy as np
import graddog as gd
import graddog.math as ops
from graddog.trace import Trace
from graddog.functions import sin, cos, tan, exp, log, sigmoid

def test_basic():
    def poly(x):
        return 2*x**4
    def trig(x):
        return sin(x)
    x1_, x1__ = gd.trace(poly, 2, return_second_deriv = True)
    assert x1_[0][0] == pytest.approx(64)
    assert x1__[0][0] == pytest.approx(96)
    x2_, x2__ = gd.trace(trig, np.pi, return_second_deriv = True)
    assert x2_[0][0] == pytest.approx(-1)
    assert x2__[0][0] == pytest.approx(0)
    non_scalar = [0, np.pi/2,np.pi,3*np.pi/2, 2*np.pi]
    with pytest.raises(ValueError):
        x1_, x1__ = gd.trace(trig, non_scalar, return_second_deriv = True)
    with pytest.raises(ValueError):
        x1_, x1__ = gd.trace(trig, non_scalar, return_second_deriv = True, mode='forward')
    x3_, x3__ = gd.trace(poly, 2, return_second_deriv = True, verbose=True)
    assert x3_[0][0] == pytest.approx(64)
    assert x3__[0][0] == pytest.approx(96)
    

def test_multiple_inputs():
    seed1 = [1, 2, 3, 4]
    def f1(x, y, z, w):
        return 2*x*y + w*z/y
    x1_, x1__ = gd.trace(f1, seed1, return_second_deriv = True)
    assert x1_[0][0] == pytest.approx(4)
    assert x1_[0][1] == pytest.approx(-1)
    assert x1__[1][1] == pytest.approx(3)
    assert x1__[3][2] == pytest.approx(0.5)
    seed2 = [np.pi] * 4
    def trig(w, x, y, z):
            return sin(x) + cos(z)*y + sin(w)
    x2_, x2__ = gd.trace(trig, seed2, return_second_deriv = True)
    assert x2_[0][1] == pytest.approx(-1)
    assert x2_[0][3] == pytest.approx(0)
    assert x2__[2][2] == pytest.approx(0)
    assert x2__[3][3] == pytest.approx(np.pi)

def test_math_errors():
    x = Trace('x', 3, {'x' : 1.0}, [])    
    y = Trace('y', 3, {'y' : 1.0}, [])
    z = 3
    with pytest.raises(ValueError):
        a = ops.new_double_deriv_one_parent(x, 'relu')
    with pytest.raises(ValueError):
        b = ops.new_double_deriv_two_parents(x, 'relu', y)
