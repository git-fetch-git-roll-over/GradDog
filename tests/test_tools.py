import pytest 
import numpy as np
from graddog.trace import Trace, Variable, one_parent, two_parents
from graddog.functions import sin, arcsin, sinh, cos, arccos, cosh, tan, arctan, tanh, exp, log, sigmoid, sqrt
import graddog.tools as tools

def test_plot_derivative_sq():
    def f(x):
        return x**2 + 1
    xs, ys = tools.plot_derivative(f, 0, 5, n_pts=5)
    txs = np.linspace(0, 5, 5)
    tys = [2*x for x in txs]
    assert (xs == txs).all() 
    assert ys == tys

def test_plot_derivative_cube():
    def f(x):
        return x**3 + 1
    xs, ys = tools.plot_derivative(f, 0, 5, n_pts=5)
    txs = np.linspace(0, 5, 5)
    tys = [3*x**2 for x in txs]
    assert (xs == txs).all() 
    assert ys == tys

def test_plot_derivative_trig():
    def f(x):
        return sin(x)
    xs, ys = tools.plot_derivative(f, 0, 5, n_pts=5)
    txs = np.linspace(0, 5, 5)
    tys = [cos(x) for x in txs]
    assert (xs == txs).all() 
    assert ys == tys

def test_find_extrema():
    def sq(x):
        x**2 + 1
    def quadratic(x):
        a=4
        xoffset = 3
        yoffset = 0
        return a*(x-xoffset)**2 + yoffset    
    x = tools.find_extrema_firstorder(quadratic, -10, 10)
    assert x[0] == pytest.approx(2.92929292929292)
    assert x[1] == pytest.approx(3.13131313131313)
    # x == (2.929292929292929, 3.1313131313131315)
    x1 = tools.find_extrema_firstorder(sq, 0, 5, n_pts=5)
    assert x1[0] == pytest.approx(0)
    x2 = tools.find_extrema_firstorder(sq, 4, 5, n_pts=5)
    assert x2 == None
    

def test_find_increasing():
    def sq(x):
        x**2 + 3
    x1, y1 = tools.find_increasing(sq, 0, 10, n_pts=5)
    assert ([x1[0], y1[0]] == [2.5 , 5])
    x2 = tools.find_increasing(sq, -10, 0, n_pts=5)
    assert x2 == None

def test_find_decreasing():
    def sq(x):
        x**2 + 3
    x1 = tools.find_decreasing(sq, 0, 10, n_pts=5)
    assert x1 == None
    x2, y2 = tools.find_decreasing(sq, -10, 0, n_pts=5)
    assert ([x2[0], y2[0]] == [-10, -20])

def test_plot_with_tangent_line():
    def quadratic(x):
        a=4
        xoffset = 3
        yoffset = 0
        return a*(x-xoffset)**2 + yoffset    
    x, val = tools.plot_with_tangent_line(quadratic, 5, 0.5, 10)
    assert x[0] == 0.5
    assert x[-1] == 10
    for i in range(len(x)):
        assert val[i] == pytest.approx(quadratic(x[i]))