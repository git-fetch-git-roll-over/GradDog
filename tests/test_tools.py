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

def plot_with_tangent_line():
    pass

def plot_with_normal_line():
    pass
