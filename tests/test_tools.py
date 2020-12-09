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
