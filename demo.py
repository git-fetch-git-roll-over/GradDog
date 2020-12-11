# :)
import numpy as np
import graddog as gd
from graddog.functions import sin, cos, tan, exp, log
from graddog.tools import plot_derivative, plot_with_tangent_line

seed0 = 1.5
def f0(x):
    return x**3 - 4*x + cos(exp(-sin(tan(log(x)))))

seed1 = [1,2]
def f1(x, y):
    return x*y + exp(x*y)

seed2 = 3.0
def f2(x):
    return [x**2, x**3, x**4]

seed3 = [1.0,2.0,3.0]
def f3(v):
    return v[0] + 3*v[2]**2

seed4 = [1,2,3]
def f4(v):
    return [v[0] + 3*v[2]**2, v[1] - v[0], v[2] + sin(v[1])]

seed5 = np.array([3,2,1])
def f5(x, y, z):
    return [exp(-(sin(x) - cos(y))**2), sin(- log(x) ** 2 + tan(z))]

seed6 = np.array([1,2,3])
def f6(v):
    return v**2 + 2*v + 1

seed7 = seed6
f7 = f0

seed8 = np.arange(6)
def f8(v):
    return v[0]*v[1] + v[2]*v[3] + v[4]*v[5]

seed9 = np.arange(2)
def f9(v):
    return [i*v[0] + i**2 * v[1] for i in range(50)]

seed10 = np.linspace(-np.pi, np.pi, 100)
def f10(v):
    return sin(v)**2 + tan(v)**2

seed11 = np.linspace(1,10, num=9)
def f11(v):
    return v**2 + 1

seeds = [seed0, seed1, seed2, seed3, seed4, seed5, seed6, seed7, seed8, seed9, seed10, seed11]
fs = [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11]

def run_demos():
    for i, f in enumerate(fs):
        print('demo', i)
        f_ = gd.trace(f, seeds[i])
        print(f_)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

def hessian_demo():
    seed1 = [1, 2, 3, 4]
    def f1(x, y, z, w):
        return 2*x*y + w*z/y
    f_, f__ = gd.trace(f1, seed1, return_second_deriv = True, verbose = True)
    print(f_)
    print(f__)

def plotting_demo():
    plot_derivative(sin, 1, 5)
    plot_with_tangent_line(sin, 0, -np.pi, np.pi)

