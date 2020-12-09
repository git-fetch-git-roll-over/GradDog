# :)
import numpy as np
import matplotlib.pyplot as plt
import time
import graddog as gd
from graddog.functions import sin, cos, tan, exp, log
from graddog.tools import plot_with_tangent_line
from graddog.functions import sin, cos, tan, exp, log, sigmoid
from graddog.tools import plot_derivative

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

seed5 = np.ones(3)
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
    return log(v, base = 2)   

seeds = [seed0, seed1, seed2, seed3, seed4, seed5, seed6, seed7, seed8, seed9, seed10, seed11]
fs = [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11]

def run_demos():
    for i, f in enumerate(fs):
        print('demo', i)
        f_ = gd.trace(f, seeds[i], verbose = True)
        print(f_)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


def time_demo(f, seed):
    fig, ax = plt.subplots()
    n_boot = 10
    times = np.zeros(shape= (n_boot,2))
    for boot in range(n_boot):
        t0 = time.time()
        gd.trace(f, seed, mode = 'forward')
        t_forward = time.time() - t0

        t0 = time.time()
        gd.trace(f, seed, mode = 'reverse')
        t_reverse = time.time() - t0
        times[boot,:] = [t_forward, t_reverse]
    plt.hist(times[:,0], bins = 10, alpha = 0.3, color = 'blue', label = 'forward', density = True)
    plt.hist(times[:,1], bins = 10, alpha = 0.3, color = 'red', label = 'reverse', density = True)
    plt.legend()
    plt.title('f : R --> R50')
    plt.xlabel('time')
    plt.ylabel('frequency')
    plt.show()



def test_hessian():
    seed1 = [1.5]
    def f1(x):
        return 2*x**2

    seed2 = [1,2]
    def f2(x, y):
        return x*y + exp(x*y)

    new_fs = [f1, f2]
    new_seeds = [seed1, seed2]
    for i, f in enumerate(new_fs):
        f_, f__ = gd.trace(f, new_seeds[i], return_second_deriv = True, verbose = True)
        print(f_)
        print(f__)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#test_hessian()
#run_demos()


#plot_derivative(lambda x : sin(log(x)), .207, 2)


def cubic(x):
    return 2*x**3
plot_with_tangent_line(cubic, 3, -10, 10, n_pts=1000, figsize=(6,6), xlabel='x', ylabel='y', plotTitle='Function with tangent line')

