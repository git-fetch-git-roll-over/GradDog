# Tools to help the user engage with our package
# :)
import numpy as np
import matplotlib.pyplot as plt
import graddog as gd
from graddog.functions import sin, cos, tan, exp, log

# def fun(x):
#     return 3*x**4 - 6*x**3
# xs, ys = plot_derivative(fun, -2, 4)

# xs = linearly spaced array b/n -2 and 4
# ys = derivative values evaluated @ each x in xs

# def fun2(x):
#     return x**6 - 10*x**5 + x**4 -7*x**3

# xs, ys = plot_derivative(fun2, -5, 5)


# def quadratic(x):
#     a=4
#     xoffset = 3
#     yoffset = 0
#     return a*(x-xoffset)**2 + yoffset


    
def plot_derivative(function, xmin, xmax, n_pts=100, figsize=(6,6), xlabel='x', ylabel='y', plotTitle='Derivative'):
    '''
    Plot the derivative of a function between xmin and xmax, using n_pts linearly spaced points to evaluate it.
    
    Inputs:
    function[function]: the function of which you'd like to plot the derivative (must only take single input)
    xmin: lower bound on which to calculate derivative
    xmax: upper bound on which to calculate derivative
    n_pts [int](Default: 100): how many points to use when evaluating derivative (more = better resolution but slower)
    figsize [tuple](Default: (6,6)): figsize in inches (see matplotlib documentation)
    xlabel [string](Default: 'x'): Label for x axis of plot
    ylabel [string](Default: 'y'): Label for y axis of plot
    plotTitle [string](Default: 'Derivative'): Label for title of plot
    
    Outputs: 
    xs: the array of linearly spaced x values between xmin and xmax
    ys: the derivative evaluated at the values in xs
    '''
    xs = np.linspace(xmin, xmax, num=n_pts)
    diag_mtx = gd.trace(function, xs)    
    # Assemble values by pulling our diagonal matrix entries
    ys = []
    for i in range(n_pts):
        ys.append(diag_mtx[i][i]) # append the derivative
        
    plt.figure(figsize=figsize)
    plt.plot(xs, ys)
    plt.title(plotTitle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xmin, xmax)
    plt.show()
    return xs, ys

# xs_EX = find_extrema_firstorder(fun, -2, 4)
# xs_EX == 0.0
# x = find_extrema_firstorder(quadratic, -10, 10)
# x == (2.929292929292929, 3.1313131313131315)

def find_extrema_firstorder(function, xmin, xmax, n_pts=100, tolerance = 1e-10):
    '''
    Locate the point where the derivative is closest to zero on the given interval.
    
    Inputs:
    function[function]: the function of which you'd like to plot the derivative. (one scalar input)
    xmin: lower bound on which to calculate derivative
    xmax: upper bound on which to calculate derivative
    n_pts [int](Default: 100): how many points to use when evaluating derivative (more = better resolution but slower)
    tolerance [float](Default: 1e-10): how close to zero should a value be before it's considered an extrema?
    
    Outputs:
    If it can locate your extrema exactly, it will return only the x value(s) of the extrema.
    If not it will return:
    xs: a tuple containing the two x values between which the extrema is located
    '''
    xs = np.linspace(xmin, xmax, num=n_pts)
    diag_mtx = gd.trace(function, xs)    
    # Assemble values by pulling our diagonal matrix entries
    ys = []
    for i in range(n_pts):
        ys.append(diag_mtx[i][i]) # append the derivative
    ys = np.array(ys)    

    zeroidx = np.where(ys < tolerance)[0].astype(int)
    if len(zeroidx) != 0:
        return xs[zeroidx]
    else:
        decreasingIDX = np.where(ys < tolerance)[0].astype(int)
        increasingIDX = np.where(ys > tolerance)[0].astype(int)

        if len(decreasingIDX) == 0 or len(increasingIDX) == 0:
            print(f'No extrema located in the interval {xmin} to {xmax}.')
            return None

        if decreasingIDX[-1] == increasingIDX[0]-1: # Function goes from decreasing -> increasing
            print(f'Extrema located between x={xs[decreasingIDX[-1]]} and {xs[increasingIDX[0]]}')
            return (xs[decreasingIDX[-1]],xs[increasingIDX[0]])
        elif increasingIDX[-1] == decreasingIDX[0]-1: #Function goes from inc -> dec
            print(f'Extrema located between x={xs[increasingIDX[-1]]} and {xs[decreasingIDX[0]]}')
            return (xs[increasingIDX[-1]],xs[decreasingIDX[0]])

        
# xinc, yinc = find_increasing(quadratic, -10, 10)

def find_increasing(function, xmin, xmax, n_pts=100):
    '''
    Locate the region in a given interval where function is increasing.
    
    Inputs:
    function[function]: the function that you want to locate increasing regions on  (one scalar input)
    xmin: lower bound of interval
    xmax: upper bound of interval
    n_pts (default = 100): how many points to use when evaluating derivative (more = better resolution but slower)
    
    Outputs:
    xs: the x values where the function is increasing
    ys: the y values where the function is increasing
    '''
    xs = np.linspace(xmin, xmax, num=n_pts)
    diag_mtx = gd.trace(function, xs)    
    # Assemble values by pulling our diagonal matrix entries
    ys = []
    for i in range(n_pts):
        ys.append(diag_mtx[i][i]) # append the derivative
    ys = np.array(ys)    
    idx = np.where(ys > 0)[0].astype(int)
    if len(idx) == 0:
        print(f'No increasing values located in the interval {xmin} to {xmax}')
        return None
    return xs[idx], ys[idx]


# xdec, ydec = find_decreasing(quadratic, -10, 10)

def find_decreasing(function, xmin, xmax, n_pts=100):
    '''
    Locate the region in a given interval where function is decreasing.
    
    Inputs:
    function[function]: the function that you want to locate decreasing regions on  (one scalar input)
    xmin: lower bound of interval
    xmax: upper bound of interval
    n_pts (default = 100): how many points to use when evaluating derivative (more = better resolution but slower)
    
    Outputs:
    xs: the x values where the function is decreasing
    ys: the y values where the function is decreasing
    '''
    xs = np.linspace(xmin, xmax, num=n_pts)
    diag_mtx = gd.trace(function, xs)    
    # Assemble values by pulling our diagonal matrix entries
    ys = []
    for i in range(n_pts):
        ys.append(diag_mtx[i][i]) # append the derivative
        
    ys = np.array(ys)    
    idx = np.where(ys < 0)[0].astype(int)
    if len(idx) == 0:
        print(f'No decreasing values located in the interval {xmin} to {xmax}')
        return None
    return xs[idx], ys[idx]

# xs, functionvalues = plot_with_tangent_line(quadratic, 5,  -10, 10)

def plot_with_tangent_line(function, xtangent, xmin, xmax, n_pts=100, figsize=(6,6), xlabel='x', ylabel='y', plotTitle='Function with tangent line'):
    '''
    Plot the a function between xmin and xmax, with a tangent line at xtangent, using n_pts linearly spaced points to evaluate it.
    
    Inputs:
    function[function]: the function you'd like to plot.
    xtangent: value at which you want the tangent line to intersect the function
    xmin: lower bound on which to calculate derivative
    xmax: upper bound on which to calculate derivative
    n_pts [int](Default: 100): how many points to use when plotting function (more = better resolution but slower)
    figsize [tuple](Default: (6,6)): figsize in inches (see matplotlib documentation)
    xlabel [string](Default: 'x'): Label for x axis of plot
    ylabel [string](Default: 'y'): Label for y axis of plot
    plotTitle [string](Default: 'Derivative'): Label for title of plot
    
    Outputs: 
    xs: the array of linearly spaced x values between xmin and xmax
    ys: the derivative evaluated at the values in xs
    '''
    deriv = gd.trace(function,xtangent)
    xs = np.linspace(xmin, xmax, num=n_pts)
    values = function(xs)
    plt.figure(figsize=figsize)
    derivativevalue = deriv[0][0]
    print(f'At the point x={xtangent}, the function has a slope of {derivativevalue}')
    plt.plot(xs, values)
    plt.plot(xs, derivativevalue*xs)
    plt.title(plotTitle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xmin, xmax)
    plt.show()
    return xs, values

# THIS ONE DOESN'T TOTALLY WORK RIGHT YET 
def plot_with_normal_line(function, xnormal, xmin, xmax, n_pts=100, figsize=(6,6), xlabel='x', ylabel='y', plotTitle='Function with normal line'):
    '''
    Plot the a function between xmin and xmax, with a normal line at xnormal, using n_pts linearly spaced points to evaluate it.
    
    Inputs:
    function[function]: the function you'd like to plot.
    xnormal: value at which you want the normal line to intersect the function
    xmin: lower bound on which to calculate derivative
    xmax: upper bound on which to calculate derivative
    n_pts [int](Default: 100): how many points to use when plotting function (more = better resolution but slower)
    figsize [tuple](Default: (6,6)): figsize in inches (see matplotlib documentation)
    xlabel [string](Default: 'x'): Label for x axis of plot
    ylabel [string](Default: 'y'): Label for y axis of plot
    plotTitle [string](Default: 'Derivative'): Label for title of plot
    
    Outputs: 
    xs: the array of linearly spaced x values between xmin and xmax
    ys: the derivative evaluated at the values in xs
    '''
    deriv = gd.trace(function,xnormal)
    xs = np.linspace(xmin, xmax, num=n_pts)
    values = function(xs)
    ynormal = function(xnormal)
#     print(ynormal)
    derivativevalue = deriv[0][0]
    print(f'At the point x={xnormal}, the function has a slope of {derivativevalue}')
    plt.figure(figsize=figsize)
    slope = (-1) /(derivativevalue)
    print(f'At the point x={xnormal}, the slope of the tangent line is {slope}')
    line = slope*(xs-xnormal) + ynormal
    plt.plot(xs, values)
    plt.plot(xs, line)
    plt.title(plotTitle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xmin, xmax)
    plt.show()
    return xs, values

    
    
