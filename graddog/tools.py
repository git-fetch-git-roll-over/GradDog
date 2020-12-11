# Tools to help the user engage with our package
# :)
import numpy as np
import matplotlib.pyplot as plt
import graddog as gd
from graddog.functions import sin, cos, tan, exp, log

    
def plot_derivative(function, xmin, xmax, n_pts=100, figsize=(6,6), xlabel='x', ylabel='y', plotTitle='Derivative', verbose = False):
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
    
    '''
    xs = np.linspace(xmin, xmax, num=n_pts)
    # derivative comes as a matrix since we are passing in a vectorized input to a single-variable function
    # the only entries in the derivative matrix are therefore the diagonals, since it is a single-variable function
    f_ = gd.trace(function, xs, verbose = verbose)
    y_ders = [f_[i,i] for i in range(n_pts)]

    
    plt.figure(figsize=figsize)
    plt.plot(xs, y_ders, color = 'red', label = 'f\'(x)')
    plt.title(plotTitle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xmin, xmax)
    plt.legend()
    plt.show()

    return xs, y_ders


def find_extrema_firstorder(function, xmin, xmax, n_pts=100, tolerance = 1e-10, verbose = False):
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
    # derivative comes as a matrix since we are passing in a vectorized input to a single-variable function
    # the only entries in the derivative matrix are therefore the diagonals, since it is a single-variable function
    f_ = gd.trace(function, xs, verbose = verbose)
    y_ders = np.array([f_[i,i] for i in range(n_pts)])

    zeroidx = np.where(np.abs(y_ders) < tolerance)[0].astype(int)
    if len(zeroidx) != 0:
        return xs[zeroidx]
    else:
        decreasingIDX = np.where(y_ders < tolerance)[0].astype(int)
        increasingIDX = np.where(y_ders > tolerance)[0].astype(int)

        if len(decreasingIDX) == 0 or len(increasingIDX) == 0:
            print(f'No extrema located in the interval {xmin} to {xmax}.')
            return None

        if decreasingIDX[-1] == increasingIDX[0]-1: # Function goes from decreasing -> increasing
            print(f'Extrema located between x={xs[decreasingIDX[-1]]} and {xs[increasingIDX[0]]}')
            return (xs[decreasingIDX[-1]],xs[increasingIDX[0]])
        elif increasingIDX[-1] == decreasingIDX[0]-1: #Function goes from inc -> dec
            print(f'Extrema located between x={xs[increasingIDX[-1]]} and {xs[decreasingIDX[0]]}')
            return (xs[increasingIDX[-1]],xs[decreasingIDX[0]])

def find_increasing(function, xmin, xmax, n_pts=100, verbose = False):
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
    # derivative comes as a matrix since we are passing in a vectorized input to a single-variable function
    # the only entries in the derivative matrix are therefore the diagonals, since it is a single-variable function
    f_ = gd.trace(function, xs, verbose = verbose)
    y_ders = np.array([f_[i,i] for i in range(n_pts)])

    idx = np.where(y_ders > 0)[0].astype(int)
    if len(idx) == 0:
        print(f'No increasing values located in the interval {xmin} to {xmax}')
        return None
    return xs[idx], y_ders[idx]


# xdec, ydec = find_decreasing(quadratic, -10, 10)

def find_decreasing(function, xmin, xmax, n_pts=100, verbose = False):
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
    # derivative comes as a matrix since we are passing in a vectorized input to a single-variable function
    # the only entries in the derivative matrix are therefore the diagonals, since it is a single-variable function
    f_ = gd.trace(function, xs, verbose = verbose)
    y_ders = np.array([f_[i,i] for i in range(n_pts)])

    idx = np.where(y_ders < 0)[0].astype(int)
    if len(idx) == 0:
        print(f'No decreasing values located in the interval {xmin} to {xmax}')
        return None
    return xs[idx], y_ders[idx]

# xs, functionvalues = plot_with_tangent_line(quadratic, 5,  -10, 10)

def plot_with_tangent_line(function, xtangent, xmin, xmax, n_pts=100, figsize=(6,6), xlabel='x', ylabel='y', plotTitle='Function with tangent line', verbose = False):
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
    deriv = gd.trace(function,xtangent, verbose = verbose)
    xs = np.linspace(xmin, xmax, num=n_pts)
    values = function(xs)
    ytangent = function(xtangent)
    plt.figure(figsize=figsize)
    derivativevalue = deriv[0,0]
    print(f'At the point x={xtangent}, the function has a slope of {derivativevalue}')
    plt.plot(xs, values)
    plt.plot(xs, derivativevalue*(xs-xtangent) + ytangent)
    plt.title(plotTitle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xmin, xmax)
    plt.show()
    return xs, values
