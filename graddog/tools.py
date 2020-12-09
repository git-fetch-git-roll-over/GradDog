# Tools to help the user engage with our package
# :)
import numpy as np
import matplotlib.pyplot as plt
import graddog as gd
from graddog.functions import sin, cos, tan, exp, log

def plot_derivative(function, xmin, xmax, n_pts=100, figsize=(6,6), xlabel='x', ylabel='y', plotTitle='Derivative'):
    '''
    Plot the derivative of a function between xmin and xmax, using n_pts linearly spaced points to evaluate it.
    
    Inputs:
    function[function]: the function of which you'd like to plot the derivative.
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
    xs: the x values of the extrema
    ys: the y values of the extrema
    '''
    xs = np.linspace(xmin, xmax, num=n_pts)
    diag_mtx = gd.trace(function, xs)    
    # Assemble values by pulling our diagonal matrix entries
    ys = []
    for i in range(n_pts):
        ys.append(diag_mtx[i][i]) # append the derivative
        
    idx = np.where(np.abs(ys) < tolerance)[0].astype(int)
    if len(idx) == 0:
        print(f'No extrema located in the interval {xmin} to {xmax}')
        return None
    return xs[idx], ys[idx]

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
        
    idx = np.where(ys > 0)[0].astype(int)
    if len(idx) == 0:
        print(f'No increasing values located in the interval {xmin} to {xmax}')
        return None
    return xs[idx], ys[idx]
  
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
        
    idx = np.where(ys < 0)[0].astype(int)
    if len(idx) == 0:
        print(f'No decreasing values located in the interval {xmin} to {xmax}')
        return None
    return xs[idx], ys[idx]

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
    
    plt.plot(xs, values)
    plt.plot(xs, deriv*xs)
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
#     print(derivativevalue)
    plt.figure(figsize=figsize)
    slope = (-1) /(derivativevalue)
#     print(slope)
    line = slope*(xs-xnormal) + ynormal
    plt.plot(xs, values)
    plt.plot(xs, line)
    plt.title(plotTitle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xmin, xmax)
    plt.show()
    return xs, values

    
    
