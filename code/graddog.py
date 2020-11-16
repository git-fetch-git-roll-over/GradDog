# This is our automatic differentiation package
# which takes in composition of different elementary functions
# combines different elementary func values and derivatives via forward mode
from functions import Variable, Function #, sin, cos, exp, log
# that file should be called functions not ad

# 
'''
peyton pseudocode 
we want this to do something like 
(assuming composite_function is something like sin(2*x) + cos(3*y) )

# get the innermost function from the first part of the comp.fxn
inner = composite_function[0].inner # Or something!! the [] would work if functions.py implements __getitem__
## e.g. inner = 2*x, inner.der = 2, composite_function[0] is sin(2x)
# Chain rule
cf1der = composite_function[0].der * inner.der
inner2 = composite_function[1].inner
fprime = np.empty(composite_function.shape)

for i, function in enumerate(composite_function):
    # Chain rule
    fprime[i] = function.der * function.inner.der

# And then reassemble the composite function from the things in fprime
actual_derivative = fprime[0] + fprime[1]

'''


#def gradient():
