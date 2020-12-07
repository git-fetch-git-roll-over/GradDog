# :)
import numpy as np
import matplotlib.pyplot as plt
import graddog as gd
from graddog.functions import sin, cos, tan, exp, log

seed0 = 0.5
def f0(x):
	return x**3 - 4*x + cos(exp(-sin(tan(log(x)))))

seed1 = [1,2]
def f1(x, y):
	return x*y + exp(x*y)

seed2 = 3
def f2(x):
	return [x**2, x**3, x**4]

seed3 = [1,2,3]
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

seeds = [seed0, seed1, seed2, seed3, seed4, seed5, seed6, seed7]
fs = [f0, f1, f2, f3, f4, f5, f6, f7]

def run_demos():
	for i, f in enumerate(fs):
		print('demo', i)
		gd.trace(f, seeds[i])
		print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')



run_demos()