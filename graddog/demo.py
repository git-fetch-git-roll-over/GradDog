import numpy as np
from variable import get_xy, get_xyz
from functions import sin, cos, tan, exp, log
from functions import VectorFunction as vec

def demo1():
	x, y, z = get_xyz(seed = [np.pi/2,np.pi/3,np.pi/4])
	f = vec([
		exp(-(sin(x) - cos(y))**2), 
		sin(- log(x) ** 2 + tan(z))
	])
	print('trace table of forward pass')
	print(f.trace_table)
	print('Jacobian of f')
	print(f.der)

def demo2():
	x, y = get_xy(seed = [1,2])
	f = x*y + exp(x*y)
	print('trace table of forward pass')
	print(f.trace_table)
	print('Jacobian of f')
	print(f.der)

demo2()

