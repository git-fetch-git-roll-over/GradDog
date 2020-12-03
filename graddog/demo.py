import numpy as np
from graddog.variable import get_x, get_xy, get_xyz, trace, get_vars
from graddog.functions import sin, cos, tan, exp, log
from graddog.functions import VectorFunction as vec
from graddog.compgraph import CompGraph

def demo0():
	pass
	# print('demo 0')

	# x = get_x(seed = [0.])

	# # f : R --> R

	# f = x**3 - 4*x + sin(log(tan(exp(cos(x/2)))))

	# f.name = 'A function R --> R using variables'
	# print(f.der)
	# print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

def demo1():
	print('demo 1')

	x, y = get_xy(seed = [1,2])

	# f : Rm --> R

	f = x*y + exp(x*y)

	f.name = 'A function Rm --> R using variables'
	print(f.der)
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

def demo2():
	print('demo 2')

	x, y, z = get_xyz(seed = [np.pi/2,np.pi/3,np.pi/4])

	# f : Rm --> Rn

	f = vec([
		exp(-(sin(x) - cos(y))**2), 
		sin(- log(x) ** 2 + tan(z))
	])

	f.name = 'A function Rm --> Rn using variables'
	print(f.der)
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

def demo3():
	print('demo 3')
	CompGraph.reset()

	# f : Rm --> R

	def f(v):
		
		return v[0] + 3*v[2]**2

	f = trace(f, [1,2,3])

	f.name = 'A function Rm --> R using vectors'
	print(f.der)
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

def demo4():
	print('demo 4')
	CompGraph.reset()

	# f : Rm --> Rn

	def f(v):
		return [v[0] + 3*v[2]**2, v[1] - v[0], v[2] + sin(v[1])]

	f = trace(f, [1,2,3])

	f.name = 'A function Rm --> Rn using vectors'
	print(f.der)
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

def demo5():
	print('demo 5')
	CompGraph.reset()

	w1, w2, w3, w4, w5 = get_vars(['w1', 'w2', 'w3', 'w4', 'w5'], seed = [2,1,1,1,1])

	# f : Rm --> R

	f = w1*w2*w3*w4*w5
	CompGraph.show_trace_table()
	CompGraph.reverse_mode()
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

def run_demos():
	l = [demo0, demo1, demo2, demo3, demo4, demo5]
	for d in l:
		d()


