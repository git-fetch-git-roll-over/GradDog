import numpy as np
from graddog.variable import get_x, get_xy, get_xyz, trace, get_vars
from graddog.functions import sin, cos, tan, exp, log
from graddog.functions import VectorFunction as vec
from graddog.compgraph import CompGraph

def demo0():
	print('demo 0')

	def f(x):
		return x**3 - 4*x + cos(exp(-sin(tan(log(x)))))

	seed = [1.0]
	assert seed[0] > 0 # input to log must be positive
	f = trace(f, seed)
	f.name = 'A function R --> R using variable-input'
	CompGraph.show_trace_table()
	CompGraph.forward_mode()
	CompGraph.reverse_mode()
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

def demo1():
	print('demo 1')

	#x, y = get_xy(seed = [1,2])

	# f : Rm --> R

	def f(x, y):
		return x*y + exp(x*y)
	seed = [1, 2]
	f = trace(f, seed)
	f.name = 'A function Rm --> R using variable-input'
	CompGraph.show_trace_table()
	CompGraph.forward_mode()
	CompGraph.reverse_mode()
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


def demo2():
	print('demo 2')

	def f(v):
		
		return v[0] + 3*v[2]**2

	f = trace(f, [1,2,3])

	f.name = 'A function Rm --> R using vector-input'
	CompGraph.show_trace_table()
	CompGraph.forward_mode()
	CompGraph.reverse_mode()
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

def demo3():
	print('demo 3')

	def f(v):
		return [v[0] + 3*v[2]**2, v[1] - v[0], v[2] + sin(v[1])]

	f = trace(f, [1,2,3])

	f.name = 'A function Rm --> Rn using a vector-input vector-output design'
	#print(f.der)
	CompGraph.show_trace_table()
	CompGraph.forward_mode()
	CompGraph.reverse_mode()
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


def demo4():
	print('demo 4')

	def f(x, y, z):
		return [exp(-(sin(x) - cos(y))**2), sin(- log(x) ** 2 + tan(z))]

	f = trace(f, seed = [np.pi/2,np.pi/3,np.pi/4])
	f.name = 'A function Rm --> Rn using a variable-input vector-output design'
	CompGraph.show_trace_table()
	CompGraph.forward_mode()
	CompGraph.reverse_mode()
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

def run_demos():
	l = [demo0, demo1, demo2, demo3, demo4]
	for d in l:
		d()


