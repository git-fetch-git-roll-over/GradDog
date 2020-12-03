import numpy as np
from graddog.variable import get_xy, get_xyz, trace
from graddog.functions import sin, cos, tan, exp, log
from graddog.functions import VectorFunction as vec
from graddog.compgraph import CompGraph

def demo1():
	print('demo 1')
	CompGraph.reset()
	x, y = get_xy(seed = [1,2])
	f = x*y + exp(x*y)
	f.name = 'Cost'
	print(f.der)
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

def demo2():
	print('demo 2')
	CompGraph.reset()
	x, y, z = get_xyz(seed = [np.pi/2,np.pi/3,np.pi/4])
	f = vec([
		exp(-(sin(x) - cos(y))**2), 
		sin(- log(x) ** 2 + tan(z))
	])
	f.name = 'Cost'
	print(f.der)
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

def demo3():
	print('demo 3')
	CompGraph.reset()
	def f(v):
		# f : Rm --> R
		return v[0] + 3*v[2]**2

	# convert a function of an iterable into a Trace object
	f = trace(f, [1,2,3])

	print(f.der)
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

def demo4():
	print('demo 4')
	CompGraph.reset()
	def f(v):
		# f : Rm --> Rn
		return [v[0] + 3*v[2]**2, v[1] - v[0], v[2] + sin(v[1])]

	# convert a function of an iterable into a Trace object
	f = trace(f, [1,2,3])
	f.name = 'A sample function from Rm --> Rn'
	print(f.der)
	CompGraph.show_trace_table()
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

#demo1()
#demo2()
#demo3()
#demo4()
