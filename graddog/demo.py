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

	# equivalent to
	# x, y = Variable('x', 1), Variable('y', 2)

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
	#print(f.der)
	CompGraph.show_trace_table()
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

def demo6():
	print('demo 6')
	CompGraph.reset()

	x, y = get_xy(seed = [1,2])

	f = x*y + exp(x*y)

	CompGraph.show_trace_table()

	print('Since v4 = exp(v3), the partial derivative dv4/dv3 = exp(v3.val)')

	print('Since v3 = v1 * v2, the partial derivatives are dv3/dv1 = v2.val and dv3/dv2 = v1.val')

	CompGraph.show_partials()

	print(CompGraph.partial_deriv('v4', 'v3'), np.exp(2))

	print(CompGraph.partial_deriv('v3', 'v1'), CompGraph.partial_deriv('v3', 'v2'))


def Physics_Demo(x0=2, v=4):
	print('Physics Demo: A Simple Harmonic Oscillator')
	print('Motion: position x = x0 + v*t^2')
	print('x0 =', x0)
	print('v =', v)
#     def x(t):
#         return x0 + v*t^2
#     function_to_Trace(x, [5])

	t = Variable('t', 5)
	x = x0 + v*t**2
	print('dx/dt =', x.der)

	# SECOND DERIVATIVE 
	a = x.second_der['t']
	print('Requires an acceleration of', a)
    
    
    


