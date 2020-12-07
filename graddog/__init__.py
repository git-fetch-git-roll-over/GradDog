# :)
import numpy as np
from graddog.variable import Variable
from graddog.functions import sin, cos, tan, exp, log
from graddog.compgraph import CompGraph


def trace(f, seed):
	'''
	Infers the dimension of input from the seed
	Dimension of output inferred in CompGraph

	Therefore f can be

	f: R --> R using explicit single-variable input

	f: Rm --> R using explicit multi-variable input

	f: R --> Rn using explicit single-variable input and explicit vector output

	f: Rm --> R using explicit multi-variable input

	f: Rm --> R using explicit vector input

	f: Rm --> Rn using explicit vector input and explicit vector output

	f: Rm --> Rn using explicit multi-variable input and explicit vector output

	f: Rm --> Rn using IMPLICIT vector input and IMPLICIT vector output
	'''

	# for now, always reset the CompGraph when tracing a new function
	CompGraph.reset()

	try:# if multidimensional
		M = len(seed) # get the dimension of the input
		seed = np.array(seed)
	except TypeError:
		M = 1
		seed = np.array([seed])
	new_variable_names = [f'v{m+1}' for m in range(M)]
	new_vars = np.array([Variable(new_variable_names[i], seed[i]) for i in range(M)])

	if M > 1:
		# multi-variable input
		try:
			# as a vector
			f(new_vars)
		except TypeError:
			# as variables
			f(*new_vars)
	else:
		# single-variable input
		f(new_vars[0])

def show(f):
	CompGraph.show_trace_table()
	CompGraph.reverse_mode()
	CompGraph.forward_mode()
	
