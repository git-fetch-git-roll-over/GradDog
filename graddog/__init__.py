import numpy as np
from graddog.variable import Variable
from graddog.functions import sin, cos, tan, exp, log
from graddog.compgraph import CompGraph

def hello():
	print('hi')

def trace(f, seed):
	CompGraph.reset()
	M = len(seed) # dimension of the input
	new_var_names = [f'v{m+1}' for m in range(M)]
	new_vars = np.array([Variable(new_var_names[i], seed[i]) for i in range(M)])
	print('just created new variables', new_vars)
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
	CompGraph.forward_mode()
	CompGraph.reverse_mode()
