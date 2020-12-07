import numbers
from graddog.trace import Trace
#from graddog.functions import VectorFunction
import numpy as np
from graddog.compgraph import CompGraph


class Variable(Trace):

	def __init__(self, name, val):
		# by default, the derivative of a variable with respect to itself is 1.0
		# the parents of a variable is an emptylist
		# the operation of a variable is, for now, represented as an emptystring
		super().__init__(name, val, {name : 1.0}, [])
		self._name = name

def get_vars(names, seed):
	assert len(names) == len(seed)
	return list(Variable(names[i], seed[i]) for i in range(len(names)))

def trace(f, seed):
	CompGraph.reset()
	M = len(seed) # dimension of the input
	new_var_names = [f'v{m+1}' for m in range(M)]
	new_vars = list(get_vars(new_var_names, seed))

	try:
		# when f takes the input as a vector
		f(new_vars)
	except TypeError:
		# when f takes the input explicitly as variables
		f(*new_vars)
