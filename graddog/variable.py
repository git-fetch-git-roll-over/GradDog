import numbers
from trace import Trace
#from functions import VectorFunction
import numpy as np

# TODO: change assert statements to ValueError exception handlers

# TODO: better name for trace function
# reminder: the trace function exists to create a trace object based on a function

'''
Changes I made to Max's code:
	1. delete val related methods in Variable class as we alread have them in the bae Trace class
	2. moved all the get_x and get_vars methods into Variable class.

Questions to Max's code:
	1. _name vs _trace_name?
'''

class Variable(Trace):
	'''
	This is a class to create Varaible instance
	'''
	def __init__(self, formula, val):
		'''
		The constructor for Variable class.

		Parameters:
			formula: formula of the variable. e.g. x, y, z and etc (string)
			val: value of the variable (float)

		Attributes: 
			_formula: formula of the variable (string)
			_val: value of the variable (float)
			_der: dicionary that stores the derivatives of the variable
			_name: name of the trace ??
			_trace_name: name of the trace

		'''
		super().__init__(formula, val, {formula : 1.0}, is_var = True)
		#self._name = name

	def get_x(seed):
		assert len(seed) == 1
		return get_vars(['x'], seed)[0]

	def get_y(seed):
		assert len(seed) == 1
		return get_vars(['y'], seed)[0]

	def get_xy(seed):
		assert len(seed) == 2
		return get_vars(['x', 'y'], seed)

	def get_xyz(seed):
		assert len(seed) == 3
		return get_vars(['x', 'y', 'z'], seed)

	def get_abc(seed):
		assert len(seed) == 3
		return get_vars(['a', 'b', 'c'], seed)

	def get_vars(names, seed):
		assert len(names) == len(seed)
		return list(Variable(names[i], seed[i]) for i in range(len(names)))


def trace(f, seed):
	M = len(seed)
	new_vars = list(get_vars([f'x{m+1}' for m in range(M)], seed))
	new_var_names = list(map(lambda v : v._formula, new_vars))
	result = f(new_vars)

	def add_zeros_to_der(t):
		for x in new_var_names:
			if x not in t._der:
				t._der[x] = 0.0
	try:
		for t in result:
			add_zeros_to_der(t)
		result = VectorFunction(result)
	except TypeError:
		add_zeros_to_der(result)
	return result




