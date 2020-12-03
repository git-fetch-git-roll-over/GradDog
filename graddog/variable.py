import numbers
from graddog.trace import Trace
from graddog.functions import VectorFunction
import numpy as np

# TODO: change assert statements to ValueError exception handlers

# TODO: better name for trace function
# reminder: the trace function exists to create a trace object based on a function

class Variable(Trace):

	def __init__(self, name, val):
		# by default, the derivative of a variable with respect to itself is 1.0

		super().__init__(name, val, {name : 1.0}, is_var = True)
		self._name = name

	@property
	def val(self):
		'''
		Returns non-public attribute _val
		'''
		return self._val

	@val.setter
	def val(self, new_val):
		'''
		This resets the _val of a Variable instance
		'''
		if isinstance(new_val, numbers.Number):
			self._val = new_val

		else:
			raise TypeError('Value should be numerical')

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
