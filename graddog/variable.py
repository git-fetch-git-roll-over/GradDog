import numbers
from graddog.trace import Trace
from graddog.functions import VectorFunction
import numpy as np


class Variable(Trace):

	def __init__(self, name, val):
		# by default, the derivative of a variable with respect to itself is 1.0
		# the parents of a variable is an emptylist
		# the operation of a variable is, for now, represented as an emptystring
		super().__init__(name, val, {name : 1.0}, [])
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
	M = len(seed) # dimension of the input
	new_var_names = [f'v{m+1}' for m in range(M)]
	new_vars = list(get_vars(new_var_names, seed))

	try:
		# when f takes the input as a vector
		result = f(new_vars)
	except TypeError:
		# when f takes the input explicitly as variables
		result = f(*new_vars)

	try:
		# when f outputs a list
		return VectorFunction(result)
	except TypeError:
		# when f outputs a single value
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
