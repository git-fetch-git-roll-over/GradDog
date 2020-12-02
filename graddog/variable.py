import numbers
from trace import Trace

class Variable(Trace):

	def __init__(self, name, val):
		# by default, the derivative of a variable with respect to itself is 1.0
		super().__init__(name, val, {name : 1.0}, is_var = True)

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

def get_vars(names, seed):
	assert len(names) == len(seed)
	return (Variable(names[i], seed[i]) for i in range(len(names)))
