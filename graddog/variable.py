import numbers
from graddog.trace import Trace
from graddog.functions import VectorFunction
import numpy as np

# TODO: change assert statements to ValueError exception handlers

# TODO: better name for trace function
# reminder: the trace function exists to create a trace object based on a function

'''
Changes I made to Max's code:
	1. deleted val related methods in Variable class as we alread have them in the bae Trace class
	2. moved all the get_x and get_vars methods into Variable class.
	3. added all the docstrings

Questions to Max's code:
	1. _name vs _trace_name?
'''

class Variable(Trace):
	'''
	This is a class to create Varaible instance
	'''
	def __init__(self, name, val):
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
		super().__init__(name, val, {name : 1.0}, is_var = True)
		self._name = name


def function_to_Trace(f, seed):
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
	'''
	This method is to create a Variable instance with name 'x' 
	and value in the seed (list).

	Parameters:
		seed: a list of values for Variable instance x.

	Returns a Variable instance x with customized value. 
	'''
	if len(seed) != 1:
		raise ValueError('Length of seed must be 1.')
	return get_vars(['x'], seed)[0]

def get_y(seed):
	'''
	This method is to create a Variable instance with name 'y' 
	and value in the seed (list).

	Parameters:
		seed: a list of values for Variable instance y.

	Returns a Variable instance y with customized value. 
	'''
	if len(seed) != 1:
		raise ValueError('Length of seed must be 1.')
	return get_vars(['y'], seed)[0]

def get_xy(seed):
	'''
	This method is to create both Variable instance with name 'x' and 'y' 
	and values in the seed (list).

	Parameters:
		seed: a list of values for Variable instance x and y.

	Returns two Variable instances x and y with customized values. 
	'''
	if len(seed) != 2:
		raise ValueError('Length of seed must be 2.')
	return get_vars(['x', 'y'], seed)

def get_xyz(seed):
	'''
	This method is to create both Variable instance with name 'x', 'y' 
	and 'z' with values in the seed (list).

	Parameters:
		seed: a list of values for Variable instance x, y and z.

	Returns three Variable instances x, y and z with customized values. 
	'''
	if len(seed) != 3:
		raise ValueError('Length of seed must be 3.')
	return get_vars(['x', 'y', 'z'], seed)

def get_abc(seed):
	'''
	This method is to create both Variable instance with name 'a', 'b' 
	and 'c' with values in the seed (list).

	Parameters:
		seed: a list of values for Variable instance a, b and c.

	Returns three Variable instances a, b and c with customized values. 
	'''
	if len(seed) != 3:
		raise ValueError('Length of seed must be 3.')
	return get_vars(['a', 'b', 'c'], seed)

def get_vars(names, seed):
	'''
	This method is to create Variable instances with name in the 
	names list and values in the seed (list).

	Parameters:
		names: a list of names.
		seed: a list of values.

	Returns Variable instances with customized names and values. 
	'''
	if len(names) != len(seed):
		raise ValueError("Lengths of seed and name lists must match")
	return list(Variable(names[i], seed[i]) for i in range(len(names)))
