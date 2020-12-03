# :)
import numpy as np
import pandas as pd
import graddog.calc_rules as calc_rules
from graddog.compgraph import CompGraph
import numbers

# TODO: dunder methods for comparison operators like __lt__ <

# TODO: figure out how to recursively print a trace object's FULL formula 
# because a Trace element no longer stores its full own formula
# e.g.

# f = sin(x) - cos(3y)

# v1 = x
# v2 = y
# v3 = sin(v1)
# v4 = 3*v2
# v5 = cos(v4)
# v6 = v3 - v5 <------ f



# TODO: add missing docstrings

# TODO (optional): replace hard-coded strings with Ops strings


class Trace:
	'''
	This is a class for creating single Trace element.
	'''
	def __init__(self, formula, val, der, is_var = False):
		'''
		The constructor for Trace class.

		Adds the new trace element to the CompGraph

		'''
		self._formula = formula
		self._name = 'output'
		if isinstance(val, numbers.Number):
			self._val = val
		else:
			raise TypeError('Value should be numerical')
		self._der = der

		CG = CompGraph.instance
		try:
			if is_var:
				self._trace_name = CG.add_var(self)
			else:
				self._trace_name = CG.add_trace(self)
		except AttributeError:
			CompGraph(self)
			self._trace_name = 'v1'
			
	@property
	def name(self):
		'''
		Returns non-public attribute _name
		'''
		return self._name

	@name.setter
	def name(self, new_name):
		'''
		This resets the _name of a Trace instance
		'''
		self._name = new_name

	@property
	def val(self):
		'''
		Returns non-public attribute _val
		'''
		return self._val

	@val.setter
	def val(self, new_val):
		'''
		This resets the _val of a Trace instance
		'''
		if isinstance(new_val, numbers.Number):
			self._val = new_val
		else:
			raise TypeError('Value should be numerical')

	@property
	def der(self):
		'''
		Returns non-public attribute _der

		If the function is single-variable, returns as a scalar instead of a dictionary

		Optional parameter: key, for example 'x', so that the user can call f.der('x')
		'''

		if len(self._der) == 1:
			return list(self._der.values())[0]
		return self._der

	def der_wrt(self, key):
		try:
			return self._der[key]
		except KeyError:
			return 0

	def __repr__(self): 
		# s = f"~~~~~~~~~~~~~  {self._name}  ~~~~~~~~~~~~~~\n"
		# s += f"formula: {self._formula}\n\nvalue: {self._val:.3f}\n\nderivative: {self.der}\n"
		# s += "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n"
		# return s
		return self._name

	@property
	def trace_table(self):

		'''
		Returns the string representation of the current CompGraph object
		'''
		print('Trace table of a forward pass')
		return repr(CompGraph.instance)

	@property
	def comp_graph(self):
		print('Comp graph : outs & ins')
		return repr(CompGraph.instance.outs) + '\n' + repr(CompGraph.instance.ins)

	def __eq__(self, other):
		try:
			return self.val == other.val
		except AttributeError:
			return self.val == other

	def __add__(self, other):
		'''
		This allows to do addition with Trace instances or scalar numbers. 
		AttributeError is caught when input `other` is not a instance of 
		Trace class. 

		Parameters:
			other (Trace, float or int)

		Returns Trace: contains new formula, new value and new derivative.
		'''
		try: 
			new_formula =  self._trace_name + '+' + other._trace_name
			new_val = self._val + other._val
		except AttributeError: 
			new_formula = self._trace_name + '+' + str(other)
			new_val = self._val + other
		new_der =  calc_rules.deriv(self, '+', other)
		return Trace(new_formula, new_val, new_der)	

	def __radd__(self, other):
		'''
		This is called when int or float + an instance of Variable class.
		
		Returns Trace: contains new formula, new value and new derivative
		'''
		return self.__add__(other)

	def __sub__(self, other):
		'''
		This allows to do subtraction with Trace instances or scalar numbers. 
		AttributeError is caught when input `other` is not a instance of 
		Trace class. 

		Parameters:
			other (Trace, float or int)

		Returns Trace: contains new formula, new value and new derivative.
		'''
		try: 
			new_formula =  self._trace_name + '-' + other._trace_name
			new_val = self._val - other._val
		except AttributeError: 
			new_formula = self._trace_name + '-' + str(other)
			new_val = self._val - other
		new_der =  calc_rules.deriv(self, '-', other)
		return Trace(new_formula, new_val, new_der)	

	def __rsub__(self, other):
		'''
		This is called when int of float - an instance of Trace class.

		Returns Trace: contains new formula, new value and new derivative
		'''
		new_formula =  + str(other) + '-' + self._trace_name
		new_val = other - self._val
		new_der =  calc_rules.deriv(self, '-R', other)
		return Trace(new_formula, new_val, new_der)	

	def __mul__(self, other):
		'''
		This allows to do Multiplication with Trace instances or scaler number. 
		AttributeError is caught when input other is not a instance of 
		Trace class. 

		Parameters:
			other (Trace, float or int)

		Returns Trace: contains new formula, new value and new derivative.
		'''
		try: 
			new_formula = self._trace_name + '*' + other._trace_name 
			new_val = self._val * other._val
		except AttributeError: 
			if other == 0:
				new_formula = '0'
			else:
				new_formula = self._trace_name + '*' + str(other)
			new_val = self._val * other
		new_der = calc_rules.deriv(self, '*', other)
		return Trace(new_formula, new_val, new_der)

	def __rmul__(self, other):
		'''
		This is called when int of float / an instance of Trace class.

		Returns Trace: contains new formula, new value and new derivative
		'''
		return self.__mul__(other)

	def __truediv__(self, other):
		'''
		This allows to do Division with Trace instances or scalar number. 
		AttributeError is caught when input `other` is not a instance of 
		Trace class. 

		Parameters:
			other (Trace, float or int)

		Returns Trace: contains new formula, new value and new derivative.
		'''
		try: 
			new_formula = self._trace_name +  '/' + other._trace_name
			new_val = self._val / other._val
		except AttributeError: 
			new_formula = self._trace_name + '/' + str(other)
			new_val = self._val / other
		new_der = calc_rules.deriv(self, '/', other)
		return Trace(new_formula, new_val, new_der)

	def __rtruediv__(self, other):
		'''
		This is called when and int or float / Trace instance
		
		Returns Trace: contains new formula, new value and new derivative
		'''
		new_formula = str(other) + '/' + self._trace_name
		new_val = other / self._val
		new_der = calc_rules.deriv(self, '/R', other)
		return Trace(new_formula, new_val, new_der)      
	
	def __neg__(self):
		'''
		This allows to negate Trace instances itself.
		
		Returns Trace: contains instance name, (-1) * instance value and (-1) * instance derivative.
		'''
		new_formula = '-'+self._trace_name
		new_val = -self._val
		new_der = calc_rules.deriv(self, '-')
		return Trace(new_formula, new_val, new_der)   

	def __pow__(self, other):
		'''
		This allows to do Trace ^ Trace or scalar number. 
		AttributeError is caught when input `other` is not a instance of 
		Trace class. 

		Parameters:
			other (Trace, float or int)

		Returns Trace: contains new formula, new value and new derivative.
		'''
		try: 
			new_formula = self._trace_name +  '^' + other._trace_name
			new_val = self._val ** other._val
		except AttributeError: 
			if other == 0:
				new_formula = '1'
			else:
				new_formula = f'{self._trace_name}^{other}'
			new_val = self._val**other
		new_der = calc_rules.deriv(self, '^', other)
		return Trace(new_formula, new_val, new_der) 
	
	def __rpow__(self, other):
		'''
		This is called when int or float ^ Trace instance
		
		Returns Trace: contains new formula, new value and new derivative
		'''
		new_formula = f'{other}^{self._trace_name}' 
		new_val = other ** self._val
		new_der = calc_rules.deriv(self, '^R', other)
		return Trace(new_formula, new_val, new_der) 


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
# 	assert len(seed) == 1
	if len(seed) != 1:
		raise ValueError('Length of seed must be 1.')
	return get_vars(['x'], seed)[0]

def get_y(seed):
	if len(seed) != 1:
		raise ValueError('Length of seed must be 1.')
	return get_vars(['y'], seed)[0]

def get_xy(seed):
	if len(seed) != 2:
		raise ValueError('Length of seed must be 2.')
	return get_vars(['x', 'y'], seed)

def get_xyz(seed):
	if len(seed) != 3:
		raise ValueError('Length of seed must be 3.')
	return get_vars(['x', 'y', 'z'], seed)

def get_abc(seed):
	if len(seed) != 3:
		raise ValueError('Length of seed must be 3.')
	return get_vars(['a', 'b', 'c'], seed)

def get_vars(names, seed):
	if len(names) != len(seed):
		raise ValueError("Lengths of seed and name lists must match")
	return list(Variable(names[i], seed[i]) for i in range(len(names)))
