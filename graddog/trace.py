# :)
import numpy as np
import pandas as pd
import graddog.calc_rules as calc_rules
from graddog.compgraph import CompGraph
# from graddog.functions import VectorFunction
import numbers

'''
Changes I made to Max's code:
	1. Changed Trace.repr(), Trace.eq() and added ne()
	2. added all the docstring


Questions to Max's code:
	1. def der_wrt:
		If a varaible doesn't exist, would it just gives 0 as the derivative?
	2. Do we need other comparison dunder methods, like __lt__  other than eq and ne?
	3. What is the difference between _name and _trace_name
'''

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
		It adds the new trace element to the CompGraph.

		Parameters:
			formula: formula of the variable. e.g. x, y, z and etc (string)
			val: value of the variable (float)
			der: dicionary that stores the derivatives of the variable

		Attributes: 
			_formula: formula of the variable (string)
			_val: value of the variable (float)
			_der: dicionary that stores the derivatives of the variable
			_name: name of the trace ??
			_trace_name: name of the trace

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
		Returns non-public attribute _name.
		'''
		return self._name

	@name.setter
	def name(self, new_name):
		'''
		This resets the _name of a Trace instance.
		'''
		if isinstance(new_name, str):
			self._name = new_name
		else:
			raise TypeError("Name should be a string")

	@property
	def val(self):
		'''
		Returns non-public attribute _val.
		'''
		return self._val

	@val.setter
	def val(self, new_val):
		'''
		This resets the _val of a Trace instance. It raises TypeError if new_val
		passed in is not numerical.

		Parameter: 
			new_val: new value of the Trace instannce (float)
		'''
		if isinstance(new_val, numbers.Number):
			self._val = new_val
		else:
			raise TypeError('Value should be numerical')

	@property
	def der(self):
		'''
		Returns non-public attribute _der.
		If the function is single-variable, returns as a scalar instead of a dictionary.
		'''
		if len(self._der) == 1:
			return list(self._der.values())[0]
		return self._der

	def der_wrt(self, key):
		'''
		Returns derivative with respect to specific variable.

		Parameter:
			key: the name of the variable (string)
		'''
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
		'''
		Compare if other is equal to self by evaluating its formula and value. AttributeError is
		caught if other is not a Trace object

		Parameters: 
			other: an object

		Returns True if self == other; otherwise, False
		'''
		try:
			return (self._formula == other._formula) and (self.val == other.val)
		except AttributeError:
			return False


	def __ne__(self, other):
		'''
		Compare if other is not equal to self by evaluating its formula and value. AttributeError is 
		caught is other is not a Trace object.

		Parameters: 
			other: an object

		Returns True if self != other; otherwise, False
		'''
		try:
			return (self._formula != other._formula) or (self.val != other.val)
		except AttributeError:
			return True

	def __add__(self, other):
		'''
		This allows to do addition with Trace instances or scalar numbers. 
		AttributeError is caught when input `other` is not a instance of 
		Trace class. 

		Parameters:
			other: an object (Trace, float or int)

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
			other: an object (Trace, float or int)

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
			other: an object (Trace, float or int)

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
			other: an object (Trace, float or int)

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
			other: an object (Trace, float or int)

		Returns Trace: contains new formula, new value and new derivative.
		'''
		try: 
			new_formula = self._trace_name +  '^' + other._trace_name
			new_val = self._val ** other._val
		except AttributeError: 
			# if other == 0:
			# 	new_formula = '1'
			# else:
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


