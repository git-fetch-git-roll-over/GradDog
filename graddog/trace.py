# :)
import numpy as np
import pandas as pd
import calc_rules
from compgraph import CompGraph

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
		self._val = val
		self._der = der

		CG = CompGraph.instance
		try:
			if is_var:
				self._trace_name = CG.add_var(formula, val)
			else:
				self._trace_name = CG.add_trace(formula, val, der)
		except AttributeError:
			CompGraph(formula, val)
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
	def der(self, key = None):
		'''
		Returns non-public attribute _der

		If the function is single-variable, returns as a scalar instead of a dictionary

		Optional parameter: key, for example 'x', so that the user can call f.der('x')
		'''

		if key:
			assert key in CompGraph.instance.var_names
			return self._der[key]

		if len(self._der) == 1:
			return self._der.values()[0]
		return self._der

	def __repr__(self): 
		
		s = f"~~~~~~~~~~~~~  {self._name}  ~~~~~~~~~~~~~~\n"
		s += f"formula: {self._formula}\n\nvalue: {self._val:.3f}\n\nderivative: {self.der}\n"
		s += "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n"
		return s

	@property
	def trace_table(self):
		'''
		Returns the string representation of the current CompGraph object
		'''
		return repr(CompGraph.instance)

	@property
	def comp_graph(self):
		return CompGraph.instance.graph

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
			new_der = calc_rules.deriv(self, '^', other)
		except AttributeError: 
			if other == 0:
				new_formula = '1'
			else:
				new_formula = f'({self._trace_name}^{other})'
			new_val = self._val**other
		new_der = calc_rules.deriv(self, '^', other)
		return Trace(new_formula, new_val, new_der) 
	
	def __rpow__(self, other):
		'''
		This is called when int or float ^ Trace instance
		
		Returns Trace: contains new formula, new value and new derivative
		'''
		new_formula = f'{other}^({self._trace_name})' 
		new_val = other ** self._val
		new_der = calc_rules.deriv(self, '^R', other)
		return Trace(new_formula, new_val, new_der) 

