# :)
import numpy as np
from collections.abc import Iterable
import numbers
import pandas as pd
import graddog.math as math
from graddog.compgraph import CompGraph


class Trace:
	'''
	This is a class for creating single Trace element.
	'''
	def __init__(self, formula, val, der, parents, op = None, param = None):
		'''
		The constructor for Trace class.
		Adds the new trace element to the CompGraph
		'''
		self._formula = formula

		# val stores the value
		self._val = val

		# der stores the derivative
		self._der = der

		# parents stores the 1 or 2 parent Trace object(s)
			# for example:
			# if v3 = v1+v2, then v3._parents = [v1, v2]
			# if v5 = sin(v4), then v5._parents = [v4]
		# is an empty list [] if this is a variable
		self._parents = parents

		# op stores the operation: '+', 'sin', etc
		# 
		# op is None if this is a variable
		self._op = op

		# optional parameter for a function, e.g. the base of a logarithm
		# default is None
		self._param = param

		# accesses the current CompGraph to know what this Trace's tracename should be
		# because we number the traces in order of their creation in the computational graph
		self._trace_name = CompGraph.add_trace(self)

		#name by default is the trace name, but can be changed to something like 'Cost' or 'Objective' or 'Loss'
		self._name = self._trace_name

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

	def __repr__(self): 
		return self._name


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
			other (Trace, float or int)
		Returns Trace: contains new formula, new value and new derivative.
		'''
		return two_parents(self, math.Ops.add, other)


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
		return two_parents(self, math.Ops.sub, other)

	def __rsub__(self, other):
		'''
		This is called when int or float - an instance of Trace class.
		Returns Trace: contains new value and new derivative
		'''
		return one_parent(self, math.Ops.sub_R, other)

	def __mul__(self, other):
		'''
		This allows to do Multiplication with Trace instances or scaler number. 
		AttributeError is caught when input other is not a instance of 
		Trace class. 
		Parameters:
			other (Trace, float or int)
		Returns Trace: contains new formula, new value and new derivative.
		'''
		return two_parents(self, math.Ops.mul, other)

	def __rmul__(self, other):
		'''
		This is called when int or float / an instance of Trace class.
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
		return two_parents(self, math.Ops.div, other)

	def __rtruediv__(self, other):
		'''
		This is called when and int or float / Trace instance
		
		Returns Trace: contains new formula, new value and new derivative
		'''
		return one_parent(self, math.Ops.div_R, other, formula = f'{other}/{self._trace_name}')    
	
	def __neg__(self):
		'''
		This allows to negate Trace instances itself.
		
		Returns Trace: contains instance name, (-1) * instance value and (-1) * instance derivative.
		'''
		return one_parent(self, math.Ops.sub_R, 0, formula = f'-{self._trace_name}')

	def __pow__(self, other):
		'''
		This allows to do Trace ^ Trace or scalar number. 
		AttributeError is caught when input `other` is not a instance of 
		Trace class. 
		Parameters:
			other (Trace, float or int)
		Returns Trace: contains new formula, new value and new derivative.
		'''
		return two_parents(self, math.Ops.power, other)
	
	def __rpow__(self, other):
		'''
		This is called when int or float ^ Trace instance
		
		Returns Trace: contains new formula, new value and new derivative
		'''
		return one_parent(self, math.Ops.exp, other, formula = f'{other}^{self._trace_name}')



class Variable(Trace):
	'''
	This is a class to create Variable instance
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
		if not isinstance(val, numbers.Number):
			raise TypeError('Value of variable should be numerical')
		super().__init__(name, val, {name : 1.0}, [])
		self._name = name

def one_parent(t : Trace, op, param = None, formula = None):
	'''
	Creates a trace from one parent, with an optional parameter and optional formula
	Due to error handling in other files, this function is guaranteed to only be called when t
	'''
	if param and not isinstance(param, numbers.Number):
		raise TypeError("Parameter must be scalar type")

	try:
		new_formula =  f'{op}({t._trace_name})'
	except AttributeError:
		raise TypeError('Input t must be of type Trace')
	if formula:
		new_formula = formula
	val = math.val(t, op, param)
	der =  math.deriv(t, op, param)
	parents = [t]
	return Trace(new_formula, val, der, parents, op, param)	

def two_parents(t1 : Trace, op, t2, formula = None):
	'''
	Creates a trace from two parents, with an optional formula
	'''
	# t2 is either a trace or a scalar, never a list
	try: 
		# when t2 is a trace
		new_formula =  t1._trace_name + op + t2._trace_name
		val = math.val(t1, op, t2)
		der =  math.deriv(t1, op, t2)
		parents = [t1, t2]
		return Trace(new_formula, val, der, parents, op)
		
	except AttributeError:
		if isinstance(t2, numbers.Number):
            # when t2 is actually a constant, not a trace, and this should really be a one parent trace
			return one_parent(t1, op, t2, formula = f'{t1._trace_name}{op}{t2}')
		else:
			raise TypeError("Input must be numerical or Trace instance")














