# :)
import numpy as np
import numbers


class Trace:
	'''
	This is a class for creating single Trace element.


	'''
	def __init__(self, formula, val, der):
		'''
		The constructor for Trace class.

		'''
		self._formula = formula
		self._name = formula
		self._val = val
		self._der = der

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

	@property
	def der(self):
		'''
		Returns non-public attribute _der
		'''
		return self._der

	@property
	def jacobian(self):
		'''
		Returns a numpy array of the trace's jacobian vector
		'''
		return np.array([self._der[x] for x in sorted(self._der)])

	def __repr__(self): 
		formatted_dict = {x : np.round(self._der[x], 3) for x in self._der}
		return f"{self._name} value: {self._val:.3f}; derivative: {formatted_dict}"


	def __add__(self, other):
		'''
		This allows to do addition with Trace instances or scalar numbers. 
		AttributeError is caught when input `other` is not a instance of 
		Trace class. 

		Parameters:
			other (Trace, float or int)

		Returns Trace: contains new name, new value and new derivative.
		'''
		try: 
			new_formula = self._formula + '+' + other._formula
			new_val = self._val + other._val


			########################################################
			new_der =  {}
			for x in self._der:
				if x in other._der:

					# a) x in both f and g
					new_der[x] = self._der[x] + other._der[x]

				else:

					# b) x in f only
					new_der[x] = self._der[x]

			for x in other._der:
				if x not in self._der:

					# c) x in g in only
					new_der[x] = other._der[x]

			#######################################################

			return Trace(new_formula, new_val, new_der)
	
		except AttributeError: 
			new_formula = self._formula + '+' + str(other)
			new_val = self._val + other
			new_der = self._der

			return Trace(new_formula, new_val, new_der)

	def __radd__(self, other):
		'''
		This is called when int of float + an instance of Varibale class.
		
		Returns Trace: contains new name, new value and new derivative
		'''
		return self.__add__(other)


	def __sub__(self, other):

		#hehehehehe this works
		return self + -other

	def __rsub__(self, other):
		'''
		This is called when int of float - an instance of Trace class.

		Returns Trace: contains new name, new value and new derivative
		'''
		return -self + other


	def __mul__(self, other):
		'''
		This allows to do Multiplication with Trace instances or scaler number. 
		AttributeError is caught when input other is not a instance of 
		Trace class. 

		Parameters:
			other (Trace, float or int)

		Returns Trace: contains new name, new value and new derivative.
		'''
		try: 
			new_formula = '(' + self._formula + '*' + other._formula + ')'
			new_val = self._val * other._val

			##########################################################
			new_der = {}
			for x in self._der:
				if x in other._der:
					# a) x in both f and g
					new_der[x] = self._der[x] * other._val + other._der[x] * self._val
				else:
					# b) x in f only
					new_der[x] = self._der[x] * other._val
			for x in other._der:
				if x not in self._der:
					# c) x in g in only
					new_der[x] = other._der[x] * self._val
			########################################################

			return Trace(new_formula, new_val, new_der)

		except AttributeError: 
			new_formula = '(' + self._formula + '*' + str(other) + ')'
			new_val = self._val * other
			new_der = {x : self._der[x] * other for x in self._der}

			return Trace(new_formula, new_val, new_der)

	def __rmul__(self, other):
		'''
		This is called when int of float / an instance of Trace class.

		Returns Trace: contains new name, new value and new derivative
		'''
		return self.__mul__(other)


	def __truediv__(self, other):
		'''
		This allows to do Division with Trace instances or scalar number. 
		AttributeError is caught when input `other` is not a instance of 
		Trace class. 

		Parameters:
			other (Trace, float or int)

		Returns Trace: contains new name, new value and new derivative.
		'''
		try: 
			new_formula = '(' + self._formula +  '/' + other._formula + ')'
			new_val = self._val / other._val

			##########################################################
			new_der = {}
			for x in self._der:
				if x in other._der:
					# a) x in both f and g
					new_der[x] = (self._der[x]*other._val - self._val*other._der[x])/(other._val**2)
				else:
					# b) x in f only
					new_der[x] = self._der[x] / other._val
			for x in other._der:
				if x not in self._der:
					# c) x in g in only
					new_der[x] = (-self._val * other._der[x]) / (other._val)**2
			##########################################################

			return Trace(new_formula, new_val, new_der)

		except AttributeError: 

			new_val = self._val / other
			new_formula = '(' + self._formula + '/' + str(other) + ')'
			new_der = {x : self._der[x] / other for x in self._der}

			return Trace(new_formula, new_val, new_der)


	def __rtruediv__(self, other):
		'''
		This is called when and int or float / Trace instance
		
		Returns Trace: contains new name, new value and new derivative
		'''
		new_formula = '(' + str(other) + '/' + self._formula + ')'
		new_val = other / self._val
		new_der = {x : (-other * self._der[x]) / ((self._val)**2) for x in self._der} 

		return Trace(new_formula, new_val, new_der)        

	
	def __neg__(self):
		'''
		This allows to negate Trace instances itself.
		
		Returns Trace: contains instance name, (-1) * instance value and (-1) * instance derivative.
		'''
		if self._formula[0] == '-':
			new_formula = self._formula[1:]
		else:
			new_formula = '-'+self._formula
		new_val = -self._val
		new_der = {x : -self._der[x] for x in self._der}
		return Trace(new_formula, new_val, new_der)


	def __pow__(self, other):
		'''
		This allows to do Trace ^ Trace or scalar number. 
		AttributeError is caught when input `other` is not a instance of 
		Trace class. 

		Parameters:
			other (Trace, float or int)

		Returns Trace: contains new name, new value and new derivative.
		'''
		try: 
			new_formula = '(' + self._formula +  '^' + other._formula + ')'
			new_val = self._val ** other._val

			##########################################################
			new_der = {}
			for x in self._der:
				if x in other._der:
					# x in both f and g
					new_der[x] = new_val * (other._der[x]*np.log(self._val) + other._val*self._der[x]/self._val)
				else:
					# x in f only
					new_der[x] = other._val * self._der[x] * (self._val ** (other._val - 1))
			for x in other._der:
				if x not in self._der:
					# x in g only
					new_der[x] = new_val*other._der[x]*np.log(self._val)
			##########################################################
			
			return Trace(new_formula, new_val, new_der)

		except AttributeError: 

			if other == 0:
				return 1
			else:
				new_der = {x : other*self._val**(other-1)*self._der[x] for x in self._der}
				return Trace(f'{self._formula}^{other}', self._val**other, new_der)
	
	def __rpow__(self, other):
		'''
		This is called when int or float ^ Trace instance
		
		Returns Trace: contains new name, new value and new derivative
		'''
		new_formula = f'{other}^({self._formula})' 
		new_val = other ** self._val
		new_der = {x : new_val * np.log(other) * self._der[x] for x in self._der}
		
		return Trace(new_formula, new_val, new_der)
   



class Variable(Trace):

	def __init__(self, name, val):
		# by default, the derivative of a variable with respect to itself is 1.0
		super().__init__(name, val, {name : 1.0})

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





# major to do
# when we reset the val of a variable it should be able to re-trigger the entire forward pass
# :/







