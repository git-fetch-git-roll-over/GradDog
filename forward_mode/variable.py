import numpy as np
import numbers

class Variable:
	'''
	This is a class for creating single input variable.

	Attributes:
		_name (str): name of the variable.
		_val (float): value of the variable.
		_der (float): derivative of the variable (default: 1.0).
	'''
	def __init__(self, name, val, der=1.0):
		'''
		The constructor for Variable class.

		Parameters:
			_name (str): name of the variable.
			_val (float): value of the variable.
			_der (float): derivative of the variable (default: 1.0).
		'''
		if isinstance(val, numbers.Number) and isinstance(der, numbers.Number):
			self._name = name
			self._val = val
			self._der = der
		else:
			raise TypeError('Value or derivative must be numerical.')

	@property
	def name(self):
		'''
		Returns non-public attribute _name
		'''
		return self._name

	@name.setter
	def name(self, new_name):
		'''
		This resets the _name of an Variable instance
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
		This resets the _val of an Variable instance
		'''
		assert isinstance(new_val, numbers.Number), TypeError(
			'Value should be numerical')
		self._val = new_val

	@property
	def der(self):
		'''
		Returns non-public attribute _der
		'''
		return self._der


	def __repr__(self): 
		return f"{self._name} value: {self._val:.3f}; derivative: {self._der:.3f}"


	def __add__(self, other):
		'''
		This allows to do addition with Variable instances or scaler number. 
		AttributeError is caught when input other is not a instance of 
		Variable class. 

		Parameters:
			other (Variable, float or int)

		Returns Variable: contains new name, new value and new derivative.
		'''
		try: 
			new_name = self._name + '+' + other._name
			new_val = self._val + other._val
			new_der = self._der + other._der

			return Variable(new_name, new_val, new_der)
	
		except AttributeError: 
			new_name = self._name + '+' + str(other)
			new_val = self._val + other
			new_der = self._der

			return Variable(new_name, new_val, new_der)

	def __radd__(self, other):
		'''
		This is called when int of float + an instance of Varibale class.
		
		Returns Variable: contains new name, new value and new derivative
		'''
		return self.__add__(other)


	def __sub__(self, other):
		'''
		This allows to do substraction with Variable instances or scaler number. 
		AttributeError is caught when input other is not a instance of 
		Variable class. 

		Parameters:
			other (Variable, float or int)

		Returns Variable: contains new name, new value and new derivative.
		'''
		try:
			new_name = self._name + '-' + other._name
			new_val = self._val - other._val
			new_der = self._der - other._der

			return Variable(new_name, new_val, new_der)
		
		except AttributeError: 
			new_name = self._name + '-' + str(other)
			new_val = self._val - other
			new_der = self._der

			return Variable(new_name, new_val, new_der)

	def __rsub__(self, other):
		'''
		This is called when int of float - an instance of Varibale class.

		Returns Variable: contains new name, new value and new derivative
		'''
		return -self.__add__(other)


	def __mul__(self, other):
		'''
		This allows to do Multiplication with Variable instances or scaler number. 
		AttributeError is caught when input other is not a instance of 
		Variable class. 

		Parameters:
			other (Variable, float or int)

		Returns Variable: contains new name, new value and new derivative.
		'''
		try: 
			new_name = self._name + '*' + other._name
			new_val = self._val * other._val
			new_der = self._der * other._val + other._der * self._val

			return Variable(new_name, new_val, new_der)

		except AttributeError: 
			new_name = self._name + '*' + str(other)
			new_val = self._val * other
			new_der = self._der * other

			return Variable(new_name, new_val, new_der)

	def __rmul__(self, other):
		'''
		This is called when int of float / an instance of Varibale class.

		Returns Variable: contains new name, new value and new derivative
		'''
		return self.__mul__(other)


	def __truediv__(self, other):
		'''
		This allows to do Division with Variable instances or scaler number. 
		AttributeError is caught when input other is not a instance of 
		Variable class. 

		Parameters:
			other (Variable, float or int)

		Returns Variable: contains new name, new value and new derivative.
		'''
		try: 
			new_name = self._name +  '/' + other._name
			new_val = self._val / other._val
			new_der = (self._der*other._val - self._val*other._der)/(other._val**2)

			return Variable(new_name, new_val, new_der)

		except AttributeError: 

			new_val = self._val / other
			new_name = self._name + '/' + str(other)
			new_der = self._der / other

			return Variable(new_name, new_val, new_der)


	def __rtruediv__(self, other):
		'''
		This is called when and int or float / Variable instance
		
		Returns Variable: contains new name, new value and new derivative
		'''
		new_name = str(other) + '/' + self._name
		new_val = other / self._val
		new_der = (-other * self._der) / (self._val)**2

		return Variable(new_name, new_val, new_der)        

	
	def __neg__(self):
		'''
		This allows to negate Variable instances itself.
		
		Returns Variable: contains instance name, (-1) * instance value and (-1) * instance derivative.
		'''
		return Variable(f'-{self._name}', -self._val, -self._der)


	def __pow__(self, other):
		'''
		This allows to do Variable ^ Variable or scaler number. 
		AttributeError is caught when input other is not a instance of 
		Variable class. 

		Parameters:
			other (Variable, float or int)

		Returns Variable: contains new name, new value and new derivative.
		'''
		try: 
			log_self = Variable(f'log{self._name}', np.log(self._val), self._der/self._val)
			new_name = self._name +  '^' + other._name
			new_val = self._val ** other._val
			new_der = (log_self._der*other._val + other._der*log_self._val)*new_val
			
			return Variable(new_name, new_val, new_der)

		except AttributeError: 

			if other == 0:
				return 1
			else:
				return Variable(f'{self._name}^{other}', self._val**other, other*self._val**(other-1)*self._der)
	
	def __rpow__(self, other):
		'''
		This is called when int or float ^ Variable instance
		
		Returns Variable: contains new name, new value and new derivative
		'''
		new_name = f'{other}^{self._name}' 
		new_val = other ** self._val
		new_der = new_val * np.log(other)
		
		return Variable(new_name, new_val, new_der)

















    
    