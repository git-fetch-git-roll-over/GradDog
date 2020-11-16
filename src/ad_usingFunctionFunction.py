### This is based on Max's "Function" class
### I renamed it to "Variable" becuase we are making variables here
### I just double checked and made sure everything is working
### This is the Variable class that takes in the name of the variable, value, and it's default deri is 1
### For M2, we are only required to take in a sigle input
### The output is computed from a scaler function
### The Jacobian is also a scaler

import numpy as np

class Variable:

	def __init__(self, name, val, der=1):
		self.name = name
		self.val = val
		self.der = der

	def __repr__(self):
		return f"{self.name}: value of {self.val}; derivative of {self.der}"

	def set_value(self, val):
		self.val = val


	## Addition
	def __add__(self, other):
		try: # when other is a function object
			new_name = self.name + '+' + other.name
			new_val = self.val + other.val
			new_der = self.der + other.der

			return Variable(new_name, new_val, new_der)

		except AttributeError: # when other is a scaler number
			new_name = self.name + '+' + str(other)
			new_val = self.val + other
			new_der = self.der

			return Variable(new_name, new_val, new_der)

	def __radd__(self, other):
		return self.__add__(other)


	## Substraction
	def __sub__(self, other):
		try: # when other is a function object
			new_name = self.name + '-' + other.name
			new_val = self.val - other.val
			new_der = self.der - other.der

			return Variable(new_name, new_val, new_der)

		except AttributeError: # when other is a scaler number
			new_name = self.name + '-' + str(other)
			new_val = self.val - other
			new_der = self.der

			return Variable(new_name, new_val, new_der)

	def __rsub__(self, other):
		# max added - , because 
		# other - self = - (self - other)
		return -self.__sub__(other)


	## Multiplication
	def __mul__(self, other):
		try: # when other is a function object
			new_name = self.name + '*' + other.name
			new_val = self.val * other.val
			new_der = self.der * other.val + other.der * self.val

			return Variable(new_name, new_val, new_der)

		except AttributeError: # when other is a scaler number
			new_name = self.name + '*' + str(other)
			new_val = self.val * other
			new_der = self.der * other

			return Variable(new_name, new_val, new_der)

	def __rmul__(self, other):
		return self.__mul__(other)


	## Division
	def __truediv__(self, other):
		try: # when other is a funciton object
			new_name = self.name +  '/' + other.name
			new_val = self.val / other.val
			new_der = (self.der*other.val - self.val*other.der)/(other.val**2)
			return Variable(new_name, new_val, new_der)

		except AttributeError: 
			new_name = self.name + '/' + str(other)
			new_val = self.val / other
			new_der = self.der / other

			return Variable(new_name, new_val, new_der)

	def __rtruediv__(self, other):
		new_name = str(other) + '/' + self.name
		new_val = other / self.val
		new_der = (-other * self.der) / (self.val)**2

		return Variable(new_name, new_val, new_der)        

	## Negation
	def __neg__(self):
		return Variable(f'-{self.name}', -self.val, -self.der)


	## Power: here we assume we are only taking scaler number
	def __pow__(self, scaler):
		if scaler == 0:
			return 1
		return Variable(f'{self.name}^{scaler}', self.val**scaler, scaler*self.val**(scaler-1)*self.der)


    
    
### Here I'm implementing a basic 'Function' function
### This function takes in 2 arguments: 'func' and an instance of "Varaible" class

def Function(func, var:Variable, base=np.e):

    if func == 'sin':
        return Variable(f'sin{var.name}', np.sin(var.val), np.cos(var.val)*var.der)
    
    elif func == 'cos':
        return Variable(f'cos{var.name}', np.cos(var.val), -np.sin(var.val)*var.der)

    elif func == 'tan':
        return Variable(f'tan{var.name}', np.tan(var.val), var.der/(np.cos(var.val)**2))

    elif func == 'exp':
        if base==np.e:
            return Variable(f'e^{var.name}', np.exp(var.val), np.exp(var.val)*var.der)
        else: 
            return Variable(f'{str(base)}^{var.name}', np.power(base, var.val), np.power(base, var.val)*np.log(base)*var.der)

    elif func == 'log':
        if base==np.e:
            return Variable(f'log{var.name}', np.exp(var.val), np.exp(var.val)*var.der)
        else: 
            return Variable(f'log_{str(base)}{var.name}', np.log(var.val)/np.log(base), var.der/(var.val*np.log(base)))

    
    
