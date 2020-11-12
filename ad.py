# Start writing code here...
# here goes nothin


'''

TODO:
multiple variables

trace table (using temporary variables v1, v2, ...)
# f(x, y) = exp(-(sin(x) - cos(y))^2)


| Trace | Element operation | Element value | Deriv 1 		   		   | Deriv 2                | [Deriv 1, Deriv 2]
| ------| -----------       | ------------- | -------- 		   		   | -------                | ----------------
| x1    | x1      			| pi/2   		| 1    	  		   		   | 0	                    | [1, 0]
| x2    | x2      			| pi/3   		| 0       		   		   | 1   	                | [0, 1]
| v1    | sin(x1)   		| 1   			| cos(x1)   	   		   | 0	                    | [0, 0]
| v2    | cos(x2)      	 	| 1/2   		| 0   			   		   | -sin(x2)               | [0, -sqrt(3)/2]
| v3    | v1 - v2      	 	| 1/2   		| cos(x1)   	   		   | sin(x2) 	            | [0, sqrt(3)/2]
| v4    | (v3)^2      	 	| 1/4   		| 2(v3)cos(x1)     		   | 2(v3)sin(x2)           | [0, sqrt(3)/2]
| v5    | -v4	      	 	| -1/4   		| -2(v3)cos(x1)    		   | -2(v3)sin(x2)          | [0, -sqrt(3)/2]
| v6    | exp(v5)      	 	| exp(-1/4)   	| -2(v3)cos(x1)exp(v5)     | -2(v3)sin(x2)exp(v5)   | [0, exp(-1/4)sqrt(3)/2]

Therfore the partial derivative of f with respect to x at (pi/2, pi/3) is 0, and 
the partial derivative of f with respect to y at (pi/2, pi/3) is exp(-1/4)sqrt(3)/2

'''

import numpy as np

class Function:

	def __init__(self, name, val, der):
		# using a dictionary for inputs for now
		self.name = name
		self.val = val
		self.der = der

	def __repr__(self):
		return self.name

	def __add__(self, other):
		# self + other
		try:
			return Function(name = self.name+'+'+other.name, val = self.val + other.val, der = self.der + other.der)
		except AttributeError:
			return Function(name = self.name + f'+{other}', val = self.val + other, der = self.der)

	def __radd__(self, other):
		# other + self
		try:
			return Function(name = other.name+'+'+self.name, val = self.val + other.val, der = self.der + other.der)
		except AttributeError:
			return Function(name = f'{other}+'+self.name, val = self.val + other, der = self.der)

	def __sub__(self, other):
		# self - other
		try:
			return Function(name = self.name+'-'+other.name, val = self.val - other.val, der = self.der - other.der)
		except AttributeError:
			return Function(name = self.name + f'-{other}', val = self.val - other, der = self.der)

	def __rsub__(self, other):
		# other - self
		try:
			return Function(name = other.name+'-'+self.name, val = other.val - self.val, der = other.der - self.der)
		except AttributeError:
			return Function(name = f'{other}-'+self.name, val = other - self.val, der = -self.der)

	def __mul__(self, other):
		# self * other
		try:
			return Function(name = self.name+'*'+other.name, val = self.val*other.val, der = self.der*other.val + self.val*other.der)
		except AttributeError:
			return Function(name = self.name + f'*{other}', val = self.val*other, der = self.der*other)

	def __rmul__(self, other):
		# other * self
		try:
			return Function(name = other.name+'*'+self.name, val = self.val*other.val, der = self.der*other.val + self.val*other.der)
		except AttributeError:
			return Function(name = f'{other}*' + self.name, val = self.val*other, der = self.der*other)

	def __div__(self, other):
		# self/other
		try:
			return Function(name = self.name+'/'+other.name, val = self.val / other.val, der = (self.der*other.val - self.val*other.der)/(other.val**2))
		except AttributeError:
			return Function(name = self.name + f'/{other}', val = self.val / other, der = self.der / other)

	def __rdiv__(self, other):
		# other/self
		try:
			return Function(name = other.name+'/'+self.name, val = other.val / self.val, der = (other.der*self.val - self.der*other.val)/(self.val**2))
		except AttributeError:
			return Function(name = f'{other}/' + self.name, val = other / self.val, der = - (self.der*other)/(self.val**2))

	def __neg__(self):
		# -self
		return Function(name = f'-({self.name})', val = -self.val, der = -self.der)

	def __pow__(self, other):
		# self ** other (exponentiation)
		# currently assumes that other is a scalar, not a Function object
		# I currently don't know how to implement the derivative of f ** g when f and g are both function...
		return Function(name = f'({self.name})**{other}', val = self.val ** other, der = other*self.val**(other - 1)*self.der)

def variable(name, val):
	return Function(name = name, val = val, der = 1)

def sin(x):
	return Function(name = f'sin({x})', val = np.sin(x.val), der = np.cos(x.val)*x.der)

def cos(x):
	return Function(name = f'cos({x})', val = np.cos(x.val), der = -np.sin(x.val)*x.der)

def tan(x):
	return Function(name = f'tan({x})', val = np.tan(x.val), der = x.der/(np.cos(x.val)**2))

def exp(x):
	return Function(name = f'exp({x})', val = np.exp(x.val), der = np.exp(x.val)*x.der)

def log(x):
	return Function(name = f'log({x})', val = np.log(x.val), der = x.der/x.val)
		
def display_func(f):
	print(f, 'evaluated at x = ', x.val)
	print(f.val)
	print('Derivative:', f.der)

def test_der(f, f_der):
	# currently assumes that f is a Function
	# allows for f_der to be a function or a scalar
	# distinguishes between exact equality, good approximation, and bad approximation

	try:
		f_der_val = f_der.val
	except AttributeError:
		f_der_val = f_der
	try:
		assert f.der == f_der_val
		print('Passed deriv test :)')
	except AssertionError:
		precision = 1e-15
		if abs(f.der - f_der_val) < precision:
			print('close enough')
		else:
			print('not good enough')

def random_polynomial_test(x):
	# creates a giant random polynomial called p
	# then compares p.der with a manually-computed polynomial derivative p_der

	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# 							create p(x)

	#choose a random polynomial degree in [1, 20)
	n = np.random.choice(np.arange(1,20))

	#generate random floats in (0, 100) for coefficients
	#size = n + 1 because there is 1 constant term and n poly_terms
	coefs = 100*np.random.random(size = n + 1)

	# the exponents for the polynomial are 1, ..., n
	exps = np.arange(1, n + 1)

	# the polynomial terms are the constant 1 followed by the function objects x, x**2, ..., x**n  
	poly_terms = np.array([1] + [x**d for d in exps])

	#constructs the polynomial as a dot product, e.g., p(x) = 3 + 4*x + 5*x**2 can be re-written as np.dot([3, 4, 5], [1, x, x**2])
	p = np.dot(coefs, poly_terms)
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# 							create p'(x)

	# calculate the new terms for the derivative: 
	# x**d ---> d*x**(d - 1) for d = 1, ..., n

	#ignore the first coefficient from the original polynomial because it represents the scalar term whose deriv is always 0
	deriv_coefs = (coefs[1:] * exps)*x.der

	# the exponents should be 1, ..., n - 1 for the derivative
	deriv_exps = exps[:-1]
	deriv_poly_terms = np.array([1] + [x**d for d in deriv_exps])

	#again use dot product to make the polynomial
	p_der = np.dot(deriv_coefs, deriv_poly_terms)
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


	test_der(p, p_der)


x = variable('x', np.pi/3)

random_polynomial_test(x)
random_polynomial_test(sin(x))
random_polynomial_test(cos(x))
random_polynomial_test(tan(x))
random_polynomial_test(exp(x))
random_polynomial_test(log(x))


f = exp(-(sin(x) - cos(x))**2.0)

#manually write out this function's derivative to test it
f_der = exp(-(sin(x) - cos(x))**2.0)*(-2.0*(sin(x) - cos(x))*(cos(x) + sin(x)))

test_der(f, f_der)

random_polynomial_test(f)






