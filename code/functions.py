'''
TODO:
reset variable value
multiple variables
trace table (using temporary variables v1, v2, ...), add a row to the trace table every time a Function object is constructed

'''
import numpy as np

class Function:

	def __init__(self, name, val, der):
		self.name = name
		self.val = val
		self.der = der

	def __repr__(self):
		return self.name

	def __add__(self, other):
		# self + other
		try:
			return Function(name = f'{self.name}+{other.name}', val = self.val + other.val, der = self.der + other.der)
		except AttributeError:
			return Function(name = f'{self.name}+{other}', val = self.val + other, der = self.der)

	def __radd__(self, other):
		# other + self
		try:
			return Function(name = f'{other.name}+{self.name}', val = self.val + other.val, der = self.der + other.der)
		except AttributeError:
			return Function(name = f'{other}+{self.name}', val = self.val + other, der = self.der)

	def __sub__(self, other):
		# self - other
		try:
			return Function(name = f'{self.name}-{other.name}', val = self.val - other.val, der = self.der - other.der)
		except AttributeError:
			return Function(name = f'{self.name}-{other}', val = self.val - other, der = self.der)

	def __rsub__(self, other):
		# other - self
		try:
			return Function(name = f'{other.name}-{self.name}', val = other.val - self.val, der = other.der - self.der)
		except AttributeError:
			return Function(name = f'{other}-{self.name}', val = other - self.val, der = -self.der)

	def __mul__(self, other):
		# self * other
		try:
			return Function(name = f'{self.name}*{other.name}', val = self.val*other.val, der = self.der*other.val + self.val*other.der)
		except AttributeError:
			return Function(name = f'{self.name}*{other}', val = self.val*other, der = self.der*other)

	def __rmul__(self, other):
		# other * self
		try:
			return Function(name = f'{other.name}*{self.name}', val = self.val*other.val, der = self.der*other.val + self.val*other.der)
		except AttributeError:
			return Function(name = f'{other}*{self.name}', val = self.val*other, der = self.der*other)

	def __div__(self, other):
		# self/other
		try:
			return Function(name = f'{self.name}/{other.name}', val = self.val / other.val, der = (self.der*other.val - self.val*other.der)/(other.val**2))
		except AttributeError:
			return Function(name = f'{self.name}/{other}', val = self.val / other, der = self.der / other)

	def __rdiv__(self, other):
		# other/self
		try:
			return Function(name = f'{other.name}+{self.name}', val = other.val / self.val, der = (other.der*self.val - self.der*other.val)/(self.val**2))
		except AttributeError:
			return Function(name = f'{other}/{self.name}', val = other / self.val, der = - (self.der*other)/(self.val**2))

	def __neg__(self):
		# -self
		return Function(name = f'-({self.name})', val = -self.val, der = -self.der)

	def __pow__(self, other):
		# self ** other (exponentiation)
		# currently assumes that other is a scalar, not a Function object
		# I currently don't know how to implement the derivative of f ** g when f and g are both functions...
		if other == 0:
			return 1
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
		
def display_func(f, x):
	print(f'f(x = {x.val} = {f.val}')
	print(f'f\'(x = {x.val} = {f.der}')

def test_der(f, f_der):
	# tests whether f_der is actually the derivative of f

	# currently assumes that f is a Function object
	# allows for f_der to be a function or a scalar
	# distinguishes between exact equality, good approximation, and bad approximation


	# store in f_der_val whether f_der is a Function or a scalar
	try:
		f_der_val = f_der.val
	except AttributeError:
		f_der_val = f_der


	diff = abs(f.der - f_der_val)
	try:
		assert diff == 0
		print(' :) Passed deriv test')
	except AssertionError:
		precision = 1e-15
		if diff < precision:
			print('close enough : diff = ', diff)
		else:
			print('not good enough : diff = ', diff)


########### Demonstration ############
# create a variable with a name and a value
x = variable('x', np.pi/3)

# create any combination & composition of elementary functions
f = exp(-(sin(x) - cos(x))**2.0)

# here I manually write out this function's derivative to test it
f_der = exp(-(sin(x) - cos(x))**2.0)*(-2.0*(sin(x) - cos(x))*(cos(x) + sin(x)))

test_der(f, f_der)

############# Testing ################

def random_polynomial_test(x):
	# not necessary but i wrote it anyway out of curiosity
	# creates a giant random polynomial called p
	# then compares p.der with a manually-computed polynomial derivative p_der

	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~create p(x)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~						
	# choose a random polynomial degree in [1, 20)
	n = np.random.choice(np.arange(1,20))

	# generate random floats in (0, 100) for coefficients
	# size = n + 1 because there is 1 constant term and n poly_terms
	coefs = 100*np.random.random(size = n + 1)

	# the exponents for the polynomial are 0, ..., n
	exps = np.arange(n + 1)

	# the polynomial terms are the constant 1 followed by the function objects x, x**2, ..., x**n, because x**0 automatically evatuates to the scalar 1 instead of a Function object
	poly_terms = np.array([x**d for d in exps])

	# constructs the polynomial as a dot product, e.g., p(x) = 3 + 4*x + 5*x**2 can be re-written as np.dot([3, 4, 5], [1, x, x**2])
	p = np.dot(coefs, poly_terms)
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~create p'(x)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~							
	# ignore the first coefficient from the original polynomial because it represents the scalar term whose deriv is always 0
	deriv_coefs = x.der*((coefs * exps)[1:])

	# the exponents should be 0, ..., n - 1 for the derivative
	deriv_exps = exps[:-1]
	deriv_poly_terms = np.array([x**d for d in deriv_exps])

	# again use dot product to make the polynomial
	p_der = np.dot(deriv_coefs, deriv_poly_terms)
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	test_der(p, p_der)

print('Extra testing')
random_polynomial_test(x)
random_polynomial_test(f)
random_polynomial_test(sin(x))
random_polynomial_test(cos(x))
random_polynomial_test(tan(x))
random_polynomial_test(exp(x))
random_polynomial_test(log(x))















