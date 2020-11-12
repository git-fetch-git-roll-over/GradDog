# Start writing code here...
# here goes nothin


'''

TODO:
sin, cos, tan, exp, log
generator that yields multiple variables


# from GradDog.functions import exp, sin, cos, variables

# x, y = variables(2)
# #exp, sin, cos are all instances of the Function base class

# f = exp(-(sin(x) - cos(y))^2)








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

	def __init__(self, params):#name, val, upper, left = None, right = None, op = None, der = 1):
		self.name = params['name']
		self.val = params['val']
		self.der = params['der']

	def __repr__(self):
		return self.name

	def __add__(self, other):
		# self + other

		try:

			new_name = self.name+'+'+other.name
			new_val = self.val + other.val
			new_der = self.der + other.der
	
		except AttributeError:

			new_name = self.name + f'+{other}'
			new_val = self.val + other
			new_der = self.der

		params = {
			'name' : new_name,
			'val' : new_val,
			'der' : new_der
		}

		return Function(params)

	def __radd__(self, other):
		# other + self

		try:

			new_name = other.name+'+'+self.name
			new_val = self.val + other.val
			new_der = self.der + other.der
	
		except AttributeError:

			new_name = f'{other}+'+self.name
			new_val = self.val + other
			new_der = self.der

		params = {
			'name' : new_name,
			'val' : new_val,
			'der' : new_der
		}

		return Function(params)

	def __sub__(self, other):
		# self - other

		try:

			new_name = self.name+'-'+other.name
			new_val = self.val - other.val
			new_der = self.der - other.der
	
		except AttributeError:

			new_name = self.name + f'-{other}'
			new_val = self.val - other
			new_der = self.der

		params = {
			'name' : new_name,
			'val' : new_val,
			'der' : new_der
		}

		return Function(params)

	def __rsub__(self, other):
		# other - self

		try:

			new_name = other.name+'-'+self.name
			new_val = other.val - self.val
			new_der = other.der - self.der
	
		except AttributeError:

			new_name = f'{other}-'+self.name
			new_val = other - self.val
			new_der = -self.der

		params = {
			'name' : new_name,
			'val' : new_val,
			'der' : new_der
		}

		return Function(params)

	def __mul__(self, other):
		# self * other

		try:

			new_name = self.name+'*'+other.name
			new_val = self.val*other.val
			new_der = self.der*other.val + self.val*other.der
	
		except AttributeError:

			new_name = self.name + f'*{other}'
			new_val = self.val*other
			new_der = self.der*other

		params = {
			'name' : new_name,
			'val' : new_val,
			'der' : new_der
		}

		return Function(params)

	def __rmul__(self, other):
		# other * self

		try:

			new_name = other.name+'*'+self.name
			new_val = self.val * other.val
			new_der = self.der*other.val + self.val*other.der
	
		except AttributeError:

			new_name = f'{other}*' + self.name
			new_val = self.val * other
			new_der = self.der * other

		params = {
			'name' : new_name,
			'val' : new_val,
			'der' : new_der
		}

		return Function(params)

	def __div__(self, other):
		# self/other

		try:

			new_name = self.name+'/'+other.name
			new_val = self.val / other.val
			new_der = (self.der*other.val - self.val*other.der)/(other.val**2)
	
		except AttributeError:

			new_name = self.name + f'/{other}'
			new_val = self.val / other
			new_der = self.der / other

		params = {
			'name' : new_name,
			'val' : new_val,
			'der' : new_der
		}

		return Function(params)

	def __rdiv__(self, other):
		# other/self

		try:

			new_name = other.name+'/'+self.name
			new_val = other.val / self.val
			new_der = (other.der*self.val - self.der*other.val)/(self.val**2)
	
		except AttributeError:

			new_name = f'{other}/' + self.name
			new_val = other / self.val
			new_der = - (self.der*other)/(self.val**2)

		params = {
			'name' : new_name,
			'val' : new_val,
			'der' : new_der
		}

		return Function(params)

	def __neg__(self):
		
		new_name = f'-({self.name})'
		new_val = -self.val
		new_der = -self.der

		params = {
			'name' : new_name,
			'val' : new_val,
			'der' : new_der
		}

		return Function(params)

	def __pow__(self, other):
		#currently assumes that other is a scalar, not Function object
		
		new_name = f'({self.name})**{other}'
		new_val = self.val ** other
		new_der = other*self.val**(other - 1)*self.der

		params = {
			'name' : new_name,
			'val' : new_val,
			'der' : new_der
		}

		return Function(params)


class Variable(Function):

	def __init__(self, name, val):
		params = {
			'name' : name,
			'val' : val,
			'der' : 1
		}
		super().__init__(params)


def sin(x):
	params = {
		'name' : f'sin({x})',
		'val' : np.sin(x.val),
		'der' : np.cos(x.val)*x.der
		}
	return Function(params)

def cos(x):
	params = {
		'name' : f'cos({x})',
		'val' : np.cos(x.val),
		'der' : -np.sin(x.val)*x.der
		}
	return Function(params)

def tan(x):
	params = {
		'name' : f'tan({x})',
		'val' : np.tan(x.val),
		'der' : x.der/(np.cos(x.val)**2)
		}
	return Function(params)

def exp(x):
	params = {
		'name' : f'exp({x})',
		'val' : np.exp(x.val),
		'der' : np.exp(x.val)*x.der
		}
	return Function(params)

def log(x):
	params = {
		'name' : f'log({x})',
		'val' : np.log(x.val),
		'der' : x.der/x.val
		}
	return Function(params)
		
def display_func(f):
	print(f, 'evaluated at x = ', x.val)
	print(f.val)
	print('Derivative:', f.der)

def test_der(f, f_der):

	try:
		check = (f.der == f_der.val)
	except AttributeError:
		check = (f.der == f_der)
	
	assert check



def random_polynomial_test(x):

	#efficient numpy random number generator
	generator = np.random.default_rng()

	#choose a random polynomial degree in [1, 20)
	n = generator.choice(np.arange(1,20))

	#generate random floats in (0, 100) for coefficients
	#size = n + 1 because there is 1 constant term and n poly_terms
	coefs = 100*generator.random(size = n + 1)
	exps = np.arange(1, n + 1)

	poly_terms = np.array([1] + [x**d for d in exps])
	#constructs the polynomial as a dot product
	#e.g.
	# 3 + 4*x + 5*x**2 = np.dot([3, 4, 5], [1, x, x**2])
	p = np.dot(coefs, poly_terms)


	
	#calculate the new coefficients for the derivative
	#ignore the first coefficient from the original polynomial because it represents the scalar term whose deriv is always 0
	deriv_coefs = (coefs[1:] * exps)*x.der

	deriv_exps = exps[:-1]
	deriv_poly_terms = np.array([1] + [x**d for d in deriv_exps])

	#again use dot product to make the polynomial
	p_der = np.dot(deriv_coefs, deriv_poly_terms)

	#use the above testing function
	#which uses an assert statement to make sure that p_der contains the value of the derivative of p
	test_der(p, p_der)



x = Variable('x', np.pi/3)


f = exp(-(sin(x) - cos(x))**2.0)

#manually write out this function's derivative to test it
f_der = exp(-(sin(x) - cos(x))**2.0)*(-2.0*(sin(x) - cos(x))*(cos(x) + sin(x)))

test_der(f, f_der)

display_func(f)




