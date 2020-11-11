# Start writing code here...
# here goes nothin


'''

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

#first write a function to produce a trace


#should be implemented as a trace table
class Function:

	def __init__(self, name, val, left = None, right = None, op = None, der = 1):
		self.name = name
		self.val = val
		self.left = left
		self.right = right
		self.op = op
		self.der = der

	def __repr__(self):
		return self.name

	def __radd__(self, other):

		try:

			new_f = Function(name = other.name+'+'+self.name, val = self.val + other.val, left = other, right = self, op = '+', der = self.der + other.der)
		
		except AttributeError:

			new_f = Function(name = f'+{other}'+self.name, val = self.val + other, left = other, right = self, op = '+', der = self.der)

		return new_f

	def __add__(self, other):

		try:

			new_f = Function(name = self.name+'+'+other.name, val = self.val + other.val, left = other, right = self, op = '+', der = self.der + other.der)
		
		except AttributeError:

			new_f = Function(name = self.name+f'+{other}', val = self.val + other, left = other, right = self, op = '+', der = self.der)

		return new_f

	def __rmul__(self, other):

		try:

			new_f = Function(name = other.name+'*'+self.name, val = self.val*other.val, left = other, right = self, op = '*', der = self.der*other.val + other.der*self.val)
		
		except AttributeError:

			new_f = Function(name = f'{other}*'+self.name, val = self.val*other, left = other, right = self, op = '*', der = self.der*other)

		return new_f

	def __mul__(self, other):

		try:

			new_f = Function(name = self.name+'*'+other.name, val = self.val*other.val, left = other, right = self, op = '*', der = self.der*other.val + other.der*self.val)
		
		except AttributeError:

			new_f = Function(name = self.name+f'*{other}', val = self.val*other, left = other, right = self, op = '*', der = self.der*other)

		return new_f

	def __pow__(self, other):
		
		new_f = Function(name = f'({self.name})**{other}', val = self.val**other, op = '**', der = other*self.val**(other - 1)*self.der)
		return new_f

	def eval(self, val):

		if self.op == '+':
			try:
				l_eval = self.left.eval(val)
			except AttributeError:
				l_eval = self.left

			try:
				r_eval = self.right.eval(val)
			except AttributeError:
				r_eval = self.right

			return l_eval + r_eval
		elif self.op == '*':
			try:
				l_eval = self.left.eval(val)
			except AttributeError:
				l_eval = self.left

			try:
				r_eval = self.right.eval(val)
			except AttributeError:
				r_eval = self.right

			return l_eval * r_eval

		else:
			return val

class Sine(Function):
    def __init__(self, name, val):
        super().__init__(name, val)
        # Can we modify this to return a different name
        self.der = Cosine(name, val)

class Cosine(Function):
    def __init__(self, name, val):
        super().__init__(name, val)
        self.der = -1 * Sine(name, val)


class Variable(Function):

	def __init__(self, name, val):
		super().__init__(name, val)



x = Variable('x', 10)
#y = Variable('y')

a, b, n = 6, 2, 3

f = a*x**n + b
#s = Sine('sin', x)
#print(s.der)

print(f.name, f.val, f.der)

