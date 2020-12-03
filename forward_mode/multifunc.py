import numpy as np
from forward_mode.trace import Variable
from forward_mode.functions import sin, cos, tan, exp, log
from functools import reduce

class MultiFunc:

	def __init__(self, funcs):
		self.funcs = funcs

		# use the map and reduce functions to combine
		# all of the variables from all the functions in funcs
		# into a single sorted list called total_vars
		self.total_vars = sorted(reduce(lambda s, t : s.union(t), list(map(lambda f : set(f.der.keys()), funcs))))
		
		self.calculate_jacobian()
	
	def calculate_jacobian(self):
		N = len(self.total_vars)
		M = len(self.funcs)

		j = np.zeros(shape = (M, N))

		for m in range(M):
			der = self.funcs[m].der
			for n in range(N):
				x = self.total_vars[n]
				try:
					j[m,n] = der[x]
				except KeyError:
					# a key error means that x is not a variable defined within
					# the scope of the current function self.funcs[m]
					# therefore, the derivative of that function
					# with respect to x is zero
					j[m,n] = 0
		self.jacobian = j
					






