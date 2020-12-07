import numbers
from graddog.trace import Trace
#from graddog.functions import VectorFunction
import numpy as np
from graddog.compgraph import CompGraph

# TODO: properly format the test that val is a number


class Variable(Trace):

	def __init__(self, name, val):
		# by default, the derivative of a variable with respect to itself is 1.0
		# the parents of a variable is an emptylist

		# if not numbers.isinstance(val):
		# 	raise TypeError('val needs to be a number')
		super().__init__(name, val, {name : 1.0}, [])
		self._name = name
