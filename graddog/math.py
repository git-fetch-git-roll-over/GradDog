# :)
import numpy as np

# TODO: convert lambda expressions to closure functions if appropriate...

# TODO: How do we distinguish that this file is meant for internal use and not for the user?

class Ops:

	# all of the math operations currently implemented
	# include a value and derivative rule for each operation
	# can include UNARY operators (with or without a numerical parameters) which are called 'one_parent'
	# can include BINARY operators which are called 'two_parent'

	# _R denotes that the operator is applied with a trace to the RIGHT
	# certain _R operators like __radd__ and __rmul__ actually have no difference from __add__ and __mul__
	# but __rsub__ and __rdiv__, for example, have different derivatives than __sub__ and __div__

	val = 'val'
	der = 'der'

	add = '+'
	sub = '-'
	sub_R = '-R'
	mul = '*'
	div = '/'
	div_R = '/R'
	power = '^'
	sin = 'sin'
	cos = 'cos'
	tan = 'tan'
	exp = 'exp'
	log = 'log'
	sqrt = 'sqrt' 
	sigm = 'sigm' 
	sinh = 'sinh'
	cosh = 'cosh'
	tanh = 'tanh'
	arcsin = 'arcsin'
	arccos = 'arccos'
	arctan = 'arctan'
	
	one_parent_rules = {
	# value rules must be (t, param) --> R
	# derivative rules must be (t, param) --> R, a partial derivative w.r.t. t
	# it's fine if the value or derivative rule doesnt reference the param, its to make the code more concise
		add: {
			val : lambda t, param : t.val+param, der : lambda t, param : 1.0},
		sub: {
			val : lambda t, param : t.val-param, der : lambda t, param : 1.0},
		sub_R: {
			val : lambda t, param : param-t.val, der : lambda t, param : -1.0},
		mul: {
			val : lambda t, param : t.val*param, der : lambda t, param : param},
		div: {
			val : lambda t, param : t.val/param, der : lambda t, param : 1/param},
		div_R: {
			val : lambda t, param : param/t.val, der : lambda t, param : -param/((t._val)**2)},
		power: {
			val : lambda t, param : t.val**param, der : lambda t, param : param*t.val**(param-1)},
		sin: {
			val : lambda t, param : np.sin(t.val), der : lambda t, param : np.cos(t.val)},
		arcsin: {
			val: lambda t, param: np.arcsin(t.val), der: lambda t, param: 1/(np.sqrt(1-t.val**2))},
		cos: {
			val : lambda t, param : np.cos(t.val), der : lambda t, param : -np.sin(t.val)},
		arccos: {
			val: lambda t, param: np.arccos(t.val), der: lambda t, param: -1/(np.sqrt(1-t.val**2))},
		tan: {
			val : lambda t, param : np.tan(t.val), der : lambda t, param : 1/(np.cos(t.val)**2)},
		arctan: {
			val: lambda t, param: np.arctan(t.val), der: lambda t, param: 1/(1+t.val**2)},
		exp: {
			val : lambda t, param : np.power(param, t.val), der : lambda t, param : np.power(param, t.val)*np.log(param)},
		log: {
			val : lambda t, param : np.log(t.val)/np.log(param), der : lambda t, param : 1/(t.val*np.log(param))},
		sqrt: {
			val : lambda t, param : t.val**0.5, der : lambda t, param : 1/(2*t.val**0.5)},
		sigm: {
			val : lambda t, param : 1/(1 + np.exp(-t.val)), der : lambda t, param : param*t.val**(param-1)},
		sinh: {
			val : lambda t, param : (np.exp(t.val) - np.exp(-t.val))/2, der : lambda t, param : (np.exp(t.val) + np.exp(-t.val))/2},
		cosh: {
			val : lambda t, param : (np.exp(t.val) + np.exp(-t.val))/2, der : lambda t, param : (np.exp(t.val) - np.exp(-t.val))/2},
		tanh: {
			val : lambda t, param : (np.exp(t.val) - np.exp(-t.val))/(np.exp(t.val) + np.exp(-t.val)), der : lambda t, param : 4/((np.exp(t.val) + np.exp(-t.val))**2)}
	}

	two_parent_rules = {
	# value rules must be (t1, t2) --> R
	# derivative rules must be (t1, t2) --> R2, a partial derivative w.r.t. t1 and a partial derivative w.r.t. t2
		add : {
			val : lambda t1, t2 : t1.val + t2.val, der : lambda t1, t2 : (1.0, 1.0)},
		sub : {
			val : lambda t1, t2 : t1.val - t2.val, der : lambda t1, t2 : (1.0, -1.0)},
		mul : {
			val : lambda t1, t2 : t1.val * t2.val, der : lambda t1, t2 : (t2.val, t1.val)},
		div : {
			val : lambda t1, t2 : t1.val / t2.val, der : lambda t1, t2 : (1/t2.val, -t1.val/(t2.val**2))},
		power : {
			val : lambda t1, t2 : t1.val ** t2.val, der : lambda t1, t2 : (t2.val*(t1.val**(t2.val-1)), (t1.val**t2.val)*np.log(t1.val))}
	}

	def double_deriv_add(t1, t2, t3):
		return 0
	def double_deriv_sub(t1, t2, t3):
		return 0
	def double_deriv_sub_R(t1, t2, t3):
		return 0
	def double_deriv_mul(t1, t2, t3):
		#return 1
		if t2._trace_name != t3._trace_name:
			return 1
		else:
			return 0
	def double_deriv_exp(t1, t2, t3):
		return np.exp(t2.val)

	double_der_rules = {
	# double derivative rules must be (t1, t2, t3) --> R
		add: double_deriv_add, 
		sub: double_deriv_sub,
		sub_R: double_deriv_sub_R,
		mul: double_deriv_mul,
		exp: double_deriv_exp
	}



def deriv_one_parent(t, op, param = None):
	# derivative of a trace with one parent
	try:
		d_op_dt = Ops.one_parent_rules[op]['der'](t, param)
		return {t._trace_name : d_op_dt}
	except KeyError:
		raise ValueError('need to implement operation', op)

def deriv_two_parents(t1, op, t2):
	# derivative of a trace with two parents
	try:
		d_op_dt1, d_op_dt2 = Ops.two_parent_rules[op]['der'](t1, t2)
		return {t1._trace_name : d_op_dt1, t2._trace_name : d_op_dt2}
	except KeyError:
		raise ValueError('need to implement operation', op)

def val_one_parent(t, op, param = None):
	# value of a trace with one parent
	try:
		return Ops.one_parent_rules[op]['val'](t, param)
	except KeyError:
		raise ValueError('need to implement operation', op)

def val_two_parents(t1, op, t2):
	# value of a trace with two parents
	try:
		return Ops.two_parent_rules[op]['val'](t1, t2)
	except KeyError:
		raise ValueError('need to implement operation', op)

def deriv(t, op, other = None):
	if other is None:
		return deriv_one_parent(t, op)
	try:
		# if other is a trace
		other_val = other.val
		return deriv_two_parents(t, op, other)
	except AttributeError:
		# if other is a param, AKA just a number
		return deriv_one_parent(t, op, other)

def val(t, op, other = None):
	if other is None:
		return val_one_parent(t, op)
	try:
		# if other is a trace it has a cal
		# causes an AttributeError to force val_1 to get called instead :P 
		# congrats you have found a lazy thing i have done to avoid solving the problem for good :)
		other_val = other.val
		return val_two_parents(t, op, other)
	except AttributeError:
		# if other is a param, AKA just a number
		return val_one_parent(t, op, other)

def double_deriv(t1, t2, t3):
	'''
	Returns double derivative of t1 w.r.t. t2 w.r.t. t3

	e.g. d^2f/dxdy or d^2g/dz^2
	'''
	return Ops.double_der_rules[t1._op](t1, t2, t3)







