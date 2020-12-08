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

	#### used internally
	val = 'val'
	der = 'der'
	double_der = 'double_der'

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
	def double_deriv_div(t1, t2, t3):
		#return 1
		if t2._trace_name != t3._trace_name:
			return -t3.val**0.5
		else:
			return 0
	def double_deriv_div_R(t1, t2, t3):
		#return 1
		if t2._trace_name != t3._trace_name:
			return 1
		else:
			return 0
	def double_deriv_power(t1, t2, t3):
		pass
	def double_deriv_sin(t1, t2, t3):
		pass
	def double_deriv_cos(t1, t2, t3):
		pass
	def double_deriv_tan(t1, t2, t3):
		pass
	def double_deriv_exp(t1, t2, t3):
		pass
	def double_deriv_log(t1, t2, t3):
		pass
	def double_deriv_sqrt(t1, t2, t3):
		pass
	def double_deriv_sigm(t1, t2, t3):
		pass
	def double_deriv_sinh(t1, t2, t3):
		pass
	def double_deriv_cosh(t1, t2, t3):
		pass
	def double_deriv_tanh(t1, t2, t3):
		pass
	def double_deriv_arcsin(t1, t2, t3):
		pass
	def double_deriv_arccos(t1, t2, t3):
		pass
	def double_deriv_arctan(t1, t2, t3):
		pass
	
	one_parent_rules = {
	# Contains the value and derivative rules for all one-parent operations
	# value rules must be (t, param) --> R
	# derivative rules must be (t, param) --> R, a partial derivative w.r.t. t
	# it's fine if the value or derivative rule doesnt reference the param, its to make the code more concise
		add: {
			val : lambda t, param : t.val+param, der : lambda t, param : 1.0, double_der : double_deriv_add}, 
		sub: {
			val : lambda t, param : t.val-param, der : lambda t, param : 1.0, double_der : double_deriv_sub},
		sub_R: {
			val : lambda t, param : param-t.val, der : lambda t, param : -1.0, double_der : double_deriv_sub_R},
		mul: {
			val : lambda t, param : t.val*param, der : lambda t, param : param, double_der : double_deriv_mul},
		div: {
			val : lambda t, param : t.val/param, der : lambda t, param : 1/param, double_der : double_deriv_div},
		div_R: {
			val : lambda t, param : param/t.val, der : lambda t, param : -param/((t._val)**2), double_der : double_deriv_div_R},
		power: {
			val : lambda t, param : t.val**param, der : lambda t, param : param*t.val**(param-1), double_der : double_deriv_power},
		sin: {
			val : lambda t, param : np.sin(t.val), der : lambda t, param : np.cos(t.val), double_der : double_deriv_sin},
		arcsin: {
			val: lambda t, param: np.arcsin(t.val), der: lambda t, param: 1/(np.sqrt(1-t.val**2)), double_der : double_deriv_arcsin},
		cos: {
			val : lambda t, param : np.cos(t.val), der : lambda t, param : -np.sin(t.val), double_der : double_deriv_cos},
		arccos: {
			val: lambda t, param: np.arccos(t.val), der: lambda t, param: -1/(np.sqrt(1-t.val**2)), double_der : double_deriv_arccos},
		tan: {
			val : lambda t, param : np.tan(t.val), der : lambda t, param : 1/(np.cos(t.val)**2), double_der : double_deriv_tan},
		arctan: {
			val: lambda t, param: np.arctan(t.val), der: lambda t, param: 1/(1+t.val**2), double_der : double_deriv_arctan},
		exp: {
			val : lambda t, param : np.power(param, t.val), der : lambda t, param : np.power(param, t.val)*np.log(param), double_der : double_deriv_exp},
		log: {
			val : lambda t, param : np.log(t.val)/np.log(param), der : lambda t, param : 1/(t.val*np.log(param)), double_der : double_deriv_log},
		sqrt: {
			val : lambda t, param : t.val**0.5, der : lambda t, param : 1/(2*t.val**0.5), double_der : double_deriv_sqrt},
		sigm: {
			val : lambda t, param : 1/(1 + np.exp(-t.val)), der : lambda t, param : param*t.val**(param-1), double_der : double_deriv_sigm},
		sinh: {
			val : lambda t, param : (np.exp(t.val) - np.exp(-t.val))/2, der : lambda t, param : (np.exp(t.val) + np.exp(-t.val))/2, double_der : double_deriv_sinh},
		cosh: {
			val : lambda t, param : (np.exp(t.val) + np.exp(-t.val))/2, der : lambda t, param : (np.exp(t.val) - np.exp(-t.val))/2, double_der : double_deriv_cosh},
		tanh: {
			val : lambda t, param : (np.exp(t.val) - np.exp(-t.val))/(np.exp(t.val) + np.exp(-t.val)), der : lambda t, param : 4/((np.exp(t.val) + np.exp(-t.val))**2), double_der : double_deriv_tanh}
	}

	two_parent_rules = {
	# Contains the value and derivative rules for all two-parent operations
	# value rules must be (t1, t2) --> R
	# derivative rules must be (t1, t2) --> R2, a partial derivative w.r.t. t1 and a partial derivative w.r.t. t2
		add : {
			val : lambda t1, t2 : t1.val + t2.val, der : lambda t1, t2 : (1.0, 1.0), double_der : double_deriv_add},
		sub : {
			val : lambda t1, t2 : t1.val - t2.val, der : lambda t1, t2 : (1.0, -1.0), double_der : double_deriv_sub},
		mul : {
			val : lambda t1, t2 : t1.val * t2.val, der : lambda t1, t2 : (t2.val, t1.val), double_der : double_deriv_mul},
		div : {
			val : lambda t1, t2 : t1.val / t2.val, der : lambda t1, t2 : (1/t2.val, -t1.val/(t2.val**2)), double_der : double_deriv_div},
		power : {
			val : lambda t1, t2 : t1.val ** t2.val, der : lambda t1, t2 : (t2.val*(t1.val**(t2.val-1)), (t1.val**t2.val)*np.log(t1.val)), double_der : double_deriv_power}
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
	Returns double derivative of t1 w.r.t. both t2 and t3

	e.g. d^2f/dxdy or d^2g/dz^2
	'''
	return Ops.one_parent_rules[t1._op][Ops.double_der](t1, t2, t3)







