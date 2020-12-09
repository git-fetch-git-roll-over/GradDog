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
		if t2._trace_name != t3._trace_name:
			return 1
		else:
			return 0
	def double_deriv_div(t1, t2, t3):
		if t2._trace_name != t3._trace_name:
			return -t3.val**-2
		else:
			return 0
	def double_deriv_div_R(t1, t2, t3):
		if t2._trace_name != t3._trace_name:
			return 1
		else:
			return 0
	def double_deriv_power(t1, t2, t3):
		if t2._trace_name != t3._trace_name:
			return (t2.val**(t3.val-1))*(t3.val*np.log(t2.val) + 1)
		else:
			return t3.val*(t3.val-1)*(t2.val**(t3.val-2))
	def double_deriv_sin(t1, t2, t3):
		return -np.sin(t2.val)
	def double_deriv_cos(t1, t2, t3):
		return -np.cos(t2.val)
	def double_deriv_tan(t1, t2, t3):
		return 2*np.sin(t2.val)/(np.cos(t2.val)**3)
	def double_deriv_exp(t1, t2, t3):
		return np.array([[np.exp(t2.val)]])
	def double_deriv_log(t1, t2, t3):
		return -t2.val**-2
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
			val : lambda t, param : t+param, der : lambda t, param : 1.0, double_der : lambda t, param : np.array([[0]])}, 
		sub: {
			val : lambda t, param : t-param, der : lambda t, param : 1.0, double_der : double_deriv_sub},
		sub_R: {
			val : lambda t, param : param-t, der : lambda t, param : -1.0, double_der : double_deriv_sub_R},
		mul: {
			val : lambda t, param : t*param, der : lambda t, param : param, double_der : lambda t, param : np.array([[0]])},
		div: {
			val : lambda t, param : t/param, der : lambda t, param : 1/param, double_der : double_deriv_div},
		div_R: {
			val : lambda t, param : param/t, der : lambda t, param : -param/((t._val)**2), double_der : double_deriv_div_R},
		power: {
			val : lambda t, param : t**param, der : lambda t, param : param*t**(param-1), double_der : lambda t, param : np.array([[param*(param - 1)*(t**(param-2))]])},
		sin: {
			val : lambda t, param : np.sin(t), der : lambda t, param : np.cos(t), double_der : double_deriv_sin},
		arcsin: {
			val: lambda t, param: np.arcsin(t), der: lambda t, param: 1/(np.sqrt(1-t**2)), double_der : double_deriv_arcsin},
		cos: {
			val : lambda t, param : np.cos(t), der : lambda t, param : -np.sin(t), double_der : double_deriv_cos},
		arccos: {
			val: lambda t, param: np.arccos(t), der: lambda t, param: -1/(np.sqrt(1-t**2)), double_der : double_deriv_arccos},
		tan: {
			val : lambda t, param : np.tan(t), der : lambda t, param : 1/(np.cos(t)**2), double_der : double_deriv_tan},
		arctan: {
			val: lambda t, param: np.arctan(t), der: lambda t, param: 1/(1+t**2), double_der : double_deriv_arctan},
		exp: {
			val : lambda t, param : np.power(param, t), der : lambda t, param : np.power(param, t)*np.log(param), double_der : lambda t, param : np.array([[np.power(param, t)*np.log(param)*np.log(param)]])},
		log: {
			val : lambda t, param : np.log(t)/np.log(param), der : lambda t, param : 1/(t*np.log(param)), double_der : double_deriv_log},
		sqrt: {
			val : lambda t, param : t**0.5, der : lambda t, param : 1/(2*t**0.5), double_der : double_deriv_sqrt},
		sigm: {
			val : lambda t, param : 1/(1 + np.exp(-t)), der : lambda t, param : np.exp(-t)/((1+np.exp(t))**2), double_der : double_deriv_sigm},
		sinh: {
			val : lambda t, param : (np.exp(t) - np.exp(-t))/2, der : lambda t, param : (np.exp(t) + np.exp(-t))/2, double_der : double_deriv_sinh},
		cosh: {
			val : lambda t, param : (np.exp(t) + np.exp(-t))/2, der : lambda t, param : (np.exp(t) - np.exp(-t))/2, double_der : double_deriv_cosh},
		tanh: {
			val : lambda t, param : (np.exp(t) - np.exp(-t))/(np.exp(t) + np.exp(-t)), der : lambda t, param : 4/((np.exp(t) + np.exp(-t))**2), double_der : double_deriv_tanh}
	}

	two_parent_rules = {
	# Contains the value and derivative rules for all two-parent operations
	# value rules must be (t1, t2) --> R
	# derivative rules must be (t1, t2) --> R2, a partial derivative w.r.t. t1 and a partial derivative w.r.t. t2
		add : {
			val : lambda t1, t2 : t1 + t2, der : lambda t1, t2 : (1.0, 1.0), double_der : lambda t1, t2 : np.array([[0, 0],[0, 0]])},
		sub : {
			val : lambda t1, t2 : t1 - t2, der : lambda t1, t2 : (1.0, -1.0), double_der : lambda t1, t2 : np.array([[0, 0],[0, 0]])},
		mul : {
			val : lambda t1, t2 : t1 * t2, der : lambda t1, t2 : (t2, t1), double_der : lambda t1, t2 : np.array([[0, 1],[1, 0]])},
		div : {
			val : lambda t1, t2 : t1 / t2, der : lambda t1, t2 : (1/t2, -t1/(t2**2)), double_der : lambda t1, t2 : np.array([[0, -t2**-2],[-t2**-2, 2*t1*t2**-3]])},
		power : {
			val : lambda t1, t2 : t1 ** t2, der : lambda t1, t2 : (t2*(t1**(t2-1)), (t1**t2)*np.log(t1)), double_der : lambda t1, t2 : np.array([[t2*(t2-1)*(t1**(t2-2)), (t1**(t2-1))*(t2*np.log(t1) + 1)],[(t1**(t2-1))*(t2*np.log(t1) + 1), np.log(t1)*np.log(t1)*t1**t2]])}

	}

def deriv_one_parent(t, op, param = None):
	# derivative of a trace with one parent
	try:
		# t is a trace
		d_op_dt = Ops.one_parent_rules[op]['der'](t.val, param)
		return {t._trace_name : d_op_dt}
	except AttributeError:
		# t is a scalar
		d_op_dt = Ops.one_parent_rules[op]['der'](t, param)
		return {t._trace_name : d_op_dt}
	except KeyError:
		raise ValueError('need to implement operation', op)

def deriv_two_parents(t1, op, t2):
	# derivative of a trace with two parents
	try:
		d_op_dt1, d_op_dt2 = Ops.two_parent_rules[op]['der'](t1.val, t2.val)
		return {t1._trace_name : d_op_dt1, t2._trace_name : d_op_dt2}
	except KeyError:
		raise ValueError('need to implement operation', op)

def val_one_parent(t, op, param = None):
	# value of a trace with one parent and optional scalar parameter
	try:
		# t is a trace
		return Ops.one_parent_rules[op]['val'](t.val, param)
	except AttributeError:
		# t is a scalar
		return Ops.one_parent_rules[op]['val'](t, param)
	except KeyError:
		raise ValueError('need to implement operation', op)


def val_two_parents(t1, op, t2):
	# value of a trace with two parents
	try:
		return Ops.two_parent_rules[op]['val'](t1.val, t2.val)
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
		# if other is a scalar
		return deriv_one_parent(t, op, other)

def val(t, op, other = None):
	if other is None:
		return val_one_parent(t, op)
	try:
		# if other is a trace
		other_val = other.val
		return val_two_parents(t, op, other)
	except AttributeError:
		# if other is a scalar
		return val_one_parent(t, op, other)


def double_deriv(t1, t2, t3):
	'''
	Returns double derivative of t1 w.r.t. both t2 and t3

	e.g. d^2f/dxdy or d^2g/dz^2
	'''
	return Ops.one_parent_rules[t1._op][Ops.double_der](t1, t2, t3)

def new_double_deriv_one_parent(t, op, param = None):
	return Ops.one_parent_rules[op][Ops.double_der](t.val, param)

def new_double_deriv_two_parents(t1, op, t2):
	return Ops.two_parent_rules[op][Ops.double_der](t1.val, t2.val)






