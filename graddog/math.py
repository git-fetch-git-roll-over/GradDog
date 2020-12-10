# :)
import numpy as np


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
	in_domain = 'in_domain'


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
		add: {
			val : lambda t, param : t+param, 
			der : lambda t, param : 1.0, 
			double_der : lambda t, param : 0}, 
		sub: {
			val : lambda t, param : t-param, 
			der : lambda t, param : 1.0, 
			double_der : lambda t, param : 0},
		sub_R: {
			val : lambda t, param : param-t, 
			der : lambda t, param : -1.0, 
			double_der : lambda t, param : 0},
		mul: {
			val : lambda t, param : t*param, 
			der : lambda t, param : param, 
			double_der : lambda t, param : 0},
		div: {
			in_domain: lambda t, param: param != 0, 
			val : lambda t, param : t/param, 
			der : lambda t, param : 1/param,
			double_der : lambda t, param : 0},
		div_R: {
			in_domain: lambda t, param: t != 0, 
			val : lambda t, param : param/t, 
			der : lambda t, param : -param/(t**2),
			double_der : lambda t, param : 2*param/(t**3)},
		power: {
			val : lambda t, param : t**param, 
			der : lambda t, param : param*t**(param-1), 
			double_der : lambda t, param : param*(param - 1)*(t**(param-2))},
		sin: {
			val : lambda t, param : np.sin(t), 
			der : lambda t, param : np.cos(t),
			double_der : lambda t, param : -np.sin(t)},
		arcsin: {
			in_domain: lambda t, param: t >= -1 and t <= 1, 
			val: lambda t, param: np.arcsin(t), 
			der: lambda t, param: 1/(np.sqrt(1-t**2))},
		cos: {
			val : lambda t, param : np.cos(t), 
			der : lambda t, param : -np.sin(t),
			double_der : lambda t, param : -np.cos(t)},
		arccos: {
			in_domain: lambda t, param: t >= -1 and t <= 1, 
			val: lambda t, param: np.arccos(t), 
			der: lambda t, param: -1/(np.sqrt(1-t**2))},
		tan: {
			val : lambda t, param : np.tan(t), 
			der : lambda t, param : 1/(np.cos(t)**2)},
		arctan: {
			val: lambda t, param: np.arctan(t), 
			der: lambda t, param: 1/(1+t**2)},
		exp: {
			val : lambda t, param : np.power(param, t), 
			der : lambda t, param : np.power(param, t)*np.log(param), 
			double_der : lambda t, param : np.power(param, t)*np.log(param)*np.log(param)},
		log: {
			in_domain: lambda t, param: t > 0 and param > 0, 
			val : lambda t, param : np.log(t)/np.log(param), 
			der : lambda t, param : 1/(np.log(param)*t),
			double_der : lambda t, param : -1/(np.log(param)*t**2)},
		sqrt: {
			in_domain: lambda t, param: t >= 0, 
			val : lambda t, param : t**0.5, 
			der : lambda t, param : 1/(2*t**0.5), 
			double_der : lambda t, param : -1/(4*t**1.5)},
		sigm: {
			val : lambda t, param : 1/(1 + np.exp(-t)), 
			der : lambda t, param : np.exp(-t)/((1+np.exp(t))**2)},
		sinh: {
			val : lambda t, param : (np.exp(t) - np.exp(-t))/2, 
			der : lambda t, param : (np.exp(t) + np.exp(-t))/2},
		cosh: {
			val : lambda t, param : (np.exp(t) + np.exp(-t))/2, 
			der : lambda t, param : (np.exp(t) - np.exp(-t))/2},
		tanh: {
			val : lambda t, param : (np.exp(t) - np.exp(-t))/(np.exp(t) + np.exp(-t)), 
			der : lambda t, param : 4/((np.exp(t) + np.exp(-t))**2)}
	}

	two_parent_rules = {
		add : {
			val : lambda t1, t2 : t1 + t2, 
			der : lambda t1, t2 : (1.0, 1.0), 
			double_der : lambda t1, t2 : np.array([[0, 0],[0, 0]])},
		sub : {
			val : lambda t1, t2 : t1 - t2, 
			der : lambda t1, t2 : (1.0, -1.0), 
			double_der : lambda t1, t2 : np.array([[0, 0],[0, 0]])},
		mul : {
			val : lambda t1, t2 : t1 * t2, 
			der : lambda t1, t2 : (t2, t1), 
			double_der : lambda t1, t2 : np.array([[0, 1],[1, 0]])},
		div : {
			val : lambda t1, t2 : t1 / t2, 
			der : lambda t1, t2 : (1/t2, -t1/(t2**2)), 
			double_der : lambda t1, t2 : np.array([[0, -1/(t2**2)],[-1/(t2**2), 2*t1/(t2**3)]])},
		power : {
			val : lambda t1, t2 : t1 ** t2, 
			der : lambda t1, t2 : (t2*(t1**(t2-1)), (t1**t2)*np.log(t1)), 
			double_der : lambda t1, t2 : np.array([[t2*(t2-1)*(t1**(t2-2)), (t1**(t2-1))*(t2*np.log(t1) + 1)],[(t1**(t2-1))*(t2*np.log(t1) + 1), np.log(t1)*np.log(t1)*t1**t2]])}

	}

def deriv_one_parent(t, op, param = None):
	# derivative of a trace with one parent
	try:
		d_op_dt = Ops.one_parent_rules[op]['der'](t.val, param)
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

	References the parent traces of t1 because the order of the arguments matters
	'''
	parents = t1._parents
	if len(parents) == 1:
		return new_double_deriv_one_parent(t2, t1._op, t1._param)
	else:
		double_deriv = new_double_deriv_two_parents(parents[0], t1._op, parents[1])
		if t2._trace_name != t3._trace_name:
			return double_deriv[0,1]
		else:
			if t2._trace_name == parents[0]._trace_name:
				return double_deriv[0,0]
			else:
				return double_deriv[1,1]


def new_double_deriv_one_parent(t, op, param = None):
	try:
		return Ops.one_parent_rules[op][Ops.double_der](t.val, param)
	except KeyError:
		raise ValueError(f'Double derivative currently not implemented for operation {op}')

def new_double_deriv_two_parents(t1, op, t2):
	try:
		return Ops.two_parent_rules[op][Ops.double_der](t1.val, t2.val)
	except KeyError:
		raise ValueError(f'Double derivative currently not implemented for operation {op}')

def in_domain(t, op, param = None):
	return Ops.one_parent_rules[op][Ops.in_domain](t, param)






