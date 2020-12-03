# :)
import numpy as np

class Ops:

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
	const_exp=  '^'
	const_exp_R = '^R'
	const_mul = '*'
	const_div = '/'
	const_divR = '/R'
	const_add = '+'
	const_sub = '-'
	const_subR = '-R'

	deriv_rules = {
	'sin' : lambda t, param : np.cos(t.val),
	'cos' : lambda t, param : -np.sin(t.val),
	'tan' : lambda t, param : 1/(np.cos(t.val)**2),
	'exp' : lambda t, param : np.power(param, t.val)*np.log(param),
	'log' : lambda t, param : 1/(t.val*np.log(param)),
	'sqrt' : lambda t, param : 1/(2*t.val**0.5),
	'sigm' : lambda t, param : np.exp(-t.val)/((1 + np.exp(-t.val))**2),
	'sinh' : lambda t, param : (np.exp(t.val) + np.exp(-t.val))/2,
	'cosh' : lambda t, param : (np.exp(t.val) - np.exp(-t.val))/2,
	'tanh' : lambda t, param : 4/((np.exp(t.val) + np.exp(-t.val))**2),
	'^' : lambda t, param : param*t._val**(param-1),
	'^R' : lambda t, param : param ** t._val * np.log(param),
	'*' : lambda t, param : param,
	'/' : lambda t, param : 1/param,
	'/R' : lambda t, param : -param / ((t._val)**2),
	'+' : lambda t, param : 1,
	'-' : lambda t, param : 1,
	'-R' : lambda t, param : -1
	}

	bi_deriv_rules = {
	'+' : [lambda t1, t2, x: t1._der[x] + t2._der[x], lambda t1, t2, x: t1._der[x], lambda t1, t2, x: t2._der[x]],
	'-' : [lambda t1, t2, x: t1._der[x] - t2._der[x], lambda t1, t2, x: t1._der[x], lambda t1, t2, x: -t2._der[x]],
	'*' : [lambda t1, t2, x: t1._der[x]*t2._val + t2._der[x]*t1._val, lambda t1, t2, x: t1._der[x]*t2._val, lambda t1, t2, x: t2._der[x]*t1._val],
	'/' : [lambda t1, t2, x : (t1._der[x]*t2._val - t1._val*t2._der[x])/(t2._val**2), lambda t1, t2, x : t1._der[x] / t2._val, lambda t1, t2, x : (-t1._val * t2._der[x]) / (t2._val)**2],
	'^' : [lambda t1, t2, x: (t1.val ** t2.val) * (t2._der[x]*np.log(t1._val) + t2._val*t1._der[x]/t1._val), lambda t1, t2, x: t2._val * t1._der[x] * (t1._val ** (t2._val - 1)), lambda t1, t2, x: (t1.val ** t2.val) * t2._der[x]*np.log(t1._val)]
	}

def deriv_1(t, op, partial_only = False, param = None):
	'''
	Deriv of single-input operators, or double-digit operators with one scalar input
	'''

	# d_op_dx = d_op_dt * dt_dx
	d_op_dt = Ops.deriv_rules[op](t, param)

	if partial_only: return d_op_dt

	return {x : t._der[x] * d_op_dt for x in t._der}

def deriv_2(t1, op, t2, partial_only = False):

	'''
	Deriv of double-input operators

	for each variable x, specify *three* derivative rules for each operator, for each of the following cases:
	a) when x in both t1 and t2
	b) when x in t1 and not in t2
	c) when x in t2 and not in t1
	'''
	new_der = {}

	for x in t1._der:
		if x in t2._der:
			new_der[x] = Ops.bi_deriv_rules[op][0](t1, t2, x) #function to combine derivs when x is in the domain of both functions
		else:
			new_der[x] = Ops.bi_deriv_rules[op][1](t1, t2, x) #function to combine derivs when x is in the domain of t1 only
	for x in t2._der:
		if x not in t1._der:
			new_der[x] = Ops.bi_deriv_rules[op][2](t1, t2, x) #function to combine derivs when x is in the domain of t2 only

	return new_der

def deriv(t, op, other = None, partial_only = False):

	result = None

	if not other:
		result = deriv_1(t, op, partial_only)
	else:
		try:
			result = deriv_2(t, op, other, partial_only)
		except AttributeError:
			result = deriv_1(t, op, partial_only, other)

	return result


