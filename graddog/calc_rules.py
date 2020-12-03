# :)
import numpy as np

def deriv_1(t, op, partial_only = False, param = None):
	'''
	Deriv of single-input operators, or double-digit operators with one scalar input
	'''
	deriv_rules = {
	'sin' : lambda t : np.cos(t.val),
	'cos' : lambda t : -np.sin(t.val),
	'tan' : lambda t : 1/(np.cos(t.val)**2),
	'exp' : lambda t : np.power(param, t.val)*np.log(param),
	'log' : lambda t : 1/(t.val*np.log(param)),
	'sqrt' : lambda t : 1/(2*t.val**0.5),
	'sigm' : lambda t : np.exp(-t.val)/((1 + np.exp(-t.val))**2),
	'sinh' : lambda t : (np.exp(t.val) + np.exp(-t.val))/2,
	'cosh' : lambda t : (np.exp(t.val) - np.exp(-t.val))/2,
	'tanh' : lambda t : 4/((np.exp(t.val) + np.exp(-t.val))**2),
	'^' : lambda t : param*t._val**(param-1),
	'^R' : lambda t : param ** t._val * np.log(param),
	'*' : lambda t : param,
	'/' : lambda t : 1/param,
	'/R' : lambda t : -param / ((t._val)**2),
	'+' : lambda t : 1,
	'-' : lambda t : 1,
	'-R' : lambda t : -1
	}

	# d_op_dx = d_op_dt * dt_dx
	d_op_dt = deriv_rules[op](t)

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
	deriv_rules = {
	'+' : [lambda t1, t2, x: t1._der[x] + t2._der[x], lambda t1, t2, x: t1._der[x], lambda t1, t2, x: t2._der[x]],
	'-' : [lambda t1, t2, x: t1._der[x] - t2._der[x], lambda t1, t2, x: t1._der[x], lambda t1, t2, x: -t2._der[x]],
	'*' : [lambda t1, t2, x: t1._der[x]*t2._val + t2._der[x]*t1._val, lambda t1, t2, x: t1._der[x]*t2._val, lambda t1, t2, x: t2._der[x]*t1._val],
	'/' : [lambda t1, t2, x : (t1._der[x]*t2._val - t1._val*t2._der[x])/(t2._val**2), lambda t1, t2, x : t1._der[x] / t2._val, lambda t1, t2, x : (-t1._val * t2._der[x]) / (t2._val)**2],
	'^' : [lambda t1, t2, x: (t1.val ** t2.val) * (t2._der[x]*np.log(t1._val) + t2._val*t1._der[x]/t1._val), lambda t1, t2, x: t2._val * t1._der[x] * (t1._val ** (t2._val - 1)), lambda t1, t2, x: (t1.val ** t2.val) * t2._der[x]*np.log(t1._val)]
	}

	new_der = {}

	for x in t1._der:
		if x in t2._der:
			new_der[x] = deriv_rules[op][0](t1, t2, x) #function to combine derivs when x is in the domain of both functions
		else:
			new_der[x] = deriv_rules[op][1](t1, t2, x) #function to combine derivs when x is in the domain of t1 only
	for x in t2._der:
		if x not in t1._der:
			new_der[x] = deriv_rules[op][2](t1, t2, x) #function to combine derivs when x is in the domain of t2 only

	return new_der

def deriv(t, op, other = None, partial_only = False):

	result = None

	if not other:
		result = deriv_1(t, op, partial_only)
	else:
		try:
			result = deriv_2(t, op, other, partial_only)
		except AttributeError:
			result = deriv_1(t, op, partial_only, param = other)

	return result


