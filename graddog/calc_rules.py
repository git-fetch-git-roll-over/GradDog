# :)
import numpy as np
from graddog.compgraph import CompGraph

# TODO: convert lambda expressions to closure functions where appropriate

# TODO: how can this be organized better?

# TODO: How do we distinguish that this file is meant for internal use and not for the user?

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
	const_div_R = '/R'
	const_add = '+'
	const_sub = '-'
	const_sub_R = '-R'

	deriv_rules = {
	sin : lambda t, param : np.cos(t.val),
	cos : lambda t, param : -np.sin(t.val),
	tan : lambda t, param : 1/(np.cos(t.val)**2),
	exp : lambda t, param : np.power(param, t.val)*np.log(param),
	log : lambda t, param : 1/(t.val*np.log(param)),
	sqrt : lambda t, param : 1/(2*t.val**0.5),
	sigm : lambda t, param : np.exp(-t.val)/((1 + np.exp(-t.val))**2),
	sinh : lambda t, param : (np.exp(t.val) + np.exp(-t.val))/2,
	cosh : lambda t, param : (np.exp(t.val) - np.exp(-t.val))/2,
	tanh : lambda t, param : 4/((np.exp(t.val) + np.exp(-t.val))**2),
	const_exp: lambda t, param : param*t._val**(param-1),
	const_exp_R : lambda t, param : param ** t._val * np.log(param),
	const_mul: lambda t, param : param,
	const_div: lambda t, param : 1/param,
	const_div_R : lambda t, param : -param / ((t._val)**2),
	const_add: lambda t, param : 1.0,
	const_sub: lambda t, param : 1.0,
	const_sub_R : lambda t, param : -1.0
	}

def deriv_1(t, op, partial = False, param = None):
	'''
	Deriv of single-input operators, or double-input operators with one scalar input
	'''

	# d_op_dx = d_op_dt * dt_dx
	d_op_dt = Ops.deriv_rules[op](t, param)

	result = {x : t._der[x] * d_op_dt for x in t._der}

	if partial: return result, {t._trace_name : d_op_dt}

	return result, None

def deriv_2(t1, op, t2, partial = False):

	'''
	Deriv of double-input operators

	for each variable x, specify *three* derivative rules for each operator, for each of the following cases:
	a) when x in both t1 and t2
	b) when x in t1 and not in t2
	c) when x in t2 and not in t1
	'''

	bi_deriv_rules = {
	'+' : [lambda t1, t2, x: t1._der[x] + t2._der[x], lambda t1, t2, x: t1._der[x], lambda t1, t2, x: t2._der[x]],
	'-' : [lambda t1, t2, x: t1._der[x] - t2._der[x], lambda t1, t2, x: t1._der[x], lambda t1, t2, x: -t2._der[x]],
	'*' : [lambda t1, t2, x: t1._der[x]*t2._val + t2._der[x]*t1._val, lambda t1, t2, x: t1._der[x]*t2._val, lambda t1, t2, x: t2._der[x]*t1._val],
	'/' : [lambda t1, t2, x : (t1._der[x]*t2._val - t1._val*t2._der[x])/(t2._val**2), lambda t1, t2, x : t1._der[x] / t2._val, lambda t1, t2, x : (-t1._val * t2._der[x]) / (t2._val)**2],
	'^' : [lambda t1, t2, x: (t1.val ** t2.val) * (t2._der[x]*np.log(t1._val) + t2._val*t1._der[x]/t1._val), lambda t1, t2, x: t2._val * t1._der[x] * (t1._val ** (t2._val - 1)), lambda t1, t2, x: (t1.val ** t2.val) * t2._der[x]*np.log(t1._val)]
	}

	new_der = {}

	for x in t1._der:
		if x in t2._der:
			new_der[x] = bi_deriv_rules[op][0](t1, t2, x) #function to combine derivs when x is in the domain of both functions
		else:
			new_der[x] = bi_deriv_rules[op][1](t1, t2, x) #function to combine derivs when x is in the domain of t1 only
	for x in t2._der:
		if x not in t1._der:
			new_der[x] = bi_deriv_rules[op][2](t1, t2, x) #function to combine derivs when x is in the domain of t2 only

	if partial: 
		if op == '+' : return new_der, {t1._trace_name : 1.0, t2._trace_name : 1.0}
		elif op == '-' : return new_der, {t1._trace_name : 1.0, t2._trace_name : -1.0}
		elif op == '*' : return new_der, {t1._trace_name : t2.val, t2._trace_name : t1.val}
		elif op == '/' : return new_der, {t1._trace_name : 1/t2.val, t2._trace_name : -t1.val/(t2.val**2)}

		elif op == '^' : return {t1._trace_name : 1.0, t2._trace_name : 1.0} # skip implementing this for now
		
	else:
		return new_der, None

def deriv(t, op, other = None, partial = False):

	result = None

	if other is None:
		result, partial = deriv_1(t, op, partial)
	else:
		try:
			result, partial = deriv_2(t, op, other, partial)
		except AttributeError:
			result, partial = deriv_1(t, op, partial, other)
	if partial:
		return partial
	return result


