# :)
import numpy as np

def deriv_1(t, op, param = None):
	'''
	Deriv of single-input operators, or double-digit operators with one scalar input
	'''
	op_dict = {
	'sin' : lambda t : {x : np.cos(t.val)*t._der[x] for x in t._der},
	'cos' : lambda t : {x : -np.sin(t.val)*t._der[x] for x in t._der},
	'tan' : lambda t : {x : t._der[x]/(np.cos(t.val)**2) for x in t._der},
	'exp' : lambda t : {x : np.exp(t.val)*t._der[x] for x in t._der},
	'log' : lambda t : {x : t._der[x]/t.val for x in t._der},
	'sqrt' : lambda t : {x : t._der[x]/(2*t.val**0.5) for x in t._der},
	'sigm' : lambda t : {x : t._der[x]*np.exp(-t.val)/((1 + np.exp(-t.val))**2) for x in t._der},
	'sinh' : lambda t : {x : t._der[x]*(np.exp(t.val) + np.exp(-t.val))/2 for x in t._der},
	'cosh' : lambda t : {x : t._der[x]*(np.exp(t.val) - np.exp(-t.val))/2 for x in t._der},
	'tanh' : lambda t : {x : t._der[x]*4/((np.exp(t.val) + np.exp(-t.val))**2) for x in t._der},
	'^' : lambda t : {x : param*t._val**(param-1)*t._der[x] for x in t._der},
	'^R' : lambda t : {x : param ** t._val * np.log(param) * t._der[x] for x in t._der},
	'*' : lambda t : {x : t._der[x] * param for x in t._der},
	'/' : lambda t : {x : t._der[x] / param for x in t._der},
	'/R' : lambda t : {x : (-param * t._der[x]) / ((t._val)**2) for x in t._der},
	'+' : lambda t : t._der,
	'-' : lambda t : t._der,
	'-R' : lambda t : {x : -t._der[x] for x in t._der}
	}
	if param:
		if op == 'exp': return lambda t : {x : np.power(param, t.val)*np.log(param)*t._der[x] for x in t._der}
		elif op == 'log': return lambda t : {x : t._der[x]/(t.val*np.log(param)) for x in t._der}
	return op_dict[op](t)

def deriv_2(t1, op, t2):

	'''
	Deriv of double-input operators

	for each variable x, specify *three* derivative rules for each operator, for each of the following cases:
	a) when x in both t1 and t2
	b) when x in t1 and not in t2
	c) when x in t2 and not in t1
	'''
	op_dict = {
	'+' : [lambda t1, t2, x: t1._der[x] + t2._der[x], lambda t1, t2, x: t1._der[x], lambda t1, t2, x: t2._der[x]],
	'-' : [lambda t1, t2, x: t1._der[x] - t2._der[x], lambda t1, t2, x: t1._der[x], lambda t1, t2, x: -t2._der[x]],
	'*' : [lambda t1, t2, x: t1._der[x]*t2._val + t2._der[x]*t1._val, lambda t1, t2, x: t1._der[x]*t2._val, lambda t1, t2, x: t2._der[x]*t1._val],
	'/' : [lambda t1, t2, x : (t1._der[x]*t2._val - t1._val*t2._der[x])/(t2._val**2), lambda t1, t2, x : t1._der[x] / t2._val, lambda t1, t2, x : (-t1._val * t2._der[x]) / (t2._val)**2],
	'^' : [lambda t1, t2, x: (t1.val ** t2.val) * (t2._der[x]*np.log(t1._val) + t2._val*t1._der[x]/t1._val), lambda t1, t2, x: t2._val * t1._der[x] * (t1._val ** (t2._val - 1)), lambda t1, t2, x: (t1.val ** t2.val) * t2._der[x]*np.log(t1._val)]
	}

	new_der = {}

	for x in t1._der:
		if x in t2._der:
			new_der[x] = op_dict[op][0](t1, t2, x) #function to combine derivs when x is in the domain of both functions
		else:
			new_der[x] = op_dict[op][1](t1, t2, x) #function to combine derivs when x is in the domain of t1 only
	for x in t2._der:
		if x not in t1._der:
			new_der[x] = op_dict[op][2](t1, t2, x) #function to combine derivs when x is in the domain of t2 only

	return new_der

def deriv(t, op, other = None):

	if not other:
		return deriv_1(t, op)
	else:
		try:
			return deriv_2(t, op, other)
		except AttributeError:
			return deriv_1(t, op, param = other)


