# :)
import numpy as np

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
	power = '^'
	mul = '*'
	div = '/'
	div_R = '/R'
	add = '+'
	sub = '-'
	sub_R = '-R'

	one_parent_rules = {
	add: lambda t, param : t.val + param,
	sub: lambda t, param : t.val - param,
	sub_R : lambda t, param : param - t.val,
	mul: lambda t, param : t.val*param,
	div: lambda t, param : t.val/param,
	div_R : lambda t, param : param/t.val,
	power: lambda t, param : t._val**param,
	sin : lambda t, param: np.sin(t.val),
	cos : lambda t, param : np.cos(t.val),
	tan : lambda t, param : np.tan(t.val),
	exp : lambda t, param : np.power(param, t.val),
	log : lambda t, param : np.log(t.val)/np.log(param),
	sqrt : lambda t, param : t.val**0.5,
	sigm : lambda t, param : 1/(1 + np.exp(-t.val)),
	sinh : lambda t, param : (np.exp(t.val) - np.exp(-t.val))/2,
	cosh : lambda t, param : (np.exp(t.val) + np.exp(-t.val))/2,
	tanh : lambda t, param : (np.exp(t.val) - np.exp(-t.val))/(np.exp(t.val) + np.exp(-t.val)),
	}

	one_parent_deriv_rules = {
	add: lambda t, param : 1.0,
	sub: lambda t, param : 1.0,
	sub_R : lambda t, param : -1.0,
	mul: lambda t, param : param,
	div: lambda t, param : 1/param,
	div_R : lambda t, param : -param / ((t._val)**2),
	power : lambda t, param : param * t._val ** (param - 1),
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
	}

	two_parent_rules = {
	add : lambda t1, t2 : t1.val + t2.val,
	sub : lambda t1, t2 : t1.val - t2.val,
	mul : lambda t1, t2 : t1.val * t2.val,
	div : lambda t1, t2 : t1.val / t2.val,
	power : lambda t1, t2 : t1.val ** t2.val,
	}

	two_parent_deriv_rules = {
	add : lambda t1, t2 : (1.0, 1.0),
	sub : lambda t1, t2 : (1.0, -1.0),
	mul : lambda t1, t2 : (t2.val, t1.val),
	div : lambda t1, t2 : (1/t2.val, -t1.val/(t2.val**2)),
	power : lambda t1, t2 : (t2.val*(t1.val**(t2.val-1)), (t1.val**t2.val)*np.log(t1.val)),
	}

def deriv_1(t, op, param = None):
	# derivative of a trace with one parent
	try:
		d_op_dt = Ops.one_parent_deriv_rules[op](t, param)
		return {t._trace_name : d_op_dt}
	except KeyError:
		raise ValueError('need to implement operation', op)
	

def deriv_2(t1, op, t2):
	# derivative of a trace with two parents
	try:
		t2_val = t2.val # causes an AttributeError to force deriv_1_new to get called instead :P
		d_op_dt1, d_op_dt2 = Ops.two_parent_deriv_rules[op](t1, t2)
		return {t1._trace_name : d_op_dt1, t2._trace_name : d_op_dt2}
	except KeyError:
		raise ValueError('need to implement operation', op)


def deriv(t, op, other = None):
	if other is None:
		return deriv_1(t, op)
	try:
		# if other is a trace
		return deriv_2(t, op, other)
	except AttributeError:
		# if other is a param, AKA just a number
		return deriv_1(t, op, other)


