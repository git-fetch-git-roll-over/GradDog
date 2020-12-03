import numpy as np
import pandas as pd
import calc_rules as calc_rules

# TODO: come up with a better name for this class

# TODO: add docstrings and examples

# TODO: choose better location for reverse_mode

# TODO: fix redundancy between add_var and add_trace

# TODO: figure out how to pass in parent nodes as a param to add_trace instead of the godawful read_formula method

class CompGraph:

	# implements the singleton design pattern 
	# so that only one instance of a CompGraph object ever exists

	class __CompGraph:
		def __init__(self, trace):
			self.reset()
			self.add_var(trace)

		def __str__(self):
			return repr(self)

		def reset(self):
			self.size = 0
			self.num_vars = 0
			self.var_names = []
			self.outs = {}
			self.ins = {}
			self.traces = {}
			self.partials = {} # use for reverse mode
			self.table = pd.DataFrame(columns = ['trace_name', 'label', 'formula', 'val'])

		def add_var(self, var):
			formula, val = var._formula, var.val

			if formula in self.var_names:
				self.reset()
				return self.add_var(var)
			else:
				self.var_names.append(formula)
				
				if self.size > 0:
					# create new column for derivatives with respect to this new var
					self.table['der_'+formula] = np.zeros(shape = (self.size,))
				else:
					self.table['der_'+formula] = [1.0]

				new_trace_name = self.new_trace_name()

				self.traces[new_trace_name] = var

				self.outs[new_trace_name] = []
				self.ins[new_trace_name] = []

				# create new row for this new var
				self.table.loc[self.size - 1] = [new_trace_name, 'INPUT', formula, val] + [0.0 for _ in range(self.num_vars)] + [1.0]

				self.num_vars += 1
				return new_trace_name

		def read_formula(self, formula):
			if formula[0]=='-':
				trace1 = self.traces[formula[1:]]
				op = '-R'
				trace2 = 0
			elif '(' in formula:
				i1 = formula.index('(')
				op = formula[:i1]
				trace1 = self.traces[formula[i1+1:-1]]
				trace2 = None
				if op in ['exp', 'log'] : trace2 = np.e
			else:
				trace1 = self.traces[formula[:2]]
				try:
					op, trace2 = formula[2:-2], self.traces[formula[-2:]]
				except KeyError:
					op, trace2 = formula[2:-1], float(formula[-1:])
			return trace1, op, trace2

		def add_trace(self, trace):
			formula, val, der = trace._formula, trace.val, trace._der

			# if you are calculating a term already in the table, just look it up (e.g. f = x*y + exp(x*y))
			already = self.table.loc[self.table['formula'] == formula]
			if not already.empty:
				trace_name = already['trace_name'].values[0]
				return trace_name

			# create new row for this trace element
			new_trace_name = self.new_trace_name()

			self.ins[new_trace_name] = []
			for x in self.table['trace_name'].values:
				if x in formula:
					self.ins[new_trace_name].append(x)
					self.outs[x].append(new_trace_name)

			self.outs[new_trace_name] = []

			self.traces[new_trace_name] = trace

			t, op, other = self.read_formula(formula)

			partial_der = calc_rules.deriv(t, op, other, partial = True)

			self.partials[new_trace_name] = partial_der

			derivs = []
			for x in self.var_names:
				if x in der:
					derivs.append(der[x])
				else:
					derivs.append(0.0)

			self.table.loc[self.size - 1] = [new_trace_name, '', formula, val] + derivs
			return new_trace_name

		def new_trace_name(self):
			self.size += 1
			return 'v' + str(self.size)

		def reverse_mode_der(self):
			res = {}
			i = self.size - 1
			trace = self.table.loc[i]['trace_name']
			n_traces = int(trace[1:])
			res = {trace : 1.0}
			while i > 0:
				i -= 1
				trace = self.table.loc[i]['trace_name']
				r = 0
				for out_ in self.outs[trace]:
					d1 = res[out_] 
					d2 = self.partials[out_][trace]
					r += d1 * d2
				res[trace] = r
			return {x : res[self.get_trace_name(x)] for x in self.var_names}

		def get_trace_name(self, var_name):
			return self.table.loc[self.table['formula'] == var_name]['trace_name'].iloc[0]

		@property
		def comp_graph(self):
			return {'in' : self.ins, 'out' : self.outs}

		def __repr__(self):
			return repr(self.table)

	instance = None

	def __init__(self, trace):
		if not CompGraph.instance:
			CompGraph.instance = CompGraph.__CompGraph(trace)

	def __getattr__(self, name):
		return getattr(self.instance, name)

	def __repr__(self):
		if CompGraph.instance:
			return repr(CompGraph.instance)

	def show_trace_table():
		if CompGraph.instance:
			print(repr(CompGraph.instance))

	def show_comp_graph():
		if CompGraph.instance:
			print(CompGraph.instance.comp_graph)

	def show_partials():
		if CompGraph.instance:
			print(CompGraph.instance.partials)

	def reset():
		if CompGraph.instance:
			CompGraph.instance.reset()

	def reverse_mode():
		if CompGraph.instance:
			print(CompGraph.instance.reverse_mode_der())



