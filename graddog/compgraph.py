import numpy as np
import pandas as pd

class CompGraph:

	# implements the singleton design pattern 
	# so that only one instance of a CompGraph object ever exists

	class __CompGraph:
		def __init__(self, formula, val):
			self.reset()
			self.add_var(formula, val)

		def __str__(self):
			return repr(self)

		def reset(self):
			self.size = 0
			self.num_vars = 0
			self.var_names = []
			self.outs = {}
			self.ins = {}
			self.table = pd.DataFrame(columns = ['trace_name', 'label', 'formula', 'val'])

		def add_var(self, formula, val):

			if formula in self.var_names:
				self.reset()
				self.add_var(formula, val)
			else:
				'''
				TODO: instead of resetting the table when a variable is replaced,
				just delete the old info from the table that's no longer relevant
				'''
				self.var_names.append(formula)
				
				if self.size > 0:
					# create new column for derivatives with respect to this new var
					self.table['der_'+formula] = np.zeros(shape = (self.size,))
				else:
					self.table['der_'+formula] = [1.0]

				new_trace_name = self.new_trace_name()

				self.outs[new_trace_name] = []
				self.ins[new_trace_name] = []

				# create new row for this new var
				self.table.loc[self.size - 1] = [new_trace_name, 'INPUT', formula, val] + [0.0 for _ in range(self.num_vars)] + [1.0]

				self.num_vars += 1
				return new_trace_name

		def add_trace(self, formula, val, der):
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

		@property
		def comp_graph(self):
			return {'in' : self.ins, 'out' : self.outs}

		def __repr__(self):
			return repr(self.table)

	instance = None

	def __init__(self, formula, val):
		if not CompGraph.instance:
			CompGraph.instance = CompGraph.__CompGraph(formula, val)

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

	def reset():
		if CompGraph.instance:
			CompGraph.instance.reset()



