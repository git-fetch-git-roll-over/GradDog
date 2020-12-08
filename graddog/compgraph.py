# :)
import numpy as np
import pandas as pd
import graddog.math as math
from itertools import combinations_with_replacement

# TODO: come up with a better name for this class

# TODO: add docstrings and examples

class CompGraph:

	# implements the singleton design pattern 
	# so that only one instance of a CompGraph object ever exists

	class __CompGraph:
		def __init__(self):
			self.reset()

		def __str__(self):
			return repr(self)

		def reset(self):
			'''
			Sets all the attributes to their initial values
			'''
			self.size = 0

			#store the number of variables, as well as the variable names
			self.num_vars = 0
			self.var_names = []

			# store the graph connections in two separate dictionaries, ins (AKA parents) and outs (AKA children)
			self.outs = {}
			self.ins = {}

			# store the actual trace objects to avoid repeated calculations
			self.traces = {}

			# store the partial derivatives
			self.partials = {}

			#set up the table as a pandas DataFrame for now. makes things simpler tbh.
			self.table = pd.DataFrame(columns = ['trace_name', 'label', 'formula', 'val', 'partial1', 'partial2'])

		def get_existing_trace(self, formula):
			# if you are calculating a term already in the table, just look it up 
			# (e.g. in the function f = x*y + exp(x*y) we only need to compute x*y once)
			already = self.table.loc[self.table['formula'] == formula]
			if not already.empty:
				trace_name = already['trace_name'].values[0]
				return self.traces[trace_name]
			else:
				return None

		def get_label_string(self, op, new_trace_name, formula):
			# when adding a trace, if it is a variable, label it 'INPUT' and add it to the variables
			# otherwise, if it is not a variable, the row is labelled 'OUTPUT' unless later it becomes a parent, in which case the OUTPUT label is removed
			if op is None:
				variable_name = formula
				if variable_name in self.var_names:
					self.reset()
				self.var_names.append(variable_name)
				self.num_vars += 1
				label_string = 'INPUT'
			else:
				label_string = 'OUTPUT'
			return label_string

		def update_computational_graph(self, new_trace_name, parents):
			# the 'ins' of the new trace are the parents
			self.ins[new_trace_name] = list(map(lambda p : p._trace_name, parents))
			# the 'outs' of each parent now include the new trace
			# and now the parents can no longer be outputs, so we ensure their label_string is '' instead of 'OUTPUT'
			for x in parents:
				self.outs[x._trace_name].append(new_trace_name)
				row_index = int(x._trace_name[1:]) - 1
				f = self.table.loc[row_index]['formula']
				if f not in self.var_names:
					self.table.at[row_index, 'label'] = ''
			# the 'outs' of the new trace start out empty
			self.outs[new_trace_name] = []

		def add_trace(self, trace):

			# unpack trace data
			formula, val, der, parents, op, param = trace._formula, trace._val, trace._der, trace._parents, trace._op, trace._param

			# check if we can avoid a repeated calculation
			existing_trace = self.get_existing_trace(formula)
			if existing_trace:
				return existing_trace._trace_name

			# get new trace name
			new_trace_name = self.new_trace_name()

			# get new label
			label_string = self.get_label_string(op, new_trace_name, formula)
		
			# update computational graph
			self.update_computational_graph(new_trace_name, parents)

			# add this new trace to the dictionary of traces so far
			self.traces[new_trace_name] = trace

			# calculate partial derivatives for the table
			derivs = self.partial_derivs_for_table(new_trace_name, der, op, parents, param)

			# update trace table
			self.add_trace_table_row(new_trace_name, label_string, formula, val, derivs)

			return new_trace_name

		def add_trace_table_row(self, new_trace_name, label_string, formula, val, partial_derivs_list):
			# add new row to the trace table
			self.table.loc[self.size - 1] = [new_trace_name, label_string, formula, val] + partial_derivs_list

		def partial_derivs_for_table(self, new_trace_name, der, op, parents, param):
			'''
			Partial derivative(s) of this trace
			'''
			self.partials[new_trace_name] = der
			partial_derivs_list = list(der.values())
			if len(partial_derivs_list) == 1:
				partial_derivs_list.append('NaN')
			return partial_derivs_list

		def new_trace_name(self):
			# creates trace names v1, v2, v3, etc.
			self.size += 1
			return 'v' + str(self.size)

		def get_variable_row(self, var_name):
			return int(self.table.loc[self.table['formula'] == var_name]['trace_name'].iloc[0][1:]) - 1

		def outputs(self):
			return self.table.loc[self.table['label'] == 'OUTPUT']['trace_name'].values

		def forward_mode_der(self):
			'''
			step FORWARDS through the trace table, calculate derivatives along the way in trace_derivs
			
			Returns numpy array of derivatives df_i/dx_j for each output function f_i w.r.t. each input variable x_j
			'''
			trace_derivs = {self.get_trace_name(x) : np.eye(self.num_vars)[i,:] for i, x in enumerate(self.var_names)}
			for row in range(self.num_vars, self.size):
				trace_name = self.table.loc[row]['trace_name']
				d_trace_d_chilren = np.array([[self.partials[trace_name][in_] for in_ in self.ins[trace_name]]])
				d_children_d_vars = np.vstack([trace_derivs[in_] for in_ in self.ins[trace_name]])
				trace_derivs[trace_name] = np.dot(d_trace_d_chilren, d_children_d_vars)
			return np.array([trace_derivs[output][0] for output in self.outputs()])

		def reverse_mode_der(self):
			'''
			step BACKWARDS through the trace table, calculate derivatives along the way in trace_derivs
			
			Returns numpy array of derivatives df_i/dx_j for each output function f_i w.r.t. each input variable x_j
			'''
			trace_derivs = {x : np.eye(self.num_outputs())[:,i].reshape(-1,1) for i, x in enumerate(self.outputs())}
			for row in reversed(range(self.size)):
				trace_name = self.table.loc[row]['trace_name']
				label = self.table.loc[row]['label']
				if label != 'OUTPUT':
					if self.outs[trace_name] == []:
						d_outs_d_trace = np.zeros(shape=(self.num_outputs(), 1))
					else:
						d_outs_d_children = np.hstack([trace_derivs[out_] for out_ in self.outs[trace_name]])
						d_children_d_trace = np.array([[self.partials[out_][trace_name] for out_ in self.outs[trace_name]]])
						d_outs_d_trace = np.dot(d_outs_d_children, d_children_d_trace.T)
					trace_derivs[trace_name] = d_outs_d_trace
			return np.hstack([trace_derivs[x] for x in list(map(lambda x : self.get_trace_name(x), self.var_names))])

		def get_trace_name(self, var_name):
			# lookup a variable in the table to get its trace name
			return self.table.loc[self.table['formula'] == var_name]['trace_name'].iloc[0]

		def num_outputs(self):
			return len(self.table.loc[self.table['label'] == 'OUTPUT'])

		@property
		def comp_graph(self):
			return {'in' : self.ins, 'out' : self.outs}

		def __repr__(self):
			return repr(self.table)

		def hessian(self):
			'''
			Implements the edge-pushing algorithm

			https://par.nsf.gov/servlets/purl/10039361

			Returns BOTH the jacobian (first derivative) and hessian (second derivative)

			Requires that the user has traced a function with only a single output
			'''
			
			if self.num_outputs() > 1:
				raise ValueError('Hessian can only be calculated for scalar function')

			# is this a hack tho #
			l = self.size
			for v in self.partials:
				for u in self.partials:
					if u not in self.partials and u != v:
						self.partials[v][u] = 0
			############################################



			v_l = f'v{l}'
			S = {self.size + 1 : set([v_l])}
			h = {k : { f'v{j+1}' : {f'v{i+1}' : 0.0 for i in range(l)} for j in range(l)} for k in range(1, l + 2)}
			v_bar = { f'v{i}' : 0.0 for i in range(1,self.size)}
			v_bar[v_l] = 1.0

			# step backward through the trace table
			m = self.num_vars
			for k in reversed(range(m+1,l + 1)):	

				##### make S
				v_k = f'v{k}'
				print('v ~', v_k)
				S[k] = S[k + 1]
				if v_k in S[k]:
					S[k].remove(v_k)
				S[k] = S[k].union(self.ins[v_k])
				print('S ~~', S[k])
				#### make v_bar
				for v in self.ins[v_k]:
					v_bar[v] += self.partials[v_k][v] * v_bar[v_k]
				print('v bar ~~~', v_bar)
				#### build current hessian layer
				for v_i, v_j in combinations_with_replacement(S[k], 2):
					#if v_i in self.ins[v_k] or v_j in self.ins[v_k]:
					print('vi vj ~~~~', v_i, v_j)
					a = h[k+1][v_i][v_j]
					b = self.partials[v_k][v_i]*h[k+1][v_j][v_k]
					c = self.partials[v_k][v_j]*h[k+1][v_i][v_k]
					d = self.partials[v_k][v_i]*self.partials[v_k][v_j]*h[k+1][v_k][v_k]
					t1, t2, t3 = self.traces[v_k], self.traces[v_i], self.traces[v_j]
					e = math.double_deriv(t1, t2, t3) #* v_bar[v_k]
					f = a+b+c+d+e
					print(a, b, c, d, e, f)
					h[k][v_i][v_j] = f
					if v_i in self.var_names and v_j in self.var_names:
						h[k][v_j][v_i] = f

					print('h ~~~~~', h[k])
			j = np.array([[v_bar[self.get_trace_name(v)] for v in self.var_names]])
			h = np.array([[h[m+1][f'v{i+1}'][f'v{j+1}'] for j in range(m)] for i in range(m)])
			return j, h

	instance = None

	def __init__():
		if not CompGraph.instance:
			CompGraph.instance = CompGraph.__CompGraph()

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

	def derivative(mode = 'forward'):
		if mode == 'forward':
			CompGraph.forward_mode()
		elif mode == 'reverse':
			CompGraph.reverse_mode()
		else:
			raise ValueError('Mode attribute must be forward or reverse')

	def forward_mode():
		if CompGraph.instance:
			return CompGraph.instance.forward_mode_der()

	def reverse_mode():
		if CompGraph.instance:
			return CompGraph.instance.reverse_mode_der()

	def add_trace(trace):
		if not CompGraph.instance:
			CompGraph.instance = CompGraph.__CompGraph()
		return CompGraph.instance.add_trace(trace)

	def num_outputs():
		if CompGraph.instance:
			return CompGraph.instance.num_outputs()

	def hessian():
		CompGraph.show_trace_table()
		if CompGraph.instance:
			return CompGraph.instance.hessian()



