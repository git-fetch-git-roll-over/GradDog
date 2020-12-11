# :)
import numpy as np
import pandas as pd
import graddog.math as math
from itertools import combinations_with_replacement


class CompGraph:

	# implements the singleton design pattern 
	# so that only one instance of a CompGraph object ever exists

	class __CompGraph:
		def __init__(self):
			self.reset()

		def reset(self):
			'''
			Sets all the attributes to their initial values
			'''
			self.size = 0

			#store the number of variables, as well as the variable names
			self.num_vars = 0
			self.variables = []

			# store the graph connections in two separate dictionaries, ins (AKA parents) and outs (AKA children)
			self.outs = {}
			self.ins = {}

			# store the actual trace objects to avoid repeated calculations
			self.traces = {}

			# store the partial derivatives
			self.partials = {}

			#set up the table as a pandas DataFrame for now. makes things simpler tbh.
			self.table = pd.DataFrame(columns = ['trace_name', 'input', 'output', 'formula', 'val', 'partial1', 'partial2'])

		def add_trace(self, trace):
			'''
			Adds a trace to the trace table and stores the relevant partial derivatives

			Returns the newly generated trace_name
			'''

			# unpack trace data
			formula, val, der, parents, op, param = trace._formula, trace._val, trace._der, trace._parents, trace._op, trace._param

			# check if we can avoid a repeated calculation
			existing_trace = self.get_existing_trace(formula)
			if existing_trace:
				return existing_trace._trace_name
			else:
				# get new trace name
				new_trace_name = self.new_trace_name()

				# get new label
				is_input, is_output = self.get_labels(op, formula)
			
				# update computational graph
				self.update_computational_graph(new_trace_name, parents)

				# add this new trace to the dictionary of traces so far
				self.traces[new_trace_name] = trace

				# calculate partial derivatives to be formatted for the table.
				derivs = self.partial_derivs_for_table(new_trace_name, der, op, parents, param)

				# update trace table
				self.add_trace_table_row(new_trace_name, is_input, is_output, formula, val, derivs)

				# return new_trace_name to the Trace class
				return new_trace_name

		def get_existing_trace(self, formula):
			'''
			if you are calculating a term already in the table, just look it up 
			for example,
			f(x,y) = x*y + exp(x*y) 
			we only need to compute x*y once
			'''
			already = self.table.loc[self.table['formula'] == formula]
			if not already.empty:
				trace_name = already['trace_name'].values[0]
				return self.traces[trace_name]
			else:
				return None

		def get_labels(self, op, formula):
			'''
			when adding a trace, if it is a variable, label it 'INPUT' and add it to the variables
			otherwise, if it is not a variable, the row is labelled 'OUTPUT'
			later on, if the trace becomes a parent, its OUTPUT label is removed
			'''
			if op is None:
				self.add_new_variable(formula)
				is_input, is_output = True, False
			else:
				is_input, is_output = False, False
			return is_input, is_output

		def add_new_variable(self, formula):
			'''
			Adds the new variable with the name `formula` to self.variables

			Design choice: if the newly created variable is already in the set of variables, reset
			'''
			if formula in self.variables:
				self.reset()
			else:
				self.variables.append(formula)
				self.num_vars += 1

		def update_computational_graph(self, new_trace_name, parents):
			'''
			Updates the information in self.ins and self.outs
			For the parents who are no longer outputs, remove their 'OUTPUT' label

			self.ins represents the parents of each trace. 
			self.outs represents the children of each trace. 
			if v3 = v1*v2, 
			self.ins[v3] = [v1, v2]
			self.outs[v1] = self.outs[v2] = [v3]

			'''
			self.ins[new_trace_name] = list(map(lambda p : p._trace_name, parents))
			for x in parents:
				self.outs[x._trace_name].append(new_trace_name)
				row_index = int(x._trace_name[1:]) - 1
				f = self.table.loc[row_index]['formula']
				self.table.at[row_index, 'output'] = False
			self.outs[new_trace_name] = []

		def add_trace_table_row(self, new_trace_name, is_input, is_output, formula, val, partial_derivs_list):
			'''
			Formats a new row in the trace table dataframe
			'''
			self.table.loc[self.size - 1] = [new_trace_name, is_input, is_output, formula, val] + partial_derivs_list

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
			'''
			creates trace names v1, v2, v3, etc.
			'''
			self.size += 1
			return f'v{self.size}'

		def get_variable_row(self, var_name):
			return int(self.table.loc[self.table['formula'] == var_name]['trace_name'].iloc[0][1:]) - 1

		def get_trace_row(self, trace_name):
			return int(trace_name[1:])-1

		def outputs(self):
			return self.table.loc[self.table['output'] == True]['trace_name'].values

		def get_trace_name(self, var_name):
			# lookup a variable in the table to get its trace name
			return self.table.loc[self.table['formula'] == var_name]['trace_name'].iloc[0]

		def num_outputs(self):
			return len(self.outputs())

		def __repr__(self):
			return repr(self.table)

		def label_outputs(self, output):
			for o in output:
				t = o._trace_name
				row = self.get_trace_row(t)
				self.table.loc[row,'output'] = True

		def forward_mode_der(self, output, verbose = False):
			'''
			step FORWARDS through the trace table, calculate derivatives along the way in trace_derivs

			The gradient for each trace is computed as the vector `d_v_d_variables`
			
			Returns the Jacobian matrix of derivatives df_i/dx_j for each output function f_i w.r.t. each input variable x_j

			'''
			self.label_outputs(output)
			if verbose:
				CompGraph.show_trace_table()
			
			# Initialize derivative for each input
			self.trace_derivs = {self.get_trace_name(x) : np.eye(self.num_vars)[i,:] for i, x in enumerate(self.variables)}
			
			for row in range(self.num_vars, self.size):

				v = self.table.loc[row]['trace_name']

				d_v_d_parents = np.array([[self.partials[v][in_] for in_ in self.ins[v]]])

				d_parents_d_variables = np.vstack([self.trace_derivs[in_] for in_ in self.ins[v]])

				d_v_d_variables = np.dot(d_v_d_parents, d_parents_d_variables)

				self.trace_derivs[v] = d_v_d_variables

			return np.vstack([self.trace_derivs[output] for output in self.outputs()])

		def reverse_mode_der(self, output, verbose = False):
			'''
			step BACKWARDS through the trace table, calculate derivatives along the way in trace_derivs

			The gradient for each trace is computed as the vector `d_outputs_d_v`
			
			Returns the Jacobian matrix of derivatives df_i/dx_j for each output function f_i w.r.t. each input variable x_j
			
			'''
			self.label_outputs(output)
			if verbose:
				CompGraph.show_trace_table()
			
			# Initialize derivative for each output
			self.trace_derivs = {x : np.eye(self.num_outputs())[:,i].reshape(-1,1) for i, x in enumerate(self.outputs())}
			
			for row in reversed(range(self.size)):

				v, is_output = self.table.loc[row]['trace_name'], self.table.loc[row]['output']

				if not is_output:

					if self.outs[v] == []:
						if verbose:
							print(f'{v} was a variable with no effect on any of the outputs')

						d_outputs_d_v = np.zeros(shape=(self.num_outputs(), 1))

					else:

						d_outputs_d_children = np.hstack([self.trace_derivs[out_] for out_ in self.outs[v]])

						d_children_d_v = np.array([[self.partials[out_][v] for out_ in self.outs[v]]])

						d_outputs_d_v = np.dot(d_outputs_d_children, d_children_d_v.T)

					self.trace_derivs[v] = d_outputs_d_v

			return np.hstack([self.trace_derivs[x] for x in list(map(lambda x : self.get_trace_name(x), self.variables))])

		def hessian(self, output, verbose = False):
			'''
			Implements the edge-pushing algorithm by Gower and Mello
			Specific implementation details from Wang, Pothen, and Hovland: https://par.nsf.gov/servlets/purl/10039361

			Returns BOTH the jacobian (first derivative) and hessian (second derivative)

			Requires that the user has traced a scalar-output function f:Rm --> R
			'''
			self.label_outputs(output)
			if self.num_outputs() > 1:
				raise ValueError('Hessian can only be calculated for scalar function')

			
			### ensure all partial derivatives exist ###
			l = self.size
			for v in self.partials:
				for u in self.partials:
					if u not in self.partials[v]:
						if u != v:
							self.partials[v][u] = 0
						else:
							self.partials[v][u] = 1
			############################################

			if verbose:
				CompGraph.show_trace_table()


			## initialize variables

			# stores the live variables
			S = {self.size + 1 : set([f'v{self.size}'])}

			# stores the Hessian as a double-nested dictionary
			# h[k][v_i][v_j] gives the 2nd derivative of the output w.r.t. v_i and v_j during iteration k of the algorithm
			h = {k : { f'v{j+1}' : {f'v{i+1}' : 0.0 for i in range(l)} for j in range(l)} for k in range(1, l + 2)}

			# stores the adjoints: the derivative of the output w.r.t. each trace
			v_bar = { f'v{i}' : 0.0 for i in range(1,self.size)}
			v_bar[f'v{self.size}'] = 1.0

			# step backward through the trace table
			m = self.num_vars
			for k in reversed(range(m+1,l + 1)):
				v_k = f'v{k}'

				##### update live variable set S #####
				S[k] = S[k + 1]
				if v_k in S[k]:
					S[k].remove(v_k)
				S[k] = S[k].union(self.ins[v_k])
				######################################


				#### accumulate adjoints v_bar #######
				for v in self.ins[v_k]:
					v_bar[v] += self.partials[v_k][v] * v_bar[v_k]
				######################################


				#### build current hessian layer
				for v_i, v_j in combinations_with_replacement(S[k], 2):

					######### calculate double derivative
					if v_i in self.ins[v_k] and v_j in self.ins[v_k]:
						double_deriv = math.double_deriv(self.traces[v_k], self.traces[v_i], self.traces[v_j]) * v_bar[v_k]
					else:
						double_deriv = 0
					############################################

					# add the relevant values from the previous layer to the double_deriv
					new_value = h[k+1][v_i][v_j] + self.partials[v_k][v_i]*h[k+1][v_j][v_k] + self.partials[v_k][v_j]*h[k+1][v_i][v_k] + self.partials[v_k][v_i]*self.partials[v_k][v_j]*h[k+1][v_k][v_k] + double_deriv
					
					# add new value to the current layer of the hessian matrix
					h[k][v_i][v_j] = new_value
					h[k][v_j][v_i] = new_value

			j = np.array([[v_bar[self.get_trace_name(v)] for v in self.variables]])
			h = np.array([[h[m+1][f'v{i+1}'][f'v{j+1}'] for j in range(m)] for i in range(m)])
			return j, h

	instance = None

	def __init__():
		if not CompGraph.instance:
			CompGraph.instance = CompGraph.__CompGraph()

	def show_trace_table():
		if CompGraph.instance:
			print(repr(CompGraph.instance))

	def reset():
		if CompGraph.instance:
			CompGraph.instance.reset()

	def forward_mode(output, verbose):
		if CompGraph.instance:
			return CompGraph.instance.forward_mode_der(output, verbose)

	def reverse_mode(output, verbose):
		if CompGraph.instance:
			return CompGraph.instance.reverse_mode_der(output, verbose)

	def add_trace(trace):
		if not CompGraph.instance:
			CompGraph.instance = CompGraph.__CompGraph()
		return CompGraph.instance.add_trace(trace)

	def num_outputs():
		if CompGraph.instance:
			return CompGraph.instance.num_outputs()

	def hessian(output, verbose):
		if CompGraph.instance:
			return CompGraph.instance.hessian(output, verbose)


