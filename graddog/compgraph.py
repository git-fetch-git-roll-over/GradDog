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
			self.variables = []

			# store the graph connections in two separate dictionaries, ins (AKA parents) and outs (AKA children)
			self.outs = {}
			self.ins = {}

			# store the actual trace objects to avoid repeated calculations
			self.traces = {}

			# store the partial derivatives
			self.partials = {}

			# store the double derivatives
			self.doubles = {}

			#set up the table as a pandas DataFrame for now. makes things simpler tbh.
			self.table = pd.DataFrame(columns = ['trace_name', 'label', 'formula', 'val', 'partial1', 'partial2'])

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
				label_string = self.get_label_string(op, formula)
			
				# update computational graph
				self.update_computational_graph(new_trace_name, parents)

				# add this new trace to the dictionary of traces so far
				self.traces[new_trace_name] = trace

				# calculate partial derivatives to be formatted for the table.
				derivs = self.partial_derivs_for_table(new_trace_name, der, op, parents, param)

				# update trace table
				self.add_trace_table_row(new_trace_name, label_string, formula, val, derivs)

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

		def get_label_string(self, op, formula):
			'''
			when adding a trace, if it is a variable, label it 'INPUT' and add it to the variables
			otherwise, if it is not a variable, the row is labelled 'OUTPUT'
			later on, if the trace becomes a parent, its OUTPUT label is removed
			'''
			if op is None:
				self.add_new_variable(formula)
				label_string = 'INPUT'
			else:
				label_string = 'OUTPUT'
			return label_string

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
				if f not in self.variables:
					self.table.at[row_index, 'label'] = ''
			self.outs[new_trace_name] = []

		def add_trace_table_row(self, new_trace_name, label_string, formula, val, partial_derivs_list):
			'''
			Formats a new row in the trace table dataframe
			'''
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
			'''
			creates trace names v1, v2, v3, etc.
			'''
			self.size += 1
			return f'v{self.size}'

		def get_variable_row(self, var_name):
			return int(self.table.loc[self.table['formula'] == var_name]['trace_name'].iloc[0][1:]) - 1

		def outputs(self):
			return self.table.loc[self.table['label'] == 'OUTPUT']['trace_name'].values

		def get_trace_name(self, var_name):
			# lookup a variable in the table to get its trace name
			return self.table.loc[self.table['formula'] == var_name]['trace_name'].iloc[0]

		def num_outputs(self):
			return len(self.outputs())

		@property
		def comp_graph(self):
			return {'in' : self.ins, 'out' : self.outs}

		def __repr__(self):
			return repr(self.table)

		def forward_mode_der(self):
			'''
			step FORWARDS through the trace table, calculate derivatives along the way in trace_derivs
			
			Returns the Jacobian matrix of derivatives df_i/dx_j for each output function f_i w.r.t. each input variable x_j

			'''
			self.trace_derivs = {self.get_trace_name(x) : np.eye(self.num_vars)[i,:] for i, x in enumerate(self.variables)}
			for row in range(self.num_vars, self.size):
				v = self.table.loc[row]['trace_name']
				d_v_d_parents = np.array([[self.partials[v][in_] for in_ in self.ins[v]]])
				d_parents_d_variables = np.vstack([self.trace_derivs[in_] for in_ in self.ins[v]])
				d_v_d_variables = np.dot(d_v_d_parents, d_parents_d_variables)
				self.trace_derivs[v] = d_v_d_variables
			return np.array([self.trace_derivs[output][0] for output in self.outputs()])

		def reverse_mode_der(self):
			'''
			step BACKWARDS through the trace table, calculate derivatives along the way in trace_derivs
			
			Returns the Jacobian matrix of derivatives df_i/dx_j for each output function f_i w.r.t. each input variable x_j
			
			'''
			self.trace_derivs = {x : np.eye(self.num_outputs())[:,i].reshape(-1,1) for i, x in enumerate(self.outputs())}
			for row in reversed(range(self.size)):
				v, label = self.table.loc[row]['trace_name'], self.table.loc[row]['label']
				if label != 'OUTPUT':
					if self.outs[v] == []:
						d_outputs_d_v = np.zeros(shape=(self.num_outputs(), 1))
					else:
						d_outputs_d_children = np.hstack([self.trace_derivs[out_] for out_ in self.outs[v]])
						d_children_d_v = np.array([[self.partials[out_][v] for out_ in self.outs[v]]])
						d_outputs_d_v = np.dot(d_outputs_d_children, d_children_d_v.T)
				else:
					d_outputs_d_v = np.ones(shape=(self.num_outputs(), 1))
				self.trace_derivs[v] = d_outputs_d_v
			return np.hstack([self.trace_derivs[x] for x in list(map(lambda x : self.get_trace_name(x), self.variables))])

		def tensor_product(self, W, u, v):
			print('TENSOR')
			print(W, u, v)
			assert u.shape[0] == 1 and v.shape[0] == 1
			J, K = u.shape[1], v.shape[1]
			assert W.shape[0] == J and W.shape[1] == K
			return np.array([[W[j,k]*u[0,j]*v[0,k] for k in range(K)] for j in range(J)])

		def double_deriv(self, i):
			v_i = f'v{i}'
			t = self.traces[v_i]
			op, parents, param = t._op, t._parents, t._param
			if len(parents) == 1:
				return math.new_double_deriv_one_parent(parents[0], op, param)
			elif len(parents) == 2:
				return math.new_double_deriv_two_parents(parents[0], op, parents[1])
			else:
				raise ValueError('what')

		def hessian(self):
			'''
			Implements the edge-pushing algorithm?

			

			Returns BOTH the jacobian (first derivative) and hessian (second derivative)

			Requires that the user has traced a scalar-output function f:Rm --> R
			'''
			
			if self.num_outputs() > 1:
				raise ValueError('Hessian can only be calculated for scalar function')

			j = self.forward_mode_der()
			print('derivative', j)
			#self.doubles = self.get_doubles(op, parents)

			print(self.trace_derivs)

			l = self.size
			m = self.num_vars
			W = np.array([[0.]])
			v_bar_T = np.array([[self.table.loc[l - 1]['val']]]) # the initial y-value from the forward pass?
			for i in reversed(range(m+1,l+1)):
				print('~ v', i)
				v_i = f'v{i}'
				D_i = np.array([[self.partials[v_i][in_] for in_ in self.ins[v_i]]])
				print('~~ D', D_i, D_i.shape)
				W = np.dot(W, D_i)
				#W += np.dot(np.dot(v_bar_T, self.double_deriv(i)), self.trace_derivs[f'v{i-1}'])
				W += v_bar_T @ self.double_deriv(i) @ self.trace_derivs[f'v{i-1}']
				v_bar_T = v_bar_T @ D_i
				#W = self.tensor_product(W, D_i, D_i)
				print('~~~ W', W, W.shape)

			return v_bar_T, W






			

			# is this a hack tho #
			# l = self.size
			# for v in self.partials:
			# 	for u in self.partials:
			# 		if u not in self.partials[v]:
			# 			self.partials[v][u] = 0
			#print(self.partials)
			############################################









			# https://mlubin.github.io/pdf/edge_pushing_julia.pdf

			#initialize
			# v_bar = {i : 0.0 for i in range(1,self.size)}
			# v_bar[l] = 1.0
			# H = {i : {j : set([]) for j in range(1, l + 1)} for i in range(1, l + 1)}
			
			# m = self.num_vars
			# for i in reversed(range(m+1, l+1)):
			# 	v_i = f'v{i}'
			# 	print(H)
			# 	print('~ v', v_i)
			# 	print('~~ vbar', v_bar)
			# 	####### pushing step
			# 	for p in H:
			# 		print('~~~ p', p)
			# 		if p <= i and i in H[p] and H[p][i] != set():
			# 			print('ding')
			# 			if p != i:

			# 				for v_j in self.ins[v_i]:
			# 					j = int(v_j[1:])
			# 					if j == p:
			# 						v_p = f'v{p}'

			# 						######### PUSH 3
			# 						new_set = H[p][p].union([self.partials[v_i][v_p]*h_ for h_ in H[p][i]])
			# 						H[p][p] = new_set

			# 					else:

			# 						######### PUSH 1
			# 						new_set = H[j][p].union([self.partials[v_i][v_j]*h_ for h_ in H[p][i]])
			# 						H[j][p] = new_set
			# 						H[p][j] = new_set

			# 			else:
			# 				######### PUSH 2
			# 				for v_j, v_k in combinations_with_replacement(self.ins[v_i], 2):
			# 					j, k = int(v_j[1:]), int(v_k[1:])
			# 					new_set = H[j][k].union([self.partials[v_i][v_k]*self.partials[v_i][v_j]*h_ for h_ in H[i][i]])
			# 					H[j][k] = new_set
			# 					H[k][j] = new_set
			# 	#################################################

			# 	####### creating step
			# 	for v_j, v_k in combinations_with_replacement(self.ins[v_i], 2):
			# 		j, k = int(v_j[1:]), int(v_k[1:])
			# 		print('~~~ j k', j, k)
			# 		t1, t2, t3 = self.traces[v_i], self.traces[v_k], self.traces[v_j]
			# 		new_set = H[j][k].union([v_bar[i]*math.double_deriv(t1, t2, t3)])
			# 		H[j][k] = new_set
			# 		H[k][j] = new_set
			# 	#################################################

			# 	####### adjoint step
			# 	for v_j in self.ins[v_j]:
			# 		j = int(v_j[1:])
			# 		v_bar[j] += v_bar[i]*self.partials[v_i][v_j]
			# 	#################################################
			# print(v_bar)
			# print(H)












			# https://par.nsf.gov/servlets/purl/10039361

			# v_l = f'v{l}'
			# S = {self.size + 1 : set([v_l])}
			# h = {k : { f'v{j+1}' : {f'v{i+1}' : 0.0 for i in range(l)} for j in range(l)} for k in range(1, l + 2)}
			# v_bar = { f'v{i}' : 0.0 for i in range(1,self.size)}
			# v_bar[v_l] = 1.0

			# # step backward through the trace table
			# m = self.num_vars
			# for k in reversed(range(m+1,l + 1)):


			# 	##### make live variable set S
			# 	v_k = f'v{k}'
			# 	#print('v ~', v_k)
			# 	S[k] = S[k + 1]
			# 	if v_k in S[k]:
			# 		S[k].remove(v_k)
			# 	S[k] = S[k].union(self.ins[v_k])
			# 	#print('S ~~', S[k])


			# 	#### update current first derivatives v_bar
			# 	for v in self.ins[v_k]:
			# 		v_bar[v] += self.partials[v_k][v] * v_bar[v_k]
			# 	#print('v bar ~~~', v_bar)


			# 	#### build current hessian layer
			# 	for v_i, v_j in combinations_with_replacement(S[k], 2):
			# 		#if v_i in self.ins[v_k] or v_j in self.ins[v_k]:
			# 		#print('vi vj ~~~~', v_i, v_j)
			# 		a = h[k+1][v_i][v_j]
			# 		b = self.partials[v_k][v_i]*h[k+1][v_j][v_k]
			# 		c = self.partials[v_k][v_j]*h[k+1][v_i][v_k]
			# 		d = self.partials[v_k][v_i]*self.partials[v_k][v_j]*h[k+1][v_k][v_k]
			# 		t1, t2, t3 = self.traces[v_k], self.traces[v_i], self.traces[v_j]
			# 		e = math.double_deriv(t1, t2, t3) #* v_bar[v_k]
			# 		#print(a, b, c, d, e)
			# 		f = a+b+c+d+e
			# 		#print(f)
			# 		h[k][v_i][v_j] = f
			# 		# if v_i in self.variables and v_j in self.variables:
			# 		# 	h[k][v_j][v_i] = f

			# 		#print('h ~~~~~', h[k])
			# j = np.array([[v_bar[self.get_trace_name(v)] for v in self.variables]])
			# h = np.array([[h[m+1][f'v{i+1}'][f'v{j+1}'] for j in range(m)] for i in range(m)])
			# return j, h

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




