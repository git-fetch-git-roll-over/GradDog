import numpy as np
import pandas as pd
import graddog.calc_rules as calc_rules

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

			# stores the trace_names of the Trace objects that are the current inputs and outputs of the computational graph
			# This is for convenience. These attributes could be calculated based on self.ins and self.outs, but this is easier
			self.outputs = []
			self.inputs = []

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
				self.inputs.append(new_trace_name)
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
			formula, val, parents, op, param = trace._formula, trace._val, trace._parents, trace._op, trace._param

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

			# calculate partial derivatives
			partial_derivs_list = self.calculate_partial_deriv_list(new_trace_name, op, parents, param)

			# update trace table
			self.add_trace_table_row(new_trace_name, label_string, formula, val, partial_derivs_list)
			
			return new_trace_name

		def add_trace_table_row(self, new_trace_name, label_string, formula, val, partial_derivs_list):
			# add new row to the trace table
			self.table.loc[self.size - 1] = [new_trace_name, label_string, formula, val] + partial_derivs_list

		def calculate_partial_deriv_list(self, new_trace_name, op, parents, param):
			############# calculate the partial derivatives of a trace with respect to its children
			# save these partial derivatives in the dictionary self.partials for use in forward and reverse mode

			# reresent the partial derivatives as a 2-element list to go in the trace table
			# if the new trace has two parents (like v3 = v1*v2), then the partial derivs will be the derivs w.r.t. v1 and v2
			# if the new trace has one parent (like v5 = sin(v4)), then the partial derivs will be the deriv w.r.t. v4, followed by 0
			
			########################################################################
			if op: #only enters this if statement when the new trace is not a variable
				try:
					t, other = parents[0], parents[1]
				except IndexError:
					t, other = parents[0], param
				partial_der = calc_rules.deriv(t, op, other)
			else: # the derivative of a variable w.r.t. itself is 1
				partial_der = {new_trace_name : 1.0}
			self.partials[new_trace_name] = partial_der
			partial_derivs_list = list(partial_der.values())
			if len(partial_derivs_list) == 1:
				partial_derivs_list.append(0.0)
			return partial_derivs_list

		def new_trace_name(self):
			# creates trace names v1, v2, v3, etc.
			self.size += 1
			return 'v' + str(self.size)

		def get_variable_row(self, var_name):
			return int(self.table.loc[self.table['formula'] == var_name]['trace_name'].iloc[0][1:]) - 1

		def forward_mode_der(self):
			self.outputs = self.table.loc[self.table['label'] == 'OUTPUT']['trace_name'].values
			# stores d_trace/d_x for each trace for each input variable x, initialized with 1s and 0s for the derivatives of variables
			trace_derivs = {self.get_trace_name(x) : np.eye(self.num_vars)[i,:] for i, x in enumerate(self.var_names)}
			# step FORWARDS through the trace table 
			# if there are n variables (for example, 3 variables x, y, and z)
			# then start at row n + 1 (for example, 4) in the trace table
			for row in range(self.num_vars, self.size):
				trace_name = self.table.loc[row]['trace_name']
				d_trace_d_chilren = np.array([[self.partials[trace_name][in_] for in_ in self.ins[trace_name]]])
				d_children_d_vars = np.vstack([trace_derivs[in_] for in_ in self.ins[trace_name]])
				trace_derivs[trace_name] = np.dot(d_trace_d_chilren, d_children_d_vars)
			return np.array([trace_derivs[output][0] for output in self.outputs])

		def reverse_mode_der(self):
			self.outputs = self.table.loc[self.table['label'] == 'OUTPUT']['trace_name'].values
			# stores d_f/d_trace for each trace for each output term f
			trace_derivs = {x : np.eye(len(self.outputs))[:,i].reshape(-1,1) for i, x in enumerate(self.outputs)}

			# step BACKWARDS through the trace table 
			# start at the last row 
			# and as you step backwards, skip all the rows labelled 'OUTPUT'
			for row in reversed(range(self.size)):
				trace_name = self.table.loc[row]['trace_name']
				label = self.table.loc[row]['label']

				# skip the rows labelled 'OUTPUT'
				if label != 'OUTPUT':

					# if this trace element leads to anything at all in the outputs, calculate derivatives
					if self.outs[trace_name] != []:
						d_outs_d_children = np.hstack([trace_derivs[out_] for out_ in self.outs[trace_name]])
						d_children_d_trace = np.array([[self.partials[out_][trace_name] for out_ in self.outs[trace_name]]])
						d_outs_d_trace = np.dot(d_outs_d_children, d_children_d_trace.T)

					# reaches this if statement if this trace has no outputs, but it's not an output
					else: #i.e., it was a variable that was used in the calculation
						#so the derivatives of the outputs with respect to this trace are all zero
						d_outs_d_trace = np.zeros(shape=(len(self.outputs), 1))
					
					trace_derivs[trace_name] = d_outs_d_trace
			return np.hstack([trace_derivs[x] for x in list(map(lambda x : self.get_trace_name(x), self.var_names))])

		def get_trace_name(self, var_name):
			# lookup a variable in the table to get its trace name
			return self.table.loc[self.table['formula'] == var_name]['trace_name'].iloc[0]

		@property
		def comp_graph(self):
			return {'in' : self.ins, 'out' : self.outs}

		def __repr__(self):
			return repr(self.table)

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
			print(CompGraph.instance.forward_mode_der())

	def reverse_mode():
		if CompGraph.instance:
			print(CompGraph.instance.reverse_mode_der())

	def add_trace(trace):
		if not CompGraph.instance:
			CompGraph.instance = CompGraph.__CompGraph()
		return CompGraph.instance.add_trace(trace)



