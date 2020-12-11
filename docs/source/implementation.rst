Implementation Details
======================

Core data structure
--------------------
Our core data structures are ``numpy.array``, ``list``, ``dictionary`` and ``pandas.DataFrame``. 

Both ``numpy.array`` and ``list`` can be used to contain the ``seed`` variable when there are multiple inputs to the customized functions. ``numpy.array`` here, however, plays a even more important role in our vectorized implementation of forward mode and reverse mode.

We chose ``pandas.DataFrame`` to store all the information during the forward-pass trace of a function. While the fact that it is a bit heavy-weight, it is very easy to store all the traces information during package development. At the same time, it is easy for the users to visualize all the traces information in a table format. 


Core classes
-------------
Our main classes are ``Trace``, ``Variable``, ``Ops`` and ``CompGraph``.

``Trace`` class is a base class that overloads all the operational dunder methods, such as addition, substraction, multiplication, division, negation, power with other ``Trace`` instances, ``int`` or ``float``.

``Variable`` is a subclass of ``Trace`` class with its value attribute set by a user's seed and its derivative attribute defaulted to 1.

``Ops`` class lives in ``math.py`` module and it stores all the mathematical rules defining our currently implemented functions.

``CompGraph`` class is the engine that does all the computational work and stores all the relevant traces information. It implements the ``Singleton`` design pattern so that ease-of-use between files is simplified and so that only one instance is ever created to reduce computational overheard. Both forward mode and reverse mode automatic differentiation are computed from the CompGraph class, as well as the second order derivative.

Attributes
--------------------
To be able to calculate the partial derivatives, ``GradDog`` contains a couple of very important attributes in some core classes described above. 

``Trace`` class:
    | **_formula**: ``str`` that describes the trace element's formula
    | **_val**: ``int`` or ``float``stores the value of the trace element.  
    | **_der**: ``dictionary`` storing the partial derivatives of the trace with respect to its predecessing traces.
    | **_parents**: ``list`` that stores references to the trace's predecessors ([] if the trace is a Variable)
    | **_op**: ``str`` stores the operation used to create the trace element (``None`` if the trace is a Variable)
    | **_param**: ``str``, used to store any numerical parameter used to create the trace (for example, the base of a logarithm). ``None`` by default.
    | **_trace_name**: ``str``, stores the unique trace name for each ``Trace`` instance when they are created, as determined by the CompGraph.


``CompGraph`` class:
    | **size**: ``int``, stores the amount of intermediate trace elements created when parsing a function
    | **num_vars**: ``int``, stores the number of variables the function takes as input
    | **var_names**: ``list``, stores all the variable names.
    | **outs**: ``dictionary``, stores the "out" direction of the computational graph, i.e. the children of each trace element
    | **ins**: ``dictionary``, stores the "in" direction of the computational graph, i.e. the parents of each trace element
    | **traces**: ``dictionary``, stores references to the Trace objects with their trace names as keys
    | **partials**: ``dictionary``, containing its current trace name as the key and its partial derivatives as the value. 
    | **table**: ``pandas.DataFrame``, contains all the trace information used to compute derivatives and display information about the function to the user.

External Dependencies
---------------------
Although our ``GradDog`` is very intelligent, it needs a couple of external packages in order to do automatic differentiation. The external dependencies are ``numpy``, ``matplotlib``, ``pandas``, and ``pytest``.


Elementary functions
---------------------
``GradDog`` can calculate the derivatives for a number of elementary functionis. They are included, but not limited to, :math:`sin, arcsin, cos, arccos, tan, arctan, exp, log, sinh, cosh, tanh, sqrt` and :math:`sigmoid`. All the functions can be used by a simple import statement and please see our tutorial for more instructions. 


Installation
-------------
For installation, we have chosen ``wheels`` because they are smaller in size than a source distribution (sdist) file for the same package, and therefore are faster. They are also faster because installing from a wheel allows one to skip the build step required to install source distributions. Wheels are also very consistent, automatically generate the correct .pyc files for the interpreter, and require no compiler (even for compiled extension modules).



