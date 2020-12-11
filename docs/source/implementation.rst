Implementation Details
======================

Core data structure
--------------------
Our core data structures are ``numpy.array``, ``list``, ``dictionary`` and ``pandas.DataFrame``. 

We chose ``pandas.DataFrame`` to store all the function traces. While the fact that it is a bit heavy-weight, it is very easy to store all the traces information during package development. At the same time, it is easy for the users to visualize all the traces information in a table format. 

Both ``numpy.array`` and ``list`` can be used to contain the ``seed`` variable when there are multiple inputs to the customized functions. ``numpy.array`` here, however, plays a even more important role by storing the derivative matrix and conducting corresponding matrix operations.

To better store partial derivatives, we decided to use ``dictionary`` as it is flexible to index with specified `trace` name to obtain the partial derivatives. 

Core classes
-------------
Our main classes are ``Trace``, ``Variable``, ``Ops`` and ``CompGraph``.

``Trace`` class is a base class that overloads all the operational dunder methods, such as addition, substraction, multiplication, division, negation, power with other ``Trace`` instances, ``int`` or ``float``.

``Variable`` is a subclass of ``Trace`` class with a few default attributes values, which will be further discussed in the following section.

``Ops`` class lives in ``math.py`` module and it stores all the partial derivative rules for elementary functions or composite functions that utilize the chain rules. 

``CompGraph`` class is the enginee that does all the computational work and stores all the relevant traces information. It implemented ``Singleton`` class structure and for each function, only one instance is ever created. This removes the trouble of redundent instantiations by any mistakes. 

Important attributes
--------------------
To be able to calculate the partial derivatives, ``GradDog`` contains a couple of very important attributes in some core classes described above. 

``Trace`` class:
    | **_formula**: ``str`` that describes the variable formula. 
    | **_val**: ``int`` or ``float``stores the value of the variable.  
    | **_der**: ``dictionary`` storing the derivatives with its ``_formula`` as the key.  
    | **_parents**: ``list`` that stores predecessing traces.  
    | **_op**: ``str`` stores the operation and ``None`` is the default value.  
    | **_param**: ``str``, used to customize values for ``log()`` and ``exp()`` functions and ``None`` is the default value.  
    | **_trace_name**: ``str``, stores the trace namefor each ``Trace`` instances when they are created.   

``Variable`` class (subclass of ``Trace``):
    | **_der**: ``dictionary``, default as ``{'name':1.0}`` and here `name` is the name of the variable and 1.0 is its default derivative.  
    | **_parent**: ``list``, starting out as a empty.


``CompGraph`` class:
    | **size**: ``int``, stores the amount of intermediate steps, also known as traces.
    | **num_vars**: ``int``, stores the number of variable created.
    | **var_names**: ``list``, stores all the variable names.
    | **outs**: ``dictioary``, stores the current trace name as the key and its successing traces names as the values.
    | **ints**: ``dictionary``, stores the current trace name as the key and its predecessing trace names as the values. 
    | **traces**: ``dictionary``, storing the currenct trace name as the key and the actual trace instance as its value. 
    | **partials**: ``dictionary``, containing its current trace name as the key and its partial derivatives as the value. 
    | **table**: ``pandas.DataFrame``, that contains all the traces information in the order when they are created. 

External Dependencies
---------------------
Although our ``GradDog`` is very intelligent, it needs a couple of external packages in order to do automatic differentiation. The external dependencies are ``numpy``, ``matplotlib``, ``numbers``, ``collections``, ``itertools`` and ``pandas``. 


Elementary functions
---------------------
``GradDog`` can calculate the derivatives for a number of elementary functionis. They are included, but not limited to, :math:`sin, arcsin, cos, arccos, tan, arctan, exp, log, sinh, cosh, tanh, sqrt` and :math:`sigmoid`. All the functions can be used by a simple import statement and please see our tutorial for more instructions. 


Installation
-------------
For installation, we have chosen ``wheels`` because they are smaller in size than a source distribution (sdist) file for the same package, and therefore are faster. They are also faster because installing from a wheel allows one to skip the build step required to install source distributions. Wheels are also very consistent, automatically generate the correct .pyc files for the interpreter, and require no compiler (even for compiled extension modules).



