AutoDiff: Software Library Documentation
==================================================

..
    Authors:  Peyton Benac, Max Cembalest, Ivan Shu and Seeam Shahid Noor
    ^^^^^^^^^^^^^^^^
Introduction
============
In engineering, statistical modelling, and countless other scientific disciplines, derivatives are calculated to measure change in a dynamic system. It is crucial for professionals of all kinds who work with quantitative systems to have access to software that can calculate derivatives quickly, efficiently, and with a simple-to-use interface. That is why ``TensorFlow`` exists. Our software package, ``AutoDiff``, will try to do the same.

``AutoDiff`` performs automatic differentiation to machine precision, as well as improves computational costs. This documentation walks through some of the mathematics of automatic differentiation, as well as some basic information about the usage of our library.

.. image:: ../../images/dog.jpg
  :width: 600

Background
==========
Computing gradients the old-fashioned way (by hand) is certainly feasible for many mathematical functions that appear in many applications. A key step in almost every derivative is known as the chain rule, which applies whenever our function's inner structure is a composition of functions, e.g.,

:math:`\\f(x, y) = g(h(x, y))`

Here's an example of computing the gradients of the function 

:math:`\\f(x, y) = \sin(2x + 3y)e^{-x^2}`

The partial derivative $\partial f/\partial x$ can be calculated using the product rule and the chain rule:

:math:`\partial f/\partial x = \sin(2x + 3y)(-2x)e^{-x^2} + 2\cos(2x + 3y)e^{-x^2}`

The partial derivative $\partial f/\partial y$ only needs to be calculated using the chain rule, since a term only involving x is constant with respect to y:

:math:`\partial f/\partial y = 3\cos(2x + 3y)e^{-x^2}`


We can represent the function's individual operations in a graphical format:

By parsing the structure of the function :math:`\\f(x, y)` into its atomic building blocks, we can construct a trace table from which we can read the results of the chain rule at every step of the computation.

.. image:: ../../images/func_graph.jpeg
  :width: 600

The fact that this process is **automatic** comes very much in handy when computing gradients of more complex functions, for example:

:math:`f(x,y) = 3sin(e^{6x - ln(4y^3 - 3x^2)} + 17\sqrt{y^5-tanh(x)})17(4x^{1/3}+29yln(x - y))^{7/19}`

Yeah, we would prefer not to do that by hand.

How to use package
==================

How to install
--------------

Go to the directory from which you want to run this package, and then open a command line prompt. 

* Visit ``https://github.com/git-fetch-git-roll-over/AutoDiff.git`` and follow the cloning instructions to clone a copy of the repository. This will create an ``AutoDiff`` directory.  (Key command: ``git clone https://github.com/git-fetch-git-roll-over/AutoDiff.git``)
* ``cd AutoDiff`` to go inside the directory
* ``virtualenv autodiff`` to create a virtual environment (It is optional but recommended to use a virtual environment.)
* ``source autodiff/bin/activate`` to activate the environment
* ``pip install -r requirements.txt`` to install the necessary dependencies
* ``cd forward_mode`` to go inside the directory containing the modules
* create your own driver script, following the basic demos below
Basic Demo
----------

Say there is a simple function and we want to get the derivative at :math:`x=4`:

:math:`f(x) = sin(x) + cos(x)`

First, import the full ``graddog`` module (recommended alias: ``gd``) and the functions from the ``functions`` module::

    import graddog as gd
    from graddog.functions import sin, cos, tan, exp, log

Then, create the function you would like to evaluate the derivative of (See `this link <http://introtopython.org/introducing_functions.html>`_ for more information on creating your own Python functions)::
    def f(x):
        return x**2 + 3x + exp(x, base=2)*sin(2*x)
Choose a "seed value", which is the value at which the derivative will be evaluated::
    seed = 3
   
Last, perform the ``trace`` on the function and the seed. (This function has an optional ``mode`` parameter which can be used to select either the forward or reverse mode of automatic differentiation.)::

    gd.trace(f, seed)

    >>> Computing forward mode
    >>>[[22.81331607]]

Both the values and the derivative values are wrapped up in attributes belonging to these objects. For example::

    f1.val
    >>> -0.7568024953079282
    f2.val
    >>> -0.6536436208636119
    f1.der
    >>> -0.6536436208636119
    f2.der
    >>> 0.7568024953079282
    f.der
    >>> 0.10315887444431626

Software organization
=====================

Directory Structure
-------------------

With our main directory ``cs107-FinalProject``, our package is organized as follows:

* dependencies in ``requirements.txt``
* continuous integration information in ``.travis.yml`` and coverage information in ``codecov.yml``
* documentation in ``\docs``
* test suite in ``\tests``
* code in ``\graddog``
* images in ``\images``

Basic Modules
-------------

For users::
    Our code consists of 3 user-facing modules
    * __init__, defining the ``trace`` function used to return derivatives
    * functions, definining elementary functions that are compatible with the trace function.
    * tools, defining useful tools to plot derivatives and estimate extrema.
    
For developers::
    Our code consists of (4,5,6) total modules
    * __init__
    * functions
    * compgraph
    * math
    * trace
    * tools
    


How to Test
------------

Both ``TravisCI`` and ``CodeCov`` are used and our test suite will live inside our repo in a ``\tests`` directory. Our project currently has 100% coverage.

Future Installation 
-------------------

As of now, we primarily distribute our code by having clients clone our github repository, but we hope to also make it pip-installable. For the future, we plan to package our project using `wheels <https://www.python.org/dev/peps/pep-0427/>`_. This should make it easily pip installable. The wheel name for our package is ``graddog-1.3-py3-none-any.whl``.


Implementation Details
======================

Descriptions
------------



For installation, we have chosen wheels because they are smaller in size than a source distribution (sdist) file for the same package, and therefore are faster. They are also faster because installing from a wheel allows one to skip the build step required to install source distributions. Wheels are also very consistent, automatically generate the correct .pyc files for the interpreter, and require no compiler (even for compiled extension modules).


Extension
=============
Reverse Mode
----------

Higher Order Derivatives
-------------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

    license
    help


..
    Indices and tables
    ==================

    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`
