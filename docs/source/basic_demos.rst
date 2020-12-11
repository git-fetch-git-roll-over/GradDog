Basic Demos
===========

.. role:: raw-html(raw)
   :format: html

First, import the full ``graddog`` module (recommended alias: ``gd``) and the functions from the ``functions`` module::

    import graddog as gd
    from graddog.functions import sin, cos, tan, exp, log

:math:`R`:raw-html:`&rarr;`:math:`R`:
--------------------------------------
Then, create the function you would like to evaluate the derivative of (See `this link <http://introtopython.org/introducing_functions.html>`_ for more information on creating your own Python functions)::

    def f0(x):
        return x**2 + 3*x + exp(x, base=2)*sin(2*x)

Now you can perform the ``trace`` on the function with a "seed value", i.e. the values of the inputs at which you would like to evaluate the derivative. Here, we choose a seed value of 3. For convention, we denote the derivative of f as ``f_``::

 f_ = gd.trace(f0, 3, mode = 'forward')
 >>> Computing forward mode derivative...
 f_
 >>> [[22.81331607]]

This function has an optional ``mode`` parameter which can be used to select either the forward or reverse mode of automatic differentiation. The default mode is decided based on the computational costs of each mode for a particular function. For this tutorial page, we will specify forward mode for all demonstrations, although this parameter is optional. To see reverse mode examples, go to the Reverse Mode page of the documentation)::

    


:math:`R^{m}`:raw-html:`&rarr;`:math:`R`:
------------------------------------------

Next, let's try an function with more variables. The idea is the same but with the ``seed`` variable as a ``list`` containing the values for the variables. Here, we have 2 variables named :math:`x` and :math:`y` and the values are 1 and 2::

    seed1 = [1,2]
    def f1(x, y):
        return x*y + exp(x*y)

    f_ = gd.trace(f1, seed1, mode = 'forward')
    >>> Computing forward mode derivative...
    f_
    >>> [[16.7781122  8.3890561]]



:math:`R`:raw-html:`&rarr;`:math:`R^{n}`:
-----------------------------------------

What about the function mapping from one-deminsional input to multidemensional outputs. ``graddog`` can also simply achieve that with the following example::

    seed2 = 3
    def f2(x):
        return [x**2, x**3, x**4]

    f_ = gd.trace(f2, seed2, mode = 'forward')
    >>> Computing forward mode derivative...
    f_
    >>> [[  6.]
         [ 27.]
         [108.]]

This example creates a single variable :math:`x` with value equal to 3 (``seed2 = 3``) and the output is :math:`x^{2}`, :math:`x^{3}` and :math:`x^{4}` casted in a ``list``. We then passed both ``f2`` and ``seed2`` into ``gd.trace()``, which computes the derivative matrix in forward mode. Forward mode is more effecient in this case as our output demension is greater than the input demension. 


:math:`R^{m}`:raw-html:`&rarr;`:math:`R^{n}`:
---------------------------------------------

Now, you might be thinking can our ``graddog`` calculate derivatives for functions mapping between mulitdemensional input and output? Absolutely it can, our ``graddog`` is well-trained. To illustrate, we can construct the same way for both ``seed3`` variable and ``f3`` function as the following demo ::

    seed3 = np.ones(3)
    def f3(x, y, z):
        return [exp(-(sin(x)-cos(y))**2), sin(-log(x)**2+tan(z))]

    f_ = gd.trace(f3, seed3, mode = 'forward')
    >>> Computing forward mode derivative...
    f_
    >>> [[-0.29722477 -0.46290015  0.        ]
         [ 0.          0.          0.04586154]]


Debugging:
---------------------------------------------
For debugging purposes, we also added an optional parameter ``verbose``, which displays for the user the trace table from the computational trace, as well as the relevant assumptions about the input and output types and dimensions::


    f_ = gd.trace(f1, seed1, mode = 'forward', verbose = True)
    >>> Inferred 2-dimensional input
    >>> Scanning the computational graph...
    >>> ...inferred the inputs are variables...
    >>> ...finished
    >>> Inferred 1-dimensional output
    >>> [v5]
    >>> Computing forward mode derivative...
    >>>       trace_name  input output  formula      val  partial1 partial2
    >>>     0         v1   True  False       v1        1  1.000000      NaN
    >>>     1         v2   True  False       v2        2  1.000000      NaN
    >>>     2         v3  False  False    v1*v2        2  2.000000        1
    >>>     3         v4  False  False  exp(v3)  7.38906  7.389056      NaN
    >>>     4         v5  False   True    v3+v4  9.38906  1.000000        1

    f_
    >>> [[16.7781122  8.3890561]]






