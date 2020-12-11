About
======

Introduction
------------
In engineering, statistical modelling, and countless other scientific disciplines, derivatives are calculated to measure change in a dynamic system. It is crucial for professionals of all kinds who work with quantitative systems to have access to software that can calculate derivatives quickly, efficiently, and with a simple-to-use interface. That is why ``TensorFlow`` exists. Our software package, ``GradDog``, will try to do the same.

``GradDog`` performs automatic differentiation to machine precision, at a low computational cost. This documentation walks through some of the mathematics of automatic differentiation, as well as some basic information about the usage of our package.

.. image:: ../../images/dog.jpg
  :width: 600

Background
----------
Computing gradients the old-fashioned way (by hand) is certainly feasible for many mathematical functions that appear in many applications. A key step in almost every derivative is known as the chain rule, which applies whenever our function's inner structure is a composition of functions, e.g.,

:math:`\\f(x, y) = g(h(x, y))`

Here's an example of computing the gradients of the function 

:math:`\\f(x, y) = \sin(2x + 3y)e^{-x^2}`

The partial derivative :math:`\partial f/\partial x` can be calculated using the product rule and the chain rule:

:math:`\partial f/\partial x = \sin(2x + 3y)(-2x)e^{-x^2} + 2\cos(2x + 3y)e^{-x^2}`

The partial derivative :math:`\partial f/\partial y` only needs to be calculated using the chain rule, since a term only involving x is constant with respect to y:

:math:`\partial f/\partial y = 3\cos(2x + 3y)e^{-x^2}`


We can represent the function's individual operations in a graphical format:

By parsing the structure of the function :math:`\\f(x, y)` into its atomic building blocks, we can construct a trace table from which we can read the results of the chain rule at every step of the computation.

.. image:: ../../images/func_graph.jpeg
  :width: 600

The fact that this process is **automatic** comes very much in handy when computing gradients of more complex functions, for example:

:math:`f(x,y) = 3sin(e^{6x - ln(4y^3 - 3x^2)} + 17\sqrt{y^5-tanh(x)})17(4x^{1/3}+29yln(x - y))^{7/19}`

Yeah, we would prefer not to do that by hand. 

The ``GradDog`` package is able to do automatic differeniation to any numerical functions in `forward mode` and another new feature we have have included is the `reverse mode`. Please go to `Reverse Mode <https://graddog.readthedocs.io/en/latest/reverse_mode.html>`_ section for more details. 


Broader Impact and Inclusivity Statement
----------------------------------------
The ``GradDog`` package is able to calculate both derivateives through automatic differentiation in both `forward mode` and `reverse mode`. It calculates to machine precision and saves a great amount of computational costs compared to both conventional finite differences and symbolic derivatives methods. However, one downside to note is that ``GradDog`` does not keep track of the mathmatical formula that composes the derivative matrix. If the user were a student, who were trying to use this package for education purpose to understand the process of automatic differentiation, this package might mitigate the overall learning experience. ``GradDog`` is simply designed and developed to provide a convenient avenue to calculate derivatives given any numerical functions. It is meant to act as a small tool to help to solve users' questions.  In writing our documentation and designing our package, we have attempted to reduce the number of assumptions we are making about a user's background.  We do not believe that this package has risks of any major negative impacts, as it does not, for example, replace any existing jobs or access sensitive user information.

The ``GradDog`` package is an open source project and welcomes any contributors from all over the world with different background. The four major developers of ``GradDog`` are either undergraduate or graduate students at Harvard University, an environment that promotes diversity. We will treat every pull request equally, with exactly the same review and approval process. Each time, when a pull request is created by an outside contributor, all the main developers will schedule a time to review it together. We will be making every effort to make sure we are only examing the code based on its idea rather than who initiated the request. If there are any ambiguities or issues about the code, we will reach out to the contributors and make sure to address the misunderstandings or any questions they have. This serves our larger goal of contributing to the movement to make open source code development more inclusive.





