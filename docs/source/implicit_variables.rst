Implicit Variables
==================

.. role:: raw-html(raw)
   :format: html

Instead of giving explicitly the variable names, we could also implicitly pass in a vector without listing the variable names. Our ``GradDog``  package could solve this issue with no problems by creating the variables on its own. Here comes a demo and as always, we import our ``graddog`` module first::

    import graddog as gd
    from graddog.functions import sin, cos, tan, exp, log

    seed4 = [1,2,3]
    def f4(v):
        return v[0]+3*v[2]**2

    gd.trace(f4, seed4)

    >>> Computing reverse mode derivatives...
    >>> [[ 1.  0. 18.]]

So here we are creating variable ``seed4 = [1,2,3]`` and the parameter `v` in the function `f4` indicates that it is ``sequence`` object. This implicitly tells our ``GradDog`` that to create a list of variables of :math:`x_{1}`, :math:`x_{2}` and :math:`x_{3}` with specifed values in ``seed4``. Our `f4` specifically defines that the funciton output is :math:`x_{1} + 3\times x_{3}^{2}`. Similarly, we could also create a function mapping from :math:`R^{m}`:raw-html:`&rarr;`:math:`R^{n}`: 
::

    seed5 = [1,2,3]
    def f5(v):
        return [v[0]+3*v[2]**2, v[1]-v[0], v[2]+sin(v[1])]

    gd.trace(f5, seed5)

    >>> Computing forward mode derivatives...
    >>> [[ 1.          0.         18.        ]
        [-1.          1.          0.        ]
        [ 0.         -0.41614684  1.        ]]


We could also apply same function to differnt variables by simply writing out the formula on the ``sequence`` object `v` itself. For example, ``v**2 + 2*v + 1`` means that we are applying :math:`f(x) = x^{2} + 2 \times x + 1` to all the variables created. ::

    seed6 = np.array([1,2,3])
    def f6(v):
        return v**2 + 2*v + 1

    gd.trace(f6, seed6)

    >>> Computing forward mode derivatives...
    >>> [[4. 0. 0.]
        [0. 6. 0.]
        [0. 0. 8.]]













