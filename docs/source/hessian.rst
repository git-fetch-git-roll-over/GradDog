Hessian (Extension)
========================


The Hessian is a matrix representing the second derivative of a function with respect to each pair of its input variables.

Example::

    import graddog as gd
    from graddog.functions import sin, cos, tan, exp, log

    seed = [1,2]
    def f(x, y):
        return x*y + exp(x*y)
    f_, f__ = gd.trace(f, seed, return_second_deriv = True)
    >>> Computing reverse mode
    f_
    >>>[[16.7781122  8.3890561]]
    f__
    >>>[[29.5562244 23.1671683]
        [23.1671683  7.3890561]]





