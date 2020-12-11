Reverse Mode (Extension)
========================

What's great about our ``GradDog`` package is that it can perform either `forward mode` and `reverse mode` of automatic differentiation with the ``gd.trace`` function.

The user defines a function ``f: Rm --> Rn`` and a variable ``seed`` and passes them into ``gd.trace()``, with an optional parameter ``mode``, which can be set to ``forward`` or ``reverse``. If unspecified, the algorithm uses the dimensions of the inputs and outputs of ``f`` to determine which mode to compute in: if m > n, reverse mode is more efficient, and if m <= n, forward mode is more efficient::

    import graddog as gd
    from graddog.functions import sin, cos, tan, exp, log

    seed = 3
    def f(x):
        return x**2 + 3*x + exp(x, base=2)*sin(2*x)

    f_ = gd.trace(f, seed, mode='forward')
    >>> Computing forward mode derivatives...

    print(f_)
    >>> [[22.81331607]]

Now switching to ``mode='reverse'``::

    f_ = gd.trace(f, seed, mode='reverse')
    >>> Computing reverse mode derivatives...
    
    print(f_)
    >>> [[22.81331607]]

See? Our ``GradDog`` is very smart.

