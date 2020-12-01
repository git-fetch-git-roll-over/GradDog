import pytest
import numpy as np
from forward_mode.trace import Variable, Trace

def test_name_reset():
    x1 = Variable('x', 2)
    x1.name = 'new_x'
    assert str(x1) == str(Variable('new_x', 2))

def test_val_reset():
    x1 = Variable('x', 2)
    x1._val = 4
    assert str(x1) == str(Variable('x', 4))

def test_val_reset_error():
    x1 = Variable('x', 2)
    with pytest.raises(TypeError):
        x1._val = 'four'

def test_variable_add():
    x1 = Variable('x', 2)
    x2 = Variable('x', 2)
    assert str(x1 + x2) == str(Variable('x+x', 4, 2))

def test_variable_radd():
    x1 = 4
    x2 = Variable('x2', 2)
    assert str(x1+x2) == str(Variable('x2+4', 6))

def test_variable_sub():
    x1 = Variable('x', 2)
    x2 = Variable('x', 2)
    assert str(x1-x2) == str(Variable('x-x', 0, 0))

def test_variable_sub_error():
    x1 = Variable('x', 4)
    x2 = 4
    assert str(x1-4) == str(Variable('x-4', 0))

def test_variable_rsub():
    x1 = 4
    x2 = Variable('x', 4)
    assert str(x1-x2) == str(Variable('-x+4', 0, -1))


def test_variable_mul():
    x1 = Variable('x', 2)
    x2 = Variable('x', 2)
    assert str(x1 * x2) == str(Variable('x*x', 4, 4))

def test_variable_div():
    x1 = Variable('x', 2)
    x2 = Variable('x', 2)
    assert str(x1 / x2) == str(Variable('x/x', 1.0, 0))

def test_variable_div_error():
    x1 = Variable('x', 3)
    x2 = 3
    assert str(x1/x2) == str(Variable('x/3', 1, (1/3)))

def test_variable_rdiv():
    x1 = 3
    x2 = Variable('x', 3)
    assert str(x1/x2) == str(Variable('3/x', 1, (-1/3)))

def test_variable_neg():
    x1 = Variable('x', 2)
    x2 = -x1
    assert "-" + x1._formula ==  x2._formula 
    assert -x1._val == x2._val
    assert -x1._der == x2._der

def test_variable_pow():
    x1 = Variable('x', 2)
    x2 = x1 ** 2
    x3 = x1 ** 0 
    x4 = x1 ** x1
    x5 = 2 ** x1
    
    assert x2._val == 4
    assert x2._der['x'] == 4
    assert x3 == 1 
    assert x2._formula == f"({x1._formula}^2)"
    assert x4._formula == f"({x1._formula}^{x1._formula})"
    assert x4._val == 4
    assert x5._val == 4
    assert x5._formula == str(2) + f"^({x1._formula})"

def test_polynom():
    x1 = Variable('x', 5)
    f = 3*x1**2 + 2*x1 + 5
    assert f._der['x'] == 32

def test_string_input():
    with pytest.raises(TypeError):
        x_str = Variable('s', 'hello World')
