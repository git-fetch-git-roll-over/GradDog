import pytest
import numpy as np
from forward_mode.variable import Variable

def test_variable_add():
    x1 = Variable('x', 2)
    x2 = Variable('x', 2)
    assert str(x1 + x2) == str(Variable('x+x', 4, 2))

def test_variable_sub():
    x1 = Variable('x', 2)
    x2 = Variable('x', 2)
    assert str(x1 - x2) == str(Variable('x-x', 0, 0))

def test_variable_mul():
    x1 = Variable('x', 2)
    x2 = Variable('x', 2)
    assert str(x1 * x2) == str(Variable('x*x', 4, 4))

def test_variable_div():
    x1 = Variable('x', 2)
    x2 = Variable('x', 2)
    # assert str(x1 / x2) == str(Variable('x/x', 1, 0))

def test_variable_neg():
    x1 = Variable('x', 2)
    x2 = -x1
    assert "-" + x1.name ==  x2.name 
    assert -x1.val == x2.val
    assert -x1.der == x2.der

def test_variable_pow():
    x1 = Variable('x', 2)
    x2 = x1 ** 2
    x3 = x1 ** 0 
    x4 = x1 ** x1
    x5 = 2 ** x1
    assert x2.val == 4
    assert x2.der == 4
    assert x3 == 1 
    assert x2.name == x1.name + "^" + str(2)
    assert x4.name == x1.name + "^" + x1.name 
    assert x4.val == 4
    assert x5.val == 4
    assert x5.name == str(2) + "^" + x1.name

def test_polynom():
    x1 = Variable('x', 5)
    f = 3*x1**2 + 2*x1 + 5
    assert f.der == 32

def test_string_input():
    with pytest.raises(TypeError):
        x_str = Variable('s', 'hello World')
