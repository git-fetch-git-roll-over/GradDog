import pytest
import numpy as np
from graddog.variable import Variable, get_x, get_xy, get_xyz, function_to_Trace, get_vars

def test_val_reset_error():
    x1 = Variable('x', 2)
    with pytest.raises(TypeError):
        x1.val = 'four'
        
def test_name_reset():
    x1 = Variable('x', 2)
    x1.name = 'x1'
    assert x1.name == 'x1'

def test_val_reset():
    x1 = Variable('x', 2)
    x1.val = 4
    assert x1.val == 4

    
def test_getvars():
    a, b, c = get_vars(['a', 'b', 'c'], [3, 2, 1])
    assert a._val == 3
    assert b._val == 2
    assert c._val == 1
    assert a._formula == 'a'
    assert b._formula == 'b'
    assert c._formula == 'c'
    
def test_string_input():
    with pytest.raises(TypeError):
        x_str = get_x(['CS107'])
        
def test_getx():
    x = get_x([1])
    assert x._val == 1
    assert x._formula == 'x'
   
    
def test_ValueError_gets():
    with pytest.raises(ValueError):
        len1list = [1]
        len2list = [3,4]
        len3list = [5,6,7]
        x_str = get_x(len3list)
        x, y, z = get_xyz(len2list)
        x, y = get_xy(len1list)
        a, b, c, d, e = get_vars(['a', 'b', 'c', 'd', 'e'], len2list)
        
#UI DEFINING COMMENTS IN ALL CAPS
def test_variable_add():
    x1 = Variable('x', 2)
    x2 = Variable('y', 5)
    f = x1 + x2
    assert f.val == 7
    #FORMULA SHOULD BE ACCESSIBLE VIA ATTRIBUTE OF F
    assert f.formula == 'x + y' # and/or contains 'x' 'x' '+'
    #IT COULD HAVE A DIFFERENT (hidden?) ATTRIBUTE TO STORE THE TRACE FORMULA/NAME INFO
    assert f._trace_formula == 'v1 + v2'
    assert x1._trace_id == v1
    assert x2._trace_id == v2
    assert f._trace_id == v3
    
def test_variable_radd():
    x1 = 4
    x2 = Variable('x', 2)
    f = 4+ x2 #x1 + 4
    assert f.val == 6
    assert f.formula == 'x+4'
    
def test_variable_sub():
    x1 = Variable('x', 2)
    x2 = Variable('y', 5)
    f = x2 - x1
    assert f.val == 3
    #FORMULA SHOULD BE ACCESSIBLE VIA ATTRIBUTE OF F
    assert f.formula == 'y - x' # and/or contains 'y' 'x' '-'
    #IT COULD HAVE A DIFFERENT (hidden?) ATTRIBUTE TO STORE THE TRACE FORMULA/NAME INFO
    assert f._trace_formula == 'v2 - v1'
    assert f._trace_id == 'v3'
    
def test_variable_sub_error():
    x1 = Variable('x', 4)
    x2 = 4
    f = x2 - x1
    assert f.val == np.approx(0, abs=1e-6)
    #FORMULA SHOULD BE ACCESSIBLE VIA ATTRIBUTE OF F
    assert f.formula == '4 - x' # and/or contains 'y' 'x' '-'
    #IT COULD HAVE A DIFFERENT (hidden?) ATTRIBUTE TO STORE THE TRACE FORMULA/NAME INFO
    assert f._trace_formula == '4 - v1'
    assert f._trace_id == 'v2'

def test_variable_rsub():
    x1 = Variable('x', 4)
    x2 = 4
    f = x1 - x2
    assert f.val == np.approx(0, abs=1e-6)
    #FORMULA SHOULD BE ACCESSIBLE VIA ATTRIBUTE OF F
    assert f.formula == 'x - 4' # and/or contains 'y' 'x' '-'
    #IT COULD HAVE A DIFFERENT (hidden?) ATTRIBUTE TO STORE THE TRACE FORMULA/NAME INFO
    assert f._trace_formula == 'v1 - 4'
    assert f._trace_id == 'v2'
    
# def test_variable_mul():
#     x1 = Variable('x', 2)
#     x2 = Variable('x', 2)
#     assert str(x1 * x2) == str(Variable('x*x', 4, 4))

# def test_variable_div():
#     x1 = Variable('x', 2)
#     x2 = Variable('x', 2)
#     assert str(x1 / x2) == str(Variable('x/x', 1.0, 0))

# def test_variable_div_error():
#     x1 = Variable('x', 3)
#     x2 = 3
#     assert str(x1/x2) == str(Variable('x/3', 1, (1/3)))

# def test_variable_rdiv():
#     x1 = 3
#     x2 = Variable('x', 3)
#     assert str(x1/x2) == str(Variable('3/x', 1, (-1/3)))

# def test_variable_neg():
#     x1 = Variable('x', 2)
#     x2 = -x1
#     assert "-" + x1._formula ==  x2._formula 
#     assert -x1._val == x2._val
#     assert -x1._der == x2._der

# def test_variable_pow():
#     x1 = Variable('x', 2)
#     x2 = x1 ** 2
#     x3 = x1 ** 0 
#     x4 = x1 ** x1
#     x5 = 2 ** x1
    
#     assert x2._val == 4
#     assert x2._der['x'] == 4
#     assert x3 == 1 
#     assert x2._formula == f"{x1._formula}^2"
#     assert x4._formula == f"{x1._formula}^{x1._formula}"
#     assert x4._val == 4
#     assert x5._val == 4
#     assert x5._formula == str(2) + f"^{x1._formula}"

# def test_polynom():
#     x1 = Variable('x', 5)
#     f = 3*x1**2 + 2*x1 + 5
#     assert f._der['x'] == 32

# def test_string_input():
#     with pytest.raises(TypeError):
#         x_str = Variable('s', 'hello World')
