import pytest
import numpy as np
from graddog.trace import Trace, get_x, get_xy, get_xyz, function_to_Trace, get_vars
from graddog.functions import sin, cos, tan, exp, log

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

