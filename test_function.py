import pytest
import numpy as np
from code.variable import Variable
from code.function import Function

def test_sin():
    x1 = Variable('x', 3)
    g = Function.sin(x1)
    assert g.der == Function.cos(x1).val
    
def test_string_input():
    with pytest.raises(TypeError):
        x_str = Variable('s', 'CS107')
        
def test_log_base2():
    x = Variable('x', 32)
    base = 2
    f = Function.log(x, base=base)
    assert f.val == 5
    assert f.der == 1/(x.val * np.log(base))
    
def test_exp_base2():
    x = Variable('x', 5)
    base = 2
    f = Function.exp(x, base=base)
    assert f.val == 32
    assert f.der == (base**x.val)*np.log(base)