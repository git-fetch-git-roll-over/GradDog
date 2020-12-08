import pytest
import numpy as np
from graddog.trace import Trace
from graddog.functions import sin, cos, tan, exp, log
from graddog.variable import get_x, get_xy, get_xyz, function_to_Trace, get_vars
from graddog.functions import VectorFunction as vec
from graddog.compgraph import CompGraph


    
        
def test_RMtoR():
    def f(v):
        return v[0] + exp(v[1]) + 6*v[2]**2
    f = function_to_Trace(f, [1,2,4])
    assert f.der['x1'] == 1.0
    assert f.der['x2'] == pytest.approx(np.exp(2))
    assert f.der['x3'] == 48.0
    assert f.val == pytest.approx(104.38905609893065)

def test_RMtoRN():
    CompGraph.reset()
    def f(v):
    #     return [v[0] + 3*v[2]**2, v[1] - v[0], v[2] + sin(v[1]), exp(v[0])+sin(v[2])]
        return [v[0]+v[1], v[1]-v[2], cos(v[2]), exp(v[3])*sin(v[2])]
    f = function_to_Trace(f, [1,2,3,4])
    assert f.der.shape == (4,4)
    assert f.der[0][0] == 1.0
    assert f.der[0][1] == 1.0
    assert f.der[1][1] == 1.0
    assert f.der[1][2] == -1.0
    assert f.der[2][2] == pytest.approx(-0.14112001)
    assert f.der[3][2] == pytest.approx(-54.05175886)
    assert f.der[3][3] == pytest.approx(7.70489137)
    
def test_partialderivatives():
    CompGraph.reset()
    x, y = get_xy([3,4])
    
    f = x*y + exp(x/y)
    assert CompGraph.partial_deriv('v3', 'v2') == 3
    assert CompGraph.partial_deriv('v3', 'v1') == 4
    assert f.der['x'] == pytest.approx(4.52925)
    assert f.der['y'] == pytest.approx(2.6030624968851)
    assert f.val == pytest.approx(12 + np.exp(3/4))
