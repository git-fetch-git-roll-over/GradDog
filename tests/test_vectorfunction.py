import pytest
import numpy as np
from graddog.Variable import Variable
from graddog.functions import sin, cos, tan, exp, log
from graddog.multifunc import VectorFunction as vec

# Partial Derivatives example
def test_threevar():
    x = Variable('x', 3)
    y = Variable('y', 4)
    z = Variable('z', 2)

    f = 2*y**2 + x ** z - 3*x/y 
    # manually compute the partial derivatives to test
    f_x = -3/y + z*x**(z-1)
    f_y = 3*x/(y**2) + 4*y
    f_z = (x**z)*log(x)
    assert f._der['x'] == f_x.val
    assert f._der['y'] == f_y.val
    assert f._der['z'] == f_z.val

    # Jacobian example

    x = Variable('x', np.pi/2)
    y = Variable('y', np.pi/3)
    z = Variable('z', np.pi/4)

    f = vec([exp(-(sin(x) - cos(y))**2), - log(x) ** 2 + tan(z)])
    manual_jacobian = [[-4.76877943e-17, -6.74461263e-1,  0.0], [-5.74972958e-01,  0.0,  2.0]]
    for i in range(f.jacobian.shape[0]):
        for j in range(f.jacobian.shape[1]):
            assert f.jacobian[i][j] == pytest.approx(manual_jacobian[i][j])

    
def test_fivevar():
    a = Variable('a', 3)
    b = Variable('b', 5)
    c = Variable('c', 7)
    d = Variable('d', 11)
    e = Variable('e', -2)
    
    f = 2*log(a) + sin(b) + 2**c + 4*e
    g = 2*a + 3*b - cos(d)
    
    f_a = 2/a
    f_b = cos(b)
    f_c = (1/np.log(2)*c)
    f_d = 0
    f_e = 4
    
    g_a = 2
    g_b = 3
    g_c = 0
    g_d = sin(d)
    g_e = 0
    
    assert f_a == f._der['a']
    assert f_b == f._der['b']
    assert f_c == f._der['c']
    assert f_d == f._der['d']
    assert f_e == f._der['e']
    
    assert g_a == g._der['a']
    assert g_b == g._der['b']
    assert g_c == g._der['c']
    assert g_d == g._der['d']
    assert g_e == g._der['e']
    
    h = vec([f, g])
    manual_h_jacobian = [[ 0.66666667,  0.28366219, 88.72283911,  0.        ,  4.        ],[ 2.        ,  3.        ,  0.        , -0.99999021,  0.        ]]
    assert h.jacobian == manual_h_jacobian