import numpy as np
from trace import Variable
from functions import sin, cos, tan, exp, log



# currently, variables are uniquely represented by their string
x = Variable('x', 3)
y = Variable('y', 4)
z = Variable('z', 2)


f = 2*y**2 + x ** z - 3*x/y 

# manually compute the derivatives to test

f_x = -3/y + z*x**(z-1)

f_y = 3*x/(y**2) + 4*y

f_z = (x**z)*log(x)
print(f)

assert f._der['x'] == f_x.val
assert f._der['y'] == f_y.val
assert f._der['z'] == f_z.val

print('Jacobian:')
print(f.jacobian)

assert f.jacobian[0] == f_x.val
assert f.jacobian[1] == f_y.val
assert f.jacobian[2] == f_z.val

