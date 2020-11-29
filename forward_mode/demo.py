# currently, variables are uniquely represented by their string
import numpy as np
from trace import Variable
from functions import sin, cos, tan, exp, log
from multifunc import MultiFunc





# Partial Derivatives example

x = Variable('x', 3)
y = Variable('y', 4)
z = Variable('z', 2)

f = 2*y**2 + x ** z - 3*x/y 

# manually compute the partial derivatives to test

f_x = -3/y + z*x**(z-1)

f_y = 3*x/(y**2) + 4*y

f_z = (x**z)*log(x)
print(f)

#print(f.der)

assert f._der['x'] == f_x.val
assert f._der['y'] == f_y.val
assert f._der['z'] == f_z.val




# Jacobian example

x = Variable('x', np.pi/2)
y = Variable('y', np.pi/3)
z = Variable('z', np.pi/4)

f = MultiFunc([
	exp(-(sin(x) - cos(y))**2), 
	- log(x) ** 2 + tan(z)
])
print(f.jacobian)