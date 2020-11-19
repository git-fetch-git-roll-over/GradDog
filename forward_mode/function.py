import numpy as np
from forward_mode.variable import Variable

class Function:
    '''
    This is a class for creating different elementary functions.
    '''
    def sin(var:Variable):
        '''
        This allows to create sin().

        Parameters:
            var (Variable instance)

        Return Variable that constitues sin() elementary function
        '''
        return Variable(f'sin{var.name}', np.sin(var.val), np.cos(var.val)*var.der)

    def cos(var:Variable):
        '''
        This allows to create cos().

        Parameters:
            var (Variable instance)

        Return Variable that constitues cos() elementary function
        '''
        return Variable(f'cos{var.name}', np.cos(var.val), -np.sin(var.val)*var.der)

    def tan(var:Variable):
        '''
        This allows to create tan().

        Parameters:
            var (Variable instance)

        Return Variable that constitues tan() elementary function
        '''
        return Variable(f'tan{var.name}', np.tan(var.val), var.der/(np.cos(var.val)**2))

    def exp(var:Variable, base=np.e):
        '''
        This allows to create exp().

        Parameters:
            var (Variable instance)
            base (int, or float)

        Return Variable that constitues exp() elementary function with input base (default=e)
        '''
        if base==np.e:
            return Variable(f'e^{var.name}', np.exp(var.val), np.exp(var.val)*var.der)
        else: 
            return Variable(f'{str(base)}^{var.name}', np.power(
                base, var.val), np.power(base, var.val)*np.log(base)*var.der)

    def log(var:Variable, base=np.e):
        '''
        This allows to create log().

        Parameters:
            var (Variable instance)
            base (int, or float)

        Return Variable that constitues log() elementary function with input base (default=e)
        '''
        if base==np.e:
            return Variable(f'log({var.name})', np.log(var.val), var.der/var.val)
        else: 
            return Variable(f'log{str(base)}({var.name})', np.log(var.val)/np.log(base), var.der/(var.val*np.log(base)))


    
