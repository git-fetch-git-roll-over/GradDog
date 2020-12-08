# :)
import numpy as np
from graddog.trace import Variable
from graddog.compgraph import CompGraph


def trace(f, seed, mode = None, return_hessian = False):
    '''
    Optional parameter mode
    default is 'forward'
    Infers the dimension of input from the seed
    Dimension of output inferred in CompGraph
    Therefore f can be
    f: R --> R using explicit single-variable input
    f: Rm --> R using explicit multi-variable input
    f: R --> Rn using explicit single-variable input and explicit vector output
    f: Rm --> R using explicit multi-variable input
    f: Rm --> R using explicit vector input
    f: Rm --> Rn using explicit vector input and explicit vector output
    f: Rm --> Rn using explicit multi-variable input and explicit vector output
    f: Rm --> Rn using IMPLICIT vector input and IMPLICIT vector output
    '''

    # for now, always reset the CompGraph when tracing a new function
    CompGraph.reset()

    try:# if multidimensional input
        M = len(seed) # get the dimension of the input
        seed = np.array(seed)
    except TypeError: # if sinledimensional input
        M = 1
        seed = np.array([seed])
    new_variable_names = [f'v{m+1}' for m in range(M)]
    new_vars = np.array([Variable(new_variable_names[i], seed[i]) for i in range(M)])

    if M > 1:
        # multi-variable input
        try:
            # as a vector
            f(new_vars)
        except TypeError:
            # as variables
            f(*new_vars)
    else:
        # single-variable input
        f(new_vars[0])

    N = CompGraph.num_outputs()

    if return_hessian:
        if N == 1:
            print('Computing first AND second derivative with reverse mode')
            return CompGraph.hessian()
        else:
            raise ValueError('Can only compute Hessian for f:Rm --> R')

    if mode is None:
    # go with more efficient algorithm if mode parameter is not specified by the user
        if M > N :
            mode = 'reverse'
        else:
            mode = 'forward'
    if mode == 'reverse':
        print('Computing reverse mode')
        return CompGraph.reverse_mode()
    elif mode == 'forward':
        print('Computing forward mode')
        return CompGraph.forward_mode()
    else:
        raise ValueError('Didnt recognize mode, should be forward or reverse')

















