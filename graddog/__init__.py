# :)
import numpy as np
from graddog.trace import Variable
from graddog.compgraph import CompGraph


def trace(f, seed, mode = None, return_second_deriv = False, verbose = False):

    '''
    f : a function
    seed: a vector/list of scalars. If f is single-dimensional, seed can be a scalar

    Optional parameter mode
        When mode = None, this function infers the more efficient mode from the number of input and output variables

    Optional parameter return_second_deriv (default is False)
        When return_second_deriv = True, this function returns f' AND f''

    f can be
    f: R --> R using explicit single-variable input
    f: Rm --> R using explicit multi-variable input
    f: R --> Rn using explicit single-variable input and explicit vector output
    f: Rm --> R using explicit vector input
    f: Rm --> Rn using explicit vector input and explicit vector output
    f: Rm --> Rn using explicit multi-variable input and explicit vector output
    f: Rm --> Rn using IMPLICIT vector input and IMPLICIT vector output
    '''

    ######################## make your Variable objects #########################
    # for now, always reset the CompGraph when tracing a new function
    CompGraph.reset()


    # infer the dimensionality of the input
    try:# if multidimensional input
        M = len(seed) # get the dimension of the input
        seed = np.array(seed)
    except TypeError: # if single-dimensional input
        M = 1
        seed = np.array([seed])
    if verbose:
        print(f'Inferred {M}-dimensional input')

    # create new variables
    names = [f'v{i+1}' for i in range(M)]
    new_variables = np.array([Variable(names[i], seed[i]) for i in range(M)])
    #############################################################################




    ################ Trace the function ##############
    if verbose:
        print('Scanning the computational graph...')
    # Apply f to the new variables
    # Infer the way f was meant to be applied
    if M > 1:
        # multi-variable input

        try:
            # as a vector
            output = f(new_variables)
            if verbose:
                print('...inferred the input is a vector...')
        except TypeError:
            # as variables
            output = f(*new_variables)
            if verbose:
                print('...inferred the inputs are variables...')

    else:
        # single-variable input
        output = f(new_variables[0])
        if verbose:
            print('...inferred the input is a variable...')
    if verbose:
        print('...finished')
    ############################################



    ################ Get Outputs #################
    try:
        N = len(output)
    except AttributeError:
        N = 1
        output = [output]
    except TypeError:
        N = 1
        output = [output]
    if verbose:
        print(f'Inferred {N}-dimensional output')
        print(output)
    ##############################################



    ##################### Second Derivative #########################
    if return_second_deriv:
        if mode is not None and mode.lower() != 'reverse':
            raise ValueError('Second derivative is automatically calculated in reverse mode')
        if N > 1:
            raise ValueError('Can only compute second derivative for scalar output f')
        print('Computing reverse mode first AND second derivative...')
        return CompGraph.hessian(output, verbose)   
    ######################################################




    ####### get user-defined mode or infer the more efficient mode ##########
    if mode is None:
        if M > N :
            mode = 'reverse'
        else:
            mode = 'forward'
    else:
        mode = mode.lower()
    ######################################################################



    


    ############## First Derivative ####################
    #if verbose:
    print(f'Computing {mode} mode derivative...')
    if mode == 'forward':
        return CompGraph.forward_mode(output, verbose)
    elif mode == 'reverse':
        return CompGraph.reverse_mode(output, verbose)
    else:
        raise ValueError('Didnt recognize mode, should be forward or reverse')
    ########################################################




