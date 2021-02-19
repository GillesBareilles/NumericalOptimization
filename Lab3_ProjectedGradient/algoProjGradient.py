#!/usr/bin/env python
# coding: utf-8

# # Projected Gradient-based algorithms
#
# In this notebook, we code our Projected gradient-based optimization algorithms.
# We consider here
# * Positivity constraints
# * Interval constraints

### 1. Projected Gradient algorithms (for positivity or interval constraints)


import numpy as np
import timeit

def positivity_gradient_algorithm(f , f_grad , x0 , step , PREC , ITE_MAX ):
    x = np.copy(x0)
    stop = PREC*np.linalg.norm(f_grad(x0) )

    epsilon = PREC*np.ones_like(x0)

    x_tab = np.copy(x)
    print("------------------------------------\n Constant Stepsize gradient\n------------------------------------\nSTART    -- stepsize = {:0}".format(step))
    t_s =  timeit.default_timer()
    for k in range(ITE_MAX):

        ## update iterate
        # TODO: complete by gradient step projected onto the set "x >= 0"
        g = f_grad(x)
        x = x

        ## Log current iterate
        x_tab = np.vstack((x_tab,x))

        ## Check stoping criterion
        # TODO: update stopping criterion
        if np.linalg.norm(g) < stop:
           break

    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations / {:.6f}s -- final value: {:f} at point ({:.2f},{:.2f})\n\n".format(k,t_e-t_s,f(x),x[0],x[1]))
    return x,x_tab


### 1.b. Constant stepsize projected gradient algorithm for interval constraints


import numpy as np
import timeit

def interval_gradient_algorithm(f , f_grad , x0 , infbound , supbound , step , PREC , ITE_MAX ):
    x = np.copy(x0)
    stop = PREC*np.linalg.norm(f_grad(x))

    epsilon = PREC*np.ones_like(x0)

    x_tab = np.copy(x)
    print("------------------------------------\n Constant Stepsize gradient\n------------------------------------\nSTART    -- stepsize = {:0}".format(step))
    t_s =  timeit.default_timer()
    for k in range(ITE_MAX):

        ## update iterate
        # TODO: complete by gradient step projected onto the set "x >= 0"
        g = f_grad(x)
        x = x

        ## Log current iterate
        x_tab = np.vstack((x_tab,x))

        ## Check stoping criterion
        # TODO: update stopping criterion
        if np.linalg.norm(g) < stop:
           break

    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations / {:.6f}s -- final value: {:f} at point ({:.2f},{:.2f})\n\n".format(k,t_e-t_s,f(x),x[0],x[1]))
    return x,x_tab
