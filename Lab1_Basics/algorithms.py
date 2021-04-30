#!/usr/bin/env python
# coding: utf-8

import numpy as np
import timeit

def gradient_algorithm(f , f_grad , x0 , step , PREC , ITE_MAX ):
    x = np.copy(x0)
    stop = PREC*np.linalg.norm(f_grad(x0) )

    x_tab = np.copy(x)
    print("------------------------------------\n Constant Stepsize gradient\n------------------------------------\nSTART    -- stepsize = {:0}".format(step))
    t_s =  timeit.default_timer()
    for k in range(ITE_MAX):
        g = f_grad(x)
        x = x - step*g  #######  ITERATION

        x_tab = np.vstack((x_tab,x))

        if np.linalg.norm(g) < stop:
            break
    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations / {:.6f}s -- final value: {:f} at point ({:.2f},{:.2f})\n\n".format(k,t_e-t_s,f(x),x[0],x[1]))
    return x,x_tab



def newton_algorithm(f , f_grad_hessian , x0 , PREC , ITE_MAX ):
    x = np.copy(x0)
    g0,H0 = f_grad_hessian(x0)
    stop = PREC*np.linalg.norm(g0 )

    x_tab = np.copy(x)
    print("------------------------------------\nNewton's algorithm\n------------------------------------\nSTART")
    t_s =  timeit.default_timer()
    for k in range(ITE_MAX):

        g,H = f_grad_hessian(x)
        x = x - np.linalg.solve(H,g)  #######  ITERATION

        x_tab = np.vstack((x_tab,x))

        if np.linalg.norm(g) < stop:
            break
    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations / {:.6f}s -- final value: {:f} at point ({:.2f},{:.2f})\n\n".format(k,t_e-t_s,f(x),x[0],x[1]))
    return x,x_tab


def gradient_adaptive_algorithm(f , f_grad , x0 , step , PREC , ITE_MAX ):
    x = np.copy(x0)
    stop = PREC*np.linalg.norm(f_grad(x0) )

    x_tab = np.copy(x)
    print("------------------------------------\nAdaptative Stepsize gradient\n------------------------------------\nSTART    -- stepsize = {:0}".format(step))
    t_s =  timeit.default_timer()
    for k in range(ITE_MAX):

        g = f_grad(x)
        x_prev = np.copy(x)

        x = x - step*g  #######  ITERATION

	### COMPLETE

        x_tab = np.vstack((x_tab,x))

        if np.linalg.norm(g) < stop:
            break
    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations / {:.6f}s -- final value: {:f} at point ({:.2f},{:.2f})\n\n".format(k,t_e-t_s,f(x),x[0],x[1]))
    return x,x_tab
