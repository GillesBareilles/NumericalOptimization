#!/usr/bin/env python
# coding: utf-8

# # Regularized Problem
#
# In this lab, we add an $\ell_1$ regularization to promote sparsity of the iterates. The function (below) is non-smooth but it has a smooth part, $f$, and a non-smooth part, $g$, that we will treat with proximal operations.
#
# \begin{align*}
# \min_{x\in\mathbb{R}^d } F(x) := \underbrace{ \frac{1}{m}  \sum_{i=1}^m  \log( 1+\exp(-b_i \langle a_i,x \rangle) ) + \frac{\lambda_2}{2} \|x\|_2^2}_{f(x)} + \underbrace{\lambda_1 \|x\|_1 }_{g(x)}.
# \end{align*}

import numpy as np
import csv
from sklearn import preprocessing

### File reading
file = open('ionosphere.data')

d = 34
n = d+1 # Variable size + intercept

m = 351 # Number of examples

lam2 = 0.001 # regularization best:0.001

A = np.zeros((m,d))
b = np.zeros(m)

reader = csv.reader(file, delimiter=',')
i = 0
for row in reader:
    A[i] = np.array(row[:d])
    if row[d] == 'b':
        b[i] = -1.0
    else:
        b[i] =  1.0
    i+=1

scaler = preprocessing.StandardScaler().fit(A)
A = scaler.transform(A)

# Adding an intercept
A_inter = np.ones((m,n))
A_inter[:,:-1] = A
A = A_inter


# Lipschitz constant of gradient of f
L = 0.25*max(np.linalg.norm(A,2,axis=1))**2 + lam2




## Oracles
### Related to function f
def f(x):
    l = 0.0
    for i in range(A.shape[0]):
        if b[i] > 0 :
            l += np.log( 1 + np.exp(-np.dot( A[i] , x ) ) )
        else:
            l += np.log( 1 + np.exp(np.dot( A[i] , x ) ) )
    return l/m + lam2/2.0*np.dot(x,x)

def f_grad(x):
    g = np.zeros(n)
    for i in range(A.shape[0]):
        if b[i] > 0:
            g += -A[i]/( 1 + np.exp(np.dot( A[i] , x ) ) )
        else:
            g += A[i]/( 1 + np.exp(-np.dot( A[i] , x ) ) )
    return g/m + lam2*x

def f_grad_hessian(x):
    g = np.zeros(n)
    H = np.zeros((n,n))
    for i in range(A.shape[0]):
        if b[i] > 0:
            g += -A[i]/( 1 + np.exp(np.dot( A[i] , x ) ) )
            H +=  (np.exp(np.dot( A[i] , x ))/( 1 + np.exp(np.dot( A[i] , x ) ) )**2)*np.outer(A[i],A[i])
        else:
            g += A[i]/( 1 + np.exp(-np.dot( A[i] , x ) ) )
            H +=  (np.exp(-np.dot( A[i] , x ))/( 1 + np.exp(-np.dot( A[i] , x ) ) )**2)*np.outer(A[i],A[i])
    g =  g/m + lam2*x
    H = H/m + lam2*np.eye(n)
    return g,H


## Prediction Function
def prediction(w,PRINT=False):
    pred = np.zeros(A.shape[0])
    perf = 0
    for i in range(A.shape[0]):
        p = 1.0/( 1 + np.exp(-np.dot( A[i] , w ) ) )
        if p>0.5:
            pred[i] = 1.0
            if b[i]>0:
                correct = "True"
                perf += 1
            else:
                correct = "False"
            if PRINT:
                print("True class: {:d} \t-- Predicted: {} \t(confidence: {:.1f}%)\t{}".format(int(b[i]),1,(p-0.5)*200,correct))
        else:
            pred[i] = -1.0
            if b[i]<0:
                correct = "True"
                perf += 1
            else:
                correct = "False"
            if PRINT:
                print("True class: {:d} \t-- Predicted: {} \t(confidence: {:.1f}%)\t{}".format(int(b[i]),-1,100-(0.5-p)*200,correct))
    return pred,float(perf)/A.shape[0]


