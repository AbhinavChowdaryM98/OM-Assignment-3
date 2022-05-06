import numpy as np
from numpy.linalg import eigh
import random
import matplotlib.pyplot as plt

def fun(a,b):
    return 5*(a**2 + b**2) - a*b - 11*a + 11*b + 11

def grad_fun(a,b):
    return (10*a- b - 11, 10*b - a + 11)

#Equation is 5x1**2 + 5x2**2 -x1x2 - 11x1 + 11x2 + 11
hess = [[10,-1],[-1,10]]
eval, evec = eigh(np.array(hess))
largest = max(eval)
alpha = [random.uniform(0, 2/largest) for i in range(5)]
eps = 1e-6

for i in range(5):
    x = [(1,0)]
    k = 0
    f = [fun(x[0][0], x[0][1])]
    while grad_fun(x[-1][0], x[-1][1])[0]**2 + grad_fun(x[-1][0], x[-1][1])[1]**2 > eps**2:
        alpha_temp = alpha[i]
        x_new = (x[-1][0] - alpha_temp*(grad_fun(x[-1][0], x[-1][1])[0]), x[-1][1] - alpha_temp*(grad_fun(x[-1][0], x[-1][1])[1]))
        x.append(x_new)
        k+=1
        f.append(fun(x_new[0], x_new[1]))
    [p, q] = np.meshgrid(x[:][0], x[:][1])
    fig, ax = plt.subplots(1, 1)
    r = fun(p,q)
    ax.contourf(p,q,r)
    ax.set_title('Contour Plot at alpha = '+str(alpha[i]))
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.show()
    print("Took ", k, " iterations to converge when alpha is ", alpha[i])

alpha2 = random.uniform(2/largest, 1)
for i in range(1):
    x = [(1,0)]
    k = 0
    f = [fun(x[0][0], x[0][1])]
    while grad_fun(x[-1][0], x[-1][1])[0]**2 + grad_fun(x[-1][0], x[-1][1])[1]**2 > eps**2:
        alpha_temp = alpha2
        x_new = (x[-1][0] - alpha_temp*(grad_fun(x[-1][0], x[-1][1])[0]), x[-1][1] - alpha_temp*(grad_fun(x[-1][0], x[-1][1])[1]))
        x.append(x_new)
        k+=1
        f.append(fun(x_new[0], x_new[1]))
    [p, q] = np.meshgrid(x[:][0], x[:][1])
    fig, ax = plt.subplots(1, 1)
    r = fun(p,q)
    ax.contourf(p,q,r)
    ax.set_title('Contour Plot at alpha = '+str(alpha2))
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.show()
    print("Took ", k, " iterations to converge when alpha is ", alpha2)