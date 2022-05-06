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
print("lambda max is :", largest)
alpha = [random.uniform(0, 2/largest) for i in range(5)]
# alpha = [(2/largest)*0.7 for i in range(5)]
eps = 1e-6

for i in range(5):
    x = [(0.1,0.2)]
    k = 0
    # f = [fun(x[0][0], x[0][1])]
    while grad_fun(x[-1][0], x[-1][1])[0]**2 + grad_fun(x[-1][0], x[-1][1])[1]**2 > eps**2:
        alpha_temp = alpha[i]
        x_new = (x[-1][0] - alpha_temp*(grad_fun(x[-1][0], x[-1][1])[0]), x[-1][1] - alpha_temp*(grad_fun(x[-1][0], x[-1][1])[1]))
        x.append(x_new)
        k+=1
        # f.append(fun(x_new[0], x_new[1]))
    print(x[-1])
    new_x = np.linspace(min([i[0] for i in x]), max([i[0] for i in x]), 100)
    new_y = np.linspace(min([i[1] for i in x]), max([i[1] for i in x]), 100)
    [p, q] = np.meshgrid(new_x, new_y)
    fig, ax = plt.subplots(1, 1)
    r = fun(p,q)
    ax.contourf(p,q,r)
    ax.set_title('Contour Plot at alpha = '+str(alpha[i]))
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    lar = max(max([i[0] for i in x])-min([i[0] for i in x]), max([i[1] for i in x])-min([i[1] for i in x]))
    for j in range(1, len(x)):
        plt.arrow(x[j-1][0], x[j-1][1], x[j][0]-x[j-1][0],x[j][1]-x[j-1][1], head_width= lar/75,length_includes_head=True)
    plt.show()
    print("Took ", k, " iterations to converge when alpha is ", alpha[i])

alpha2 = random.uniform(2/largest, 1)
for i in range(1):
    x = [(0.1,0.2)]
    k = 0
    f = [fun(x[0][0], x[0][1])]
    while grad_fun(x[-1][0], x[-1][1])[0]**2 + grad_fun(x[-1][0], x[-1][1])[1]**2 > eps**2:
        if(x[k][0] > 1e30 or x[k][1] > 1e30):
            break
        alpha_temp = alpha2
        x_new = (x[-1][0] - alpha_temp*(grad_fun(x[-1][0], x[-1][1])[0]), x[-1][1] - alpha_temp*(grad_fun(x[-1][0], x[-1][1])[1]))
        x.append(x_new)
        k+=1
        f.append(fun(x_new[0], x_new[1]))
    print(x[-1])
    new_x = np.linspace(min([i[0] for i in x]), max([i[0] for i in x]), 100)
    new_y = np.linspace(min([i[1] for i in x]), max([i[1] for i in x]), 100)
    [p, q] = np.meshgrid(new_x, new_y)
    fig, ax = plt.subplots(1, 1)
    r = fun(p,q)
    ax.contourf(p,q,r)
    ax.set_title('Contour Plot at alpha = '+str(alpha2))
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    # plt.arrow(p,q, grad_fun(p,q)[0],grad_fun(p,q)[0])
    lar = max(max([i[0] for i in x])-min([i[0] for i in x]), max([i[1] for i in x])-min([i[1] for i in x]))
    for j in range(1, len(x)):
        plt.arrow(x[j-1][0], x[j-1][1], x[j][0]-x[j-1][0],x[j][1]-x[j-1][1], head_width= lar/75,length_includes_head=True)
    plt.show()
    print("Took ", k, " iterations to converge when alpha is ", alpha2)