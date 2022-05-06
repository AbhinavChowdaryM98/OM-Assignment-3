import math
import numpy as np
# import random

def fun(c,b):
    return math.exp(c + 3*b - 0.1) + math.exp(c - b*3 - 0.1) + math.exp(-1*c - 0.1)

def fun_x1(c,b):
    return math.exp(c + 3*b - 0.1) + math.exp(c - b*3 - 0.1) - math.exp(-1*c - 0.1)

def fun_x2(c,b):
    return 3*math.exp(c + 3*b - 0.1) - 3*math.exp(c - b*3 - 0.1)
eps = 0.000001

def hess_inv(a):
    a11 = math.exp(a[0] + 3*a[1] - 0.1) + math.exp(a[0] - a[1]*3 - 0.1) + math.exp(-1*a[0] - 0.1)
    a12 = 3*math.exp(a[0] + 3*a[1] - 0.1) - 3*math.exp(a[0] - a[1]*3 - 0.1)
    a22 = 9*math.exp(a[0] + 3*a[1] - 0.1) + 9*math.exp(a[0] - a[1]*3 - 0.1)
    det = a11 * a22 - (a12**2)
    return [[a22/det, -1 * (a12/det)], [-1 * (a12/det), a11/det]]


k = 0
alpha = [1]
x = [(0,1)]
# p = []
beta = 0.1
tow = 0.7
while fun_x1(x[k][0],x[k][1])**2 + fun_x2(x[k][0],x[k][1])**2 > eps**2:
    a = 1
    print(k, "xk: ", x[k])
    hk_inv = hess_inv(x[k])
    while fun(x[k][0] - a*fun_x1(x[k][0],x[k][1]), x[k][1] - a*fun_x2(x[k][0],x[k][1])) > fun(x[k][0],x[k][1]) - a*beta*np.matmul(np.array([fun_x1(x[k][0],x[k][1]), fun_x2(x[k][0],x[k][1])]), np.matmul(np.array(hk_inv), np.array([fun_x1(x[k][0],x[k][1]), fun_x2(x[k][0],x[k][1])]).T)):
        a *= tow
    alpha.append(a)
    dk = (-1)*np.matmul(np.array(hk_inv), np.array([fun_x1(x[k][0],x[k][1]), fun_x2(x[k][0],x[k][1])]).T)
    x.append((x[k][0] + a*dk[0], x[k][1] + a*dk[1]))
    k+=1

print("Took k = ", k, " iterations to converge")