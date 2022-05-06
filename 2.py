import math
import numpy as np
# import random
import matplotlib.pyplot as plt

def fun(c,b):
    return np.exp(c + 3*b - 0.1) + np.exp(c - b*3 - 0.1) + np.exp(-1*c - 0.1)

def fun_x1(c,b):
    return np.exp(c + 3*b - 0.1) + np.exp(c - b*3 - 0.1) - np.exp(-1*c - 0.1)

def fun_x2(c,b):
    return 3*np.exp(c + 3*b - 0.1) - 3*np.exp(c - b*3 - 0.1)
eps = 0.000001

def hess_inv(a):
    a11 = np.exp(a[0] + 3*a[1] - 0.1) + np.exp(a[0] - a[1]*3 - 0.1) + np.exp(-1*a[0] - 0.1)
    a12 = 3*np.exp(a[0] + 3*a[1] - 0.1) - 3*np.exp(a[0] - a[1]*3 - 0.1)
    a22 = 9*np.exp(a[0] + 3*a[1] - 0.1) + 9*np.exp(a[0] - a[1]*3 - 0.1)
    det = a11 * a22 - (a12**2)
    return [[a22/det, -1 * (a12/det)], [-1 * (a12/det), a11/det]]

def hess(a):
    a11 = np.exp(a[0] + 3*a[1] - 0.1) + np.exp(a[0] - a[1]*3 - 0.1) + np.exp(-1*a[0] - 0.1)
    a12 = 3*np.exp(a[0] + 3*a[1] - 0.1) - 3*np.exp(a[0] - a[1]*3 - 0.1)
    a22 = 9*np.exp(a[0] + 3*a[1] - 0.1) + 9*np.exp(a[0] - a[1]*3 - 0.1)
    return [[a11, a12], [a12, a22]]

k = 0
alpha = [1]
x = [(0.1,0.3)]
# p = []
beta = 0.01
tow = 0.9
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

fig, ax = plt.subplots(1, 1)
new_x = np.linspace(min([i[0] for i in x]), max([i[0] for i in x]), 100)
new_y = np.linspace(min([i[1] for i in x]), max([i[1] for i in x]), 100)
[p, q] = np.meshgrid(new_x, new_y)
r = fun(p,q)
ax.contourf(p,q,r)
ax.set_title('Contour Plot')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
lar = max(max([i[0] for i in x])-min([i[0] for i in x]), max([i[1] for i in x])-min([i[1] for i in x]))
for j in range(1, len(x)):
    plt.arrow(x[j-1][0], x[j-1][1], x[j][0]-x[j-1][0],x[j][1]-x[j-1][1], head_width= lar/75,length_includes_head=True)

# hess_k = np.array(hess(x[k]))
plt.show()
for i in range(k):
    hess_k = np.array(hess(x[i]))
    plt.close()
    fig, ax = plt.subplots(1, 1)
    ax.contourf(p,q,r)
    ax.set_title('Contour Plot')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    lar = max(max([i[0] for i in x])-min([i[0] for i in x]), max([i[1] for i in x])-min([i[1] for i in x]))
    for j in range(1, len(x)):
        plt.arrow(x[j-1][0], x[j-1][1], x[j][0]-x[j-1][0],x[j][1]-x[j-1][1], head_width= lar/75,length_includes_head=True)
    for listx, listy in zip(p, q):
        for ai, bi in zip(listx, listy):
            tmp = np.array([ai, bi]) - np.array(x[i])
            if (np.matmul(tmp, np.matmul(hess_k, tmp.T))) <= 1:
                plt.plot(ai, bi,marker="o", markersize=4, markeredgecolor="black", markerfacecolor="yellow")
    # plt.show()
    plt.savefig(str(i+1) + '.png')
    print("Completed iteration"+str(i+1))

print("Took k = ", k, " iterations to converge")