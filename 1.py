import math
import numpy as np
import random
# from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


def fun(c,b):
    return np.exp(c + 3*b - 0.1) + np.exp(c - b*3 - 0.1) + np.exp(-1*c - 0.1)

def fun_x1(c,b):
    return np.exp(c + 3*b - 0.1) + np.exp(c - b*3 - 0.1) - np.exp(-1*c - 0.1)

def fun_x2(c,b):
    return 3*np.exp(c + 3*b - 0.1) - 3*np.exp(c - b*3 - 0.1)
eps = 0.000001


#Armijo-Goldstein Line Search
c1 = random.uniform(0, 1/3)
print("c1 = ", c1)
alpha_init = 1
k = 0
x_a = [[0.1,0.1]]
# f = [fun(0.1,0.1)]
while fun_x1(x_a[k][0],x_a[k][1])**2 + fun_x2(x_a[k][0],x_a[k][1])**2 > eps**2:
    dx_1 = fun_x1(x_a[k][0],x_a[k][1])
    dx_2 = fun_x2(x_a[k][0],x_a[k][1])
    a = alpha_init
    while fun(x_a[k][0],x_a[k][1]) - (1 - c1)*a*(dx_1**2 + dx_2**2) > fun(x_a[k][0]-a*dx_1, x_a[k][1]-a*dx_2) or fun(x_a[k][0]-a*dx_1, x_a[k][1]-a*dx_2) > fun(x_a[k][0],x_a[k][1]) - (c1)*a*(dx_1**2 + dx_2**2):
        a = a/2
    x_a.append([x_a[k][0]-a*dx_1, x_a[k][1]-a*dx_2])
    # f.append(fun(x_a[k][0]-a*dx_1, x_a[k][1]-a*dx_2))
    k+=1

print("Took ", k, " iterations to converge in Armijo-Goldstein Line Search.")

# fig = plt.figure()
# ax = plt.axes(projection ='3d')
# p = [i[0] for i in x_a]
# q = [i[1] for i in x_a]
# r = f
# ax.plot3D(p, q, r, 'green')
# ax.set_title('3D line plot')
# plt.show()

fig, ax = plt.subplots(1, 1)
new_x = np.linspace(min([i[0] for i in x_a]), max([i[0] for i in x_a]), 100)
new_y = np.linspace(min([i[1] for i in x_a]), max([i[1] for i in x_a]), 100)
[p, q] = np.meshgrid(new_x, new_y)
r = fun(p,q)
ax.contourf(p,q,r)
ax.set_title('Contour Plot at c = '+str(c1))
ax.set_xlabel('x1')
ax.set_ylabel('x2')
lar = max(max([i[0] for i in x_a])-min([i[0] for i in x_a]), max([i[1] for i in x_a])-min([i[1] for i in x_a]))
for j in range(1, len(x_a)):
    plt.arrow(x_a[j-1][0], x_a[j-1][1], x_a[j][0]-x_a[j-1][0],x_a[j][1]-x_a[j-1][1], head_width= lar/75,length_includes_head=True)
plt.show()


#Backtracking-Armijo Line Search

k = 0
alpha = [1]
x = [(0.1,0.1)]
p = []
beta = 0.1
tow = 0.7
f2 = [fun(0.1,0.1)]
while fun_x1(x[k][0],x[k][1])**2 + fun_x2(x[k][0],x[k][1])**2 > eps**2:
    a = 1
    #while fun(x[k][0] - a[0]*fun_x1(x[k][0],x[k][1]), x[k][1] - a[1]*fun_x2(x[k][0],x[k][1])) < fun(x[k][0],x[k][1]) - (a[0]*beta*fun_x1(x[k][0],x[k][1]) + a[1]*beta*fun_x2(x[k][0],x[k][1])):
    while fun(x[k][0] - a*fun_x1(x[k][0],x[k][1]), x[k][1] - a*fun_x2(x[k][0],x[k][1])) > fun(x[k][0],x[k][1]) - a*beta*(fun_x1(x[k][0],x[k][1])**2 + fun_x2(x[k][0],x[k][1])**2):
        a *= tow
    alpha.append(a)
    x.append((x[k][0] - a*fun_x1(x[k][0],x[k][1]), x[k][1] - a*fun_x2(x[k][0],x[k][1])))
    f2.append(fun(x[k][0] - a*fun_x1(x[k][0],x[k][1]), x[k][1] - a*fun_x2(x[k][0],x[k][1])))
    k+=1
    #print(k , '\n', x[k])


# fig = plt.figure()
# ax = plt.axes(projection ='3d')
# p = [i[0] for i in x]
# q = [i[1] for i in x]
# r = f2
# ax.plot3D(p, q, r, 'green')
# ax.set_title('3D line plot')
# plt.show()


fig, ax = plt.subplots(1, 1)
new_x = np.linspace(min([i[0] for i in x]), max([i[0] for i in x]), 100)
new_y = np.linspace(min([i[1] for i in x]), max([i[1] for i in x]), 100)
[p, q] = np.meshgrid(new_x, new_y)
r = fun(p,q)
ax.contourf(p,q,r)
ax.set_title('Contour Plot of Backtracking-Armijo Line Search')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
lar = max(max([i[0] for i in x])-min([i[0] for i in x]), max([i[1] for i in x])-min([i[1] for i in x]))
for j in range(1, len(x)):
    plt.arrow(x[j-1][0], x[j-1][1], x[j][0]-x[j-1][0],x[j][1]-x[j-1][1], head_width= lar/75,length_includes_head=True)
plt.show()

print("Took ", k, " iterations to converge in Backtracking Armijo Line Search.")