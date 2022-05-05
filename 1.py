import math
import numpy as np



#Backtracking-Armijo Line Search

def fun(c,b):
    return math.exp(c + 3*b - 0.1) + math.exp(c - b*3 - 0.1) + math.exp(-1*c - 0.1)

def fun_x1(c,b):
    return math.exp(c + 3*b - 0.1) + math.exp(c - b*3 - 0.1) - math.exp(-1*c - 0.1)

def fun_x2(c,b):
    return 3*math.exp(c + 3*b - 0.1) - 3*math.exp(c - b*3 - 0.1)
eps = 0.000001

k = 0
alpha = [1]
x = [(0,1)]
p = []
beta = 0.1
tow = 0.7
while fun_x1(x[k][0],x[k][1])**2 + fun_x2(x[k][0],x[k][1])**2 > eps**2:
    a = 1
    #while fun(x[k][0] - a[0]*fun_x1(x[k][0],x[k][1]), x[k][1] - a[1]*fun_x2(x[k][0],x[k][1])) < fun(x[k][0],x[k][1]) - (a[0]*beta*fun_x1(x[k][0],x[k][1]) + a[1]*beta*fun_x2(x[k][0],x[k][1])):
    while fun(x[k][0] - a*fun_x1(x[k][0],x[k][1]), x[k][1] - a*fun_x2(x[k][0],x[k][1])) > fun(x[k][0],x[k][1]) - a*beta*(fun_x1(x[k][0],x[k][1])**2 + fun_x2(x[k][0],x[k][1])**2):
        a *= tow
    alpha.append(a)
    x.append((x[k][0] - a*fun_x1(x[k][0],x[k][1]), x[k][1] - a*fun_x2(x[k][0],x[k][1])))
    k+=1
    print(k , '\n', x[k])

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection ='3d')
# a = np.linspace(0,k,k)
# b = x[a][0]
# c = x[a][1]
'''p = x[:][0]
q = x[:][1]
print(p)
r = fun(p,q)
ax.plot3D(p, q, r, 'green')
#ax.set_title('3D line plot geeks for geeks')
plt.show()
'''
print("Took ", k, " iterations to converge.")