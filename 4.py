import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.linalg import eigh

#Part 1
#f(x) = 10x1**2 + 10x1x2 + x2**2 + 4*x1 - 10*x2 + 2 over [-3, 3] x [-3, 3]
print("Part1")
def fun1(a,b):
    return 10*(a**2) + 10*a*b + b**2 + 4*a - 10*b + 2

def grad_fun1(a,b):
    return (20*a + 10*b + 4, 10*a + 2*b - 10)

def hess_fun1():
    return [[20, 10], [10, 2]]

x1_init = (1.8, -4)
a1 = 0.01
pos = False
neg = False
f = fun1(x1_init[0], x1_init[1])

val = []

for i in range(200):
    x_new = (x1_init[0]+a1*(np.cos(i*math.pi/100)), x1_init[1]+a1*(np.sin(i*math.pi/100)))
    f_new = fun1(x_new[0], x_new[1])
    if(f_new < f):
        neg = True
    elif f_new > f:
        pos = True
    val.append(f_new - f)
x = [i*math.pi/100 for i in range(200)]
plt.plot(x, val)
plt.show()
if pos and neg:
    print("Therfore given point (", x1_init[0], ", ", x1_init[1], ") is a saddle point")
elif pos == True:
    print("Therfore given point (", x1_init[0], ", ", x1_init[1], ") is a local minima")

else:
    print("Therfore given point (", x1_init[0], ", ", x1_init[1], ") is a local maxima")
    
eval, evec = eigh((hess_fun1()))

print("Grad of f is: ", grad_fun1(x1_init[0], x1_init[1]))
print("Eigen values of Hess of f are: ", eval)

#Part2
#f(x) = 16*x1**2 + 8x1x2 + 10x2**2 + 12x1 - 6x2 + 2
print("Part2")

def fun2(a,b):
    return 16*(a**2) + 8*a*b + 10*(b**2) + 12*a - 6*b + 2

def grad_fun2(a,b):
    return (32*a + 8*b + 12, 8*a + 20*b - 6)

def hess_fun2():
    return [[32, 8], [8, 20]]

x2_init = (-0.5, 0.5)
a1 = 0.01
pos = False
neg = False
f2 = fun2(x2_init[0], x2_init[1])

val2 = []

for i in range(200):
    x_new = (x2_init[0]+a1*(np.cos(i*math.pi/100)), x2_init[1]+a1*(np.sin(i*math.pi/100)))
    f_new = fun2(x_new[0], x_new[1])
    if(f_new < f2):
        neg = True
    elif f_new > f2:
        pos = True
    val2.append(f_new - f2)
    
plt.plot(x, val2)
plt.show()
if pos and neg:
    print("Therfore given point (", x2_init[0], ", ", x2_init[1], ") is a saddle point")
elif pos == True:
    print("Therfore given point (", x2_init[0], ", ", x2_init[1], ") is a local minima")

else:
    print("Therfore given point (", x2_init[0], ", ", x2_init[1], ") is a local maxima")
    
eval2, evec2 = eigh((hess_fun2()))

print("Grad of f is: ", grad_fun2(x2_init[0], x2_init[1]))
print("Eigen values of Hess of f are: ", eval2)