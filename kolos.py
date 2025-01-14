import main
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import time
import math
from numpy.polynomial import polynomial as P

poly_kol = [1, -23, 179, -517, 360]
new_roots = P.polyroots(poly_kol)
print(new_roots)

a = 14
b = 16
def fun(x):
    return np.cos(np.exp(x/5))

def dfun(x):
    return -(np.exp(x/5) * np.sin(np.exp(x/5))) / 5

def ddfun(x):
    return -(np.exp(x/5) * (np.sin(np.exp(x/5)) + np.exp(x/5) * np.cos(np.exp(x/5)))) / 25

epsilon = 0.085
iteration = 9

newton_kol = main.newton(fun, dfun, ddfun, a, b, epsilon, iteration)

print(newton_kol)


a = 24
b = 25
def fun(x):
    return np.sin(x - 1) + np.exp(-(x - 3)**2)

epsilon = 0.018
iteration = 11

bisection_kol = main.bisection(a, b, fun, epsilon, iteration)

print(bisection_kol)