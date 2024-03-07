import numpy as np
from mpmath import hyper
import scipy.special as sc
def Density_J(l, a, b ,t ,x):
    multiplier = np.exp(-l * t)
    first_m = l * b**a * t * x**(a-1) * np.exp(-b * x)/ sc.factorial(a-1)

    u_input1 = np.array([])
    u_input2 = np.linspace(1 + 1/a, 2, a)
    u_input3 = l * t * (b * x/a)**a

    first_u = np.array([float(hyper(u_input1, u_input2, i,  maxterms=10**6)) for i in u_input3])
    return np.log(multiplier) + np.log(first_m) + np.log(first_u)

def optimize_DensityJ(param, a = 1, T= 1/4, H = None):
    l, b = param
    return -np.sum(Density_J(l, a, b, T, H))


def Density_RelaxJ(l, b ,t, x):
    multiplier = np.exp(-l * t)
    first_m = np.exp(-b * x) * np.sqrt(l * b * t/x)

    i_input = 2 * np.sqrt(l * b * t * x)
    first_i = sc.i1(i_input)
    return np.log(multiplier) + np.log(first_m) + np.log(first_i)

def optimize_RelaxJ(params, T= 1/4, H = None):
    l, b = params
    return -np.sum(Density_RelaxJ(l, b, T, H))

