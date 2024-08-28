import numpy as np
import pandas as pd

import scipy.special as sc
from scipy.integrate import quad
from scipy.special import erfc, expi, poch
import mpmath as mp
from mpmath import hyper, pcfd, log

import os
import warnings
import pickle
from collections.abc import Iterable

def load_data(file_dir, file_name):
    return_data = pd.read_excel(os.path.join(file_dir, file_name), sheet_name=0)
    cet_data = pd.read_excel(os.path.join(file_dir, file_name), sheet_name=1)

    B = cet_data['CET-1 ratio (phase-in)'].values/100
    J = np.tan(np.pi * 0.5 - np.pi * B) + 1 / np.tan(np.pi * (1 - B[0]))
    return return_data,  cet_data, B, J, cet_data['Names Date']

# [CoCo Evaluation]
def CDensity_derivative(l1, b, a, x, t):
    m1 = l1 * b ** a * np.power(x, a - 1) * np.exp(-l1 * t - b * x) / sc.factorial(a - 1)
    m11 = l1 * t * np.power(b * x, a) / poch(1 + a, a)
    u1_input1 = np.array([])
    u1_input2 = np.linspace(2 + 1 / a, 3, a)
    u1_input3 = l1 * t * np.power(b * x / a, a)
    # first_u = np.array([float(hyper(u1_input1, u1_input2, i, maxterms=10 ** 6)) for i in u1_input3])
    #first_u = float(hyper(u1_input1, u1_input2, u1_input3, maxterms=10 ** 6))
    first_u = hyper(u1_input1, u1_input2, u1_input3, maxterms=10 ** 6)

    m2 = l1 * t - 1
    u2_input1 = np.array([])
    u2_input2 = np.linspace(1 + 1 / a, 2, a)
    u2_input3 = l1 * t * np.power(b * x / a, a)
    # second_u = np.array([float(hyper(u2_input1, u2_input2, i, maxterms=10 ** 6)) for i in u2_input3])
    #second_u = float(hyper(u2_input1, u2_input2, u2_input3, maxterms=10 ** 6))
    second_u = hyper(u2_input1, u2_input2, u2_input3, maxterms=10 ** 6)
    #return m1 * (m11 * first_u - m2 * second_u)
    return float(m1*(m11 * first_u - m2 * second_u))



def I1(l1, a, b, t):
    # integrand = lambda y: (1- b*y/(l1 * a * t)) * CDensity_derivative(l1, b, a, y, t)
    integrand = lambda y: (1 - b * y / (l1 * a * t)) * np.exp(Density_J(l1, a, b, t, np.array([y])))
    lower = 0
    upper = l1 * a * t / b
    return quad(integrand, lower, upper, epsabs=1e-4)[0]


def I2(l1, a, b, t, x):
    integrand = lambda y: CDensity_derivative(l1, b, a, y, t)
    lower = 0
    upper = x
    return quad(integrand, lower, upper, epsabs=1e-4)[0]


def SupPr(l1, a, b, t, x):
    if t == 0:
        return 0
    c1 = 1 - np.exp(-l1 * t)

    # int2_func = lambda y: CDensity_derivative(l1, b, a, y, t)
    #np.exp(Density_J(l1, a, b, 10, np.array([7])))
    int2_func = lambda y: np.exp(Density_J(l1, a, b, t, np.array([y])))

# density_value = []
# for input_ in range(100):
#    density_value.append(int2_func(input_))
# plt.plot(range(100), density_value)
# plt.show()

    # print(int2_func(np.array([5])))
    c2 = -quad(int2_func, 0, x + l1 * a * t / b)[0]

    m3 = lambda s: I1(l1, a, b, t - s)
    input_CDensity = lambda s: x + l1 * a * s / b
    input_I2 = lambda s: x + l1 * a * s / b

    int3_func = lambda s: m3(s) * (
            np.exp(Density_J(l1, a, b, s, np.array([input_CDensity(s)]))) -
            b / a * (np.exp(-l1 * s) - 1 / l1 * I2(l1, a, b, s, input_I2(s)))
    )

    c3 = quad(int3_func, 0, t, epsabs=1e-4)[0]
    return c1 + c2 + c3


def SupPr_Approx(l1, b, t, x, a):
    with open('./spline_0814/SupPr.pkl', 'rb') as inp:
        spline = pickle.load(inp)
    if isinstance(t, Iterable):
        return np.array([spline(l1, a, b, ti, x) for ti in t])
    return spline(l1, a, b, t, x)


def CDS_spread(k1, xi1, k2, xi2, l2, l32, muV, SigmaV, t, l31r, t0=0):
    const = E(k1, xi1, k2, xi2, l2, l32, muV, SigmaV, t, 1, l31r, t0)

    return -np.log(const) / (t - t0)

def E(k1, xi1, k2, xi2, l2, l32, muV, SigmaV, t, u, l31r, t0=0):

    return E1(k1, xi1, t, u, t0, l31r) * E2(k2, xi2, l2, l32, muV, SigmaV, t, u, t0)

def E1(k1, xi1, t, u, t0=0, l31r = 0):
    #Todo: fix t==0 taking an array input
    # if t == 0:
    #     return 1
    if t0 == 0:
        l31r = 0
    m11 = k1 ** 2 * (t - t0) / (2 * xi1 ** 2)
    m12_input = np.sqrt(2 * xi1 ** 2 * (t - t0) ** 2 * u)
    m12 = np.tanh(m12_input) / m12_input - 1
    log_c1 = m11 * m12

    m2_input = np.sqrt(2 * xi1 ** 2 * (t - t0) ** 2 * u)
    m2 = k1 * 1/np.cosh(m2_input)/xi1**2
    c2 = - l31r**2 * np.sqrt(u) * np.tanh(m2_input)/ np.sqrt(2 * xi1 **2)
    add2 = m2 * l31r + c2


    m3_input = np.sqrt(2 * xi1 ** 2 * (t - t0) ** 2 * u)
    m3 = np.sqrt(1 / np.cosh(m3_input))
    return (np.exp(log_c1 + add2)) * m3


def a_func(muV, SigmaV, case):
    if case == 1:
        numerator = muV + SigmaV
        denominator = np.sqrt(2 * SigmaV ** 2)
        return 1 - 0.5 * erfc(numerator / denominator)
    elif case == 2:
        numerator1 = muV + SigmaV
        denominator1 = np.sqrt(2 * SigmaV ** 2)
        numerator2 = muV + 2 * SigmaV
        denominator2 = np.sqrt(2 * SigmaV ** 2)
        return 0.5 * (erfc(numerator1 / denominator1) - erfc(numerator2 / denominator2))
    elif case == 3:
        numerator1 = muV + 2 * SigmaV
        denominator1 = np.sqrt(2 * SigmaV ** 2)
        numerator2 = muV + 3 * SigmaV
        denominator2 = np.sqrt(2 * SigmaV ** 2)
        return 0.5 * (erfc(numerator1 / denominator1) - erfc(numerator2 / denominator2))
    elif case == 4:
        numerator = muV + 3 * SigmaV
        denominator = np.sqrt(2 * SigmaV ** 2)
        return 0.5 * erfc(numerator / denominator)


def E2(k2, xi2, l2, l32, muV, SigmaV, t, u, t0=0): #ToDo: k2, xi2, l32 needs optimization
    # Todo: fix t==0 taking an array input
    # if t == 0:
    #     return 1
    m11 = -u * l32
    m12 = 1 - np.exp(-k2 * (t - t0))
    c1 = - m11 * m12 / k2

    c2 = 0
    for i in range(1, 5):
        Ei_input1 = i * u * xi2 * np.exp(-k2 * t0) / k2
        Ei_input2 = i * u * xi2 * np.exp(-k2 * t) / k2
        c2 += l2 / k2 * a_func(muV, SigmaV, i) * np.exp(-i * xi2 * u / k2) * (expi(Ei_input1) - expi(Ei_input2))

    c3 = -l2 * (t - t0)
    return np.exp(c1 + c2 + c3)

def equityconvert_coco(r, K, T, t0, l1, a, b, c, e, p, q, Jbar, M, w, w_bar,
                       k1, xi1, k2, xi2, l2, l32, muV, SigmaV, Sigma, ignore_gov = False):
    m11 = K * np.exp(-r * (T-t0))
    #m12 = 1 - SupPr(l1, a, b, T, Jbar)
    m12 = 1 - SupPr_Approx(l1, b, T - t0, Jbar, a)
    c1 = m11 * m12

    c2 = 0
    for i in range(1, M + 1):
        #c2 += c * np.exp(-r * k * (T / M)) * (1 - SupPr(l1, a, b, k * T / M, Jbar))
        c2 += c * np.exp(-r * (i * T/M - t0) * (1 - SupPr_Approx(l1, b, i * T/M -t0, Jbar, a))) # ToDo: check 4.2.2 ti =t0?



    l1_tilde = l1 * (psi1(p, a, b, e) + 1)

    if isinstance(t0, Iterable) == False:
        t0 = [t0]


    M5 = int(np.floor(52 * T))
    k_index = np.floor(t0 * M5 /T)

    price_list = []
    for i in range(len(t0)):
        t0i = t0[i]
        ri = r[i]
        c3 = 0
        for j in range(int(k_index[i]), M5 + 1):
            #m32 = np.exp(- Q(p, q, ri, Sigma, l1, l2, a, b, e, muV, SigmaV) * Ti/M5 *j)
            m32 = np.exp(- Q(p, q, ri, Sigma, l1, l2, a, b, e, muV, SigmaV) * (T/M5 * j -t0i) )
            #ToDo: confirm ignore_gov can be done through value assiginment
            if ignore_gov:
                w_bar = 1
                m31 = w_bar * (1 - w) * K #ToDo: check 4.2.2 (St0/S0)**p
                m33 = 1
                m34 = 1


            else:
                k_tilde = k1 + p * Sigma * xi1
                l2_tilde = l2 * (psi2(muV, SigmaV, p) + 1)
                muV_tilde = muV + p * SigmaV ** 2

                m31 = w_bar * (1 - w) * K
                # m33 = E1(k_tilde, xi1,  j * Ti/M5, 1)
                # m34 = E2(k2, xi2, l2_tilde, l32, muV_tilde, SigmaV, j * Ti/M5, 1)

                m33 = E1(k_tilde, xi1,  T / M5 * j - t0i, 1)
                m34 = E2(k2, xi2, l2_tilde, l32, muV_tilde, SigmaV, T / M5 * j - t0i, 1)


            # m35 = SupPr_Approx(l1_tilde, b + e * p, (j + 1) * Ti/M5, Jbar, a)
            # m36 = SupPr_Approx(l1_tilde, b + e * p, j * Ti/M5, Jbar, a)
            # [change Ti to T]
            m35 = SupPr_Approx(l1_tilde, b + e * p, (j + 1) * T/M5 - t0i, Jbar, a)
            m36 = SupPr_Approx(l1_tilde, b + e * p, j * T/M5 - t0i, Jbar, a)

            c3 += m31 * (m32 * m33 * m34 * (m35 - m36))
        price_list.append(c1[i] + c2[i] + c3)

    if len(t0) == 1:
        return price_list[0]
    else:
        return price_list

def writedown_coco(r, K, T, l1, a, b, c, Jbar, M, w, w_bar, k1, xi1, k2, xi2, l2, l32, muV, SigmaV):
    m11 = K * np.exp(-r * T)
    #m12 = 1 - SupPr(l1, a, b, T, Jbar)
    m12 = 1 - SupPr_Approx(l1, a, b, T, Jbar)
    c1 = m11 * m12

    c2 = 0
    for k in range(1, M + 1):
        #c2 += c * np.exp(-r * k * (T / M)) * (1 - SupPr(l1, a, b, k * T / M, Jbar))
        c2 += c * np.exp(-r * k * (T / M)) * (1 - SupPr_Approx(l1, a, b, k * T / M, Jbar))

    c3 = 0
    NN = 12 * T
    m31 = w_bar * (1 - w) * K

    for j in range(0, NN):
        m32 = np.exp(-r * T / NN * j)
        m33 = E1(k1, xi1, j * T / NN, 1) * E2(k2, xi2, l2, l32, muV, SigmaV, j * T / NN, 1)
        #m34 = SupPr(l1, a, b, (j + 1) * (T / NN), Jbar) - SupPr(l1, a, b, j * (T / NN), Jbar)
        m34 = SupPr_Approx(l1, a, b, (j + 1) * (T / NN), Jbar) - SupPr_Approx(l1, a, b, j * (T / NN), Jbar)
        c3 += m31 * m32 * m33 * m34
    return c1 + c2 + c3


def psi1(u, a, b, e):
    return (1 + u * e / b) ** (-a) - 1


def psi2(muV, sigmaV, u):
    return np.exp(muV * u + sigmaV ** 2 * u ** 2 / 2) - 1


def Q(p, q, r, sigma, l1, l2, a, b, e, muV, SigmaV): # q is dividend yield
    c1 = p * q
    c2 = (1 - p) * r
    c3 = 0.5 * p * (1 - p) * sigma ** 2
    c4 = l1 * (p * psi1(1, a, b, e) - psi1(p, a, b, e))
    c5 = l2 * (p * psi2(muV, SigmaV, 1) - psi2(muV, SigmaV, p))
    return c1 + c2 + c3 + c4 + c5


def Density_J(l, a, b ,t ,x):
    log_multiplier = -l * t
    #first_m = l * b**a * t * x**(a-1) * np.exp(-b * x)/ sc.factorial(a-1)
    log_firstm = np.log(l) + a * np.log(b) +  np.log(t) + (a-1) * np.log(x) - b * x - np.log(sc.factorial(a-1))

    u_input1 = np.array([])
    u_input2 = np.linspace(1 + 1/a, 2, a)
    u_input3 = l * t * (b * x/a)**a

    log_firstu = np.array([float(log(hyper(u_input1, u_input2, i,  maxterms=10**6))) for i in u_input3])
    #return np.log(multiplier) + np.log(first_m) + np.log(first_u)
    return log_multiplier + log_firstm + log_firstu

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




def log_psi(x, v, up, a = None, b = None, d =1 / 252, e= None):
    log_m1 = ((a-1)/2)  * np.log(d) - 1/2 * np.log(2 * np.pi * up ** 2)
    log_m2 = a * np.log(b * up / e)

    log1 = -(x - v*d)**2 / (2 * up ** 2 * d)
    log2 = (e * (x - v * d) + b * up ** 2 * d) ** 2 / ((2 * e * up)**2 *d)

    d_input = (e * (x - v * d ) + b * up ** 2 * d) / (e * up * np.sqrt(d))
    final_add = [float(log_m1 + log_m2 + log1[i] + log2[i] + mp.log(pcfd(-a, d_input[i]))) for i in range(len(d_input))]
    return np.array(final_add)

def Density_stock(l1, a, b, mu, sigma, l2, muV, sigmaV, e, x, d= 1/252):
    v = mu + muV/d
    up = np.sqrt(sigma**2 + sigmaV**2/d)


    psi1 = np.exp(log_psi(x, v, up, a = a, b = b, d = d, e = e))
    psi2 = np.exp(log_psi(x, mu, sigma, a = a, b = b, d = d, e = e))

    first = l1 * l2 * d**2 * psi1

    m2 = l2 * d * (1- l1 * d) /np.sqrt( 2 * np.pi *up**2 * d)
    log2 = -(x-v*d)**2 / (2 * up**2 * d)

    third = l1 * d * (1-l2*d) * psi2

    m4 = (1 - l1*d) * (1 - l2*d) / np.sqrt(2 * np.pi * sigma**2 *d)
    log4 = -(x - mu * d)**2 / (2 * sigma**2 * d)

    return first + m2 * np.exp(log2) + third + m4 * np.exp(log4)

def optimize_stock(param, l1 = None, a= None, b = None, d = 1/252, x = None):
    mu, sigma, l2, muV, SigmaV, e = param
    log_density = np.log(Density_stock(l1, a, b, mu, sigma, l2, muV, SigmaV, e, x, d))
    inf_flag = np.isinf(log_density)
    if sum(inf_flag) != 0:
        warnings.warn("Warning: log_density contains invalid value given the current parameters, masking it with the minimal of valid values")
        log_density[inf_flag] = min(log_density[~inf_flag])
    return -np.sum(log_density)

def optimize_cds(param, l2=None, muV=None, sigmaV=None, T=None, t0=None, cds_price=None):
    k1, xi1, k2, xi2, l32 = param
    model_price = CDS_spread(k1, xi1, k2, xi2, l2, l32, muV, sigmaV, T, 0, t0)
    loss = np.abs(cds_price - model_price) / cds_price
    return np.mean(loss)

from sklearn.neighbors import KernelDensity
from scipy.stats import iqr
def KDE_estimate(fit_data, eval_data):
    silverman_bw = 0.9 * min(np.std(fit_data), iqr(fit_data) / 1.34) * fit_data.shape[0] ** (-1 / 5)
    KDE_model = KernelDensity(bandwidth=silverman_bw).fit(fit_data.reshape(-1, 1))
    log_density = KDE_model.score_samples(eval_data.reshape(-1, 1))
    return np.exp(log_density)
