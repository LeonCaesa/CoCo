import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sc
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize
from scipy.stats import iqr
from mpmath import hyper, pcfd
import mpmath as mp
import warnings
import os

from utils import *


def load_data(file_dir):
    return_data = pd.read_excel(os.path.join(file_dir, 'Credit_Suisse_Data_13-23.xlsx'), sheet_name=0)
    cet_data = pd.read_excel(os.path.join(file_dir, 'Credit_Suisse_Data_13-23.xlsx'), sheet_name=1)

    B = cet_data['CET-1 ratio (phase-in)'].values
    J = np.tan(np.pi * 0.5 - np.pi * B) + 1 / np.tan(np.pi * (1 - B[0]))

    return return_data,  cet_data, B, J


def Callback_RelaxJ(Xi):
    global Nfeval, H
    print('{0: 4d}     {1:.4f}     {2: .4f}    {3:.4f} '.format(Nfeval, Xi[0], Xi[1], optimize_RelaxJ(Xi, H=H)))
    Nfeval += 1

def Callback_DensityJ(Xi):
    global Nfeval, H, a
    print('{0: 4d}     {1:.4f}     {2: .4f}    {3:.4f} '.format(Nfeval, Xi[0], Xi[1], optimize_DensityJ(Xi, a=a, H=H)))
    Nfeval += 1

def KDE_estimate(fit_data, eval_data):
    silverman_bw = 0.9 * min(np.std(fit_data), iqr(fit_data) / 1.34) * fit_data.shape[0] ** (-1 / 5)
    KDE_model = KernelDensity(bandwidth=silverman_bw).fit(fit_data.reshape(-1, 1))
    log_density = KDE_model.score_samples(eval_data.reshape(-1, 1))
    return np.exp(log_density)



if __name__ == '__main__':

    return_data, cet_data, B, J = load_data('../data')
    Diff_J = np.diff(J)

    J_grids = np.linspace(-3, 3, 200).reshape(-1, 1)
    H_grids = np.linspace(1.1, 1.25, 200).reshape(-1, 1)

    PJ_grids = KDE_estimate(Diff_J, J_grids)
    H = Diff_J - min(Diff_J) + 0.01
    PH_grids = KDE_estimate(H, H_grids)

    # [Plotting]
    fig = plt.figure(figsize=(6, 12))
    ax1 = fig.add_subplot(311)
    ax1.plot(J_grids, PJ_grids, label = 'Diff_J Density')
    plt.legend()

    ax2 = fig.add_subplot(312)
    ax2.plot(B, label = 'B', color = 'red')
    plt.legend()

    ax3 = fig.add_subplot(313)
    ax3.plot(H, label='H', color = 'blue')
    plt.legend()
    plt.show()

    # [Optimization]
    l, b, T = [1, 1, 1/4]
    Test_RelaxJ = Density_RelaxJ(l, b, T, H)
    Test_DensityJ = Density_J(l, 1, b, T, H)
    bounds = [(0, 100), (0, 150)]
    init, Nfeval = [[0.1, 0.1], 1]
    sorted_list = []
    for a in range(1, 6):
        Nfeval = 1
        print('==========optimization for a = %d===========' % a)
        res = minimize(optimize_DensityJ, init, args=(a, T, H), method='Nelder-Mead', options={'maxiter': 100}, bounds=bounds,
                       callback = Callback_DensityJ, tol=0.001)
        print('lambda: %.4f, beta %.4f' % (res.x[0], res.x[1]))
        sorted_list.append([res.fun, *res.x, a])
        if a ==1:
            l1_a1, b_a1 = res.x
    sorted_list = sorted(sorted_list)
    loss, l1_a3, b_a3, a3 = sorted_list[0]

    # [Stock estimation]
    RET = pd.read_excel(os.path.join('../data', 'Credit_Suisse_Data_13-23.xlsx'), sheet_name=0)[
              'Log-returns (without Dividends)'].values[:2570]
    plt.plot(RET)
    plt.show()

    def log_psi(x, v, up, a = None, b = None, d =1 / 252, e= None):
        log_m1 = ((a-1)/2)  * np.log(d) - 1/2 * np.log(2 * np.pi * up ** 2)
        log_m2 = a * np.log(b * up / e)

        log1 = -(x - v*d)**2 / (2 * up ** 2 * d)
        log2 = (e * (x - v * d) + b * up ** 2 * d) ** 2 / ((2 * e * up)**2 *d)

        d_input = (e * (x - v * d ) + b * up ** 2 * d) / (e * up * np.sqrt(d))
        final_add = [float(log_m1 + log_m2 + log1[i] + log2[i] + mp.log(pcfd(-a, d_input[i]))) for i in range(len(d_input))]
        return np.array(final_add)

    #TODO: check evaluation is correct
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

    def callback_stock(Xi):
        global Nfeval, d, RET, l1, a, b
        print('{0: 4d}     mu:{1:.4f}     sigma:{2: .4f}    l2:{3:.4f}    muV:{4:.4f}    sigmaV:{5:.4f}    e:{6:.4f}    loss:{7:.4f}'.format(Nfeval,
            Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], Xi[5],
            optimize_stock(Xi, x= RET,  l1= l1, a = a, b = b, d= d)))
        Nfeval += 1



    param_a1, a1 = [[-0.0425323, 0.272093, 12.0023, -0.012407,  0.0649111, 0.0606179], 1] #full length, a = 1
    mu_a1, sigma_a1, l2_a1, muV_a1, sigmaV_a1, e_a1 = param_a1
    Point_DensityStock = Density_stock(l1_a1, a1, b_a1, mu_a1, sigma_a1, l2_a1, muV_a1, sigmaV_a1, e_a1, RET, d=1 / 252)
    print(np.sum(np.log(Point_DensityStock)), optimize_stock(param_a1, l1=l1_a1, a=a1, b=b_a1, d=1 / 252, x=RET))

    param_a3, a3 = [[0.465593, 0.224503, 26.0084, -0.00140472, 0.0514438,  0.539438], 3]  # full length, a = 3
    mu_a3, sigma_a3, l2_a3, muV_a3, sigmaV_a3, e_a3 = param_a3
    Point_DensityStock = Density_stock(l1_a3, a3, b_a3, mu_a3, sigma_a3, l2_a3, muV_a3, sigmaV_a3, e_a3, RET, d = 1/252)
    print( np.sum(np.log(Point_DensityStock)), optimize_stock(param_a3, l1=l1_a3, a=a3, b=b_a3, d=1/252, x=RET))


    # [Density Plot]
    RET_grids = np.linspace(-0.15, 0.15, 100)
    Eval_Density_a1 = Density_stock(l1_a1, a1, b_a1, mu_a1, sigma_a1, l2_a1, muV_a1, sigmaV_a1, e_a1, RET_grids)
    Eval_Density_a3 = Density_stock(l1_a3, a3, b_a3, mu_a3, sigma_a3, l2_a3, muV_a3, sigmaV_a3, e_a3, RET_grids)
    Data_DensityStock = KDE_estimate(RET, RET_grids)

    # silverman_bw = 0.9 * min(np.std(RET), iqr(RET)/1.34) * RET.shape[0]**(-1/5)
    # KDE_model = KernelDensity(bandwidth=silverman_bw, kernel = 'gaussian').fit(RET.reshape(-1, 1))
    # log_density = KDE_model.score_samples(RET_grids.reshape(-1, 1))
    # Data_DensityStock = np.exp(log_density)

    plt.plot(RET_grids, Eval_Density_a1, label = 'Model Density (a=1)')
    plt.legend()

    plt.plot(RET_grids, Eval_Density_a3, label='Model Density (a=3)')
    plt.legend()

    plt.plot(RET_grids, Data_DensityStock, label = 'Data Density')
    plt.legend()
    plt.show()



    # [CoCo Evaluation]
    from scipy.integrate import quad
    from scipy.stats import hypsecant as sech
    from scipy.special import erfc, expi
    def CDensity_derivative(l1, b, a, x, t):
        m1 = l1 * b ** a * x**(a-1) * np.exp( -l1 * t - b * x)/ sc.factorial(a-1)
        m11 = l1 * t * (b * x)**a / (1 + a)
        u1_input1 = np.array([])
        u1_input2 = np.linspace(2 + 1 / a, 3, a)
        u1_input3 = l1 * t * (b * x / a) ** sc.special.poch(1+a, a)
        first_u = np.array([float(hyper(u1_input1, u1_input2, i, maxterms=10 ** 6)) for i in u1_input3])

        m2 = l1 * t - 1
        u2_input1 = np.array([])
        u2_input2 = np.linspace(1 + 1 / a, 2, a)
        u2_input3 = l1 * t * (b * x / a) ** a
        second_u = np.array([float(hyper(u2_input1, u2_input2, i, maxterms=10 ** 6)) for i in u2_input3])
        return m1 * m11 * first_u - m2 * second_u
    def I1(l1, a, b, t):
        integrand = lambda y: 1- b*y/(l1 * a * t) * CDensity_derivative(l1, b, a, y, t)
        lower = 0
        upper = 1 * a * t / b
        return quad(integrand, lower, upper)[0]
    def I2(l1, a, b, t, x):
        integrand = lambda y: CDensity_derivative(l1, b, a, y, t)
        lower = 0
        upper = x
        return quad(integrand, lower, upper)[0]

    def SupPr(l1, a, b, t, x):
        c1 = 1 - np.exp(-l1 * t)

        int2_func = lambda y: CDensity_derivative(l1, b, a, y, t)
        c2 = -quad(int2_func, 0, x + l1 * a * t /b)

        m3 = lambda s: I1(l1, a, b, t - s)
        input_CDensity = lambda s: x + l1 * a * s/ b
        input_I2 = lambda s: x + l1 * a * s/b
        int3_func = lambda s: m3(s) * (
            CDensity_derivative(l1, b, a, input_CDensity(s), t) -
            b / a * np.exp(-l1 * s) - 1 / l1 * I2(s, input_I2(s))
        )
        c3 = quad(int3_func, 0, t)
        return c1 + c2 + c3

    def E1(k1, xi1, t, u, t0 =0):
        m11 = k1 * (t - t0)/ (2 * xi1**2)
        m12_input = np.sqrt(2 * xi1**2 * (t-t0)**2 *u)
        m12 = np.tanh(m12_input) / m12_input -1
        log_c1 = m11 * m12

        # ToDo: t0 !=0
        # m2_input = 2 * k1**2 * (t-t0)**2 * u
        # m21 = (np.sech(m2_input) - 1)/k1**2 * k1 * t0 + xi1 *

        m3_input = np.sqrt(2 * xi1**2 * (t-t0)**2)
        m3 = np.sqrt(sech(m3_input))
        return np.exp(log_c1) * m3

    def a_func (muV, SigmaV, case):
        if case==1:
            numerator = muV + SigmaV
            denominator = np.sqrt( 2 * SigmaV**2)
            return 1 - 0.5 * erfc( numerator / denominator)
        elif case==2:
            numerator1 = muV + SigmaV
            denominator1 = np.sqrt(2 * SigmaV ** 2)
            numerator2 = muV + 2 * SigmaV
            denominator2 = np.sqrt(2 * SigmaV ** 2)
            return 0.5 * (erfc(numerator1/denominator1) - erfc(numerator2/denominator2))
        elif case==3:
            numerator1 = muV + 2 * SigmaV
            denominator1 = np.sqrt(2 * SigmaV ** 2)
            numerator2 = muV + 3 * SigmaV
            denominator2 = np.sqrt(2 * SigmaV ** 2)
            return 0.5 * (erfc(numerator1 / denominator1) - erfc(numerator2 / denominator2))
        elif case==4:
            numerator = muV + 3 * SigmaV
            denominator = np.sqrt(2 * SigmaV ** 2)
            return 0.5 * erfc ( numerator / denominator)

    def E2(k2, xi2, l2, l30, muV, SigmaV, t, u, t0 = 0):
        m11 = -u * l30
        m12 = 1 - np.exp(-k2 * (t- t0))
        c1 = - m11 * m12 / k2

        c2 = 0
        for i in range(1, 5):
            Ei_input1 = i * u * k2 * np.exp( -k2 *t0) / k2
            Ei_input2 = i * u * k2 * np.exp( -k2 *t) / k2
            c2 += a_func(muV, SigmaV, i) * np.exp( -i * xi2 * u /k2)  * (expi(Ei_input1) - expi(Ei_input2))

        c3 = -l2 * (t-t0)
        return np.exp(c1 + c2 - c3)

    def writedown_coco(r, K, T, l1, a, b, c,Jbar, M, w, k1, xi1, k2, xi2, l2, l30,  muV, SigmaV):
        m11 = K * np.exp( -r * T)
        m12 = 1 - SupPr( l1, np.ceil(a), b, T, Jbar)
        c1 = m11 * m12

        c2 = 0
        for k in range(1, M+1):
            c2 += c * np.exp ( -r * k * (T / M)) * (1 - SupPr( l1, np.ceil(a), b, k * T/M, Jbar))

        c3 = 0
        NN = 12 * T
        m31 = (1-w) * K
        for j in range(1, NN+1):
            m32 = np.exp(-r * T / NN * j)
            m33 = E1(k1, xi1, j * T/ NN, 1) * E2(k2, xi2, l2, l30, muV, SigmaV, j * T/NN, 1)
            m34 = SupPr(l1, a, b, (j+1) * (T / NN), Jbar) - SupPr(l1, a, b, j * (T / NN), Jbar)
            c3 += m31 * m32 * m33 * m34
        return c1 + c2 + c3

    def psi1(p, a, b):
        return (1 + p/b)**(-a) -1
    def psi2(muV, sigmaV, u):
        return np.exp(muV + sigmaV **2 * u**2 /2 -1)
    def Q(p, q, r, sigma, l1, l2, a, b):
        c1 = p * q
        c2 = (1-p) * r
        c3 = 0.5 * p * (1-p) * sigma**2
        c4 = l1 * (p * psi1(1, a, b) - psi1(p))
        c5 = l2 * (p * psi2(1) - psi2(p))
        return c1 + c2  +c3 + c4 + c5

    #ToDo: check e, p
    def equityconvert_coco(r, K, T, l1, a, b, c, e, p, Jbar, M, w, k1, xi1, k2, xi2, l2, l30, muV, SigmaV):
        m11 = K * np.exp(-r * T)
        m12 = 1 - SupPr(l1, np.ceil(a), b, T, Jbar)
        c1 = m11 * m12

        c2 = 0
        for k in range(1, M + 1):
            c2 += c * np.exp(-r * k * (T / M)) * (1 - SupPr(l1, np.ceil(a), b, k * T / M, Jbar))
        c3 = 0
        NN = 12 * T
        m31 = w * K
        for j in range(0, NN):
            m32 = np.exp(Q(p, q, r, sigma, l1, l2, a, b) * T/NN *j)
            m33 = E1(k1 + p * sigma * k1, xi1, j * T/NN, 1)
            m34 = E2(k2, xi2, l2 * (1+psi2( np.ceil(p))), l30, muV + p * SigmaV**2, SigmaV, j * T/NN, 1)
            m35 = SupPr(l1 * (1 + psi1( np.ceil(p), a, b)), np.ceil(a), b + e * p, (j +1) * T/NN, Jbar)
            m36 = SupPr(l1 * (1 + psi1( np.ceil(p), a ,b)), np.ceil(a), b + e * p, j * T/NN, Jbar)
            c3 += m32 * m33 * m34 * m35 * m36
        c3 *= m31
        return c3









    print('end')