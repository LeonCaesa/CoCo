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


def load_data(file_dir, start_idx = 0):
    return_data = pd.read_excel(os.path.join(file_dir, 'Credit_Suisse_Data_13-23.xlsx'), sheet_name=0).iloc[start_idx:]
    cet_data = pd.read_excel(os.path.join(file_dir, 'Credit_Suisse_Data_13-23.xlsx'), sheet_name=1).iloc[start_idx:]

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

def KDE_estimate(fit_data, eval_data, bw_selection = 'silverman'):
    KDE_model = KernelDensity(bandwidth=bw_selection).fit(fit_data.reshape(-1, 1))
    log_density = KDE_model.score_samples(eval_data.reshape(-1, 1))
    return np.exp(log_density)



if __name__ == '__main__':

    return_data, cet_data, B, J = load_data('../data', start_idx = 0)
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
    sorted_list = sorted(sorted_list)
    loss, l1, b, a = sorted_list[0]

    # [Stock estimation]

    #RET = pd.read_excel(os.path.join('../data', 'Credit_Suisse_Data_13-23.xlsx'), sheet_name=0)['Log-returns (without Dividends)'].values[997:2520]
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
        #loglast_d = np.log(sc.pbdv(-a, d_input)[0])

        #loglast_d = np.log(float(pcfd(-a, d_input)))
        #inf_flag = np.isinf(loglast_d)
        # return log_m1 + log_m2 + log1 + log2 + loglast_d #TODO: check large negative return -0.52 causing error

        final_add = [float(log_m1 + log_m2 + log1[i] + log2[i] + mp.log(pcfd(-a, d_input[i]))) for i in range(len(d_input))]
        return np.array(final_add)

    #TODO: check evaluation is correct
    def Density_stock(l1, a, b, mu, sigma, l2, muV, sigmaV, e, x, d= 1/252):
        v = mu + muV/d
        up = np.sqrt(sigma**2 + sigmaV**2/d)


        psi1 = np.exp(log_psi(x, v, up, a = a, b = b, d = d, e = e))
        #print('v, up, psi1 is %.2f, %.2f, %.2f' %(v, up, np.sum(psi1)))
        psi2 = np.exp(log_psi(x, mu, sigma, a = a, b = b, d = d, e = e))
        #print('mu, sigma, psi2 is %.2f, %.2f, %.2f' % (mu, sigma, np.sum(psi2)))

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


    #param = [0.186287, 0.225406, 41.0965, 0.0113117, 0.0343969, 0.822461] #
    param = [-0.0425323, 0.272093, 12.0023, -0.012407,  0.0649111, 0.0606179] #full length
    mu_hat, sigma_hat, l2, muV, sigmaV, e = param
    Point_DensityStock = Density_stock(l1, a, b, mu_hat, sigma_hat, l2, muV, sigmaV, e, RET, d = 1/252)
    print( np.sum(np.log(Point_DensityStock)), optimize_stock(param, l1=l1, a=a, b=b, d=1 / 252, x = RET))

    # mu, sigma, l2, muV, SigmaV, e
    # bounds = [(-1, 1), (0.01, 1), (0, None), (-1,1), (0.01, 1), (0, None)]
    # init = [-0.123329, 0.202993, 4.8, 0.067659, 0.0179964, 1.79029]
    # Nfeval, d = [1, 1/252]
    # res2 = minimize(optimize_stock, init, args=(98.1804, 5, 126.174, d, RET), method='Nelder-Mead', options={'maxiter': 5000}, bounds=bounds,
    #                callback=callback_stock, tol=0.001)

    # [Density Plot]
    RET_grids = np.linspace(-0.15, 0.15, 100)
    Eval_DensityStock = Density_stock(l1, a, b, mu_hat, sigma_hat, l2, muV, sigmaV, e, RET_grids)
    #Data_DensityStock = KDE_estimate(RET, RET_grids)

    #silverman_bw = 1.06 * np.std(RET) * RET.shape[0]**(-1/5)

    #A = min(np.std(RET), iqr(RET)/1.34)
    #silverman_bw = ((4 * A **5)/ (3 * RET.shape[0]))**(1/5)

    silverman_bw = 0.9 * min(np.std(RET), iqr(RET)/1.34) * RET.shape[0]**(-1/5)
    KDE_model = KernelDensity(bandwidth=silverman_bw, kernel = 'gaussian').fit(RET.reshape(-1, 1))
    log_density = KDE_model.score_samples(RET_grids.reshape(-1, 1))
    Data_DensityStock = np.exp(log_density)

    plt.plot(RET_grids, Eval_DensityStock, label = 'Model Density')
    plt.legend()

        # TODO: check why RET_grids density doesn't match up
    plt.plot(RET_grids, Data_DensityStock, label = 'Data Density')
    plt.legend()
    plt.show()



    # [CoCo Evaluation]
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

    print('end')