import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sc
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize
from mpmath import hyper, pcfd
import mpmath as mp
import warnings
def Density_J(l, a, b ,t ,x):
    multiplier = np.exp(-l * t)
    first_m = l * b**a * t * x**(a-1) * np.exp(-b * x)/ sc.factorial(a-1)

    u_input1 = np.array([])
    u_input2 = np.linspace(1 + 1/a, 2, a)
    u_input3 = l * t * (b * x/a)**a

    #first_u = sc.hyp1f1(u_input1, u_input2, u_input3)
    first_u = np.array([float(hyper(u_input1, u_input2, i,  maxterms=10**6)) for i in u_input3])
    return np.log(multiplier) + np.log(first_m) + np.log(first_u)

def optimize_DensityJ(param, a = 1, T= 1/4, H = None):
    l, b = param
    return -np.sum(Density_J(l, a, b, T, H))

def Callback_DensityJ(Xi):
    global Nfeval, H, a
    print('{0: 4d}     {1:.4f}     {2: .4f}    {3:.4f} '.format(Nfeval, Xi[0], Xi[1], optimize_DensityJ(Xi, a=a, H = H)))
    Nfeval += 1


def Density_RelaxJ(l, b ,t, x):
    multiplier = np.exp(-l * t)
    first_m = np.exp(-b * x) * np.sqrt(l * b * t/x)

    i_input = 2 * np.sqrt(l * b * t * x)
    first_i = sc.i1(i_input)
    return np.log(multiplier) + np.log(first_m) + np.log(first_i)

def optimize_RelaxJ(params, T= 1/4, H = None):
    l, b = params
    return -np.sum(Density_RelaxJ(l, b, T, H))

def Callback_RelaxJ(Xi):
    global Nfeval, H
    print('{0: 4d}     {1:.4f}     {2: .4f}    {3:.4f} '.format(Nfeval, Xi[0], Xi[1], optimize_RelaxJ(Xi, H = H)))
    Nfeval += 1

if __name__ == '__main__':
    #return_data = pd.read_excel('../data/Credit_Suisse_Data_17-23.xlsx', sheet_name=0)
    #cet_data = pd.read_excel('../data/Credit_Suisse_Data_17-23.xlsx', sheet_name = 1)
    #B = cet_data['CET-1 ratio'].values

    return_data = pd.read_excel('../data/Credit_Suisse_Data_13-23.xlsx', sheet_name=0).iloc[16:]
    cet_data = pd.read_excel('../data/Credit_Suisse_Data_13-23.xlsx', sheet_name = 1).iloc[16:]

    B = cet_data['CET-1 ratio (phase-in)'].values
    J = np.tan(np.pi * 0.5 - np.pi  * B ) + 1 / np.tan(np.pi * (1-B[0]))
    Diff_J = np.diff(J)

    KDE_J = KernelDensity(bandwidth= 'silverman').fit(Diff_J.reshape(-1,1))
    J_grids = np.linspace(-3, 3, 200).reshape(-1,1)
    PJ_grids = np.exp(KDE_J.score_samples(J_grids))
    H = Diff_J - min(Diff_J) + 0.01

    KDE_H = KernelDensity(bandwidth= 'silverman').fit(H.reshape(-1,1))
    H_grids = np.linspace(1.1, 1.25, 200).reshape(-1, 1)
    PH_grids = np.exp(KDE_H.score_samples(H_grids))

    # [debug density 2.1.3]
    # H_grids = np.linspace(0.001, 1, 100)
    # # Density_J(l, a, b ,t ,x):
    # #Test_DensityJ_Mine = Density_J(12.020007680203374, 1, 100, T, H_grids)
    # Test_DensityJ_Weixuan = np.exp(Density_J(27.74, 2, 46.3, 1/4, H))
    # # Test_DensityJ_Mine = np.exp(Density_RelaxJ(12.020007680203374, 100, 1/4, H_grids))
    # #Test_DensityJ_Weixuan2 = np.exp(Density_RelaxJ(32.28, 26.95, 1/4, H_grids))
    # # plt.plot(H_grids, Test_DensityJ_Weixuan1, label='liang')
    # plt.plot(H_grids, Test_DensityJ_Weixuan, label='weixuan')
    # plt.legend()
    # plt.show()

    # [plotting]
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




    bounds = [(0, 100), (0, 100)]
    init = [0.1, 0.1]
    Nfeval = 1
    res = minimize(optimize_RelaxJ, init, args=(T, H), method='Nelder-Mead', options={'maxiter': 100}, bounds=bounds, callback=Callback_RelaxJ, tol = 0.001)
    l1, b = res.x
    print('lambda: %.4f, beta %.4f' % (l1, b))

    #[Re-estimation for alpha = 1...6]
    sorted_list = []
    for a in range(1, 6):
        Nfeval = 1
        print('==========optimization for a = %d===========' % a)
        res = minimize(optimize_DensityJ, init, args=(a, T, H), method='Nelder-Mead', options={'maxiter': 100}, bounds=bounds,
                       callback=Callback_DensityJ, tol=0.001)
        print('lambda: %.4f, beta %.4f' % (res.x[0], res.x[1]))
        sorted_list.append([res.fun, *res.x, a])
    sorted_list = sorted(sorted_list)
    loss, l1, b, a = sorted_list[0]
    # [For stock estimation]
    RET = return_data['Returns'].values
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
        global Nfeval, d, RET #TODO: change a_temp
        print('{0: 4d}     mu:{1:.4f}     sigma:{2: .4f}    l2:{3:.4f}    muV:{4:.4f}    sigmaV:{5:.4f}    e:{6:.4f}    loss:{7:.4f}'.format(Nfeval,
            Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], Xi[5],
            optimize_stock(Xi, x= RET,  l1= 98.1804, a = 5, b = 126.174, d= d)))
        Nfeval += 1


    #l1_temp, a_temp, b_temp, mu, sigma, l2, muV, sigmaV, e = [1] * 9
    #Test_DensityStock = Density_stock(l1_temp, a_temp, b_temp, mu, sigma, l2, muV, sigmaV, e, RET)
    param = [0.186287, 0.225406, 41.0965, 0.0113117, 0.0343969, 0.822461]
    mu_hat, sigma_hat, l2, muV, sigmaV, e = param

    RET_point = RET
    Point_DensityStock = Density_stock(l1, a, b, mu_hat, sigma_hat, l2, muV, sigmaV, e, RET_point, d = 1/252)

    print( np.sum(np.log(Point_DensityStock)), optimize_stock(param, l1=l1, a=a, b=b, d=1 / 252, x=RET))

    # mu, sigma, l2, muV, SigmaV, e
    bounds = [(-1, 1), (0.01, 1), (0, None), (-1,1), (0.01, 1), (0, None)]
    init = [-0.123329, 0.202993, 4.8, 0.067659, 0.0179964, 1.79029]
    Nfeval = 1
    d = 1/252
    res2 = minimize(optimize_stock, init, args=(98.1804, 5, 126.174, d, RET), method='Nelder-Mead', options={'maxiter': 5000}, bounds=bounds,
                   callback=callback_stock, tol=0.001)

    # [Density Plot]
    mu_hat, sigma_hat, l2, muV, sigmaV, e = [-0.123329, 0.202993, 2.3138, 0.067659, 0.0179964, 1.79029]
    RET_grids = np.linspace(-0.15, 0.15, 100)
    Eval_DensityStock = Density_stock(98.1804, 5, 126.174, mu_hat, sigma_hat, l2, muV, sigmaV, e, RET_grids)
    plt.plot(RET_grids, Eval_DensityStock)
    plt.show()


    print('end')