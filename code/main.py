import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sc
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize


def Density_J(l, a, b ,t ,x):
    multiplier = np.exp(-l * t)
    first_m = l * b**a * t * x**(a-1) * np.exp(-b * x)/ sc.factorial(a-1)

    u_input1 = np.array([])# TODO: check taking only two inputs
    u_input2 = np.linspace(1, 2, a)
    u_input3 = l * t * (b * x/a)**a

    first_u = sc.hyp1f1(u_input1, u_input2, u_input3)
    return multiplier *(first_m * first_u)

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

    i_input = 2 * np.sqrt(l * b * t *x)
    first_i = sc.j0(i_input)
    return multiplier * first_m * first_i

def optimize_RelaxJ(params, T= 1/4, H = None):
    l, b = params
    return -np.sum(Density_RelaxJ(l, b, T, H))

def Callback_RelaxJ(Xi):
    global Nfeval, H
    print('{0: 4d}     {1:.4f}     {2: .4f}    {3:.4f} '.format(Nfeval, Xi[0], Xi[1], optimize_RelaxJ(Xi, H = H)))
    Nfeval += 1

if __name__ == '__main__':
    return_data = pd.read_excel('../data/Credit_Suisse_Data_17-23.xlsx', sheet_name=0)
    cet_data = pd.read_excel('../data/Credit_Suisse_Data_17-23.xlsx', sheet_name = 1)
    B = cet_data['CET-1 ratio'].values
    J = np.tanh(np.pi * 0.5 - np.pi  * B ) + 1 / np.tan(np.pi * (1-B[0]))
    Diff_J = np.diff(J)

    KDE_J = KernelDensity(bandwidth= 'silverman').fit(Diff_J.reshape(-1,1))
    J_grids = np.linspace(-3, 3, 200).reshape(-1,1)
    PJ_grids = np.exp(KDE_J.score_samples(J_grids))
    H = Diff_J + 1.2

    KDE_H = KernelDensity(bandwidth= 'silverman').fit(H.reshape(-1,1))
    H_grids = np.linspace(1.1, 1.25, 200).reshape(-1, 1)
    PH_grids = np.exp(KDE_H.score_samples(H_grids))


    # [plotting]
    fig = plt.figure(figsize=(6, 12))
    ax1 = fig.add_subplot(311)
    ax1.plot(J_grids, PJ_grids, label = 'Diff_J Density')
    plt.legend()

    ax2 = fig.add_subplot(312)
    ax2.plot(1-B, label = 'B', color = 'red')
    plt.legend()

    ax3 = fig.add_subplot(313)
    ax3.plot(H, label='H', color = 'blue')
    plt.legend()
    plt.show()


    # [Optimization]
    l, b, T = [1, 1, 1/4]
    Test_RelaxJ = Density_RelaxJ(l, b, T, H)

    bounds = [(0, None), (0, None)]
    init = [0.1, 0.1]
    Nfeval = 1
    res = minimize(optimize_RelaxJ, init, args=(T, H), method='L-BFGS-B', options={'maxiter': 50000}, bounds=bounds, callback=Callback_RelaxJ, tol = 0.001)
    l1, b = res.x
    print('lambda: %.4f, beta %.4f' % (l1, b))

    #[Re-estimation for alpha = 1...6]
    # for a in range(1, 6):
    #     init = res.x
    #     Nfeval = 1
    #     print('==========optimization for a = %d===========' % a)
    #     res = minimize(optimize_DensityJ, init, args=(a, T, H), method='L-BFGS-B', options={'maxiter': 50000}, bounds=bounds,
    #                    callback=Callback_RelaxJ, tol=0.001)
    #     print('lambda: %.4f, beta %.4f' % (res.x[0], res.x[1]))


    # [For stock estimation]
    RET = return_data['Returns'].values
    plt.plot(RET)
    plt.show()

    def psi(x, v, u_, a = None, b = None, d = 1/252, e= None):
        m1 = d ** ((a-1)/2) / np.sqrt( 2* np.pi * u_**2)
        m2 = (b * u_/e ) **a

        log1 = -(x - v*d)**2 / (2 * u_**2 * d)
        log2 = (e * (x- v*d) + b * u_**2 * d)**2

        d_input = (e * (x - v *d) + b * u_**2 *d) / (e * u_ * np.sqrt(d))
        last_d = sc.pbdv(-a, d_input)[0] # TODO: check -a is correct

        return m1 * m2 * np.exp(log1 + log2) * last_d #TODO: check large negative return -0.52 causing error

    def Density_stock(l1, a, b, mu, sigma, l2, muV, sigmaV, e, x, d= 1/252):
        v = mu + muV/d
        u_ = np.sqrt(sigma**2 + sigmaV**2/d)

        psi1 = psi(x, v, u_, a = a, b = b, d = d, e = e)
        psi2 = psi(x, mu, sigma, a = a, b = b, d = d, e = e) # TODO: different between note and mathematica code
        #psi2 = psi1

        first = l1 * l2 * d**2 * psi1

        m2 = l2 * d * (1- l1 * d) /np.sqrt( 2 * np.pi *u_**2 * d)
        log2 = -(x-v*d)**2 / (2 * u_**2 * d)

        third = l1 * d * (1-l2*d) * psi2

        m4 = (1 - l1*d) * (1 - l2*d) / np.sqrt(2 * np.pi * sigma**2 *d)
        log4 = -(x - mu * d)**2 / (2 * u_**2 * d)

        return first + m2 * np.exp(log2) + third + m4 * np.exp(log4)

    def optimize_stock(param, l1 = None, a= None, b = None, d = 1/252, x = None):
        mu, sigma, l2, muV, SigmaV, e = param
        return -np.sum(np.log(Density_stock(l1, a, b, mu, sigma, l2, muV, SigmaV, e, x, d)))

    def callback_stock(Xi):
        global Nfeval, l1, a_temp, b,  d, RET #TODO: change a_temp
        print('{0: 4d}     {1:.4f}     {2: .4f}    {3:.4f}    {4:.4f}    {5:.4f}    {6:.4f}    {7:.4f}'.format(Nfeval,
            Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], Xi[5],
            optimize_stock(Xi, x= RET,  l1= l1, a = a_temp, b = b, d= d)))
        Nfeval += 1


    l1_temp, a_temp, b_temp, mu, sigma, l2, muV, sigmaV, e = [1] * 9
    Test_DensityStock = Density_stock(l1_temp, a_temp, b_temp, mu, sigma, l2, muV, sigmaV, e, RET)

    bounds = [(-1, 1), (0.01, 1), (0, None), (-1,1), (0.01, 1), (0, None)]
    init = [0, 0.5, l1, 0, 0.5, 0.5]
    Nfeval = 1
    d = 1/252
    res2 = minimize(optimize_stock, init, args=(l1, 1, b, d, RET), method='L-BFGS-B', options={'maxiter': 50000}, bounds=bounds,
                   callback=callback_stock, tol=0.001)

    print('end')