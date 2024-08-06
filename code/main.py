import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import os
import time
from utils import *




def Callback_RelaxJ(Xi):
    global Nfeval, H
    print('{0: 4d}     {1:.4f}     {2: .4f}    {3:.4f} '.format(Nfeval, Xi[0], Xi[1], optimize_RelaxJ(Xi, H=H)))
    Nfeval += 1

def Callback_DensityJ(Xi):
    global Nfeval, H, a
    print('{0: 4d}     {1:.4f}     {2: .4f}    {3:.4f} '.format(Nfeval, Xi[0], Xi[1], optimize_DensityJ(Xi, a=a, H=H)))
    Nfeval += 1

def callback_stock(Xi):
    global Nfeval, RET, l1_a3, a3, b_a3
    print('{0: 4d}     mu:{1:.4f}     sigma:{2: .4f}    l2:{3:.4f}    muV:{4:.4f}    sigmaV:{5:.4f}    e:{6:.4f}    loss:{7:.4f}'.format(Nfeval,
        Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], Xi[5],
        optimize_stock(Xi, x= RET,  l1= l1_a3, a = a3, b = b_a3, d= 1/252)))
    Nfeval += 1

def Callback_CDS(Xi):
    global Nfeval, l2_a3, muV_a3, sigmaV_a3, cds_values#, t_cds, T_cds
    print(
        '{0: 4d}     k1:{1:.4f}     xi1:{2: .4f}    k2:{3:.4f}    xi2:{4:.4f}     l32:{5:.4f}    loss:{6:.4f}'.format(
            Nfeval,
            Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], optimize_cds(Xi, l2_a3, muV_a3, sigmaV_a3, 5, 0, cds_values
                         )))
    Nfeval += 1

if __name__ == '__main__':
    #file_name = 'Lloyds_Data_13-23.xlsx'
    file_name = 'Credit_Suisse_Data_13-23.xlsx'
    #file_name = 'China_Construction_Bank_Data_13-23.xlsx'
    save_fig = False

    return_data, cet_data, B, J, B_date = load_data('../data', file_name)
    Diff_J = np.diff(J)

    J_grids = np.linspace(-3, 3, 200).reshape(-1, 1)


    PJ_grids = KDE_estimate(Diff_J, J_grids)
    H = Diff_J - min(Diff_J) + 0.01

    # H_grids = np.linspace(1.1, 1.25, 200).reshape(-1, 1)
    # PH_grids = KDE_estimate(H, H_grids)

    # [Plotting]
    fig = plt.figure(figsize=(6, 12))
    ax1 = fig.add_subplot(311)
    ax1.plot(J_grids, PJ_grids, label = 'Diff_J Density')
    plt.legend()

    ax2 = fig.add_subplot(312)
    ax2.plot(B, label='B', color = 'red')
    plt.legend()

    ax3 = fig.add_subplot(313)
    ax3.plot(H, label='H', color = 'blue')
    plt.legend()
    plt.show()

    # [1. Latent J Optimization]
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
        if a == 1:
            l1_a1, b_a1 = res.x
    sorted_list = sorted(sorted_list)
    param_table = pd.DataFrame(sorted_list, columns= ['loss', 'l1', 'b', 'a'])
    #param_table.to_csv('../param/J_' + file_name.split('.')[0] + '.csv', index = False)
    loss, l1_a3, b_a3, a3 = sorted_list[0]

    # [Density Plot]
    H_grids = np.linspace(0, 1, 100)
    # Eval_Density_a1 = Density_stock(l1_a1, a1, b_a1, mu_a1, sigma_a1, l2_a1, muV_a1, sigmaV_a1, e_a1, RET_grids)
    Eval_JDensity_a3 = np.exp(Density_J(l1_a3, int(a3), b_a3, T, H_grids))
    Data_JDensity_data = KDE_estimate(H, H_grids)

    plt.plot(H_grids, Data_JDensity_data, label='Kernel')
    plt.plot(H_grids, Eval_JDensity_a3, linestyle='--', label='Fitted')
    plt.legend()
    if save_fig:
        plt.savefig('../figure/Hdensity_'+ file_name.split('_')[0]+'.jpg', dpi=600)
    plt.show()

    # [2. Stock optimization]
    if file_name == 'Credit_Suisse_Data_13-23.xlsx':
        RET_data = pd.read_excel(os.path.join('../data', file_name), sheet_name=0)
        S_name = 'Price or Bid/Ask Average'
        RET = RET_data['Log-returns (without Dividends)'].values[:2570]
    else:
        RET_data = pd.read_excel(os.path.join('../data', file_name), sheet_name=0)
        S_name = 'Price or Bid/Ask Average'
        RET = RET_data['Log-returns without dividends'].values

    # Figure to be added
    S_data = RET_data[[S_name, 'Names Date']]
    S_data.columns = ['S', 'Date']
    S_data = S_data.set_index('Date')
    S_data.plot(label = 'S')
    if save_fig:
        plt.savefig('../figure/S_'+ file_name.split('_')[0]+'.jpg', dpi=600)
    plt.show()


    # Figure to be added
    plt.plot(pd.to_datetime(B_date), B, label = 'B')
    plt.legend()
    if save_fig:
        plt.savefig('../figure/B_' + file_name.split('_')[0] + '.jpg', dpi=600)
    plt.show()

    param_a1, a1 = [[-0.0425323, 0.272093, 12.0023, -0.012407,  0.0649111, 0.0606179], 1] #full length, a = 1
    mu_a1, sigma_a1, l2_a1, muV_a1, sigmaV_a1, e_a1 = param_a1
    Point_DensityStock = Density_stock(l1_a1, a1, b_a1, mu_a1, sigma_a1, l2_a1, muV_a1, sigmaV_a1, e_a1, RET, d = 1/ 252)
    print(np.sum(np.log(Point_DensityStock)), optimize_stock(param_a1, l1=l1_a1, a=a1, b=b_a1, d=1 / 252, x=RET))

    param_a3, a3 = [[0.465593, 0.224503, 26.0084, -0.00140472, 0.0514438,  0.539438], a3]  # full length, a = 3
    mu_a3, sigma_a3, l2_a3, muV_a3, sigmaV_a3, e_a3 = param_a3
    Point_DensityStock = Density_stock(l1_a3, a3, b_a3, mu_a3, sigma_a3, l2_a3, muV_a3, sigmaV_a3, e_a3, RET, d = 1/252)
    print( np.sum(np.log(Point_DensityStock)), optimize_stock(param_a3, l1=l1_a3, a=a3, b=b_a3, d=1/252, x=RET))


    #mu, sigma, l2, muV, SigmaV, e = param
    # bounds = [(-1, 1), (0, 1), (0, None), (-1, 1), (0, 1), (0, None)]
    # #init, Nfeval = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 1]
    # init, Nfeval = [param_a3, 1]
    # stock_res = minimize(optimize_stock, init, args=(l1_a3, a3, b_a3, 1/252, RET), method='Nelder-Mead',
    #                     options={'maxiter': 100}, bounds=bounds, callback = callback_stock, tol=0.001)
    # mu_a3, sigma_a3, l2_a3, muV_a3, sigmaV_a3, e_a3 = stock_res.x
    # stock_param = pd.DataFrame([stock_res.fun, l1_a3, b_a3, a3, mu_a3, sigma_a3, l2_a3, muV_a3, sigmaV_a3, e_a3], index = ['loss', 'l1', 'b', 'a', 'mu', 'sigma', 'l2', 'muV', 'sigmaV', 'e']).T
    # stock_param.to_csv('../param/Stock_' + file_name.split('.')[0] + '.csv', index = False)

    stock_param = pd.read_csv('../param/Stock_' + file_name.split('.')[0] + '.csv')
    loss_a3, l1_a3, b_a3, a3, mu_a3, sigma_a3, l2_a3, muV_a3, sigmaV_a3, e_a3 = list(stock_param.values.squeeze())

    # [Density Plot]
    RET_grids = np.linspace(-0.15, 0.15, 100)
    #Eval_Density_a1 = Density_stock(l1_a1, a1, b_a1, mu_a1, sigma_a1, l2_a1, muV_a1, sigmaV_a1, e_a1, RET_grids)
    Eval_Density_a3 = Density_stock(l1_a3, a3, b_a3, mu_a3, sigma_a3, l2_a3, muV_a3, sigmaV_a3, e_a3, RET_grids)
    Data_DensityStock = KDE_estimate(RET, RET_grids)

    # plt.plot(RET_grids, Eval_Density_a1, label = 'Model Density (a=1)')
    # plt.legend()


    plt.plot(RET_grids, Data_DensityStock, label='Kernel')
    plt.plot(RET_grids, Eval_Density_a3, linestyle='--', label='Fitted')
    plt.legend()
    if save_fig:
       plt.savefig('../figure/Sdensity_'+ file_name.split('_')[0]+'.jpg', dpi=600)
    plt.show()


    # [Evaluation part]
    # LLyolds: 2001 - 2011, Quarterly CET1 ratio, K, r(term structure), W(B_) = 5%, T,

    W = 0.05 # for llyods
    C0 = 6.3/100 # for llyods
    wbar = 1 # for llyods
    S0 = 150
    K = 100
    c = 6.75 / 2
    w = 0.5
    r = 0.0168
    q = 0.02
    T = 5
    M = 2 * T

    k1 = 0.09
    xi1 = 0.001
    k2 = 7.2
    xi2 = 0.04

    l32 = 0.1
    l31r = 0.1


    # Question: how is B bar determined? Is C0 the initial CET1 ratio?, range of Jbar for spline grid?

    Jbar = np.tan( np.pi * (1 - 2 * W)/2) + 1/ np.tan(np.pi * (1-C0))
    param_a3, a3 = [[0.465593, 0.224503, 26.0084, -0.00140472, 0.0514438, 0.539438], 3]  # full length, a = 3
    mu_a3, sigma_a3, l2_a3, muV_a3, sigmaV_a3, e_a3 = param_a3

    oneyear_cds = pd.read_csv('../data/CSGN1YEUAM=R Overview.csv').set_index('Date')#['Price']
    oneyear_cds['T'] = 1
    fiveyear_cds = pd.read_csv('../data/CSGN5YEUAM=R Overview.csv').set_index('Date')  # ['Price']
    fiveyear_cds['T'] = 5
    tenyear_cds = pd.read_csv('../data/CSGN10YEUAM=R Overview.csv').set_index('Date')#['Price']
    tenyear_cds['T'] = 10


    cds_data = pd.concat([oneyear_cds, tenyear_cds, fiveyear_cds])[['Price', 'T']]
    cds_data['Price'] = cds_data['Price'].apply(lambda x: x.replace(',', '')).astype(float)/100
    cds_data.index = pd.to_datetime(cds_data.index)
    cds_data = cds_data.sort_index()
    date_flag = cds_data.index.isin(pd.date_range(start='3/1/2023', end='3/17/2023'))
    cds_values = cds_data['Price'].values[date_flag]
    date_values = cds_data.index[date_flag]
    mat_values = cds_data['T'].values[date_flag]


    import seaborn as sns
    plt.figure(figsize = (4, 4))
    cds_data['T'] = cds_data['T'].astype(str)
    sns.scatterplot(data=cds_data[date_flag].sort_values('T'), x='Date', y='Price', hue='T', style='T')
    #plt.legend(loc = 'upper left')
    plt.ylabel('CDS Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_fig:
        plt.savefig('../figure/CDS_'+ file_name.split('_')[0]+'.jpg', dpi=600)
    plt.show()
    def optimize_cds(param, l2=None, muV=None, sigmaV=None, T = None , t0 = None, cds_price = None):
        #k1, xi1, k2, xi2, l31r, l32 = param
        #model_price = CDS_spread(k1, xi1, k2, xi2, l2, l32, muV, sigmaV, T, l31r, t0)
        k1, xi1, k2, xi2, l32 = param
        model_price = CDS_spread(k1, xi1, k2, xi2, l2, l32, muV, sigmaV, T, 0, t0)
        loss = np.abs(cds_price - model_price)/ cds_price
        return np.mean(loss)

    #k1, xi1, k2, xi2, l31r, l32
    init, Nfeval = [[1.5, 0.5, 1.5, 0.5, 0.5], 1]
    bounds = [(0, 5), (0.01, 2), (1, 10), (0.01, 2), (0.1, 10)]
    #t_cds = np.array([ (i - date_values[0]).days/252 for i in date_values])
    #T_cds = np.array([i + 5 for i in t_cds])
    res = minimize(optimize_cds, init, args=(l2_a3, muV_a3, sigmaV_a3, mat_values, 0, cds_values), method='Nelder-Mead', options={'maxiter': 100},
                   callback = Callback_CDS, bounds=bounds, tol=0.001)
    k1, xi1, k2, xi2, l32 = res.x


    def optimize_wdcoco(param, r= None, K = None, T = None, l1_a3= None, a3 = None, b_a3 = None, c = None,
                        Jbar = None, M = None,  k1 = None, xi1 = None, k2= None, xi2 = None, l2_a3 = None,
                        l32 = None, muV_a3 = None, sigmaV_a3 = None, coco_price = None):
        w, w_bar, p = param
        model_price =writedown_coco(r, K, T, l1_a3, a3, b_a3, c,
                                    Jbar, M, w, k1, xi1, k2, xi2,
                                    l2_a3, l32, muV_a3, sigmaV_a3)
        loss = np.abs(model_price - coco_price)/ coco_price
        return np.mean(loss)


    init, Nfeval = [[0.3, 0.3, 0.3], 1]
    bounds = [(0, 1), (0, 1), (0, 1)]
    res = minimize(optimize_wdcoco, init, args=(l2_a3, muV_a3, sigmaV_a3, mat_values, 0, cds_values), method='Nelder-Mead', options={'maxiter': 100},
                   callback = None, bounds=bounds, tol=0.001)


    start = time.time()
    wd_value = writedown_coco(r, K, T, l1_a3, a3, b_a3, c, Jbar, M, w, wbar, k1, xi1, k2, xi2, l2_a3, l32, muV_a3, sigmaV_a3)
    end = time.time()
    # print(wd_value, end - start)

    p = 0.6
    start = time.time()
    ec_value = equityconvert_coco(r, K, T, l1_a3, a3, b_a3, c, e_a3, p, q, Jbar, M, w, wbar, k1, xi1, k2, xi2, l2_a3, l32, muV_a3, sigmaV_a3, sigma_a3) #ToDo: CoCo w.r.t. p
    end = time.time()
    print(ec_value, end - start)

    print('end')