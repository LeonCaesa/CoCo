from scipy.optimize import minimize
import matplotlib.pyplot as plt
from utils import *
import seaborn as sns
import pandas as pd
import time


def optimize_convertcoco(param, r=None, q=None, K=None, T=None, t0=None, c=None, M=None,
                         coco_price=None,  # data
                         l1=None, a=None, b=None,  # latent params
                         k1=None, xi1=None, k2=None, xi2=None, l32=None, ignore_gov=None,  # intervention params
                         sigma=None, l2=None, muV=None, sigmaV=None, e=None, St=None):
    w, p, Jbar, w_bar = param
    model_price = equityconvert_coco(r, K, T, t0,
                                     l1, a, b,
                                     c, e, p, q,
                                     Jbar, M, w, w_bar,
                                     k1, xi1, k2, xi2,
                                     l2, l32, muV, sigmaV, sigma,
                                     ignore_gov, St)

    loss = np.sqrt(np.mean((model_price - coco_price) ** 2))
    return loss

def Callback_CoCo_CaseStudy(Xi):
    global Nfeval, r, q, K, T, t0, c, Jbar, M, coco_price, l1_a3, a3, b_a3, k1, xi1, k2, xi2, l32, ignore_gov, sigma_a3, l2_a3, muV_a3, sigmaV_a3, e_a3, St
    func_values = optimize_convertcoco(Xi, r=r, q=q, K=K, T=T, t0=t0, c=c,  M=M, coco_price=coco_price,
                         l1=l1_a3, a=a3, b=b_a3,  # latent params
                         k1=k1, xi1=xi1, k2=k2, xi2=xi2, l32=l32, ignore_gov=ignore_gov,  # intervention params
                         sigma=sigma_a3, l2=l2_a3, muV=muV_a3, sigmaV=sigmaV_a3, e=e_a3, St = St)


    # print('{0: 4d}     w:{1:.4f}     p:{2: .4f}    loss:{3:.4f}'.format(
    #         Nfeval, Xi[0], Xi[1], func_values))
    print('{0: 4d}     w:{1:.4f}     p:{2: .4f}     Jbar:{3: .4f}    wbar:{4: .4f}    loss:{5:.4f}'.format(
                Nfeval, Xi[0], Xi[1], Xi[2], Xi[3], func_values))
    Nfeval += 1

if __name__ == '__main__':
    # [Read stock parameters]
    # file_name = 'Lloyds_Data_13-23.xlsx'
    file_name = 'Credit_Suisse_Data_13-23.xlsx'
    # file_name = 'China_Construction_Bank_Data_13-23.xlsx'
    save_fig = False
    save_param = False

    intervention_param = pd.read_csv('../param/Intervention_' + file_name.split('.')[0] + '.csv')
    (loss_a3, l1_a3, b_a3, a3, mu_a3, sigma_a3,
     l2_a3, muV_a3, sigmaV_a3, e_a3,
     k1, xi1, k2, xi2, l32) = list(intervention_param.values.squeeze())

    # [Read coco data]
    data = pd.read_excel('../data/Charlie1124/case4/data-case4.xlsx')
    data = data.set_index('Date')
    #date_flag = data.index.isin(pd.date_range(start='3/1/2023', end='3/17/2023'))
    date_flag = data.index.isin(pd.date_range(start='1/1/2020', end='3/17/2023'))
    data = data[date_flag]

    data = data.dropna(how='any')
    St = data['Stock'].values
    RET = data['return without dividend'] / 100

    r = data['r'].values / 100
    coco_price = data['CoCo'].values
    T,c = [7, 7.25/100/2]
    maturity = pd.to_datetime(data.index[0]) + pd.DateOffset(years=T)
    t0 = np.array(range(data.index.shape[0])) / 252

    K = 100
    M = 14
    ignore_gov = False
    q = 0# ToDo: check


    init, Nfeval = [[0.3, 0.3, 0.3, 0.3], 1]
    bounds = [(0, 1), (0, 1), (0, 1), (0.1, 1)]
    res = minimize(optimize_convertcoco, init, args=( r, q, K, T, t0, c, M, coco_price,  # data
                                                     l1_a3, a3, b_a3,  # latent params
                                                     k1, xi1, k2, xi2, l32, ignore_gov,  # intervention params
                                                     sigma_a3, l2_a3, muV_a3, sigmaV_a3, e_a3, St),
                   method='Nelder-Mead', options={'maxiter': 100},
                   callback=Callback_CoCo_CaseStudy, bounds=bounds, tol=0.001)


    print('end')