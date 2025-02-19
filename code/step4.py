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
                         sigma=None, l2=None, muV=None, sigmaV=None, e=None, St=None, cet = None):
    w, p, Jbar, w_bar, gamma = param
    model_price = equityconvert_coco(r, K, T, t0,
                                     l1, a, b,
                                     c, e, p, q,
                                     Jbar, M, w, w_bar,
                                     k1, xi1, k2, xi2,
                                     l2, l32, muV, sigmaV, sigma,
                                     ignore_gov, St, gamma, cet)

    loss = np.sqrt(np.mean((model_price - coco_price) ** 2))
    return loss

def Callback_CoCo_CaseStudy(Xi):
    global Nfeval, r, q, K, T, t0, c, Jbar, M, coco_price, l1_a3, a3, b_a3, k1, xi1, k2, xi2, l32, ignore_gov, sigma_a3, l2_a3, muV_a3, sigmaV_a3, e_a3, St, cet_value
    func_values = optimize_convertcoco(Xi, r=r, q=q, K=K, T=T, t0=t0, c=c,  M=M, coco_price=coco_price,
                         l1=l1_a3, a=a3, b=b_a3,  # latent params
                         k1=k1, xi1=xi1, k2=k2, xi2=xi2, l32=l32, ignore_gov=ignore_gov,  # intervention params
                         sigma=sigma_a3, l2=l2_a3, muV=muV_a3, sigmaV=sigmaV_a3, e=e_a3, St = St, cet = cet_value)


    # print('{0: 4d}     w:{1:.4f}     p:{2: .4f}    loss:{3:.4f}'.format(
    #         Nfeval, Xi[0], Xi[1], func_values))
    print('{0: 4d}     w:{1:.4f}     p:{2: .4f}     Jbar:{3: .4f}    wbar:{4: .4f}    gamma:{5: .4f}    loss:{6:.4f}'.format(
                Nfeval, Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], func_values))
    Nfeval += 1

if __name__ == '__main__':
    # [Read stock parameters]
    # file_name = 'Lloyds_Data_13-23.xlsx'
    file_name = 'Credit_Suisse_Data_13-23.xlsx'
    # file_name = 'China_Construction_Bank_Data_13-23.xlsx'
    save_fig = True
    save_param = True

    # [added cet0 data]
    # file_name = 'China_Construction_Bank_Data_13-23.xlsx'
    date_range = ['1/1/2020', '3/17/2023']
    return_data, cet_data, B, J, B_date = load_data('../data', file_name, date_range=date_range)

    intervention_param = pd.read_csv('../param/Intervention0131_' + file_name.split('.')[0] + '.csv')
    (loss_a3, l1_a3, b_a3, a3, mu_a3, sigma_a3,
     l2_a3, muV_a3, sigmaV_a3, e_a3,
     k1, xi1, k2, xi2, l32) = list(intervention_param.values.squeeze())

    # [Read coco data]
    data = pd.read_excel('../data/Charlie1124/case4/data-case4.xlsx')
    data = data.set_index('Date')
    #date_flag = data.index.isin(pd.date_range(start='3/1/2023', end='3/17/2023'))
    date_flag = data.index.isin(pd.date_range(start=date_range[0], end=date_range[1]))
    data = data[date_flag]

    data = data.dropna(how='any')
    St = data['Stock'].values
    RET = data['return without dividend'] / 100

    r = data['r'].values / 100
    coco_price = data['CoCo'].values
    T,c = [7, 7.25/100/2]
    maturity = pd.to_datetime(data.index[0]) + pd.DateOffset(years=T)
    t0 = np.array(range(data.index.shape[0])) / 252

    cet_data['Date'] = pd.to_datetime(cet_data['Names Date'])
    cet_smoothed = cet_data.set_index('Date').resample('D').interpolate()
    cet_value = cet_smoothed.loc[cet_smoothed.index.isin(data.index), 'CET-1 ratio (phase-in)'].values/100



    K = 100
    M = 14
    ignore_gov = False
    q = 0# ToDo: check

    # w, p, Jbar, w_bar = param

    # loss, p, w, Jbar, wbar, gamma
    # 9.2835, 0.852, 0.05, 0.6717, 0.3896, 0.021
    p, w, Jbar, wbar, gamma =  [0.9943, 0.0002, 0.1967, 0.9998, 0.0015]
    model_price = equityconvert_coco(r, K, T, t0,
                                     l1_a3, a3, b_a3,
                                     c, e_a3, p, q,
                                     Jbar, M, w, wbar,
                                     k1, xi1, k2, xi2,
                                     l2_a3, l32, muV_a3, sigmaV_a3, sigma_a3,
                                     ignore_gov=ignore_gov, St=St, gamma = gamma,
                                     cet = cet_value
                                     )
# import matplotlib.pyplot as plt
# plt.plot(model_price, label = 'model')
# plt.plot(coco_price, label ='actual' )
# plt.legend()

    # w, p, Jbar, w_bar, gamma = param
    init, Nfeval = [[0.05, 0.8520, 0.3896, 0.3, 0.0210], 1]
    bounds = [(0, 1), (0.05, 0.995), (0, 10), (0.1, 1), (0, 1)]
    res = minimize(optimize_convertcoco, init, args=( r, q, K, T, t0, c, M, coco_price,  # data
                                                     l1_a3, a3, b_a3,  # latent params
                                                     k1, xi1, k2, xi2, l32, ignore_gov,  # intervention params
                                                     sigma_a3, l2_a3, muV_a3, sigmaV_a3, e_a3, St, cet_value),
                   method='Nelder-Mead', options={'maxiter': 100},
                   callback=Callback_CoCo_CaseStudy, bounds=bounds, tol=0.001)
    coco_param = pd.DataFrame([res.fun, *res.x],
                              columns=['loss', 'p', 'w', 'Jbar', 'wbar', 'gamma'])
    if save_param:
        coco_param.to_csv('../param/CoCo0117_case4.csv', index=False)




    print('end')