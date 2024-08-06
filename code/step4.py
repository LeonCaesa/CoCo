from scipy.optimize import minimize
import matplotlib.pyplot as plt
from utils import *
import seaborn as sns
import pandas as pd
import time



if __name__ == '__main__':
    # file_name = 'Lloyds_Data_13-23.xlsx'
    file_name = 'Credit_Suisse_Data_13-23.xlsx'
    # file_name = 'China_Construction_Bank_Data_13-23.xlsx'
    save_fig = False
    save_param = False

    intervention_param = pd.read_csv('../param/Intervention_' + file_name.split('.')[0] + '.csv')
    (loss_a3, l1_a3, b_a3, a3, mu_a3, sigma_a3,
     l2_a3, muV_a3, sigmaV_a3, e_a3,
     k1, xi1, k2, xi2, l32) = list(intervention_param.values.squeeze())

    def optimize_convertcoco(param, r= None, K = None, T = None, c = None, Jbar = None, M = None, coco_price = None, # data
                        l1=None, a=None, b=None, # latent params
                        k1 = None, xi1 = None, k2= None, xi2 = None, l32 = None, ignore_gov = None,# intervention params
                        l2 = None, muV = None, sigmaV = None, e = None):
        w, w_bar, p = param

        model_price = equityconvert_coco(r, K, T,
                                         l1, a, b,
                                         c, e, p, q,
                                         Jbar, M, w, w_bar,
                                         k1, xi1, k2, xi2,
                                         l2, l32, muV, sigmaV,
                                         ignore_gov = ignore_gov)

        loss = np.abs(model_price - coco_price)/ coco_price
        return np.mean(loss)


    init, Nfeval = [[0.3, 0.3, 0.3], 1]
    bounds = [(0, 1), (0, 1), (0, 1)]
    res = minimize(optimize_convertcoco, init, args=(l2_a3, muV_a3, sigmaV_a3, mat_values, 0, cds_values),


                   method='Nelder-Mead', options={'maxiter': 100},
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