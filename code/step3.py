from scipy.optimize import minimize
import matplotlib.pyplot as plt
from utils import *
import seaborn as sns
import pandas as pd


def Callback_CDS(Xi):
    global Nfeval, l2_a3, muV_a3, sigmaV_a3, cds_values#, t_cds, T_cds
    print(
        '{0: 4d}     k1:{1:.4f}     xi1:{2: .4f}    k2:{3:.4f}    xi2:{4:.4f}     l32:{5:.4f}    loss:{6:.4f}'.format(
            Nfeval,
            Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], optimize_cds(Xi, l2_a3, muV_a3, sigmaV_a3, 5, 0, cds_values
                         )))
    Nfeval += 1

if __name__ == '__main__':
    # file_name = 'Lloyds_Data_13-23.xlsx'
    file_name = 'Credit_Suisse_Data_13-23.xlsx'
    # file_name = 'China_Construction_Bank_Data_13-23.xlsx'
    save_fig = False
    save_param = False


    stock_param = pd.read_csv('../param/Stock_' + file_name.split('.')[0] + '.csv')
    loss_a3, l1_a3, b_a3, a3, mu_a3, sigma_a3, l2_a3, muV_a3, sigmaV_a3, e_a3 = list(stock_param.values.squeeze())


    # ToDo: confirm what is W and C0, and the range of Jbar
    W = 0.1
    C0 = 0.11
    Jbar = np.tan(np.pi * (1 - 2 * W) / 2) + 1 / np.tan(np.pi * (1 - C0))

    # [loading data]
    oneyear_cds = pd.read_csv('../data/CSGN1YEUAM=R Overview.csv').set_index('Date')  # ['Price']
    oneyear_cds['T'] = 1
    fiveyear_cds = pd.read_csv('../data/CSGN5YEUAM=R Overview.csv').set_index('Date')  # ['Price']
    fiveyear_cds['T'] = 5
    tenyear_cds = pd.read_csv('../data/CSGN10YEUAM=R Overview.csv').set_index('Date')  # ['Price']
    tenyear_cds['T'] = 10

    cds_data = pd.concat([oneyear_cds, tenyear_cds, fiveyear_cds])[['Price', 'T']]
    cds_data['Price'] = cds_data['Price'].apply(lambda x: x.replace(',', '')).astype(float) / 100
    cds_data.index = pd.to_datetime(cds_data.index)
    cds_data = cds_data.sort_index()
    date_flag = cds_data.index.isin(pd.date_range(start='3/1/2023', end='3/17/2023'))
    cds_values = cds_data['Price'].values[date_flag]
    date_values = cds_data.index[date_flag]
    mat_values = cds_data['T'].values[date_flag]


    #[plotting of CDS curve]
    plt.figure(figsize=(4, 4))
    cds_data['T'] = cds_data['T'].astype(str)
    sns.scatterplot(data=cds_data[date_flag].sort_values('T'), x='Date', y='Price', hue='T', style='T')
    plt.ylabel('CDS Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_fig:
        plt.savefig('../figure/CDS_' + file_name.split('_')[0] + '.jpg', dpi=600)
    plt.show()

    # [optimization part]
    init, Nfeval = [[1.5, 0.5, 1.5, 0.5, 0.5], 1]
    bounds = [(0, 5), (0.01, 2), (1, 10), (0.01, 2), (0.1, 10)]
    intervention_res = minimize(optimize_cds, init, args = (l2_a3, muV_a3, sigmaV_a3, mat_values, 0, cds_values), method='Nelder-Mead',
                   options = {'maxiter': 100},
                   callback = Callback_CDS, bounds = bounds, tol = 0.001)
    k1, xi1, k2, xi2, l32 = intervention_res.x

    if save_param:
        intervention_param = pd.DataFrame([intervention_res.fun,
                                           l1_a3, b_a3, a3,
                                           mu_a3, sigma_a3,l2_a3, muV_a3, sigmaV_a3, e_a3,
                                           k1, xi1, k2, xi2, l32],
                                   index=['loss', 'l1', 'b', 'a',
                                          'mu', 'sigma', 'l2', 'muV', 'sigmaV', 'e',
                                          'k1', 'xi1', 'k2', 'xi2', 'l32']).T

        intervention_param.to_csv('../param/Intervention_' + file_name.split('.')[0] + '.csv', index=False)

    loss_a3, l1_a3, b_a3, a3, mu_a3, sigma_a3, l2_a3, muV_a3, sigmaV_a3, e_a3, k1, xi1, k2, xi2, l32 = list(intervention_param.values.squeeze())