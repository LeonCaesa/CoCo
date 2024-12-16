import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from utils import *


def Callback_Stock(Xi):
    global Nfeval, RET, l1_a3, a3, b_a3
    print('{0: 4d}     mu:{1:.4f}     sigma:{2: .4f}    l2:{3:.4f}    muV:{4:.4f}    sigmaV:{5:.4f}    e:{6:.4f}    loss:{7:.4f}'.format(Nfeval,
        Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], Xi[5],
        optimize_stock(Xi, x= RET,  l1= l1_a3, a = a3, b = b_a3, d= 1/252)))
    Nfeval += 1

if __name__ == '__main__':
    # file_name = 'Lloyds_Data_13-23.xlsx'
    file_name = 'Credit_Suisse_Data_13-23.xlsx'
    # file_name = 'China_Construction_Bank_Data_13-23.xlsx'
    #date_range = ['1/1/2020', '3/17/2023']
    date_range = None
    return_data, cet_data, B, J, B_date = load_data('../data', file_name, date_range=date_range)

    save_fig = True
    save_param = True
    if file_name == 'Credit_Suisse_Data_13-23.xlsx':
        RET_data = pd.read_excel(os.path.join('../data', file_name), sheet_name=0)
        date_range = ['1/1/2020', '3/17/2023']
        date_flag = RET_data['Names Date'].isin(pd.date_range(start = date_range[0], end = date_range[1]))
        RET_data = RET_data[date_flag]
        S_name = 'Price or Bid/Ask Average'
        RET = RET_data['Log-returns (without Dividends)'].values[:2570]
    else:
        RET_data = pd.read_excel(os.path.join('../data', file_name), sheet_name=0)
        S_name = 'Price or Bid/Ask Average'
        RET = RET_data['Log-returns without dividends'].values


    # [Plotting --- S figure]
    S_data = RET_data[[S_name, 'Names Date']]
    S_data.columns = ['S', 'Date']
    S_data = S_data.set_index('Date')
    S_data.plot(label='S')
    if save_fig:
        plt.savefig('../figure/S1216_' + file_name.split('_')[0] + '.jpg', dpi=600)
    plt.show()

    # [Plotting --- B figure]
    plt.plot(pd.to_datetime(B_date), B, label='B')
    plt.legend()
    if save_fig:
        plt.savefig('../figure/B1216_' + file_name.split('_')[0] + '.jpg', dpi=600)
    plt.show()
    # [load parameters from step1]
    param_table = pd.read_csv('../param/J1216_' + file_name.split('.')[0] + '.csv')
    loss, l1_a3, b_a3, a3 = param_table.iloc[0]


    # [Optimization part]
    bounds = [(-1, 1), (0, 1), (0, None), (-1, 1), (0, 1), (0, None)]
    init, Nfeval = [[0.465593, 0.224503, 26.0084, -0.00140472, 0.0514438, 0.539438], 1]
    stock_res = minimize(optimize_stock, init, args=(l1_a3, a3, b_a3, 1/252, RET), method='Nelder-Mead',
                        options={'maxiter': 100}, bounds=bounds, callback = Callback_Stock, tol=0.001)

    mu_a3, sigma_a3, l2_a3, muV_a3, sigmaV_a3, e_a3 = stock_res.x
    stock_param = pd.DataFrame([stock_res.fun, l1_a3, b_a3, a3, mu_a3, sigma_a3, l2_a3, muV_a3, sigmaV_a3, e_a3], index = ['loss', 'l1', 'b', 'a', 'mu', 'sigma', 'l2', 'muV', 'sigmaV', 'e']).T

    if save_param:
        stock_param.to_csv('../param/Stock1216_' + file_name.split('.')[0] + '.csv', index = False)
    loss_a3, l1_a3, b_a3, a3, mu_a3, sigma_a3, l2_a3, muV_a3, sigmaV_a3, e_a3 = list(stock_param.values.squeeze())

    # [Density Plot]
    RET_grids = np.linspace(-0.15, 0.15, 100)
    Eval_Density_a3 = Density_stock(l1_a3, a3, b_a3, mu_a3, sigma_a3, l2_a3, muV_a3, sigmaV_a3, e_a3, RET_grids)
    Data_DensityStock = KDE_estimate(RET, RET_grids)

    plt.plot(RET_grids, Data_DensityStock, label='Kernel')
    plt.plot(RET_grids, Eval_Density_a3, linestyle='--', label='Fitted')
    plt.legend()
    if save_fig:
        plt.savefig('../figure/Sdensity1216_' + file_name.split('_')[0] + '.jpg', dpi=600)
    plt.show()

    print('Done')