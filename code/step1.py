from scipy.optimize import minimize
import matplotlib.pyplot as plt
from utils import *

def Callback_DensityJ(Xi):
    global Nfeval, H, a
    print('{0: 4d}     {1:.4f}     {2: .4f}    {3:.4f} '.format(Nfeval, Xi[0], Xi[1], optimize_DensityJ(Xi, a=a, H=H)))
    Nfeval += 1



if __name__ == '__main__':
    # file_name = 'Lloyds_Data_13-23.xlsx'
    file_name = 'Credit_Suisse_Data_13-23.xlsx'
    # file_name = 'China_Construction_Bank_Data_13-23.xlsx'
    save_fig = False
    save_param = False

    return_data, cet_data, B, J, B_date = load_data('../data', file_name)
    Diff_J = np.diff(J)
    H = Diff_J - min(Diff_J) + 0.01


    J_grids = np.linspace(-3, 3, 200).reshape(-1, 1)
    PJ_grids = KDE_estimate(Diff_J, J_grids)


    # [Plotting]
    fig = plt.figure(figsize=(6, 12))
    ax1 = fig.add_subplot(311)
    ax1.plot(J_grids, PJ_grids, label='Diff_J Density')
    plt.legend()

    ax2 = fig.add_subplot(312)
    ax2.plot(B, label='B', color='red')
    plt.legend()

    ax3 = fig.add_subplot(313)
    ax3.plot(H, label='H', color='blue')
    plt.legend()
    plt.show()

    # [1. Latent J Optimization]
    l, b, T = [1, 1, 1 / 4]
    Test_RelaxJ = Density_RelaxJ(l, b, T, H)
    Test_DensityJ = Density_J(l, 1, b, T, H)
    bounds = [(0, 100), (0, 150)]
    init, Nfeval = [[0.1, 0.1], 1]
    sorted_list = []
    for a in range(1, 6):
        Nfeval = 1
        print('==========optimization for a = %d===========' % a)
        res = minimize(optimize_DensityJ, init, args=(a, T, H), method='Nelder-Mead', options={'maxiter': 100},
                       bounds=bounds,
                       callback=Callback_DensityJ, tol=0.001)
        print('lambda: %.4f, beta %.4f' % (res.x[0], res.x[1]))
        sorted_list.append([res.fun, *res.x, a])
        if a == 1:
            l1_a1, b_a1 = res.x
    sorted_list = sorted(sorted_list)
    param_table = pd.DataFrame(sorted_list, columns=['loss', 'l1', 'b', 'a'])
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
        plt.savefig('../figure/Hdensity_' + file_name.split('_')[0] + '.jpg', dpi=600)
    plt.show()


    if save_param:
        param_table.to_csv('../param/J_' + file_name.split('.')[0] + '.csv', index = False)