import os.path
import random
from utils import *
import scipy as sp
import scipy.interpolate
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import pickle
import pandas as pd
import time

random.seed(3)
np.random.seed(3)
data_size = 10001
# ToDo: confirm it is okay to fix a = 3
l1_grids = np.array([random.random() * 100 for p in range(0, data_size)])
b_grids = np.array([random.random() * 150 for p in range(0, data_size)])
t_grids = np.array([random.random() * 10 for p in range(0, data_size)])
#x_grids = np.linspace(1.1, 1.25, data_size)

B0_grids = np.random.uniform(0.05, 0.2, data_size)
x_grids = np.tan( np.pi * (1 - 2 * 0.05)/2) + 1/ np.tan(np.pi * (1-B0_grids))# + 13
a_grids = np.random.choice(5, data_size) + 1

# Pr_table = pd.DataFrame([l1_grids, b_grids, t_grids, x_grids, a_grids]).T
# Pr_table.columns = ['lambda1', 'beta', 't', 'jbar', 'a']
# print('Min of simulated parameters are')
# print(Pr_table.min())
# print('Max of simulated parameters are')
# print(Pr_table.max())

# import time
# i = 0
# start = time.time()
# test = SupPr(l1_grids[i], a_grids[i], b_grids[i], t_grids[i], x_grids[i])
# end = time.time()
# print(end-start)

if os.path.exists('./spline_approx.csv'):
    Pr_table = pd.read_csv('./spline_approx.csv')
    func_values = Pr_table.iloc[:, -1].values
    l1_grids = Pr_table.iloc[:, 0].values
    b_grids = Pr_table.iloc[:, 1].values
    t_grids = Pr_table.iloc[:, 2].values
    x_grids = Pr_table.iloc[:, 3].values
    a_grids = Pr_table.iloc[:, 4].values
else:
    time_list = []
    func_list = []
    for i in range(data_size):
        start = time.time()
        func_values = SupPr(l1_grids[i], a_grids[i], b_grids[i], t_grids[i], x_grids[i])
        end = time.time()
        time_list.append(end - start)
        func_list.append(func_values)

        Pr_table = pd.DataFrame([l1_grids[:(i+1)], b_grids[:(i+1)], t_grids[:(i+1)], x_grids[:(i+1)], a_grids[:(i+1)], func_list[:(i+1)], time_list[:(i+1)]]).T
        Pr_table.columns = ['l1', 'b', 't', 'x', 'a', 'func', 'time']
        Pr_table.to_csv('./spline_approx_loop.csv', index = False)


#L1_grids, B_grids, T_grids, X_grids = np.meshgrid(l1_grids, b_grids, [0.25] * data_size, x_grids)

if os.path.exists('./SupPr.pkl'):
    with open('./SupPr.pkl', 'rb') as inp:
        spline = pickle.load(inp)
else:
    na_flag = ~np.isnan(func_values)
    spline = sp.interpolate.Rbf(l1_grids[na_flag], a_grids[na_flag],
                                b_grids[na_flag], t_grids[na_flag],
                                x_grids[na_flag], func_values[na_flag],
                                function='multiquadric')
    with open('./SupPr.pkl', 'wb') as file:
        pickle.dump(spline ,file)
#Z = spline(L1_grids, B_grids, T_grids, X_grids)
func_approx = spline(l1_grids, a_grids, b_grids, t_grids, x_grids)
#func_approx = [Z[i, i ,i , i] for i in range(data_size)]

# [time plot]
import time
SupPr_time = {'actual':[],
              'approx':[]}
eval_index = np.random.randint(0, data_size, 100)

for i in eval_index:
    start = time.time()
    SupPr_i = SupPr(l1_grids[i], int(a_grids[i]), b_grids[i], t_grids[i], x_grids[i])
    end = time.time()
    SupPr_time['actual'].append(end - start)


    start_spline = time.time()
    SupPr_approxi = spline(l1_grids[i], a_grids[i], b_grids[i], t_grids[i], x_grids[i])
    end_spline = time.time()
    SupPr_time['approx'].append(end_spline - start_spline)

fig, ax = plt.subplots()
ax.boxplot(SupPr_time.values())
plt.ylabel('Eval time (s)')
ax.set_xticklabels(SupPr_time.keys())
plt.savefig('../figure/approx_time.png', dpi=300)

avg_times = np.mean(np.array(SupPr_time['actual']))/ np.mean(np.array(SupPr_time['approx']))

# [approx plot]
param_names = [r'$\lambda_1$', r'$\alpha$', r'$\beta$', 't', 'x']
param_list = [l1_grids, a_grids, b_grids, t_grids, x_grids]

for i in range(len(param_names)):
    fig = plt.figure(figsize=(6,6))
    plt.plot(param_list[i], func_values, 'o', label = 'actual', fillstyle='none', alpha=0.8)
    plt.plot(param_list[i], func_approx, '+', label = 'approx', alpha=0.8)
    plt.xlabel(param_names[i])
    plt.ylabel(r'$P(\lambda_1, \alpha, \beta, t, x)$')
    plt.legend()
    fig.tight_layout()
    plt.savefig('../figure/spline_' + param_names[i] + '.png', dpi=300)


#plt.show()

