import os.path
import random
from utils import *
import scipy as sp
import scipy.interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pickle

data_size = 51
# ToDo: confirm it is okay to fix a = 3
l1_grids = np.array([random.random() * 100 for p in range(0, data_size)])
b_grids = np.array([random.random() * 150 for p in range(0, data_size)])
t_grids = np.array([random.random() for p in range(0, data_size)])
x_grids = np.linspace(1.1, 1.25, data_size)


func_values = [SupPr(l1_grids[i], 3, b_grids[i], t_grids[i], x_grids[i]) for i in range(data_size)]
#L1_grids, B_grids, T_grids, X_grids = np.meshgrid(l1_grids, b_grids, [0.25] * data_size, x_grids)

if os.path.exists('./SupPr.pkl'):
    with open('./SupPr.pkl', 'rb') as inp:
        spline = pickle.load(inp)
else:
    spline = sp.interpolate.Rbf(l1_grids, b_grids, t_grids, x_grids, func_values, function='thin_plate',smooth=5, episilon=5)
    with open('./SupPr.pkl', 'wb') as file:
        pickle.dump(spline ,file)
#Z = spline(L1_grids, B_grids, T_grids, X_grids)
Z = spline(l1_grids, b_grids, t_grids, x_grids)
func_approx = [Z[i, i ,i , i] for i in range(data_size)]

fig = plt.figure(figsize=(10,6))
plt.plot(func_values, 'o', label = 'real', fillstyle='none')
plt.plot(func_approx, '+', label = 'approx')
plt.legend()
plt.show()

