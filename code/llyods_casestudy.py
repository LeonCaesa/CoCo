import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from utils import *


optimize_stock = False
save_param = False



def optimize_stock_casestudy(param, a = None, d = 1/252, x = None):
    mu, sigma, l2, muV, SigmaV, e, l1, b = param
    log_density = np.log(Density_stock(l1, a, b, mu, sigma, l2, muV, SigmaV, e, x, d))
    inf_flag = np.isinf(log_density)
    if sum(inf_flag) != 0:
        warnings.warn("Warning: log_density contains invalid value given the current parameters, masking it with the minimal of valid values")
        log_density[inf_flag] = min(log_density[~inf_flag])
    return -np.sum(log_density)
def Callback_Stock_CaseStudy(Xi):
    global Nfeval, RET, a
    print('{0: 4d}     mu:{1:.4f}     sigma:{2: .4f}    l2:{3:.4f}    muV:{4:.4f}    sigmaV:{5:.4f}    e:{6:.4f}    l1:{7:4f}    b:{8:4f}    loss:{9:.4f}'.format(Nfeval,
        Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], Xi[5],
        Xi[6], Xi[7],
        optimize_stock_casestudy(Xi, a =a, x= RET,  d= 1/252)))
    Nfeval += 1


def Callback_CoCo_CaseStudy(Xi):
    global Nfeval, wbar, r, q, K, T, t0, c, Jbar, M, coco_price, l1, a, b, k1, xi1, k2, xi2, l32, ignore_gov, sigma, l2, muV, sigmaV, eta
    func_values = optimize_convertcoco(Xi, w_bar=wbar, r=r, q=q, K=K, T=T, t0=t0, c=c, Jbar=Jbar, M=M, coco_price=coco_price,
                         l1=l1, a=a, b=b,  # latent params
                         k1=k1, xi1=xi1, k2=k2, xi2=xi2, l32=l32, ignore_gov=ignore_gov,  # intervention params
                         sigma=sigma, l2=l2, muV=muV, sigmaV=sigmaV, e=eta)

    # print('{0: 4d}     w:{1:.4f}     p:{2: .4f}    loss:{3:.4f}'.format(
    #         Nfeval, Xi[0], Xi[1], func_values))
    print('{0: 4d}     w:{1:.4f}     p:{2: .4f}     Jbar:{3: .4f}    loss:{4:.4f}'.format(
                Nfeval, Xi[0], Xi[1], Xi[2], func_values))            
    Nfeval += 1




file_name = 'Lloyd (09-11).xlsx'
data = pd.read_excel('../data/Charlie0907/Lloyd (09-11) coc and stock.xlsx')
data = data.set_index('Dates')
data = data.dropna(how = 'any')
RET = data['return without dividend (in percentage)']/100

r = data['interest rate r'].values/100
coco_price = data['CoCo Price'].values
maturity = pd.to_datetime('2019-11-23')
T = 10
#t0 = T - np.array((maturity - data.index).days.astype(int)/365) #ToDo: confirm 365 makes more sense
t0 = np.array(range(data.index.shape[0]))/252

if optimize_stock:
    bounds = [(-1, 1), (0, 1), (0, 100), (-1, 1), (0, 1), (0, None), (0, 100), (0, 150)]

    sorted_list = []
    for a in range(1, 5):
        #mu, sigma, l2, muV, SigmaV, e, l1, b
        init, Nfeval = [[0.3225, 0.2305, 32.95, -0.0017, 0.0407, 0.4365, 21.64, 22.49], 1]
        stock_res = minimize(optimize_stock_casestudy, init, args=(a, 1/252, RET), method='Nelder-Mead',
                        options={'maxiter': 100}, bounds=bounds, callback=Callback_Stock_CaseStudy, tol=0.001)

        sorted_list.append([stock_res.fun, *stock_res.x, a])
    sorted_list = sorted(sorted_list)
    stock_param = pd.DataFrame(sorted_list, columns= ['loss', 'mu', 'sigma', 'l2', 'muV', 'sigmaV', 'eta', 'l1', 'b', 'a'])

if save_param:
    stock_param.to_csv('../param/J_' + file_name.split('.')[0] + 'casestudy.csv', index=False)
else:
    stock_param = pd.read_csv('../param/J_' + file_name.split('.')[0] + 'casestudy.csv')
loss, mu, sigma, l2, muV, sigmaV, eta, l1, b, a = stock_param.iloc[0].values



RET_grids = np.linspace(-0.2, 0.2, 100)
Eval_Density = Density_stock(l1, a, b, mu, sigma, l2, muV, sigmaV, eta, RET_grids)
Data_DensityStock = KDE_estimate(RET.values, RET_grids)
plt.plot(RET_grids, Eval_Density, linestyle='--', label='Fitted')
plt.plot(RET_grids, Data_DensityStock, label='Kernel')
plt.legend()
plt.show()
#plt.savefig('../figure/Sdensity_' + file_name.split('_')[0] + '.jpg', dpi=600)

def optimize_convertcoco(param, w_bar = None, r=None, q=None, K=None, T=None,t0 = None, c=None, Jbar=None, M=None, coco_price=None,  # data
                         l1=None, a=None, b=None,  # latent params
                         k1=None, xi1=None, k2=None, xi2=None, l32=None, ignore_gov=None,  # intervention params
                         sigma = None, l2=None, muV=None, sigmaV=None, e=None):
    #w, p = param
    w, p, Jbar = param
    model_price = equityconvert_coco(r, K, T, t0,
                                     l1, a, b,
                                     c, e, p, q,
                                     Jbar, M, w, w_bar,
                                     k1, xi1, k2, xi2,
                                     l2, l32, muV, sigmaV, sigma,
                                     ignore_gov=ignore_gov)

    loss = np.abs(model_price - coco_price) / coco_price
    return np.mean(loss)


W = 0.05 # for llyods
C0 = 6.3/100 # for llyods
wbar = 1 # for llyods
q = 0 # for llyods

#K = 77.5158
K = 100
c = 0.15/2
M = 20
Jbar = np.tan( np.pi * (1 - 2 * W)/2) + 1/ np.tan(np.pi * (1-C0))

#Jbar_range = np.tan( np.pi * (1 - 2 * W)/2) + 1/ np.tan(np.pi * (1-C0))


# intervention params
k1, xi1, k2, xi2, l32, ignore_gov = [None, None, None, None, None, True]

p = 0.49
w = 1.0
Jbar = 0.1
model_price = equityconvert_coco(r, K, T, t0,
                                 l1, a, b,
                                 c, eta, p, q,
                                 Jbar, M, w, wbar,
                                 k1, xi1, k2, xi2,
                                 l2, l32, muV, sigmaV, sigma,
                                 ignore_gov=ignore_gov)

plt.plot(model_price, '+', label = 'estimated')
plt.plot(coco_price, 'o', label = 'actual')
plt.legend()
plt.show()

# init, Nfeval = [[1, 0.49], 1]
# bounds = [(0, 1), (0, 1)]
init, Nfeval = [[1, 0.49, Jbar], 1]
bounds = [(0, 1), (0, 1), (0.01, 4.9)]

res = minimize(optimize_convertcoco, init, args=(wbar, r, q, K, T, t0, c, Jbar, M, coco_price,  # data
                         l1, a, b,  # latent params
                         k1, xi1, k2, xi2, l32, ignore_gov,  # intervention params
                         sigma, l2, muV, sigmaV, eta),
               method='Nelder-Mead', options={'maxiter': 100},
               callback=Callback_CoCo_CaseStudy, bounds=bounds, tol=0.001)
print('end')