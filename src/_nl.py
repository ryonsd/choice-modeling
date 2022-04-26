import numpy as np
import pandas as pd
import math
import time
from scipy import stats, optimize

import matplotlib.pyplot as plt
from IPython.display import display, clear_output
%matplotlib inline


data = pd.read_csv("./data_hyodo.csv", index_col=0)
data["鉄道"] = (data["alt"] == 1).astype("int8")
data["バス"] = (data["alt"] == 2).astype("int8")
data["mode_type"] = ((data["alt"] == 1) | (data["alt"] == 2)).astype("int8")
data = data.loc[:, ["chid", "mode_type", "alt", "選択機関", "総時間", "総費用", "年齢", "保有台数", "鉄道", "バス"]]

# 対数尤度関数
def function(param):
    data["V"] = list((param[:6] * data.iloc[:, 4:10]).sum(axis=1))
    logsum = data.groupby(["chid", "mode_type"]).apply(lambda d: np.log(sum(np.exp(d.V))))

    LL = 0
    for c in data.chid.unique():
        mode_, alt_ = data[(data["選択機関"] == 1) & (data.chid == c)][["mode_type", "alt"]].values[0]
        Pm = np.exp(param[6] * logsum[c][mode_]) / sum(np.exp(param[6] * logsum[c]))
        Pr = np.exp(data[(data.chid == c) & (data.alt == alt_)].V.values) / sum(np.exp(data[(data.chid == c) & (data.mode_type == mode_)].V))
        LL += np.log(Pm * Pr)[0]
        
    return -LL

# 結果のリアルタイム可視化 (Jupyter Notebook)
def cbf(x):
    global count, f
    count += 1
    clear_output(wait = True)
    plt.cla()
    f.append(-function(x))
    plt.scatter(range(count), f, color='#1f77b4')
    plt.xlim([0, 100])
    plt.ylim([-2000, 0])
    display(fig) 

    print("iter:", count)
    print("f:", -function(x))
    print("param:", x)    


# 初期パラメータ
param = [1, 1, 1, 1, 1, 1, 1]
# パラメータの定義域
bounds = ((-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (0, np.inf))


count = 0
f = []
plt.ion()
fig = plt.figure()

# 最適化実行
result = optimize.minimize(function, param, method="L-BFGS-B", bounds=bounds, callback=cbf)
print(result)

# 結果
# 計算に１時間ほどかかりますが結果は合ってます
#
# fun: 1283.673505153972
# hess_inv: <7x7 LbfgsInvHessProduct with dtype=float64>
#      jac: array([-0.00052296, -0.00013642, -0.00143245, -0.00118234, -0.00025011,
#       -0.00097771, -0.00029559])
#  message: b'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH'
#     nfev: 416
#      nit: 42
#   status: 0
#  success: True
#        x: array([-0.89786429,  0.01277061,  1.17845223,  0.50026018,  0.03898127,
#       -0.43393897,  4.87268973])

