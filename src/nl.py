import numpy as np
import pandas as pd
import math
import time
from scipy import stats, optimize

import matplotlib.pyplot as plt
from IPython.display import display, clear_output

#=================================================================
# class for Nested Logit Model
class NL:
    def __init__(self, data, factor, nest_name, choice="choice", alt="alt", vis_process=False):
        self.data = data
        self.factor = factor
        self.nest_name = nest_name
        self.choice = choice
        self.alt = alt
        self.vis_process = vis_process
        
        # for visualize estimating process
        self.count = 0
        self.f = []
        plt.ion()
        self.fig = plt.figure()
    
    #-------------------------------------------------------------------
    # Log-likelihood
    def LL(self, beta):
        self.data["V"] = list((beta[:-1] * self.data[self.factor]).sum(axis=1))
        logsum = self.data.groupby(["chid", self.nest_name]).apply(lambda d: np.log(sum(np.exp(d.V))))

        LL = 0
        for c in self.data.chid.unique():
            nest_, alt_ = self.data[(self.data["choice"] == 1) & (self.data.chid == c)][[self.nest_name, "alt"]].values[0]
            Pm = np.exp(beta[-1] * logsum[c][nest_]) / sum(np.exp(beta[-1] * logsum[c]))
            Pr = np.exp(self.data[(self.data.chid == c) & (self.data[self.alt] == alt_)].V.values) / sum(np.exp(self.data[(self.data.chid == c) & (self.data[self.nest_name] == nest_)].V))
            LL += np.log(Pm * Pr)[0]

        return -LL
    
    #-------------------------------------------------------------------
    # visualize estimating process
    def cbf(self, x):
        self.count += 1
        clear_output(wait = True)
#         plt.cla()
        self.f.append(-self.LL(x))
        plt.scatter(range(self.count), self.f, color='#1f77b4')
        plt.xlim([0, 100])
        plt.ylim([-2000, 0])
        plt.pause(0.2)
        plt.draw()
        self.fig.clear()

        print("iter:", self.count)
        print("f:", -self.LL(x))
        print("param:", x)    
    
    #-------------------------------------------------------------------
    # estimates model and print results
    def run(self, beta, bounds):
        print("number of choice")
        print(self.data[self.data[self.choice] == True].groupby(self.alt).count()[self.choice])
        print("")
        print("estimating ...")

        if self.vis_process == True:
            result = optimize.minimize(self.LL, beta, method="L-BFGS-B", bounds=bounds, callback=self.cbf)
        else:
            result = optimize.minimize(self.LL, beta, method="L-BFGS-B", bounds=bounds)
        
        beta_opt = result.x

        # print("")
        # print("estimated parameter")
        # print(self.factor, "η")
        # np.set_printoptions(precision=3, suppress=True)
        # print(beta_opt)

        LL = -self.LL(beta_opt)
        LL0 = -self.LL(np.zeros(len(beta)))

        # statistics
        print("")
        print("calculate statistics...")
    
        hess = numdifftools.core.Hessian(function)(beta_opt)
        stdev = np.sqrt(np.diagonal(np.linalg.inv(hess)))

        t = beta_opt / stdev
        p = np.zeros(len(beta))
        signif_codes = []
        N = len(self.data.chid.unique())
        for i in range(len(beta)):
            alpha = stats.t.cdf(abs(t[i]), df=N-1)
            p[i] = (1-alpha) * 2
            if p[i] < 0.001:
                signif_codes.append("***")
            elif p[i] < 0.01:
                signif_codes.append("**")
            elif p[i] < 0.05:
                signif_codes.append("*")
            elif p[i] < 0.1:
                signif_codes.append(".")
            else:
                signif_codes.append("")

        beta_opt_3 = [round(beta_opt[n], 3) for n in range(len(beta_opt))]
        stdev_3 = [round(stdev[n], 3) for n in range(len(stdev))]
        t_3 = [round(t[n], 3) for n in range(len(t))]
        p_3 = [round(p[n], 3) for n in range(len(p))]

        summary = pd.DataFrame({"Coefficients": self.factor.append("eta"),
                     "Estimate": beta_opt_3,
                     "Std. Error": stdev_3,
                     "t-value": t_3,
                     "p-value": p_3,
                     "": signif_codes})

        print("-"*80)
        print("Estimated parameters")
        print(summary)
        print()
        print("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        
        print("-"*80)
        print("Evaluation index")
        print("log-Likelihood:", LL)
        print("McFadden R2:", 1-(LL/LL0))
        print("Adjusted McFadden R2:", 1-((LL-len(param))/LL0)) 

########################################################################
if __name__ == '__main__':
    data = pd.read_csv("../data/hyodo.csv", index_col=0)
    data["train"] = (data["alt"] == 1).astype("int8")
    data["bus"] = (data["alt"] == 2).astype("int8")
    # train and bus are nested as public transportation
    data["public"] = ((data["alt"] == 1) | (data["alt"] == 2)).astype("int8")
    data = data.loc[:, ["chid", "alt", "public", "choice", "time", "fare", "age", "number_own_cars", "train", "bus"]]

    factor = ["time", "fare", "age", "number_own_cars", "train", "bus"]
    nl = NL(data=data, factor=factor, nest_name="public", vis_process=True)
    beta0 = np.ones(len(factor)+1) # add a parameter of eta that logsum paramter
    bounds = ((-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (0, np.inf))

    nl.run(beta0, bounds)