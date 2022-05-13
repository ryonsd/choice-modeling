import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy import optimize
import numdifftools

#=================================================================
# class for Multinomial Logit Model
class MNL:
    def __init__(self, data, factor, choice="choice", alt="alt"):
        self.data = data
        self.factor = factor
        self.choice = choice
        self.alt = alt
        
        self.n_alt = len(self.data[self.alt].unique())
        self.n_chid = int(len(self.data) / self.n_alt)
        self.n_factor = len(self.factor)
    
    #-------------------------------------------------------------------
    # Log-likelihood   
    def LL(self, beta):

        # params
        a = beta[0:self.n_alt-1]
        b = beta[self.n_alt-1:self.n_alt+self.n_factor]

        LL_ = 0
        for n in range(self.n_chid):
            term1 = 0
            term2 = 0
            for i in range(n * self.n_alt, (n+1) * self.n_alt):
                const = 0 if i%self.n_alt == 0 else a[i%self.n_alt-1]
                bx = sum(b[k]*self.data[x][i] for k, x in enumerate(self.factor)) + const
                term2 += np.exp(bx)
                if self.data[self.choice][i] is np.bool_(True):
                    term1 = bx
            LL_ += term1 - np.log(term2)

        return -LL_  
    
    #-------------------------------------------------------------------
    # Log-likelihood when parameters values are zero
    def LL0(self, beta):

        # params
        a = beta[0:self.n_alt-1]
        b = beta[self.n_alt-1:self.n_alt+self.n_factor]

        LL_ = 0
        for n in range(self.n_chid):
            term1 = 0
            term2 = 0
            for i in range(n * self.n_alt, (n+1) * self.n_alt):
                const = 0 if i%self.n_alt == 0 else a[i%self.n_alt-1]
                bx = const
                term2 += np.exp(bx)
                if self.data[self.choice][i] is np.bool_(True):
                    term1 = bx
            LL_ += term1 - np.log(term2)

        return -LL_
    
    #-------------------------------------------------------------------
    # list of coefficents names
    def coefficients(self):
        coeff = self.data[self.alt].unique()[1:self.n_alt].tolist() 
        for i in range(self.n_alt-1):
            coeff[i] = coeff[i]+":(intercept)"
        coeff = coeff + self.factor 
        return coeff
    
    #-------------------------------------------------------------------
    # estimates model and print results
    def run(self, beta):
        # about data
        print("number of choice")
        print(self.data[self.data[self.choice] == True].groupby(self.alt).count()[self.choice])
        
        # estimate
        print("")
        print("estimating...")
        result = optimize.minimize(self.LL, beta, method="L-BFGS-B")
        result0 = optimize.minimize(self.LL0, beta, method="L-BFGS-B")

        beta_opt = result.x
        hess = numdifftools.core.Hessian(mnl.LL)(beta_opt)
        stdev = np.sqrt(np.diagonal(np.linalg.inv(hess)))

        LL = -result.fun
        LL0 = -result0.fun
        
        # evaluate
        n_beta = self.n_alt - 1 + self.n_factor 
    
        t = np.zeros(n_beta)
        p = np.zeros(n_beta)
        signif_codes = []
        N = len(self.data.chid.unique())
        for i in range(n_beta):
            t[i] = beta_opt[i] / stdev[i]
            alpha = stats.t.cdf(abs(t[i]), df=N-1)
            p[i] = (1-alpha) * 2
            if p[i] < 0.001:
                signif_codes.append("***")
            elif p[i] < 0.01:
                signif_codes.append("**")
            elif p[i] < 0.05:
                signif_codes.append("*")
            elif p[i] < 0.1:
                signif_codes[i].append(".")
            else:
                signif_codes[i].append("")

        summary = pd.DataFrame({"Coefficients": mnl.coefficients(),
                     "Estimate": beta_opt,
                     "Std. Error": stdev,
                     "t-value": t,
                     "p-value": p,
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
        print("Adjusted McFadden R2:", 1-((LL-n_beta)/LL0))   

########################################################################
if __name__ == '__main__':
    # A data "Mode" from R package "mlogit" is used
    # Choose a transportation from "car", "carpool", "bus", and "rail" based on "cost" and "time".
    data_long = pd.read_csv("../data/mode_long.csv", index_col=0) 

    mnl = MNL(data=data_long, factor=["cost", "time"])

    num_params_estimated = 5
    beta0 = np.zeros(num_params_estimated)

    mnl.run(beta0)