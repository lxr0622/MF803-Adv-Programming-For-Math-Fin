import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt  
import seaborn as sns
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
from sklearn.linear_model import LinearRegression

### Question 1 ###

## (a) ##
# read data as dataframe
ff_df = pd.read_csv("F-F_Research_Data_Factors_daily.CSV", skiprows = 4)
ff_df = ff_df.drop([(ff_df.shape[0] - 1)]) # delete last row
ff_df = ff_df.set_index(ff_df.columns[0]) # make date as index
ff_df.index = pd.to_datetime(ff_df.index) # set datetime formate for date
ff_df = ff_df.loc[ff_df.index.year >= 2010] # choose data from 2010,consistent wtih ETF data
# check missing value of data
print('Number of missing value for Fama-French factors data:')
print(ff_df.isnull().sum(axis=0))

## (b) ##
ff_covar = ff_df.iloc[:,0:3].cov()
ff_corr = ff_df.iloc[:,0:3].corr()
# plot covar matrix
sns.set(font_scale = 1)
plt.figure()
ax1 = plt.axes()
sns.heatmap(ff_covar, xticklabels = ff_df.columns[0:3].tolist(),yticklabels = ff_df.columns[0:3].tolist(),annot=True,ax = ax1)
ax1.set_title('Q1(b):covariance matrix of daily return for FF factors')
plt.figure()
ax2 = plt.axes()
sns.heatmap(ff_corr, xticklabels = ff_df.columns[0:3].tolist(),yticklabels = ff_df.columns[0:3].tolist(),annot=True,ax = ax2)
ax2.set_title('Q1(b):correlation matrix of daily return for FF factors')

## (c) ##
roll_corr_list = [] # 90-day rolling correlation among factor returns
index_list = [[0,1],[1,2],[0,2]]
for i,j in index_list:
    roll_corr = []
    factor1 = ff_df.iloc[:,i]
    factor2 = ff_df.iloc[:,j]
    for k in range ((len(factor1) - 90)):
        roll_corr.append(np.corrcoef(factor1[k:k+90],factor2[k:k+90])[0,1])
    roll_corr_list.append(roll_corr)
# plot
plt.figure()
for i in range(len(roll_corr_list)):
    plt.plot(ff_df.index[90:len(ff_df.index)],roll_corr_list[i],label = ff_df.columns[i] + ' vs ' + ff_df.columns[j])
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("90-day rolling correlation")
    plt.title("Q1(c):90-day rolling correlation among each factor")
    
## (d) ##
stat0,p_value0 = sm.stats.diagnostic.kstest_normal(ff_df.iloc[:,0]) # Kolmogorov-Smirnov test for Mkt - Rf
stat1,p_value1 = sm.stats.diagnostic.kstest_normal(ff_df.iloc[:,1]) # Kolmogorov-Smirnov test for SMB
stat2,p_value2 = sm.stats.diagnostic.kstest_normal(ff_df.iloc[:,2]) # Kolmogorov-Smirnov test for HML
print("p value for normality test of Mkt - Rf", p_value0)
print("p value for normality test of SMB", p_value1)
print("p value for normality test of HML", p_value2)


## (e) ##
# df of ETF files
yf.pdr_override()
etf_list = ['XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY']
etf_df_list = []
for etf in etf_list:
    etf_df_list.append(pdr.get_data_yahoo(etf, start="2009-12-31", end="2018-07-31"))
# adj close for each ETFs
list_cls = []
for df in etf_df_list:
    list_cls.append(df['Adj Close'].values.tolist())
etf_df = pd.DataFrame(np.array(list_cls).T)
etf_df.columns = etf_list
etf_df = etf_df.apply(pd.to_numeric)
etf_df.index = etf_df_list[0].index
etf_df.index = pd.to_datetime(etf_df.index)
# daily return of ETF
ret_list = []
for etf_name in etf_list:
    etf = etf_df[etf_name].values.tolist()
    ret = []
    for i in range(1,len(etf)):
        ret.append((etf[i]-etf[i-1])/etf[i-1])
    ret_list.append(ret)

beta_list = [] # betas for entire historical data
roll_beta1_list = [] # beta1 for rolling 90-day data
roll_beta2_list = [] # beta2 for rolling 90-day data
roll_beta3_list = [] # beta3 for rolling 90-day data
res_list = [] # residual for each ETF
for i in range(len(ret_list)):
    reg = LinearRegression()
    coef = reg.fit(ff_df.iloc[:,0:3],ret_list[i]).coef_.tolist()
    beta_list.append(coef)
    y_pred = reg.predict(ff_df.iloc[:,0:3])
    res_list.append([a - b for a, b in zip(y_pred, ret_list[i])])
    roll_beta1 = []
    roll_beta2 = []
    roll_beta3 = []
    for j in range((len(ret_list[i]) - 90)):
        reg = LinearRegression()
        coef = reg.fit(ff_df.iloc[j:j+90,0:3],ret_list[i][j:j+90]).coef_.tolist()
        roll_beta1.append(coef[0])
        roll_beta2.append(coef[1])
        roll_beta3.append(coef[2])
    roll_beta1_list.append(roll_beta1)
    roll_beta2_list.append(roll_beta2)
    roll_beta3_list.append(roll_beta3)

print("Betas of entire historical data for each ETF")
for i in range(len(etf_list)):
    print(etf_list[i],": beta1 : %.5f, beta2 : %.5f, beta3 : %.5f" % (beta_list[i][0],beta_list[i][1],beta_list[i][2]))
    
# plot
color_list = plt.cm.rainbow(np.linspace(0, 1, 10))
plt.figure()
for i in range(len(etf_list)):
    plt.plot(ff_df.index[90:len(ff_df.index)],roll_beta1_list[i],label = etf_list[i],color = color_list[i])
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("90-day rolling beta1")
    plt.title("Q1(e):90-day rolling beta1 between each ETF and Fama-French factors")
plt.figure()
for i in range(len(etf_list)):
    plt.plot(ff_df.index[90:len(ff_df.index)],roll_beta2_list[i],label = etf_list[i],color = color_list[i])
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("90-day rolling beta2")
    plt.title("Q1(e):90-day rolling beta2 between each ETF and Fama-French factors")
plt.figure()
for i in range(len(etf_list)):
    plt.plot(ff_df.index[90:len(ff_df.index)],roll_beta3_list[i],label = etf_list[i],color = color_list[i])
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("90-day rolling beta3")
    plt.title("Q1(e):90-day rolling beta3 between each ETF and Fama-French factors")

## (f) ##
mean_res = []
var_res = []
for i in range(len(etf_list)):
    mean_res.append(np.mean(res_list[i]))
    var_res.append(np.var(res_list[i]))
    
print("mean and variance of residual for each ETF F-F model")
for i in range(len(etf_list)):
    print(etf_list[i],": mean: ",mean_res[i],"var: ",var_res[i])

# QQ plot for residuals/Kolmogorov-Smirnov test
fig = plt.figure()
pvalue_list = []
for i in range(1, 10):
    ax = fig.add_subplot(3, 3, i)
    sm.qqplot(np.asarray(res_list[i-1]),ax = ax)
    ax.text(-2,0.01,etf_list[i-1])
    pvalue_list.append(sm.stats.diagnostic.kstest_normal(res_list[i-1])[1])
print("By Kolmogorov-Smirnov test for residuals")
for i in range(len(etf_list)):
    print(etf_list[i],": p-value: ",pvalue_list[i])
    
    
    
### Question 2 ###
class Bachelier():
    def __init__(self,r = 0,sigma = 10,S0 = 100,K = 100,T = 1,n_steps = 100,N = 10000):
        self.r = r
        self.sigma = sigma
        self.S0 = S0
        self.K = K
        self.T = T
        self.n_steps = n_steps
        self.N = N
        print("constructor is called")
    
    def __del__(self):
        print("destructor is called")
    
    # Monte Carlo for all of simulated paths    
    def mc_simu(self,S0):
        dt = self.T/self.n_steps
        all_simu = []
        for i in range(self.N):
            simu = [S0]
            for j in range(self.n_steps):
                dW = np.random.normal(0,np.sqrt(dt))
                simu.append(self.sigma*dW + simu[j])
            all_simu.append(simu)
        return np.array(all_simu)
    
    # lookback put option price
    def lookback_put(self,S0):
        min_S = np.min(self.mc_simu(S0),axis = 1)
        payoff = [np.maximum(self.K - S, 0) for S in min_S]
        P = 1 / self.N * np.sum(payoff) * np.exp(-self.r * self.T)
        return P
    
    # delta by FDM
    def delta(self,eps):
        return (self.lookback_put(self.S0 + eps) - self.lookback_put(self.S0 - eps))/(2 * eps)
        
        
## (b) ##
bachelier_model = Bachelier() 
end_simu = bachelier_model.mc_simu(100)[:,-1]    
plt.figure()
plt.hist(end_simu)
plt.xlabel("ending value of simulation")
plt.title("Q2(b):Histogram of ending value for simulations")
print("p value of Kolmogorov-Smirnov test for ending value of simulations: ",sm.stats.diagnostic.kstest_normal(end_simu )[1])

## (c) ##
put = bachelier_model.lookback_put(100)
print("price of a Lookback put option:",put)

## (d) ##
eps_list = [0.01,0.1,0.3,0.5,1,3,5,10]
delta_list = []
for eps in eps_list:
    delta_list.append(bachelier_model.delta(eps))

plt.figure()
plt.plot(eps_list, delta_list)
plt.xlabel("eps")
plt.ylabel("delta")
plt.title("delta vs eps")        
        
        