import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.linear_model import LinearRegression

### Question 1 ###

## (a) ##
print("Q1 part(a)")
# all of data files
filename_list = ['SPY','XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY']
file_list = [s + '.csv' for s in filename_list]

# dataframe of all data files
df_list = []
for file in file_list:
    df_list.append(pd.read_csv(file))
    
# dataframe of adj close for each ETFs
list_close = []
for df in df_list:
    list_close.append(df['Adj Close'].values.tolist())
df_close = pd.DataFrame(np.array(list_close).T)
df_close.columns = filename_list
df_close = df_close.apply(pd.to_numeric)
df_close.index = df_list[0]['Date']
df_close.index = pd.to_datetime(df_close.index)

# check missing value of ETF data
print('Number of missing value for each ETF close price:')
print(df_close.isnull().sum(axis=0))


## (b) ##
print("Q1 part(b)")
# length of years
year_length = (df_close.index[-1] - df_close.index[0]).days/365
# return of ETFs
return_list = []
for filename in filename_list:
    return_list.append((df_close[filename].iloc[-1]/df_close[filename].iloc[0])**(1/year_length)-1)
return_dict = dict(zip(filename_list, return_list)) 
print("annualized return of each ETF: ",return_dict)
# standard deviation of ETFs
std_list = []
for filename in filename_list:
    std_list.append(df_close[filename].std()*np.sqrt(252))
std_dict = dict(zip(filename_list, std_list)) 
print("standard deviation of each ETF: ",std_dict)


## (c) ##
print("Q1 part(c)")
ret_day_list = [] # list for daily return of ETF
ret_month_list = [] # list for monthly return of ETF
df_close_month = df_close.asfreq('BM').dropna() # monthly close
for filename in filename_list:
    close_day = df_close[filename].values.tolist()
    close_month = df_close_month[filename].values.tolist()
    ret_day = []
    ret_month = []
    for i in range(1,len(close_day)):
        ret_day.append((close_day[i]-close_day[i-1])/close_day[i-1])
    for i in range(1,len(close_month)):
        ret_month.append((close_month[i]-close_month[i-1])/close_month[i-1])
    ret_day_list.append(ret_day)
    ret_month_list.append(ret_month)
covar_day = np.cov(np.array(ret_day_list)) # covar matrix for daily return
covar_month = np.cov(np.array(ret_month_list)) # covar matrix for monthly return
#covar_day = np.corrcoef(np.array(ret_day_list)) # corr matrix for daily return
#covar_month = np.corrcoef(np.array(ret_month_list)) # corr matrix for monthly return
# plot covar matrix
sns.set(font_scale=0.8)
plt.figure()
ax1 = plt.axes()
sns.heatmap(covar_day, xticklabels = filename_list,yticklabels = filename_list,annot=True,ax = ax1)
ax1.set_title('Q1(c):covariance matrix of daily return')
plt.figure()
ax2 = plt.axes()
sns.heatmap(covar_month, xticklabels = filename_list,yticklabels = filename_list,annot=True,ax = ax2)
ax2.set_title('Q1(c):covariance matrix of monthly return')
# comment 
print("Covariance of monthly return is higher than daily return")


## (d) ##
print("Q1 part(d)")
roll_corr_list = [] # 90-day rolling correlation between each ETF and S&P500
close0 = df_close.iloc[:,0].values.tolist()
for i in range(1,len(filename_list)):
    close1 = df_close.iloc[:,i].values.tolist()
    roll_corr = []
    for j in range ((len(close1) - 90)):
        roll_corr.append(np.corrcoef(close0[j:j+90],close1[j:j+90])[0,1])
    roll_corr_list.append(roll_corr)
# plot
color_list = plt.cm.rainbow(np.linspace(0, 1, 9))
plt.figure()
for i in range(len(roll_corr_list)):
    plt.plot(df_close.index[90:len(df_close.index)],roll_corr_list[i],label = filename_list[i+1] + ' vs S&P500',color = color_list[i])
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("90-day rolling correlation")
    plt.title("Q1(d):90-day rolling correlation between each ETF and S&P500")
# comment
print("Rolling correlation is unstable over time")
print("In certain season, correlation varies significantly: like XLU and XLE, their correlation with S&P moves to negative value during the certain searson of year")


## (e) ##
print("Q1 part(e)")
beta_list = [] # beta for entire historical data
roll_beta_list = [] # beta for rolling 90-day data
for i in range(1,len(ret_day_list)):
    reg = LinearRegression()
    coef = reg.fit(np.asarray(ret_day_list[i]).reshape(-1,1),np.asarray(ret_day_list[0])).coef_.item()
    beta_list.append(coef)
    roll_beta = []
    for j in range((len(close1) - 90)):
        reg = LinearRegression()
        coef = reg.fit(np.asarray(ret_day_list[0][j:j+90]).reshape(-1,1), np.asarray(ret_day_list[i][j:j+90])).coef_.item()
        roll_beta.append(coef)
    roll_beta_list.append(roll_beta)
beta_dict = dict(zip(filename_list[1:], beta_list)) 
print("beta of each ETF for entire historical data: ",beta_dict)
# plot
color_list = plt.cm.rainbow(np.linspace(0, 1, 9))
plt.figure()
for i in range(len(roll_corr_list)):
    plt.plot(df_close.index[90:len(df_close.index)],roll_beta_list[i],label = filename_list[i+1],color = color_list[i])
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("90-day rolling beta")
    plt.title("Q1(e):90-day rolling beta between each ETF and S&P500")
# comment
print("Rolling beta is unstable over time")
print("Rolling beta and rolling correlation move together with time: like XLU, when correlation move to negative value, the beta become lower.")


## (f) ##
print("Q1 part(f)")
alpha_list = [] # alpha for each ETF
ret_day_df = pd.DataFrame(np.array(ret_day_list).T)
for i in range(ret_day_df.shape[1]):
    alpha_list.append(ret_day_df.iloc[:,i].autocorr(lag = 1))
alpha_dict = dict(zip(filename_list, alpha_list)) 
print("alpha of each ETF by auto-correlation: ",alpha_dict)
print("The alphas of ETF are between -0.01 and -0.1, which means auto correlation is weak")



### Question 2 ###

## (a) ##
print("Q2 part(a)")
# parameters
r = 0.03 # risk free rate
S_0 = 100 # begining underlying price
K = 100 # strike price
sigma = 0.25 # volatility of underlying price
T = 1 # Time to maturity in year
n_steps = 100 # number of time steps for simulation
N = 10000 # number of simulations

# MC simulation for Europe Put option pricing
def MC_Put_EU(r,S_0,K,sigma,T,n_steps,N):
    dt = T/n_steps # time interval
    S_simu = [] # terminal underlying price of all simulations
    payoff = [] # payoff of all simulations
    for i in range(N):
        S = S_0
        for j in range(n_steps):
#            W = np.random.normal(0,1)
#            S = S*np.exp((r - 0.5 * sigma ** 2) * dt + (sigma * np.sqrt(dt) * W))
            dW = np.random.normal(0,np.sqrt(dt))
            S = S + sigma * S * dW + r * S * dt
        S_simu.append(S)
        payoff.append(np.maximum(K - S, 0)) 
    P = 1 / N * np.sum(payoff) * np.exp(-r * T) # put price
    return S_simu,payoff,P

S_simu,payoff,P_mc_eu = MC_Put_EU(r,S_0,K,sigma,T,n_steps,N)

mean_S = np.mean(S_simu) # mean of terminal underlying value
var_S = np.var(S_simu) # variance of terminal underlying value
print("mean of terminal underlying value: ", mean_S)
print("variance of terminal underlying value: ", var_S)
print("As mean is close to 100 and variance is close to 25^2, it's consistent with the assumption")


## (b) ##
print("Q2 part(b)")
mean_P = np.mean(payoff) # mean of put
std_P = np.std(payoff) # standard deviation of put
print("mean of put: ", mean_P)
print("standard deviation of put: ", std_P)
# histogram for put
plt.figure()
plt.hist(payoff)
plt.xlabel("Put Price")
plt.title("Q2(b):Histogram of MC Simulation for Put Pricing")


## (c) ##
print("Q2 part(c)")
print("price of a European put option by taking the average discounted payoff across all path: ", P_mc_eu)


## (d) ##
print("Q2 part(d)")
# BS formula for Put option pricing
def BS_Put(r,S_t,K,sigma,t,T):
    d1 =  1 / (sigma * np.sqrt(T - t)) * (np.log(S_t / K) + (r + 0.5 * sigma ** 2) * (T - t))
    d2 = d1 - sigma * np.sqrt(T - t)
    P = K * np.exp(-r * (T - t)) * stats.norm.cdf(-d2) - S_t * stats.norm.cdf(-d1)
    return P
P_bs = BS_Put(r,S_0,K,sigma,0,T)
print("Put price by BS formula is: ",P_bs )
print("Put price by MC simulation is closed to the price by BS formula")


## (e) ##
print("Q2 part(e)")
# MC simulation for lookback Put option pricing
def MC_Put_lookback(r,S_0,K,sigma,T,n_steps,N):
    dt = T/n_steps # time interval
    payoff = [] # payoff of all simulations
    for i in range(N):
        S = [S_0]
        for j in range(n_steps):
#            W = np.random.normal()
#            S.append(S[-1]*np.exp((r - 0.5 * sigma ** 2) * dt + (sigma * np.sqrt(dt) * W)))
            dW = np.random.normal(0,np.sqrt(dt))
            S.append(S[-1]*sigma*dW + S[-1])
        payoff.append(np.maximum(K - min(S), 0))
    P = 1 / N * np.sum(payoff) * np.exp(-r * T) # put price
    return P    
P_mc_lookback = MC_Put_lookback(r,S_0,K,sigma,T,n_steps,N)
print("Price of a lookback put option by taking the average discounted payoff across all path: ", P_mc_lookback)


## (f) ##
print("Q2 part(f)")
premium = P_mc_lookback - P_mc_eu
print("Premium that the buyer is charged for the extra optionality embedded in the lookback is: ", premium)
print("Premium is highest when the option contract starts and is lowest when it's at the maturity. It will never be negative since the buyer can choose the lowest underlying price.")


## (g) ##
print("Q2 part(g)")
put_eu_list = []
put_lookback_list = []
premium_list = []
sigma_list = [0.1,0.5,1.0]
for sigma in sigma_list:
    put_eu_list.append(MC_Put_EU(r,S_0,K,sigma,T,n_steps,N)[2])
    put_lookback_list.append(MC_Put_lookback(r,S_0,K,sigma,T,n_steps,N))
    premium_list.append(put_lookback_list[-1] - put_eu_list[-1])    
print("Europe Put with vary sigma: ", dict(zip(sigma_list, put_eu_list)))
print("Lookback Put with vary sigma: ", dict(zip(sigma_list, put_lookback_list)))
print("Premium with vary sigma: ", dict(zip(sigma_list, premium_list)))
print("When sigma increases from 0.1 to 1, Europe put price increases, lookback price increases and its premium increases as well")
    
    
    
    










