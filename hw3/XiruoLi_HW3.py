import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import statsmodels.api as sm
from scipy import stats
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf

# (a)
# read data
yf.pdr_override()
spy_df = pdr.get_data_yahoo('spy', start="2010-01-01", end="2018-09-30")
vix_df = pdr.get_data_yahoo('^vix', start="2010-01-01", end="2018-09-30")
spy_cls = spy_df["Adj Close"]
vix_cls = vix_df["Adj Close"]
date = spy_df.index

# (b)
# ACF plot
plt.figure()
sm.graphics.tsa.plot_acf(spy_cls, lags=20) 
plt.title("SPY ACF plot")
plt.figure()
sm.graphics.tsa.plot_acf(vix_cls, lags=20)
plt.title("VIX ACF plot")


# (c)
# daily corr
corr_day = np.corrcoef(spy_cls,vix_cls)[0][1]
print("daily correlation between SPY and VIX is %.2f" %corr_day)
# monthly corr
spy_cls_month = spy_cls.asfreq('BM').dropna() 
vix_cls_month = vix_cls.asfreq('BM').dropna()
corr_month = np.corrcoef(spy_cls_month,vix_cls_month)[0][1]
print("monthly correlation between SPY and VIX is %.2f" %corr_month)   

# (d)
# 90-day rolling correlation
roll_corr = [] 
for i in range ((len(spy_cls) - 90)):
    roll_corr.append(np.corrcoef(spy_cls[i:i+90],vix_cls[i:i+90])[0,1])
plt.figure()
plt.plot(date[90:len(date)],roll_corr, label = "correlation")
plt.axhline(np.mean(roll_corr), color="red",label = "average correlation")
plt.title(" 90-day rolling correlation between SPY and VIX")
plt.legend()

# (e)
roll_var = []
for i in range ((len(spy_cls) - 90)):
    roll_var.append(np.std(spy_cls[i:i+90]) * np.sqrt(250/90))
premium = [x - y for x, y in zip(vix_cls[90:], roll_var)]
plt.figure()
plt.plot(date[90:], roll_var, label = "realized volatilities")
plt.plot(date[90:], vix_cls[90:], label = "implied volatilities")
plt.plot(date[90:], premium, label = "premium")
plt.legend()
plt.title("volatilities comparison")
print("When:", date[premium.index(np.min(premium)) + 90], "premium has the minimum value: ", np.min(premium))
print("When:", date[premium.index(np.max(premium)) + 90], "premium has the maximum value: ", np.max(premium))

# (f)
def Option_BS(r,S_t,K,sigma,t,T):
    d1 =  1 / (sigma * np.sqrt(T - t)) * (np.log(S_t / K) + (r + 0.5 * sigma ** 2) * (T - t))
    d2 = d1 - sigma * np.sqrt(T - t)
    C = S_t * stats.norm.cdf(d1) - stats.norm.cdf(d2) * K * np.exp(-r * (T - t))
    P = K * np.exp(-r * (T - t)) * stats.norm.cdf(-d2) - S_t * stats.norm.cdf(-d1)
    return C, P

r = 0
T = 1/12
t = 0
put_list = []
call_list = []

for i in range((len(spy_cls) - 21)):
    S_0 = spy_cls[i]
    K = S_0
    sigma = vix_cls[i] / 100
    call_list.append(Option_BS(r,S_0,K,sigma,t,T)[0])
    put_list.append(Option_BS(r,S_0,K,sigma,t,T)[1])

plt.figure()
plt.plot(date[21:],put_list,color = "r")
plt.title("put price by BS formula")
plt.figure()
plt.plot(date[21:],call_list,color = "b")
plt.title("call price by BS formula")


# (g)
payoff = []
PL = []
for i in range(21,len(spy_cls)):
    payoff.append(abs(spy_cls[i] - spy_cls[i-21]))
    PL.append(abs(spy_cls[i] - spy_cls[i-21]) - put_list[i - 21] - call_list[i - 21])
plt.plot(date[21:],payoff,label = "payoff")
plt.plot(date[21:],PL,label = "P&L")
plt.axhline(np.mean(PL), color="red",label = "average P&L")
plt.legend()
plt.title("P&L and payoff of straddle strategy")
print("average P&L is: %.2f" %np.mean(PL))

# (h)
plt.figure()
plt.scatter(premium, PL[69:], label = "P&L", alpha = 0.5)
plt.title("P&L vs premium")
plt.xlabel("premium")
plt.ylabel("P&L")
print("correlation between P&L and premium is:%.2f" %np.corrcoef(premium, PL[69:])[0][1])

#fig, ax1 = plt.subplots()
#color = 'tab:red'
#ax1.set_xlabel('date')
#ax1.set_ylabel('P&L', color=color)
#ax1.scatter(date[21:], PL, color=color,alpha = 0.5)
#ax1.tick_params(axis='y', labelcolor=color)
#
#ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#color = 'tab:blue'
#ax2.set_ylabel('premium', color=color)  # we already handled the x-label with ax1
#ax2.scatter(date[90:],premium, color=color,alpha = 0.5)
#ax2.tick_params(axis='y', labelcolor=color)
#plt.title("P&L vs premium")
#fig.tight_layout()  # otherwise the right y-label is slightly clipped



