import matplotlib.pyplot as plt  


#(a)(b)(c) 
def SwaptoForward(sw):
    fwd = {}.fromkeys(list(sw))
    fwd[1] = (1 + sw[1]/2) ** 2 - 1 # first year forward rate
    # 2 - 30 year forward rate
    for i in range(1,len(sw)):
        year1 = list(sw.keys())[i-1]
        year2 = list(sw.keys())[i]
        s1 = (1 + sw[year1] / 2) ** (year1 * 2)
        s2 = (1 + sw[year2] / 2 ) ** (year2 * 2)
        fwd[year2] = (s2 / s1) ** (1 / (year2 - year1)) - 1
    return fwd

swap_rate = {1:0.028438, 2:0.03060, 3:0.03126, 4:0.03144, 5:0.03150, 7:0.03169, 10:0.03210, 30:0.03237}
forward_rate = SwaptoForward(swap_rate)
for k,v in forward_rate.items():
    print(k,"year forward rate is: ",round(v,5))

plt.figure()
plt.plot(list(swap_rate),list(swap_rate.values()), label = "swap rate")
plt.plot(list(forward_rate),list(forward_rate.values()), label = "forward rate")
plt.xlabel("year")
plt.ylabel("rate")
plt.legend()
plt.title("swap rate vs forward rate")


#(d)
def breakeven_15(fwd):
    pre = 1
    year_last = 0
    for year in list(fwd)[:-1]:
        pre *= (1 + fwd[year]) ** (year - year_last)
        year_last = year
    fair_rate = 2 * (((1 + fwd[30]) ** 5 * pre) ** (1/30) - 1)
    return fair_rate

breakeven_rate_15 = round(breakeven_15(forward_rate),5)
print("breakeven swap rate of 15 year swap is:",breakeven_rate_15)
    

#(e)
def discount(sw):
    zero_rate = {}.fromkeys(list(sw))
    discount_factor = {}.fromkeys(list(sw))
    for year in list(sw):
        discount_factor[year] = 1 / (1 + sw[year]/2) ** (2 * year)
        zero_rate[year] = (1 / discount_factor[year]) ** (1 / year) - 1
    return zero_rate,discount_factor

zero_rate,discount_factor = discount(swap_rate)     
for k,v in zero_rate.items():
    print(k,"year zero rate is: ",round(v,5))
for k,v in discount_factor.items():
    print(k,"year discount factor is: ",round(v,5))
    
plt.figure()
plt.plot(list(swap_rate),list(swap_rate.values()), label = "swap rate")
plt.plot(list(zero_rate),list(zero_rate.values()), label = "zero rate")
plt.xlabel("year")
plt.ylabel("rate")
plt.legend()
plt.title("swap rate vs zero rate")   


#(f)
def breakeven(fwd):
    sw = {}.fromkeys(list(fwd))
    pre = 1
    year_last = 0
    for year in list(sw):
        pre *= (1 + fwd[year]) ** (year - year_last)
        year_last = year
        sw[year] = (pre ** (1 / (2 * year)) - 1) * 2
    return sw

forward_rate1 = {}.fromkeys(list(forward_rate))
swap_rate1 = {}.fromkeys(list(swap_rate))
for i in list(forward_rate1):
    forward_rate1[i] = forward_rate[i] + 0.01
    swap_rate1[i] = swap_rate[i] + 0.01

swap_rate2 = breakeven(forward_rate1)
for k,v in swap_rate2.items():
    print(k,"year breakeven swap rate is: ",round(v,5))
    
plt.figure()
plt.plot(list(swap_rate1),list(swap_rate1.values()), label = "swap rate with direct shift 100 basis points")
plt.plot(list(swap_rate2),list(swap_rate2.values()), label = "breakeven swap rate")
plt.xlabel("year")
plt.ylabel("rate")
plt.legend()
plt.title("breakeven swap rate vs swap rate with direct shift")   

#(g)
swap_rate3 = {1:0.028438, 2:0.03060, 3:0.03126, 4:0.03194, 5:0.03250, 7:0.03319, 10:0.03460, 30:0.03737}
for k,v in swap_rate3.items():
    print(k,"year bear swap rate is: ",round(v,5))

#(h)
forward_rate3 = SwaptoForward(swap_rate3)
for k,v in forward_rate3.items():
    print(k,"year bear forward rate is: ",round(v,5))
    
plt.figure()
plt.plot(list(swap_rate),list(swap_rate.values()), label = "swap rate")
plt.plot(list(forward_rate),list(forward_rate.values()), label = "forward rate")
plt.plot(list(swap_rate3),list(swap_rate3.values()),label = "bear swap rate")
plt.plot(list(forward_rate3),list(forward_rate3.values()),label = "bear forward rate")
plt.xlabel("year")
plt.ylabel("rate")
plt.legend()
plt.title("swap rate vs forward rate")
    
#(g)
swap_rate4 = {1:0.023438, 2:0.0281, 3:0.02976, 4:0.03044, 5:0.03100, 7:0.03169, 10:0.03210, 30:0.03237}
for k,v in swap_rate4.items():
    print(k,"year bull swap rate is: ",round(v,5))
    
#(h)
forward_rate4 = SwaptoForward(swap_rate4)
for k,v in forward_rate4.items():
    print(k,"year bull forward rate is: ",round(v,5))

plt.figure()
plt.plot(list(swap_rate),list(swap_rate.values()), label = "swap rate")
plt.plot(list(forward_rate),list(forward_rate.values()), label = "forward rate")
plt.plot(list(swap_rate4),list(swap_rate4.values()), label = "bull swap rate")
plt.plot(list(forward_rate4),list(forward_rate4.values()), label = "bull forward rate")
plt.xlabel("year")
plt.ylabel("rate")
plt.legend()
plt.title("swap rate vs forward rate")
    




    
    
    