#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
#xout = np.array([2.2100  , 5.6300  , 9.0500 , 12.4700 , 15.8900 , 19.3100 , 22.7300 , 26.1500 , 29.5700 , 32.9900])
xout = np.array([   4.4200, 12.2600, 20.1000  ,27.9400,  35.7800  ,43.6200,51.4600 , 59.3000 , 67.1400 , 74.9800]) # reg
w = xout[1] - xout[0]
#ns = np.array([26  ,  8  ,  4  ,  2  ,  4  ,  5  ,  3  ,  1  ,  3   , 1]).astype(np.float)
ns = np.array([  21 ,13   , 6 ,3   , 0 , 0   , 1 ,  1,  0 , 1]).astype(np.float)

#ns /= np.sum(ns)

f = lambda x, k : k * np.exp(- k * x) 
F = lambda x, k : 1 - np.exp(- k * x) 

print ns.size
def fit(k):
    J = 0.0
    h = w / 2.0
    n = sum(ns)
    for i in range(len(xout)):
        x = xout[i]
        fi = ns[i] # 频数
        a = F(x + h, k) - F(x - h, k)
        cost = (fi - a * n) ** 2
        #J = np.max(cost, J)
        J += cost / (a * n)
    return J
print fit(1.0 / 16.9457)
fewaf

t = 0.0
minJ = np.inf
bestT = -1
for i in range(10000):
    t += 0.0001
    e = fit(t)
    if e < minJ:
        minJ = e
        bestT = t

print minJ, bestT
ns /= sum(ns)
ns /= w
plt.plot(xout, ns, '')
#bestT = 1 / 9.1912
fits = [f(x, bestT) for x in xout]
print fits
plt.plot(xout, fits, 'r')
plt.show()
