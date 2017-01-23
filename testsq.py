#coding=utf-8
from ServiceQueue import *

import Queue
import random
import numpy as np
import copy

def get_rande(beta):
    #print -np.log(random.random()) / beta
    return -np.log(random.random()) / beta
    #return np.random.exponential(beta)

def get_randn(par):
    mean, std = par
    r = np.random.randn()
    return r * std + mean


u = 1
yy = 0.0
for k in range(u):
    t = 0.0
    interval = 0.1
    n = 100000
    beta = 5.156
    enter_time = get_rande(beta)
    ss = 9.028
    q = ServiceQueue(get_rande, ss)
    #for i in range(1000):
    while n > 0:
        q.update(interval)
        t += interval
        while t > enter_time:
            n -= 1
            p = People()
            q.push(p, enter_time)
            enter_time += get_rande(beta)
            #print "=", get_rande(beta)
        while q.isok():
            q.pop()
    while not q.empty():
        q.update(interval)
        if q.isok():
            q.pop()
    #yy += q.avg_wait_time()
    yy += q.avg_size() #5.607
print yy / u 
