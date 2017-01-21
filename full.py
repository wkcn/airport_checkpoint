#coding=utf-8
import numpy as np
import time
import Queue
import matplotlib.pyplot as plt

def get_randn(mean, std):
    r = np.random.randn()
    return r * std + mean

#global variance
t = 0.0

class People:
    def __init__(self):
        self.intime = 0.0
        self.service_time = 0.0
        self.flag = 0
    def qtime(self):
        return t - self.intime



class ServiceQueue:
    def __init__(self, func):
        self.num = 0
        self.single_time = 0.0
        self.get_service_time = func
        self.ok_time = self.get_service_time()
        self.queue = Queue.Queue()
        self.push_num = 0
        self.pop_num = 0
        self.pop_tot_time = 0
        self.pop_wait_time = 0
    def push(self, people):
        self.num += 1
        self.push_num += 1
        self.queue.put((people, t))
    def update(self, interval):
        if not self.empty():
            self.single_time += interval
    def isok(self):
        return self.single_time >= self.ok_time and not self.empty()
    def pop(self):
        #print "out: %.1lf" % self.single_time
        self.single_time = 0.0 # -delay
        peo,intime = self.queue.get() 
        self.pop_tot_time += max(t - intime, 0.0)
        self.pop_wait_time += max(t - intime - self.ok_time, 0.0)
        self.pop_num += 1
        self.ok_time = self.get_service_time()
        self.num -= 1
        return peo
    def size(self):
        return self.num
    def empty(self):
        return self.num == 0
    def avg_wait_time(self):
        # 等待时间
        if self.pop_num:
            return self.pop_wait_time * 1.0 / self.pop_num
        return 0.0
    def avg_tot_time(self):
        #逗留时间：等待 + 服务
        if self.pop_num:
            return self.pop_tot_time * 1.0 / self.pop_num
        return 0.0

def print_time(t):
    minute = int(t / 60)
    second = t - minute * 60
    print "%d:%.1lf" % (minute, second)

def get_rande(beta):
    return np.random.exponential(beta)

beta_tsa = 9.1912 # 信誉乘客 mean(delta_tsa)
beta_reg = 12.9457 # 普通乘客 mean(delta_reg)
betas = [beta_tsa, beta_reg]

tcheck = (11.2125, 3.7926)
wave = (11.6359, 5.8616) 
repack = (16.9848, 5.8616)
PAR = [None for _ in range(3)]
QS = [[[],[]] for _ in range(3)]
#A region parameter
PAR[0] = [[tcheck, tcheck], [tcheck]]
#B region parameter
PAR[1] = [[wave, wave, wave, wave], [wave]]
#C region parameter
PAR[2] = [[repack, repack, repack],[repack]]

#build queue
for a in range(3): # 3 regions
    for g in range(2): # 2 kinds of people
        v = QS[a][g]
        for p in PAR[a][g]:
            mu, sig = p
            v.append(ServiceQueue(lambda : get_randn(mu, sig)))

enter_time = [get_rande(betas[i]) for i in range(2)]

def JoinQ(qlayer, peo):
    bestqid = None
    bestnum = -1
    for q in QS[qlayer][peo.flag]:
        if q.size() < bestnum or bestqid == None:
            bestqid = q
            bestnum = q.size()
    bestqid.push(peo)

interval = 0.1
need_time = 8 * 60 + 50
people_in = [0,0]
people_out = [0,0]
for i in range(int(need_time / interval)):
    t += interval
    for i in range(2):
        if t >= enter_time[i]:
            enter_time[i] += betas[i]
            people_in[i] += 1
            peo = People()
            peo.flag = i
            peo.intime = t
            JoinQ(0, peo)

    for a in range(3): # 3 regions
        for g in range(2): # 2 kinds of people
            v = QS[a][g]
            for q in v:
                q.update(interval)
                if q.isok():
                    opeo = q.pop()
                    if a == 2:
                        #go out C region
                        people_out[opeo.flag] += 1
                    else:
                        JoinQ(a + 1, opeo)

print "=========="
for a in range(3):
    print "Region: %c" % chr(ord('A') + a)
    for g in range(2):
        print " Kind: %s" % (["tsa", "reg"])[g]
        v = QS[a][g]
        for q in v:
            print "  Size: %d, WaitTime: %lf" % (q.size(), q.avg_wait_time())
for i in range(2):
    print "OutK: %s" % (["tsa", "reg"])[i]
    print "%d / %d" % (people_out[i], people_in[i])
