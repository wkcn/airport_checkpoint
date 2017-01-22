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
        self.pop_tot_time = 0.0
        self.pop_wait_time = 0.0
        self.update_times = 0
        self.tot_size = 0
        self.last_push_time = t
        self.last_pop_time = t
        self.tot_push_interval = 0.0
        self.tot_pop_interval = 0.0
        self.tot_service_time = 0.0
        self.free_time = 0.0
        self.stable_time = 0.0 # p < 1
    def push(self, people):
        self.num += 1
        self.push_num += 1
        self.tot_push_interval += t - self.last_push_time
        self.last_push_time = t
        self.queue.put((people, t))
    def update(self, interval):
        if not self.empty():
            self.single_time += interval
        else:
            #queue is empty
            self.free_time += interval
        if self.get_p() < 1:
            self.stable_time += interval
        self.update_times += 1
        self.tot_size += self.size()
    def isok(self):
        return self.single_time >= self.ok_time and not self.empty()
    def pop(self):
        #print "out: %.1lf" % self.single_time
        self.tot_pop_interval += t - self.last_pop_time
        self.last_pop_time = t
        self.single_time = 0.0 # -delay
        peo,intime = self.queue.get() 
        self.pop_tot_time += max(t - intime, 0.0)
        self.pop_wait_time += max(t - intime - self.ok_time, 0.0)
        self.tot_service_time += self.ok_time
        self.pop_num += 1
        self.ok_time = max(0.0, self.get_service_time()) # may be minus ? 
        self.num -= 1
        return peo
    def size(self):
        return self.num
    def avg_size(self):
        if self.update_times > 0:
            return self.tot_size * 1.0 / self.update_times
        return 0.0
    def in_rate(self):
        if self.tot_push_interval > 0:
            return self.push_num * 1.0 / self.tot_push_interval
        return 0.0
    def out_rate(self):
        if self.tot_pop_interval > 0:
            return self.pop_num * 1.0 / self.tot_pop_interval
        return 0.0
    def get_p(self):
        out_r = self.out_rate()
        if out_r > 0:
            return self.in_rate() * 1.0 / out_r
        return 0.0
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
wave = (11.6359 + 6.6462, 5.8616) 
repack = (11.9848, 5.8616)
PAR = [None for _ in range(3)]
QS = [[[],[]] for _ in range(3)]
#A region parameter
PAR[0] = [[tcheck], [tcheck]]
#B region parameter
PAR[1] = [[wave], [wave]]
#C region parameter
PAR[2] = [[repack],[repack]]

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
print "total_time: %d" % need_time
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

for a in range(3):
    print "=========="
    print "Region: %c" % chr(ord('A') + a)
    for g in range(2):
        print " Kind: %s" % (["tsa", "reg"])[g]
        v = QS[a][g]
        for q in v:
            print "  Avg_Size: %.1lf, WaitTime: %.1lf" % (q.avg_size(), q.avg_wait_time())
            print "  in/out(minute): %.3lf/%.3lf, p: %.1lf" % (q.in_rate() * 60, q.out_rate() * 60, q.get_p())
            print "  free/stable: %.2lf/%.2lf" % (q.free_time, q.stable_time)
for i in range(2):
    print "OutK: %s" % (["tsa", "reg"])[i]
    print "%d / %d" % (people_out[i], people_in[i])
