#coding=utf-8
import numpy as np
import time
import Queue
import matplotlib.pyplot as plt
import copy
import random
import threading

def get_randn(par):
    mean, std = par
    r = np.random.randn()
    return r * std + mean

class PEOPLE_KIND:
    NORMAL = 0
    CRASH_QUEUE = 1
class People:
    def __init__(self):
        self.intime = 0.0
        self.wait_time = 0.0
        self.tot_time = 0.0
        self.flag = 0
        self.kind = PEOPLE_KIND.NORMAL



class ServiceQueue:
    CRASH_TIME_PER_PEOPLE = 4.0
    def __init__(self, func, par):
        self.num = 0
        self.func = func
        self.par = copy.copy(par)
        self.single_time = 0.0
        self.ok_time = self.get_service_time()
        self.queue = Queue.Queue()
        self.push_num = 0
        self.pop_num = 0
        self.pop_tot_time = 0.0
        self.pop_wait_time = 0.0
        self.update_times = 0
        self.tot_size = 0
        self.last_push_time = 0.0
        self.last_pop_time = 0.0
        self.tot_push_interval = 0.0
        self.tot_pop_interval = 0.0
        self.tot_service_time = 0.0
        self.free_time = 0.0
        self.stable_time = 0.0 # p < 1
        self.tot_time = 0.0
    def get_service_time(self):
        return self.func(self.par)
    def push(self, people):
        t = self.tot_time
        self.num += 1
        self.push_num += 1
        self.tot_push_interval += t - self.last_push_time
        self.last_push_time = t
        if random.randint(0,8) == 3:
            pass
            people.kind = PEOPLE_KIND.CRASH_QUEUE
        if people.kind == PEOPLE_KIND.NORMAL:
            self.queue.put((people, t))
        elif people.kind == PEOPLE_KIND.CRASH_QUEUE:
            #crash queue
            w = 0
            if self.avg_service_time() > 0:
                w = int(np.ceil(self.queue.qsize() * 1.0 * ServiceQueue.CRASH_TIME_PER_PEOPLE / self.avg_service_time()))
            ntime = w * ServiceQueue.CRASH_TIME_PER_PEOPLE
            nq = Queue.Queue()
            for _ in range(w):
                if not self.queue.empty():
                    nq.put(self.queue.get())
            people.wait_time += ntime 
            people.tot_time += ntime
            nq.put((people,t))
            while not self.queue.empty():
                nq.put(self.queue.get())
            self.queue = nq

    def update(self, interval):
        if not self.empty():
            self.single_time += interval
        else:
            #queue is empty
            self.free_time += interval
        if self.get_p() < 1:
            self.stable_time += interval
        self.update_times += 1
        self.tot_time += interval
        self.tot_size += self.size()
    def isok(self):
        return self.single_time >= self.ok_time and not self.empty()
    def pop(self):
        t = self.tot_time
        #print "out: %.1lf" % self.single_time
        self.tot_pop_interval += t - self.last_pop_time
        self.last_pop_time = t
        self.single_time = 0.0 # -delay
        peo,intime = self.queue.get() 
        tot_time = max(t - intime, 0.0)
        wait_time = max(t - intime - self.ok_time, 0.0)
        self.pop_tot_time += tot_time 
        self.pop_wait_time += wait_time 
        self.tot_service_time += self.ok_time
        self.pop_num += 1
        self.ok_time = max(0.0, self.get_service_time()) # may be minus ? 
        self.num -= 1

        peo.tot_time += tot_time 
        peo.wait_time += wait_time 
        return peo
    def size(self):
        return self.num
    def avg_size(self):
        if self.update_times > 0:
            return self.tot_size * 1.0 / self.update_times
        return 0.0
    def avg_service_time(self):
        if self.pop_num > 0:
            return self.tot_service_time * 1.0 / self.pop_num
        return 0.0
    def service_rate(self):
        if self.avg_service_time() > 0:
            return 1.0 / self.avg_service_time()
        return 0.0
    def in_rate(self):
        if self.tot_time > 0:
            return self.push_num * 1.0 / self.tot_time
        return 0.0
    def out_rate(self):
        if self.tot_time > 0:
            return self.pop_num * 1.0 / self.tot_time
        return 0.0
    def get_p(self):
        s_r = self.service_rate()
        if s_r > 0:
            return self.in_rate() * 1.0 / s_r
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

def AddNormal(a, b):
    mu = a[0] + b[0]
    sig = np.sqrt(a[1] ** 2 + b[1] ** 2)
    return (mu, sig)

beta_tsa = 9.1912# 信誉乘客 mean(delta_tsa)
beta_reg = 12.9457# 普通乘客 mean(delta_reg)
betas = [beta_tsa, beta_reg]

tcheck = (11.2125, 3.7926)

wave_tsa = (7.5070, 4.7082) 
wave_reg = (15.0141, 6.6583)

rpack_tsa = (3.3350, 5.9169)
rpack_reg = (6.6701, 8.3678)

PAR = [None for _ in range(3)]

#A region parameter
PAR[0] = [[tcheck for _ in range(1)], [tcheck for _ in range(1)]]
#B region parameter
PAR[1] = [[wave_tsa for _ in range(1)], [wave_reg for _ in range(2)]]
#C region parameter
PAR[2] = [[rpack_tsa for _ in range(1)],[rpack_reg for _ in range(1)]]

class Simulation(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        self.simulation()
    def simulation(self):
        QS = [[[],[]] for _ in range(3)]
#build queue
        for a in range(3): # 3 regions
            for g in range(2): # 2 kinds of people
                v = QS[a][g]
                for p in PAR[a][g]:
                    mu, sig = p
                    v.append(ServiceQueue(get_randn, (mu,sig)))

        enter_time = [get_rande(betas[i]) for i in range(2)]

        def JoinQ(qlayer, peo, anygo = False):
            bestqid = None
            bestnum = -1
            for i in range(2):
                if peo.flag == 1 and i != 1:
                    continue
                if peo.flag == 0 and i != 0 and not anygo:
                    continue
                for q in QS[qlayer][i]:
                    if q.size() < bestnum or bestqid == None:
                        bestqid = q
                        bestnum = q.size()
            bestqid.push(peo)

        interval = 0.1
        need_time = 10 * 60# + 50
        people_in = [0,0]
        people_out = [0,0]
        people_ok = []
#for i in range(int(sim_time / interval)):
        t = 0.0
        while True:
            t += interval
            if t > need_time:
                break
                if people_in[0] + people_in[1] == people_out[0] + people_out[1]:
                    print "sim_time: %.1lf" % t
                    break
            if t < need_time:
                for i in range(2):
                    if t >= enter_time[i]:
                        enter_time[i] += betas[i]
                        people_in[i] += 1
                        peo = People()
                        peo.flag = i
                        peo.intime = t
                        JoinQ(0, peo, False)

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
                                people_ok.append(opeo)
                            else:
                                JoinQ(a + 1, opeo)
        for a in range(3):
            print "=========="
            print "Region: %c" % chr(ord('A') + a)
            for g in range(2):
                print " Kind: %s" % (["tsa", "reg"])[g]
                v = QS[a][g]
                for q in v:
                    print "  Avg_Size: %.1lf, WaitTime: %.1lf, ServiceTime: %.1lf" % (q.avg_size(), q.avg_wait_time(),q.avg_service_time())
                    print "  in/out/service(minute): %.3lf/%.3lf/%.3lf, p: %.1lf" % (q.in_rate() * 60, q.out_rate() * 60, q.service_rate() * 60, q.get_p())
                    print "  free/stable: %.2lf/%.2lf" % (q.free_time, q.stable_time)
        for i in range(2):
            print "OutK: %s" % (["tsa", "reg"])[i]
            print "%d / %d" % (people_out[i], people_in[i])

        min_wait_time = np.inf
        max_wait_time = -np.inf
        tot_wait_time = 0.0

        min_tot_time = np.inf
        max_tot_time = -np.inf
        tot_tot_time = 0.0
        for p in people_ok:
            min_wait_time = min(min_wait_time, p.wait_time)
            max_wait_time = max(max_wait_time, p.wait_time)
            tot_wait_time += p.wait_time

            min_tot_time = min(min_tot_time, p.tot_time)
            max_tot_time = max(max_tot_time, p.tot_time)
            tot_tot_time += p.tot_time

        people_ok_num = len(people_ok)
        avg_wait_time = tot_wait_time * 1.0 / people_ok_num
        avg_tot_time = tot_tot_time * 1.0 / people_ok_num

        print min_wait_time, max_wait_time, avg_wait_time
        print min_tot_time, max_tot_time, avg_tot_time

        self.result = [avg_tot_time]


sims = [Simulation() for _ in range(1)]
for s in sims:
    s.simulation()#start()

co = 0.0
for s in sims:
    #s.join()
    co += s.result[0]

print "T: ", co / len(sims)
print "end"
