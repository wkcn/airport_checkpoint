#coding=utf-8
import numpy as np
import time
import NQueue as Queue
import matplotlib.pyplot as plt
import copy
import random

need_time = 20 * 60# + 50

beta_tsa = 9.1912# 信誉乘客 mean(delta_tsa)
beta_reg = 12.9457# 普通乘客 mean(delta_reg)

tcheck = (11.2125, 3.7926)

wave_tsa = (7.5070, 4.7082) 
wave_reg = (15.0141, 6.6583)

rpack_tsa = (3.3350, 5.9169)
rpack_reg = (6.6701, 8.3678)

SLOWER_RATE = 0.0
CRASH_QUEUE_RATE = 0.0

MAX_QUEUE_SIZE = [8, 12, 12]
INIT_QUEUE_SIZE = [(1,1), (1,1), (1,1)]

PAR = [None for _ in range(3)]

def get_randn(par):
    mean, std = par
    r = np.random.randn()
    return r * std + mean

class PEOPLE_KIND:
    NORMAL = 0
    CRASH_QUEUE = 1
class People:
    def __init__(self):
        self.wait_time = 0.0
        self.tot_time = 0.0
        self.flag = 0
        self.kind = PEOPLE_KIND.NORMAL
        self.slower = False



class ServiceQueue:
    CRASH_TIME_PER_PEOPLE = 4.0
    WINDOW_SIZE = 10 * 60.0
    def __init__(self, func, par):
        self.num = 0
        self.func = func
        self.par = copy.copy(par)
        self.single_time = 0.0
        self.queue = Queue.Queue()
        self.ok_time = self.get_service_time()
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
        self.opened = False # Default Close

        self.push_rec = Queue.Queue()
        self.push_rec_top = None

    def get_service_time(self):
        stime = max(1.0, self.func(self.par))
        if not self.queue.empty():
            if self.queue.top()[0].slower:
                stime *= 1.5
        return stime
    def running(self):
        return self.opened or not self.empty()
    def push(self, people):
        t = self.tot_time
        self.num += 1
        self.push_num += 1 # history push_num
        self.tot_push_interval += t - self.last_push_time
        self.last_push_time = t

        self.push_rec.put((t, self.push_num))

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
        peo,intime = self.queue.get() 
        tot_time = max(t - intime, 0.0)
        wait_time = max(t - intime - self.ok_time, 0.0)
        self.pop_tot_time += tot_time 
        self.pop_wait_time += wait_time 
        self.tot_service_time += self.ok_time
        self.pop_num += 1

        self.single_time = 0.0 # -delay
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
        '''
        if self.pop_num > 0:
            return self.tot_service_time * 1.0 / self.pop_num
        return 0.0
        '''
        return self.par[0]
    def service_rate(self):
        if self.avg_service_time() > 0:
            return 1.0 / self.avg_service_time()
        return 0.0
    def in_rate(self):
        '''
        if self.tot_time > 0:
            return self.push_num * 1.0 / self.tot_time
        return 0.0
        '''
        #using slide window
        if self.push_rec.empty():
            return 0.0
        while self.push_rec_top == None or self.tot_time - self.push_rec_top[0] > self.WINDOW_SIZE:
            self.push_rec_top = self.push_rec.get()
        T = self.tot_time - self.push_rec_top[0]
        n = self.push_num - self.push_rec_top[1]
        if T > 0:
            return n * 1.0 / T 
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

#A region parameter
#PAR[0] = [[tcheck for _ in range(8)], [tcheck for _ in range(8)]]
PAR[0] = (tcheck, tcheck)
#B region parameter
#PAR[1] = [[wave_tsa for _ in range(12)], [wave_reg for _ in range(12)]]
PAR[1] = (wave_tsa, wave_reg)
#C region parameter
#PAR[2] = [[rpack_tsa for _ in range(8)],[rpack_reg for _ in range(8)]]
PAR[2] = (rpack_tsa, rpack_reg)


class World:
    # per minute
    COST = 0.2 
    PAY_OFFICER = 0.27 
    PAY_MACHINE = 0.14 
    # second
    CHECK_INTERVAL = 60 * 10.0
    PREDICT_TIME = 10.0 * 60
    def __init__(self):
        self.QS = [[[],[]] for _ in range(3)]
        self.betas = [beta_tsa, beta_reg]
        #build queue
        for a in range(3): # 3 regions
            for g in range(2): # 2 kinds of people
                v = self.QS[a][g]
                iq = INIT_QUEUE_SIZE[a][g]
                mu, sig = PAR[a][g]
                for _ in range(iq):
                    sq = ServiceQueue(get_randn, (mu, sig))
                    sq.opened = True
                    v.append(sq)
        self.enter_time = [get_rande(self.betas[i]) for i in range(2)]
        self.t = 0.0
        self.cost = 0.0
        self.cost_s = [0.0, 0.0, 0.0]
        self.people_in = [0,0]
        self.people_out = [0,0]

        sn = 20
        self.people_out_tot_time = [Queue.SQueue(sn),Queue.SQueue(sn)]
        self.people_out_wait_time = [Queue.SQueue(sn), Queue.SQueue(sn)]

        self.people_all_tot_time = Queue.SQueue(sn)
        self.people_all_wait_time = Queue.SQueue(sn)

        self.last_check_t = 0.0
        #init num
        self.cpa = INIT_QUEUE_SIZE[0][0]
        self.cpb = INIT_QUEUE_SIZE[1][0]
        self.cna = INIT_QUEUE_SIZE[0][1]
        self.cnb = INIT_QUEUE_SIZE[1][1]
        #record: 
        self.rec_io = [] # [(in, out), ...]
        self.rec_cost = []
        self.rec_avg_wait_time = []
    def add_valid(self, p):
        pa, pb, na, nb = p
        if self.cpa + self.cna + pa + na >= MAX_QUEUE_SIZE[0]:
            return False
        #Assume b == c
        if self.cpb + self.cnb + pb + nb >= MAX_QUEUE_SIZE[1]:
            return False
        f = lambda p : p > 0.0
        if pa and not self.qexist(self.QS[0][0], f):
            return False
        if na and not self.qexist(self.QS[0][1], f):
            return False
        if pb and not (self.qexist(self.QS[1][0], f) or self.qexist(self.QS[2][0], f)):
            return False
        if nb and not (self.qexist(self.QS[1][1], f) or self.qexist(self.QS[2][1], f)):
            return False
        return True
    def qexist(self, qs, f):
        for q in qs:
            if f(q.get_p()):
                return True
        return False
    def dec_valid(self, p):
        pa, pb, na, nb = p
        if self.cpa - pa < 1:
            return False
        if self.cna - na < 1:
            return False
        if self.cpb - pb < 1:
            return False
        if self.cnb - nb < 1:
            return False
        return True
    def open(self, p):
        pa, pb, na, nb = p
        # (precheckA, prechceckB, normalA, normalB)
        '''
        for a in range(3): # 3 regions
            for g in range(2): # 2 kinds of people
                v = self.QS[a][g]
                iq = INIT_QUEUE_SIZE[a][g]
                mu, sig = PAR[a][g]
                for _ in range(iq):
                    sq = ServiceQueue(get_randn, (mu, sig))
                    sq.opened = True
                    v.append(sq)
        '''
        if pa:
            self.cpa += 1
            a, g = 0, 0
            mu, sig = PAR[a][g]
            sq = ServiceQueue(get_randn, (mu, sig))
            sq.opened = True
            self.QS[a][g].append(sq)
        if na:
            self.cna += 1
            a, g = 0, 1
            mu, sig = PAR[a][g]
            sq = ServiceQueue(get_randn, (mu, sig))
            sq.opened = True
            self.QS[a][g].append(sq)
        if pb:
            self.cpb += 1
            a, g = 1, 0
            mu, sig = PAR[a][g]
            sq = ServiceQueue(get_randn, (mu, sig))
            sq.opened = True
            self.QS[a][g].append(sq)
            self.balance(self.QS[a][g])
            a, g = 2, 0
            mu, sig = PAR[a][g]
            sq = ServiceQueue(get_randn, (mu, sig))
            sq.opened = True
            self.QS[a][g].append(sq)
        if nb:
            self.cnb += 1
            a, g = 1, 1
            mu, sig = PAR[a][g]
            sq = ServiceQueue(get_randn, (mu, sig))
            sq.opened = True
            self.QS[a][g].append(sq)
            self.balance(self.QS[a][g])
            a, g = 2, 1
            mu, sig = PAR[a][g]
            sq = ServiceQueue(get_randn, (mu, sig))
            sq.opened = True
            self.QS[a][g].append(sq)

    def close(self, p):
        pa, pb, na, nb = p
        # (precheckA, prechceckB, normalA, normalB)
        if pa:
            a, g = 0, 0
            self.QS[a][g][-1].opened = False
        if na:
            a, g = 0, 1
            self.QS[a][g][-1].opened = False
        if pb:
            a, g = 1, 0
            self.QS[a][g][-1].opened = False
            a, g = 2, 0
            self.QS[a][g][-1].opened = False
        if nb:
            a, g = 1, 1
            self.QS[a][g][-1].opened = False
            a, g = 2, 1
            self.QS[a][g][-1].opened = False

    def get_min_pq(self, qs):
        best = None
        minp = np.inf
        for q in qs:
            if q.get_p() < minp:
                minp = q.get_p()
                best = q
        return best

    def balance(self, q):
        #TODO
        pass

    def JoinQ(self, qlayer, peo, anygo = False):
        bestqid = None
        bestnum = -1
        for i in range(2):
            if peo.flag == 1 and i != 1:
                continue
            if peo.flag == 0 and i != 0 and not anygo:
                continue
            for q in self.QS[qlayer][i]:
                if q.opened:
                    if q.size() < bestnum or bestqid == None:
                        bestqid = q
                        bestnum = q.size()
        bestqid.push(peo)

    def ut_cost(self):
        j = 0.0
        # PAX
        n = 0
        m = [0,0,0] # machine
        for a in range(3): # 3 regions
            for g in range(2): # 2 kinds of people
                for q in self.QS[a][g]:
                    n += q.size()
                    if q.running():
                        m[a] += 1
        w1 = self.COST * 1.0 * n
        # m[1] = m[2]
        # B region and C region
        w2 = (2 * self.PAY_OFFICER + self.PAY_MACHINE) * 1.0 * m[1] 
        # id check
        w3 = (1 * self.PAY_OFFICER ) * m[0]
        #print w1, w2, w3
        '''
        self.cost_s[0] += w1
        self.cost_s[1] += w2
        self.cost_s[2] += w3
        j += w1 + w2 + w3
        return j / 60.0 # per second
        '''
        return (w1 / 60.0, w2 / 60.0, w3 / 60.0)

    def update(self, interval, allow_check = True):
        self.t += interval
        w1, w2, w3 = self.ut_cost()
        self.cost += (w1 + w2 + w3) * interval
        self.cost_s[0] += w1 * interval
        self.cost_s[1] += w2 * interval
        self.cost_s[2] += w3 * interval
        for i in range(2):
            if self.t >= self.enter_time[i]:
                self.enter_time[i] += get_rande(self.betas[i])
                self.people_in[i] += 1
                peo = People()
                peo.flag = i
                peo.slower = (random.random() < SLOWER_RATE)
                if random.random() < CRASH_QUEUE_RATE:
                    peo.kind = PEOPLE_KIND.CRASH_QUEUE
                self.JoinQ(0, peo, True)

        for a in range(3): # 3 regions
            for g in range(2): # 2 kinds of people
                v = self.QS[a][g]
                for q in v:
                    q.update(interval)
                    if q.isok():
                        opeo = q.pop()
                        if a == 2:
                            #go out C region
                            self.people_out[opeo.flag] += 1

                            if allow_check:
                                self.people_out_tot_time[opeo.flag].put(opeo.tot_time)
                                self.people_out_wait_time[opeo.flag].put(opeo.wait_time)
                                self.people_all_tot_time.put(opeo.tot_time)
                                self.people_all_wait_time.put(opeo.wait_time)
                        else:
                            self.JoinQ(a + 1, opeo)
        if allow_check and self.t - self.last_check_t > self.CHECK_INTERVAL:
            #Check
            self.check()
            self.last_check_t = self.t
        self.delete_empty_queue()
        if allow_check:
            self.record()

    def delete_empty_queue(self):
        #delete empty queue
        for a in range(3): # 3 regions
            for g in range(2): # 2 kinds of people
                dlist = []
                qs = self.QS[a][g]
                for i in range(len(qs)):
                    if not qs[i].running():
                        dlist.append(i)
                if len(dlist):
                    nqs = []
                    for i in range(len(qs)):
                        if i not in dlist:
                            nqs.append(qs[i])
                    self.QS[a][g] = nqs
                    k = len(dlist)
                    if a == 0 and g == 0:
                        self.cpa -= k
                    if a == 0 and g == 1:
                        self.cna -= k
                    if a == 1 and g == 0:
                        self.cpb -= k
                    if a == 1 and g == 1:
                        self.cnb -= k
    def record(self):
        #recording
        cin = self.people_in[0] + self.people_in[1]
        cout = self.people_out[0] + self.people_out[1]
        self.rec_io.append( (cin, cout) )
        self.rec_cost.append(self.cost)
        self.rec_avg_wait_time.append(self.people_all_wait_time.get_avg())

    def get_copy_world(self):
        w = copy.deepcopy(self) 
        #todo: update world beta
        return w
    def check(self):
        # (precheckA, prechceckB, normalA, normalB)
        '''
        combines = [
                (0,0,0,0),
                (0,0,1,0),
                (0,0,1,1),
                (0,0,0,1),
                (0,1,0,0),
                (0,1,1,0),
                (0,1,0,1),
                (0,1,1,1)
        ]
        '''
        combines = []
        for i in range(1, 4 * 4):
            combines.append((i & 8 > 0, i & 4 > 0, i & 2 > 0, i & 1 > 0))


        bestp = None
        minCost = np.inf

        #combines exclude [F,F,F,F]
        for p in combines:
            if self.add_valid(p):
                wcost = self.test_world(p)
                if wcost < minCost:
                    minCost = wcost
                    bestp = p


        notCost = None
        notOpen = True
        dothing = False
        notp = [False, False, False, False]
        if bestp:
            notCost = self.test_world(notp)
            if minCost < notCost:
                if self.is_good_change(notCost, minCost):
                    notOpen = False
                    dothing = True
                    self.open(bestp)
                    print "Open: ", bestp, notCost, minCost - notCost
                    nc = []
                    for p in combines:
                        conflict = False
                        for i in range(4):
                            if bestp[i] and p[i]:
                                conflict = True
                                break
                        if not conflict:
                            nc.append(p)
                    bestp2, minCost2 = self.find_dec_best(nc)
                    if minCost2 < minCost:
                        if self.is_good_change(notCost, minCost2):
                            self.close(bestp2)
                            print "Then Close: ", bestp2, minCost, minCost2 - minCost
        if notOpen:
            bestp2, minCost2 = self.find_dec_best(combines)
            if bestp2:
                if notCost == None: 
                    notCost = self.test_world(notp)
                    if minCost2 < notCost:
                        if self.is_good_change(notCost, minCost2):
                            dothing = True
                            self.close(bestp2)
                            print "Close: ", bestp2, notCost, minCost2 - notCost
        if not dothing:
            print "No thing to do"

    def is_good_change(self, noCost, bestCost):
        noa = noCost - self.cost
        cha = noCost - bestCost
        r = cha * 1.0 / noa
        return r > 0.25

    def find_dec_best(self, combines):
        bestp2 = None
        minCost2 = np.inf
        for p in combines:
            if self.dec_valid(p):
                wcost2 = self.test_world(p, False)
                if wcost2 < minCost2:
                    minCost2 = wcost2
                    bestp2 = p
        return bestp2, minCost2
    def test_world(self, p, needopen = True):
        interval = 0.1
        #assume p is valid
        w = self.get_copy_world()

        w.betas[0] = 1.0 / w.QS[0][0][0].in_rate()
        w.betas[1] = 1.0 / w.QS[0][1][0].in_rate()
        #print w.betas
        
        if needopen:
            w.open(p)
        else:
            w.close(p)
        for _ in range(int(self.PREDICT_TIME * 1.0 / interval)):
            w.update(interval, False)
        return w.cost




class Simulation():#threading.Thread):
    def __init__(self):
        pass
        #threading.Thread.__init__(self)
    def run(self):
        self.simulation()
    def simulation(self):
        world = World()
        interval = 0.1
        while True:
            if world.t > need_time:
                break
            world.update(interval)



        for a in range(3):
            print "=========="
            print "Region: %c" % chr(ord('A') + a)
            for g in range(2):
                print " Kind: %s" % (["tsa", "reg"])[g]
                v = world.QS[a][g]
                for q in v:
                    print "  Avg_Size: %.1lf, WaitTime: %.1lf, ServiceTime: %.1lf" % (q.avg_size(), q.avg_wait_time(),q.avg_service_time())
                    print "  in/out/service(minute): %.3lf/%.3lf/%.3lf, p: %.1lf" % (q.in_rate() * 60, q.out_rate() * 60, q.service_rate() * 60, q.get_p())
                    print "  free/stable: %.2lf/%.2lf" % (q.free_time, q.stable_time)
        for i in range(2):
            print "OutK: %s" % (["tsa", "reg"])[i]
            print "%d / %d" % (world.people_out[i], world.people_in[i])

        print world.people_in, world.people_out
        print "cost: %.1lf" % world.cost
        print world.cost_s

        self.result = [world.people_all_tot_time.get_avg()]


sims = [Simulation() for _ in range(1)]
for s in sims:
    s.simulation()#start()

co = 0.0
for s in sims:
    #s.join()
    co += s.result[0]

print "T: ", co / len(sims)
print "end"
