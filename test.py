#coding=utf-8
import numpy as np
import time
import Queue

def get_randn(mean, std):
    r = np.random.randn()
    return r * std + mean

class ServiceQueue:
    def __init__(self, func):
        self.num = 0
        self.single_time = 0.0
        self.get_service_time = func
        self.ok_time = self.get_service_time()
        self.queue = Queue.Queue()
    def push(self, flag = 0):
        self.num += 1
        self.queue.put(flag)
    def update(self, interval):
        if not self.empty():
            self.single_time += interval
    def isok(self):
        return self.single_time >= self.ok_time and not self.empty()
    def pop(self):
        #print "out: %.1lf" % self.single_time
        self.single_time = 0.0 # -delay
        self.ok_time = self.get_service_time()
        self.num -= 1
        return self.queue.get()
    def size(self):
        return self.num
    def empty(self):
        return self.num == 0

def print_time(t):
    minute = int(t / 60)
    second = t - minute * 60
    print "%d:%.1lf" % (minute, second)

def get_rande(beta):
    return np.random.exponential(beta)

beta = 9.1912 # 信誉乘客 mean(delta_tsa)
beta2 = 10.9143 # 普通乘客 mean(delta_reg)
betas = [beta]
#enterq = ServiceQueue(lambda :np.random.exponential(beta))
sq1 = ServiceQueue(lambda :get_randn(10.1889, 2.9793)) #信誉乘客IDCheck1
sq2 = ServiceQueue(lambda :get_randn(12.5286, 4.5313)) #普通乘客IDCheck2
sqitem = ServiceQueue(lambda :get_randn(28.6207, 14.0901)) #信誉乘客time to get scanned property
sqitem2 = ServiceQueue(lambda :get_randn(28.6207, 14.0901)) #普通乘客time to get scanned property
t = 0.0
interval = 0.01
need_time = 8 * 60 + 50
#enter_time = get_rande(beta) 
#enter_time2 = get_rande(beta2) 
enter_time = [get_rande(betas[i]) for i in range(len(betas))]
people_num = 0
people_check = 0
people_item = 0
for i in range(int(need_time / interval)):
    t += interval
    '''
    if t >= enter_time:
        #print_time(t)
        enter_time += get_rande(beta)
        people_num += 1
        sq1.push()
    '''
    for i in range(len(betas)):
        if t >= enter_time[i]:
            enter_time[i] += get_rande(betas[i]) 
            people_num += 1
            sq1.push(i)

    sq1.update(interval)
    if sq1.isok():
        flag = sq1.pop()
        #print flag
        sqitem.push()
        #print_time(t)
        #print sq1.size()
        people_check += 1
    sqitem.update(interval)
    if sqitem.isok():
        sqitem.pop()
        people_item += 1

print "check: %d / %d" % (people_check, people_num)
print "item: %d" % people_item
print "out: %.2lf" % (people_item * 1.0 / people_num)

