#coding=utf-8
import copy
import NQueue as Queue

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

        
        #self.ok_time = None #self.get_service_time()
        self.serve_st_time = None
        self.serve_len = None

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

        self.out_queue = Queue.Queue()

    def get_service_time(self):
        stime = max(0.0, self.func(self.par))
        if not self.queue.empty():
            if self.queue.top()[0].slower:
                stime *= 1.5
        return stime
    def running(self):
        return self.opened or not self.empty()

    def push(self, people, t):
        if self.queue.empty():
            self.serve_st_time = t
            self.serve_len = self.get_service_time()
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
        self.tot_time += interval
        if not self.empty():
            #serving
            while not self.queue.empty() and self.tot_time >= self.serve_st_time + self.serve_len:
                #finish time
                t = self.serve_st_time + self.serve_len
                peo, intime = self.queue.get()

                self.tot_pop_interval += t - self.last_pop_time
                self.last_pop_time = t

                tot_time = t - intime
                wait_time = self.serve_st_time - intime

                self.pop_tot_time += tot_time 
                self.pop_wait_time += wait_time 
                self.tot_service_time += self.serve_len
                
                self.pop_num += 1

                peo.tot_time += tot_time 
                peo.wait_time += wait_time 

                self.out_queue.put((peo, t))
                
                if self.queue.empty():
                    self.serve_st_time = None
                else:
                    self.serve_st_time = t
                    self.serve_len = self.get_service_time()

        else:
            #queue is empty
            self.free_time += interval
        if self.get_p() < 1:
            self.stable_time += interval
        self.update_times += 1
        self.tot_size += self.size()
    def isok(self):
        return not self.out_queue.empty() 
    def pop(self):
        peo, t = self.out_queue.get()
        self.num -= 1
        return peo, t

        '''
        return peo, t
        '''
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
        #constant
        return self.par[0]
    def service_rate(self):
        if self.avg_service_time() > 0:
            return 1.0 / self.avg_service_time()
        return 0.0
    def in_rate(self):
        '''
        if not USE_SLIDE_WINDOW:
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

