class Queue3:
    def __init__(self):
        self.siz = 0 
        self.alc = 10
        self.data = [None for _ in range(self.alc)]
        self.i = 0
        self.j = 0
    def step(self, k):
        k += 1
        if k >= self.alc:
            return 0
        return k
    def empty(self):
        return self.siz == 0
    def put(self, x):
        if self.siz >= self.alc:
            nalc = self.alc * 4 
            nd = [None for _ in range(nalc)]
            for i in range(self.siz):
                nd[i] = self.data[self.i]
                self.i = self.step(self.i)
            self.i = 0
            self.j = self.siz
            self.alc = nalc
            self.data = nd
        self.siz += 1
        self.data[self.j] = x
        self.j = self.step(self.j)
    def get(self):
        x = self.data[self.i]
        self.i = self.step(self.i)
        self.siz -= 1
        return x
    def qsize(self):
        return self.siz

class Queue:
    def __init__(self):
        self.siz = 0 
        self.data = []
    def empty(self):
        return self.siz == 0
    def put(self, x):
        self.siz += 1
        self.data.append(x)
    def get(self):
        self.siz -= 1
        x = self.data[0]
        del self.data[0]
        return x
    def qsize(self):
        return self.siz

if __name__ == "__main__":
    import time
    def test(q):
        t = time.time()
        for i in range(1000):
            q.put(i)
        k = 0
        while not q.empty():
            if k != q.get():
                print "kk"
            k += 1
        for i in range(2000, 4000):
            q.put(i)
        while not q.empty():
            q.get(),
        print time.time() - t
    test(Queue())
    test(Queue2())

