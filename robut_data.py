#robut_data.py
from multiprocessing import Queue, Process

#from pathos.multiprocessing import Queue, Process 
#parallel data gen

def get_supervised_batchsize(fn, batchsize=200):
    #takes a generation function and outputs lists of optimal size
    remainder = [], []
    while True:
        preS, preA = remainder
        S, A = fn()
        S, A = preS+S, preA+A
        ln = len(S)

        if ln > batchsize:
            yield S[:batchsize], A[:batchsize]
            remainder = S[batchsize:], A[batchsize:]
            continue
        elif ln < batchsize:
            remainder = S, A
            continue
        elif ln == batchsize:
            yield S, A
            remainder = [], []
            continue
        else: assert 0, "uh oh, not a good place"

class GenData:
    def __init__(self, fn, n_processes=20, max_size=100, batchsize=200):
        ##what needs to happen:
        def consumer(Q):
            iterator = get_supervised_batchsize(fn, batchsize=batchsize)
            while True:
                try:
                    # get a new message
                    size = Q.qsize()
                    #print(size)
                    if size < max_size:
                        # process the data
                        ret = next(iterator)
                        Q.put( ret )
                except ValueError as e:
                    print("I think you closed the thing while it was running, but that's okay")
                    break
                except Exception as e:
                    print("error!", e)
                    break

        self.Q = Queue()
        print("started queue ...")

        # instantiate workers
        self.workers = [Process(target=consumer, args=(self.Q,))
               for i in range(n_processes)]

        for w in self.workers:
            w.start()
        print("started parallel workers, ready to work!")

    def batchIterator(self):
        while True:
            yield self.Q.get()
        #yield from get_supervised_batchsize(self.Q.get, batchsize=batchsize) #is this a slow way of doing this??
        
    def kill(self):
        #KILL stuff
        # tell all workers, no more data (one msg for each)
        # join on the workers
        for w in self.workers:
            try:
                w.close() #this will cause a valueError apparently??
            except ValueError:
                print("killed a worker")
                continue

def makeTestdata(synth=True, challenge=False, max_num_ex=4, include_const=False):
    import sys
    import os
    sys.path.append(os.path.abspath('./'))
    sys.path.append(os.path.abspath('./ec'))
    from makeTextTasks import makeTasks, loadPBETasks
    from type import arrow, tlist, tcharacter
    tasks = []
    if synth:
        tasks = makeTasks() 
    if challenge:
        challenge_tasks, _ = loadPBETasks()
        tasks = tasks + challenge_tasks

    tasklist = []
    for task in tasks:
        if task.request == arrow(tlist(tcharacter), tlist(tcharacter)):
            if include_const: 
                if not task.stringConstants==[]: continue
            inputs = [''.join(x[0]) for x, _ in task.examples[:max_num_ex]]
            outputs = [''.join(y) for _, y in task.examples[:max_num_ex]]
            tasklist.append( (inputs, outputs) )

    return tasklist


if __name__ == '__main__':
    tasks = makeTestdata(synth=True, challenge=True, max_num_ex=4)

    max_len = 0
    for task in tasks:
        inputs, outputs = task
        for i in inputs:   
            if len(i) > max_len: max_len = len(i)
        for o in outputs:
            if len(o) > max_len: max_len = len(o)

    # from ROBUT import get_supervised_sample
    # import time

    # fn = get_supervised_sample
    # print("normal, unparallelized:")
    # t = time.time()
    # for i, (S, A) in enumerate(get_supervised_batchsize(fn, batchsize=2000)):
    #     assert len(S) == 2000
    #     if i >= 40 - 1: break
    # tot = time.time() - t
    # print(f"unparallelized average time for 20 batches: {tot/20} sec ", flush=True)


    # dataqueue = GenData(fn, n_processes=10, max_size=10000)

    # print("waiting 5 seconds ...")
    # time.sleep(5)

    # t = time.time()
    # for i, (S, A) in enumerate( dataqueue.batchIterator(batchsize=2000) ):
    #     assert len(S) == 2000
    #     if i >= 40 - 1: break
    # tot = time.time() - t
    # print(f"parallelized average time for 20 batches: {tot/20} sec ")

    # dataqueue.kill()


