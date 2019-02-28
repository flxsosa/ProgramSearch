import heapq
import random

import torch
import torch.nn as nn

class Module(nn.Module):
    """Wrapper over torch Module class that handles GPUs elegantly"""
    def __init__(self):
        super(Module, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def tensor(self, array):
        return self.device(torch.tensor(array))
    def device(self, t):
        if self.use_cuda: return t.cuda()
        else: return t
    def finalize(self):
        if self.use_cuda: self.cuda()


class PQ(object):
    """why the fuck does Python not wrap this in a class
    This is a priority queue, a.k.a. max heap"""

    def __init__(self):
        self.h = []
        self.index2value = {}
        self.nextIndex = 0

    def push(self, priority, v):
        self.index2value[self.nextIndex] = v
        heapq.heappush(self.h, (-priority, self.nextIndex))
        self.nextIndex += 1

    def popMaximum(self):
        i = heapq.heappop(self.h)[1]
        v = self.index2value[i]
        del self.index2value[i]
        return v

    def __iter__(self):
        for _, v in self.h:
            yield self.index2value[v]

    def __len__(self): return len(self.h)

def getArgument(requestedType, graph):
    '''
    Returns arguments of a given type
    '''
    # Return integers for integer types
    if requestedType.isInteger:
        return random.choice(range(requestedType.lower, requestedType.upper + 1))

    # Otherwise, return object types
    choices = [o for o in graph.objects() if requestedType.instance(o)]
    # If any are found, return them
    if choices: 
        return random.choice(choices)
    # Otherwise, return None
    else: 
        return None