import heapq

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

class LayerNorm(Module):
    "Adapted from http://nlp.seas.harvard.edu/2018/04/03/attention.html"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

        self.finalize()

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


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

def saveMatrixAsImage(m, fn):
    import scipy.misc
    scipy.misc.imsave(fn, m)

def showMatrixAsImage(*m):
    import matplotlib.pyplot as plot

    n = len(m)
    f,a = plot.subplots(n,1)

    for i,_m in enumerate(m):
        a[i].imshow(_m)

    plot.show()

NEGATIVEINFINITY = float('-inf')
