import heapq

import pickle

import numpy as np

import torch
import torch.nn as nn

import math

def rotationMatrix(x,y,z):
    x = math.pi*x/180.
    y = math.pi*y/180.
    z = math.pi*z/180.
    sx = math.sin(x)
    cx = math.cos(x)
    sz = math.sin(z)
    cz = math.cos(z)
    sy = math.sin(y)
    cy = math.cos(y)
    
    rx = np.array([[1,0,0],
                   [0,cx,-sx],
                   [0,sx, cx]])
    ry = np.array([[cy,0,sy],
                   [0 ,1,0],
                   [-sy,0,cy]])
    rz = np.array([[cz,-sz,0],
                   [sz,cz,0],
                   [0,0,1]])
    return rz@ry@rx

class Module(nn.Module):
    """Wrapper over torch Module class that handles GPUs elegantly"""
    def __init__(self):
        super(Module, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def tensor(self, array):
        """Convert something to a tensor if it is not already a tensor (does nothing otherwise), and then moves to device"""
        if type(array) == torch.Tensor: return self.device(array)
        return self.device(torch.tensor(array))
    def device(self, t):
        if t.dtype == torch.float64: t = t.float()
        if self.use_cuda: return t.cuda()
        return t
    def finalize(self):
        if self.use_cuda: self.cuda()

class IdentityLayer(Module):
    def __init__(self):
        super(IdentityLayer, self).__init__()
        self.finalize()
    def forward(self, x): return x

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

def binary_cross_entropy(y,t, epsilon=10**-10, average=True):
    """y: tensor of size B, elements <= 0. each element is a log probability.
    t: tensor of size B, elements in [0,1]. intended target.
    returns: 1/B * - \sum_b t*y + (1 - t)*(log(1 - e^y + epsilon))"""

    B = y.size(0)
    log_yes_probability = y
    log_no_probability = torch.log(1 - y.exp() + epsilon)
    assert torch.all(log_yes_probability <= 0.)
    assert torch.all(log_no_probability <= 0.)
    correctYes = t
    correctNo = 1 - t
    ce = -(correctYes*log_yes_probability + correctNo*log_no_probability).sum()
    if average: ce = ce/B
    return ce

    
def load_checkpoint(fn):
    """wrapper over torch.load which places checkpoints on the CPU if a GPU is not available"""
    print(f"Loading checkpoint {fn}. cuda? {torch.cuda.is_available()}")
    m = torch.load(fn, map_location=torch.device('cpu'))
    print(f"loaded {m}")
    if torch.cuda.is_available():
        print(f"moving to GPU...")
        m.cuda()
        print(f"{m} has been moved!")
    return m

class PointerDictionary:
    def __init__(self):
        """dict-like object that uses pointer equality. O(N) access."""
        self.d = []

    def __contains__(self,o):
        return any( op is o for op,_ in self.d  )
    def __getitem__(self,o):
        for op,v in self.d:
            if o is op: return v
        raise KeyError(o)
    def __delitem__(self,k):
        if not (k in self): raise KeyError(o)
        self.d = [(kp,v) for kp,v in self.d if not (kp is k) ]
    def __setitem__(self,k,v):
        if k in self: del self[k]
        self.d.append((k,v))
    
