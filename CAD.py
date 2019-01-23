import numpy as np
from pointerNetwork import *
from programGraph import *
from CNN import *

import random


RESOLUTION = 32

import torch
import torch.nn as nn

class CSG():
    lexicon = ['+','-','t','c','r'] + list(map(str,range(64)))

    def __init__(self):
        self._rendering = None

    def __ne__(self, o): return not (self == o)

    def execute(self):
        if self._rendering is None: self._rendering = self.render()
        return self._rendering
    
    def render(self, w=None, h=None):
        w = w or RESOLUTION
        h = h or RESOLUTION
        
        a = np.zeros((w,h))
        for x in range(w):
            for y in range(h):
                if (x,y) in self:
                    a[x,y] = 1
        return a

    @staticmethod
    def parseLine(tokens):
        if len(tokens) == 0: return None
        if tokens[0] == '+':
            if len(tokens) != 3: return None
            if not isinstance(tokens[2],CSG): return None
            if not isinstance(tokens[1],CSG): return None
            return Union(tokens[0],tokens[1])
        if tokens[0] == '-':
            if len(tokens) != 3: return None
            if not isinstance(tokens[2],CSG): return None
            if not isinstance(tokens[1],CSG): return None
            return Difference(tokens[0],tokens[1])
        if tokens[0] == 't':
            if len(tokens) != 4: return None
            if not isinstance(tokens[3],CSG): return None
            try:
                return Translation((int(tokens[1]),int(tokens[2])),
                                   tokens[3])
            except: return None
        if tokens[0] == 'r':
            if len(tokens) != 3: return None
            try:
                return Rectangle(int(tokens[1]),
                                 int(tokens[2]))
            except: return None
        if tokens[0] == 'c':
            if len(tokens) != 2: return None
            try: return Circle(int(tokens[1]))
            except: return None
        return None
        

class Rectangle(CSG):
    def __init__(self, w, h):
        super(Rectangle, self).__init__()
        self.w = w
        self.h = h

    def children(self): return []

    def __eq__(self, o):
        return isinstance(o, Rectangle) and o.w == self.w and o.h == self.h

    def __hash__(self):
        return hash(('r',self.w,self.h))

    def serialize(self):
        return ('r',str(self.w),str(self.h))

    def __contains__(self, p):
        return p[0] >= 0 and p[1] >= 0 and \
            p[0] < self.w and p[1] < self.h

class Circle(CSG):
    def __init__(self, r):
        super(Circle, self).__init__()
        self.r = r

    def children(self): return []

    def __eq__(self, o):
        return isinstance(o, Circle) and o.r == self.r
    def __hash__(self):
        return hash(('c', str(self.r)))

    def serialize(self):
        return ('c',str(self.r))

    def __contains__(self, p):
        return p[0]*p[0] + p[1]*p[1] <= self.r

class Translation(CSG):
    def __init__(self, p, child):
        super(Translation, self).__init__()
        self.v = p
        self.child = child

    def children(self): return [self.child]

    def serialize(self):
        return ('t',str(self.v[0]),str(self.v[1]),self.child)

    def __eq__(self, o):
        return isinstance(o, Translation) and o.v == self.v and self.child == o.child

    def __hash__(self):
        return hash(('t', self.v, self.child))

    def __contains__(self, p):
        p = (p[0] - self.v[0],
             p[1] - self.v[1])
        return p in self.child

class Union(CSG):
    def __init__(self, a, b):
        super(Union, self).__init__()
        self.elements = frozenset({a,b})

    def children(self): return list(self.elements)

    def serialize(self):
        return ('+',list(self.elements)[0],list(self.elements)[1])

    def __eq__(self, o):
        return isinstance(o, Union) and o.elements == self.elements

    def __hash__(self):
        return hash(('u', self.elements))

    def __contains__(self, p):
        return any( p in e for e in self.elements )

class Difference(CSG):
    def __init__(self, a, b):
        super(Difference, self).__init__()
        self.a, self.b = a, b
        
    def children(self): return [self.a, self.b]

    def serialize(self):
        return ('-',self.a,self.b)

    def __eq__(self, o):
        return isinstance(o, Difference) and self.a == o.a and self.b == o.b
    
    def __contains__(self, a, b):
        return p in self.a and (not (p in self.b))


"""Neural networks"""
class ObjectEncoder(CNN):
    def __init__(self):
        super(ObjectEncoder, self).__init__(channels=2,
                                            inputImageDimension=RESOLUTION)

    def forward(self, spec, obj):
        return super(ObjectEncoder, self).forward(np.stack([spec, obj]))
        

class SpecEncoder(CNN):
    def __init__(self):
        super(SpecEncoder, self).__init__(channels=1,
                                          inputImageDimension=RESOLUTION)


"""Training"""
def randomScene(resolution=64):
    def quadrilateral():
        w = random.choice(range(int(resolution/2))) + 3
        h = random.choice(range(int(resolution/2))) + 3
        x = random.choice(range(resolution - w))
        y = random.choice(range(resolution - h))
        return Translation((x,y),
                           Rectangle(w,h))

    def circular():
        r = random.choice(range(int(resolution/4))) + 2
        x = random.choice(range(resolution - r*2)) + r
        y = random.choice(range(resolution - r*2)) + r
        return Translation((x,y),
                           Circle(r))
    s = None
    for _ in range(random.choice([1])):
        o = quadrilateral() if random.choice([True,False]) else circular()
        if s is None: s = o
        else: s = Union(s,o)
    return s

def trainCSG(m, getProgram, maxSteps=100000):
    print("cuda?",m.use_cuda)
    optimizer = torch.optim.Adam(m.parameters(), lr=0.001, eps=1e-3, amsgrad=True)

    s = getProgram()
    g = ProgramGraph.fromRoot(s)
    print(f"Training on graph:\n{g.prettyPrint()}")

    for iteration in range(maxSteps):
        l = m.gradientStepTrace(optimizer, s.execute(), g)
        print(f"Trace loss {sum(l)}\tAverage per-move loss {sum(l)/len(l)}")
        if sum(l)/len(l) < 1.:
            print(f"1-move loss is small! Trying a sample...")
            sample = m.sample(s.execute(), maxMoves=5)
            if sample is None: print("Failed to get a correct sample")
            else: print(sample.prettyPrint())
        
    

if __name__ == "__main__":
    m = ProgramPointerNetwork(ObjectEncoder(), SpecEncoder(), CSG)
    trainCSG(m, lambda: randomScene())
