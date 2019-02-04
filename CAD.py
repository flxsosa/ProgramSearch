import pickle
import numpy as np
from pointerNetwork import *
from programGraph import *
from SMC import *
from MCTS import *
from ForwardSample import *
from CNN import *

import time
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

    def IoU(self, other):
        return (self.execute()*other.execute()).sum()/(self.execute() + other.execute() - self.execute()*other.execute()).sum()
    
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
            if tokens[1] == tokens[2]: return None
            return Union(tokens[1],tokens[2])
        if tokens[0] == '-':
            if len(tokens) != 3: return None
            if not isinstance(tokens[2],CSG): return None
            if not isinstance(tokens[1],CSG): return None
            return Difference(tokens[1],tokens[2])
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
        if isinstance(obj, list): # batched - expect a single spec and multiple objects
            spec = np.repeat(spec[np.newaxis,:,:],len(obj),axis=0)
            obj = np.stack(obj)
            return super(ObjectEncoder, self).forward(np.stack([spec, obj],1))
        else: # not batched
            return super(ObjectEncoder, self).forward(np.stack([spec, obj]))
        

class SpecEncoder(CNN):
    def __init__(self):
        super(SpecEncoder, self).__init__(channels=1,
                                          inputImageDimension=RESOLUTION)


"""Training"""
def randomScene(resolution=32, maxShapes=3, minShapes=1, verbose=False):
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
    numberOfShapes = 0
    desiredShapes = random.choice(range(minShapes, 1 + maxShapes))
    while numberOfShapes < desiredShapes:
        o = quadrilateral() if random.choice([True,False]) else circular()
        if s is None: s = o
        else:
            if (s.execute()*o.execute()).sum() > 0.5: continue
            s = Union(s,o)
        numberOfShapes += 1
    if verbose:
        import matplotlib.pyplot as plot
        print(ProgramGraph.fromRoot(s).prettyPrint())
        plot.imshow(s.execute())
        plot.show()
    
    return s

def trainCSG(m, getProgram, trainTime=None, checkpoint=None):
    print("cuda?",m.use_cuda)
    assert checkpoint is not None, "must provide a checkpoint path to export to"
    
    optimizer = torch.optim.Adam(m.parameters(), lr=0.001, eps=1e-3, amsgrad=True)
    
    startTime = time.time()
    reportingFrequency = 100
    totalLosses = []
    movedLosses = []
    distanceLosses = []
    iteration = 0

    while trainTime is None or time.time() - startTime < trainTime:
        s = getProgram()
        g = ProgramGraph.fromRoot(s)
        l,dl = m.gradientStepTrace(optimizer, s.execute(), g)
        totalLosses.append(sum(l))
        movedLosses.append(sum(l)/len(l))
        distanceLosses.append(sum(dl)/len(dl))
        if sum(l) < 2. and iteration%5 == 0:
            print(f"loss is small! Trying a sample. For reference, here is the goal graph:\n{g.prettyPrint()}")
            sample = m.sample(s.execute(), maxMoves=5)
            if sample is None: print("Failed to get a correct sample")
            else: print(f"Got the following sample:\n{sample.prettyPrint()}")
            print()

        if iteration%reportingFrequency == 0:
            print(f"\n\nAfter {iteration} gradient steps...\n\tTrace loss {sum(totalLosses)/len(totalLosses)}\t\tMove loss {sum(movedLosses)/len(movedLosses)}\t\tdistance loss {sum(distanceLosses)/len(distanceLosses)}\n{iteration/(time.time() - startTime)} grad steps/sec")
            totalLosses = []
            movedLosses = []
            distanceLosses = []
            with open(checkpoint,"wb") as handle:
                pickle.dump(m, handle)

        iteration += 1

def testCSG(m, getProgram):
    def reward(s, g):
        if len(g) == 0: return 0
        return max((s*o).sum()/(s + o - s*o).sum() for o_ in g.objects() for o in [o_.execute()] )
    searchers = [ForwardSample(m, defaultTimeout=5., reward=reward),
                 SMC(m, particles=100),
                 MCTS(m, beamSize=50, rolloutDepth=10, reward=reward, defaultTimeout=5.),
                 MCTS(m, beamSize=10, rolloutDepth=10, reward=reward, defaultTimeout=5.),
                 MCTS(m, beamSize= 0, rolloutDepth=10, reward=reward, defaultTimeout=5.)]
    # Map from index of searcher to list of rewards
    rewards = [[] for _ in searchers]
                 
    
    for _ in range(100):
        spec = getProgram()
        print("Trying to explain the program:")
        print(ProgramGraph.fromRoot(spec).prettyPrint())
        print()
        for i, s in enumerate(searchers):
            samples = s.infer(spec.execute())
            if not isinstance(samples, list): samples = [samples]
            rewards[i].append(max(reward(spec.execute(), sample) for sample in samples ))
            print(f"{s}\t{rewards[i][-1]}")
        print()
    for searcher, rs in zip(searchers, rewards):
        print(f"Search algorithm {searcher}")
        print(f"Rewards {rs}")
        print(f"Average {sum(rs)/len(rs)}\tMedian {list(sorted(rs))[len(rs)//2]}\tHits {sum(r > 0.99 for r in rs )}")
        if isinstance(searcher, MCTS):
            print(f"MCTS spent {searcher.beamTime} beaming, {searcher.rollingTime} doing rollouts, {searcher.distanceTime} calculating value")
        
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("mode", choices=["train","test"])
    parser.add_argument("--checkpoint", default="checkpoints/CSG.pickle")
    parser.add_argument("--maxShapes", default=2,
                            type=int)
    parser.add_argument("--trainTime", default=None, type=float,
                        help="Time in hours to train the network")
    parser.add_argument("--attention", default=1, type=int,
                        help="Number of rounds of self attention to perform upon objects in scope")
    parser.add_argument("--heads", default=2, type=int,
                        help="Number of attention heads")
    parser.add_argument("--hidden", "-H", type=int, default=256,
                        help="Size of hidden layers")
    arguments = parser.parse_args()

    if arguments.mode == "train":
        m = ProgramPointerNetwork(ObjectEncoder(), SpecEncoder(), CSG,
                                  attentionRounds=arguments.attention,
                                  heads=arguments.heads,
                                  H=arguments.hidden)
        trainCSG(m, lambda: randomScene(maxShapes=arguments.maxShapes),
                 trainTime=arguments.trainTime*60*60 if arguments.trainTime else None,
                 checkpoint=arguments.checkpoint)
    elif arguments.mode == "test":
        with open(arguments.checkpoint,"rb") as handle:
            m = pickle.load(handle)
        testCSG(m, lambda: randomScene(maxShapes=arguments.maxShapes, minShapes=arguments.maxShapes))
