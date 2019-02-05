import pickle
import numpy as np

from API import *

from pointerNetwork import *
from programGraph import *
from SMC import *
from ForwardSample import *
from CNN import *

import time
import random


RESOLUTION = 32

import torch
import torch.nn as nn

class CSG(Program):
    lexicon = ['+','-','t','c','r'] + list(range(RESOLUTION))

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


class Rectangle(CSG):
    token = 'r'
    type = CSG
    argument_types = (int, int)
    
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
        return (self.__class__.token, self.w, self.h)

    def __contains__(self, p):
        return p[0] >= 0 and p[1] >= 0 and \
            p[0] < self.w and p[1] < self.h

class Circle(CSG):
    token = 'c'
    type = CSG
    argument_types = (int,)
    
    def __init__(self, r):
        super(Circle, self).__init__()
        self.r = r

    def children(self): return []

    def __eq__(self, o):
        return isinstance(o, Circle) and o.r == self.r
    def __hash__(self):
        return hash(('c', str(self.r)))

    def serialize(self):
        return (self.__class__.token, self.r)

    def __contains__(self, p):
        return p[0]*p[0] + p[1]*p[1] <= self.r

class Translation(CSG):
    token = 't'
    type = CSG
    argument_types = (int,)
    
    def __init__(self, p, child):
        super(Translation, self).__init__()
        self.v = p
        self.child = child

    def children(self): return [self.child]

    def serialize(self):
        return ('t', self.v[0], self.v[1], self.child)

    def __eq__(self, o):
        return isinstance(o, Translation) and o.v == self.v and self.child == o.child

    def __hash__(self):
        return hash(('t', self.v, self.child))

    def __contains__(self, p):
        p = (p[0] - self.v[0],
             p[1] - self.v[1])
        return p in self.child

class Union(CSG):
    token = '+'
    type = CSG
    argument_types = (CSG, CSG)
    
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
    token = '-'
    type = CSG
    argument_types = (CSG, CSG)
    
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

dsl = DSL([Rectangle, Circle, Translation, Union, Difference],
          lexicon=CSG.lexicon)

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
def randomScene(resolution=32, maxShapes=3, verbose=False):
    random.seed(0)
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
    desiredShapes = random.choice(range(1, 1 + maxShapes))
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

        if iteration%reportingFrequency == 0:
            print(f"\n\nAfter {iteration} gradient steps...\n\tTrace loss {sum(totalLosses)/len(totalLosses)}\t\tMove loss {sum(movedLosses)/len(movedLosses)}\t\tdistance loss {sum(distanceLosses)/len(distanceLosses)}\n{iteration/(time.time() - startTime)} grad steps/sec")
            totalLosses = []
            movedLosses = []
            distanceLosses = []
            with open(checkpoint,"wb") as handle:
                pickle.dump(m, handle)

        iteration += 1

def testCSG(m, getProgram, timeout):
    solvers = [SMC(m), ForwardSample(m)]
    loss = lambda spec, program: -max( o.IoU(spec) for o in program.objects() ) if len(program) > 0 else -1.

    testResults = [[] for _ in solvers]

    for _ in range(3):
        spec = getProgram()
        print("Trying to explain the program:")
        print(ProgramGraph.fromRoot(spec).prettyPrint())
        print()
        for n, solver in enumerate(solvers):
            testResults[n].append(solver.infer(spec.execute(), loss, timeout))

    plotTestResults(testResults, timeout,
                    ["SMC", "FS"])

def plotTestResults(testResults, timeout,
                    names):
    import matplotlib.pyplot as plot

    def averageLoss(n, T):
        results = testResults[n] # list of list of results, one for each test case
        # Filter out results that occurred after time T
        results = [ [r for r in rs if r.time <= T]
                    for rs in results ]
        losses = [ min(r.loss for r in rs) for rs in results ]
        return sum(losses)/len(losses)

    plot.figure()
    plot.xlabel('Time')
    plot.ylabel('Average Loss')

    for n in range(len(testResults)):
        xs = list(np.arange(0,timeout,0.1))
        plot.plot(xs, [averageLoss(n,x) for x in xs],
                  label=names[n])
    plot.legend()
    plot.show()
        
        
    
        
    

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
    parser.add_argument("--timeout", default=5, type=float,
                        help="Test time maximum timeout")
    arguments = parser.parse_args()

    if arguments.mode == "train":
        m = ProgramPointerNetwork(ObjectEncoder(), SpecEncoder(), dsl,
                                  attentionRounds=arguments.attention,
                                  heads=arguments.heads,
                                  H=arguments.hidden)
        trainCSG(m, lambda: randomScene(maxShapes=arguments.maxShapes),
                 trainTime=arguments.trainTime*60*60 if arguments.trainTime else None,
                 checkpoint=arguments.checkpoint)
    elif arguments.mode == "test":
        with open(arguments.checkpoint,"rb") as handle:
            m = pickle.load(handle)
        testCSG(m, lambda: randomScene(maxShapes=arguments.maxShapes), arguments.timeout)
