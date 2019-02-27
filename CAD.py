import pickle
import numpy as np

from API import *

from randomSolver import *
from pointerNetwork import *
from programGraph import *
from SMC import *
from ForwardSample import *
from MCTS import MCTS
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

    def __repr__(self):
        return str(self)

    def __ne__(self, o): return not (self == o)

    def execute(self):
        if self._rendering is None: self._rendering = self.render()
        return self._rendering

    def IoU(self, other):
        if isinstance(other, CSG): other = other.execute()
        return (self.execute()*other).sum()/(self.execute() + other - self.execute()*other).sum()
    
    def render(self, w=None, h=None):
        w = w or RESOLUTION
        h = h or RESOLUTION
        
        a = np.zeros((w,h))
        for x in range(w):
            for y in range(h):
                if (x,y) in self:
                    a[x,y] = 1
        return a

# The type of CSG's
tCSG = BaseType(CSG)

class Rectangle(CSG):
    token = 'r'
    type = arrow(integer(0, RESOLUTION - 1), integer(0, RESOLUTION - 1), tCSG)
    
    def __init__(self, w, h):
        super(Rectangle, self).__init__()
        self.w = w
        self.h = h

    def toTrace(self): return [self]

    def __str__(self):
        return f"(r {self.w} {self.h})"

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
    type = arrow(integer(0, RESOLUTION - 1), tCSG)
    
    def __init__(self, r):
        super(Circle, self).__init__()
        self.r = r

    def toTrace(self): return [self]
        
    def __str__(self):
        return f"(c {self.r})"

    def children(self): return []

    def __eq__(self, o):
        return isinstance(o, Circle) and o.r == self.r
    def __hash__(self):
        return hash(('c', str(self.r)))

    def serialize(self):
        return (self.__class__.token, self.r)

    def __contains__(self, p):
        return p[0]*p[0] + p[1]*p[1] <= self.r*self.r

class Translation(CSG):
    token = 't'
    type = arrow(integer(0, RESOLUTION - 1), integer(0, RESOLUTION - 1), tCSG, tCSG)
    
    def __init__(self, x, y, child):
        super(Translation, self).__init__()
        self.v = (x, y)
        self.child = child

    def toTrace(self): return self.child.toTrace() + [self]
    
    def __str__(self):
        return f"(t {self.v} {self.child})"

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
    type = arrow(tCSG, tCSG, tCSG)
    
    def __init__(self, a, b):
        super(Union, self).__init__()
        self.elements = [a,b]

    def toTrace(self):
        return self.elements[0].toTrace() + self.elements[1].toTrace() + [self]

    def __str__(self):
        return f"(+ {str(self.elements[0])} {str(self.elements[1])})"

    def children(self): return self.elements

    def serialize(self):
        return ('+',list(self.elements)[0],list(self.elements)[1])

    def __eq__(self, o):
        return isinstance(o, Union) and tuple(o.elements) == tuple(self.elements)

    def __hash__(self):
        return hash(('u', tuple(self.elements)))

    def __contains__(self, p):
        return any( p in e for e in self.elements )

class Difference(CSG):
    token = '-'
    type = arrow(tCSG, tCSG, tCSG)
    
    def __init__(self, a, b):
        super(Difference, self).__init__()
        self.a, self.b = a, b

    def toTrace(self):
        return self.a.toTrace() + self.b.toTrace() + [self]

    def __str__(self):
        return f"(- {self.a} {self.b})"
        
    def children(self): return [self.a, self.b]

    def serialize(self):
        return ('-',self.a,self.b)

    def __eq__(self, o):
        return isinstance(o, Difference) and self.a == o.a and self.b == o.b

    def __hash__(self):
        return hash(('-', hash(self.a), hash(self.b)))
    
    def __contains__(self, p):
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
def randomScene(resolution=32, maxShapes=3, minShapes=1, verbose=False, export=None):
    dc = 8 # number of distinct coordinates
    def quadrilateral():
        choices = [c
                   for c in range(resolution//(dc*2), resolution, resolution//dc) ]
        w = random.choice([2,5])
        h = random.choice([2,5])
        x = random.choice(choices)
        y = random.choice(choices)
        return Translation(x,y,
                           Rectangle(w,h))

    def circular():
        r = random.choice([2,4])
        choices = [c
                   for c in range(resolution//(dc*2), resolution, resolution//dc) ]
        x = random.choice(choices)
        y = random.choice(choices)
        return Translation(x,y,
                           Circle(r))
    s = None
    numberOfShapes = 0
    desiredShapes = random.choice(range(minShapes, 1 + maxShapes))
    for _ in range(desiredShapes):
        o = quadrilateral() if random.choice([True,False]) else circular()
        if s is None: s = o
        else:
            if (s.execute()*o.execute()).sum() > 0.5: continue
            s = Union(s,o)
        numberOfShapes += 1
    if verbose:
        print(s)
        print(ProgramGraph.fromRoot(s, oneParent=True).prettyPrint())
        import matplotlib.pyplot as plot
        plot.imshow(s.execute())
        plot.show()
    if export:
        import matplotlib.pyplot as plot
        plot.imshow(s.execute())
        plot.savefig(export)
    
    return s

def trainCSG(m, getProgram, trainTime=None, checkpoint=None):
    print("cuda?",m.use_cuda)
    assert checkpoint is not None, "must provide a checkpoint path to export to"
    
    optimizer = torch.optim.Adam(m.parameters(), lr=0.001, eps=1e-3, amsgrad=True)
    
    startTime = time.time()
    reportingFrequency = 100
    totalLosses = []
    movedLosses = []
    iteration = 0

    while trainTime is None or time.time() - startTime < trainTime:
        s = getProgram()
        l = m.gradientStepTrace(optimizer, s.execute(), s.toTrace())
        totalLosses.append(sum(l))
        movedLosses.append(sum(l)/len(l))

        if iteration%reportingFrequency == 0:
            print(f"\n\nAfter {iteration} gradient steps...\n\tTrace loss {sum(totalLosses)/len(totalLosses)}\t\tMove loss {sum(movedLosses)/len(movedLosses)}\n{iteration/(time.time() - startTime)} grad steps/sec")
            totalLosses = []
            movedLosses = []
            with open(checkpoint,"wb") as handle:
                pickle.dump(m, handle)

        iteration += 1

def testCSG(m, getProgram, timeout, export):
    oneParent = m.oneParent
    solvers = [# RandomSolver(dsl),
               # MCTS(m, reward=lambda l: 1. - l),
               # SMC(m),
               ForwardSample(m, maximumLength=18)]
    loss = lambda spec, program: 1-max( o.IoU(spec) for o in program.objects() ) if len(program) > 0 else 1.

    testResults = [[] for _ in solvers]

    for _ in range(30):
        spec = getProgram()
        print("Trying to explain the program:")
        print(ProgramGraph.fromRoot(spec, oneParent=oneParent).prettyPrint())
        print()
        for n, solver in enumerate(solvers):
            testSequence = solver.infer(spec.execute(), loss, timeout)
            testResults[n].append(testSequence)
            for result in testSequence:
                print(f"After time {result.time}, achieved loss {result.loss} w/")
                print(result.program.prettyPrint())
                print()

    plotTestResults(testResults, timeout,
                    defaultLoss=1.,
                    names=[# "MCTS","SMC", 
                           "FS"],
                    export=export)

def plotTestResults(testResults, timeout, defaultLoss=None,
                    names=None, export=None):
    import matplotlib.pyplot as plot

    def averageLoss(n, T):
        results = testResults[n] # list of list of results, one for each test case
        # Filter out results that occurred after time T
        results = [ [r for r in rs if r.time <= T]
                    for rs in results ]
        losses = [ min([defaultLoss] + [r.loss for r in rs]) for rs in results ]
        return sum(losses)/len(losses)

    plot.figure()
    plot.xlabel('Time')
    plot.ylabel('Average Loss')

    for n in range(len(testResults)):
        xs = list(np.arange(0,timeout,0.1))
        plot.plot(xs, [averageLoss(n,x) for x in xs],
                  label=names[n])
    plot.legend()
    if export:
        plot.savefig(export)
    else:
        plot.show()
        
        
    
        
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("mode", choices=["train","test","demo"])
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
    parser.add_argument("--oneParent", default=False, action='store_true')
    arguments = parser.parse_args()

    if arguments.mode == "demo":
        for n in range(100):
            randomScene(export=f"/tmp/CAD_{n}.png",maxShapes=arguments.maxShapes)
        import sys
        sys.exit(0)
        
            

    if arguments.mode == "train":
        m = ProgramPointerNetwork(ObjectEncoder(), SpecEncoder(), dsl,
                                  oneParent=arguments.oneParent,
                                  attentionRounds=arguments.attention,
                                  heads=arguments.heads,
                                  H=arguments.hidden)
        trainCSG(m, lambda: randomScene(maxShapes=arguments.maxShapes),
                 trainTime=arguments.trainTime*60*60 if arguments.trainTime else None,
                 checkpoint=arguments.checkpoint)
    elif arguments.mode == "test":
        with open(arguments.checkpoint,"rb") as handle:
            m = pickle.load(handle)
        testCSG(m,
                lambda: randomScene(maxShapes=arguments.maxShapes, minShapes=arguments.maxShapes), arguments.timeout,
                export=f"figures/CAD_{arguments.maxShapes}_shapes.png")
