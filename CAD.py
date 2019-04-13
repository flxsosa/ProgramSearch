import os
import pickle
import numpy as np

from API import *

from a2c import *
from randomSolver import *
from pointerNetwork import *
from programGraph import *
from SMC import *
from ForwardSample import *
from MCTS import MCTS
from beamSearch import *
from CNN import *

import time
import random


RESOLUTION = 32

import torch
import torch.nn as nn
import torch.nn.functional as F

class BadCSG(Exception): pass

class CSG(Program):
    lexicon = ['+','-','t','c','r','tr','tc'] + list(range(RESOLUTION))

    def __init__(self):
        self._rendering = None

    def clearRendering(self):
        self._rendering = None
        for c in self.children(): c.clearRendering()

    def heatMapTarget(self):
        hm = np.zeros((RESOLUTION,RESOLUTION,3)) > 1.
        for c in self.children():
            hm = np.logical_or(hm,c.heatMapTarget())
        return hm


    def __repr__(self):
        return str(self)

    def __eq__(self,o):
        return self.serialize() == o.serialize()

    def __ne__(self, o): return not (self == o)

    def __hash__(self): return hash(self.serialize())

    def execute(self):
        if self._rendering is None: self._rendering = self.render()
        return self._rendering

    def IoU(self, other):
        if isinstance(other, CSG): other = other.execute()
        return (self.execute()*other).sum()/(self.execute() + other - self.execute()*other).sum()
    
    def render(self, r=None):
        r = r or RESOLUTION
        
        a = np.zeros((r,r,r))
        for x in range(r):
            for y in range(r):
                for z in range(r):
                    if (RESOLUTION*x/r, RESOLUTION*y/r, RESOLUTION*z/r) in self:
                        a[x,y,z] = 1
        return a

    def removeDeadCode(self): return self

    def depthMaps(self, r=None):
        """Returns six depth maps"""
        r = r or RESOLUTION
        vs = self.render(r=r)

        span = np.arange(0.,1.,1./r)

        x, y, z = np.indices((r,r,r))/float(r)
        
        maps = []

        # collapsing Z (front)
        ds = np.zeros((r,r,r))
        ds[:,:,:] = span

        maps.append(np.amax((vs*z), axis=2))

        # collapsing Z (behind)
        ds[:,:,:] = np.flip(span)
        maps.append(np.amax((vs*np.flip(z,2)), axis=2))

        maps = [m
                for n,i in enumerate(np.indices((r,r,r))/float(r))
                for m in [np.amax(vs * i, axis=n),
                          np.amax(vs * np.flip(i,n), axis=n)] ]

        return maps
        

    def show(self):
        """Open up a new window and show the CSG"""
        import matplotlib.pyplot as plot
        from mpl_toolkits.mplot3d import Axes3D
        columns = 2
        f = plot.figure()

        maps = self.depthMaps(32)

        a = f.add_subplot(1, 1 + len(maps), 1,
                          projection='3d')
        a.voxels(self.execute(), edgecolor='k')

        for i,m in enumerate(maps):
            a = f.add_subplot(1, 1 + len(maps), 2 + i)
            a.imshow(1 - m, cmap='Greys',
                     vmin=0., vmax=1.)
            a.get_xaxis().set_visible(False)
            a.get_yaxis().set_visible(False)
        
        plot.show()
        
                
# The type of CSG's
tCSG = BaseType(CSG)

class Cuboid(CSG):
    token = 'cuboid'
    type = arrow(integer(0, RESOLUTION - 1), integer(0, RESOLUTION - 1), integer(0, RESOLUTION - 1),
                 integer(0, RESOLUTION - 1), integer(0, RESOLUTION - 1), integer(0, RESOLUTION - 1),
                 tCSG)
    
    def __init__(self, x0, y0, z0, x1, y1, z1):
        super(Cuboid, self).__init__()
        if x1 <= x0: raise ParseFailure()
        if y1 <= y0: raise ParseFailure()
        if z1 <= z0: raise ParseFailure()
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1

    def toTrace(self): return [self]

    def render(self,r=None):
        r = r or RESOLUTION

        a = np.zeros((r,r,r))
        a[int(self.x0*r/RESOLUTION):int(self.x1*r/RESOLUTION),
          int(self.y0*r/RESOLUTION):int(self.y1*r/RESOLUTION),
          int(self.z0*r/RESOLUTION):int(self.z1*r/RESOLUTION)] = 1

        return a
        

    def __str__(self):
        return f"(cuboid {self.x0} {self.y0} {self.z0} ; {self.x1} {self.y1} {self.z1})"

    def children(self): return []

    def serialize(self):
        return (self.__class__.token, self.x0, self.y0, self.z0, self.x1, self.y1, self.z1)

    def __contains__(self, p):
        return p[0] >= self.x0 and p[1] >= self.y0 and p[2] >= self.z0 and \
            p[0] < self.x1 and p[1] < self.y1 and p[2] < self.z1

    def flipX(self):
        return Cuboid(RESOLUTION - self.x1, self.y0, self.z0,
                      RESOLUTION - self.x0, self.y1, self.z1)

    def flipY(self):
        return Cuboid(self.x0, RESOLUTION - self.y1, self.z0,
                      self.x1, RESOLUTION - self.y0, self.z1)

    def flipZ(self):
        return Cuboid(self.x0, self.y1, RESOLUTION - self.z0,
                      self.x1, self.y0, RESOLUTION - self.z1)

# Cuboid(9,9,2,
#        18,27,8).show()


class Sphere(CSG):
    token = 'sphere'
    type = arrow(integer(0, RESOLUTION - 1), integer(0, RESOLUTION - 1), integer(0, RESOLUTION - 1),
                 integer(0, RESOLUTION - 1),
                 tCSG)
    
    def __init__(self, x,y,z,r):
        super(Sphere, self).__init__()
        self.r = r
        self.x = x
        self.y = y
        self.z = z

    def toTrace(self): return [self]

    def render(self,r=None):
        r = r or RESOLUTION
        x, y, z = np.indices((r,r,r))/float(r)
        a = np.zeros((r,r,r))

        dx = (x - self.x/RESOLUTION)
        dy = (y - self.y/RESOLUTION)
        dz = (z - self.z/RESOLUTION)
        distance2 = dx*dx + dy*dy + dz*dz

        return 1.*(distance2 <= (self.r/RESOLUTION)*(self.r/RESOLUTION))
        
    def __str__(self):
        return f"(sphere ({self.x}, {self.y}, {self.z}) {self.r})"

    def children(self): return []

    def serialize(self):
        return (self.__class__.token, self.x,self.y,self.z,self.r)

    def __contains__(self, p):
        return (p[0]-self.x)*(p[0]-self.x) + (p[1] - self.y)*(p[1] - self.y) + (p[2] - self.z)*(p[2] - self.z)\
            <= self.r*self.r

    def flipX(self):
        return Sphere(RESOLUTION - self.x, self.y, self.z, self.r)

    def flipY(self):
        return Sphere(self.x, RESOLUTION - self.y, self.z, self.r)

    def flipZ(self):
        return Sphere(self.x, self.y, RESOLUTION - self.z, self.r)

Sphere(3,16,25,9).show()

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
        return ('+',self.elements[0],self.elements[1])

    def __eq__(self, o):
        return isinstance(o, Union) and tuple(o.elements) == tuple(self.elements)

    def __hash__(self):
        return hash(('u', tuple(hash(e) for e in self.elements )))

    def __contains__(self, p):
        return any( p in e for e in self.elements )

    def removeDeadCode(self):
        a = self.elements[0].removeDeadCode()
        b = self.elements[1].removeDeadCode()

        self = Union(a,b)

        me = self.render()
        l = a.render()
        r = b.render()
        if np.all(me == l): return a
        if np.all(me == r): return b
        return self

    def flipX(self): return Union(self.elements[0].flipX(),
                                  self.elements[1].flipX())
    def flipY(self): return Union(self.elements[0].flipY(),
                                  self.elements[1].flipY())
    def flipZ(self): return Union(self.elements[0].flipZ(),
                                  self.elements[1].flipZ())

    def render(self,r=None):
        r = r or RESOLUTION
        a = 1.*(self.elements[0].render(r))
        b = 1.*(self.elements[1].render(r))
        return a + b - a*b        

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

    def removeDeadCode(self):
        a = self.a.removeDeadCode()
        b = self.b.removeDeadCode()

        self = Difference(a,b)

        me = self.render()
        l = a.render()
        r = b.render()
        if np.all(l < r): raise BadCSG()
        if np.all(me == l): return a
        if np.all(me == r): return b
        return self

    def flipY(self): return Difference(self.a.flipY(),
                                       self.b.flipY())
    def flipX(self): return Difference(self.a.flipX(),
                                       self.b.flipX())
    def flipZ(self): return Difference(self.a.flipZ(),
                                       self.b.flipZ())

    def render(self,r):
        a = 1.*self.a.render(r)
        b = 1.*self.b.render(r)
        return np.clip(a - b,
                       0., 1.)

dsl = DSL([Union, Difference, Cuboid, Sphere],
          lexicon=CSG.lexicon)

"""Neural networks"""
class ObjectEncoder(CNN):
    def __init__(self):
        super(ObjectEncoder, self).__init__(channels=2,
                                            inputImageDimension=RESOLUTION)

    def forward(self, spec, obj):
        if isinstance(spec, list):
            # batching both along specs and objects
            assert isinstance(obj, list)
            B = len(spec)
            assert len(obj) == B
            return super(ObjectEncoder, self).forward(np.stack([np.stack([s,o]) for s,o in zip(spec, obj)]))
        elif isinstance(obj, list): # batched - expect a single spec and multiple objects
            spec = np.repeat(spec[np.newaxis,:,:],len(obj),axis=0)
            obj = np.stack(obj)
            return super(ObjectEncoder, self).forward(np.stack([spec, obj],1))
        else: # not batched
            return super(ObjectEncoder, self).forward(np.stack([spec, obj]))
        

class SpecEncoder(CNN):
    def __init__(self):
        super(SpecEncoder, self).__init__(channels=1,
                                          inputImageDimension=RESOLUTION)


class HeatMap(Module):
    def __init__(self):
        super(HeatMap, self).__init__()

        h = 64
        o = 3
        def residual():
            return ResidualCNNBlock(h)
        self.model = nn.Sequential(nn.Conv2d(1, h, 3, padding=1),
                                   nn.ReLU(),
                                   residual(), residual(), residual(), residual(),
                                   nn.Conv2d(h, o, 3, padding=1))

        self.finalize()

    def forward(self, x):
        return self.model(x)


def learnHeatMap(checkpoint='checkpoints/hm.p'):
    data = getTrainingData('CSG_data.p')
    B = 32

    if os.path.exists(checkpoint):
        print("Found checkpoint - going to do a demo, and then resume training")
        with open(checkpoint,'rb') as handle:
            hm = pickle.load(handle)
        with torch.no_grad():
            ps = [data() for _ in range(B)]
            xs = hm.device(torch.tensor(np.array([p.execute() for p in ps ])).float())
            xs = xs.unsqueeze(1)
            ys = hm.tensor(torch.tensor(1.*np.array([p.heatMapTarget() for p in ps])).float())
            ys = ys.permute(*[0,3,1,2])
            predictions = F.sigmoid(hm(xs)).cpu().numpy()

        os.system("mkdir  -p data/hm/")

        for b in range(B):
            i = ps[b].execute()
            saveMatrixAsImage(i, f"data/hm/{b}.png")
            t = ps[b].heatMapTarget()
            for j in range(3):
                saveMatrixAsImage(t[:,:,j], f"data/hm/{b}_{j}_target.png")
                saveMatrixAsImage(predictions[b,j,:,:], f"data/hm/{b}_{j}_prediction.png")
    else:        
        hm = HeatMap()
    startTime = time.time()
    optimizer = torch.optim.Adam(hm.parameters(), lr=0.001, eps=1e-3, amsgrad=True)
    i = 0
    while time.time() < startTime + 3600*5:
        i += 1
        
        hm.zero_grad()
        
        ps = [data() for _ in range(B)]
        xs = hm.device(torch.tensor(np.array([p.execute() for p in ps ])).float())
        xs = xs.unsqueeze(1)
        ys = hm.tensor(torch.tensor(1.*np.array([p.heatMapTarget() for p in ps])).float())
        ys = ys.permute(*[0,3,1,2])
        predictions = hm(xs)
        l = F.binary_cross_entropy_with_logits(predictions,ys,reduction='sum')/B
        l.backward()
        optimizer.step()

        if i%1000 == 1:
            print(l.data)
            with open(checkpoint,'wb') as handle:
                pickle.dump(hm,handle)
        
    


    
"""Training"""
def randomScene(resolution=32, maxShapes=3, minShapes=1, verbose=False, export=None,
                nudge=False, disjointUnion=True, translate=True):
    import matplotlib.pyplot as plot

    def translation(x,y,child):
        if not nudge: return Translation(x,y,child)
        for _ in range(x):
            child = Translation(1,0,child)
        for _ in range(y):
            child = Translation(0,1,child)
        return child
    dc = 16 # number of distinct coordinates
    choices = [c
               for c in range(resolution//(dc*2), resolution, resolution//dc) ]
    def quadrilateral():
        if not translate:
            x0 = random.choice(choices[:-1])
            y0 = random.choice(choices[:-1])
            x1 = random.choice([x for x in choices if x > x0 ])
            y1 = random.choice([y for y in choices if y > y0 ])
            return TRectangle(x0,y0,x1,y1)
            
        while True:
            
            w = random.choice(range(2, resolution, resolution//dc))
            h = random.choice(range(2, resolution, resolution//dc))
            x = random.choice(choices)
            y = random.choice(choices)
            if x + w < resolution and y + h < resolution:
                return translation(x,y,
                                   Rectangle(w,h))

    def circular():
        choices = [c
                   for c in range(resolution//(dc*2), resolution, resolution//dc) ]

        if not translate:
            r = random.choice(range(2,10,2))
            x = random.choice([x for x in choices if x - r > 0 and x + r < resolution ])
            y = random.choice([y for y in choices if y - r > 0 and y + r < resolution ])
            return TCircle(x,y,r)
        while True:
            r = random.choice(range(2,10,2))
            choices = [c
                       for c in range(resolution//(dc*2), resolution, resolution//dc) ]
            x = random.choice(choices)
            y = random.choice(choices)
            if x - r >= 0 and x + r < resolution and y - r >= 0 and y + r < resolution:
                return translation(x,y,
                                   Circle(r))
    while True:
        s = None
        numberOfShapes = 0
        desiredShapes = random.choice(range(minShapes, 1 + maxShapes))
        for _ in range(desiredShapes):
            o = quadrilateral() if random.choice([True,False]) else circular()
            if s is None:
                s = o
            else:
                if disjointUnion:
                    if (o.render()*s.render()).sum() > 0.5: continue
                    s = Union(s,o)
                    continue
                
                if random.choice([True,False]):
                    new = Union(s,o)
                elif True or random.choice([True,False]):
                    new = Difference(s,o)
                else:
                    new = Difference(o,s)
                # Change at least ten percent of the pixels
                oldOn = s.render().sum()
                newOn = new.render().sum()
                pc = abs(oldOn - newOn)/oldOn
                if pc < 0.1 or pc > 0.6:
                    continue
                s = new
        try:
            finalScene = s.removeDeadCode()
            assert np.all(finalScene.render() == s.render())
            s = finalScene
            break
        except BadCSG:
            continue
    
    if verbose:
        print(s)
        print(ProgramGraph.fromRoot(s, oneParent=True).prettyPrint())
        s.show()
    if export:
        plot.imshow(s.execute())
        plot.savefig(export)
        plot.imshow(s.highresolution(256))
        plot.savefig(f"{export}_hr.png")
        if not translate:
            for j in range(3):
                plot.imshow(s.heatMapTarget()[:,:,j])
                plot.savefig(f"{export}_hm_{j}.png")
    
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

    B = 16

    while trainTime is None or time.time() - startTime < trainTime:
        ss = [getProgram() for _ in range(B)]
        ls = m.gradientStepTraceBatched(optimizer, [(s, s.toTrace())
                                                    for s in ss])
        for l in ls:
            totalLosses.append(sum(l))
            movedLosses.append(sum(l)/len(l))
        iteration += 1
        if iteration%reportingFrequency == 1:
            print(f"\n\nAfter {iteration*B} training examples...\n\tTrace loss {sum(totalLosses)/len(totalLosses)}\t\tMove loss {sum(movedLosses)/len(movedLosses)}\n{iteration*B/(time.time() - startTime)} examples/sec\n{iteration/(time.time() - startTime)} grad steps/sec")
            totalLosses = []
            movedLosses = []
            with open(checkpoint,"wb") as handle:
                pickle.dump(m, handle)



def testCSG(m, getProgram, timeout, export):
    random.seed(0)
    oneParent = m.oneParent
    print(f"One parent restriction?  {oneParent}")
    solvers = [# RandomSolver(dsl),
               # MCTS(m, reward=lambda l: 1. - l),
               SMC(m),
        BeamSearch(m, criticCoefficient=1),
                BeamSearch(m, criticCoefficient=0.),
               ForwardSample(m, maximumLength=18)]
    loss = lambda spec, program: 1-max( o.IoU(spec) for o in program.objects() ) if len(program) > 0 else 1.

    testResults = [[] for _ in solvers]

    os.system("mkdir data/test")

    for ti in range(30):
        spec = getProgram()
        print("Trying to explain the program:")
        print(ProgramGraph.fromRoot(spec, oneParent=oneParent).prettyPrint())
        print()
        saveMatrixAsImage(spec.highresolution(256), "data/test/%03d.png"%ti)
        for n, solver in enumerate(solvers):
            print(f"Running solver {solver.name}")
            solver.maximumLength = len(ProgramGraph.fromRoot(spec).nodes) + 1
            testSequence = solver.infer(spec, loss, timeout)
            if len(testSequence) == 0:
                testSequence = [SearchResult(ProgramGraph([]), 1., 0., 1)]
            testResults[n].append(testSequence)
            for result in testSequence:
                print(f"After time {result.time}, achieved loss {result.loss} w/")
                print(result.program.prettyPrint())
                print()
            if len(testSequence) > 0:
                obs = testSequence[-1].program.objects()
                if len(obs) == 0:
                    bestProgram = np.zeros((256,256))
                else:
                    bestProgram = max(obs, key=lambda bp: bp.IoU(spec)).highresolution(256)
                saveMatrixAsImage(bestProgram,
                                  "data/test/%03d_%s.png"%(ti,solver.name))
                

    plotTestResults(testResults, timeout,
                    defaultLoss=1.,
                    names=[s.name for s in solvers],
                    export=export)

def plotTestResults(testResults, timeout, defaultLoss=None,
                    names=None, export=None):
    import matplotlib.pyplot as plot

    def averageLoss(n, predicate):
        results = testResults[n] # list of list of results, one for each test case
        # Filter out results that occurred after time T
        results = [ [r for r in rs if predicate(r)]
                    for rs in results ]
        losses = [ min([defaultLoss] + [r.loss for r in rs]) for rs in results ]
        return sum(losses)/len(losses)

    plot.figure()
    plot.xlabel('Time')
    plot.ylabel('Average Loss')
    plot.ylim(bottom=0.)
    for n in range(len(testResults)):
        xs = list(np.arange(0,timeout,0.1))
        plot.plot(xs, [averageLoss(n,lambda r: r.time < x) for x in xs],
                  label=names[n])
    plot.legend()
    if export:
        plot.savefig(export)
    else:
        plot.show()
    plot.figure()
    plot.xlabel('Evaluations')
    plot.ylabel('Average Loss')
    plot.ylim(bottom=0.)
    for n in range(len(testResults)):
        xs = list(range(max(r.evaluations for tr in testResults[n] for r in tr )))
        plot.plot(xs, [averageLoss(n,lambda r: r.evaluations < x) for x in xs],
                  label=names[n])
    plot.legend()
    if export:
        plot.savefig(f"{export}_evaluations.png")
    else:
        plot.show()
        
        
    
def makeTrainingData():
    import os
    import matplotlib.pyplot as plot
    
    data = {} # Map from image to (size, {programs})
    lastUpdate = time.time()
    n_samples = 0
    maximumSize = 0
    startTime = time.time()
    while time.time() < startTime + 3600*5:
        n_samples += 1
        program = randomScene(maxShapes=20, minShapes=5, disjointUnion=False, translate=False)
        size = len(program.toTrace())
        im = program.render()
        im = tuple( im[x,y] > 0.5
                    for x in range(RESOLUTION)
                    for y in range(RESOLUTION) )
        program.clearRendering()
        if im in data:
            oldSize, oldPrograms = data[im]
            if oldSize < size:
                pass
            elif oldSize == size:
                data[im] = (size, {program}|oldPrograms)
            elif oldSize > size:
                data[im] = (size, {program})
            else:
                assert False
        else:
            data[im] = (size, {program})
            maximumSize = max(maximumSize,size)

        if time.time() > lastUpdate + 10:
            print(f"After {n_samples} samples; {n_samples/(time.time() - startTime)} samples per second")
            for sz in range(maximumSize):
                n = sum(size == sz
                        for (size,_) in data.values() )
                print(f"{n} images w/ programs of size {sz}")
            print()
            lastUpdate = time.time()

    with open('CSG_data.p','wb') as handle:
        pickle.dump(data, handle)


def getTrainingData(path):
    import copy
    
    with open(path,'rb') as handle:
        data = pickle.load(handle)
    print(f"Loaded {len(data)} images from {path}")
    print(f"Contains {sum(len(ps) for _,ps in data.values() )} programs")
    data = [list(ps) for _,ps in data.values()]

    def getData():
        programs = random.choice(data)
        # make a deep copy because we are caching the renders, and we want these to be garbage collected
        p = copy.deepcopy(random.choice(programs))
        if random.choice([True,False]):
            # print("X flip")
            # showMatrixAsImage(p.highresolution(128),
            #                   p.flipX().highresolution(128))
            p = p.flipX()
        if random.choice([True,False]):
            # print("Y flip")
            # showMatrixAsImage(p.highresolution(128),
            #                   p.flipY().highresolution(128))
            p = p.flipY()
            
        return p

    return getData
        
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("mode", choices=["imitation","exit","test","demo","makeData","heatMap",
                                         "critic"])
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
    parser.add_argument("--nudge", default=False, action='store_true')
    parser.add_argument("--oneParent", default=False, action='store_true')
    parser.add_argument("--noTranslate", default=True, action='store_true')
    parser.add_argument("--disjointUnion", default=False, action='store_true')
    
    arguments = parser.parse_args()
    arguments.translate = not arguments.noTranslate

    if arguments.mode == "demo":
        startTime = time.time()
        ns = 50
        for _ in range(ns):
            randomScene(maxShapes=arguments.maxShapes, minShapes=arguments.maxShapes, disjointUnion=arguments.disjointUnion, nudge=arguments.nudge, translate=arguments.translate).execute()
        print(f"{ns/(time.time() - startTime)} (renders + samples)/second")
        for n in range(100):
            randomScene(export=f"/tmp/CAD_{n}.png",maxShapes=arguments.maxShapes,
                        minShapes=arguments.maxShapes,
                        nudge=arguments.nudge, disjointUnion=arguments.disjointUnion, 
                        translate=arguments.translate)
        import sys
        sys.exit(0)
        
            

    if arguments.mode == "imitation":
        m = ProgramPointerNetwork(ObjectEncoder(), SpecEncoder(), dsl,
                                  oneParent=arguments.oneParent,
                                  attentionRounds=arguments.attention,
                                  heads=arguments.heads,
                                  H=arguments.hidden)
        trainCSG(m, getTrainingData('CSG_data.p'),
                 trainTime=arguments.trainTime*60*60 if arguments.trainTime else None,
                 checkpoint=arguments.checkpoint)
    elif arguments.mode == "critic":
        with open("checkpoints/imitation.pickle","rb") as handle:
            m = pickle.load(handle)
        critic = A2C(m)
        def R(spec, program):
            if len(program) == 0 or len(program) > len(spec.toTrace()): return False
            spec = spec.execute() > 0.5
            for o in program.objects():
                if np.all((o.execute() > 0.5) == spec): return True
            return False
        critic.train(
            lambda: randomScene(maxShapes=arguments.maxShapes, minShapes=arguments.maxShapes, nudge=arguments.nudge, translate=arguments.translate, disjointUnion=arguments.disjointUnion), 
            R)
        
    elif arguments.mode == "heatMap":
        learnHeatMap()
    elif arguments.mode == "makeData":
        makeTrainingData()
    elif arguments.mode == "exit":
        with open(arguments.checkpoint,"rb") as handle:
            m = pickle.load(handle)
        searchAlgorithm = BeamSearch(m, maximumLength=arguments.maxShapes*3 + 1)
        loss = lambda spec, program: 1-max( o.IoU(spec) for o in program.objects() ) if len(program) > 0 else 1.
        searchAlgorithm.train(getTrainingData('CSG_data.p'),
                              # lambda: randomScene(maxShapes=arguments.maxShapes, nudge=arguments.nudge,
                              #                     disjointUnion=arguments.disjointUnion,
                              #                     translate=arguments.translate),
                              loss=loss,
                              policyOracle=lambda spec: spec.toTrace(),
                              timeout=1,
                              exitIterations=-1)
    elif arguments.mode == "test":
        with open(arguments.checkpoint,"rb") as handle:
            m = pickle.load(handle)
        testCSG(m,
                getTrainingData('CSG_data.p'),#lambda: randomScene(maxShapes=arguments.maxShapes, minShapes=arguments.maxShapes, nudge=arguments.nudge, translate=arguments.translate, disjointUnion=arguments.disjointUnion),
                arguments.timeout,
                export=f"figures/CAD_{arguments.maxShapes}_shapes.png")
