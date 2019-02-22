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


"""
TODO:
- [X] copy
- [ ] rendering code
- [ ] data generation code
- [ ] random code
- [ ] pointer network graph code
"""

class Fig(Program):
    #first, version without loops+variables
    lexicon = ['add', 'circle', 'rect', 'line', 'copy'] + list(range(RESOLUTION))

    def __init__(self):
        self._rendering = None

    def __ne__(self, o): return not (self == o)

    def execute(self):
        if self._rendering is None: self._rendering = self.render()
        return self._rendering

    # def IoU(self, other):
    #     if isinstance(other, CSG): other = other.execute()
    #     return (self.execute()*other).sum()/(self.execute() + other - self.execute()*other).sum()
    
    def render(self, w=None, h=None):
        raise NotImplementedError
        # w = w or RESOLUTION
        # h = h or RESOLUTION
        
        # a = np.zeros((w,h))
        # for x in range(w):
        #     for y in range(h):
        #         if (x,y) in self:
        #             a[x,y] = 1
        # return a

tFig = BaseType(Fig)
tbool = BaseType(bool)

class Rectangle(Fig):
    token = 'rect'
    #corner to corner
    type = arrow(integer(0, RESOLUTION - 1), 
                 integer(0, RESOLUTION - 1),
                 integer(0, RESOLUTION - 1),
                 integer(0, RESOLUTION - 1), 
                 tFig)
    
    def __init__(self, x1, y1, x2, y2):
        super(Rectangle, self).__init__()
        self.l1 = (x1, y1)
        self.l2 = (x2, y2)

    def children(self): return [] #list(self.l1+self.l2) 

    def __eq__(self, o):
        return isinstance(o, Rectangle) and self.l1 == self.l1 and o.l2 == self.l2

    def __hash__(self):
        return hash(('rect',) + self.l1 + self.l2 )

    def serialize(self):
        return (self.__class__.token,) + self.l1 + self.l2 # TODO

    def __contains__(self, p):
        # TODO
        assert False

class Circle(Fig):
    token = 'circle'
    #defined with coords of center and radius
    type = arrow(integer(0, RESOLUTION - 1), 
             integer(0, RESOLUTION - 1),
             integer(0, RESOLUTION - 1), 
             tFig)
    
    def __init__(self, x, y, r):
        super(Circle, self).__init__()
        self.l = (x,y)
        self.r = r

    def children(self): return [] #list(self.l)

    def __eq__(self, o):
        return isinstance(o, Circle) and o.l == self.l
    def __hash__(self):
        return hash(('circle',) + (self.l,) + (self.r,) )

    def serialize(self):
        return (self.__class__.token,) + self.l + self.r

    def __contains__(self, p):
        # TODO
        assert False

class Line(Fig):
    token = 'line'
    argument_types = (int, int, int, int, bool)
    type = arrow(integer(0, RESOLUTION - 1), 
             integer(0, RESOLUTION - 1),
             integer(0, RESOLUTION - 1),
             integer(0, RESOLUTION - 1),
             tbool,
             tFig)

    def __init__(self, x1, y1, x2, y2, arrow):
        super(Line, self).__init__()
        self.start = (x1, y1)
        self.end = (x2, y2)
        self.arrow = arrow

    def children(self): return []

    def serialize(self):
        if self.arrow:
            return ('line',) + self.start + self.end + ( 'arrow',) 
        else:
            return ('line',) + self.start + self.end

    def __eq__(self, o):
        return isinstance(o, Line) and o.start == self.start and self.end == o.end and self.arrow == o.arrow

    def __hash__(self):
        return hash(('line',) + self.start + self.end + (self.arrow,) )

    def __contains__(self, p):
        assert False

class Copy(Fig):
    token = 'copy'
    #type sig: Figure, number of copies, and vector x, y of translate
    type = arrow(tFig, integer(0, RESOLUTION - 1), 
                 integer(-RESOLUTION, RESOLUTION - 1),
                 integer(-RESOLUTION, RESOLUTION - 1),
                 tFig)
    
    def __init__(self, child, nCopies, x, y):
        super(Copy, self).__init__()
        self.child = child
        self.nCopies = nCopies
        self.trans = (x,y)

    def children(self): return [self.child]

    def serialize(self):
        return ('copy', self.child, self.nCopies, self.trans)

    def __eq__(self, o):
        return isinstance(o, Copy) and o.child == self.child and self.nCopies == o.nCopies and self.trans == o.trans

    def __hash__(self):
        return hash(('copy', self.child, self.nCopies, self.trans))

    def __contains__(self, p):
        assert False
        return (p in self.child)

class Add(Fig):
    token = 'add'
    
    type = arrow(tFig, 
                tFig,
                tFig)

    def __init__(self, a, b):
        super(Add, self).__init__()
        self.elements = frozenset({a,b})

    def children(self): return list(self.elements)

    def serialize(self):
        return ('Add',list(self.elements)[0],list(self.elements)[1])

    def __eq__(self, o):
        return isinstance(o, Add) and o.elements == self.elements

    def __hash__(self):
        return hash(('add', self.elements))

    def __contains__(self, p):
        return any( p in e for e in self.elements )


dsl = DSL([Rectangle, Circle, Line, Add, Copy],
          lexicon=Fig.lexicon)


###############

"""Training"""
# ======= DATA GENERATION =======
def random_scene(resolution=RESOLUTION, maxShapes=6, minShapes=3, verbose=False, export=None):
    def quadrilateral():
        w = random.choice(range(int(resolution/8))) + 3
        h = random.choice(range(int(resolution/8))) + 3
        x = random.choice(range(resolution - w))
        y = random.choice(range(resolution - h))
        return Rectangle(x, y, x+w, y+h)

    def circular():
        r = random.choice(range(int(resolution/16))) + 2
        x = random.choice(range(resolution - r*2)) + r
        y = random.choice(range(resolution - r*2)) + r
        return Circle(x,y,r)

    def line():
        w = random.choice(range(int(resolution/4))) + 3
        h = random.choice(range(int(resolution/4))) + 3
        x = random.choice(range(resolution - w))
        y = random.choice(range(resolution - h))
        a = random.choice([True,False])
        return Line(x, y, x+w, y+h, a)

    def copy(fig):
        nTimes = random.choice(range(2, 5))
        dx = random.choice(range(1, 4))
        dy = random.choice(range(1, 4))
        return Copy(fig, nTimes, dx, dy)

    s = None
    numberOfShapes = 0
    desiredShapes = random.choice(range(minShapes, 1 + maxShapes))

    while numberOfShapes < desiredShapes:
        #randomly pick one
        o = [quadrilateral, circular, line][random.choice(range(3))]()
        if random.choice([True,False]):
            o = copy(o)
        if s is None:
            s = o
        else:
            s = Add(s,o)
        numberOfShapes += 1
    if verbose:
        raise NotImplementedError
    if export:
        raise NotImplementedError
    return s

def test_random():
    from randomSolver import RandomSolver

    # using program as spec for now .. just for random
    loss = lambda spec, program: program == spec 

    r_solver = RandomSolver(dsl)
    program = random_scene()
    testSequence = r_solver.infer(program, loss, 10)

    print (testSequence)
    # for idx, ppp in enumerate(testSequence):
    #     ppp.program.get_root().render(f'tan_drawings/{idx}.png')
    # program.render('tan_drawings/goal.png')

if __name__ == '__main__':
    # test_constructor()
    test_random()