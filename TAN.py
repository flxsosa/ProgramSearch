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
from PIL import Image, ImageDraw


RESOLUTION = 3
ORIENTATIONS = [1, 2, 3, 4]
# We have a 6 x 8 x 8 ZA WARLDUUUUU
# each cell can be of 6 states of filled-ness
#     empty --- 1 --- 2 --- 3 --- 4 --- full
#         ..    xx    x.    .x    xx    xx
#         ..    x.    xx    xx    .x    xx
# channel 0 --- 1 --- 2 --- 3 --- 4 --- 5

# render constants
RENDER_SCALE = 100


class TAN(Program):
    # lexicon = ['P1','P2','P3','P4','P5','P6','P7'] +\
    #           ORIENTATIONS +\
    #           list(range(RESOLUTION))

    lexicon = ['P1','P2','P3','P4'] +\
              ORIENTATIONS +\
              list(range(RESOLUTION))

    def to_np_raw(self):
        ret = np.zeros((6, RESOLUTION, RESOLUTION))
        for ch in range(6):
            for x in range(RESOLUTION):
                for y in range(RESOLUTION):
                    if (ch, x, y) in self:
                        ret[ch, x, y] = 1
        return ret

    def legal(self):
        self_np = self.to_np_raw()
        mass = 0
        for x in range(RESOLUTION):
            for y in range(RESOLUTION):
                # multiple channel is hot
                if np.sum(self_np[:, x, y]) == 1:
                    idx = np.argmax(self_np[:, x, y])
                    if idx == 0:
                        mass += 0
                    elif idx == 5:
                        mass += 2
                    else:
                        mass += 1
                if np.sum(self_np[:, x, y]) == 2:
                    if self_np[0, x, y] == 1:
                        return False
                    if self_np[5, x, y] == 1:
                        return False
                    if self_np[1, x, y] == self_np[3, x, y] == 1\
                            or self_np[2, x, y] == self_np[4, x, y] == 1:
                        self_np[1,x,y] = 0
                        self_np[2,x,y] = 0
                        self_np[3,x,y] = 0
                        self_np[4,x,y] = 0
                        self_np[5,x,y] = 1
                        mass += 2
                    else:
                        return False
                if np.sum(self_np[:, x, y]) > 2:
                    return False

        if mass == self.mass():
            return self_np
        else:
            return False

    def to_np(self):
        return self.legal()

    def to_board(self):
        self_np = self.to_np()
        ret = np.zeros((RESOLUTION, RESOLUTION))
        for x in range(RESOLUTION):
            for y in range(RESOLUTION):
                ret[x, y] = np.argmax(self_np[:,x,y])
        return ret

    def tan_distance(self, spec_np):
        my_np = self.to_np()
        acc = np.sum(np.logical_and(my_np, spec_np)) / (np.sum(my_np + spec_np) / 2)
        return 1 - acc

    # note: this render is NOT what execute does, this literally render a png thats it
    def render(self, name = "board.png"):
        render_board(self.to_board(), name)

    def execute(self):
        return self.to_np()

# The type of TAN's
tTAN = BaseType(TAN)

class P(TAN):
    """
    THE BASIC PIECES
    """
    type = arrow(integer(1, 4), 
                 integer(0, RESOLUTION - 1), 
                integer(0, RESOLUTION - 1), 
                tTAN)

    def __init__(self, o, x, y):
        super(P, self).__init__()
        self.o, self.x, self.y = o, x, y

    def children(self): return []

    def __eq__(self, other):
        return isinstance(other, TAN)\
               and other.__class__.token == self.__class__.token\
               and other.o == self.o and other.x == self.x and other.y == self.y

    def __hash__(self):
        return hash((self.__class__.token, self.o, self.x, self.y))

    def serialize(self):
        return [self.__class__.token] + [self.o, self.x, self.y]


class P1(P):
    '''
    SMALL_TRIANGLE, PIECES P1 and P2 
    xx
    x.       orient 1

    x.
    xx       orient 2

    .x
    xx       orient 3

    xx
    .x       orient 4
    '''
    token = 'P1'
    def __contains__(self, p):
        cc, yy, xx = p
        if xx == self.x and yy == self.y:
            return int(self.o) == cc
        else:
            return False

    def mass(self):
        return 1

class P2(P1):
    '''
    EXACTLY SAME AS P1 except it is P2 :D :D :D 
    '''
    token = 'P2'

class P3(P):
    '''
    SMALL SQUARE

    xx
    xx       orient 1 / 2 / 3 / 4 all same

    '''
    token = 'P3'
    def __contains__(self, p):
        cc, yy, xx = p
        if xx == self.x and yy == self.y:
            return cc == 5
        else:
            return False

    def mass(self):
        return 2

class P4(P):
    '''
    MED TRIANGLE

    '''
    token = 'P4'
    def __contains__(self, p):
        cc, yy, xx = p
        # at the coordinate of exact positioning
        if xx == self.x and yy == self.y:
            if self.o == 1: return cc == 2
            if self.o == 2: return cc == 3
            if self.o == 3: return cc == 3
            if self.o == 4: return cc == 4
        # one down
        if xx == self.x and yy == self.y + 1:
            if self.o == 1: return cc == 1
            if self.o == 3: return cc == 4
        # one right
        if xx == self.x + 1 and yy == self.y:
            if self.o == 2: return cc == 2
            if self.o == 4: return cc == 1
        else:
            return False

    def mass(self):
        return 2

class P5(P):
    '''
    MED TRIANGLE

    '''
    token = 'P5'
    def __contains__(self, p):
        cc, yy, xx = p
        # at the coordinate of exact positioning
        if xx == self.x and yy == self.y:
            if self.o == 1: return cc == 2
            if self.o == 2: return cc == 3
            if self.o == 3: return cc == 3
            if self.o == 4: return cc == 4
        # one down
        if xx == self.x and yy == self.y + 1:
            if self.o == 1: return cc == 4
            if self.o == 3: return cc == 1
        # one right
        if xx == self.x + 1 and yy == self.y:
            if self.o == 2: return cc == 1
            if self.o == 4: return cc == 2
        else:
            return False

    def mass(self):
        return 2

# A Helper Class that only computes things and is not fundamentally useful
class Add(TAN):
    token = '+'
    type = arrow(tTAN, tTAN, tTAN)

    def __init__(self, tan1, tan2):
        super(Add, self).__init__()
        self.elements = frozenset([tan1, tan2])

    def children(self): return list(self.elements)

    def __eq__(self, o):
        return isinstance(o, Add) and o.elements == self.elements

    def __hash__(self):
        return hash(('+', self.elements))

    def __contains__(self, p):
        return any( p in e for e in self.elements )

    def mass(self):
        return sum([p.mass() for p in self.elements])

    def serialize(self):
        return ['+'] + self.children()

# ======= PICTURE RENDERING ========
def render_board(board, name):
    def board_2_coords(board):
        coords = []
        for x in range(RESOLUTION):
            for y in range(RESOLUTION):
                # we need to index backwards
                board_cell = board[y][x]
                if board_cell == 0:
                    continue
                if board_cell == 1:
                    coords.append( [x,y,x+1,y,x,y+1] )
                    continue
                if board_cell == 2:
                    coords.append( [x,y,x+1,y+1,x,y+1] )
                    continue
                if board_cell == 3:
                    coords.append( [x+1,y,x+1,y+1,x,y+1] )
                    continue
                if board_cell == 4:
                    coords.append( [x,y,x+1,y,x+1,y+1] )
                    continue
                if board_cell == 5:
                    coords.append( [x,y,x+1,y,x,y+1] )
                    coords.append( [x+1,y,x+1,y+1,x,y+1] )
                    continue
        return coords

    def draw(coords):
        im = Image.new('RGBA', (RENDER_SCALE*RESOLUTION, RENDER_SCALE*RESOLUTION))
        draw = ImageDraw.Draw(im)

        for c in coords:
            c = RENDER_SCALE * np.array(c)
            draw.polygon([(c[0],c[1]),(c[2],c[3]),(c[4],c[5])], fill = 'black')
        im.save("./{}".format(name), 'PNG')

    coords = board_2_coords(board)
    draw(coords)

class TanGraph(ProgramGraph):

    def policyOracle(self, currentGraph):
        yield from (self.nodes - currentGraph.nodes)

    def action_space(self):
        return [P1, P2, P3, P4], ORIENTATIONS, range(RESOLUTION), range(RESOLUTION)

    def extend(self, newNode):
        return TanGraph(self.nodes | {newNode})

# ======= DATA GENERATION =======
def random_scene(resolution=RESOLUTION, export=None):
    def random_oxy(x_lim, y_lim):
        o = random.choice(ORIENTATIONS)
        x = random.choice(range(x_lim))
        y = random.choice(range(y_lim))
        return o, x, y

    def random_P1():
        return P1(*random_oxy(resolution, resolution))
    def random_P2():
        return P2(*random_oxy(resolution, resolution))
    def random_P3():
        return P3(*random_oxy(resolution, resolution))
    def random_P4():
        return P4(*random_oxy(resolution, resolution))
    def random_P5():
        return P5(*random_oxy(resolution, resolution))

    ret_args = [random_P1(), random_P2(), random_P3(), random_P4()]
    
    ret = ret_args[0]
    for x in ret_args[1:]:
        ret = Add(ret, x)

    if ret.legal() is not False:
        return ret
    else:
        return random_scene()

def random_graph():
    return TanGraph(random_scene())

# =================== something ===================
dsl = DSL([P1, P2, P3, P4, Add],
          lexicon=TAN.lexicon)

def test_constructor():
    scene1 = random_scene()
    print (scene1.mass())
    print (scene1.to_np_raw())
    print (scene1.to_board())
    scene1.render('tan_drawings/board.png')

    spec = random_scene().to_np()
    print (scene1.tan_distance(spec))

def test_random():
    from randomSolver import RandomSolver
    loss = lambda spec, program: program.get_root().tan_distance(spec)

    r_solver = RandomSolver(dsl)
    program = random_scene()
    testSequence = r_solver.infer(program.execute(), loss, 100)

    print (testSequence)
    for idx, ppp in enumerate(testSequence):
        ppp.program.get_root().render(f'tan_drawings/{idx}.png')
    program.render('tan_drawings/goal.png')

if __name__ == '__main__':
    # test_constructor()
    test_random()
