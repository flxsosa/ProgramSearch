import pickle
import numpy as np

from API import *

import time
import random
from TAN import random_scene, Add, P1, P2, P3, P4, RESOLUTION

# UP, DOWN, LEFT, RIGHT, SPIN, COMMIT
ACTIONS = ['U', 'D', 'L', 'R', 'S', 'C']

# TAN ENVIRONMENT 
class TAN_ENV:

    def __init__(self):
        print ("hi i live")
        self.tan = None
        self.cur_prog = None
        self.scratch = None
        self.pieces = []

    def render(self):

        committed = np.zeros(self.tan.to_np().shape) if self.cur_prog is None else self.cur_prog.to_np()
        scratch = self.scratch[0](*self.scratch[1:]).to_np()
        return (self.spec, committed, scratch)

    def render_pix(self):
        self.tan.render('tan_drawings/spec.png')
        if self.cur_prog is not None:
            self.cur_prog.render('tan_drawings/committed.png')
        self.scratch[0](*self.scratch[1:]).render('tan_drawings/scratch.png')

    def reset(self):
        # the spec
        self.tan = random_scene()
        self.spec = self.tan.to_np()

        # commited, nothing here yet
        self.cur_prog = None

        # we gonna try piece 1 first
        self.pieces = [P2, P3, P4]
        self.scratch = [P1, 1, 0, 0]

        return self.render()

    # return nxt state, reward, done
    def step(self, move):
        assert move in ACTIONS

        if move in ['U', 'L', 'D', 'R', 'S']:
            scratch_old = self.scratch
            constr, o, x, y = self.scratch
            if move == 'U':
                y = max(y - 1, 0)
            if move == 'L':
                x = max(x - 1, 0)
            if move == 'D':
                y = min(RESOLUTION - 1, y + 1)
            if move == 'R':
                x = min(RESOLUTION - 1, x + 1)
            if move == 'S':
                o = o % 4 + 1
            self.scratch = [constr, o, x, y]

            tentative = self.scratch[0](*self.scratch[1:])
            if tentative.legal() is False:
                self.scratch = scratch_old

            nxt_state = self.render()
            reward = 0.0
            return nxt_state, reward, False

        if move == 'C':
            if self.cur_prog is None:
                self.cur_prog = self.scratch[0](*self.scratch[1:])
            else:
                to_add = self.scratch[0](*self.scratch[1:])
                self.cur_prog = Add(self.cur_prog, to_add)
                if self.cur_prog.legal() is False:
                    return None, 0.0, True

            if len(self.pieces) == 0:
                reward = 0.0
                if self.cur_prog.tan_distance(self.tan.to_np()) == 0:
                    reward = 1.0
                return None, reward, True
            else:
                self.scratch = [self.pieces[0], 1, 0, 0]
                self.pieces = self.pieces[1:]

            next_state = self.render()
            reward = 0.0
            return next_state, reward, False
# =================== something ===================
def test_env():
    tenv = TAN_ENV()
    cur_state = tenv.reset()
    done = False
    while not done:
        tenv.render_pix()
        a = input('input\n')
        cur_state, r, done = tenv.step(a)
        print ("reward ", r)


if __name__ == '__main__':
    test_env()

