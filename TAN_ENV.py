import pickle
import numpy as np

from API import *

import time
import random
from TAN import random_scene, Add, P1, P2, P3, P4, P5, RESOLUTION, decompose

# UP, DOWN, LEFT, RIGHT, SPIN, COMMIT
ACTIONS = ['U', 'D', 'L', 'R', 'S', 'C']

BLANK = np.zeros((6,RESOLUTION, RESOLUTION))
BLANK[0, :, :] = 1.0

# TAN ENVIRONMENT 
class TAN_ENV:

    def __init__(self):
        print ("hi i live")
        self.tan = None
        self.cur_prog = None
        self.scratch = None
        self.pieces = []

    def clone_state(self):
        return self.tan,
               self.spec
               self.cur_prog
               self.pieces
               self.scratch

    def load_state(self, cloned_state):
        tan, spec, cur_prog, pieces, scratch = cloned_state
        self.tan = tan
        self.spec = spec
        self.cur_prog = cur_prog
        self.pieces = pieces
        self.scratch = scratch

    def render(self):

        committed = BLANK if self.cur_prog is None else self.cur_prog.to_np()
        scratch = self.scratch[0](*self.scratch[1:]).to_np()

        together = np.concatenate((self.spec, committed, scratch))
        return together

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
        self.pieces = [P2, P3, P4, P5]
        self.scratch = [P1, 1, 0, 0]

        return self.render()

    # get an oracular move from the environment
    def oracle(self):
        cur_piece = self.scratch[0]
        cur_o, cur_x, cur_y = self.scratch[1:]
        pieces = decompose(self.tan)
        for p in pieces:
            if p.__class__.token == cur_piece.token:
                o,x,y = p.o, p.x, p.y
                if o != cur_o:
                    return 'S'
                if cur_x < x:
                    return 'R'
                if cur_x > x:
                    return 'L'
                if cur_y < y:
                    return 'D'
                if cur_y > y:
                    return 'U'
                return 'C'
        assert 0, "should not happen"

    # return nxt state, reward, done
    def step(self, move):
        assert move in ACTIONS
        sm_neg = -0.1

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
            return nxt_state, sm_neg, False

        if move == 'C':
            if self.cur_prog is None:
                self.cur_prog = self.scratch[0](*self.scratch[1:])
            else:
                to_add = self.scratch[0](*self.scratch[1:])
                self.cur_prog = Add(self.cur_prog, to_add)
                if self.cur_prog.legal() is False:
                    return None, sm_neg, True

            if len(self.pieces) == 0:
                reward = sm_neg
                if self.cur_prog.tan_distance(self.tan.to_np()) == 0:
                    reward = 1.0
                return None, reward, True
            else:
                self.scratch = [self.pieces[0], 1, 0, 0]
                self.pieces = self.pieces[1:]

            next_state = self.render()
            return next_state, sm_neg, False

# ================ training, rollouts and such ==================
def get_rollout(env, policy, max_iter = 50):
    cur_state = env.reset()
    done = False
    trace = []
    iter_k = 0
    while not done:
        iter_k += 1
        if iter_k > max_iter:
            break
        oracle_a = env.oracle()
        a = policy.act(cur_state)
        next_state, r, done = env.step(a)
        # next_state, r, done = env.step(a)
        trace.append((cur_state, a, oracle_a, r, next_state))
        cur_state = next_state
    return trace

def get_supervision(env, max_iter = 50):
    cur_state = env.reset()
    done = False
    trace = []
    iter_k = 0
    while not done:
        iter_k += 1
        if iter_k > max_iter:
            break
        oracle_a = env.oracle()
        next_state, r, done = env.step(oracle_a)
        # next_state, r, done = env.step(a)
        trace.append((cur_state, oracle_a, r, next_state))
        cur_state = next_state
    return trace

def train_dagger(env, student, max_iter):
    init_state = env.reset()
    s_a_agg = []
    batch_size = 10

    for i in range(1000000):

        # learning
        trace = get_rollout(env, student, max_iter)
        state_sample = [x[0] for x in trace]
        action_sample = [x[2] for x in trace]
        s_a_agg += list(zip(state_sample, action_sample))

        if i % 100 == 0:
            print ("======== diagnostics ==========")
            print ("Oracle Actions")
            print (action_sample)
            print ("Student Actions")
            print ([x[1] for x in trace])

        if len(s_a_agg) > batch_size:
            for i in range(len(s_a_agg) // batch_size // 10):
                sub_sample = random.sample(s_a_agg, batch_size)
                sub_states, sub_actions = [x[0] for x in sub_sample], [x[1] for x in sub_sample]
                student.learn_supervised(sub_states, sub_actions)

def train_supervised(env, student, max_iter):
    init_state = env.reset()
    s_a_agg = []

    for i in range(1000000):
        # learning
        trace = get_supervision(env, max_iter)
        states = [x[0] for x in trace]
        actions = [x[1] for x in trace]
        loss = student.learn_supervised(states, actions)

        if i % 1000 == 0:
            print (f"loss {loss}")
            trace = get_rollout(env, student, max_iter)
            try:
                env.render_pix()
            except:
                pass
            state_sample = [x[0] for x in trace]
            action_sample = [x[2] for x in trace]
            s_a_agg += list(zip(state_sample, action_sample))
            print ("======== diagnostics ==========")
            print ("Oracle Actions")
            print (action_sample)
            print ("Student Actions")
            print ([x[1] for x in trace])

            student.save("tan1.mdl")

def beam_search(goal_spec, student, max_iter, beam_width):
    env = TAN_ENV()
    env.reset()
    tan, spec, cur_prog, pieces, scratch = env.clone_state()
    spec = goal_spec

    init_state = [tan, spec, cur_prog, pieces, scratch]

    beam = [init_state for _ in range(beam_width)]

    for i in range(max_iter):
        pass


# =================== something ===================
def test_env():
    tenv = TAN_ENV()
    cur_state = tenv.reset()
    print (cur_state.shape)
    done = False
    while not done:
        tenv.render_pix()
        oracle_move = tenv.oracle()
        print ("oracle says ", oracle_move)
        a = input('input\n')
        nxt_state, r, done = tenv.step(a)
        print ("reward ", r)
        cur_state = nxt_state

def test_ro():
    from fcnet import Agent
    env = TAN_ENV()
    agent = Agent(18*RESOLUTION*RESOLUTION, ACTIONS)

    trace = get_rollout(env, agent, max_iter = 50)
    print (trace)

def run_train_supervised():
    from fcnet import Agent, MEMAgent
    env = TAN_ENV()
    agent = Agent(18*RESOLUTION*RESOLUTION, ACTIONS)
    train_supervised(env, agent, RESOLUTION*2*5)

def test_supervised():
    from fcnet import Agent, MEMAgent
    env = TAN_ENV()
    agent = Agent(18*RESOLUTION*RESOLUTION, ACTIONS)
    agent.load("tan.mdl")
    for i in range(10):
        ro = get_rollout(env, agent, RESOLUTION*2*5)
        env.render_pix()
        print ("sequence . . . ")
        print ([x[1] for x in ro])
        input("yo")

def run_train_dagger():
    from fcnet import Agent, MEMAgent
    env = TAN_ENV()
    agent = Agent(18*RESOLUTION*RESOLUTION, ACTIONS)
    train_dagger(env, agent, RESOLUTION*2*5)

if __name__ == '__main__':
    # test_env()
    # test_ro()
    # test_supervised()
    run_train_dagger()

