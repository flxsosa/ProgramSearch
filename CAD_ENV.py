#CAD env

from CAD import Rectangle, Circle, Translation, Union, Difference
#from CAD import randomScene
from programGraph import ProgramGraph
from fcnet import Agent
import itertools
import numpy as np
import random

NUM_SCRATCH = 2
RESOLUTION = 32

SHAPE_SIZE = RESOLUTION/8 #STARTING WITH FIXED SIZE ...

ACTION_PRIMS = [('+', 2), ('-', 2), ('Circ', 1), ('Rect', 1), ('U', 1), ('D', 1), ('L', 1), ('R', 1), ('Commit', 1)] #maybe bigger or smaller ...

def makeFullActionSet():
	actions = []
	SCRATCHES = list(range(NUM_SCRATCH))
	for action_type, arity in ACTION_PRIMS:
		actions += list( (action_type, ) + x for x in itertools.product(SCRATCHES,repeat=arity))
	return actions

ACTIONS = makeFullActionSet() #TODO, make combinatorial

BLANK = np.zeros((RESOLUTION, RESOLUTION)) #whatever it is
#ASSUME THERE IS SOMETHING CALLED A MODEL



def export(array, name):
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plot
	plot.imshow(array)
	plot.savefig(name)

def randomSceneAndTrace(resolution=32, maxShapes=5, minShapes=4, verbose=False, export=None):
    trace = []
    def quadrilateral():
        choices = [c
                   for c in range(resolution//8, resolution, resolution//4) ]
        w = SHAPE_SIZE #random.choice([2,5])
        h = SHAPE_SIZE #random.choice([2,5])
        x = random.choice(choices)
        y = random.choice(choices)
        action_list = []
        action_list.append( ('Rect', 0) )
        for i in range(0, x):
            action_list.append( ( 'D', 0) ) # TODO direction
        for i in range(0, y):
            action_list.append( ( 'R', 0) ) # TODO direction
        action_list.append( ('Commit', 0) )
        return Translation(x,y,
                           Rectangle(w,h)), action_list

    def circular():
        r = SHAPE_SIZE #random.choice([2,4])
        choices = [c
                   for c in range(resolution//8, resolution, resolution//4) ]
        x = random.choice(choices)
        y = random.choice(choices)

        action_list = []
        action_list.append( ('Circ', 0) )
        for i in range(0, x):
            action_list.append( ( 'D', 0) ) # TODO direction
        for i in range(0, y):
            action_list.append( ( 'R', 0) ) # TODO direction
        action_list.append( ('Commit', 0) )
        return Translation(x,y,
                           Circle(r)), action_list
    s = None
    numberOfShapes = 0
    i = 0
    desiredShapes = random.choice(range(minShapes, 1 + maxShapes))
    trace = []
    while numberOfShapes < desiredShapes and i < desiredShapes*5:
        o, action_list = quadrilateral() if random.choice([True,False]) else circular()
        if s is None:
            s = o
            trace.extend(action_list)
        else:
            if (s.execute()*o.execute()).sum()/(o.execute()).sum() > 0.3:
                #print("continue tripped")
                continue
            s = Union(s,o)
            trace.extend(action_list)
        numberOfShapes += 1
        i += 1
    return s, trace

class CAD_ENV:

	def __init__(self):
		self.goal = None
		self.cur_prog = None
		self.scratch = [None for _ in range(NUM_SCRATCH)]
		#self.actions = # TODO

	def render(self):
		committed = BLANK if self.cur_prog is None else self.cur_prog.execute()
		scratchboards = (BLANK if scr is None else scr.execute() for scr in  self.scratch)
		spec = BLANK if self.goal is None else self.goal.execute()

		return np.concatenate((spec, committed) + tuple(scratchboards) ) 

	def render_pix(self):
		#export goal		
		s = self.goal.execute() if self.goal is not None else BLANK 
		export(s, 'spec.png')
		#export committed
		s = self.cur_prog.execute() if self.cur_prog is not None else BLANK
		export(s, 'committed.png')
		#export scratch
		for i, scr in enumerate(self.scratch):
			s = scr.execute() if scr is not None else BLANK
			export(s, f'scratch_{i}.png')

	def reset(self, debug=None):

		if debug:
			self.goal, self.action_trace = debug
		else:
			self.goal, self.action_trace = randomSceneAndTrace() # uses random scene which 
		self.spec = self.goal.execute()

		self.cur_prog = None

		return self.render()

	def oracle(self):
		#TODO
		raise NotImplementedError
		#should return an action

	def step(self, move, verbose=False):
		assert move in ACTIONS
		sm_neg = -0.1 #some small negative reward
		operator = move[0]
		loc = move[1:]
		#TODO: for each action possible, match it and modify state accordingly
		if operator == 'Commit':
			if self.cur_prog is None:
				self.cur_prog = self.scratch[loc[0]]
			else:
				self.cur_prog = Union(self.cur_prog, self.scratch[loc[0]])

			self.scratch[loc[0]] = None

			#if done:
			if (self.cur_prog.render() == self.goal.render()).all():
				return self.render(), 1.0, True
			else:
				return self.render(), sm_neg, False

		elif operator in ['U', 'L', 'D', 'R']:
			if self.scratch[loc[0]] is None:
				if verbose: print("invalid action", move) #TODO idk if i want this
				return self.render(), sm_neg, False
			else:
				if operator == 'U':
					self.scratch[loc[0]] = Translation(-1, 0, self.scratch[loc[0]]) #check this
				if operator == 'L':
					self.scratch[loc[0]] = Translation(0, -1, self.scratch[loc[0]]) #check this
				if operator == 'D':
					self.scratch[loc[0]] = Translation(1, 0, self.scratch[loc[0]]) #check this
				if operator == 'R':
					self.scratch[loc[0]] = Translation(0, 1, self.scratch[loc[0]]) #check this
				return self.render(), sm_neg, False


		elif operator in ['Circ', 'Rect']:
			if self.scratch[loc[0]] is not None:
				if verbose: print("invalid action", move) #TODO idk if i want this
				return self.render(), sm_neg, False		
			else:
				self.scratch[loc[0]] = {'Circ':Circle(SHAPE_SIZE), 'Rect': Rectangle(SHAPE_SIZE, SHAPE_SIZE)}[operator]
				return self.render(), sm_neg, False 

		elif operator in ['+', '-']:
			#loc[0] becomes the union, the other becomes None
			if self.scratch[loc[0]] is None or self.scratch[loc[1]] is None:
				if verbose: print("invalid action", move)
				return self.render(), sm_neg, False
			self.scratch[loc[0]] = {'+':Union, '-':Difference }[operator](self.scratch[loc[0]], self.scratch[loc[1]])
			self.scratch[loc[1]] = None
			return self.render(), sm_neg, False

		else: 
			assert False, 'invalid move!!'

	#TODO:SOME FORM OF GARBAGE COLLECTION

#THIS WON'T work because across branches ..
from TAN_ENV import get_rollout, get_supervision, train_supervised

def get_supervision(env, max_iter = 100, debug=None):
    cur_state = env.reset(debug=debug)
    actions = env.action_trace
    trace = []
    for action in actions:
        trace.append( (cur_state, action) )
        #print("action", action)
        next_state, r, done = env.step(action)
        #print("cur prog", env.cur_prog.serialize() if env.cur_prog is not None else "none")
        #print("action", action)

        cur_state = next_state
    return trace

def get_rollout(env, policy, max_iter = 200, debug=None):
    cur_state = env.reset(debug=debug)
    done = False
    oracle_actions = env.action_trace
    iter_k = 0
    policy_trace = []
    while not done:
        iter_k += 1
        if iter_k > max_iter:
            break

        a = policy.act(cur_state)
        next_state, r, done = env.step(a)

        policy_trace.append( (cur_state, a) )

        cur_state = next_state
    return oracle_actions, policy_trace

def train_supervised(env, student, max_iter = 200, debug=False):

    if debug: sameScene = randomSceneAndTrace()
    else: sameScene = None
    init_state = env.reset(debug=sameScene)

    for i in range(1000000):
        # learning
        trace = get_supervision(env, max_iter, debug=sameScene)
        states = [x[0] for x in trace]
        actions = [x[1] for x in trace]
        loss = student.learn_supervised(states, actions)

        if i % 100 == 0:
            print (f"loss {loss}")
            oracle_actions, policy_trace = get_rollout(env, student, max_iter, debug=sameScene)
            try:
                env.render_pix()
            except:
                "render failed"
                pass
            
            print ("======== diagnostics ==========")
            print ("Oracle Actions")
            print (oracle_actions)
            print ("Student Actions")
            print ([x[1] for x in policy_trace])

            student.save("CAD5.mdl")

def test_env():
    cenv = CAD_ENV()
    cur_state = cenv.reset()
    print (cenv.cur_prog)
    print (cenv.goal)
    print (cenv.action_trace)
    done = False
    while not done:
        cenv.render_pix()
        oracle_action = cenv.action_trace.pop()
        print ("oracle said ", oracle_action) #this doesn't make sense anymore
        a = input('input\n')
        nxt_state, r, done = cenv.step(a)
        print (np.where(cur_state != nxt_state))
        print ("reward ", r)
        cur_state = nxt_state

def test_ro(load=False): 
    from fcnet import Agent
    env = CAD_ENV()
    agent = Agent(RESOLUTION**2 * (NUM_SCRATCH + 2), ACTIONS)
    if load: agent.load('CAD4.mdl')

    for i in range(20):
        oracle_actions, policy_trace = get_rollout(env, agent, max_iter=200, debug=False)
        try:
            env.render_pix()
        except:
            "render failed"
            pass
        
        print ("======== diagnostics ==========")
        print ("Oracle Actions")
        print (oracle_actions)
        print ("Student Actions")
        print ([x[1] for x in policy_trace])


def test_train_supervised(debug=False, load=True):
    from fcnet import Agent#, MEMAgent
    env = CAD_ENV()
    agent = Agent(RESOLUTION**2 * (NUM_SCRATCH + 2), ACTIONS)
    if load: agent.load('CAD4.mdl')
    train_supervised(env, agent, debug=debug)

def test_model():
    pass

if __name__ == '__main__':
    #test_env()
    #test_ro(load=True)
    test_train_supervised(load=True, debug=False)


