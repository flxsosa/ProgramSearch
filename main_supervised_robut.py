##main_supervised_robut.py

"""
todo:

- [X] finish code for action decoding/sampling
- [X] make best_action, topk
- [X] incorporate last_action at end

- [X] cuda stuff
- [X] write a supervised training loop for robut_net
- [X] do advanced batching

- [X] start a training job
- [X] conv column_encoding

- [X] value fun
- train value fun!!
    - [X] write training loop
        - [X] batch multiple states together during rollout to lower reward variance
    - [ ] is getting the indicies the step which takes the longest???
    
    - [X] make compatible with current networks easily
    - [ ] refactor get_rollouts for speed
    - [X] run!


- [X] test make test tasks, deal with importing from EC
- [ ] robustfill baseline
    - [ ] check if PCCoder version is better to use
    - [ ] write training  for it
    - [ ] make beam search suck less
    - [ ] 

- [X] write bestFS (just not using value) 
    - [ ] test it
- [ ] parallel data gen
- [ ] parallelize rollouts?
- [ ] actually determine testing scheme ... 
- [ ] make a draft for workshop?

- [X] A* search!!!

*** in beam with value: 
- [ ] mini batch running value fun so we can do more without memory error

*** training
    - use button type head
    - modify training distribution

- [X] batched rollout
- [ ] nn modifications

- [X] beam search - make quicker

- [ ] performance debugging
- [ ] does a bigger network perform better??
- [ ] robustfill baseline
- [ ] write attentional pooling
- [ ] write attention/transformer encoder
- [ ] consider incorporating last_action elsewhere
- [ ] good way to do multiple experiments? use git-working?

- [ ] do args more effectively (probably in this file)

"""

from ROBUT import ALL_BUTTS
from ROB import get_supervised_sample
from robut_net import Agent
from load_args import args #requires
import torch
import time

from train_value_fun import train_RL

from robut_data import get_supervised_batchsize, GenData, makeTestdata
import dill

def test_gsb():
    for i, (S, A) in enumerate(get_supervised_batchsize(get_supervised_sample, 200)):
        if i >= 20: break
        print(len(A))
        print(A[:10])

def train():
    print(f"is cuda available? {torch.cuda.is_available()}")
    agent = Agent(ALL_BUTTS, value_net=True)

    try:
        agent.load(args.load_path)
        print("loaded model")
    except FileNotFoundError:
        print ("no saved model found ... training from scratch")


    num_params = sum(p.numel() for p in agent.nn.parameters() if p.requires_grad)
    print("num params:", num_params)

    enum_t2 = 0
    print_time = 0
    if not hasattr(agent, 'train_iterations'): agent.train_iterations = 0

    if args.parallel:
        dataqueue = GenData(get_supervised_sample, n_processes=args.n_processes, max_size=50000)

    for i, (S, A) in enumerate(
            dataqueue.batchIterator(batchsize=args.batchsize) if args.parallel else \
            get_supervised_batchsize(get_supervised_sample, args.batchsize) ):
        enum_t = time.time()
        if agent.train_iterations >= args.train_iterations: break
        t = time.time()
        loss = agent.learn_supervised(S,A)
        t2 = time.time()

        pt = time.time()
        if i%args.print_freq == 0 and i!=0:
            print("iteration {}, loss: {:.5f}, network time: {:.5f}, gen samples time: {:.5f}, prev print time: {:.5f}, total other time: {:.5f}".format(agent.train_iterations, loss.item(), t2-t, enum_t - enum_t2, print_time, t-t3 ))
        pt2 = time.time()
        print_time = pt2-pt

        t3 = t2
        if i%args.save_freq == 0 and i!=0:
            agent.save(args.save_path)
            print("saved model")
        if i%args.test_freq == 0 and i!=0:  
            print("testing...")
            S, A = get_supervised_sample()
            actions = agent.sample_actions(S)
            print("real actions:")
            print(A)
            print("model actions:")
            print(actions)
        enum_t2 = time.time()
    
        if hasattr(agent, 'train_iterations'):
            agent.train_iterations += 1
        else: agent.train_iterations = 1
    agent.save(args.save_path)



def play_with_trained_model():

    from ROBUT import get_rollout, ROBENV, RepeatAgent
    from ROB import generate_FIO
    print(f"is cuda available? {torch.cuda.is_available()}")
    agent = Agent(ALL_BUTTS)
    agent.load(args.save_path)
    print("loaded model")
    num_params = sum(p.numel() for p in agent.nn.parameters() if p.requires_grad)
    print("num params:", num_params)
    for i in range(100000):
        print (f"checking valid programs do not crash interpreter / env {i}")
        prog, inputs, outputs = generate_FIO(5)
        env = ROBENV(inputs, outputs)
        repeat_agent = RepeatAgent(prog.flatten())
        trace_truth = get_rollout(env, repeat_agent, 30)
        trace_sample = get_rollout(env, agent, 30)
        # print ("outputs[0]", outputs[0])
        # print("trace truth", [x[1:3] for x in trace_truth])
        # print("trace sample", [x[1:3] for x in trace_sample])

        # print ("truth trace states")
        # for x in trace_truth:
        #     print (x[-1])
        # print ("sample trace states")
        # for x in trace_sample:
        #     print (x[-1])
        # input()

def test_get_rollouts():
    global traces
    from ROB import generate_FIO
    from ROBUT import ROBENV
    print(f"is cuda available? {torch.cuda.is_available()}")
    agent = Agent(ALL_BUTTS)
    agent.load(args.save_path)
    print("loaded model")
    #prog, inputs, outputs = generate_FIO(5)

    inputs = ['aodklfj', 'aodklfj', 'aodklfj', 'aodklfj']
    outputs = ['3', '3', '3', '3']

    env = ROBENV(inputs, outputs)
    traces = agent.get_rollouts([env], n_rollouts=1000, max_iter=30)
    num_hits = sum([t[-1].reward > 0 for t in traces ])
    print (num_hits)#print(traces)

def test_beam():
    global beam
    global solutions
    from ROB import generate_FIO
    from ROBUT import ROBENV
    print(f"is cuda available? {torch.cuda.is_available()}")
    agent = Agent(ALL_BUTTS, value_net=True)
    agent.load(args.save_path)
    print("loaded model")
    prog, inputs, outputs = generate_FIO(5)
    print("desired prog:", prog.flatten())
    env = ROBENV(inputs, outputs)
    beam, solutions = agent.beam_rollout(env, beam_size=1000, max_iter=30, use_value=False)
    print("number of solutions, no value", len(solutions))
    beam, solutions = agent.beam_rollout(env, beam_size=1000, max_iter=30, use_value=True)
    print("number of solutions, w value", len(solutions))

def test_a_star():
    global beam
    global solutions
    from ROB import generate_FIO
    from ROBUT import ROBENV
    print(f"is cuda available? {torch.cuda.is_available()}")
    agent = Agent(ALL_BUTTS, value_net=True)
    agent.load(args.save_path)
    print("loaded model")

    tasklist = makeTestdata(synth=False, challenge=True, max_num_ex=4)
    print("loaded data")
    import random
    random.shuffle(tasklist)
    inputs, outputs = tasklist[0]


    print("inputs:", inputs, sep='\n\t')
    print("outputs", outputs, sep='\n\t')
    #print("ground truth program:\n\t", prog.flatten())
    env = ROBENV(inputs, outputs)

    print("BEAM SEARCH: (no value)")
    beam, solutions = agent.beam_rollout(env, beam_size=1000, max_iter=30, use_value=False)
    if len(solutions) > 0:
        print("beam search found solution:")
        print(solutions[0].pstate.past_buttons)

    print("OUR MODEL:")
    solutions = agent.a_star_rollout(env, batch_size=1, verbose=True, no_value=False)
    print("A star found solution:")
    print(solutions[0].pstate.past_buttons)
    # print("number of solutions", len(solutions))
    # print("solutions:", solutions)



def test_multistate_rollouts():
    global traces
    from ROB import generate_FIO
    from ROBUT import ROBENV
    print(f"is cuda available? {torch.cuda.is_available()}")
    agent = Agent(ALL_BUTTS)
    agent.load(args.save_path)
    print("loaded model")
    envs = []
    for i in range(50):
        prog, inputs, outputs = generate_FIO(5)
        env = ROBENV(inputs, outputs)
        envs.append(env)

    traces = agent.get_rollouts(envs, n_rollouts=20, max_iter=30)
    num_hits = sum([t[-1].reward > 0 for t in traces ])
    print (num_hits)#print(traces)

def interact_beam():
    global beam
    global solutions
    from ROB import generate_FIO
    from ROBUT import ROBENV
    print(f"is cuda available? {torch.cuda.is_available()}")
    agent = Agent(ALL_BUTTS, value_net=True)
    agent.load(args.save_path)
    print("loaded model")
    prog, inputs, outputs = generate_FIO(5)
    env = ROBENV(inputs, outputs)
    print("inputs:", inputs)
    print("outputs", outputs)
    print("gt:", prog.flatten())
    solutions = agent.interact_beam_rollout(env, beam_size=20, verbose=True)
    print("number of solutions", len(solutions))
    print("solutions:", solutions)


def test_on_real_data():
    global beam
    global solutions
    from ROB import generate_FIO
    from ROBUT import ROBENV
    print(f"is cuda available? {torch.cuda.is_available()}")
    agent = Agent(ALL_BUTTS, value_net=True)
    agent.load(args.load_path)
    print("loaded model")

    tasklist = makeTestdata(synth=False, challenge=True, max_num_ex=4)
    import random
    random.shuffle(tasklist)

    beam_solutions = 0
    astar_solutions = 0
    for i, (inputs, outputs) in enumerate(tasklist):
        print("inputs:", inputs, sep='\n\t')
        print("outputs", outputs, sep='\n\t')

        env = ROBENV(inputs, outputs)

        print("BEAM SEARCH:")
        hit, solution, stats = agent.beam_rollout(env, beam_size=1000, max_iter=30, use_value=False)
        if hit:
            print("beam search found solution:")
            print(solution.pstate.past_buttons)
            beam_solutions += 1

        print("OUR MODEL:")
        hit, solution, stats = agent.a_star_rollout(env, batch_size=1, verbose=True)
        print("OUR MODEL found solution:")
        print(solution.pstate.past_buttons)
        astar_solutions += 1

    print(f"beam solved {beam_solutions} out of {i} problems")
    print(f"a star solved {astar_solutions} out of {i} problems")

def test_smc():
    global beam
    global solutions
    from ROB import generate_FIO
    from ROBUT import ROBENV
    print(f"is cuda available? {torch.cuda.is_available()}")
    agent = Agent(ALL_BUTTS, value_net=True)
    agent.load(args.load_path)
    print("loaded model")

    #tasklist = makeTestdata(synth=False, challenge=True, max_num_ex=4)
    with open('rb_tasks.p', 'rb') as h:
        tasklist = dill.load(h)
    print("loaded data")
    import random
    random.shuffle(tasklist)

    for inputs, outputs in tasklist:

        #inputs = ['aodklfj', 'aodklfj', 'aodklfj', 'aodklfj']
        #outputs = ['3', '3', '3', '3']


        print("inputs:", inputs, sep='\n\t')
        print("outputs", outputs, sep='\n\t')
        #print("ground truth program:\n\t", prog.flatten())
        env = ROBENV(inputs, outputs)

        # print("BEAM SEARCH: (no value)")
        # hit, solution, stats = agent.beam_rollout(env, beam_size=1000, max_iter=30, use_value=False)
        # if len(solutions) > 0:
        #     print("beam search found solution:")
        #     print(solutions[0].pstate.past_buttons)

        print("SMC MODEL:")
        hit, solution, stats = agent.smc_rollout(env, max_beam_size=8192, max_iter=30, verbose=False)
        # print("forward samp")
        # hit, solution, stats = agent.forward_sample_solver(env, batch_size=1024, max_iter=30, max_nodes_expanded=2*1024*30*10, verbose=False)

        if hit:
            print("smt found solution:")
            print(solution.pstate.past_buttons)
        else: print("no solution found for that size")


if __name__=='__main__':
    #test_gsb()
    #train()
    #play_with_trained_model()
    #play_with_trained_model()
    #test_get_rollouts()
    #test_beam()
    #test_a_star()
    #test_multistate_rollouts()
    #train_value_fun()
    #interact_beam()
    #test_on_real_data()
    test_smc()