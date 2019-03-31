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

- [ ] value fun
- train value fun!!
    - [X] write training loop
        - [ ] write biased version?
        - [ ] batch multiple states together during rollout to lower reward variance
    - [ ] is getting the indicies the step which takes the longest???
    
    - [X] make compatible with current networks easily
    - [ ] refactor get_rollouts for speed
    - [X] run!

- [ ] A* search!!!

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

from ROBUT import get_supervised_sample, ALL_BUTTS
from robut_net import Agent
import arguments.args as args
import torch
import time

def get_supervised_batchsize(fn, batchsize=200):
    #takes a generation function and outputs lists of optimal size
    remainder = [], []
    while True:
        preS, preA = remainder
        S, A = fn()
        S, A = preS+S, preA+A
        ln = len(S)

        if ln > batchsize:
            yield S[:batchsize], A[:batchsize]
            remainder = S[batchsize:], A[batchsize:]
            continue
        elif ln < batchsize:
            remainder = S, A
            continue
        elif ln == batchsize:
            yield S, A
            remainder = [], []
            continue
        else: assert 0, "uh oh, not a good place"

def test_gsb():
    for i, (S, A) in enumerate(get_supervised_batchsize(get_supervised_sample, 200)):
        if i >= 20: break
        print(len(A))
        print(A[:10])

def train():
    print(f"is cuda available? {torch.cuda.is_available()}")

    agent = Agent(ALL_BUTTS, value_net=True)

    try:
        agent.load(args.save_path)
        print("loaded model")
    except FileNotFoundError:
        print ("no saved model found ... training from scratch")


    num_params = sum(p.numel() for p in agent.nn.parameters() if p.requires_grad)
    print("num params:", num_params)

    enum_t2 = 0
    print_time = 0
    for i, (S, A) in enumerate(get_supervised_batchsize(get_supervised_sample, args.batchsize)):
        enum_t = time.time()
        if i >= args.train_iterations: break
        t = time.time()
        loss = agent.learn_supervised(S,A)
        t2 = time.time()

        pt = time.time()
        if i%args.print_freq == 0 and i!=0:
            print("iteration {}, loss: {:.5f}, network time: {:.5f}, gen samples time: {:.5f}, prev print time: {:.5f}, total other time: {:.5f}".format(i, loss.item(), t2-t, enum_t - enum_t2, print_time, t-t3 ))
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
    agent.save(args.save_path)
    if hasattr(agent, 'train_iterations'):
        agent.train_iterations += 1
    else: agent.train_iterations = 1

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
    prog, inputs, outputs = generate_FIO(5)
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
    agent = Agent(ALL_BUTTS)
    agent.load(args.save_path)
    print("loaded model")
    prog, inputs, outputs = generate_FIO(5)
    env = ROBENV(inputs, outputs)
    beam, solutions = agent.beam_rollout(env, beam_size=1000, max_iter=30)
    print("number of solutions", len(solutions))

def test_a_star():
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
    solutions = agent.a_star_rollout(env, batch_size=1000, verbose=False)
    print("number of solutions", len(solutions))
    print("solutions:", solutions)

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

if __name__=='__main__':
    #test_gsb()
    #train()
    #play_with_trained_model()
    #play_with_trained_model()
    #test_get_rollouts()
    #test_beam()
    #test_a_star()
    test_multistate_rollouts()
    #train_value_fun()
