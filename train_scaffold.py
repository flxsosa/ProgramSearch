#train_scaffold.py

#different types of training:

# - [ ] normal supervised
# - [ ] normal value fun training
# - [ ] fewer buttons or something

# - [ ] robustfill base

from robut_net import Agent
from load_args import args #requires

import torch
import time
from ROBUT import ALL_BUTTS
from ROB import get_supervised_sample
from robut_data import get_supervised_batchsize, GenData, makeTestdata

def load_model():
    print(f"is cuda available? {torch.cuda.is_available()}")
    agent = Agent(ALL_BUTTS, value_net=True)
    try:
        agent.load(args.load_path)
        print("loaded model")
    except FileNotFoundError:
        print ("no saved model found ... training from scratch")
    num_params = sum(p.numel() for p in agent.nn.parameters() if p.requires_grad)
    print("num params:", num_params)
    return agent


def train_model_supervised(agent):
    enum_t2 = 0
    print_time = 0
    if not hasattr(agent, 'train_iterations'): agent.train_iterations = 0

    if args.parallel:
        dataqueue = GenData(lambda: get_supervised_sample(render_kind=args.render_kind), n_processes=args.n_processes, batchsize=args.batchsize, max_size=100)

    for i, (S, A) in enumerate(
            dataqueue.batchIterator() if args.parallel else \
            get_supervised_batchsize(lambda: get_supervised_sample(render_kind=args.render_kind), batchsize=args.batchsize) ):
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
            print("saved model", flush=True)
        if i%args.test_freq == 0 and i!=0:  
            print("testing...")
            S, A = get_supervised_sample(render_kind=args.render_kind)
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

def rl_train(agent):
    from train_value_fun import train_value_fun
    if args.rl_mode == 'value only':
        train_value_fun(agent)
    else:
        assert False


if __name__ == '__main__':
    #print some args stuff
    import sys
    print(sys.version)

    #load model or create model
    agent = load_model()
    #train
    train_model_supervised(agent)
    
    #rl train, whatever that entails
    rl_train(agent)