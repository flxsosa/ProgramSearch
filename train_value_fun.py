#learn_value_fun

from ROBUT import get_supervised_sample, ALL_BUTTS
from robut_net import Agent
import args
import torch
import time

def train_value_fun():
    print(f"is cuda available? {torch.cuda.is_available()}")

    agent = Agent(ALL_BUTTS, value_fun=True)

    try:
        agent.load(args.save_path)
        print("loaded model")
    except FileNotFoundError:
        print ("no saved model found ... training value function from scratch")


    num_params = sum(p.numel() for p in agent.nn.parameters() if p.requires_grad)
    print("num params:", num_params)

    enum_t2 = 0
    print_time = 0
    for i in range(1000):
       
       traces = get_rollouts()

       states, rewards = sample_some_from_traces(traces)

       loss = train_value_fun(states, rewards)

       