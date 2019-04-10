#learn_value_fun

from ROBUT import get_supervised_sample, ALL_BUTTS
from robut_net import Agent
import arguments.args as args
import torch
import time
import random



def sample_from_traces(traces, biased=False):
    #reward is 1 for hit and 0 for not hit
    if biased: raise NotImplementedError
    states = []
    rewards = []
    for trace in traces:
        ln = len(trace)
        idx = random.choice(range(ln))
        states.append( trace[idx].s )
        rewards.append( trace[-1].reward ) #TODO beware
    return states, rewards

def train_value_fun(mode='unbiased'):
    global traces
    from ROB import generate_FIO
    from ROBUT import ROBENV
    print(f"is cuda available? {torch.cuda.is_available()}")
    agent = Agent(ALL_BUTTS, value_net=True)
    try:
        agent.load(args.load_path)
        print("loaded model")
    except FileNotFoundError:
        print ("no saved model found ... training value function from scratch") #TODO XXX


    num_params = sum(p.numel() for p in agent.nn.parameters() if p.requires_grad)
    print("num params in policy net:", num_params)
    num_params = sum(p.numel() for p in agent.Vnn.parameters() if p.requires_grad)
    print("num params in value net:", num_params)


    for i in range(6000):
        envs = []
        for _ in range(args.n_envs_per_rollout):
            prog, inputs, outputs = generate_FIO(5)
            env = ROBENV(inputs, outputs)
            envs.append(env)
            
        ro_t = time.time()
        traces = agent.get_rollouts(envs, n_rollouts=args.n_rollouts) #TODO refactor
        ro_t2 = time.time()
        states, rewards = sample_from_traces(traces)

        t = time.time()
        loss = agent.value_fun_optim_step(states, rewards)
        t2 = time.time()

        if i%args.print_freq==0 and i!=0: print(f"iteration {i}, loss {loss.item()}, net time: {t2-t}, rollout time: {ro_t2 - ro_t}, tot other time {t-t3}")
        t3 = t2

        if i%10==0: 
            agent.save(args.save_path)
            print("Model saved")

if __name__=='__main__':
    train_value_fun()
