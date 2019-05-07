#learn_value_fun

from ROBUT import ALL_BUTTS, ROBENV
from ROB import get_supervised_sample, generate_FIO
from robut_net import Agent
from load_args import args #requires
import torch
import time
import random



def sample_from_traces(traces, biased=False, keep_all=False, get_actions=False):
    #TODO: get actions from traces    
    #reward is 1 for hit and 0 for not hit
    if biased: raise NotImplementedError
    states = []
    rewards = []
    actions = []
    prev_states = []
    for trace in traces:
        if keep_all: #instead of sampling, just use all of them for gradient update
            l = len(trace)
            r = trace[-1].reward
            #print("reward", r)
            for i, te in enumerate(trace):
                states.append( te.s )
                rewards.append( r )
                if get_actions and r == 1.0:
                    """
                    gets actions and previous states (which match the actions)
                    only if they have positive reward
                    """
                    prev_states.append(te.prev_s)
                    actions.append(te.action)

                    # if i >= l - 1: #if last state, no action
                    #     actions.append( None )
                    # else: 
                    #     actions.append ( trace[i+1].action )
                    # print("state:" )
                    # print("action:", actions[-1])
                    # x = input()
                    # if x == 'n': import pdb ; pdb.set_trace()

        else:
            assert not get_actions
            ln = len(trace)
            idx = random.choice(range(ln))
            states.append( trace[idx].s )
            rewards.append( trace[-1].reward ) #TODO beware

    return states, rewards, actions, prev_states

#depricated, i think
def filter_for_hits(states, rewards, actions):
    reward_states = []
    reward_actions = []
    for state, reward, action in zip(states, rewards, actions):
        if reward == 1 and (action is not None):
            reward_states.append(state)
            reward_actions.append(action)

    return reward_states, reward_actions

def train_RL(agent, mode='unbiased', tune_policy=False):

    num_params = sum(p.numel() for p in agent.nn.parameters() if p.requires_grad)
    print("num params in policy net:", num_params)
    num_params = sum(p.numel() for p in agent.Vnn.parameters() if p.requires_grad)
    print("num params in value net:", num_params)

    for i in range(args.rl_iterations): #TODO
        ttot1 = time.time()
        ttot2 = time.time()
        envs = []
        for _ in range(args.n_envs_per_rollout):
            _, inputs, outputs = generate_FIO(4)
            env = ROBENV(inputs, outputs)
            envs.append(env)
            
        ro_t = time.time()
        traces = agent.get_rollouts(envs, n_rollouts=args.n_rollouts) #TODO refactor
        ro_t2 = time.time()
        states, rewards, reward_actions, reward_states = sample_from_traces(traces, keep_all=True, get_actions=tune_policy)

        t = time.time()
        loss = agent.value_fun_optim_step(states, rewards)
        t2 = time.time()

        if tune_policy:

            #filter only positive rewards
            #reward_states, reward_actions = filter_for_hits(states, rewards, actions)
            print("reward states:", len(reward_states), flush=True)

            pt = time.time()
            if len(reward_states) > 0:
                ploss = agent.learn_supervised(reward_states, reward_actions)  
                ploss = ploss.item()
            else:
                ploss = 0
            pt2 = time.time()

        if tune_policy:
            if i%args.print_freq==0 and i!=0: print(f"iteration {i}, value loss {loss.item()}, value net time: {t2-t}, policy loss {ploss}, policy net time: {pt2-pt}, rollout time: {ro_t2 - ro_t}, tot time {ttot1-ttot3}")
        else:
            if i%args.print_freq==0 and i!=0: print(f"iteration {i}, value loss {loss.item()}, value net time: {t2-t}, rollout time: {ro_t2 - ro_t}, tot other time {t-t3}")
        t3 = t2
        ttot3 = ttot2

        if i%args.print_freq==0 and i!=0: 
            agent.save(args.save_path)
            print("Model saved", flush=True)




if __name__=='__main__':
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

    train_RL(agent, tune_policy=True)
