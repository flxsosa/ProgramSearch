#test_scaffold

from ROBUT import get_supervised_sample, ALL_BUTTS
from robut_net import Agent

from load_args import args #requires

import torch
import time
import dill

from robut_data import get_supervised_batchsize, GenData, makeTestdata

from collections import namedtuple

SearchResult = namedtuple("SearchResult", "hit solution stats")

def test_on_real_data():
    global beam
    global solutions
    from ROB import generate_FIO
    from ROBUT import ROBENV
    print(f"is cuda available? {torch.cuda.is_available()}")
    agent = Agent(ALL_BUTTS, value_net=True)
    agent.load(args.save_path)
    print("loaded model")

    tasklist = makeTestdata(synth=True, challenge=True, max_num_ex=4)
    import random
    random.shuffle(tasklist)

    if args.debug:
        tasklist = tasklist[:3]

    n_solutions = 0
    results = {}
    for i, (inputs, outputs) in enumerate(tasklist):
        print("inputs:", inputs, sep='\n\t')
        print("outputs", outputs, sep='\n\t', flush=True)
        env = ROBENV(inputs, outputs)

        if args.test_type == 'beam':
            print("BEAM SEARCH:")
            hit, solution, stats = agent.iterative_beam_rollout(env,
                    max_beam_size=1024,
                    max_iter=30,
                    use_value=args.use_value,
                    value_filter_multiple=2,
                    use_prev_value=args.use_prev_value) #TODO maybe full beam

        elif args.test_type == 'a_star':
            print("a_star:")
            hit, solution, stats = agent.a_star_rollout(env,
                    batch_size=1,
                    max_count=1024*30*5,
                    max_iter=30,
                    verbose=False,
                    max_num_actions_expand=800,
                    beam_size=800,
                    no_value=not args.use_value,
                    use_prev_value=args.use_prev_value) #todo add timeout

        elif args.test_type == 'smc':
            hit, solution, stats = agent.smc_rollout(env, max_beam_size=1024, max_iter=30, verbose=False, max_nodes_expanded=2*1024*30*10)

        elif args.test_type == 'sample':
            assert not args.use_value
            assert not args.use_prev_value
            hit, solution, stats = agent.forward_sample_solver(env, batch_size=1024, max_iter=30, max_nodes_expanded=2*1024*30*10, verbose=False)
        else: assert False

        if hit:
            n_solutions += 1
            print("hit!")
        results[ ( tuple(inputs), tuple(outputs) ) ] = SearchResult(hit, solution, stats)

    print(f"{args.test_type} solved {n_solutions} out of {i+1} problems")
    return results

if __name__ == '__main__':
    print(args.resultsfile)
    print(args.test_type)
    print("use value:", args.use_value)

    #load model

    #test model on dataset, depending on type of search.
    #maybe have a timeout?

    #save results somewhere with some data structure:
        #inputs, outputs, solution, search stats

    #plot results?
    results = test_on_real_data()

    #save results 
    timestr = str(int(time.time()))
    filename = args.resultsfile + timestr
    with open(filename, 'wb') as savefile:
        dill.dump(results, savefile)
        print("results file saved at", filename)





