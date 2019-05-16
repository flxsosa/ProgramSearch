#test_scaffold

from ROBUT import ALL_BUTTS
from robut_net import Agent


from load_args import args #requires

import torch
import time
import dill

from robut_data import GenData, makeTestdata

from collections import namedtuple

SearchResult = namedtuple("SearchResult", "hit solution stats")

def test_on_real_data():
    global beam
    global solutions
    from ROB import generate_FIO
    from ROBUT import ROBENV
    print(f"is cuda available? {torch.cuda.is_available()}")
    agent = Agent(ALL_BUTTS, value_net=True)
    agent.load(args.load_path)
    print("loaded model")

    if test_args.random_data:
        with open("random_tasks_noconst10.p", 'rb') as h:
            tasklist = dill.load(h)
        print("USING RANDOM TEST DATA")
    else:
        tasklist = makeTestdata(synth=True, challenge=True, max_num_ex=4)
    #import random
    #random.shuffle(tasklist)


    #print("USING SINGLE TASK*********")
    #tasklist = [[("23 Witherspoon Drive, PA", "17 Terrace Street, PA", "188 Rosegarden Way, MA", "1873 Hopkins Road, AK"), ("Witherspoon Drive (PA)", "Terrace Street (PA)", "Rosegarden Way (MA)", "Hopkins Road (AK)")]]
    #tasklist = [[("23 Witherspoon Drive, PA", "17 Terrace Street, PA", "188 Rosegarden Way, MA", "1873 Hopkins Road, AK"), ("Witherspoon Drive (PA) (PA)", "Terrace Street (PA) (PA)", "Rosegarden Way (MA) (MA)", "Hopkins Road (AK) (AK)")]]

    if test_args.debug:
        tasklist = tasklist[:3]

    n_solutions = 0
    results = {}
    for i, (inputs, outputs) in enumerate(tasklist):
        print("inputs:", inputs, sep='\n\t')
        print("outputs", outputs, sep='\n\t', flush=True)
        env = ROBENV(inputs, outputs)

        if test_args.test_type == 'beam':
            print("BEAM SEARCH:")
            hit, solution, stats = agent.iterative_beam_rollout(env,
                    max_beam_size=1024,
                    max_iter=45,
                    use_value=test_args.use_value,
                    value_filter_multiple=2,
                    use_prev_value=test_args.use_prev_value) #TODO maybe full beam

        elif test_args.test_type == 'a_star':
            print("a_star:")
            hit, solution, stats = agent.a_star_rollout(env,
                    batch_size=1,
                    max_count=1024*30*5,
                    max_iter=45,
                    verbose=False,
                    max_num_actions_expand=800,
                    beam_size=800,
                    no_value=not test_args.use_value,
                    use_prev_value=test_args.use_prev_value) #todo add timeout

        elif test_args.test_type == 'smc' and test_args.use_value:
            hit, solution, stats = agent.smc_rollout(env, max_beam_size=1024, max_iter=45, verbose=False, max_nodes_expanded=2*1024*30*10)

        elif test_args.test_type == 'sample' or (test_args.test_type == 'smc' and not test_args.use_value):
            print("doing forward sample")
            assert not test_args.use_value
            assert not test_args.use_prev_value
            hit, solution, stats = agent.forward_sample_solver(env, max_batch_size=1024, max_iter=45, max_nodes_expanded=2*1024*30*10, verbose=False)
        else: assert False

        if hit:
            n_solutions += 1
            print("hit!")
        results[ ( tuple(inputs), tuple(outputs) ) ] = SearchResult(hit, solution, stats)

        #prelim results
        with open(filename, 'wb') as savefile:
            dill.dump(results, savefile)
        print("prelim results file saved at", filename)

    print(f"{test_args.test_type} solved {n_solutions} out of {i+1} problems")
    return results

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--resultsfile", type=str)
    parser.add_argument("--test_type", type=str)
    parser.add_argument("--use_value", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--use_prev_value", action='store_true')
    parser.add_argument("--random_data", action='store_true')
    test_args, unk = parser.parse_known_args()
    print("test_args:", test_args)
    print("unknown test args:", unk)

    print(test_args.resultsfile)
    print(test_args.test_type)
    print("use value:", test_args.use_value)
    print("debug,", test_args.debug)
    print("use_prev_value", test_args.use_prev_value)



    timestr = str(int(time.time()))
    filename = test_args.resultsfile #+ timestr


    #load model

    #test model on dataset, depending on type of search.
    #maybe have a timeout?

    #save results somewhere with some data structure:
        #inputs, outputs, solution, search stats

    #plot results?
    results = test_on_real_data()

    #save results 

    with open(filename, 'wb') as savefile:
        dill.dump(results, savefile)
        print("results file saved at", filename)





