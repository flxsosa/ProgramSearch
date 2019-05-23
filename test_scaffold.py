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

track_edit_distance = True

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
    elif test_args.our_data:
        with open("full_dataset.p", 'rb') as h: #hand_made_data3.p
            tasklist = dill.load(h)
        print("USING OUR TEST DATA")
        print("len(tasklist):", len(tasklist))
    else:
        tasklist = makeTestdata(synth=True, challenge=True, max_num_ex=4)
        print("len(tasklist):", len(tasklist))
        tasklist = list(set(  (tuple(i), tuple(o)) for i, o in tasklist ))
        print("len(tasklist):", len(tasklist))
    #import random
    #random.shuffle(tasklist)


    #print("USING SINGLE TASK*********")
    #tasklist = tasklist[1:2]
    #tasklist = [[("23 Witherspoon Drive, PA", "17 Terrace Street, PA", "188 Rosegarden Way, MA", "1873 Hopkins Road, AK"), ("Witherspoon Drive (PA)", "Terrace Street (PA)", "Rosegarden Way (MA)", "Hopkins Road (AK)")]]
    #tasklist = [[("23 Witherspoon Drive, PA", "17 Terrace Street, PA", "188 Rosegarden Way, MA", "1873 Hopkins Road, AK"), ("Witherspoon Drive (PA) (PA)", "Terrace Street (PA) (PA)", "Rosegarden Way (MA) (MA)", "Hopkins Road (AK) (AK)")]]
    #tasklist = [[('Sat, Nov 10 11:03', 'Mon, Nov 7 3:14', 'Sun, Dec 4 10:00', 'Tues, Jan 13 11:37'), ("Nov 10 at 11 o'clock", "Nov 7 at 3 o'clock", "Dec 4 at 10 o'clock", "Jan 13 at 11 o'clock")], [('Sat, Aug 19, 8:19', 'Sun, Aug 14, 4:04', 'Wed, Feb 6, 7:19', 'Fri, March 6, 5:04'), ("Sat at 8 o'clock", "Sun at 4 o'clock", "Wed at 7 o'clock", "Fri at 5 o'clock")], [('18 Sept 2010', '1 Oct 2028', '14 Oct 1978', '23 Jan 1979'), ('year: 2010; month: Sept', 'year: 2028; month: Oct', 'year: 1978; month: Oct', 'year: 1979; month: Jan')], [('23 Sept 2005', '9 June 2027', '16 Oct 1982', '28 Aug 1987'), ('Sept 23 (2005)', 'June 9 (2027)', 'Oct 16 (1982)', 'Aug 28 (1987)')], [('27-Sept (Sun)', '9-July (Mon)', '2-July (Sun)', '6-March (Tues)'), ('Sun (Sept 27)', 'Mon (July 9)', 'Sun (July 2)', 'Tues (March 6)')], [('24-6-2007', '2-8-1977', '6-1-1983', '30-5-2027'), ('6/24/2007', '8/2/1977', '1/6/1983', '5/30/2027')], [('date: 26 mo: 1 year: 2003', 'date: 26 mo: 8 year: 2003', 'date: 9 mo: 6 year: 2017', 'date: 15 mo: 4 year: 1990'), ('1/26/2003', '8/26/2003', '6/9/2017', '4/15/1990')], [('1/31/1984', '8/28/1975', '10/26/1983', '11/2/2025'), ('date: 31 mo: 1 year: 1984', 'date: 28 mo: 8 year: 1975', 'date: 26 mo: 10 year: 1983', 'date: 2 mo: 11 year: 2025')], [('Wed, Aug 13, 5:54 AM', 'Fri, Nov 21, 5:43 PM', 'Mon, May 9, 7:05 AM', 'Sat, April 11, 1:57 AM'), ('Wed at 5 AM', 'Fri at 5 PM', 'Mon at 7 AM', 'Sat at 1 AM')], [('Oct 27, 4:48 PM', 'Nov 25, 7:08 AM', 'July 30, 4:58 AM', 'Oct 29, 6:15 PM'), ('Oct 27, approx. 4 PM', 'Nov 25, approx. 7 AM', 'July 30, approx. 4 AM', 'Oct 29, approx. 6 PM')], [('May 24, 6:53 PM', 'Dec 11, 8:07 AM', 'Jan 17, 4:34 AM', 'April 20, 12:08 AM'), ('May 24, at 6:53 (PM)', 'Dec 11, at 8:07 (AM)', 'Jan 17, at 4:34 (AM)', 'April 20, at 12:08 (AM)')], [('April 11, 1:44 PM', 'Sept 23, 11:01 AM', 'July 29, 11:56 PM', 'Nov 30, 6:21 PM'), ('April 11 (at 1:44 PM) ', 'Sept 23 (at 11:01 AM) ', 'July 29 (at 11:56 PM) ', 'Nov 30 (at 6:21 PM) ')], [('19/12, 4:06 PM', '10/7, 7:09 PM', '24/4, 2:13 AM', '13/1, 10:15 PM'), ('12/19, at 4:06 (PM)', '7/10, at 7:09 (PM)', '4/24, at 2:13 (AM)', '1/13, at 10:15 (PM)')], [('18/10, 4:05 AM', '19/3, 12:11 AM', '21/6, 7:01 AM', '21/6, 5:23 PM'), ('4:05 (AM) on Jan 18', '12:11 (AM) on Sept 19', '7:01 (AM) on Sept 21', '5:23 (PM) on March 21')], [('23/10, 9:39 PM', '4/10, 5:21 AM', '10/12, 10:23 AM', '26/11, 9:19 PM'), ('10/23 (at 9:39  PM) ', '10/4 (at 5:21  AM) ', '12/10 (at 10:23  AM) ', '11/26 (at 9:19  PM) ')]]
    #tasklist = [ [ ('16-10-2028', '26-7-2007', '27-9-1976', '24-7-2014'), ('10/16/2028', '7/26/2007', '9/27/1976', '7/24/2014') ] ]
    #tasklist = [ [('21/2, 2:55 PM', '18/10, 11:29 PM', '24/10, 10:56 AM', '19/1, 5:30 AM'), ('2/21, at 2:55 (PM)', '10/18, at 11:29 (PM)', '10/24, at 10:56 (AM)', '1/19, at 5:30 (AM)')] ]
    #tasklist = [ [ ('3/16/1997', '4/17/1986', '6/12/2003', '4/23/1997'), ('date: 16 mo: 3 year: 1997', 'date: 17 mo: 4 year: 1986', 'date: 12 mo: 6 year: 2003', 'date: 23 mo: 4 year: 1997') ] ]
    #tasklist = [ [('April 19, 2:45 PM', 'July 5, 8:42 PM', 'July 13, 3:35 PM', 'May 24, 10:22 PM'), ('April 19, approx. 2 PM', 'July 5, approx. 8 PM', 'July 13, approx. 3 PM', 'May 24, approx. 10 PM') ]   ]
    #tasklist = [ [('cell: 322-594-9310', 'home: 190-776-2770', 'home: 224-078-7398', 'cell: 125-961-0607'), ('(322) 5949310 (cell)', '(190) 7762770 (home)', '(224) 0787398 (home)', '(125) 9610607 (cell)') ] ]

    #tasklist = [ [ ('(137) 544 1718', '(582) 431 0370', '(010) 738 6792', '(389) 820 9649'), ('area code: 137, num: 5441718', 'area code: 582, num: 4310370', 'area code: 010, num: 7386792', 'area code: 389, num: 8209649')] ] 
    #tasklist = [ [('Wed, Aug 12, 12:35PM', 'Fri, June 17, 10:21PM', 'Mon, Aug 3, 2:50AM', 'Sun, Oct 12, 9:54AM'), ('Wed at approx. 12 PM', 'Fri at approx. 10 PM', 'Mon at approx. 2 AM', 'Sun at approx. 9 AM') ]]

    #tasklist = [[ ('4-8-1981', '28-11-1987', '22-9-2029', '13-6-1990'), ('8/4/1981', '11/28/1987', '9/22/2029', '6/13/1990') ]]

    #tasklist = [ [('349 First Road', '366 Foothill Road', '19 Wood Violet St', '246 Log Pond Lane') , ('Street:First Road, House num:349', 'Street:Foothill Road, House num:366', 'Street:Wood Violet St, House num:19', 'Street:Log Pond Lane, House num:246') ]] 
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
                    use_prev_value=test_args.use_prev_value, timeout=120 if test_args.twomin_cutoff else None,
                    track_edit_distance=track_edit_distance) #TODO maybe full beam

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
                    use_prev_value=test_args.use_prev_value, timeout=120 if test_args.twomin_cutoff else None,
                    track_edit_distance=track_edit_distance) #todo add timeout

        elif test_args.test_type == 'smc': # and test_args.use_value:
            hit, solution, stats = agent.smc_rollout(env, max_beam_size=1024, max_iter=45, verbose=False, max_nodes_expanded=2*1024*30*10, timeout=120 if test_args.twomin_cutoff else None, sample_only=not test_args.use_value, track_edit_distance=track_edit_distance)
        elif test_args.test_type == 'sample': #or (test_args.test_type == 'smc' and not test_args.use_value):
            print("doing forward sample")
            assert not test_args.use_value
            assert not test_args.use_prev_value
            hit, solution, stats = agent.forward_sample_solver(env, max_batch_size=1024, max_iter=45, max_nodes_expanded=2*1024*30*10, verbose=False, timeout=120 if test_args.twomin_cutoff else None, track_edit_distance=track_edit_distance)
        else: assert False

        if track_edit_distance:
            print("best prog found:")
            print(stats['best_ps'])

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
    parser.add_argument("--our_data", action='store_true')
    parser.add_argument("--twomin_cutoff", action='store_true')
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





