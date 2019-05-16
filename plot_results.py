#plotting
#plot_results.py
import dill
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import math

"""
stats = {
    'nodes_expanded': 0,
    'policy_runs': 0,
    'policy_gpu_runs': 0,
    'value_runs': 0,
    'value_gpu_runs': 0,
    'start_time': time.time()
    'end_time': 
        }
"""
#SearchResult = namedtuple("SearchResult", "hit solution stats")
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

LONGEST = [False, False, False, False, False, True, True, False, False, True, False, True, False, True, False, True, True, True, True, True, False, True, False, False, True, True, False, True, True, False, True, False, True, False, False, True, False, False, True, False, True, False, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, False, False, True, True, False, True, False, True, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, True, False, False, True, True, False, True, True, False, False, False, True, True, False, False, False, True, False, False, True] 
ONLY_LONGEST = False
HACK = True

def percent_solved(results, fn, x):
    tot = 0
    hits = 0
    for IO, res in results.items():
        if res.hit and fn(res.stats) <= x:
            hits += 1
        tot += 1
    return float(hits)/tot

def compute_x_axis(results_list, fn, title, granularity=200):
    maximum = 0
    for results in results_list:
        for IO, res in results.items():
            if res.hit and fn(res.stats) > maximum:
                maximum = fn(res.stats)

    mn = math.log(0.01) if title == 'Time' else math.log(1) #using title is a hack
    mx = math.log(maximum)
    rng = [ mn + (mx - mn)*i/granularity for i in range(0, granularity+1)]
    return [ math.exp(i) for i in rng]


def plot(file_list, legend_list, filename):

    #load the files:
    results_list = []
    for file in file_list:
        with open(file, 'rb') as h:
            r = dill.load(h)
            if ONLY_LONGEST:
                r = {IO: res for lng, (IO, res) in zip(LONGEST, r.items()) if lng } 

                print("NUMBER OF PROG", len(r))
            if HACK:
                if results_list:
                    r = {IO: res for IO, res in zip(results_list[0].keys(), r.values())}
                    print("NUMBER OF PROG", len(r))
            results_list.append(r)

    EXAMINE = False
    if EXAMINE:
        rs = {key: (r1, r2, r3, r4) for key, r1, r2, r3, r4 in zip(results_list[0].keys(), results_list[0].values(), results_list[1].values(), results_list[2].values(), results_list[3].values())} 
        rlist = list(rs.items())
        slist = sorted(rlist, key=lambda x: -x[1][0].stats['nodes_expanded'])

        for t in slist: print(t[0], '\n',(t[1][0].solution.pstate.past_buttons if t[1][0].solution else "SMC FAILED"),'\n', ('SAMPLE HIT' if t[1][1].solution else "SAMPLE FAILED"), '\n', t[1][0].stats['nodes_expanded'], '\n')   
        import pdb; pdb.set_trace()
    titles_fns = [
        ('Nodes expanded', lambda stats: stats['nodes_expanded']),
        #('Policy net runs', lambda stats: stats['policy_runs']),
        ('Time', lambda stats: stats['end_time'] - stats['start_time']),
        #('Value net runs', lambda stats: stats['value_runs'])
            ]

    fig, ax = plt.subplots(1, len(titles_fns), figsize=(6*len(titles_fns), 6))

    for i, (title, fn) in enumerate(titles_fns):

        x_axis = compute_x_axis(results_list, fn, title, granularity=200)
        for results, legend in zip(results_list, legend_list):

            y_axis = [percent_solved(results, fn, x) for x in x_axis]
            ax[i].semilogx(x_axis, y_axis, label=legend, linewidth=6.0, linestyle='-')#, marker="o") #, c='C6')
            #l = int(len(x_axis)/10)
            #ax[i].plot(x_axis[:l], y_axis[:l], label=legend, linewidth=6.0, linestyle='-')#, marker="o") #, c='C6')
            #ax[i].set_title(title)
            ax[i].legend(loc='lower right')#'best')
            #import pdb; pdb.set_trace()
            plt.axes(ax[i])
            plt.xlabel(title)
            plt.ylabel("Fraction of problems solved")

    savefile='plots/' + filename + str(time.time()) + '.eps'
    plt.savefig(savefile)

if __name__=='__main__':
    #savefile = 'random_data'
    savefile = 'hand_data'

    file_list = [
        './results/sample_our.p',
        './results/smc_our.p'
        ]

    legend_list = [
    'Sample',
    'SMC',
    ]

    # savefile = 'random_data_long'
    # #random data, longer than 10

    # file_list = [
    #     './results/smc_random.p',
    #     './results/smc_noval_random.p',
    #     './results/beam_val_random.p',
    #     './results/beam_noval_random.p',
    # ]

    # legend_list = [
    #     'SMC',
    #     'Sample',
    #     'Beam w/ value',
    #     'Beam w/out value',
    #     ]

    #random data
    # file_list = [
    #     './results/smc_random.p1557951900',
    #     './results/smc_noval_random.p1557951907',
    #     './results/beam_val_random.p1557951789',
    #     './results/beam_noval_random.p1557951893',
    # ]

    # legend_list = [
    #     'SMC',
    #     'Sample',
    #     'Beam w/ value',
    #     'Beam w/out value',
    #     ]

    # noscratch
    """
    file_list = [
        './results/smcnoprefixprelim.p1557854881',
        './results/smc_novalnoprefixprelim.p1557854900',
        './results/beam_valnoprefixprelim.p1557854854',
        './results/beam_novalnoprefixprelim.p1557854864'
        ]

    legend_list = [
        'SMC',
        'Sample',
        'Beam w/ value',
        'Beam w/out value',
        ]
    """
    """
    file_list = [ 
        './results/beam_val.p1557526743',
        './results/beam_noval.p1557526751',
        './results/smc.p1557526764',
        './results/smc_noval.p1557778857',
        './results/a_star.p1557526718',
        './results/a_star_noval.p1557526718',
        ]

    legend_list = [
        'Beam with value',
        'Beam without value',
        'SMC',
        'Sample',
        'A* with value',
        'A* without value'
        ]
    """

    ### workshop:
    # file_list = [
    #                 './results/smc_val_new.p1556099092',
    #                 './results/forward_sample_alpha.p1555577199',
    #                 './results/beam_val_new.p1556084609',
    #                 './results/beam_alpha.p1555229962',
    #                 './results/a_star_150k_alpha.p1555289034',
    #                 './results/a_star_no_val_150k_alpha.p1555286946',
    #                 './results/beam_val_noscratch.p1556240944', #unfinished
    #                 './results/beam_noscratch.p1556240952', #unfinished
    #                 './results/a_star_noscratch.p1556117426',
    #                 './results/a_star_noscratch.p1556130303',
    #                 './results/smc_val_noscratch.p1556173737',
    #                 './results/sample_noscratch.p1556111283'
    #             ]

    # legend_list = [
    #                 'SMC',
    #                 'Forward sample',
    #                 'Beam w/ value',
    #                 'Beam w/out value',
    #                 'A* w/ value',
    #                 'A* w/out value',
    #                 'Beam w/ value, no scratch',
    #                 'Beam w/out value, no scratch',
    #                 'A* w/ value, no scratch',
    #                 'A* w/out value, no scratch',
    #                 'SMC, no scratch',
    #                 'sample, no scratch',
    #                 ]

    # file_list = ['./results/beam_alpha.p1555229962',
    #                 './results/beam_withval_multiple_alpha.p1555370579',
    #                 './results/beam_prev_val_multiple_alpha.p1555372128',
    #                 './results/a_star_150k_alpha.p1555289034', #'./results/a_star_600k_alpha.p1555281742',
    #                 './results/a_star_no_val_150k_alpha.p1555286946',#'./results/a_star_600k_noval_alpha.p1555278678',
    #                 './results/a_star_prev_val_alpha.p1555386519',
    #                 ]

    # legend_list = ['Beam, w/out value',
    #                 'beam, w value',
    #                 'beam, prev value',
    #                 'a_star, w value',
    #                 'a_star, w/out value',
    #                 'a_star, prev value',
    #                 ]

    

    plot(file_list, legend_list, savefile)