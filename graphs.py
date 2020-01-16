from CAD import *
from utilities import *

import matplotlib.pyplot as plot

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

import pickle

def export_legend(legend, filename="figures/graphics_legend.png"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("checkpoints", nargs='+')
    parser.add_argument("--timeout","-t",type=int,default=None)
    parser.add_argument("--export","-e",type=str,default=None)
    parser.add_argument("--title","-n",type=str,default=None)
    parser.add_argument("--log","-l",action='store_true',default=False)
    parser.add_argument("--y","-y",action='store_true',default=False)
    parser.add_argument("--legend",action='store_true',default=False)

    arguments = parser.parse_args()

    assert arguments.timeout is not None
    timeout = arguments.timeout

    # the times that we will be evaluating at
    ts = np.arange(1 if arguments.log else 0,timeout,0.1)
    
    solverToResult = {}
    for fn in arguments.checkpoints:
        print()
        with open(fn,"rb") as handle:
            try:
                for solver, results in pickle.load(handle):
                    if solver == "no_REPL_beam": continue
                    print(f"From {fn}/{solver} got {len(results)} test cases")

                    for rss in results:
                        for rs in rss:
                            for program in rs.program.nodes:
                                program.clearRendering()

                    if solver not in solverToResult:
                        solverToResult[solver] = [results]
                    else:
                        solverToResult[solver].append(results)
            except Exception as e:
                print(f"Error loading {fn}")
                print(e)


    for s,rss in solverToResult.items():
        L = 0.05
        print(f"Solver {s} gets a program of length:")
        print(max(len(r.program.nodes)
                  for rs in rss 
            for r_ in rs
            for r in r_ 
            if r.loss <= L
        ))
        print(f"and the biggest number of tokens is")
        print(max(sum(len(l.serialize()) for l in r.program.nodes )
                  for rs in rss 
            for r_ in rs
            for r in r_ 
            if r.loss <= L
        ))
        
    plot.figure(figsize=(3.5,2.5))
    #plot.subplot(211)
    #plot.tight_layout();
    if '2' in arguments.title or arguments.y:
        plot.ylabel("Intersection over Union")
    else:
        plot.ylabel("")
    plot.xlabel("Time (seconds)")
    plot.ylim([0,1])
    if arguments.title:
        plot.title(arguments.title)

    def rewardAtTime(t,solver):
        """Returns a list of rewards, one for each random seed"""
        return [mean([1 - min([r.loss for r in rs if r.time <= t] + [1.])
                     for rs in rss ])
                for rss in solverToResult[solver]]
    def deviationAtTime(t,solver):
        seeds = solverToResult[solver]
        deviations = []
        for benchmark in range(len(seeds[0])):
            # collect all the performance for all the seeds for this benchmark
            losses = []
            for s in range(len(seeds)):
                assert len(seeds[s]) == len(seeds[0]) == 100
                losses.append(min([r.loss for r in seeds[s][benchmark] if r.time <= t] + [1.]))
            deviations.append(standardDeviation(losses))
        return mean(deviations)
                

    ordering = ["SMC_value","forwardSample","beam_value","beam","no_REPL"]
    name = {"SMC_value": "SMC",
            "forwardSample": "policy rollout",
            "beam_value": "beam w/ value",
            "beam": "beam w/o value",
            "no_REPL": "no REPL (CSGNet)"}
    color = {"SMC_value": "blue",
            "forwardSample": "orange",
            "beam_value": "green",
            "beam": "red",
            "no_REPL": "purple"}
    ordering = dict(zip(ordering,range(len(ordering))))
    for s in sorted(solverToResult.keys(),
                    key=lambda s: ordering[s]):
        ys = [rewardAtTime(t,s) for t in ts ]
        us = [mean(y) for y in ys ]
        #deviations = [5*standardDeviation(y) for y in ys ]
        deviations = [deviationAtTime(t,s) for t in ts ]
        if not arguments.log:
            plot.plot(ts,us,label=name[s],linewidth=1.5,color=color[s])#,basex=10)
            if "beam" not in s:
                print("providing error bars for",s)
                plot.fill_between(ts,
                                  [max(u - deviation, 0.)
                                   for u,deviation in zip(us,deviations) ],
                                  [min(u + deviation, 1.)
                                   for u,deviation in zip(us,deviations) ],
                                  alpha=0.3,
                                  color=color[s])
        else:
            plot.semilogx(ts,ys,label=name[s],linewidth=3,basex=10)        

    if arguments.export:
        if arguments.legend:
            plot.legend(loc='best',fontsize=7)
        plot.savefig(arguments.export)
        export_legend(plot.legend())
    else:
        plot.legend()
        plot.show()

    
    


    
