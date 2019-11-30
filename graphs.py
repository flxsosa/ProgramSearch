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
    parser.add_argument("--names", type=str, nargs='+', default="") #if need custom names
    arguments = parser.parse_args()

    solverToResult = {}
    for i, fn in enumerate(arguments.checkpoints):
        with open(fn,"rb") as handle:
            r = pickle.load(handle)
            if arguments.names: assert len(r) == 1
            for solver, results in r:
                if arguments.names:
                    solver = arguments.names[i]

                if solver == "no_REPL_beam": continue
                
                if solver not in solverToResult:
                    solverToResult[solver] = [results]
                else:
                    solverToResult[solver].append(results)

    timeout = arguments.timeout
    if timeout is None:
        timeout = max(r_.time
                      for rss in solverToResult.values()
                      for rs in rss 
                      for r in rs
                      for r_ in r )
        print("setting timeout to",timeout)

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
    if '2' in arguments.title:
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

    ordering = ["abs_non_modular", "abs_modular_rl", "abs_modular", "full_repl", 
                "SMC_value", "forwardSample","beam_value","beam","no_REPL"]

    name = {"SMC_value": "SMC",
            "forwardSample": "policy rollout",
            "beam_value": "beam w/ value",
            "beam": "beam w/o value",
            "no_REPL": "no REPL",
            "abs_non_modular": "SMC, non-modular abstract REPL (with spec in objectEnc)",
            "abs_modular": "SMC, modular abstract REPL",
            "full_repl": "SMC, full symbolic REPL (RL training) ",
            "abs_modular_rl": "SMC, modular abstract REPL (RL training)"
            }

    ordering = dict(zip(ordering,range(len(ordering))))
    for s in sorted(solverToResult.keys(),
                    key=lambda s: ordering[s]):
        ts = np.arange(1 if arguments.log else 0,timeout,0.1)
        ys = [rewardAtTime(t,s) for t in ts ]
        us = [mean(y) for y in ys ]
        deviations = [10*standardDeviation(y) for y in ys ]
        if not arguments.log:
            plot.plot(ts,us,label=name[s],linewidth=1.5)#,basex=10)
            if "beam" not in s:
                print(s,deviations)
                plot.fill_between(ts,
                                  [max(u - deviation, 0.)
                                   for u,deviation in zip(us,deviations) ],
                                  [min(u + deviation, 1.)
                                   for u,deviation in zip(us,deviations) ],
                                  alpha=0.3)
        else:
            plot.semilogx(ts,ys,label=name[s],linewidth=3,basex=10)        

    if arguments.export:
        plot.savefig(arguments.export)
        export_legend(plot.legend())
    else:
        plot.legend()
        plot.show()

    
    


    
